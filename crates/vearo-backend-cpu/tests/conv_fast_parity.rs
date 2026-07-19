//! Checks `conv2d_fast` against the reference `conv2d`.
//!
//! The two are deliberately not bit-equal. `conv2d` accumulates in a fixed
//! scalar order and is bit-identical to the CUDA kernel; `conv2d_fast` lowers
//! the convolution to a GEMM, which reassociates the sum. So this asserts
//! agreement to a tolerance, and that is the contract: use `conv2d` where
//! bit-equality matters, `conv2d_fast` where speed does.


// Conv parameters are conventionally single letters (n, k, c); spelling them
// out here would obscure rather than clarify.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops
)]
use vearo_backend_cpu::{conv2d, conv2d_fast};
use vearo_core::{DType, Device, Shape, Tensor};

/// Deterministic values with enough variation that cancellation would show.
fn filled(shape: [usize; 4], phase: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.13 + phase).sin() * 0.7 + ((i as f32) * 0.07).cos() * 0.3)
        .collect();
    Tensor::from_f32(&data, Shape::new(shape)).to(Device::Cpu)
}

fn check(n: usize, cin: usize, cout: usize, side: usize, k: usize, stride: usize, padding: usize) {
    vearo_backend_cpu::init();

    let x = filled([n, cin, side, side], 0.0);
    let w = filled([cout, cin, k, k], 1.7);
    let b = Tensor::from_f32(
        &(0..cout).map(|i| (i as f32) * 0.05 - 0.1).collect::<Vec<_>>(),
        [cout],
    )
    .to(Device::Cpu);

    let reference = conv2d(&x, &w, &b, stride, padding).to_vec_f32();
    let fast = conv2d_fast(&x, &w, &b, stride, padding).to_vec_f32();

    assert_eq!(
        reference.len(),
        fast.len(),
        "shape mismatch between conv2d and conv2d_fast"
    );

    // A degenerate all-zero output would make the comparison meaningless.
    let magnitude = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        magnitude > 1e-3,
        "reference output is degenerate (max |out| = {magnitude})"
    );

    let worst = reference
        .iter()
        .zip(fast.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let relative = worst / magnitude;

    println!(
        "n{n} cin{cin} cout{cout} {side}x{side} k{k} s{stride} p{padding}: \
         max|out| {magnitude:.4} abs {worst:e} rel {relative:e}"
    );
    assert!(
        relative < 1e-5,
        "conv2d_fast disagrees with the reference by {relative:e} relative \
         (absolute {worst:e}) for n{n} cin{cin} cout{cout} {side}x{side} \
         k{k} s{stride} p{padding}"
    );
}

#[test]
fn fast_conv_matches_reference_within_tolerance() {
    // Shapes chosen to cover padding on and off, stride 1 and 2, and a case
    // where the channel count is not a multiple of any obvious blocking factor.
    check(2, 3, 8, 8, 3, 1, 1);
    check(1, 1, 1, 5, 3, 1, 0);
    check(2, 4, 6, 8, 3, 2, 1);
    check(1, 5, 7, 9, 3, 1, 1);
    check(3, 16, 32, 16, 3, 1, 1);
}

#[test]
fn fast_conv_handles_no_padding_and_stride_two() {
    check(1, 2, 4, 7, 3, 2, 0);
    check(2, 3, 5, 10, 5, 2, 0);
}

/// The dtype guard must survive the faster path.
#[test]
#[should_panic(expected = "F32")]
fn fast_conv_rejects_non_f32() {
    vearo_backend_cpu::init();
    let x = Tensor::zeros(Shape::new([1, 1, 4, 4]), DType::I32);
    let w = filled([1, 1, 3, 3], 0.0);
    let b = Tensor::from_f32(&[0.0], [1]).to(Device::Cpu);
    let _ = conv2d_fast(&x, &w, &b, 1, 0);
}
