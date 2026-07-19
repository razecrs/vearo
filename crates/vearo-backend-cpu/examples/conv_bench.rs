//! Times the reference convolution against the GEMM-lowered one.
//!
//!     cargo run --release -p vearo-backend-cpu --example conv_bench
//!
//! The two are not bit-equal by design: the reference accumulates in a fixed
//! scalar order and matches the CUDA kernel exactly, while the GEMM path
//! reassociates the sum. `tests/conv_fast_parity.rs` pins the agreement to a
//! tolerance; this only measures what that buys.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops
)]

use std::time::Instant;
use vearo_backend_cpu::{conv2d, conv2d_fast};
use vearo_core::{Device, Shape, Tensor};

fn filled(shape: [usize; 4]) -> Tensor {
    let n: usize = shape.iter().product();
    let d: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin()).collect();
    Tensor::from_f32(&d, Shape::new(shape)).to(Device::Cpu)
}

fn bench(label: &str, n: usize, cin: usize, cout: usize, side: usize) {
    let x = filled([n, cin, side, side]);
    let w = filled([cout, cin, 3, 3]);
    let b = Tensor::from_f32(&vec![0.0f32; cout], [cout]).to(Device::Cpu);
    let reps = 10;

    // One untimed run each so first-touch allocation is not charged to the
    // first sample.
    let _ = conv2d(&x, &w, &b, 1, 1);
    let t = Instant::now();
    for _ in 0..reps {
        let _ = conv2d(&x, &w, &b, 1, 1);
    }
    let slow = t.elapsed().as_secs_f64() * 1000.0 / f64::from(reps);

    let _ = conv2d_fast(&x, &w, &b, 1, 1);
    let t = Instant::now();
    for _ in 0..reps {
        let _ = conv2d_fast(&x, &w, &b, 1, 1);
    }
    let fast = t.elapsed().as_secs_f64() * 1000.0 / f64::from(reps);

    println!("{label:<24} reference {slow:>8.2} ms   gemm {fast:>7.2} ms   {:>5.1}x", slow / fast);
}

fn main() {
    vearo_backend_cpu::init();
    bench("conv1 3->16 @32x32", 32, 3, 16, 32);
    bench("conv2 16->32 @16x16", 32, 16, 32, 16);
    bench("wide 64->128 @32x32", 16, 64, 128, 32);
}
