//! Checks the fused attention kernel against an unfused reference.
//!
//! The reference is plain matmul/softmax/matmul assembled from ops that already
//! have their own gradient checks, so the fused kernel has to agree with it on
//! both outputs and gradients or the fusion changed the maths.
//!
//! Comparing at tensor level rather than through `MultiHeadAttention` avoids
//! toggling the `VEARO_USE_FUSED_ATTENTION` env var, which cannot be set without
//! `unsafe` under edition 2024 and would leak across tests in the same process.

use vearo::{Device, Tensor};

const B: usize = 2;
const H: usize = 2;
const S: usize = 4;
const DK: usize = 8;

/// Deterministic, varied values so no two positions coincide by accident.
fn fill(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32;
            (x * 0.37 + phase).sin() * 0.5 + (x * 0.11 + phase).cos() * 0.25
        })
        .collect()
}

fn qkv() -> (Tensor, Tensor, Tensor) {
    let n = B * H * S * DK;
    let q = Tensor::from_f32(&fill(n, 0.0), [B, H, S, DK]).to(Device::Cpu);
    let k = Tensor::from_f32(&fill(n, 1.3), [B, H, S, DK]).to(Device::Cpu);
    let v = Tensor::from_f32(&fill(n, 2.7), [B, H, S, DK]).to(Device::Cpu);
    q.set_requires_grad(true);
    k.set_requires_grad(true);
    v.set_requires_grad(true);
    (q, k, v)
}

/// Weighted reduction: a plain sum would let different attention outputs share
/// a gradient and hide a real disagreement.
fn reduce(out: &Tensor) -> Tensor {
    let n: usize = out.shape().dims().iter().product();
    let coef: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.19).sin() + 1.5).collect();
    let coef_t = Tensor::from_f32(&coef, *out.shape()).to(Device::Cpu);
    out.mul(&coef_t)
        .sum(0, false)
        .sum(0, false)
        .sum(0, false)
        .sum(0, false)
}

fn grads(q: &Tensor, k: &Tensor, v: &Tensor) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Gradients can arrive non-contiguous (the reference path transposes k),
    // and to_vec_f32 requires a contiguous buffer.
    (
        q.grad().expect("dq").contiguous().to_vec_f32(),
        k.grad().expect("dk").contiguous().to_vec_f32(),
        v.grad().expect("dv").contiguous().to_vec_f32(),
    )
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn fused_attention_matches_unfused_reference() {
    vearo::backend_cpu::init();
    vearo::autograd::init();
    vearo::set_training(true);

    // Causal mask with rank 2, exactly as SimpleGpt builds it. The fused path
    // must broadcast it the same way the reference `add` does.
    let mut mask_data = vec![0.0f32; S * S];
    for row in 0..S {
        for col in 0..S {
            if col > row {
                mask_data[row * S + col] = -1e9;
            }
        }
    }
    let mask = Tensor::from_f32(&mask_data, [S, S]).to(Device::Cpu);
    let scale = Tensor::from_f32(&[1.0 / (DK as f32).sqrt()], [1]).to(Device::Cpu);

    // ---- reference ---------------------------------------------------------
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let (q, k, v) = qkv();
    let scores = q.matmul(&k.transpose(2, 3)).mul(&scale).add(&mask);
    let out_ref_t = scores.softmax(3).matmul(&v);
    let out_ref = out_ref_t.contiguous().to_vec_f32();
    reduce(&out_ref_t).backward();
    let (dq_ref, dk_ref, dv_ref) = grads(&q, &k, &v);

    // ---- fused -------------------------------------------------------------
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let (q2, k2, v2) = qkv();
    let out_fused_t = q2.fused_attention(&k2, &v2, Some(&mask));
    let out_fused = out_fused_t.contiguous().to_vec_f32();
    reduce(&out_fused_t).backward();
    let (dq_f, dk_f, dv_f) = grads(&q2, &k2, &v2);

    // A degenerate (all-zero) output would make every comparison below pass.
    let maxabs = out_ref.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(maxabs > 1e-4, "reference output is degenerate (max |out| = {maxabs})");

    let d_out = max_abs_diff(&out_ref, &out_fused);
    let d_q = max_abs_diff(&dq_ref, &dq_f);
    let d_k = max_abs_diff(&dk_ref, &dk_f);
    let d_v = max_abs_diff(&dv_ref, &dv_f);

    println!("max |out| = {maxabs:.6}");
    println!("output   diff = {d_out:e}");
    println!("grad q   diff = {d_q:e}");
    println!("grad k   diff = {d_k:e}");
    println!("grad v   diff = {d_v:e}");

    assert!(d_out < 1e-5, "fused output disagrees by {d_out:e}");
    assert!(d_v < 1e-4, "fused grad v disagrees by {d_v:e}");
    assert!(d_q < 1e-4, "fused grad q disagrees by {d_q:e}");
    assert!(d_k < 1e-4, "fused grad k disagrees by {d_k:e}");
}
