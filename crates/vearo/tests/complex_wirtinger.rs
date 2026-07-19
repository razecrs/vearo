//! Finite-difference gradient check for complex-valued layers.
//!
//! # Why this is the test that matters
//!
//! For a real-valued loss `L` over a complex parameter `z = a + ib`, the
//! steepest-descent direction is not the naive complex derivative but the
//! conjugate Wirtinger derivative:
//!
//! ```text
//! dL/dz_bar = 1/2 (dL/da + i dL/db)
//! z <- z - lr * 2 * dL/dz_bar = z - lr * (dL/da + i dL/db)
//! ```
//!
//! `ComplexTensor` keeps the real and imaginary parts as two ordinary real
//! tensors and builds every operation out of real ops, so the existing real
//! autograd already produces `dL/da` and `dL/db`. Updating those two tensors
//! independently with a real optimizer is therefore exactly the Wirtinger
//! update, with the factor of two folded into the learning rate. No explicit
//! conjugation is needed anywhere.
//!
//! That argument holds only if the complex product is the real composition
//! `(ac - bd) + i(ad + bc)` and autograd differentiates it correctly. A sign
//! error in either term still trains, just towards the wrong place, which is
//! precisely the failure finite differences catch and a smoke test does not.

// Test-only float formatting and index maths; lossy casts here cannot affect
// any result under test.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops
)]

use vearo::nn::complex::{ComplexLinear, ComplexTensor};
use vearo::{Device, Tensor};

const IN: usize = 3;
const OUT: usize = 2;
const BATCH: usize = 2;

/// Real-valued loss over a complex output: a weighted `sum |out|^2`.
///
/// The weights are varied on purpose. With uniform weights many different
/// outputs share a gradient, which hides sign errors in the imaginary path.
fn loss_of(out: &ComplexTensor) -> Tensor {
    let n = BATCH * OUT;
    let coef: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7).sin() + 1.6).collect();
    let c = Tensor::from_f32(&coef, [BATCH, OUT]).to(Device::Cpu);
    let re2 = out.real.mul(&out.real);
    let im2 = out.imag.mul(&out.imag);
    re2.add(&im2).mul(&c).sum(0, false).sum(0, false)
}

fn input() -> ComplexTensor {
    let re: Vec<f32> = (0..BATCH * IN).map(|i| ((i as f32) * 0.53).sin()).collect();
    let im: Vec<f32> = (0..BATCH * IN).map(|i| ((i as f32) * 0.31).cos()).collect();
    ComplexTensor::new(
        Tensor::from_f32(&re, [BATCH, IN]).to(Device::Cpu),
        Tensor::from_f32(&im, [BATCH, IN]).to(Device::Cpu),
    )
}

/// Evaluates the loss with one scalar of one parameter nudged by `delta`.
fn loss_with_nudge(layer: &ComplexLinear, which: usize, index: usize, delta: f32) -> f32 {
    let params = layer.parameters();
    let target = &params[which];
    let mut vals = target.to_vec_f32();
    vals[index] += delta;
    target.copy_from_slice(&vals);

    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let l = loss_of(&layer.forward(&input())).to_vec_f32()[0];

    // Put it back so the next probe starts from the same point.
    vals[index] -= delta;
    target.copy_from_slice(&vals);
    l
}

#[test]
fn complex_linear_gradients_match_finite_differences() {
    vearo::backend_cpu::init();
    vearo::autograd::init();
    vearo::set_training(true);

    let layer = ComplexLinear::new(IN, OUT, true, 11);
    for p in layer.parameters() {
        p.set_requires_grad(true);
    }

    // Analytic gradients from autograd.
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let out = layer.forward(&input());
    let l0 = loss_of(&out);
    let base = l0.to_vec_f32()[0];
    l0.backward();

    let params = layer.parameters();
    let names = ["w_real", "w_imag", "b_real", "b_imag"];

    // A degenerate loss would let a wrong gradient pass unnoticed.
    assert!(
        base.abs() > 1e-3,
        "loss is degenerate ({base}), the comparison below would be vacuous"
    );

    let h = 1e-3f32;
    let mut worst = 0.0f32;
    let mut worst_where = String::new();

    // Snapshot every analytic gradient before touching finite differences:
    // loss_with_nudge calls zero_gradients(), which would wipe the gradients of
    // any parameter not yet read.
    let analytic_all: Vec<Vec<f32>> = params
        .iter()
        .enumerate()
        .map(|(pi, p)| {
            p.grad()
                .unwrap_or_else(|| panic!("no gradient for {}", names[pi]))
                .contiguous()
                .to_vec_f32()
        })
        .collect();

    for (pi, analytic) in analytic_all.iter().enumerate() {
        for (idx, &a) in analytic.iter().enumerate() {
            // Central difference: error is O(h^2) rather than O(h).
            let plus = loss_with_nudge(&layer, pi, idx, h);
            let minus = loss_with_nudge(&layer, pi, idx, -h);
            let numeric = (plus - minus) / (2.0 * h);

            let scale = a.abs().max(numeric.abs()).max(1.0);
            let rel = (a - numeric).abs() / scale;
            if rel > worst {
                worst = rel;
                worst_where = format!(
                    "{}[{idx}]: analytic {a:.6}, numeric {numeric:.6}",
                    names[pi]
                );
            }
        }
    }

    println!("loss = {base:.6}");
    println!("worst relative gradient error = {worst:.3e}  ({worst_where})");
    assert!(
        worst < 2e-2,
        "complex gradients disagree with finite differences: {worst_where} \
         (relative error {worst:.3e}). Check the sign of the cross terms in \
         ComplexTensor::matmul: the product must be (ac - bd) + i(ad + bc)."
    );
}

/// The imaginary path must actually carry gradient. If `matmul` dropped or
/// zeroed a cross term the test above could still pass on a layer whose
/// imaginary weights happen not to matter.
#[test]
fn imaginary_weights_receive_gradient() {
    vearo::backend_cpu::init();
    vearo::autograd::init();
    vearo::set_training(true);

    let layer = ComplexLinear::new(IN, OUT, true, 11);
    for p in layer.parameters() {
        p.set_requires_grad(true);
    }

    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    loss_of(&layer.forward(&input())).backward();

    let g_imag = layer.w_imag.grad().expect("w_imag gradient").contiguous().to_vec_f32();
    let maxabs = g_imag.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("max |dL/dw_imag| = {maxabs:.6}");
    assert!(
        maxabs > 1e-4,
        "w_imag receives no gradient, so the imaginary path is disconnected"
    );
}
