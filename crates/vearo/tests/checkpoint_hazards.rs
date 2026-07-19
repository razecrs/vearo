//! Probes the two failure modes activation checkpointing is classically prone to.
//!
//! Checkpointing runs the forward block twice: once to get the output, once
//! during backward to rebuild the activations. Anything stateful or random
//! inside the block therefore runs twice, which is fine only if the second run
//! reproduces the first exactly.
//!
//! 1. Dropout draws a fresh mask per call. If the recomputed mask differs from
//!    the one used in the forward pass, the gradient is taken with respect to a
//!    network that was never evaluated. Masks come from a thread-local counter,
//!    so both runs below rewind it to the same point; otherwise they would draw
//!    different masks for reasons that have nothing to do with checkpointing.
//! 2. `BatchNorm` updates running mean/variance during training. Running the
//!    block twice applies that update twice per optimizer step, so the running
//!    statistics drift at double rate and eval-mode results are wrong.

// Test-only float formatting and index maths; lossy casts here cannot affect
// any result under test.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops
)]

use vearo::nn::Module;
use vearo::{Device, Tensor};

/// A block containing dropout must produce the same gradient with and without
/// checkpointing, which requires the recomputed mask to match the original.
#[test]
fn dropout_inside_checkpoint_is_reproducible() {
    vearo::backend_cpu::init();
    vearo::autograd::init();
    vearo::set_training(true);

    let x = Tensor::from_f32(&[0.5, -1.5, 2.0, 0.25, 1.0, -0.75], [2, 3]).to(Device::Cpu);
    x.set_requires_grad(true);

    // Same module instance both times, so any internal RNG state is shared.
    let drop = vearo::nn::Dropout::new(0.25, 7);

    // Reference: no checkpointing.
    vearo::set_rng_counter(0);
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let plain = drop.forward(&x).sum(0, false).sum(0, false);
    plain.backward();
    let grad_plain = x.grad().unwrap().to_vec_f32();

    // Checkpointed, rewound to the same counter so the first call inside the
    // block draws the same mask the reference run drew.
    vearo::set_rng_counter(0);
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let drop2 = vearo::nn::Dropout::new(0.25, 7);
    let ckp = vearo::autograd::checkpoint(&x, move |inp| drop2.forward(inp))
        .sum(0, false)
        .sum(0, false);
    ckp.backward();
    let grad_ckp = x.grad().unwrap().to_vec_f32();

    println!("plain grad:      {grad_plain:?}");
    println!("checkpoint grad: {grad_ckp:?}");

    // Two all-zero gradients agree trivially. If the mask dropped every unit,
    // this test would pass no matter how badly checkpointing behaved, so
    // require that some units survived and carry gradient.
    let live = grad_plain.iter().filter(|g| g.abs() > 1e-9).count();
    assert!(
        live > 0,
        "every unit was dropped, so this comparison proves nothing - \
         adjust the seed or p so some units survive"
    );
    println!("surviving units carrying gradient: {live}/{}", grad_plain.len());
    for (i, (a, b)) in grad_plain.iter().zip(grad_ckp.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "element {i}: checkpointing changed the gradient ({a} vs {b}). \
             The recomputed dropout mask differs from the one used in forward."
        );
    }
}

/// Running statistics must advance once per training step, not twice.
#[test]
fn batchnorm_running_stats_not_updated_twice() {
    vearo::backend_cpu::init();
    vearo::autograd::init();
    vearo::set_training(true);

    let x = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2, 2, 1, 2],
    )
    .to(Device::Cpu);
    x.set_requires_grad(true);

    // Reference: one plain forward advances the running stats once.
    let bn_plain = vearo::nn::BatchNorm2d::new(2, 0.1, 1e-5);
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let out = bn_plain.forward(&x).sum(0, false).sum(0, false).sum(0, false).sum(0, false);
    out.backward();
    let mean_plain = bn_plain.running_mean.borrow().to_vec_f32();
    let var_plain = bn_plain.running_var.borrow().to_vec_f32();

    // Checkpointed: the block runs twice, so a naive implementation updates the
    // running statistics twice for a single logical step.
    // BatchNorm2d is not Clone, so the module is shared through an Rc: the
    // closure and this scope must observe the same running statistics.
    let bn_ckp = std::rc::Rc::new(vearo::nn::BatchNorm2d::new(2, 0.1, 1e-5));
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let bn_moved = std::rc::Rc::clone(&bn_ckp);
    let out2 = vearo::autograd::checkpoint(&x, move |inp| bn_moved.forward(inp))
        .sum(0, false)
        .sum(0, false)
        .sum(0, false)
        .sum(0, false);
    out2.backward();
    let mean_ckp = bn_ckp.running_mean.borrow().to_vec_f32();
    let var_ckp = bn_ckp.running_var.borrow().to_vec_f32();

    println!("plain      running_mean {mean_plain:?} running_var {var_plain:?}");
    println!("checkpoint running_mean {mean_ckp:?} running_var {var_ckp:?}");
    for (i, (a, b)) in mean_plain.iter().zip(mean_ckp.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "channel {i}: running_mean advanced differently under checkpointing \
             ({a} vs {b}), so the block's statistics were updated twice."
        );
    }
}
