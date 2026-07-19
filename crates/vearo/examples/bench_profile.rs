//! Breaks a training step down by phase, to find what the benchmark gap is made of.
//!
//!     cargo run --release -p vearo --example bench_profile
//!
//! `bench_cnn` reports a whole step; this attributes the time. Optimising before
//! measuring where it goes is how you end up speeding up something that was
//! never the bottleneck.
#![allow(clippy::cast_precision_loss)]

use std::time::Instant;
use vearo::nn::Module;
use vearo::{Device, Tensor};

const BATCH: usize = 32;
const REPS: usize = 10;

/// Times `f` over `REPS` runs and returns mean milliseconds.
fn time_ms(label: &str, mut f: impl FnMut()) -> f64 {
    // One untimed run so lazy allocation is not attributed to the first sample.
    f();
    let start = Instant::now();
    for _ in 0..REPS {
        f();
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    println!("  {label:<34} {ms:>9.2} ms");
    ms
}

fn main() {
    vearo::init();
    vearo::set_training(true);

    let x = Tensor::from_f32(
        &(0..BATCH * 3 * 32 * 32)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect::<Vec<_>>(),
        [BATCH, 3, 32, 32],
    )
    .to(Device::Cpu);

    let conv1 = vearo::nn::Conv2d::new(3, 16, 3, 1, 1, true, 1);
    let pool1 = vearo::nn::MaxPool2d::new(2, 2, 0);
    let conv2 = vearo::nn::Conv2d::new(16, 32, 3, 1, 1, true, 2);
    let pool2 = vearo::nn::MaxPool2d::new(2, 2, 0);
    let fc = vearo::nn::Linear::new(32 * 8 * 8, 10, true, 3);

    println!("forward, by layer:");
    let h1 = conv1.forward(&x);
    let mut total = 0.0;
    total += time_ms("conv1 3->16 on 32x32", || {
        vearo::autograd::reset_active_tape();
        let _ = conv1.forward(&x);
    });
    let p1 = pool1.forward(&h1.relu());
    total += time_ms("pool1", || {
        vearo::autograd::reset_active_tape();
        let _ = pool1.forward(&h1.relu());
    });
    total += time_ms("conv2 16->32 on 16x16", || {
        vearo::autograd::reset_active_tape();
        let _ = conv2.forward(&p1);
    });
    let h2 = conv2.forward(&p1);
    let p2 = pool2.forward(&h2.relu());
    let flat = p2.reshape([BATCH, 32 * 8 * 8]);
    total += time_ms("fc 2048->10", || {
        vearo::autograd::reset_active_tape();
        let _ = fc.forward(&flat);
    });
    println!("  {:<34} {total:>9.2} ms", "forward total");

    // Full step, so the share spent in backward is visible by difference.
    println!();
    println!("whole step:");
    let mut params = conv1.parameters();
    params.extend(conv2.parameters());
    params.extend(fc.parameters());
    let mut opt = vearo::optim::AdamW::new(params, 1e-3, 0.9, 0.999, 1e-8, 0.0);
    let y = Tensor::from_f32(
        &(0..BATCH).map(|i| (i % 10) as f32).collect::<Vec<_>>(),
        [BATCH],
    )
    .to(Device::Cpu);

    let step_ms = time_ms("forward + backward + opt", || {
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();
        let h = pool1.forward(&conv1.forward(&x).relu());
        let h = pool2.forward(&conv2.forward(&h).relu());
        let loss = fc
            .forward(&h.reshape([BATCH, 32 * 8 * 8]))
            .cross_entropy(&y);
        loss.backward();
        opt.step();
    });

    println!();
    println!("forward is {:.0}% of the step", total / step_ms * 100.0);
    println!(
        "backward + optimizer is {:.0}%",
        (step_ms - total) / step_ms * 100.0
    );
}
