//! Benchmarks a fixed CNN training step, for comparison against `PyTorch`.
//!
//! Pairs with `scripts/bench_pytorch.py`, which builds the same model, the same
//! input shape and the same number of steps. Both report peak resident memory
//! and mean milliseconds per step so the numbers can be read side by side.
//!
//!     cargo run --release -p vearo --example bench_cnn
//!     python3 scripts/bench_pytorch.py
//!
//! Peak memory is read from `VmHWM` in `/proc/self/status`, the high-water mark
//! of resident set size, rather than current usage: a training loop that frees
//! between steps can show a small instantaneous figure while still having needed
//! a large peak.
//!
//! Vearo's CPU backend is single-threaded, so compare against
//! `bench_pytorch.py --threads 1`. The default multi-threaded `PyTorch` number is
//! also worth knowing, but it measures a different thing.
#![allow(clippy::cast_precision_loss)]

use std::time::Instant;
use vearo::nn::Module;
use vearo::{Device, Tensor};

const BATCH: usize = 32;
const CHANNELS: usize = 3;
const SIDE: usize = 32;
const CLASSES: usize = 10;
const WARMUP: usize = 5;
const STEPS: usize = 20;

struct BenchCnn {
    conv1: vearo::nn::Conv2d,
    pool1: vearo::nn::MaxPool2d,
    conv2: vearo::nn::Conv2d,
    pool2: vearo::nn::MaxPool2d,
    fc: vearo::nn::Linear,
}

impl BenchCnn {
    fn new(device: Device) -> Self {
        Self {
            conv1: vearo::nn::Conv2d::new(CHANNELS, 16, 3, 1, 1, true, 1).to(device),
            pool1: vearo::nn::MaxPool2d::new(2, 2, 0).to(device),
            conv2: vearo::nn::Conv2d::new(16, 32, 3, 1, 1, true, 2).to(device),
            pool2: vearo::nn::MaxPool2d::new(2, 2, 0).to(device),
            fc: vearo::nn::Linear::new(32 * 8 * 8, CLASSES, true, 3).to(device),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.pool1.forward(&self.conv1.forward(x).relu());
        let h = self.pool2.forward(&self.conv2.forward(&h).relu());
        let flat = h.reshape([BATCH, 32 * 8 * 8]);
        self.fc.forward(&flat)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.conv1.parameters();
        p.extend(self.conv2.parameters());
        p.extend(self.fc.parameters());
        p
    }
}

/// Peak resident set size in MiB, from `VmHWM`.
fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).map(str::to_string))
        })
        .and_then(|kb| kb.parse::<f64>().ok())
        .map_or(0.0, |kb| kb / 1024.0)
}

fn main() {
    vearo::init();
    vearo::set_training(true);

    // Deterministic inputs, so the comparison is not measuring data generation.
    let n = BATCH * CHANNELS * SIDE * SIDE;
    let xs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin()).collect();
    let ys: Vec<f32> = (0..BATCH).map(|i| (i % CLASSES) as f32).collect();

    let device = if vearo::cuda_available() { Device::Cuda(0) } else { Device::Cpu };
    let x = Tensor::from_f32(&xs, [BATCH, CHANNELS, SIDE, SIDE]).to(device);
    let y = Tensor::from_f32(&ys, [BATCH]).to(device);

    let model = BenchCnn::new(device);
    let mut opt = vearo::optim::AdamW::new(model.parameters(), 1e-3, 0.9, 0.999, 1e-8, 0.0);

    let rss_before = peak_rss_mib();

    let mut step = || {
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();
        let loss = model.forward(&x).cross_entropy(&y);
        let value = loss.to_vec_f32()[0];
        loss.backward();
        opt.step();
        value
    };

    // Warm up so allocation growth and first-touch costs are not counted.
    for _ in 0..WARMUP {
        step();
    }

    let start = Instant::now();
    let mut last = 0.0;
    for _ in 0..STEPS {
        last = step();
    }
    let elapsed = start.elapsed().as_secs_f64();

    let rss_after = peak_rss_mib();

    println!("framework      vearo (device: {:?})", device);
    println!("model          conv(3-16) pool conv(16-32) pool fc({}-{CLASSES})", 32 * 8 * 8);
    println!("input          [{BATCH}, {CHANNELS}, {SIDE}, {SIDE}]");
    println!("steps          {STEPS} (after {WARMUP} warmup)");
    println!("final loss     {last:.6}");
    println!("ms/step        {:.2}", elapsed * 1000.0 / STEPS as f64);
    println!("peak rss mib   {rss_after:.1}");
    println!("rss growth mib {:.1}", rss_after - rss_before);
}
