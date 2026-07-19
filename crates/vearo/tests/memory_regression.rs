//! Resource-regression tests: assert that repeated training steps do not grow
//! host or device memory.
//!
//! Correctness tests cannot catch a leak - every individual step produces the right
//! answer, and only the growth over many steps reveals the bug. This file covers that
//! gap. It exists because CPU arena slots were leaking ~12 per training step (the
//! refcount hooks were consulted before the device check, so once a device backend
//! registered a hook, every CPU tensor took that branch and the hook ignored non-CUDA
//! devices). That leak made high-resolution training impossible long before it made
//! anything fail.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args,
    clippy::many_single_char_names
)]

use vearo::nn::{BatchNorm2d, Conv2d, Linear, MaxPool2d, Module};
use vearo::{Device, Tensor};

fn live_cpu_slots() -> usize {
    (0..vearo::core::NUM_SHARDS)
        .map(|i| {
            let g = vearo::core::get_cpu_shard(i).lock().unwrap();
            g.slots.len() - g.free_indices.len()
        })
        .sum()
}

fn live_cuda_slots() -> usize {
    let slots = vearo::backend_cuda::CUDA_SLOTS.lock().unwrap().len();
    let free = vearo::backend_cuda::FREE_CUDA_SLOTS.lock().unwrap().len();
    slots - free
}

/// Runs `iters` full training steps and returns `(cpu_slot_growth, cuda_slot_growth)`.
fn measure_growth(device: Device, iters: usize) -> (i64, i64) {
    let (b, c, h, w) = (32usize, 3usize, 16usize, 16usize);
    let flat = 16 * 8 * 8;

    let conv = Conv2d::new(c, 16, 3, 1, 1, true, 1).to(device);
    let bn = BatchNorm2d::new(16, 0.1, 1e-5).to(device);
    let pool = MaxPool2d::new(2, 2, 0);
    let fc = Linear::new(flat, 10, true, 2).to(device);

    let mut params = conv.parameters();
    params.extend(bn.parameters());
    params.extend(fc.parameters());
    let mut opt = vearo::optim::AdamW::new(params, 1e-3, 0.9, 0.999, 1e-8, 1e-4);

    let xd: Vec<f32> = (0..b * c * h * w)
        .map(|i| (i as f32 * 0.013).sin())
        .collect();
    let yd: Vec<f32> = (0..b).map(|i| (i % 10) as f32).collect();

    vearo::set_training(true);

    let step = |opt: &mut vearo::optim::AdamW| {
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();
        let x = Tensor::from_f32(&xd, [b, c, h, w]).to(device);
        let y = Tensor::from_f32(&yd, [b]).to(device);
        let hh = pool.forward(&bn.forward(&conv.forward(&x)).relu());
        let logits = fc.forward(&hh.reshape([b, flat]));
        let loss = logits.cross_entropy(&y);
        loss.backward();
        opt.step();
    };

    // Warm up so one-time allocations are not counted as growth.
    for _ in 0..3 {
        step(&mut opt);
    }

    let cpu_before = live_cpu_slots() as i64;
    let cuda_before = live_cuda_slots() as i64;
    for _ in 0..iters {
        step(&mut opt);
    }
    (
        live_cpu_slots() as i64 - cpu_before,
        live_cuda_slots() as i64 - cuda_before,
    )
}

/// Both devices are checked in one test, on purpose.
///
/// The measurement reads the process-global CPU arena, and cargo runs `#[test]`
/// functions on parallel threads. As two separate tests, each one's allocations
/// land inside the other's before/after window, so a clean run can report growth
/// that no leak caused. Keeping them sequential in a single test makes the
/// measurement mean what it claims.
#[test]
fn test_no_memory_growth() {
    vearo::init();
    let iters = 25;

    let (cpu_only, _) = measure_growth(Device::Cpu, iters);
    assert!(
        cpu_only <= 0,
        "CPU arena leaked {} slots over {} steps ({:.2}/step) - storage is not being reclaimed",
        cpu_only,
        iters,
        cpu_only as f64 / iters as f64
    );

    let (cpu, cuda) = measure_growth(Device::Cuda(0), iters);
    assert!(
        cpu <= 0,
        "CPU arena leaked {} slots over {} CUDA steps ({:.2}/step) - host storage is not \
         being reclaimed during device training",
        cpu,
        iters,
        cpu as f64 / iters as f64
    );
    assert!(
        cuda <= 0,
        "CUDA slots leaked {} over {} steps ({:.2}/step)",
        cuda,
        iters,
        cuda as f64 / iters as f64
    );
}
