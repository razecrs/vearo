//! Integration tests for BatchNorm2d (forward/backward parity, train/eval modes).
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown
)]

// This file exercises the CUDA backend directly, so it only builds with the
// `cuda` feature. CPU coverage of the same ops lives in the other test files.
#![cfg(feature = "cuda")]

use vearo::nn::{BatchNorm2d, Module};
use vearo::{Device, Tensor};

fn setup() {
    vearo::backend_cpu::init();
    vearo::backend_cuda::init();
    vearo::autograd::init();
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn test_batchnorm_forward_train_eval_parity() {
    setup();

    let n = 2;
    let c = 3;
    let h = 4;
    let w = 4;
    let numel = n * c * h * w;

    // 1. Setup inputs (random sine wave to be deterministic)
    let x_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.1).sin()).collect();

    // Test training mode
    {
        vearo::set_training(true);
        let bn_cpu = BatchNorm2d::new(c, 0.1, 1e-5);
        let bn_cuda = bn_cpu.to(Device::Cuda(0));

        let x_cpu = Tensor::from_f32(&x_data, [n, c, h, w]);
        let x_cuda = x_cpu.to(Device::Cuda(0));

        let y_cpu = bn_cpu.forward(&x_cpu);
        let y_cuda = bn_cuda.forward(&x_cuda).to(Device::Cpu);

        assert!(
            max_diff(&y_cpu.to_vec_f32(), &y_cuda.to_vec_f32()) < 1e-5,
            "Train forward output mismatch: max diff = {}",
            max_diff(&y_cpu.to_vec_f32(), &y_cuda.to_vec_f32())
        );

        // Check running stats parity
        let rm_cpu = bn_cpu.running_mean.borrow().to_vec_f32();
        let rm_cuda = bn_cuda.running_mean.borrow().to(Device::Cpu).to_vec_f32();
        assert!(max_diff(&rm_cpu, &rm_cuda) < 1e-5, "Running mean mismatch");

        let rv_cpu = bn_cpu.running_var.borrow().to_vec_f32();
        let rv_cuda = bn_cuda.running_var.borrow().to(Device::Cpu).to_vec_f32();
        assert!(
            max_diff(&rv_cpu, &rv_cuda) < 1e-5,
            "Running variance mismatch"
        );
    }

    // Test eval mode
    {
        vearo::set_training(false);
        let bn_cpu = BatchNorm2d::new(c, 0.1, 1e-5);
        // Pre-fill running stats with some mock data to verify they are used
        let rm_mock = vec![0.5f32; c];
        let rv_mock = vec![1.5f32; c];
        *bn_cpu.running_mean.borrow_mut() = Tensor::from_f32(&rm_mock, [c]);
        *bn_cpu.running_var.borrow_mut() = Tensor::from_f32(&rv_mock, [c]);

        let bn_cuda = bn_cpu.to(Device::Cuda(0));

        let x_cpu = Tensor::from_f32(&x_data, [n, c, h, w]);
        let x_cuda = x_cpu.to(Device::Cuda(0));

        let y_cpu = bn_cpu.forward(&x_cpu);
        let y_cuda = bn_cuda.forward(&x_cuda).to(Device::Cpu);

        assert!(
            max_diff(&y_cpu.to_vec_f32(), &y_cuda.to_vec_f32()) < 1e-5,
            "Eval forward output mismatch"
        );
    }
}

#[test]
fn test_batchnorm_backward_parity() {
    setup();

    let n = 2;
    let c = 3;
    let h = 4;
    let w = 4;
    let numel = n * c * h * w;

    let x_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.1).sin()).collect();

    // Test training backward parity
    {
        vearo::set_training(true);
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        let bn_cpu = BatchNorm2d::new(c, 0.1, 1e-5);
        let x_cpu = Tensor::from_f32(&x_data, [n, c, h, w]);
        x_cpu.set_requires_grad(true);

        let y_cpu = bn_cpu.forward(&x_cpu);
        let loss_cpu = y_cpu
            .sum(0, false)
            .sum(0, false)
            .sum(0, false)
            .sum(0, false);
        loss_cpu.backward();

        let gx_cpu = x_cpu.grad().unwrap().to_vec_f32();
        let gw_cpu = bn_cpu.weight.grad().unwrap().to_vec_f32();
        let gb_cpu = bn_cpu.bias.grad().unwrap().to_vec_f32();

        // CUDA run
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        let bn_cuda = bn_cpu.to(Device::Cuda(0));
        let x_cuda = Tensor::from_f32(&x_data, [n, c, h, w]).to(Device::Cuda(0));
        x_cuda.set_requires_grad(true);

        let y_cuda = bn_cuda.forward(&x_cuda);
        let loss_cuda = y_cuda
            .sum(0, false)
            .sum(0, false)
            .sum(0, false)
            .sum(0, false);
        loss_cuda.backward();

        let gx_cuda = x_cuda.grad().unwrap().to(Device::Cpu).to_vec_f32();
        let gw_cuda = bn_cuda.weight.grad().unwrap().to(Device::Cpu).to_vec_f32();
        let gb_cuda = bn_cuda.bias.grad().unwrap().to(Device::Cpu).to_vec_f32();

        assert!(max_diff(&gx_cpu, &gx_cuda) < 1e-5, "grad_x mismatch");
        assert!(max_diff(&gw_cpu, &gw_cuda) < 1e-5, "grad_weight mismatch");
        assert!(max_diff(&gb_cpu, &gb_cuda) < 1e-5, "grad_bias mismatch");
    }

    // Test eval backward parity
    {
        vearo::set_training(false);
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        let bn_cpu = BatchNorm2d::new(c, 0.1, 1e-5);
        let x_cpu = Tensor::from_f32(&x_data, [n, c, h, w]);
        x_cpu.set_requires_grad(true);

        let y_cpu = bn_cpu.forward(&x_cpu);
        let loss_cpu = y_cpu
            .sum(0, false)
            .sum(0, false)
            .sum(0, false)
            .sum(0, false);
        loss_cpu.backward();

        let gx_cpu = x_cpu.grad().unwrap().to_vec_f32();
        let gw_cpu = bn_cpu.weight.grad().unwrap().to_vec_f32();
        let gb_cpu = bn_cpu.bias.grad().unwrap().to_vec_f32();

        // CUDA run
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        let bn_cuda = bn_cpu.to(Device::Cuda(0));
        let x_cuda = Tensor::from_f32(&x_data, [n, c, h, w]).to(Device::Cuda(0));
        x_cuda.set_requires_grad(true);

        let y_cuda = bn_cuda.forward(&x_cuda);
        let loss_cuda = y_cuda
            .sum(0, false)
            .sum(0, false)
            .sum(0, false)
            .sum(0, false);
        loss_cuda.backward();

        let gx_cuda = x_cuda.grad().unwrap().to(Device::Cpu).to_vec_f32();
        let gw_cuda = bn_cuda.weight.grad().unwrap().to(Device::Cpu).to_vec_f32();
        let gb_cuda = bn_cuda.bias.grad().unwrap().to(Device::Cpu).to_vec_f32();

        assert!(max_diff(&gx_cpu, &gx_cuda) < 1e-5, "Eval grad_x mismatch");
        assert!(
            max_diff(&gw_cpu, &gw_cuda) < 1e-5,
            "Eval grad_weight mismatch"
        );
        assert!(
            max_diff(&gb_cpu, &gb_cuda) < 1e-5,
            "Eval grad_bias mismatch"
        );
    }
}
