//! CNN on real MNIST digits - proves conv2d works inside an image classifier,
//! trained end-to-end on both the CPU and CUDA backends.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::uninlined_format_args,
    clippy::needless_range_loop,
    clippy::float_cmp
)]

use vearo::nn::{Conv2d, Linear, MaxPool2d, Module};
use vearo::{Device, Tensor};

/// Loads the first `n` MNIST rows from the CSV: (normalized pixels, labels).
fn load_mnist(n: usize) -> (Vec<f32>, Vec<f32>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/mnist_train.csv");
    let content = std::fs::read_to_string(path).expect("read data/mnist_train.csv");

    let mut images = Vec::with_capacity(n * 28 * 28);
    let mut labels = Vec::with_capacity(n);
    for line in content.lines().skip(1).take(n) {
        let mut fields = line.split(',');
        labels.push(fields.next().unwrap().parse::<f32>().unwrap());
        for px in fields {
            images.push(px.parse::<f32>().unwrap() / 255.0);
        }
    }
    (images, labels)
}

/// Trains a small CNN to overfit `n` MNIST digits on `device`; returns final train accuracy.
fn run_cnn(device: Device) -> f32 {
    let n = 16;
    let (img, lbl) = load_mnist(n);
    let x = Tensor::from_f32(&img, [n, 1, 28, 28]).to(device);
    let y = Tensor::from_f32(&lbl, [n]).to(device);

    // conv(1->8, k3 s1 p1) -> relu -> conv(8->16, k3 s2 p1) -> relu -> fc(16*14*14 -> 10)
    let conv1 = Conv2d::new(1, 8, 3, 1, 1, true, 1).to(device);
    let conv2 = Conv2d::new(8, 16, 3, 2, 1, true, 2).to(device);
    let fc = Linear::new(16 * 14 * 14, 10, true, 3).to(device);

    let mut params = conv1.parameters();
    params.extend(conv2.parameters());
    params.extend(fc.parameters());
    let mut opt = vearo::optim::AdamW::new(params, 2e-3, 0.9, 0.999, 1e-8, 0.0);

    let forward = |x: &Tensor| {
        let h = conv1.forward(x).relu();
        let h = conv2.forward(&h).relu();
        let flat = h.reshape([n, 16 * 14 * 14]);
        fc.forward(&flat)
    };

    let mut acc = 0.0;
    for epoch in 0..50 {
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        let logits = forward(&x);
        let loss = logits.cross_entropy(&y);
        loss.backward();
        opt.step();

        if epoch % 10 == 9 {
            let logits_host = forward(&x).to(Device::Cpu).to_vec_f32();
            let mut correct = 0;
            for i in 0..n {
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for c in 0..10 {
                    let v = logits_host[i * 10 + c];
                    if v > best_v {
                        best_v = v;
                        best = c;
                    }
                }
                if best as f32 == lbl[i] {
                    correct += 1;
                }
            }
            acc = correct as f32 / n as f32;
            println!(
                "[{:?}] epoch {} | loss {:.4} | train acc {:.1}%",
                device,
                epoch + 1,
                loss.to(Device::Cpu).to_vec_f32()[0],
                acc * 100.0
            );
        }
    }
    acc
}

#[test]
#[ignore = "needs data/mnist_train.csv (see README); not shipped in the repo"]
fn test_cnn_overfits_mnist_cpu() {
    vearo::init();
    let acc = run_cnn(Device::Cpu);
    assert!(acc > 0.9, "CPU CNN failed to overfit MNIST digits; train acc {}", acc);
}

#[test]
#[ignore = "needs data/mnist_train.csv (see README); not shipped in the repo"]
fn test_cnn_overfits_mnist_cuda() {
    vearo::init();
    let acc = run_cnn(Device::Cuda(0));
    assert!(acc > 0.9, "CUDA CNN failed to overfit MNIST digits; train acc {}", acc);
}

/// Trains a CNN *with max pooling* to overfit `n` MNIST digits on `device`; returns final accuracy.
fn run_cnn_pool(device: Device) -> f32 {
    let n = 16;
    let (img, lbl) = load_mnist(n);
    let x = Tensor::from_f32(&img, [n, 1, 28, 28]).to(device);
    let y = Tensor::from_f32(&lbl, [n]).to(device);

    // conv(1->8,k3s1p1) -> relu -> pool2 -> conv(8->16,k3s1p1) -> relu -> pool2 -> fc(16*7*7 -> 10)
    let conv1 = Conv2d::new(1, 8, 3, 1, 1, true, 1).to(device);
    let conv2 = Conv2d::new(8, 16, 3, 1, 1, true, 2).to(device);
    let pool = MaxPool2d::new(2, 2, 0);
    let fc = Linear::new(16 * 7 * 7, 10, true, 3).to(device);

    let mut params = conv1.parameters();
    params.extend(conv2.parameters());
    params.extend(fc.parameters());
    let mut opt = vearo::optim::AdamW::new(params, 2e-3, 0.9, 0.999, 1e-8, 0.0);

    let forward = |x: &Tensor| {
        let h = pool.forward(&conv1.forward(x).relu()); // [n, 8, 14, 14]
        let h = pool.forward(&conv2.forward(&h).relu()); // [n, 16, 7, 7]
        let flat = h.reshape([n, 16 * 7 * 7]);
        fc.forward(&flat)
    };

    let mut acc = 0.0;
    for epoch in 0..50 {
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        let logits = forward(&x);
        let loss = logits.cross_entropy(&y);
        loss.backward();
        opt.step();

        if epoch % 10 == 9 {
            let logits_host = forward(&x).to(Device::Cpu).to_vec_f32();
            let mut correct = 0;
            for i in 0..n {
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for c in 0..10 {
                    let v = logits_host[i * 10 + c];
                    if v > best_v {
                        best_v = v;
                        best = c;
                    }
                }
                if best as f32 == lbl[i] {
                    correct += 1;
                }
            }
            acc = correct as f32 / n as f32;
            println!(
                "[{:?}+pool] epoch {} | loss {:.4} | train acc {:.1}%",
                device,
                epoch + 1,
                loss.to(Device::Cpu).to_vec_f32()[0],
                acc * 100.0
            );
        }
    }
    acc
}

#[test]
#[ignore = "needs data/mnist_train.csv (see README); not shipped in the repo"]
fn test_cnn_maxpool_overfits_cpu() {
    vearo::init();
    let acc = run_cnn_pool(Device::Cpu);
    assert!(acc > 0.9, "CPU maxpool-CNN failed to overfit; train acc {}", acc);
}

#[test]
#[ignore = "needs data/mnist_train.csv (see README); not shipped in the repo"]
fn test_cnn_maxpool_overfits_cuda() {
    vearo::init();
    let acc = run_cnn_pool(Device::Cuda(0));
    assert!(acc > 0.9, "CUDA maxpool-CNN failed to overfit; train acc {}", acc);
}
