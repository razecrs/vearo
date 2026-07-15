//! CNN on real MNIST digits - proves conv2d works inside an image classifier.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::uninlined_format_args,
    clippy::needless_range_loop,
    clippy::float_cmp
)]

use vearo::Tensor;
use vearo::nn::{Conv2d, Linear, Module};

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

#[test]
fn test_cnn_overfits_mnist() {
    vearo::init();

    let n = 16;
    let (img, lbl) = load_mnist(n);
    let x = Tensor::from_f32(&img, [n, 1, 28, 28]);
    let y = Tensor::from_f32(&lbl, [n]);

    // conv(1->8, k3 s1 p1) -> relu -> conv(8->16, k3 s2 p1) -> relu -> fc(16*14*14 -> 10)
    let conv1 = Conv2d::new(1, 8, 3, 1, 1, true, 1);
    let conv2 = Conv2d::new(8, 16, 3, 2, 1, true, 2);
    let fc = Linear::new(16 * 14 * 14, 10, true, 3);

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
            let logits_c = forward(&x).contiguous();
            let mut correct = 0;
            for i in 0..n {
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for c in 0..10 {
                    let v = logits_c.get_f32(i * 10 + c);
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
                "epoch {} | loss {:.4} | train acc {:.1}%",
                epoch + 1,
                loss.get_f32(0),
                acc * 100.0
            );
        }
    }

    assert!(
        acc > 0.9,
        "CNN failed to overfit MNIST digits; train acc {}",
        acc
    );
}
