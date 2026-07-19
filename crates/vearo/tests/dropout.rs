//! Dropout: eval-mode identity, train-mode masking/scaling, and gradient flow.
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]

use vearo::Tensor;
use vearo::nn::{Dropout, Module};

fn setup() {
    vearo::backend_cpu::init();
    vearo::autograd::init();
}

#[test]
fn test_dropout_eval_is_identity() {
    setup();
    vearo::set_training(false);

    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [4]);
    let d = Dropout::new(0.5, 7);
    let y = d.forward(&x);
    assert_eq!(y.to_vec_f32(), x.to_vec_f32());
}

#[test]
fn test_dropout_train_masks_and_scales() {
    setup();
    vearo::set_training(true);

    let n = 10_000;
    let p = 0.5f32;
    let scale = 1.0 / (1.0 - p);
    let x = Tensor::from_f32(&vec![1.0f32; n], [n]);
    let d = Dropout::new(p, 42);
    let y = d.forward(&x).to_vec_f32();

    // Every element is either dropped (0) or kept and scaled by 1/(1-p).
    for &v in &y {
        assert!(v == 0.0 || (v - scale).abs() < 1e-5, "unexpected value {v}");
    }
    // Kept fraction ~ (1-p), and the expectation is preserved (mean ~ 1.0).
    let kept = y.iter().filter(|&&v| v != 0.0).count() as f32 / n as f32;
    assert!((kept - (1.0 - p)).abs() < 0.03, "kept fraction {kept}");
    let mean = y.iter().sum::<f32>() / n as f32;
    assert!((mean - 1.0).abs() < 0.05, "mean {mean}");
}

#[test]
fn test_dropout_grad_through_mask() {
    setup();
    vearo::set_training(true);

    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8]);
    x.set_requires_grad(true);

    let d = Dropout::new(0.5, 123);
    let y = d.forward(&x);
    let y_vec = y.to_vec_f32();
    let loss = y.sum(0, false);
    loss.backward();
    let g = x.grad().unwrap().to_vec_f32();

    // loss = sum(x * mask)  =>  d(loss)/dx_i = mask_i (0 if dropped, scale if kept).
    let scale = 2.0f32; // 1 / (1 - 0.5)
    for i in 0..8 {
        if y_vec[i] == 0.0 {
            assert_eq!(g[i], 0.0, "dropped elem {i} must have zero grad");
        } else {
            assert!(
                (g[i] - scale).abs() < 1e-5,
                "kept elem {i} grad {} != {scale}",
                g[i]
            );
        }
    }
}
