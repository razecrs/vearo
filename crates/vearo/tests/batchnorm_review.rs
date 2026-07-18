//! Reviewer tests for BatchNorm: true gradient correctness (numerical vs analytical,
//! not just CPU/CUDA agreement) and the running-statistics lifecycle.
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::suboptimal_flops
)]

use vearo::Tensor;
use vearo::autograd::numerical_grad;
use vearo::nn::{BatchNorm2d, Module};

fn setup() {
    vearo::backend_cpu::init();
    vearo::autograd::init();
}

/// Grad-check BatchNorm in training mode.
///
/// NOTE: the loss must NOT be a plain `sum(BN(x))`. Because `x_hat` is zero-mean by
/// construction, `sum(BN(x)) == m * beta` regardless of `x`, so grad_x would be ~0 and
/// the check would pass even with a badly wrong formula. Weighting the output by a
/// fixed varied coefficient makes the gradient non-degenerate.
#[test]
fn test_batchnorm_gradcheck_training() {
    setup();
    vearo::set_training(true);

    let (n, c, h, w) = (2usize, 3usize, 2usize, 2usize);
    let numel = n * c * h * w;
    let xd: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.37).sin() * 1.5 + 0.2)
        .collect();
    let coefd: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.71).cos() + 1.3).collect();

    let x = Tensor::from_f32(&xd, [n, c, h, w]);
    x.set_requires_grad(true);
    let gamma = Tensor::from_f32(&[1.2, 0.8, 1.5], [c]);
    gamma.set_requires_grad(true);
    let beta = Tensor::from_f32(&[0.1, -0.2, 0.3], [c]);
    beta.set_requires_grad(true);
    let rm = Tensor::from_f32(&vec![0.0f32; c], [c]);
    let rv = Tensor::from_f32(&vec![1.0f32; c], [c]);
    let coef = Tensor::from_f32(&coefd, [n, c, h, w]);

    let forward = |t: &Tensor| {
        t.batchnorm(&gamma, &beta, &rm, &rv, true, 0.1, 1e-5)
            .mul(&coef)
            .reshape([numel])
            .sum(0, false)
    };

    let out = forward(&x);
    out.backward();
    let gx_ana = x.grad().unwrap().to_vec_f32();
    let gg_ana = gamma.grad().unwrap().to_vec_f32();
    let gb_ana = beta.grad().unwrap().to_vec_f32();

    vearo::autograd::reset_active_tape();
    let gx_num = numerical_grad(forward, &x, 1e-3).to_vec_f32();

    // Guard: if the gradient were ~0 the comparison would be vacuous.
    let maxabs = gx_num.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(
        maxabs > 1e-3,
        "degenerate grad_x (max |g| = {maxabs}); test would prove nothing"
    );

    for i in 0..numel {
        let d = (gx_ana[i] - gx_num[i]).abs();
        assert!(
            d < 5e-2,
            "grad_x[{i}]: analytical={} numerical={} diff={d}",
            gx_ana[i],
            gx_num[i]
        );
    }

    vearo::autograd::reset_active_tape();
    let fwd_g = |t: &Tensor| {
        x.batchnorm(t, &beta, &rm, &rv, true, 0.1, 1e-5)
            .mul(&coef)
            .reshape([numel])
            .sum(0, false)
    };
    let gg_num = numerical_grad(fwd_g, &gamma, 1e-3).to_vec_f32();

    vearo::autograd::reset_active_tape();
    let fwd_b = |t: &Tensor| {
        x.batchnorm(&gamma, t, &rm, &rv, true, 0.1, 1e-5)
            .mul(&coef)
            .reshape([numel])
            .sum(0, false)
    };
    let gb_num = numerical_grad(fwd_b, &beta, 1e-3).to_vec_f32();

    for i in 0..c {
        assert!(
            (gg_ana[i] - gg_num[i]).abs() < 5e-2,
            "grad_gamma[{i}]: analytical={} numerical={}",
            gg_ana[i],
            gg_num[i]
        );
        assert!(
            (gb_ana[i] - gb_num[i]).abs() < 5e-2,
            "grad_beta[{i}]: analytical={} numerical={}",
            gb_ana[i],
            gb_num[i]
        );
    }
}

/// The running statistics must actually persist on the module across forwards in
/// training mode, and must stay frozen in eval mode.
#[test]
fn test_batchnorm_running_stats_lifecycle() {
    setup();

    let (n, c, h, w) = (2usize, 2usize, 2usize, 2usize);
    let numel = n * c * h * w;
    let momentum = 0.1f32;
    let bn = BatchNorm2d::new(c, momentum, 1e-5);

    // Channel 0 is all 2.0, channel 1 is all 6.0 -> batch means 2.0 / 6.0, batch var 0.
    let mut xd = vec![0.0f32; numel];
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h {
                for ww in 0..w {
                    xd[((nn * c + cc) * h + hh) * w + ww] = if cc == 0 { 2.0 } else { 6.0 };
                }
            }
        }
    }
    let x = Tensor::from_f32(&xd, [n, c, h, w]);

    // Training: running stats move toward the batch stats.
    vearo::set_training(true);
    let _ = bn.forward(&x);
    let rm = bn.running_mean.borrow().to_vec_f32();
    let rv = bn.running_var.borrow().to_vec_f32();
    assert!(
        (rm[0] - momentum * 2.0).abs() < 1e-5,
        "running_mean[0]={} expected {}",
        rm[0],
        momentum * 2.0
    );
    assert!(
        (rm[1] - momentum * 6.0).abs() < 1e-5,
        "running_mean[1]={} expected {}",
        rm[1],
        momentum * 6.0
    );
    // batch var is 0, so running_var = (1-m)*1.0 + m*0 = 0.9
    assert!(
        (rv[0] - 0.9).abs() < 1e-5,
        "running_var[0]={} expected 0.9",
        rv[0]
    );

    // Eval: running stats must be frozen.
    vearo::set_training(false);
    let before_m = bn.running_mean.borrow().to_vec_f32();
    let before_v = bn.running_var.borrow().to_vec_f32();
    let _ = bn.forward(&x);
    let after_m = bn.running_mean.borrow().to_vec_f32();
    let after_v = bn.running_var.borrow().to_vec_f32();
    assert_eq!(before_m, after_m, "eval mode must not update running_mean");
    assert_eq!(before_v, after_v, "eval mode must not update running_var");
}
