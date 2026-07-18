//! Minimal end-to-end example: train a small MLP on synthetic data.
//!
//! No dataset download, no GPU needed at runtime (this runs on the CPU backend).
//!
//!     cargo run --release -p vearo --example train_mlp
//!
//! It learns the function y = sin(3*x0) + 0.5*x1^2 from random samples, and prints
//! the loss falling. If the loss does not fall, something in the stack is broken.

use vearo::nn::{Linear, Module};
use vearo::{Device, Tensor};

const N: usize = 512; // samples
const IN: usize = 2; // input features
const HIDDEN: usize = 32;
const EPOCHS: usize = 200;

fn main() {
    // Register the CPU backend and the autograd engine. (vearo::init() would also
    // bring up CUDA; we only need the CPU here.)
    vearo::backend_cpu::init();
    vearo::autograd::init();
    vearo::set_training(true);

    // ---- synthetic dataset -------------------------------------------------
    let mut rng = vearo::nn::SimpleRng::new(7);
    let mut xs = Vec::with_capacity(N * IN);
    let mut ys = Vec::with_capacity(N);
    for _ in 0..N {
        let x0 = rng.next_uniform(-1.0, 1.0);
        let x1 = rng.next_uniform(-1.0, 1.0);
        xs.push(x0);
        xs.push(x1);
        ys.push((3.0 * x0).sin() + 0.5 * x1 * x1);
    }
    let x = Tensor::from_f32(&xs, [N, IN]).to(Device::Cpu);
    let y = Tensor::from_f32(&ys, [N, 1]).to(Device::Cpu);

    // ---- model -------------------------------------------------------------
    let fc1 = Linear::new(IN, HIDDEN, true, 1);
    let fc2 = Linear::new(HIDDEN, HIDDEN, true, 2);
    let fc3 = Linear::new(HIDDEN, 1, true, 3);

    let mut params = fc1.parameters();
    params.extend(fc2.parameters());
    params.extend(fc3.parameters());

    // AdamW: (params, lr, beta1, beta2, eps, weight_decay)
    let mut opt = vearo::optim::AdamW::new(params, 5e-3, 0.9, 0.999, 1e-8, 0.0);

    let forward = |input: &Tensor| {
        let h = fc1.forward(input).relu();
        let h = fc2.forward(&h).relu();
        fc3.forward(&h)
    };

    // ---- training loop -----------------------------------------------------
    println!("training a {IN}-{HIDDEN}-{HIDDEN}-1 MLP on {N} synthetic samples\n");
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for epoch in 0..EPOCHS {
        // Clear last step's gradients and start a fresh tape.
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        let pred = forward(&x);
        let diff = pred.sub(&y);
        let loss = diff.mul(&diff).mean(0, false); // mean squared error

        loss.backward();
        opt.step();

        let l = loss.to_vec_f32()[0];
        if epoch == 0 {
            first_loss = l;
        }
        last_loss = l;

        if epoch % 20 == 0 || epoch == EPOCHS - 1 {
            println!("epoch {epoch:3}  loss {l:.6}");
        }
    }

    println!("\nloss went {first_loss:.6} -> {last_loss:.6}");
    assert!(
        last_loss < first_loss,
        "loss did not decrease - the training stack is broken"
    );
    println!("done. the stack works end to end: forward, autograd, optimizer.");
}
