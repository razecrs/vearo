//! Vearo: a memory-efficient, Rust-native deep learning training framework.
//!
//! Re-exports all workspace crates under one roof. Check out ROADMAP.md for the plan.

/// Autograd engine.
pub use vearo_autograd as autograd;
/// CPU execution backend.
pub use vearo_backend_cpu as backend_cpu;
/// CUDA execution backend.
pub use vearo_backend_cuda as backend_cuda;
/// Core vocabulary and types.
pub use vearo_core as core;
/// Neural network layers and modules.
pub use vearo_nn as nn;
/// Optimizer implementations.
pub use vearo_optim as optim;

/// Terminal dashboard for training runs.
pub mod metrics;
pub mod tui;

// Hoisted primitives for convenience.
pub use vearo_core::{DType, Device, Shape, Tensor};
/// Training vs evaluation mode control (affects layers like dropout).
pub use vearo_core::{is_training, set_training};

/// Initializes the backend and autograd engine. Registers both CPU and CUDA backends.
///
/// Idempotent - safe to call more than once.
pub fn init() {
    vearo_backend_cpu::init();
    vearo_backend_cuda::init();
    vearo_autograd::init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_operators_and_dispatched_methods() {
        init();

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], [3]);
        let y = Tensor::from_f32(&[10.0, 20.0, 30.0], [3]);

        // Method call check
        let z_method = x.add(&y);

        // Operator check
        let z_op = &x + &y;

        // Verify equality
        assert_eq!(z_method.shape().dims(), &[3]);
        assert_eq!(z_op.shape().dims(), &[3]);

        let p = x.reshape([1, 3]).matmul(&y.reshape([3, 1]));
        assert_eq!(p.shape().dims(), &[1, 1]);
    }
}
