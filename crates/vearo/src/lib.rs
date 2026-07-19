//! Vearo: a memory-efficient, Rust-native deep learning training framework.
//!
//! Re-exports all workspace crates under one roof. Check out ROADMAP.md for the plan.

/// Autograd engine.
pub use vearo_autograd as autograd;
/// CPU execution backend.
pub use vearo_backend_cpu as backend_cpu;
/// CUDA execution backend. Present only with the `cuda` feature.
#[cfg(feature = "cuda")]
pub use vearo_backend_cuda as backend_cuda;
/// Core vocabulary and types.
pub use vearo_core as core;
/// Neural network layers and modules.
pub use vearo_nn as nn;
/// Optimizer implementations.
pub use vearo_optim as optim;

/// Terminal dashboard for training runs.
pub mod checkpoint;
pub mod metrics;
pub mod tui;

// Hoisted primitives for convenience.
pub use vearo_core::{DType, Device, Shape, Tensor};
/// Training vs evaluation mode control (affects layers like dropout).
pub use vearo_core::{is_training, set_training};
/// Recomputation state, consulted by layers with side effects so that a
/// checkpointed block replayed during backward does not apply them twice.
pub use vearo_core::{is_recomputing, next_rng_counter, rng_counter, set_recomputing, set_rng_counter};
/// Activation checkpointing.
pub use vearo_autograd::checkpoint;

/// Initializes the backends and the autograd engine.
///
/// Registers the CPU backend always, and the CUDA backend when the `cuda`
/// feature is enabled. Without that feature Vearo is CPU-only and
/// `Device::Cuda` has no registered backend, so dispatching to it fails rather
/// than silently computing on the wrong device.
///
/// Idempotent - safe to call more than once.
pub fn init() {
    vearo_backend_cpu::init();
    #[cfg(feature = "cuda")]
    vearo_backend_cuda::init();
    vearo_autograd::init();
}

/// Returns whether this build has the CUDA backend compiled in.
///
/// Lets a caller pick a device at runtime without a `cfg` of its own.
#[must_use]
pub const fn cuda_available() -> bool {
    cfg!(feature = "cuda")
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
