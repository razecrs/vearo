//! Optimizers.
#![allow(clippy::doc_markdown)]

use vearo_core::Tensor;

/// Stochastic Gradient Descent (SGD) optimizer.
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: Vec<Vec<f32>>,
}

impl SGD {
    /// Creates a new SGD optimizer.
    #[must_use]
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32, weight_decay: f32) -> Self {
        let mut velocities = Vec::new();
        for p in &params {
            velocities.push(vec![0.0; p.shape().numel()]);
        }
        Self {
            params,
            lr,
            momentum,
            weight_decay,
            velocities,
        }
    }

    /// Performs a single optimization step.
    pub fn step(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            if let Some(grad) = p.grad() {
                let grad_contiguous = grad.contiguous();
                p.sgd_step(
                    &grad_contiguous,
                    &mut self.velocities[i],
                    self.lr,
                    self.momentum,
                    self.weight_decay,
                );
            }
        }
    }

    /// The current learning rate.
    #[must_use]
    pub const fn lr(&self) -> f32 {
        self.lr
    }

    /// Sets the learning rate (e.g. driven by a schedule between steps).
    pub const fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
pub struct AdamW {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: u32,
}

impl AdamW {
    /// Creates a new AdamW optimizer.
    #[must_use]
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for p in &params {
            let numel = p.shape().numel();
            m.push(vec![0.0; numel]);
            v.push(vec![0.0; numel]);
        }
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }

    /// Performs a single optimization step.
    pub fn step(&mut self) {
        self.t += 1;
        for (i, p) in self.params.iter().enumerate() {
            if let Some(grad) = p.grad() {
                let grad_contiguous = grad.contiguous();
                p.adamw_step(
                    &grad_contiguous,
                    &mut self.m[i],
                    &mut self.v[i],
                    self.t,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                );
            }
        }
    }

    /// The current learning rate.
    #[must_use]
    pub const fn lr(&self) -> f32 {
        self.lr
    }

    /// Sets the learning rate (e.g. driven by a schedule between steps).
    pub const fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Quantized state block-wise representation.
pub struct QuantizedState {
    data: Vec<i8>,
    scales: Vec<f32>,
    block_size: usize,
}

impl QuantizedState {
    /// Creates a new `QuantizedState` initialized with zeros.
    #[must_use]
    pub fn new(numel: usize, block_size: usize) -> Self {
        let num_blocks = numel.div_ceil(block_size);
        Self {
            data: vec![0; numel],
            scales: vec![0.0; num_blocks],
            block_size,
        }
    }

    /// Quantizes the source float slice into 8-bit block-wise representation in-place.
    #[allow(clippy::needless_range_loop, clippy::cast_possible_truncation)]
    pub fn quantize(&mut self, src: &[f32]) {
        let numel = src.len();
        let block_size = self.block_size;

        for (b_idx, scale) in self.scales.iter_mut().enumerate() {
            let start = b_idx * block_size;
            let end = std::cmp::min(start + block_size, numel);
            if start >= end {
                continue;
            }

            let block = &src[start..end];
            let mut max_val = 0.0f32;
            for &val in block {
                let abs_val = val.abs();
                if abs_val > max_val {
                    max_val = abs_val;
                }
            }

            *scale = max_val / 127.0;

            if max_val > 0.0 {
                let inv_scale = 127.0 / max_val;
                for i in start..end {
                    let val = src[i];
                    let q_val = (val * inv_scale).round();
                    self.data[i] = q_val.clamp(-127.0, 127.0) as i8;
                }
            } else {
                for i in start..end {
                    self.data[i] = 0;
                }
            }
        }
    }

    /// Dequantizes the 8-bit block-wise representation back to floats in-place.
    #[allow(clippy::needless_range_loop)]
    pub fn dequantize(&self, dst: &mut [f32]) {
        let numel = dst.len();
        let block_size = self.block_size;

        for (b_idx, &scale) in self.scales.iter().enumerate() {
            let start = b_idx * block_size;
            let end = std::cmp::min(start + block_size, numel);

            for i in start..end {
                dst[i] = f32::from(self.data[i]) * scale;
            }
        }
    }
}

/// 8-bit AdamW optimizer with block-wise quantized optimizer states (m and v).
/// Saves 75% memory on optimizer states.
pub struct AdamW8bit {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<QuantizedState>,
    v: Vec<QuantizedState>,
    t: u32,
}

impl AdamW8bit {
    /// Creates a new 8-bit AdamW optimizer.
    #[must_use]
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for p in &params {
            let numel = p.shape().numel();
            m.push(QuantizedState::new(numel, 256));
            v.push(QuantizedState::new(numel, 256));
        }
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }

    /// Performs a single optimization step.
    pub fn step(&mut self) {
        self.t += 1;
        for (i, p) in self.params.iter().enumerate() {
            if let Some(grad) = p.grad() {
                let grad_contiguous = grad.contiguous();
                let numel = p.shape().numel();

                // Allocate temporary float buffers
                let mut m_f32 = vec![0.0f32; numel];
                let mut v_f32 = vec![0.0f32; numel];

                // Dequantize states
                self.m[i].dequantize(&mut m_f32);
                self.v[i].dequantize(&mut v_f32);

                // Perform update step
                p.adamw_step(
                    &grad_contiguous,
                    &mut m_f32,
                    &mut v_f32,
                    self.t,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                );

                // Re-quantize updated states
                self.m[i].quantize(&m_f32);
                self.v[i].quantize(&v_f32);
            }
        }
    }

    /// The current learning rate.
    #[must_use]
    pub const fn lr(&self) -> f32 {
        self.lr
    }

    /// Sets the learning rate.
    pub const fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Learning-rate schedule: linear warmup, then cosine decay to `min_lr`.
///
/// Matches the nanoGPT reference schedule so training runs stay comparable:
///   - `step < warmup_steps`: `lr = base_lr * (step + 1) / (warmup_steps + 1)`
///   - `step > total_steps`:  `lr = min_lr`
///   - otherwise: cosine decay from `base_lr` down to `min_lr`
pub struct CosineSchedule {
    base_lr: f32,
    min_lr: f32,
    warmup_steps: u32,
    total_steps: u32,
}

impl CosineSchedule {
    /// Creates a warmup + cosine-decay schedule.
    #[must_use]
    pub const fn new(base_lr: f32, min_lr: f32, warmup_steps: u32, total_steps: u32) -> Self {
        Self {
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }

    /// Learning rate at a given (0-indexed) step.
    // Non-FMA arithmetic is intentional: it matches nanoGPT's plain
    // `min_lr + coeff * (base - min)` so the schedule stays bit-comparable to the
    // reference. `mul_add` would round differently.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
    pub fn lr_at(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            return self.base_lr * (step as f32 + 1.0) / (self.warmup_steps as f32 + 1.0);
        }
        if step > self.total_steps {
            return self.min_lr;
        }
        // Cosine decay. `denom.max(1)` guards a degenerate warmup == total config.
        let denom = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let progress = (step - self.warmup_steps) as f32 / denom as f32;
        let coeff = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + coeff * (self.base_lr - self.min_lr)
    }
}

/// Clips the global L2 norm of `params`' gradients to `max_norm`, in place.
///
/// Returns the total gradient norm *before* clipping (the PyTorch
/// `clip_grad_norm_` convention). Parameters without a gradient are skipped;
/// call after `backward()` and before `optimizer.step()`.
///
/// # Panics
/// Panics if a gradient tensor is non-contiguous.
#[must_use]
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) -> f32 {
    let grads: Vec<Tensor> = params.iter().filter_map(Tensor::grad).collect();

    let mut total_sq = 0.0f32;
    for g in &grads {
        assert!(g.is_contiguous(), "clip_grad_norm expects contiguous grads");
        for i in 0..g.shape().numel() {
            let v = g.get_f32(i);
            total_sq = v.mul_add(v, total_sq);
        }
    }
    let total_norm = total_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for g in &grads {
            for i in 0..g.shape().numel() {
                g.set_f32(i, g.get_f32(i) * scale);
            }
        }
    }

    total_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_step() {
        vearo_backend_cpu::init();
        let x = Tensor::from_f32(&[1.0, 2.0], [2]);
        x.set_requires_grad(true);

        let grad = Tensor::from_f32(&[0.1, 0.2], [2]);
        // hand-set grad, check sgd_step directly (no backward needed)
        let mut velocity = vec![0.0, 0.0];
        x.sgd_step(&grad, &mut velocity, 0.1, 0.9, 0.01);

        // Expected value: x_new = x - lr * (momentum * v_old + grad + wd * x)
        // x_new = 1.0 - 0.1 * (0.0 + 0.1 + 0.01 * 1.0) = 1.0 - 0.1 * 0.11 = 0.989
        assert!((x.get_f32(0) - 0.989).abs() < 1e-5);
    }

    #[test]
    fn test_set_lr() {
        let mut opt = AdamW::new(vec![], 0.1, 0.9, 0.999, 1e-8, 0.0);
        assert!((opt.lr() - 0.1).abs() < 1e-6);
        opt.set_lr(0.05);
        assert!((opt.lr() - 0.05).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::suboptimal_flops)] // mirror the schedule's non-FMA formula
    fn test_cosine_schedule() {
        let s = CosineSchedule::new(0.1, 0.01, 10, 100);

        // Warmup ramps from ~0 up toward base over `warmup_steps`.
        assert!((s.lr_at(0) - 0.1 * 1.0 / 11.0).abs() < 1e-6);
        assert!((s.lr_at(9) - 0.1 * 10.0 / 11.0).abs() < 1e-6);

        // First post-warmup step (decay ratio 0) is exactly base_lr.
        assert!((s.lr_at(10) - 0.1).abs() < 1e-6);

        // Midpoint of decay: progress = 45/90 = 0.5 -> coeff 0.5.
        let expected_mid = 0.01 + 0.5 * (0.1 - 0.01);
        assert!((s.lr_at(55) - expected_mid).abs() < 1e-6);

        // End of decay hits min_lr; beyond `total_steps` stays there.
        assert!((s.lr_at(100) - 0.01).abs() < 1e-6);
        assert!((s.lr_at(200) - 0.01).abs() < 1e-6);

        // Warmup increases, decay decreases.
        assert!(s.lr_at(1) > s.lr_at(0));
        assert!(s.lr_at(60) < s.lr_at(50));
    }

    #[test]
    fn test_clip_grad_norm() {
        vearo_backend_cpu::init();
        vearo_autograd::init();

        let x = Tensor::from_f32(&[3.0, 4.0], [2]);
        x.set_requires_grad(true);
        // loss = sum(x^2) -> grad = 2x = [6, 8], so the global norm is 10.
        let loss = x.mul(&x).sum(0, false);
        loss.backward();

        let norm = clip_grad_norm(std::slice::from_ref(&x), 5.0);
        assert!(
            (norm - 10.0).abs() < 1e-4,
            "pre-clip norm should be 10, got {norm}"
        );

        // Scaled by 5/10 = 0.5 -> [3, 4].
        let g = x.grad().unwrap();
        assert!((g.get_f32(0) - 3.0).abs() < 1e-3);
        assert!((g.get_f32(1) - 4.0).abs() < 1e-3);
    }

    #[test]
    fn test_adamw_8bit_step() {
        vearo_backend_cpu::init();
        vearo_autograd::init();

        let w_data = vec![1.0f32, 2.0, 3.0, 4.0];
        
        // Standard AdamW
        let w_std = Tensor::from_f32(&w_data, [4]);
        w_std.set_requires_grad(true);
        let mut opt_std = AdamW::new(vec![w_std.clone()], 0.1, 0.9, 0.999, 1e-8, 0.01);

        // 8-bit AdamW
        let w_8bit = Tensor::from_f32(&w_data, [4]);
        w_8bit.set_requires_grad(true);
        let mut opt_8bit = AdamW8bit::new(vec![w_8bit.clone()], 0.1, 0.9, 0.999, 1e-8, 0.01);

        for _ in 0..5 {
            // Standard step
            vearo_autograd::zero_gradients();
            vearo_autograd::reset_active_tape();
            let loss_std = w_std.mul(&w_std).sum(0, false);
            loss_std.backward();
            opt_std.step();

            // 8-bit step
            vearo_autograd::zero_gradients();
            vearo_autograd::reset_active_tape();
            let loss_8bit = w_8bit.mul(&w_8bit).sum(0, false);
            loss_8bit.backward();
            opt_8bit.step();
        }

        // Verify that 8-bit AdamW matches standard AdamW within a tiny quantization error tolerance (e.g. 1e-2)
        let val_std = w_std.to_vec_f32();
        let val_8bit = w_8bit.to_vec_f32();
        println!("Std AdamW weights: {val_std:?}");
        println!("8-bit AdamW weights: {val_8bit:?}");
        for i in 0..4 {
            assert!((val_std[i] - val_8bit[i]).abs() < 1e-2);
        }
    }
}
