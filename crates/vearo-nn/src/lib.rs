//! Neural network modules.
#![allow(
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops
)]

use vearo_core::Tensor;

/// A simple deterministic random number generator.
///
/// Uses an LCG or Xorshift structure to avoid pulling in external `rand` dependencies.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Creates a new SimpleRng with a given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // Ensure state is non-zero
        let state = if seed == 0 {
            0x1234_5678_9ABC_DEF0
        } else {
            seed
        };
        Self { state }
    }

    /// Generates the next random u64 value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generates the next random f32 in the range [0.0, 1.0).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    /// Generates the next random f32 in the range [low, high).
    pub fn next_uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
}

/// The base trait for all neural network modules.
pub trait Module {
    /// Forward pass of the module.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Returns a list of all parameters (weights, biases) that require gradients.
    fn parameters(&self) -> Vec<Tensor>;
}

/// A fully connected linear layer: `y = x W^T + b`.
pub struct Linear {
    /// The weight tensor of shape `[out_features, in_features]`.
    pub weight: Tensor,
    /// The bias tensor of shape `[out_features]`.
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Creates a new Linear layer.
    ///
    /// The weights and biases are initialized using a uniform distribution
    /// in range `[-bound, bound]` where `bound = 1.0 / sqrt(in_features)`.
    ///
    /// # Panics
    /// Panics if in_features is 0.
    #[must_use]
    pub fn new(in_features: usize, out_features: usize, bias: bool, seed: u64) -> Self {
        assert!(in_features > 0, "in_features must be greater than 0");
        let mut rng = SimpleRng::new(seed);
        let bound = 1.0 / (in_features as f32).sqrt();

        let numel_w = out_features * in_features;
        let mut w_data = vec![0.0; numel_w];
        for val in &mut w_data {
            *val = rng.next_uniform(-bound, bound);
        }
        let weight = Tensor::from_f32(&w_data, [out_features, in_features]);
        weight.set_requires_grad(true);

        let bias_tensor = if bias {
            let mut b_data = vec![0.0; out_features];
            for val in &mut b_data {
                *val = rng.next_uniform(-bound, bound);
            }
            let b = Tensor::from_f32(&b_data, [out_features]);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_tensor,
        }
    }

    /// Move the layer parameters to a different device.
    #[must_use]
    pub fn to(&self, device: vearo_core::Device) -> Self {
        let weight = self.weight.to(device);
        let bias = self.bias.as_ref().map(|b| b.to(device));
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let w_t = self.weight.transpose(0, 1);
        let out = input.matmul(&w_t);
        if let Some(ref bias) = self.bias {
            out.add(bias)
        } else {
            out
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// A 2D convolution layer (NCHW input, OIHW weight).
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv2d {
    /// Creates a new `Conv2d` layer.
    ///
    /// Weights use a uniform init in `[-bound, bound]` with
    /// `bound = 1/sqrt(in_channels * kernel * kernel)` (PyTorch's default).
    ///
    /// # Panics
    /// Panics if `in_channels` or `kernel` is 0.
    #[must_use]
    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        seed: u64,
    ) -> Self {
        assert!(
            in_channels > 0 && kernel > 0,
            "in_channels and kernel must be > 0"
        );
        let mut rng = SimpleRng::new(seed);
        let fan_in = in_channels * kernel * kernel;
        let bound = 1.0 / (fan_in as f32).sqrt();

        let mut w_data = vec![0.0; out_channels * fan_in];
        for val in &mut w_data {
            *val = rng.next_uniform(-bound, bound);
        }
        let weight = Tensor::from_f32(&w_data, [out_channels, in_channels, kernel, kernel]);
        weight.set_requires_grad(true);

        let bias_tensor = if bias {
            let mut b_data = vec![0.0; out_channels];
            for val in &mut b_data {
                *val = rng.next_uniform(-bound, bound);
            }
            let b = Tensor::from_f32(&b_data, [out_channels]);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_tensor,
            stride,
            padding,
        }
    }

    /// Move the layer parameters to a different device.
    #[must_use]
    pub fn to(&self, device: vearo_core::Device) -> Self {
        Self {
            weight: self.weight.to(device),
            bias: self.bias.as_ref().map(|b| b.to(device)),
            stride: self.stride,
            padding: self.padding,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let out_c = self.weight.shape().dims()[0];
        let bias = self
            .bias
            .clone()
            .unwrap_or_else(|| Tensor::zeros([out_c], vearo_core::DType::F32));
        input.conv2d(&self.weight, &bias, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// An embedding lookup layer.
pub struct Embedding {
    /// The weight tensor of shape `[vocab_size, embedding_dim]`.
    pub weight: Tensor,
}

impl Embedding {
    /// Creates a new Embedding layer.
    ///
    /// Weights are initialized using a uniform distribution
    /// in range `[-bound, bound]` where `bound = 1.0 / sqrt(embedding_dim)`.
    #[must_use]
    pub fn new(vocab_size: usize, embedding_dim: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let bound = 1.0 / (embedding_dim as f32).sqrt();
        let mut w_data = vec![0.0; vocab_size * embedding_dim];
        for val in &mut w_data {
            *val = rng.next_uniform(-bound, bound);
        }
        let weight = Tensor::from_f32(&w_data, [vocab_size, embedding_dim]);
        weight.set_requires_grad(true);
        Self { weight }
    }

    /// Forward pass performing embedding lookup.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.embedding(&self.weight)
    }

    /// Returns the embedding weight parameter.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

/// Layer normalization module.
pub struct LayerNorm {
    /// The learnable scale parameters ($\gamma$).
    pub weight: Tensor,
    /// The learnable shift parameters ($\beta$).
    pub bias: Tensor,
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl LayerNorm {
    /// Creates a new LayerNorm layer.
    ///
    /// The weights ($\gamma$) are initialized to 1s, and biases ($\beta$) to 0s.
    #[must_use]
    pub fn new(normalized_dim: usize, eps: f32) -> Self {
        let weight = Tensor::from_f32(&vec![1.0; normalized_dim], [normalized_dim]);
        let bias = Tensor::from_f32(&vec![0.0; normalized_dim], [normalized_dim]);
        weight.set_requires_grad(true);
        bias.set_requires_grad(true);
        Self { weight, bias, eps }
    }

    /// Forward pass performing layer normalization.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.layernorm(&self.weight, &self.bias, self.eps)
    }

    /// Returns the learnable LayerNorm parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// Multi-head causal attention layer.
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

impl MultiHeadAttention {
    /// Creates a new MultiHeadAttention module.
    ///
    /// # Panics
    /// Panics if `n_embd` is not divisible by `n_head`.
    #[must_use]
    pub fn new(n_embd: usize, n_head: usize, seed: u64) -> Self {
        assert_eq!(n_embd % n_head, 0, "n_embd must be divisible by n_head");

        let q_proj = Linear::new(n_embd, n_embd, true, seed);
        let k_proj = Linear::new(n_embd, n_embd, true, seed + 1);
        let v_proj = Linear::new(n_embd, n_embd, true, seed + 2);
        let out_proj = Linear::new(n_embd, n_embd, true, seed + 3);

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_head,
            n_embd,
        }
    }

    /// Forward pass of multi-head attention.
    /// Input shape: `[B, S, D]`.
    /// Output shape: `[B, S, D]`.
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let shape = x.shape().dims();
        let b = shape[0];
        let s = shape[1];
        let d = self.n_embd;
        let h = self.n_head;
        let d_k = d / h;

        // 1. Projects Q, K, V
        let q = self.q_proj.forward(x); // [B, S, D]
        let k = self.k_proj.forward(x); // [B, S, D]
        let v = self.v_proj.forward(x); // [B, S, D]

        // 2. Reshape to [B, S, H, D_k] and transpose to [B, H, S, D_k]
        let q = q.reshape([b, s, h, d_k]).transpose(1, 2);
        let k = k.reshape([b, s, h, d_k]).transpose(1, 2);
        let v = v.reshape([b, s, h, d_k]).transpose(1, 2);

        // 3. Compute attention scores: Q K^T / sqrt(d_k)
        let k_t = k.transpose(2, 3); // [B, H, D_k, S]
        let scale = Tensor::from_f32(&[1.0 / (d_k as f32).sqrt()], [1]);
        let mut scores = q.matmul(&k_t).mul(&scale); // [B, H, S, S]

        // 4. Apply mask if present (adds mask elementwise)
        if let Some(m) = mask {
            scores = scores.add(m);
        }

        // 5. Softmax attention probabilities
        let probs = scores.softmax(3);

        // 6. Compute output: probs * V -> [B, H, S, D_k]
        let out = probs.matmul(&v);

        // 7. Transpose back: [B, H, S, D_k] -> [B, S, H, D_k] -> reshape [B, S, D]
        let out = out.transpose(1, 2).reshape([b, s, d]);

        // 8. Output projection
        self.out_proj.forward(&out)
    }

    /// Returns MHA parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}

/// A Transformer decoder block (pre-LN style).
pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
}

impl TransformerBlock {
    /// Creates a new Transformer block.
    #[must_use]
    pub fn new(n_embd: usize, n_head: usize, mlp_dim: usize, seed: u64) -> Self {
        let ln_1 = LayerNorm::new(n_embd, 1e-5);
        let attn = MultiHeadAttention::new(n_embd, n_head, seed);
        let ln_2 = LayerNorm::new(n_embd, 1e-5);

        let mlp_fc1 = Linear::new(n_embd, mlp_dim, true, seed + 10);
        let mlp_fc2 = Linear::new(mlp_dim, n_embd, true, seed + 20);

        Self {
            ln_1,
            attn,
            ln_2,
            mlp_fc1,
            mlp_fc2,
        }
    }

    /// Forward pass of the transformer block: Pre-LN architecture.
    #[must_use]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let ln_x1 = self.ln_1.forward(x);
        let attn_out = self.attn.forward(&ln_x1, mask);
        let x = x.add(&attn_out);

        let ln_x2 = self.ln_2.forward(&x);
        let mlp_out = self.mlp_fc1.forward(&ln_x2).gelu();
        let mlp_out = self.mlp_fc2.forward(&mlp_out);
        x.add(&mlp_out)
    }

    /// Returns all block parameters.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.ln_1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln_2.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

/// A simple decoder-only Generative Pretrained Transformer (GPT).
pub struct SimpleGPT {
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    n_embd: usize,
}

impl SimpleGPT {
    /// Creates a new SimpleGPT model.
    #[must_use]
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        n_embd: usize,
        n_head: usize,
        n_layer: usize,
        mlp_dim: usize,
        seed: u64,
    ) -> Self {
        let token_embedding = Embedding::new(vocab_size, n_embd, seed);
        let position_embedding = Embedding::new(max_seq_len, n_embd, seed + 1);

        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            blocks.push(TransformerBlock::new(
                n_embd,
                n_head,
                mlp_dim,
                seed + 10 + (i as u64) * 100,
            ));
        }

        let ln_f = LayerNorm::new(n_embd, 1e-5);
        let lm_head = Linear::new(n_embd, vocab_size, false, seed + 2);

        Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
            lm_head,
            n_embd,
        }
    }

    /// Forward pass of the SimpleGPT.
    /// Returns: `(logits, loss)` where `logits` has shape `[B * S, V]` and `loss` is a scalar loss tensor.
    #[must_use]
    pub fn forward(&self, x: &Tensor, targets: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        let shape = x.shape().dims();
        let b = shape[0];
        let s = shape[1];

        // 1. Generate position indices: [0, 1, 2, ..., S-1] for each batch row
        let mut pos_data = vec![0.0f32; b * s];
        for row in 0..b {
            for col in 0..s {
                #[allow(clippy::cast_precision_loss)]
                let pos_val = col as f32;
                pos_data[row * s + col] = pos_val;
            }
        }
        let pos_tensor = Tensor::from_f32(&pos_data, [b, s]);

        // 2. Lookup Embeddings
        let tok_emb = self.token_embedding.forward(x);
        let pos_emb = self.position_embedding.forward(&pos_tensor);
        let h = tok_emb.add(&pos_emb); // [B, S, D]

        // 3. Create causal mask (adds large negative values above diagonal)
        let mut mask_data = vec![0.0f32; s * s];
        for row in 0..s {
            for col in 0..s {
                if col > row {
                    mask_data[row * s + col] = -1e9;
                }
            }
        }
        let mask = Tensor::from_f32(&mask_data, [s, s]);

        // 4. Run through transformer blocks
        let mut h_current = h;
        for block in &self.blocks {
            h_current = block.forward(&h_current, Some(&mask));
        }

        // 5. Final layer normalization
        let h_ln = self.ln_f.forward(&h_current);

        // 6. LM head projection
        let h_flat = h_ln.reshape([b * s, self.n_embd]);
        let logits = self.lm_head.forward(&h_flat);

        // 7. Compute loss if targets are provided
        let loss = targets.map(|target_tensor| {
            let targets_flat = target_tensor.reshape([b * s]);
            logits.cross_entropy(&targets_flat)
        });

        (logits, loss)
    }

    /// Returns all learnable parameters in the model.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        params.extend(self.position_embedding.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_f.parameters());
        params.extend(self.lm_head.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        vearo_backend_cpu::init();
        let layer = Linear::new(3, 2, true, 42);
        assert_eq!(layer.weight.shape().dims(), &[2, 3]);
        assert!(layer.bias.is_some());
        assert_eq!(layer.bias.as_ref().unwrap().shape().dims(), &[2]);

        let input = Tensor::from_f32(&[1.0, 2.0, 3.0], [1, 3]);
        let output = layer.forward(&input);
        assert_eq!(output.shape().dims(), &[1, 2]);
    }
}
