//! Eager reverse-mode autograd tape.
//!
//! The tape is a DAG, which is the same graph the capture layer
//! will optimize in Phase 8. Ops and `.backward()` land in Phase 3.

use std::cell::RefCell;
use std::collections::HashMap;
use vearo_core::{
    DType, Shape, StorageId, Tensor, is_autograd_enabled, register_backward_hook,
    register_grad_hook, register_record_op, set_autograd_enabled,
};

/// Type of operation recorded on the tape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpType {
    /// Elementwise addition.
    Add,
    /// Elementwise subtraction.
    Sub,
    /// Elementwise multiplication.
    Mul,
    /// Elementwise division.
    Div,
    /// Matrix multiplication.
    Matmul,
    /// Tensor reshape.
    Reshape,
    /// Swapping dimensions.
    Transpose {
        /// First dimension index.
        dim0: usize,
        /// Second dimension index.
        dim1: usize,
    },
    /// Permuting dimensions.
    Permute {
        /// Target axis ordering.
        axes: Vec<usize>,
    },
    /// Elementwise `ReLU`.
    Relu,
    /// Elementwise `GELU`.
    Gelu,
    /// Softmax over a single axis.
    Softmax {
        /// The softmax axis.
        dim: usize,
    },
    /// Layer normalization.
    LayerNorm {
        /// Numerical stability epsilon represented as bits.
        eps_bits: u32,
    },
    /// Embedding lookup.
    Embedding,
    /// Categorical cross-entropy loss.
    CrossEntropy,
    /// Sum over a single axis.
    Sum {
        /// The reduced axis.
        dim: usize,
        /// Whether the reduced axis was kept as size 1.
        keep_dim: bool,
    },
    /// Mean over a single axis.
    Mean {
        /// The reduced axis.
        dim: usize,
        /// Whether the reduced axis was kept as size 1.
        keep_dim: bool,
    },
}

/// A node in the autograd computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// The unique identifier of this node.
    pub id: u32,
    /// The type of operation performed.
    pub op: OpType,
    /// The `NodeId`s of the inputs to this operation (Some(id) if tracked, None otherwise).
    pub inputs: Vec<Option<u32>>,
    /// Cloned tensor handles of the inputs (needed for gradient calculation).
    pub saved_tensors: Vec<Tensor>,
}

/// The autograd computation tape.
#[derive(Debug, Default)]
pub struct Tape {
    /// Flat vector containing all nodes in order of execution.
    pub nodes: Vec<Node>,
    /// Bumped on every `reset` so node tokens still held by leaf tensors from a
    /// previous tape become stale and are ignored.
    pub generation: u32,
}

impl Tape {
    /// Packs a `(generation, index)` pair into an opaque node token.
    const fn pack(generation: u32, index: u32) -> u64 {
        ((generation as u64) << 32) | index as u64
    }

    /// Unpacks a node token into its `(generation, index)` pair.
    const fn unpack(token: u64) -> (u32, u32) {
        #[allow(clippy::cast_possible_truncation)]
        ((token >> 32) as u32, token as u32)
    }

    /// Pushes a new node onto the tape, returning the output node's packed token.
    #[allow(clippy::cast_possible_truncation)]
    pub fn push(&mut self, op: OpType, inputs: &[&Tensor]) -> u64 {
        let mut input_ids = Vec::with_capacity(inputs.len());
        for &input in inputs {
            // Honor an existing token only if it belongs to the *current* tape;
            // a token from an earlier generation means "fresh leaf this pass".
            let live_index = input.node_id().and_then(|tok| {
                let (token_gen, idx) = Self::unpack(tok);
                (token_gen == self.generation).then_some(idx)
            });

            if let Some(idx) = live_index {
                input_ids.push(Some(idx));
            } else if input.requires_grad() {
                let leaf_id = self.nodes.len() as u32;
                self.nodes.push(Node {
                    id: leaf_id,
                    op: OpType::Add, // leaf dummy op
                    inputs: vec![],
                    saved_tensors: vec![],
                });
                input.set_node_id(Some(Self::pack(self.generation, leaf_id)));
                input_ids.push(Some(leaf_id));
            } else {
                input_ids.push(None);
            }
        }

        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            id,
            op,
            inputs: input_ids,
            saved_tensors: inputs.iter().map(|&t| t.clone()).collect(),
        });
        Self::pack(self.generation, id)
    }

    /// Clears the tape and bumps the generation so every outstanding node token
    /// (still held by leaf tensors) is invalidated.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.generation = self.generation.wrapping_add(1);
    }
}

thread_local! {
    /// Thread-local active autograd tape.
    pub static TAPE: RefCell<Tape> = RefCell::new(Tape::default());

    /// Thread-local storage for computed gradients after backward pass.
    pub static GRADIENTS: RefCell<HashMap<StorageId, Tensor>> = RefCell::new(HashMap::default());
}

/// Parses a `"<dim>_<keepbit>"` reduction op-name suffix into `(dim, keep_dim)`.
fn parse_reduce_args(suffix: &str) -> (usize, bool) {
    let parts: Vec<&str> = suffix.split('_').collect();
    let dim = parts[0].parse::<usize>().unwrap();
    let keep_dim = parts[1] == "1";
    (dim, keep_dim)
}

/// Dynamic recording callback function.
fn record_op_callback(op_name: &str, inputs: &[&Tensor], output: &mut Tensor) {
    let op = if op_name == "reshape" {
        OpType::Reshape
    } else if let Some(stripped) = op_name.strip_prefix("transpose_") {
        let parts: Vec<&str> = stripped.split('_').collect();
        let dim0 = parts[0].parse::<usize>().unwrap();
        let dim1 = parts[1].parse::<usize>().unwrap();
        OpType::Transpose { dim0, dim1 }
    } else if let Some(stripped) = op_name.strip_prefix("permute_") {
        let axes: Vec<usize> = stripped
            .split(',')
            .map(|s| s.parse::<usize>().unwrap())
            .collect();
        OpType::Permute { axes }
    } else if let Some(stripped) = op_name.strip_prefix("softmax_") {
        let dim = stripped.parse::<usize>().unwrap();
        OpType::Softmax { dim }
    } else if let Some(stripped) = op_name.strip_prefix("layernorm_") {
        let eps = stripped.parse::<f32>().unwrap();
        OpType::LayerNorm {
            eps_bits: eps.to_bits(),
        }
    } else if let Some(stripped) = op_name.strip_prefix("sum_") {
        let (dim, keep_dim) = parse_reduce_args(stripped);
        OpType::Sum { dim, keep_dim }
    } else if let Some(stripped) = op_name.strip_prefix("mean_") {
        let (dim, keep_dim) = parse_reduce_args(stripped);
        OpType::Mean { dim, keep_dim }
    } else {
        match op_name {
            "add" => OpType::Add,
            "sub" => OpType::Sub,
            "mul" => OpType::Mul,
            "div" => OpType::Div,
            "matmul" => OpType::Matmul,
            "relu" => OpType::Relu,
            "gelu" => OpType::Gelu,
            "embedding" => OpType::Embedding,
            "cross_entropy" => OpType::CrossEntropy,
            _ => panic!("Unknown operation to record: {op_name}"),
        }
    };

    TAPE.with(|tape| {
        let mut tape = tape.borrow_mut();
        let out_id = tape.push(op, inputs);
        output.set_node_id(Some(out_id));
    });
}

/// Generic broadcasting gradient reduction helper.
///
/// Sums `grad` values into a smaller `target_shape` to handle PyTorch-style broadcasting gradients.
fn reduce_to(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    if grad.shape().dims() == target_shape {
        return grad.clone();
    }
    if grad.shape().numel() == 0 {
        return Tensor::zeros(target_shape, grad.dtype());
    }
    let out_numel = target_shape.iter().product::<usize>();
    let mut out_data = vec![0.0; out_numel];

    let grad_shape = grad.shape();
    let mut iter = vearo_core::NdIterator::new(*grad_shape);
    let grad_contiguous = grad.contiguous();
    let grad_vec = grad_contiguous.to_vec_f32();

    let mut i = 0;
    loop {
        let coord = iter.coord();
        let mut target_idx = 0;
        let mut stride = 1;

        let grad_rank = grad_shape.rank();
        let target_rank = target_shape.len();
        assert!(
            grad_rank >= target_rank,
            "Grad rank {grad_rank} must be >= target rank {target_rank}"
        );
        for d in (0..target_rank).rev() {
            let grad_d = grad_rank - target_rank + d;
            let c = coord[grad_d];
            let mapped_c = if target_shape[d] == 1 { 0 } else { c };
            target_idx += mapped_c * stride;
            stride *= target_shape[d];
        }

        let val = grad_vec[i];
        out_data[target_idx] += val;

        i += 1;
        if !iter.step() {
            break;
        }
    }

    Tensor::from_f32(&out_data, target_shape)
}

/// Builds the `ReLU` gradient mask: 1.0 where `x > 0`, else 0.0.
fn relu_grad_mask(x: &Tensor) -> Tensor {
    let xc = x.contiguous();
    let xc_vec = xc.to_vec_f32();
    let mut data = vec![0.0f32; xc.shape().numel()];
    for (i, d) in data.iter_mut().enumerate() {
        if xc_vec[i] > 0.0 {
            *d = 1.0;
        }
    }
    Tensor::from_f32(&data, *xc.shape())
}

/// Builds the `GELU` gradient mask (derivative of GELU tanh approximation).
#[allow(
    clippy::items_after_statements,
    clippy::excessive_precision,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]
fn gelu_grad_mask(x: &Tensor) -> Tensor {
    let xc = x.contiguous();
    let xc_vec = xc.to_vec_f32();
    let mut data = vec![0.0f32; xc.shape().numel()];
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const COEFF: f32 = 0.044_715;
    const DERIV_COEFF: f32 = 0.134_145; // 3 * 0.044715

    for i in 0..data.len() {
        let v = xc_vec[i];
        let v3 = v * v * v;
        let inner = SQRT_2_OVER_PI * (v + COEFF * v3);
        let t = inner.tanh();
        let t_sq = t * t;
        let sech_sq = 1.0 - t_sq;
        let inner_deriv = SQRT_2_OVER_PI * (1.0 + DERIV_COEFF * v * v);

        data[i] = 0.5 * (1.0 + t) + 0.5 * v * sech_sq * inner_deriv;
    }
    Tensor::from_f32(&data, *xc.shape())
}

/// Broadcasts a single-axis reduction's output gradient back to the input shape.
///
/// If the axis was dropped (`keep_dim == false`) it is first reinserted as size 1;
/// adding a zero tensor of the input shape then expands it across the reduced axis
/// (exact - adds 0.0).
fn expand_reduce_grad(
    grad_out: &Tensor,
    input_shape: &Shape,
    dim: usize,
    keep_dim: bool,
) -> Tensor {
    let grad_keep = if keep_dim {
        grad_out.clone()
    } else {
        let mut dims = grad_out.shape().dims().to_vec();
        dims.insert(dim, 1);
        grad_out.reshape(Shape::new(&dims))
    };
    let zeros = Tensor::zeros(*input_shape, DType::F32);
    zeros.add(&grad_keep)
}

/// Dynamic backward pass execution callback.
#[allow(clippy::too_many_lines)]
fn backward_callback(output: &Tensor) {
    // Disable tape recording during backward pass execution
    set_autograd_enabled(false);

    let Some(output_token) = output.node_id() else {
        set_autograd_enabled(true);
        return;
    };
    let (_gen, output_idx) = Tape::unpack(output_token);

    // 1. Fetch nodes from the tape
    let nodes = TAPE.with(|t| t.borrow().nodes.clone());

    // 2. Map from node index to accumulated gradient Tensor
    let mut grad_map: HashMap<u32, Tensor> = HashMap::new();

    // 3. Initialize output gradient to ones
    let out_grad = Tensor::from_f32(&[1.0], *output.shape());
    grad_map.insert(output_idx, out_grad);

    // 4. Backward execution through reverse topological sort (simply reverse tape order)
    for node in nodes.iter().rev() {
        let grad_out = match grad_map.get(&node.id) {
            Some(g) => g.clone(),
            None => continue,
        };

        if node.saved_tensors.is_empty() {
            // Leaf dummy node, no inputs to propagate to
            continue;
        }

        let grads_in = match &node.op {
            OpType::Add => {
                let lhs = &node.saved_tensors[0];
                let rhs = &node.saved_tensors[1];
                let gl = grad_out.clone();
                let gr = grad_out.clone();
                vec![
                    reduce_to(&gl, lhs.shape().dims()),
                    reduce_to(&gr, rhs.shape().dims()),
                ]
            }
            OpType::Sub => {
                let lhs = &node.saved_tensors[0];
                let rhs = &node.saved_tensors[1];
                let gl = grad_out.clone();
                let neg_ones = Tensor::from_f32(&[-1.0], [1]);
                let gr = grad_out.mul(&neg_ones);
                vec![
                    reduce_to(&gl, lhs.shape().dims()),
                    reduce_to(&gr, rhs.shape().dims()),
                ]
            }
            OpType::Mul => {
                let lhs = &node.saved_tensors[0];
                let rhs = &node.saved_tensors[1];
                let gl = grad_out.mul(rhs);
                let gr = grad_out.mul(lhs);
                vec![
                    reduce_to(&gl, lhs.shape().dims()),
                    reduce_to(&gr, rhs.shape().dims()),
                ]
            }
            OpType::Div => {
                let lhs = &node.saved_tensors[0];
                let rhs = &node.saved_tensors[1];
                let gl = grad_out.div(rhs);
                let neg_ones = Tensor::from_f32(&[-1.0], [1]);
                let gr = grad_out.mul(&neg_ones).mul(lhs).div(&rhs.mul(rhs));
                vec![
                    reduce_to(&gl, lhs.shape().dims()),
                    reduce_to(&gr, rhs.shape().dims()),
                ]
            }
            OpType::Matmul => {
                let lhs = &node.saved_tensors[0];
                let rhs = &node.saved_tensors[1];

                let r_lhs = lhs.shape().rank();
                let r_rhs = rhs.shape().rank();

                let lhs_t = lhs.transpose(r_lhs - 2, r_lhs - 1);
                let rhs_t = rhs.transpose(r_rhs - 2, r_rhs - 1);

                let gl = grad_out.matmul(&rhs_t);
                let gr = lhs_t.matmul(&grad_out);

                vec![
                    reduce_to(&gl, lhs.shape().dims()),
                    reduce_to(&gr, rhs.shape().dims()),
                ]
            }
            OpType::Reshape => {
                let lhs = &node.saved_tensors[0];
                let gl = grad_out.reshape(*lhs.shape());
                vec![gl]
            }
            OpType::Transpose { dim0, dim1 } => {
                let gl = grad_out.transpose(*dim0, *dim1);
                vec![gl]
            }
            OpType::Permute { axes } => {
                let mut inv_axes = vec![0; axes.len()];
                for (i, &ax) in axes.iter().enumerate() {
                    inv_axes[ax] = i;
                }
                let gl = grad_out.permute(&inv_axes);
                vec![gl]
            }
            OpType::Relu => {
                let x = &node.saved_tensors[0];
                let mask = relu_grad_mask(x);
                vec![grad_out.mul(&mask)]
            }
            OpType::Gelu => {
                let x = &node.saved_tensors[0];
                let mask = gelu_grad_mask(x);
                vec![grad_out.mul(&mask)]
            }
            OpType::Softmax { dim } => {
                let x = &node.saved_tensors[0];
                let y = x.softmax(*dim);
                let sum_grad_y = grad_out.mul(&y).sum(*dim, true);
                let gl = y.mul(&grad_out.sub(&sum_grad_y));
                vec![gl]
            }
            OpType::LayerNorm { eps_bits } => {
                let x = &node.saved_tensors[0];
                let weight = &node.saved_tensors[1];
                let bias = &node.saved_tensors[2];
                let eps = f32::from_bits(*eps_bits);
                let (gx, gw, gb) = x.layernorm_backward(weight, bias, &grad_out, eps);
                vec![gx, gw, gb]
            }
            OpType::Embedding => {
                let x = &node.saved_tensors[0];
                let weight = &node.saved_tensors[1];
                let gw = x.embedding_backward(weight, &grad_out);
                vec![Tensor::zeros(*x.shape(), vearo_core::DType::F32), gw]
            }
            OpType::CrossEntropy => {
                let logits = &node.saved_tensors[0];
                let targets = &node.saved_tensors[1];
                let gl = logits.cross_entropy_backward(targets, &grad_out);
                vec![gl, Tensor::zeros(*targets.shape(), vearo_core::DType::F32)]
            }
            OpType::Sum { dim, keep_dim } => {
                let x = &node.saved_tensors[0];
                vec![expand_reduce_grad(&grad_out, x.shape(), *dim, *keep_dim)]
            }
            OpType::Mean { dim, keep_dim } => {
                let x = &node.saved_tensors[0];
                let expanded = expand_reduce_grad(&grad_out, x.shape(), *dim, *keep_dim);
                #[allow(clippy::cast_precision_loss)]
                let count = x.shape().dims()[*dim] as f32;
                let scale = Tensor::from_f32(&[1.0 / count], [1]);
                vec![expanded.mul(&scale)]
            }
        };

        for (i, input_opt) in node.inputs.iter().enumerate() {
            if let Some(input_id) = input_opt {
                let g = &grads_in[i];
                if let Some(existing) = grad_map.get_mut(input_id) {
                    *existing = existing.add(g);
                } else {
                    grad_map.insert(*input_id, g.clone());
                }
            }
        }
    }

    let current_gen = TAPE.with(|t| t.borrow().generation);

    // 5. Save the final calculated gradients to the thread-local storage map
    GRADIENTS.with(|g| {
        let mut g = g.borrow_mut();
        g.clear();
        for node in &nodes {
            for (i, input) in node.saved_tensors.iter().enumerate() {
                let is_leaf = input.node_id().is_some_and(|tok| {
                    let (token_gen, idx) = Tape::unpack(tok);
                    token_gen == current_gen && nodes[idx as usize].inputs.is_empty()
                });

                let grad_opt = if is_leaf {
                    node.inputs
                        .get(i)
                        .and_then(Option::as_ref)
                        .and_then(|id| grad_map.get(id))
                } else {
                    None
                };

                if let Some(grad_tensor) = grad_opt {
                    g.insert(input.storage_id(), grad_tensor.clone());
                }
            }
        }
    });

    // 6. Reset the tape to free all references and memory
    TAPE.with(|t| t.borrow_mut().reset());

    // Re-enable tape recording
    set_autograd_enabled(true);
}

/// Dynamic gradient lookup callback.
fn grad_callback(tensor: &Tensor) -> Option<Tensor> {
    GRADIENTS
        .try_with(|g| {
            g.try_borrow()
                .map(|g_ref| g_ref.get(&tensor.storage_id()).cloned())
                .ok()
                .flatten()
        })
        .ok()
        .flatten()
}

/// Dynamic drop cleanup callback.
fn drop_callback(storage_id: StorageId) {
    let _ = GRADIENTS.try_with(|g| {
        if let Ok(mut g_ref) = g.try_borrow_mut() {
            g_ref.remove(&storage_id);
        }
    });
}

/// Initializes the autograd hook overrides.
pub fn init() {
    register_record_op(record_op_callback);
    register_backward_hook(backward_callback);
    register_grad_hook(grad_callback);
    vearo_core::register_drop_hook(drop_callback);
}

/// Resets the active autograd tape for the current thread.
pub fn reset_active_tape() {
    TAPE.with(|t| t.borrow_mut().reset());
}

/// Clear all active gradients in the thread-local gradients map.
pub fn zero_gradients() {
    let _ = GRADIENTS.try_with(|g| {
        if let Ok(mut g_ref) = g.try_borrow_mut() {
            g_ref.clear();
        }
    });
}

/// Computes the numerical gradient of a scalar-valued function `f` with respect to input `x`.
///
/// # Panics
/// Panics if the function `f` does not return a scalar-valued tensor (numel == 1).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn numerical_grad<F>(mut f: F, x: &Tensor, eps: f32) -> Tensor
where
    F: FnMut(&Tensor) -> Tensor,
{
    let x_contiguous = x.contiguous();
    let numel = x_contiguous.shape().numel();
    let mut grad_data = vec![0.0; numel];

    // Disable tape recording during numerical forward evaluations
    let was_enabled = is_autograd_enabled();
    set_autograd_enabled(false);

    for i in 0..numel {
        let orig = x_contiguous.get_f32(i);

        // Perturb positive
        x_contiguous.set_f32(i, orig + eps);
        let y_plus = f(&x_contiguous);
        assert_eq!(
            y_plus.shape().numel(),
            1,
            "grad_check requires scalar-valued function (output numel == 1)"
        );
        let val_plus = y_plus.contiguous().get_f32(0);

        // Perturb negative
        x_contiguous.set_f32(i, orig - eps);
        let y_minus = f(&x_contiguous);
        let val_minus = y_minus.contiguous().get_f32(0);

        // Restore
        x_contiguous.set_f32(i, orig);

        // Compute central difference
        grad_data[i] = (val_plus - val_minus) / (2.0 * eps);
    }

    set_autograd_enabled(was_enabled);

    Tensor::from_f32(&grad_data, *x_contiguous.shape())
}

#[cfg(test)]
mod tests {
    // Exact float-equality is intentional here: these ops produce exactly
    // representable integer-valued results. `*_ana` / `*_num` naming is deliberate.
    #![allow(clippy::float_cmp, clippy::similar_names)]

    use super::*;

    #[test]
    fn test_numerical_gradient_square_sum() {
        vearo_backend_cpu::init();

        // f(x) = sum(x^2)
        // input = [1.0, 2.0, 3.0]
        // expected analytical grad = [2.0, 4.0, 6.0]
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], [3]);

        let grad = numerical_grad(
            |t| {
                let t_sq = t.mul(t);
                let ones = Tensor::from_f32(&[1.0, 1.0, 1.0], [3, 1]);
                t_sq.reshape([1, 3]).matmul(&ones).reshape([1])
            },
            &x,
            1e-3,
        );

        // Assert numerical gradient matches analytical gradient [2.0, 4.0, 6.0]
        assert_eq!(grad.shape().dims(), &[3]);
        assert!((grad.get_f32(0) - 2.0).abs() <= 1e-3);
        assert!((grad.get_f32(1) - 4.0).abs() <= 1e-3);
        assert!((grad.get_f32(2) - 6.0).abs() <= 1e-3);
    }

    #[test]
    fn test_backward_simple_add_mul() {
        vearo_backend_cpu::init();
        init();

        // z = (x + y) * w
        // x = 2.0, y = 3.0, w = 4.0
        // z = 5.0 * 4.0 = 20.0
        // dz/dx = 4.0, dz/dy = 4.0, dz/dw = 5.0
        let x = Tensor::from_f32(&[2.0], [1]);
        let y = Tensor::from_f32(&[3.0], [1]);
        let w = Tensor::from_f32(&[4.0], [1]);

        x.set_requires_grad(true);
        y.set_requires_grad(true);
        w.set_requires_grad(true);

        let sum = x.add(&y);
        let z = sum.mul(&w);

        z.backward();

        let gx = x.grad().unwrap();
        let gy = y.grad().unwrap();
        let gw = w.grad().unwrap();

        assert_eq!(gx.get_f32(0), 4.0);
        assert_eq!(gy.get_f32(0), 4.0);
        assert_eq!(gw.get_f32(0), 5.0);
    }

    #[test]
    fn test_autograd_parity_with_numerical_grad() {
        vearo_backend_cpu::init();
        init();

        // Let's test Add & Mul with broadcasting!
        // f(x, y) = sum((x + y) * x)
        // x shape: [2, 3], y shape: [1, 3] (broadcasted to [2, 3])
        // Let's check with respect to x.
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_data = [10.0, 20.0, 30.0];

        let x = Tensor::from_f32(&x_data, [2, 3]);
        let y = Tensor::from_f32(&y_data, [1, 3]);

        x.set_requires_grad(true);
        y.set_requires_grad(true);

        // Forward function
        let forward = |t_x: &Tensor| {
            let sum = t_x.add(&y);
            let prod = sum.mul(t_x);
            // sum(prod) using matmul
            let ones = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [6, 1]);
            prod.reshape([1, 6]).matmul(&ones).reshape([1])
        };

        // 1. Analytical grad via backward
        let out = forward(&x);
        out.backward();
        let grad_x_ana = x.grad().unwrap();
        let grad_y_ana = y.grad().unwrap();

        // Reset tape for numerical check
        TAPE.with(|t| t.borrow_mut().reset());

        // 2. Numerical grad
        let grad_x_num = numerical_grad(forward, &x, 1e-3);

        // Assert X gradients match
        for i in 0..6 {
            let diff = (grad_x_ana.get_f32(i) - grad_x_num.get_f32(i)).abs();
            assert!(
                diff <= 5e-2,
                "Mismatch on X at index {}: ana={}, num={}",
                i,
                grad_x_ana.get_f32(i),
                grad_x_num.get_f32(i)
            );
        }

        // Test Y gradients (numerical check for Y)
        TAPE.with(|t| t.borrow_mut().reset());
        let forward_y = |t_y: &Tensor| {
            let sum = x.add(t_y);
            let prod = sum.mul(&x);
            let ones = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [6, 1]);
            prod.reshape([1, 6]).matmul(&ones).reshape([1])
        };
        let grad_y_num = numerical_grad(forward_y, &y, 1e-3);

        // Assert Y gradients match
        for i in 0..3 {
            let diff = (grad_y_ana.get_f32(i) - grad_y_num.get_f32(i)).abs();
            assert!(
                diff <= 5e-2,
                "Mismatch on Y at index {}: ana={}, num={}",
                i,
                grad_y_ana.get_f32(i),
                grad_y_num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_two_backward_passes_reuse_leaves() {
        vearo_backend_cpu::init();
        init();

        let x = Tensor::from_f32(&[2.0], [1]);
        let y = Tensor::from_f32(&[3.0], [1]);
        x.set_requires_grad(true);
        y.set_requires_grad(true);

        // Pass 1: z = x * y -> dz/dx = y = 3, dz/dy = x = 2
        let z1 = x.mul(&y);
        z1.backward();
        assert_eq!(x.grad().unwrap().get_f32(0), 3.0, "pass 1 dx");
        assert_eq!(y.grad().unwrap().get_f32(0), 2.0, "pass 1 dy");

        // Pass 2: identical computation must give identical grads.
        let z2 = x.mul(&y);
        z2.backward();
        assert_eq!(x.grad().unwrap().get_f32(0), 3.0, "pass 2 dx");
        assert_eq!(y.grad().unwrap().get_f32(0), 2.0, "pass 2 dy");
    }

    #[test]
    fn test_autograd_relu() {
        vearo_backend_cpu::init();
        init();
        // Inputs kept away from 0 so the ReLU kink doesn't break finite differences.
        let x = Tensor::from_f32(&[1.0, -2.0, 3.0, -0.5, 4.0, -1.5], [2, 3]);
        x.set_requires_grad(true);
        let forward = |t: &Tensor| t.relu().sum(1, false).sum(0, false);

        let out = forward(&x);
        out.backward();
        let ana = x.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let num = numerical_grad(forward, &x, 1e-3);

        for i in 0..6 {
            assert!(
                (ana.get_f32(i) - num.get_f32(i)).abs() <= 5e-2,
                "relu grad mismatch at {}: ana={}, num={}",
                i,
                ana.get_f32(i),
                num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_autograd_sum() {
        vearo_backend_cpu::init();
        init();
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        x.set_requires_grad(true);
        let forward = |t: &Tensor| t.sum(1, true).sum(0, false);

        let out = forward(&x);
        out.backward();
        let ana = x.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let num = numerical_grad(forward, &x, 1e-3);

        for i in 0..6 {
            // d(sum)/dx = 1 everywhere.
            assert!((ana.get_f32(i) - 1.0).abs() <= 5e-2, "sum grad should be 1");
            assert!(
                (ana.get_f32(i) - num.get_f32(i)).abs() <= 5e-2,
                "sum grad mismatch at {}: ana={}, num={}",
                i,
                ana.get_f32(i),
                num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_autograd_mean() {
        vearo_backend_cpu::init();
        init();
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        x.set_requires_grad(true);
        let forward = |t: &Tensor| t.mean(1, false).sum(0, false);

        let out = forward(&x);
        out.backward();
        let ana = x.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let num = numerical_grad(forward, &x, 1e-3);

        for i in 0..6 {
            // d(mean over 3)/dx = 1/3 everywhere.
            assert!(
                (ana.get_f32(i) - (1.0 / 3.0)).abs() <= 5e-2,
                "mean grad should be 1/3"
            );
            assert!(
                (ana.get_f32(i) - num.get_f32(i)).abs() <= 5e-2,
                "mean grad mismatch at {}: ana={}, num={}",
                i,
                ana.get_f32(i),
                num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_autograd_matmul() {
        vearo_backend_cpu::init();
        init();

        // f(X, Y) = sum(X * Y)
        // X: [2, 3], Y: [3, 2]
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_data = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let x = Tensor::from_f32(&x_data, [2, 3]);
        let y = Tensor::from_f32(&y_data, [3, 2]);

        x.set_requires_grad(true);
        y.set_requires_grad(true);

        let forward = |t_x: &Tensor| {
            let out = t_x.matmul(&y);
            // sum of out
            let ones = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], [4, 1]);
            out.reshape([1, 4]).matmul(&ones).reshape([1])
        };

        let out = forward(&x);
        out.backward();

        let grad_x_ana = x.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let grad_x_num = numerical_grad(forward, &x, 1e-3);

        for i in 0..6 {
            let diff = (grad_x_ana.get_f32(i) - grad_x_num.get_f32(i)).abs();
            assert!(
                diff <= 5e-2,
                "Mismatch on matmul X at index {}: ana={}, num={}",
                i,
                grad_x_ana.get_f32(i),
                grad_x_num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_autograd_gelu() {
        vearo_backend_cpu::init();
        init();

        let x = Tensor::from_f32(&[-1.0, 0.0, 1.0, 2.0], [4]);
        x.set_requires_grad(true);

        let forward = |t_x: &Tensor| {
            let out = t_x.gelu();
            out.sum(0, false)
        };

        let out = forward(&x);
        out.backward();

        let grad_x_ana = x.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let grad_x_num = numerical_grad(forward, &x, 1e-4);

        for i in 0..4 {
            let diff = (grad_x_ana.get_f32(i) - grad_x_num.get_f32(i)).abs();
            assert!(
                diff <= 1e-2,
                "Mismatch on gelu at index {}: ana={}, num={}",
                i,
                grad_x_ana.get_f32(i),
                grad_x_num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_autograd_softmax() {
        vearo_backend_cpu::init();
        init();

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        x.set_requires_grad(true);

        let target = Tensor::from_f32(&[0.5, 0.5, 0.2, 0.8], [2, 2]);

        let forward = |t_x: &Tensor| {
            let sm = t_x.softmax(1);
            let out = sm.mul(&target);
            out.sum(0, false).sum(0, false)
        };

        let out = forward(&x);
        out.backward();

        let grad_x_ana = x.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let grad_x_num = numerical_grad(forward, &x, 1e-4);

        for i in 0..4 {
            let diff = (grad_x_ana.get_f32(i) - grad_x_num.get_f32(i)).abs();
            assert!(
                diff <= 1e-2,
                "Mismatch on softmax at index {}: ana={}, num={}",
                i,
                grad_x_ana.get_f32(i),
                grad_x_num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_autograd_layernorm() {
        vearo_backend_cpu::init();
        init();

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let weight = Tensor::from_f32(&[0.5, 1.5], [2]);
        let bias = Tensor::from_f32(&[0.1, -0.1], [2]);

        x.set_requires_grad(true);
        weight.set_requires_grad(true);
        bias.set_requires_grad(true);

        let forward = |t_x: &Tensor| {
            let out = t_x.layernorm(&weight, &bias, 1e-5);
            out.sum(0, false).sum(0, false)
        };

        let out = forward(&x);
        out.backward();

        let grad_x_ana = x.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let grad_x_num = numerical_grad(forward, &x, 1e-4);

        for i in 0..4 {
            let diff = (grad_x_ana.get_f32(i) - grad_x_num.get_f32(i)).abs();
            assert!(
                diff <= 1e-2,
                "Mismatch on layernorm X at {}: ana={}, num={}",
                i,
                grad_x_ana.get_f32(i),
                grad_x_num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_autograd_embedding() {
        vearo_backend_cpu::init();
        init();

        let x = Tensor::from_f32(&[0.0, 1.0], [2]);
        let weight = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        weight.set_requires_grad(true);

        let forward_weight = |t_w: &Tensor| {
            let out = x.embedding(t_w);
            out.sum(0, false).sum(0, false)
        };

        let out = forward_weight(&weight);
        out.backward();

        let grad_w_ana = weight.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let grad_w_num = numerical_grad(forward_weight, &weight, 1e-3);

        for i in 0..6 {
            let diff = (grad_w_ana.get_f32(i) - grad_w_num.get_f32(i)).abs();
            assert!(
                diff <= 1e-2,
                "Mismatch on embedding Weight at {}: ana={}, num={}",
                i,
                grad_w_ana.get_f32(i),
                grad_w_num.get_f32(i)
            );
        }
    }

    #[test]
    fn test_autograd_cross_entropy() {
        vearo_backend_cpu::init();
        init();

        let logits = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let targets = Tensor::from_f32(&[1.0, 0.0], [2]);

        logits.set_requires_grad(true);

        let forward = |t_l: &Tensor| t_l.cross_entropy(&targets);

        let out = forward(&logits);
        out.backward();

        let grad_logits_ana = logits.grad().unwrap();

        TAPE.with(|t| t.borrow_mut().reset());
        let grad_logits_num = numerical_grad(forward, &logits, 1e-3);

        for i in 0..4 {
            let diff = (grad_logits_ana.get_f32(i) - grad_logits_num.get_f32(i)).abs();
            assert!(
                diff <= 1e-2,
                "Mismatch on cross_entropy Logits at {}: ana={}, num={}",
                i,
                grad_logits_ana.get_f32(i),
                grad_logits_num.get_f32(i)
            );
        }
    }

    #[test]
    #[allow(clippy::collection_is_never_read)] // held only to keep slots alive
    fn test_stale_gradient_prevention() {
        vearo_backend_cpu::init();
        init();

        let x = Tensor::from_f32(&[2.0], [1]);
        x.set_requires_grad(true);
        let y = x.mul(&x); // y = x^2
        y.backward();

        let x_grad = x.grad().unwrap();
        assert_eq!(x_grad.get_f32(0), 4.0);

        let old_id = x.storage_id();

        // Now, we drop x and the gradient and y to free all references to the slot
        drop(x_grad);
        drop(y);
        drop(x);

        // We allocate new tensors in a loop, keeping them alive, until we reuse the old_id slot.
        let mut held_tensors = Vec::new();
        let mut reused_tensor = None;
        for _ in 0..100 {
            let t = Tensor::from_f32(&[5.0], [1]);
            if t.storage_id() == old_id {
                reused_tensor = Some(t);
                break;
            }
            held_tensors.push(t);
        }

        let t = reused_tensor.expect("Should have reused the old storage slot");
        assert!(
            t.grad().is_none(),
            "Stale gradient bug detected! Reused tensor got old gradient."
        );
    }
}
