//! CPU reference backend implementations.
// FFI into the hand-written assembly microkernels in src/asm.
#![allow(unsafe_code)]

#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::uninlined_format_args,
    clippy::similar_names,
    clippy::excessive_precision,
    clippy::items_after_statements
)]

use vearo_core::{
    BackendOps, CpuArenaShard, CpuStorage, Device, NdIterator, Shape, Tensor, get_cpu_shard,
    get_offset, register_backend_ops,
};

/// Registers the CPU backend's op implementations with `vearo-core`.
///
/// Idempotent - safe to call from every test and at application startup.
pub fn init() {
    register_backend_ops(
        Device::Cpu,
        BackendOps {
            add,
            sub,
            mul,
            div,
            matmul,
            relu,
            sum,
            mean,
            gelu,
            softmax,
            layernorm,
            layernorm_backward,
            embedding,
            embedding_backward,
            cross_entropy,
            cross_entropy_backward,
            conv2d,
            conv2d_backward,
            maxpool2d,
            maxpool2d_backward,
            avgpool2d,
            avgpool2d_backward,
            batchnorm,
            batchnorm_backward,
            fused_attention,
            fused_attention_backward,
        },
    );
}

/// Helper to lock up to 3 shards in a sorted deadlock-free order on the stack.
pub struct LockedShards {
    guards: [Option<std::sync::MutexGuard<'static, CpuArenaShard>>; 3],
    indices: [u8; 3],
    count: usize,
}

impl LockedShards {
    /// Lock up to 3 shards in a sorted deadlock-free order.
    pub fn lock(s0: u8, s1: u8, s2: u8) -> Self {
        let mut sorted = [s0, s1, s2];
        sorted.sort_unstable();

        let mut count = 1;
        if sorted[1] != sorted[0] {
            sorted[count] = sorted[1];
            count += 1;
        }
        if sorted[2] != sorted[count - 1] {
            sorted[count] = sorted[2];
            count += 1;
        }

        let mut guards = [None, None, None];
        for i in 0..count {
            let shard_idx = sorted[i] as usize;
            guards[i] = Some(
                get_cpu_shard(shard_idx)
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
        }

        let mut indices = [0; 3];
        indices[..count].copy_from_slice(&sorted[..count]);

        Self {
            guards,
            indices,
            count,
        }
    }

    /// Access a shard immutably.
    ///
    /// # Panics
    /// Panics if the requested shard is not locked by this lock manager.
    #[must_use]
    pub fn get(&self, shard_idx: u8) -> &CpuArenaShard {
        for i in 0..self.count {
            if self.indices[i] == shard_idx {
                return self.guards[i].as_ref().unwrap();
            }
        }
        unreachable!()
    }

    /// Access a shard mutably.
    ///
    /// # Panics
    /// Panics if the requested shard is not locked by this lock manager.
    pub fn get_mut(&mut self, shard_idx: u8) -> &mut CpuArenaShard {
        for i in 0..self.count {
            if self.indices[i] == shard_idx {
                return self.guards[i].as_mut().unwrap();
            }
        }
        unreachable!()
    }
}

/// Obtain the F32 backing buffer of a tensor under its shard lock by cloning the Arc.
///
/// Cloning the Arc lets the caller compute with **no locks held** and performs zero
/// copying or allocations of the underlying buffer.
fn read_f32(t: &Tensor) -> std::sync::Arc<Vec<f32>> {
    let guard = get_cpu_shard(t.storage_id().shard_idx as usize)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = match &guard.slots[t.storage_id().slot_idx as usize]
        .as_ref()
        .expect("Source slot was empty")
        .storage
    {
        CpuStorage::F32(v) => v.clone(),
        _ => panic!("Only F32 supported"),
    };
    drop(guard);
    out
}

/// Publish an F32 buffer into a (contiguous) tensor's slot under its shard lock.
fn write_f32(t: &Tensor, data: Vec<f32>) {
    let mut guard = get_cpu_shard(t.storage_id().shard_idx as usize)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let slot = guard.slots[t.storage_id().slot_idx as usize]
        .as_mut()
        .expect("Output slot was empty");
    slot.storage = CpuStorage::F32(std::sync::Arc::new(data));
    drop(guard);
}

fn elementwise_op(lhs: &Tensor, rhs: &Tensor, op: impl Fn(f32, f32) -> f32) -> Tensor {
    assert_eq!(lhs.dtype(), vearo_core::DType::F32, "Only F32 supported");
    assert_eq!(rhs.dtype(), vearo_core::DType::F32, "Only F32 supported");

    let out_shape = lhs
        .shape()
        .broadcast(rhs.shape())
        .expect("Shapes are not broadcastable");
    let out_tensor = Tensor::zeros(out_shape, vearo_core::DType::F32);

    if out_shape.numel() == 0 {
        return out_tensor;
    }

    // Copy operands out (each under its own brief lock), then compute unlocked.
    let lhs_data = read_f32(lhs);
    let rhs_data = read_f32(rhs);

    let mut out_data = vec![0.0f32; out_shape.numel()];
    if lhs.is_contiguous() && rhs.is_contiguous() && lhs.shape() == rhs.shape() {
        for i in 0..out_shape.numel() {
            out_data[i] = op(lhs_data[i], rhs_data[i]);
        }
    } else {
        let mut iter = NdIterator::new(out_shape);
        let mut i = 0;
        loop {
            let coord = iter.coord();
            let l_offset = get_offset(coord, lhs.shape(), lhs.strides());
            let r_offset = get_offset(coord, rhs.shape(), rhs.strides());
            out_data[i] = op(lhs_data[l_offset], rhs_data[r_offset]);
            i += 1;
            if !iter.step() {
                break;
            }
        }
    }

    // Output is freshly allocated and contiguous, so `out_data` is already in
    // row-major order; a single locked write publishes it.
    write_f32(&out_tensor, out_data);
    out_tensor
}

/// Adds two tensors elementwise on CPU.
///
/// # Panics
/// Panics if shapes are not broadcastable or dtypes are not F32.
#[must_use]
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    elementwise_op(lhs, rhs, |a, b| a + b)
}

/// Subtracts two tensors elementwise on CPU.
///
/// # Panics
/// Panics if shapes are not broadcastable or dtypes are not F32.
#[must_use]
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    elementwise_op(lhs, rhs, |a, b| a - b)
}

/// Multiplies two tensors elementwise on CPU.
///
/// # Panics
/// Panics if shapes are not broadcastable or dtypes are not F32.
#[must_use]
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    elementwise_op(lhs, rhs, |a, b| a * b)
}

/// Divides two tensors elementwise on CPU.
///
/// # Panics
/// Panics if shapes are not broadcastable or dtypes are not F32.
#[must_use]
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    elementwise_op(lhs, rhs, |a, b| a / b)
}

#[cfg(vearo_cpu_asm)]
unsafe extern "C" {
    fn matmul_block_4x8_avx2_asm(
        out_ptr: *mut f32,
        lhs_ptr: *const f32,
        rhs_ptr: *const f32,
        k_l: usize,
        n: usize,
    );
}

/// Matrix multiplication of two CPU tensors.
///
/// # Panics
/// Panics if ranks are less than 2, trailing dimensions do not match, or dtypes are not F32.
#[must_use]
// Scoped to this function while the assembly path is in progress. The shared
// prefix between the asm and portable branches wants hoisting once the two
// settle, and the indexed loop wants an iterator; neither is a correctness
// issue. Do not widen these to the crate.
#[allow(
    clippy::too_many_lines,
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::needless_range_loop
)]
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    assert_eq!(lhs.dtype(), vearo_core::DType::F32, "Only F32 supported");
    assert_eq!(rhs.dtype(), vearo_core::DType::F32, "Only F32 supported");

    let rank_l = lhs.shape().rank();
    let rank_r = rhs.shape().rank();
    assert!(rank_l >= 2 && rank_r >= 2, "Matmul requires rank >= 2");

    let m = lhs.shape()[rank_l - 2];
    let k_l = lhs.shape()[rank_l - 1];
    let k_r = rhs.shape()[rank_r - 2];
    let n = rhs.shape()[rank_r - 1];
    assert_eq!(k_l, k_r, "Incompatible dimensions for matmul");

    let batch_shape_l = Shape::new(&lhs.shape().dims()[..rank_l - 2]);
    let batch_shape_r = Shape::new(&rhs.shape().dims()[..rank_r - 2]);

    let out_batch_shape = batch_shape_l
        .broadcast(&batch_shape_r)
        .expect("Batch shapes are not broadcastable");
    let mut out_dims = out_batch_shape.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::new(out_dims);

    let out_tensor = Tensor::zeros(out_shape, vearo_core::DType::F32);

    if out_shape.numel() == 0 {
        return out_tensor;
    }

    let lhs_contig = lhs.contiguous();
    let rhs_contig = rhs.contiguous();
    let lhs_data = read_f32(&lhs_contig);
    let rhs_data = read_f32(&rhs_contig);
    let mut out_data = vec![0.0f32; out_shape.numel()];

    let l_stride_m = lhs_contig.strides()[rank_l - 2];
    let l_stride_k = lhs_contig.strides()[rank_l - 1];
    let r_stride_k = rhs_contig.strides()[rank_r - 2];
    let r_stride_n = rhs_contig.strides()[rank_r - 1];
    let batch_strides_l = Shape::new(&lhs_contig.strides().dims()[..rank_l - 2]);
    let batch_strides_r = Shape::new(&rhs_contig.strides().dims()[..rank_r - 2]);

    let total_rows = out_batch_shape.numel() * m;
    let compute = |out_data_slice: &mut [f32], start_row: usize| {
        let end_row = start_row + out_data_slice.len() / n;
        let mut row_idx = start_row;
        while row_idx < end_row {
            let r_m = row_idx % m;
            let batch_idx = row_idx / m;

            let mut run_asm = false;
            #[cfg(vearo_cpu_asm)]
            {
                // A full 4-row block must be available, and the CPU must
                // actually have the instructions the kernel is written in.
                if r_m + 4 <= m
                    && row_idx + 4 <= end_row
                    && is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("fma")
                {
                    run_asm = true;
                }
            }

            if run_asm {
                let mut batch_coord = [0usize; 8];
                let mut remaining = batch_idx;
                for i in (0..out_batch_shape.rank()).rev() {
                    batch_coord[i] = remaining % out_batch_shape[i];
                    remaining /= out_batch_shape[i];
                }

                let batch_coord_slice = &batch_coord[..out_batch_shape.rank()];
                let batch_offset_l = get_offset(batch_coord_slice, &batch_shape_l, &batch_strides_l);
                let batch_offset_r = get_offset(batch_coord_slice, &batch_shape_r, &batch_strides_r);

                let n_blocks = n / 8;
                for b_idx in 0..n_blocks {
                    let c_n = b_idx * 8;
                    let out_ptr = unsafe { out_data_slice.as_mut_ptr().add((row_idx - start_row) * n + c_n) };
                    let lhs_ptr = unsafe { lhs_data.as_ptr().add(batch_offset_l + r_m * k_l) };
                    let rhs_ptr = unsafe { rhs_data.as_ptr().add(batch_offset_r + c_n) };

                    #[cfg(vearo_cpu_asm)]
                    unsafe {
                        matmul_block_4x8_avx2_asm(out_ptr, lhs_ptr, rhs_ptr, k_l, n);
                    }
                }

                for c_n in (n_blocks * 8)..n {
                    for r_offset in 0..4 {
                        let mut sum = 0.0f32;
                        let curr_r_m = r_m + r_offset;
                        for r_k in 0..k_l {
                            let l_idx = batch_offset_l + curr_r_m * l_stride_m + r_k * l_stride_k;
                            let r_idx = batch_offset_r + r_k * r_stride_k + c_n * r_stride_n;
                            sum = lhs_data[l_idx].mul_add(rhs_data[r_idx], sum);
                        }
                        out_data_slice[(row_idx + r_offset - start_row) * n + c_n] = sum;
                    }
                }

                row_idx += 4;
            } else {
                let mut batch_coord = [0usize; 8];
                let mut remaining = batch_idx;
                for i in (0..out_batch_shape.rank()).rev() {
                    batch_coord[i] = remaining % out_batch_shape[i];
                    remaining /= out_batch_shape[i];
                }

                let batch_coord_slice = &batch_coord[..out_batch_shape.rank()];
                let batch_offset_l = get_offset(batch_coord_slice, &batch_shape_l, &batch_strides_l);
                let batch_offset_r = get_offset(batch_coord_slice, &batch_shape_r, &batch_strides_r);

                let row_slice = &mut out_data_slice[(row_idx - start_row) * n..(row_idx - start_row + 1) * n];
                for r_k in 0..k_l {
                    let l_idx = batch_offset_l + r_m * l_stride_m + r_k * l_stride_k;
                    let lhs_val = lhs_data[l_idx];

                    let r_k_offset = batch_offset_r + r_k * r_stride_k;
                    for r_n in 0..n {
                        let r_idx = r_k_offset + r_n * r_stride_n;
                        row_slice[r_n] = lhs_val.mul_add(rhs_data[r_idx], row_slice[r_n]);
                    }
                }

                row_idx += 1;
            }
        }
    };

    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let rows_per_thread = total_rows.div_ceil(threads.max(1));
    let chunk_size = rows_per_thread * n;

    if threads <= 1 || total_rows < 2 {
        compute(&mut out_data, 0);
    } else {
        std::thread::scope(|s| {
            let compute_ref = &compute;
            for (t, out_chunk) in out_data.chunks_mut(chunk_size).enumerate() {
                s.spawn(move || compute_ref(out_chunk, t * rows_per_thread));
            }
        });
    }

    write_f32(&out_tensor, out_data);
    out_tensor
}

/// Elementwise `ReLU` on CPU: `max(0, x)`.
///
/// # Panics
/// Panics if the dtype is not F32.
#[must_use]
pub fn relu(x: &Tensor) -> Tensor {
    assert_eq!(x.dtype(), vearo_core::DType::F32, "Only F32 supported");
    let out = Tensor::zeros(*x.shape(), vearo_core::DType::F32);
    if x.shape().numel() == 0 {
        return out;
    }

    let x_data = read_f32(x);
    let mut out_data = vec![0.0f32; x.shape().numel()];
    if x.is_contiguous() {
        for i in 0..x.shape().numel() {
            let v = x_data[i];
            out_data[i] = if v > 0.0 { v } else { 0.0 };
        }
    } else {
        let mut iter = NdIterator::new(*x.shape());
        let mut i = 0;
        loop {
            let off = get_offset(iter.coord(), x.shape(), x.strides());
            let v = x_data[off];
            out_data[i] = if v > 0.0 { v } else { 0.0 };
            i += 1;
            if !iter.step() {
                break;
            }
        }
    }

    write_f32(&out, out_data);
    out
}

/// Elementwise `GELU` on CPU using tanh approximation.
#[must_use]
pub fn gelu(x: &Tensor) -> Tensor {
    assert_eq!(x.dtype(), vearo_core::DType::F32, "Only F32 supported");
    let out = Tensor::zeros(*x.shape(), vearo_core::DType::F32);
    if x.shape().numel() == 0 {
        return out;
    }

    let x_data = read_f32(x);
    let mut out_data = vec![0.0f32; x.shape().numel()];

    // Tanh approximation constants
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const COEFF: f32 = 0.044_715;

    if x.is_contiguous() {
        for i in 0..x.shape().numel() {
            let v = x_data[i];
            let v3 = v * v * v;
            let inner = SQRT_2_OVER_PI * (v + COEFF * v3);
            out_data[i] = 0.5 * v * (1.0 + inner.tanh());
        }
    } else {
        let mut iter = NdIterator::new(*x.shape());
        let mut i = 0;
        loop {
            let off = get_offset(iter.coord(), x.shape(), x.strides());
            let v = x_data[off];
            let v3 = v * v * v;
            let inner = SQRT_2_OVER_PI * (v + COEFF * v3);
            out_data[i] = 0.5 * v * (1.0 + inner.tanh());
            i += 1;
            if !iter.step() {
                break;
            }
        }
    }

    write_f32(&out, out_data);
    out
}

/// Softmax along a single axis `dim` on CPU.
#[must_use]
pub fn softmax(x: &Tensor, dim: usize) -> Tensor {
    assert_eq!(x.dtype(), vearo_core::DType::F32, "Only F32 supported");
    let rank = x.shape().rank();
    assert!(dim < rank, "Softmax dim out of range");

    let out = Tensor::zeros(*x.shape(), vearo_core::DType::F32);
    if x.shape().numel() == 0 {
        return out;
    }

    let x_data = read_f32(x);
    let mut out_data = vec![0.0f32; x.shape().numel()];

    let dims = x.shape().dims();
    let dim_size = dims[dim];

    let mut reduced_dims = dims.to_vec();
    reduced_dims[dim] = 1;
    let reduced_shape = Shape::new(&reduced_dims);

    let mut iter = NdIterator::new(reduced_shape);
    loop {
        let mut coord = iter.coord().to_vec();

        // 1. Find max value along `dim` for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for d_idx in 0..dim_size {
            coord[dim] = d_idx;
            let off = get_offset(&coord, x.shape(), x.strides());
            let val = x_data[off];
            if val > max_val {
                max_val = val;
            }
        }

        // 2. Compute sum of exponentials
        let mut sum_exp = 0.0f32;
        for d_idx in 0..dim_size {
            coord[dim] = d_idx;
            let off = get_offset(&coord, x.shape(), x.strides());
            let val = x_data[off];
            sum_exp += (val - max_val).exp();
        }

        // 3. Write softmax values into out_data
        for d_idx in 0..dim_size {
            coord[dim] = d_idx;
            let off = get_offset(&coord, x.shape(), x.strides());
            let val = x_data[off];
            let sm_val = (val - max_val).exp() / sum_exp;

            let out_off = get_offset(&coord, x.shape(), &x.shape().contiguous_strides());
            out_data[out_off] = sm_val;
        }

        if !iter.step() {
            break;
        }
    }

    write_f32(&out, out_data);
    out
}

/// Row-major flat index into a single-axis reduction's output for an input `coord`.
fn reduced_out_index(coord: &[usize], in_dims: &[usize], dim: usize, keep_dim: bool) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for d in (0..in_dims.len()).rev() {
        if keep_dim {
            let c = if d == dim { 0 } else { coord[d] };
            let sz = if d == dim { 1 } else { in_dims[d] };
            idx += c * stride;
            stride *= sz;
        } else if d != dim {
            idx += coord[d] * stride;
            stride *= in_dims[d];
        }
    }
    idx
}

/// Sum an input over a single axis into a fresh output buffer.
///
/// Accumulation is deterministic: contributions to each output element arrive in
/// strictly increasing `dim`-index order (guaranteed by row-major iteration).
fn reduce_sum_data(x: &Tensor, dim: usize, keep_dim: bool) -> (Shape, Vec<f32>) {
    assert_eq!(x.dtype(), vearo_core::DType::F32, "Only F32 supported");
    let in_shape = *x.shape();
    let rank = in_shape.rank();
    assert!(dim < rank, "Reduction dim out of range");

    let mut out_dims: Vec<usize> = in_shape.dims().to_vec();
    if keep_dim {
        out_dims[dim] = 1;
    } else {
        out_dims.remove(dim);
    }
    let out_shape = Shape::new(&out_dims);

    let mut out_data = vec![0.0f32; out_shape.numel()];
    if in_shape.numel() == 0 {
        return (out_shape, out_data);
    }

    let x_data = read_f32(x);
    let in_dims = in_shape.dims();
    let mut iter = NdIterator::new(in_shape);
    loop {
        let coord = iter.coord();
        let in_off = get_offset(coord, &in_shape, x.strides());
        let out_idx = reduced_out_index(coord, in_dims, dim, keep_dim);
        out_data[out_idx] += x_data[in_off];
        if !iter.step() {
            break;
        }
    }

    (out_shape, out_data)
}

/// Sum over a single axis on CPU.
///
/// # Panics
/// Panics if the dtype is not F32 or `dim` is out of range.
#[must_use]
pub fn sum(x: &Tensor, dim: usize, keep_dim: bool) -> Tensor {
    let (out_shape, out_data) = reduce_sum_data(x, dim, keep_dim);
    let out = Tensor::zeros(out_shape, vearo_core::DType::F32);
    if out_shape.numel() > 0 {
        write_f32(&out, out_data);
    }
    out
}

/// Mean over a single axis on CPU (sum, then a single division by the axis length).
///
/// # Panics
/// Panics if the dtype is not F32 or `dim` is out of range.
#[must_use]
pub fn mean(x: &Tensor, dim: usize, keep_dim: bool) -> Tensor {
    let (out_shape, mut out_data) = reduce_sum_data(x, dim, keep_dim);
    #[allow(clippy::cast_precision_loss)]
    let count = x.shape().dims()[dim] as f32;
    for v in &mut out_data {
        *v /= count;
    }
    let out = Tensor::zeros(out_shape, vearo_core::DType::F32);
    if out_shape.numel() > 0 {
        write_f32(&out, out_data);
    }
    out
}

fn flat_to_coord(mut index: usize, shape: &Shape) -> Vec<usize> {
    let dims = shape.dims();
    let mut coord = vec![0; dims.len()];
    for i in (0..dims.len()).rev() {
        coord[i] = index % dims[i];
        index /= dims[i];
    }
    coord
}

/// Layer normalization forward on CPU.
#[must_use]
pub fn layernorm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.dtype(), vearo_core::DType::F32, "Only F32 supported");
    let out = Tensor::zeros(*x.shape(), vearo_core::DType::F32);
    if x.shape().numel() == 0 {
        return out;
    }

    let dims = x.shape().dims();
    let rank = dims.len();
    assert!(rank >= 1, "LayerNorm input must be at least rank 1");
    let norm_dim = dims[rank - 1];

    assert_eq!(weight.shape().rank(), 1, "LayerNorm weight must be rank 1");
    assert_eq!(bias.shape().rank(), 1, "LayerNorm bias must be rank 1");
    assert_eq!(
        weight.shape()[0],
        norm_dim,
        "Weight dimension must match norm_dim"
    );
    assert_eq!(
        bias.shape()[0],
        norm_dim,
        "Bias dimension must match norm_dim"
    );

    let x_data = read_f32(x);
    let w_data = read_f32(weight);
    let b_data = read_f32(bias);
    let mut out_data = vec![0.0f32; x.shape().numel()];

    let outer_numel = x.shape().numel() / norm_dim;

    for b in 0..outer_numel {
        let base_idx = b * norm_dim;

        // 1. Calculate mean
        let mut sum = 0.0f32;
        for i in 0..norm_dim {
            let coord = flat_to_coord(base_idx + i, x.shape());
            let off = get_offset(&coord, x.shape(), x.strides());
            sum += x_data[off];
        }
        let mean = sum / (norm_dim as f32);

        // 2. Calculate variance
        let mut sum_sq = 0.0f32;
        for i in 0..norm_dim {
            let coord = flat_to_coord(base_idx + i, x.shape());
            let off = get_offset(&coord, x.shape(), x.strides());
            let diff = x_data[off] - mean;
            sum_sq += diff * diff;
        }
        let var = sum_sq / (norm_dim as f32);
        let inv_std = 1.0 / (var + eps).sqrt();

        // 3. Normalize, scale, and shift
        for i in 0..norm_dim {
            let coord = flat_to_coord(base_idx + i, x.shape());
            let off = get_offset(&coord, x.shape(), x.strides());

            let x_hat = (x_data[off] - mean) * inv_std;
            let out_idx = base_idx + i;

            let w_coord = [i];
            let w_off = get_offset(&w_coord, weight.shape(), weight.strides());
            let b_off = get_offset(&w_coord, bias.shape(), bias.strides());

            out_data[out_idx] = x_hat * w_data[w_off] + b_data[b_off];
        }
    }

    write_f32(&out, out_data);
    out
}

/// Layer normalization backward on CPU.
#[must_use]
#[allow(clippy::similar_names)]
pub fn layernorm_backward(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    grad_out: &Tensor,
    eps: f32,
) -> (Tensor, Tensor, Tensor) {
    let grad_x = Tensor::zeros(*x.shape(), vearo_core::DType::F32);
    let grad_w = Tensor::zeros(*weight.shape(), vearo_core::DType::F32);
    let grad_b = Tensor::zeros(*bias.shape(), vearo_core::DType::F32);

    if x.shape().numel() == 0 {
        return (grad_x, grad_w, grad_b);
    }

    let dims = x.shape().dims();
    let rank = dims.len();
    assert!(rank >= 1, "LayerNorm input must be at least rank 1");
    let norm_dim = dims[rank - 1];

    assert_eq!(weight.shape().rank(), 1, "LayerNorm weight must be rank 1");
    assert_eq!(bias.shape().rank(), 1, "LayerNorm bias must be rank 1");
    assert_eq!(
        weight.shape()[0],
        norm_dim,
        "Weight dimension must match norm_dim"
    );
    assert_eq!(
        bias.shape()[0],
        norm_dim,
        "Bias dimension must match norm_dim"
    );

    let x_data = read_f32(x);
    let w_data = read_f32(weight);
    let go_data = read_f32(grad_out);

    let mut gx_data = vec![0.0f32; x.shape().numel()];
    let mut gw_data = vec![0.0f32; weight.shape().numel()];
    let mut gb_data = vec![0.0f32; bias.shape().numel()];

    let outer_numel = x.shape().numel() / norm_dim;

    for b in 0..outer_numel {
        let base_idx = b * norm_dim;

        // 1. Re-calculate mean and variance for x_hat
        let mut sum = 0.0f32;
        for i in 0..norm_dim {
            let coord = flat_to_coord(base_idx + i, x.shape());
            let off = get_offset(&coord, x.shape(), x.strides());
            sum += x_data[off];
        }
        let mean = sum / (norm_dim as f32);

        let mut sum_sq = 0.0f32;
        for i in 0..norm_dim {
            let coord = flat_to_coord(base_idx + i, x.shape());
            let off = get_offset(&coord, x.shape(), x.strides());
            let diff = x_data[off] - mean;
            sum_sq += diff * diff;
        }
        let var = sum_sq / (norm_dim as f32);
        let inv_std = 1.0 / (var + eps).sqrt();

        // 2. Compute intermediates for grad_x
        let mut sum_go_w = 0.0f32;
        let mut sum_go_w_xhat = 0.0f32;

        for i in 0..norm_dim {
            let coord = flat_to_coord(base_idx + i, x.shape());
            let go_off = get_offset(&coord, grad_out.shape(), grad_out.strides());
            let x_off = get_offset(&coord, x.shape(), x.strides());

            let x_hat = (x_data[x_off] - mean) * inv_std;

            let w_coord = [i];
            let w_off = get_offset(&w_coord, weight.shape(), weight.strides());
            let w_val = w_data[w_off];
            let go_val = go_data[go_off];

            sum_go_w += go_val * w_val;
            sum_go_w_xhat += go_val * w_val * x_hat;

            gw_data[i] += go_val * x_hat;
            gb_data[i] += go_val;
        }

        // 3. Compute grad_x
        for i in 0..norm_dim {
            let coord = flat_to_coord(base_idx + i, x.shape());
            let go_off = get_offset(&coord, grad_out.shape(), grad_out.strides());
            let x_off = get_offset(&coord, x.shape(), x.strides());
            let x_hat = (x_data[x_off] - mean) * inv_std;

            let w_coord = [i];
            let w_off = get_offset(&w_coord, weight.shape(), weight.strides());
            let w_val = w_data[w_off];
            let go_val = go_data[go_off];

            let term1 = (norm_dim as f32) * go_val * w_val;
            let term2 = sum_go_w;
            let term3 = x_hat * sum_go_w_xhat;

            gx_data[base_idx + i] = (term1 - term2 - term3) * inv_std / (norm_dim as f32);
        }
    }

    write_f32(&grad_x, gx_data);
    write_f32(&grad_w, gw_data);
    write_f32(&grad_b, gb_data);

    (grad_x, grad_w, grad_b)
}

/// Embedding lookup forward on CPU.
#[must_use]
pub fn embedding(x: &Tensor, weight: &Tensor) -> Tensor {
    assert_eq!(weight.dtype(), vearo_core::DType::F32, "Weight must be F32");
    assert_eq!(
        weight.shape().rank(),
        2,
        "Weight must be rank 2 (vocab_size, embedding_dim)"
    );

    let x_data = read_f32(x);
    let w_data = read_f32(weight);

    let weight_dims = weight.shape().dims();
    let vocab_size = weight_dims[0];
    let embedding_dim = weight_dims[1];

    let mut out_dims = x.shape().dims().to_vec();
    out_dims.push(embedding_dim);
    let out_shape = Shape::new(&out_dims);

    let out = Tensor::zeros(out_shape, vearo_core::DType::F32);
    if x.shape().numel() == 0 {
        return out;
    }

    let mut out_data = vec![0.0f32; out_shape.numel()];

    for i in 0..x.shape().numel() {
        let coord = flat_to_coord(i, x.shape());
        let off = get_offset(&coord, x.shape(), x.strides());

        let token_val = x_data[off];
        assert!(
            token_val >= 0.0,
            "Token ID cannot be negative: {}",
            token_val
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let token_id = token_val.round() as usize;
        assert!(
            token_id < vocab_size,
            "Token ID {} out of vocabulary bound {}",
            token_id,
            vocab_size
        );

        let out_base = i * embedding_dim;

        for d in 0..embedding_dim {
            let w_coord = [token_id, d];
            let w_off = get_offset(&w_coord, weight.shape(), weight.strides());
            out_data[out_base + d] = w_data[w_off];
        }
    }

    write_f32(&out, out_data);
    out
}

/// Embedding lookup backward on CPU.
#[must_use]
#[allow(clippy::similar_names)]
pub fn embedding_backward(x: &Tensor, weight: &Tensor, grad_out: &Tensor) -> Tensor {
    assert_eq!(
        weight.shape().rank(),
        2,
        "Weight must be rank 2 (vocab_size, embedding_dim)"
    );

    let weight_dims = weight.shape().dims();
    let vocab_size = weight_dims[0];
    let embedding_dim = weight_dims[1];

    let mut expected_go_dims = x.shape().dims().to_vec();
    expected_go_dims.push(embedding_dim);
    assert_eq!(
        grad_out.shape().dims(),
        expected_go_dims.as_slice(),
        "grad_out shape must be x.shape() + [embedding_dim]"
    );

    let grad_w = Tensor::zeros(*weight.shape(), vearo_core::DType::F32);
    let x_data = read_f32(x);
    let go_data = read_f32(grad_out);

    let mut gw_data = vec![0.0f32; weight.shape().numel()];

    for i in 0..x.shape().numel() {
        let coord = flat_to_coord(i, x.shape());
        let off = get_offset(&coord, x.shape(), x.strides());

        let token_val = x_data[off];
        assert!(
            token_val >= 0.0,
            "Token ID cannot be negative: {}",
            token_val
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let token_id = token_val.round() as usize;
        assert!(
            token_id < vocab_size,
            "Token ID {} out of vocabulary bound {}",
            token_id,
            vocab_size
        );

        for d in 0..embedding_dim {
            let go_coord = {
                let mut c = coord.clone();
                c.push(d);
                c
            };
            let go_off = get_offset(&go_coord, grad_out.shape(), grad_out.strides());
            gw_data[token_id * embedding_dim + d] += go_data[go_off];
        }
    }

    write_f32(&grad_w, gw_data);
    grad_w
}

/// Categorical cross-entropy loss forward on CPU.
#[must_use]
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    assert_eq!(logits.dtype(), vearo_core::DType::F32, "Logits must be F32");
    assert_eq!(
        logits.shape().rank(),
        2,
        "Logits must be rank 2 (batch_size, vocab_size)"
    );
    assert_eq!(
        targets.shape().rank(),
        1,
        "Targets must be rank 1 (batch_size)"
    );

    let logits_dims = logits.shape().dims();
    let batch_size = logits_dims[0];
    let vocab_size = logits_dims[1];

    assert!(batch_size > 0, "Batch size must be greater than 0");
    assert_eq!(
        targets.shape()[0],
        batch_size,
        "Targets shape must match batch size"
    );

    let t_data = read_f32(targets);
    let l_data = read_f32(logits);

    let mut loss_sum = 0.0f32;

    for b in 0..batch_size {
        let target_val = t_data[get_offset(&[b], targets.shape(), targets.strides())];
        assert!(
            target_val >= 0.0,
            "Target class cannot be negative: {}",
            target_val
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_class = target_val.round() as usize;
        assert!(
            target_class < vocab_size,
            "Target class {} out of vocab bounds {}",
            target_class,
            vocab_size
        );

        let mut max_logit = f32::NEG_INFINITY;
        for c in 0..vocab_size {
            let l_off = get_offset(&[b, c], logits.shape(), logits.strides());
            if l_data[l_off] > max_logit {
                max_logit = l_data[l_off];
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..vocab_size {
            let l_off = get_offset(&[b, c], logits.shape(), logits.strides());
            sum_exp += (l_data[l_off] - max_logit).exp();
        }

        let target_off = get_offset(&[b, target_class], logits.shape(), logits.strides());
        let target_logit = l_data[target_off];
        let log_softmax = target_logit - max_logit - sum_exp.ln();
        loss_sum -= log_softmax;
    }

    let mean_loss = loss_sum / (batch_size as f32);
    Tensor::from_f32(&[mean_loss], [1])
}

/// Categorical cross-entropy loss backward on CPU.
#[must_use]
pub fn cross_entropy_backward(logits: &Tensor, targets: &Tensor, grad_out: &Tensor) -> Tensor {
    assert_eq!(
        logits.shape().rank(),
        2,
        "Logits must be rank 2 (batch_size, vocab_size)"
    );
    assert_eq!(
        targets.shape().rank(),
        1,
        "Targets must be rank 1 (batch_size)"
    );

    let grad_l = Tensor::zeros(*logits.shape(), vearo_core::DType::F32);
    let logits_dims = logits.shape().dims();
    let batch_size = logits_dims[0];
    let vocab_size = logits_dims[1];

    assert!(batch_size > 0, "Batch size must be greater than 0");
    assert_eq!(
        targets.shape()[0],
        batch_size,
        "Targets shape must match batch size"
    );
    assert_eq!(grad_out.shape().numel(), 1, "grad_out must be a scalar");

    let t_data = read_f32(targets);
    let l_data = read_f32(logits);
    let go_val = read_f32(grad_out)[0];

    let mut gl_data = vec![0.0f32; logits.shape().numel()];

    for b in 0..batch_size {
        let target_val = t_data[get_offset(&[b], targets.shape(), targets.strides())];
        assert!(
            target_val >= 0.0,
            "Target class cannot be negative: {}",
            target_val
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_class = target_val.round() as usize;
        assert!(
            target_class < vocab_size,
            "Target class {} out of vocab bounds {}",
            target_class,
            vocab_size
        );

        let mut max_logit = f32::NEG_INFINITY;
        for c in 0..vocab_size {
            let l_off = get_offset(&[b, c], logits.shape(), logits.strides());
            if l_data[l_off] > max_logit {
                max_logit = l_data[l_off];
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..vocab_size {
            let l_off = get_offset(&[b, c], logits.shape(), logits.strides());
            sum_exp += (l_data[l_off] - max_logit).exp();
        }

        for c in 0..vocab_size {
            let l_off = get_offset(&[b, c], logits.shape(), logits.strides());
            let p_c = (l_data[l_off] - max_logit).exp() / sum_exp;
            let target_indicator = if c == target_class { 1.0f32 } else { 0.0f32 };

            let out_idx = b * vocab_size + c;
            gl_data[out_idx] = go_val * (p_c - target_indicator) / (batch_size as f32);
        }
    }
    write_f32(&grad_l, gl_data);
    grad_l
}


/// Convolution lowered to a matrix multiply: `im2col` then GEMM.
///
/// # Relationship to [`conv2d`]
///
/// [`conv2d`] is the reference: scalar, deterministic, and bit-identical to the
/// CUDA kernel, which was written to match its accumulation order. It is the
/// correctness oracle and must stay that way.
///
/// This path trades that for speed. Convolution becomes a single
/// `[cout, cin*kh*kw] x [cin*kh*kw, oh*ow]` product per image, handed to a
/// blocked, threaded, SIMD GEMM. That reassociates the sum, so results agree
/// with the reference to roughly 1e-5 rather than exactly. Anything relying on
/// bit-equality must use [`conv2d`].
///
/// The im2col buffer costs `cin*kh*kw*oh*ow` floats per image, which is `kh*kw`
/// times the input. That is the memory price of the speed, and it is why this is
/// a separate function rather than a replacement.
///
/// # Panics
/// Panics if dtypes are not F32 or shapes disagree.
#[must_use]
// Strides are tensor dimensions, far below isize::MAX; a wrap would mean an
// allocation larger than the address space.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_possible_wrap
)]
pub fn conv2d_fast(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> Tensor {
    assert_eq!(
        input.dtype(),
        vearo_core::DType::F32,
        "conv2d_fast input must be F32"
    );
    let ic = input.contiguous();
    let wc = weight.contiguous();
    let bc = bias.contiguous();
    let id = ic.shape().dims();
    let wd = wc.shape().dims();
    let (n, cin, h, w) = (id[0], id[1], id[2], id[3]);
    let (cout, kh, kw) = (wd[0], wd[2], wd[3]);
    assert_eq!(wd[1], cin, "conv2d_fast channel mismatch");

    let oh = (h + 2 * padding - kh) / stride + 1;
    let ow = (w + 2 * padding - kw) / stride + 1;
    let out_shape = Shape::new([n, cout, oh, ow]);
    let out = Tensor::zeros(out_shape, vearo_core::DType::F32);
    if out_shape.numel() == 0 {
        return out;
    }

    let x = read_f32(&ic);
    let wt = read_f32(&wc);
    let b = read_f32(&bc);

    let patch = cin * kh * kw;
    let spatial = oh * ow;
    let mut cols = vec![0.0f32; patch * spatial];
    let mut out_data = vec![0.0f32; out_shape.numel()];

    for nn in 0..n {
        // im2col: row r = (c, i, j), column s = (y, x_out), so the product with
        // the weight matrix laid out as [cout, patch] gives [cout, spatial].
        for c in 0..cin {
            for i in 0..kh {
                for j in 0..kw {
                    let row = (c * kh + i) * kw + j;
                    let row_off = row * spatial;
                    for y in 0..oh {
                        let ih = y * stride + i;
                        if ih < padding || ih >= h + padding {
                            continue;
                        }
                        let ih = ih - padding;
                        let src_row = ((nn * cin + c) * h + ih) * w;
                        let dst_row = row_off + y * ow;
                        for x_out in 0..ow {
                            let iw = x_out * stride + j;
                            if iw < padding || iw >= w + padding {
                                continue;
                            }
                            cols[dst_row + x_out] = x[src_row + (iw - padding)];
                        }
                    }
                }
            }
        }

        // [cout, patch] x [patch, spatial] -> [cout, spatial], row-major, so
        // column stride is 1 and row stride is the width.
        let dst = &mut out_data[nn * cout * spatial..(nn + 1) * cout * spatial];
        unsafe {
            gemm::gemm(
                cout,
                spatial,
                patch,
                dst.as_mut_ptr(),
                1,
                spatial as isize,
                false,
                wt.as_ptr(),
                1,
                patch as isize,
                cols.as_ptr(),
                1,
                spatial as isize,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }

        for co in 0..cout {
            let base = co * spatial;
            for v in &mut dst[base..base + spatial] {
                *v += b[co];
            }
        }
    }

    write_f32(&out, out_data);
    out
}

/// 2D convolution on CPU (optimized direct with padded input).
///
/// The reference implementation: deterministic, and bit-identical to the CUDA
/// kernel, which was written to match its accumulation order. Use
/// [`conv2d_fast`] when speed matters more than bit-equality.
#[must_use]
// The stride==1 path and the general path share their opening lines; keeping
// them separate is what lets the common case stay branch-free inside the loop.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::branches_sharing_code
)]
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> Tensor {
    assert_eq!(input.dtype(), vearo_core::DType::F32, "conv2d input must be F32");
    let ic = input.contiguous();
    let wc = weight.contiguous();
    let bc = bias.contiguous();

    let id = ic.shape().dims();
    let wd = wc.shape().dims();
    let (n, cin, h, w) = (id[0], id[1], id[2], id[3]);
    let (cout, kh, kw) = (wd[0], wd[2], wd[3]);
    assert_eq!(wd[1], cin, "conv2d channel mismatch");

    let oh = (h + 2 * padding - kh) / stride + 1;
    let ow = (w + 2 * padding - kw) / stride + 1;
    let out_shape = Shape::new([n, cout, oh, ow]);
    let out = Tensor::zeros(out_shape, vearo_core::DType::F32);
    if out_shape.numel() == 0 {
        return out;
    }

    let x = read_f32(&ic);
    let wt = read_f32(&wc);
    let b = read_f32(&bc);
    let mut out_data = vec![0.0f32; out_shape.numel()];

    let hp = h + 2 * padding;
    let wp = w + 2 * padding;
    let mut x_padded = vec![0.0f32; n * cin * hp * wp];
    for nn in 0..n {
        for c in 0..cin {
            for ih in 0..h {
                let src_offset = ((nn * cin + c) * h + ih) * w;
                let dest_offset = ((nn * cin + c) * hp + ih + padding) * wp + padding;
                x_padded[dest_offset..dest_offset + w].copy_from_slice(&x[src_offset..src_offset + w]);
            }
        }
    }

    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let total_tasks = n * cout;
    let tasks_per_thread = total_tasks.div_ceil(threads.max(1));

    let x_padded_ref = &x_padded;
    let wt_ref = &wt;
    let b_ref = &b;
    let out_data_ptr = out_data.as_mut_ptr() as usize;
    let out_data_len = out_data.len();

    std::thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                let start_task = t * tasks_per_thread;
                let end_task = (start_task + tasks_per_thread).min(total_tasks);
                if start_task >= end_task {
                    return;
                }

                let local_out_data = unsafe { std::slice::from_raw_parts_mut(out_data_ptr as *mut f32, out_data_len) };

                for task in start_task..end_task {
                    let nn = task / cout;
                    let co = task % cout;
                    let bias_val = b_ref[co];

                    let out_offset = ((nn * cout + co) * oh) * ow;
                    local_out_data[out_offset .. out_offset + oh * ow].fill(bias_val);

                    for c in 0..cin {
                        for i in 0..kh {
                            for j in 0..kw {
                                let w_val = wt_ref[((co * cin + c) * kh + i) * kw + j];
                                if w_val == 0.0 {
                                    continue;
                                }

                                for y in 0..oh {
                                    let x_row_offset = ((nn * cin + c) * hp + y * stride + i) * wp + j;
                                    let dest_row_offset = out_offset + y * ow;

                                    if stride == 1 {
                                        let dest_slice = &mut local_out_data[dest_row_offset .. dest_row_offset + ow];
                                        let src_slice = &x_padded_ref[x_row_offset .. x_row_offset + ow];
                                        for x_out in 0..ow {
                                            dest_slice[x_out] = w_val.mul_add(src_slice[x_out], dest_slice[x_out]);
                                        }
                                    } else {
                                        let dest_slice = &mut local_out_data[dest_row_offset .. dest_row_offset + ow];
                                        for x_out in 0..ow {
                                            dest_slice[x_out] = w_val.mul_add(x_padded_ref[x_row_offset + x_out * stride], dest_slice[x_out]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    });

    write_f32(&out, out_data);
    out
}

/// 2D convolution backward pass on CPU.
#[must_use]
// The stride==1 fast path and the general path share their opening lines; they
// are kept separate so the common case stays branch-free in the inner loop.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::branches_sharing_code,
    clippy::too_many_lines,
    // Indices address several parallel buffers at once, so an iterator over one
    // of them would not remove the indexing.
    clippy::needless_range_loop
)]
pub fn conv2d_backward(
    input: &Tensor,
    weight: &Tensor,
    grad_output: &Tensor,
    stride: usize,
    padding: usize,
) -> (Tensor, Tensor, Tensor) {
    assert_eq!(input.dtype(), vearo_core::DType::F32);
    let ic = input.contiguous();
    let wc = weight.contiguous();
    let gc = grad_output.contiguous();

    let id = ic.shape().dims();
    let wd = wc.shape().dims();
    let gd = gc.shape().dims();
    let (n, cin, h, w) = (id[0], id[1], id[2], id[3]);
    let (cout, kh, kw) = (wd[0], wd[2], wd[3]);
    let (oh, ow) = (gd[2], gd[3]);

    let grad_input = Tensor::zeros(*ic.shape(), vearo_core::DType::F32);
    let grad_weight = Tensor::zeros(*wc.shape(), vearo_core::DType::F32);
    let grad_bias = Tensor::zeros(Shape::new([cout]), vearo_core::DType::F32);

    let x = read_f32(&ic);
    let wt = read_f32(&wc);
    let go = read_f32(&gc);

    let mut gi_data = vec![0.0f32; ic.shape().numel()];
    let mut gw_data = vec![0.0f32; wc.shape().numel()];
    let mut gb_data = vec![0.0f32; cout];

    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let co_per_thread = cout.div_ceil(threads.max(1));
    let go_ref = &go;
    let gb_ptr = gb_data.as_mut_ptr() as usize;

    std::thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                let co_start = t * co_per_thread;
                let co_end = (co_start + co_per_thread).min(cout);
                if co_start >= co_end {
                    return;
                }
                let local_gb = unsafe { std::slice::from_raw_parts_mut(gb_ptr as *mut f32, cout) };
                for co in co_start..co_end {
                    let mut sum = 0.0f32;
                    for b in 0..n {
                        let base = ((b * cout + co) * oh) * ow;
                        for i in 0..(oh * ow) {
                            sum += go_ref[base + i];
                        }
                    }
                    local_gb[co] = sum;
                }
            });
        }
    });
    write_f32(&grad_bias, gb_data);

    let hp = h + 2 * padding;
    let wp = w + 2 * padding;
    let mut x_padded = vec![0.0f32; n * cin * hp * wp];
    for nn in 0..n {
        for c in 0..cin {
            for ih in 0..h {
                let src_offset = ((nn * cin + c) * h + ih) * w;
                let dest_offset = ((nn * cin + c) * hp + ih + padding) * wp + padding;
                x_padded[dest_offset..dest_offset + w].copy_from_slice(&x[src_offset..src_offset + w]);
            }
        }
    }

    let total_tasks_gw = cout * cin;
    let gw_tasks_per_thread = total_tasks_gw.div_ceil(threads.max(1));
    let x_padded_ref = &x_padded;
    let gw_ptr = gw_data.as_mut_ptr() as usize;
    let gw_data_len = gw_data.len();

    std::thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                let start_task = t * gw_tasks_per_thread;
                let end_task = (start_task + gw_tasks_per_thread).min(total_tasks_gw);
                if start_task >= end_task {
                    return;
                }

                let local_gw = unsafe { std::slice::from_raw_parts_mut(gw_ptr as *mut f32, gw_data_len) };

                for task in start_task..end_task {
                    let co = task / cin;
                    let c = task % cin;

                    for i in 0..kh {
                        for j in 0..kw {
                            let gw_offset = ((co * cin + c) * kh + i) * kw + j;
                            let mut acc = 0.0f32;

                            for nn in 0..n {
                                for y in 0..oh {
                                    let x_row_offset = ((nn * cin + c) * hp + y * stride + i) * wp + j;
                                    let go_row_offset = ((nn * cout + co) * oh + y) * ow;

                                    if stride == 1 {
                                        let src_x = &x_padded_ref[x_row_offset .. x_row_offset + ow];
                                        let src_go = &go_ref[go_row_offset .. go_row_offset + ow];
                                        for x_out in 0..ow {
                                            acc = src_go[x_out].mul_add(src_x[x_out], acc);
                                        }
                                    } else {
                                        for x_out in 0..ow {
                                            let val_x = x_padded_ref[x_row_offset + x_out * stride];
                                            let val_go = go_ref[go_row_offset + x_out];
                                            acc = val_go.mul_add(val_x, acc);
                                        }
                                    }
                                }
                            }
                            local_gw[gw_offset] = acc;
                        }
                    }
                }
            });
        }
    });
    write_f32(&grad_weight, gw_data);

    let mut gi_padded = vec![0.0f32; n * cin * hp * wp];
    let gi_padded_ptr = gi_padded.as_mut_ptr() as usize;
    let gi_padded_len = gi_padded.len();
    let total_tasks_gi = n * cin;
    let gi_tasks_per_thread = total_tasks_gi.div_ceil(threads.max(1));
    let wt_ref = &wt;

    std::thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                let start_task = t * gi_tasks_per_thread;
                let end_task = (start_task + gi_tasks_per_thread).min(total_tasks_gi);
                if start_task >= end_task {
                    return;
                }

                let local_gi_padded = unsafe { std::slice::from_raw_parts_mut(gi_padded_ptr as *mut f32, gi_padded_len) };

                for task in start_task..end_task {
                    let nn = task / cin;
                    let c = task % cin;

                    for co in 0..cout {
                        let go_offset = ((nn * cout + co) * oh) * ow;

                        for i in 0..kh {
                            for j in 0..kw {
                                let w_val = wt_ref[((co * cin + c) * kh + i) * kw + j];
                                if w_val == 0.0 {
                                    continue;
                                }

                                for y in 0..oh {
                                    let gi_row_offset = ((nn * cin + c) * hp + y * stride + i) * wp + j;
                                    let go_row_offset = go_offset + y * ow;

                                    if stride == 1 {
                                        let dest_slice = unsafe { std::slice::from_raw_parts_mut(local_gi_padded.as_mut_ptr().add(gi_row_offset), ow) };
                                        let src_slice = &go_ref[go_row_offset .. go_row_offset + ow];
                                        for x_out in 0..ow {
                                            dest_slice[x_out] = src_slice[x_out].mul_add(w_val, dest_slice[x_out]);
                                        }
                                    } else {
                                        for x_out in 0..ow {
                                            local_gi_padded[gi_row_offset + x_out * stride] = go_ref[go_row_offset + x_out].mul_add(w_val, local_gi_padded[gi_row_offset + x_out * stride]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    });

    for nn in 0..n {
        for c in 0..cin {
            for ih in 0..h {
                let src_offset = ((nn * cin + c) * hp + ih + padding) * wp + padding;
                let dest_offset = ((nn * cin + c) * h + ih) * w;
                gi_data[dest_offset..dest_offset + w].copy_from_slice(&gi_padded[src_offset..src_offset + w]);
            }
        }
    }
    write_f32(&grad_input, gi_data);

    (grad_input, grad_weight, grad_bias)
}

/// 2D max pooling (NCHW). Padded positions are treated as -inf (never selected).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn maxpool2d(input: &Tensor, kernel_size: usize, stride: usize, padding: usize) -> Tensor {
    let ic = input.contiguous();
    let id = ic.shape().dims();
    let (n, c, h, w) = (id[0], id[1], id[2], id[3]);
    let oh = (h + 2 * padding - kernel_size) / stride + 1;
    let ow = (w + 2 * padding - kernel_size) / stride + 1;
    let out_shape = Shape::new([n, c, oh, ow]);
    let out = Tensor::zeros(out_shape, vearo_core::DType::F32);
    if out_shape.numel() == 0 {
        return out;
    }

    let x = read_f32(&ic);
    let mut out_data = vec![0.0f32; out_shape.numel()];

    let total_channels = n * c;
    let in_channel_len = h * w;
    let out_channel_len = oh * ow;

    let compute = |out_channels: &mut [f32], start_channel: usize| {
        for (ch_idx, out_channel_slice) in out_channels.chunks_exact_mut(out_channel_len).enumerate() {
            let nc = start_channel + ch_idx;
            let in_offset = nc * in_channel_len;

            for y in 0..oh {
                for x_out in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    for i in 0..kernel_size {
                        let ih = y * stride + i;
                        if ih < padding || ih >= h + padding {
                            continue;
                        }
                        let ih = ih - padding;
                        for j in 0..kernel_size {
                            let iw = x_out * stride + j;
                            if iw < padding || iw >= w + padding {
                                continue;
                            }
                            let iw = iw - padding;
                            let v = x[in_offset + ih * w + iw];
                            if v > best {
                                best = v;
                            }
                        }
                    }
                    out_channel_slice[y * ow + x_out] = best;
                }
            }
        }
    };

    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let channels_per_thread = total_channels.div_ceil(threads.max(1));
    let chunk_size = channels_per_thread * out_channel_len;

    if threads <= 1 || total_channels < 2 {
        compute(&mut out_data, 0);
    } else {
        std::thread::scope(|s| {
            let compute_ref = &compute;
            for (t, out_chunk) in out_data.chunks_mut(chunk_size).enumerate() {
                s.spawn(move || compute_ref(out_chunk, t * channels_per_thread));
            }
        });
    }

    write_f32(&out, out_data);
    out
}

/// Backward for [`maxpool2d`]: routes each output gradient to the argmax input
/// (first occurrence on ties, matching the forward pass). Returns grad input.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn maxpool2d_backward(
    input: &Tensor,
    grad_out: &Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor {
    let ic = input.contiguous();
    let gc = grad_out.contiguous();
    let id = ic.shape().dims();
    let (n, c, h, w) = (id[0], id[1], id[2], id[3]);
    let gd = gc.shape().dims();
    let (oh, ow) = (gd[2], gd[3]);

    let x = read_f32(&ic);
    let go = read_f32(&gc);
    let mut gi = vec![0.0f32; ic.shape().numel()];

    let total_channels = n * c;
    let in_channel_len = h * w;
    let out_channel_len = oh * ow;

    let compute = |gi_channels: &mut [f32], start_channel: usize| {
        for (ch_idx, gi_channel_slice) in gi_channels.chunks_exact_mut(in_channel_len).enumerate() {
            let nc = start_channel + ch_idx;
            let in_offset = nc * in_channel_len;
            let out_offset = nc * out_channel_len;

            for y in 0..oh {
                for x_out in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    let mut best_idx = usize::MAX;
                    for i in 0..kernel_size {
                        let ih = y * stride + i;
                        if ih < padding || ih >= h + padding {
                            continue;
                        }
                        let ih = ih - padding;
                        for j in 0..kernel_size {
                            let iw = x_out * stride + j;
                            if iw < padding || iw >= w + padding {
                                continue;
                            }
                            let iw = iw - padding;
                            let local_idx = ih * w + iw;
                            let idx = in_offset + local_idx;
                            let v = x[idx];
                            if v > best {
                                best = v;
                                best_idx = local_idx;
                            }
                        }
                    }
                    if best_idx != usize::MAX {
                        let go_idx = out_offset + y * ow + x_out;
                        gi_channel_slice[best_idx] += go[go_idx];
                    }
                }
            }
        }
    };

    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let channels_per_thread = total_channels.div_ceil(threads.max(1));
    let chunk_size = channels_per_thread * in_channel_len;

    if threads <= 1 || total_channels < 2 {
        compute(&mut gi, 0);
    } else {
        std::thread::scope(|s| {
            let compute_ref = &compute;
            for (t, gi_chunk) in gi.chunks_mut(chunk_size).enumerate() {
                s.spawn(move || compute_ref(gi_chunk, t * channels_per_thread));
            }
        });
    }

    Tensor::from_f32(&gi, *ic.shape())
}

/// 2D average pooling (NCHW). Averages over valid (non-padded) positions only.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn avgpool2d(input: &Tensor, kernel_size: usize, stride: usize, padding: usize) -> Tensor {
    let ic = input.contiguous();
    let id = ic.shape().dims();
    let (n, c, h, w) = (id[0], id[1], id[2], id[3]);
    let oh = (h + 2 * padding - kernel_size) / stride + 1;
    let ow = (w + 2 * padding - kernel_size) / stride + 1;
    let out_shape = Shape::new([n, c, oh, ow]);
    let out = Tensor::zeros(out_shape, vearo_core::DType::F32);
    if out_shape.numel() == 0 {
        return out;
    }

    let x = read_f32(&ic);
    let mut out_data = vec![0.0f32; out_shape.numel()];

    for nn in 0..n {
        for cc in 0..c {
            for y in 0..oh {
                for x_out in 0..ow {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for i in 0..kernel_size {
                        let ih = y * stride + i;
                        if ih < padding || ih >= h + padding {
                            continue;
                        }
                        let ih = ih - padding;
                        for j in 0..kernel_size {
                            let iw = x_out * stride + j;
                            if iw < padding || iw >= w + padding {
                                continue;
                            }
                            let iw = iw - padding;
                            sum += x[((nn * c + cc) * h + ih) * w + iw];
                            count += 1;
                        }
                    }
                    out_data[((nn * c + cc) * oh + y) * ow + x_out] = sum / count as f32;
                }
            }
        }
    }

    write_f32(&out, out_data);
    out
}

/// Backward for [`avgpool2d`]: each output gradient is split evenly across the
/// valid input positions in its window. Returns grad input.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn avgpool2d_backward(
    input: &Tensor,
    grad_out: &Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor {
    let ic = input.contiguous();
    let gc = grad_out.contiguous();
    let id = ic.shape().dims();
    let (n, c, h, w) = (id[0], id[1], id[2], id[3]);
    let gd = gc.shape().dims();
    let (oh, ow) = (gd[2], gd[3]);

    let go = read_f32(&gc);
    let mut gi = vec![0.0f32; ic.shape().numel()];

    for nn in 0..n {
        for cc in 0..c {
            for y in 0..oh {
                for x_out in 0..ow {
                    let mut count = 0usize;
                    for i in 0..kernel_size {
                        let ih = y * stride + i;
                        if ih < padding || ih >= h + padding {
                            continue;
                        }
                        for j in 0..kernel_size {
                            let iw = x_out * stride + j;
                            if iw >= padding && iw < w + padding {
                                count += 1;
                            }
                        }
                    }
                    let g = go[((nn * c + cc) * oh + y) * ow + x_out] / count as f32;
                    for i in 0..kernel_size {
                        let ih = y * stride + i;
                        if ih < padding || ih >= h + padding {
                            continue;
                        }
                        let ih = ih - padding;
                        for j in 0..kernel_size {
                            let iw = x_out * stride + j;
                            if iw < padding || iw >= w + padding {
                                continue;
                            }
                            let iw = iw - padding;
                            gi[((nn * c + cc) * h + ih) * w + iw] += g;
                        }
                    }
                }
            }
        }
    }

    Tensor::from_f32(&gi, *ic.shape())
}

/// 2D Batch Normalization forward on CPU (NCHW).
#[must_use]
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments
)]
pub fn batchnorm(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Tensor {
    assert_eq!(x.dtype(), vearo_core::DType::F32, "Only F32 supported");
    let xc = x.contiguous();
    let dims = xc.shape().dims();
    assert_eq!(
        dims.len(),
        4,
        "BatchNorm2d input must be rank 4 (N, C, H, W)"
    );
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let m = (n * h * w) as f32;

    assert_eq!(gamma.shape().rank(), 1, "Gamma must be rank 1 (C)");
    assert_eq!(beta.shape().rank(), 1, "Beta must be rank 1 (C)");
    assert_eq!(gamma.shape()[0], c, "Gamma dim must match channels C");
    assert_eq!(beta.shape()[0], c, "Beta dim must match channels C");

    let out = Tensor::zeros(*x.shape(), vearo_core::DType::F32);
    if x.shape().numel() == 0 {
        return out;
    }

    let x_data = read_f32(&xc);
    let g_data = read_f32(gamma);
    let b_data = read_f32(beta);
    let mut out_data = vec![0.0f32; xc.shape().numel()];

    if training {
        let mut mean_batch = vec![0.0f32; c];
        let mut var_batch = vec![0.0f32; c];

        // 1. Compute mean for each channel
        for cc in 0..c {
            let mut sum = 0.0f32;
            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        sum += x_data[((nn * c + cc) * h + hh) * w + ww];
                    }
                }
            }
            mean_batch[cc] = sum / m;
        }

        // 2. Compute variance for each channel
        for cc in 0..c {
            let mut sum_sq = 0.0f32;
            let mean = mean_batch[cc];
            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        let diff = x_data[((nn * c + cc) * h + hh) * w + ww] - mean;
                        sum_sq += diff * diff;
                    }
                }
            }
            var_batch[cc] = sum_sq / m;
        }

        // 3. Update running stats
        let rm_data_orig = read_f32(running_mean);
        let rv_data_orig = read_f32(running_var);
        let mut rm_data_new = vec![0.0f32; c];
        let mut rv_data_new = vec![0.0f32; c];

        for cc in 0..c {
            rm_data_new[cc] = (1.0 - momentum) * rm_data_orig[cc] + momentum * mean_batch[cc];
            let bessel = if m > 1.0 { m / (m - 1.0) } else { 1.0 };
            rv_data_new[cc] =
                (1.0 - momentum) * rv_data_orig[cc] + momentum * var_batch[cc] * bessel;
        }
        write_f32(running_mean, rm_data_new);
        write_f32(running_var, rv_data_new);

        // 4. Normalize and scale/shift
        for cc in 0..c {
            let mean = mean_batch[cc];
            let var = var_batch[cc];
            let inv_std = 1.0 / (var + eps).sqrt();
            let gamma_val = g_data[cc];
            let beta_val = b_data[cc];

            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        let idx = ((nn * c + cc) * h + hh) * w + ww;
                        let x_hat = (x_data[idx] - mean) * inv_std;
                        out_data[idx] = x_hat * gamma_val + beta_val;
                    }
                }
            }
        }
    } else {
        // Eval mode: use running_mean and running_var
        let rm_data = read_f32(running_mean);
        let rv_data = read_f32(running_var);

        for cc in 0..c {
            let mean = rm_data[cc];
            let var = rv_data[cc];
            let inv_std = 1.0 / (var + eps).sqrt();
            let gamma_val = g_data[cc];
            let beta_val = b_data[cc];

            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        let idx = ((nn * c + cc) * h + hh) * w + ww;
                        let x_hat = (x_data[idx] - mean) * inv_std;
                        out_data[idx] = x_hat * gamma_val + beta_val;
                    }
                }
            }
        }
    }

    write_f32(&out, out_data);
    out
}

/// 2D Batch Normalization backward on CPU (NCHW).
#[must_use]
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub fn batchnorm_backward(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    grad_out: &Tensor,
    training: bool,
    eps: f32,
) -> (Tensor, Tensor, Tensor) {
    let xc = x.contiguous();
    let gc = grad_out.contiguous();
    let dims = xc.shape().dims();
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let m = (n * h * w) as f32;

    let grad_x = Tensor::zeros(*x.shape(), vearo_core::DType::F32);
    let grad_w = Tensor::zeros(*gamma.shape(), vearo_core::DType::F32);
    let grad_b = Tensor::zeros(*beta.shape(), vearo_core::DType::F32);

    if x.shape().numel() == 0 {
        return (grad_x, grad_w, grad_b);
    }

    let x_data = read_f32(&xc);
    let g_data = read_f32(gamma);
    let go_data = read_f32(&gc);

    let mut gx_data = vec![0.0f32; xc.shape().numel()];
    let mut gw_data = vec![0.0f32; c];
    let mut gb_data = vec![0.0f32; c];

    if training {
        let mut mean_batch = vec![0.0f32; c];
        let mut var_batch = vec![0.0f32; c];

        // 1. Recompute mean/var
        for cc in 0..c {
            let mut sum = 0.0f32;
            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        sum += x_data[((nn * c + cc) * h + hh) * w + ww];
                    }
                }
            }
            mean_batch[cc] = sum / m;
        }

        for cc in 0..c {
            let mut sum_sq = 0.0f32;
            let mean = mean_batch[cc];
            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        let diff = x_data[((nn * c + cc) * h + hh) * w + ww] - mean;
                        sum_sq += diff * diff;
                    }
                }
            }
            var_batch[cc] = sum_sq / m;
        }

        // 2. Compute grad_bias and grad_weight
        for cc in 0..c {
            let mean = mean_batch[cc];
            let var = var_batch[cc];
            let inv_std = 1.0 / (var + eps).sqrt();

            let mut sum_dy = 0.0f32;
            let mut sum_dy_xhat = 0.0f32;

            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        let idx = ((nn * c + cc) * h + hh) * w + ww;
                        let dy = go_data[idx];
                        let x_hat = (x_data[idx] - mean) * inv_std;
                        sum_dy += dy;
                        sum_dy_xhat += dy * x_hat;
                    }
                }
            }
            gb_data[cc] = sum_dy;
            gw_data[cc] = sum_dy_xhat;
        }

        // 3. Compute grad_x
        for cc in 0..c {
            let mean = mean_batch[cc];
            let var = var_batch[cc];
            let inv_std = 1.0 / (var + eps).sqrt();
            let gamma_val = g_data[cc];
            let sum_dy = gb_data[cc];
            let sum_dy_xhat = gw_data[cc];

            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        let idx = ((nn * c + cc) * h + hh) * w + ww;
                        let dy = go_data[idx];
                        let x_hat = (x_data[idx] - mean) * inv_std;
                        let gx =
                            (gamma_val * inv_std / m) * (m * dy - sum_dy - x_hat * sum_dy_xhat);
                        gx_data[idx] = gx;
                    }
                }
            }
        }
    } else {
        // Eval mode backward
        let rm_data = read_f32(running_mean);
        let rv_data = read_f32(running_var);

        // 1. Compute grad_bias and grad_weight
        for cc in 0..c {
            let mean = rm_data[cc];
            let var = rv_data[cc];
            let inv_std = 1.0 / (var + eps).sqrt();

            let mut sum_dy = 0.0f32;
            let mut sum_dy_xhat = 0.0f32;

            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        let idx = ((nn * c + cc) * h + hh) * w + ww;
                        let dy = go_data[idx];
                        let x_hat = (x_data[idx] - mean) * inv_std;
                        sum_dy += dy;
                        sum_dy_xhat += dy * x_hat;
                    }
                }
            }
            gb_data[cc] = sum_dy;
            gw_data[cc] = sum_dy_xhat;
        }

        // 2. Compute grad_x
        for cc in 0..c {
            let var = rv_data[cc];
            let inv_std = 1.0 / (var + eps).sqrt();
            let gamma_val = g_data[cc];

            for nn in 0..n {
                for hh in 0..h {
                    for ww in 0..w {
                        let idx = ((nn * c + cc) * h + hh) * w + ww;
                        let dy = go_data[idx];
                        gx_data[idx] = dy * gamma_val * inv_std;
                    }
                }
            }
        }
    }

    write_f32(&grad_x, gx_data);
    write_f32(&grad_w, gw_data);
    write_f32(&grad_b, gb_data);
    (grad_x, grad_w, grad_b)
}

/// Fused scaled dot-product attention forward pass on CPU.
#[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
pub fn fused_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
    assert_eq!(q.dtype(), vearo_core::DType::F32);
    assert_eq!(k.dtype(), vearo_core::DType::F32);
    assert_eq!(v.dtype(), vearo_core::DType::F32);

    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    let v_shape = v.shape().dims();

    let b = q_shape[0];
    let h = q_shape[1];
    let s = q_shape[2];
    let d_k = q_shape[3];

    assert_eq!(k_shape, q_shape);
    assert_eq!(v_shape, q_shape);

    let q_cont = q.contiguous();
    let k_cont = k.contiguous();
    let v_cont = v.contiguous();

    let q_data = read_f32(&q_cont);
    let k_data = read_f32(&k_cont);
    let v_data = read_f32(&v_cont);

    let mask_cont = mask.map(Tensor::contiguous);
    let mask_data = mask_cont.as_ref().map(read_f32);
    // Right-align the mask shape to rank 4. The unfused path adds the mask with
    // broadcasting, so a causal mask is usually [S, S] rather than [B, H, S, S];
    // indexing a shorter shape directly panics. Left-padding with ones keeps the
    // fused path a drop-in replacement for the broadcast add.
    let mask_shape = mask_cont.as_ref().map(|m| {
        let dims = m.shape().dims().to_vec();
        let mut padded = vec![1usize; 4_usize.saturating_sub(dims.len())];
        padded.extend(dims);
        padded
    });

    let scale = 1.0 / (d_k as f32).sqrt();
    let mut out_data = vec![0.0f32; b * h * s * d_k];

    let bh_stride = s * d_k;
    let b_stride = h * bh_stride;

    for b_idx in 0..b {
        for h_idx in 0..h {
            let offset = b_idx * b_stride + h_idx * bh_stride;

            for i in 0..s {
                let q_row = &q_data[(offset + i * d_k)..(offset + (i + 1) * d_k)];

                let mut scores = vec![0.0f32; s];
                let mut max_score = -f32::INFINITY;

                for j in 0..s {
                    let k_row = &k_data[(offset + j * d_k)..(offset + (j + 1) * d_k)];
                    let mut dot = 0.0f32;
                    for d in 0..d_k {
                        dot = q_row[d].mul_add(k_row[d], dot);
                    }
                    dot *= scale;

                    if let Some(ref m_data) = mask_data {
                        let m_shape = mask_shape.as_ref().unwrap();
                        let mb = if m_shape[0] == 1 { 0 } else { b_idx };
                        let mh = if m_shape[1] == 1 { 0 } else { h_idx };
                        let ms = if m_shape[2] == 1 { 0 } else { i };
                        let ms_col = if m_shape[3] == 1 { 0 } else { j };

                        let m_idx = mb * (m_shape[1] * m_shape[2] * m_shape[3])
                            + mh * (m_shape[2] * m_shape[3])
                            + ms * m_shape[3]
                            + ms_col;
                        dot += m_data[m_idx];
                    }

                    scores[j] = dot;
                    if dot > max_score {
                        max_score = dot;
                    }
                }

                let mut sum_exp = 0.0f32;
                for j in 0..s {
                    scores[j] = (scores[j] - max_score).exp();
                    sum_exp += scores[j];
                }
                let inv_sum = 1.0 / (sum_exp + 1e-9);
                for j in 0..s {
                    scores[j] *= inv_sum;
                }

                let out_row_offset = offset + i * d_k;
                for d in 0..d_k {
                    let mut sum_val = 0.0f32;
                    for j in 0..s {
                        let v_val = v_data[offset + j * d_k + d];
                        sum_val = scores[j].mul_add(v_val, sum_val);
                    }
                    out_data[out_row_offset + d] = sum_val;
                }
            }
        }
    }

    let out = Tensor::zeros(*q.shape(), vearo_core::DType::F32);
    write_f32(&out, out_data);
    out
}

/// Fused scaled dot-product attention backward pass on CPU.
#[allow(clippy::many_single_char_names, clippy::needless_range_loop, clippy::too_many_lines)]
pub fn fused_attention_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    grad_out: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let q_shape = q.shape().dims();
    let b = q_shape[0];
    let h = q_shape[1];
    let s = q_shape[2];
    let d_k = q_shape[3];

    let q_cont = q.contiguous();
    let k_cont = k.contiguous();
    let v_cont = v.contiguous();
    let go_cont = grad_out.contiguous();

    let q_data = read_f32(&q_cont);
    let k_data = read_f32(&k_cont);
    let v_data = read_f32(&v_cont);
    let go_data = read_f32(&go_cont);

    let mask_cont = mask.map(Tensor::contiguous);
    let mask_data = mask_cont.as_ref().map(read_f32);
    // Right-align the mask shape to rank 4. The unfused path adds the mask with
    // broadcasting, so a causal mask is usually [S, S] rather than [B, H, S, S];
    // indexing a shorter shape directly panics. Left-padding with ones keeps the
    // fused path a drop-in replacement for the broadcast add.
    let mask_shape = mask_cont.as_ref().map(|m| {
        let dims = m.shape().dims().to_vec();
        let mut padded = vec![1usize; 4_usize.saturating_sub(dims.len())];
        padded.extend(dims);
        padded
    });

    let scale = 1.0 / (d_k as f32).sqrt();

    let mut dq_data = vec![0.0f32; b * h * s * d_k];
    let mut dk_data = vec![0.0f32; b * h * s * d_k];
    let mut dv_data = vec![0.0f32; b * h * s * d_k];

    let bh_stride = s * d_k;
    let b_stride = h * bh_stride;

    for b_idx in 0..b {
        for h_idx in 0..h {
            let offset = b_idx * b_stride + h_idx * bh_stride;

            for i in 0..s {
                let q_row = &q_data[(offset + i * d_k)..(offset + (i + 1) * d_k)];

                let mut scores = vec![0.0f32; s];
                let mut max_score = -f32::INFINITY;

                for j in 0..s {
                    let k_row = &k_data[(offset + j * d_k)..(offset + (j + 1) * d_k)];
                    let mut dot = 0.0f32;
                    for d in 0..d_k {
                        dot = q_row[d].mul_add(k_row[d], dot);
                    }
                    dot *= scale;

                    if let Some(ref m_data) = mask_data {
                        let m_shape = mask_shape.as_ref().unwrap();
                        let mb = if m_shape[0] == 1 { 0 } else { b_idx };
                        let mh = if m_shape[1] == 1 { 0 } else { h_idx };
                        let ms = if m_shape[2] == 1 { 0 } else { i };
                        let ms_col = if m_shape[3] == 1 { 0 } else { j };

                        let m_idx = mb * (m_shape[1] * m_shape[2] * m_shape[3])
                            + mh * (m_shape[2] * m_shape[3])
                            + ms * m_shape[3]
                            + ms_col;
                        dot += m_data[m_idx];
                    }

                    scores[j] = dot;
                    if dot > max_score {
                        max_score = dot;
                    }
                }

                let mut sum_exp = 0.0f32;
                for j in 0..s {
                    scores[j] = (scores[j] - max_score).exp();
                    sum_exp += scores[j];
                }
                let inv_sum = 1.0 / (sum_exp + 1e-9);
                for j in 0..s {
                    scores[j] *= inv_sum;
                }

                let go_row = &go_data[(offset + i * d_k)..(offset + (i + 1) * d_k)];
                let mut dp = vec![0.0f32; s];
                for j in 0..s {
                    let mut sum_val = 0.0f32;
                    for d in 0..d_k {
                        sum_val = go_row[d].mul_add(v_data[offset + j * d_k + d], sum_val);
                    }
                    dp[j] = sum_val;
                }

                let mut sum_dp_p = 0.0f32;
                for j in 0..s {
                    sum_dp_p = dp[j].mul_add(scores[j], sum_dp_p);
                }

                let mut ds = vec![0.0f32; s];
                for j in 0..s {
                    ds[j] = scores[j] * (dp[j] - sum_dp_p);
                }

                let dq_row_offset = offset + i * d_k;
                for d in 0..d_k {
                    let mut sum_val = 0.0f32;
                    for j in 0..s {
                        sum_val = ds[j].mul_add(k_data[offset + j * d_k + d], sum_val);
                    }
                    dq_data[dq_row_offset + d] = sum_val * scale;
                }

                for j in 0..s {
                    let dv_idx = offset + j * d_k;
                    let p_ij = scores[j];
                    for d in 0..d_k {
                        dv_data[dv_idx + d] = p_ij.mul_add(go_row[d], dv_data[dv_idx + d]);
                    }
                }

                for j in 0..s {
                    let dk_idx = offset + j * d_k;
                    let ds_ij = ds[j] * scale;
                    for d in 0..d_k {
                        dk_data[dk_idx + d] = ds_ij.mul_add(q_row[d], dk_data[dk_idx + d]);
                    }
                }
            }
        }
    }

    let dq = Tensor::zeros(*q.shape(), vearo_core::DType::F32);
    let dk = Tensor::zeros(*q.shape(), vearo_core::DType::F32);
    let dv = Tensor::zeros(*q.shape(), vearo_core::DType::F32);

    write_f32(&dq, dq_data);
    write_f32(&dk, dk_data);
    write_f32(&dv, dv_data);

    (dq, dk, dv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], [3]);
        let y = Tensor::from_f32(&[4.0, 5.0, 6.0], [3]);
        let z = add(&x, &y);

        let guard = get_cpu_shard(z.storage_id().shard_idx as usize)
            .lock()
            .unwrap();
        match &guard.slots[z.storage_id().slot_idx as usize]
            .as_ref()
            .unwrap()
            .storage
        {
            CpuStorage::F32(vec) => assert_eq!(vec.as_ref(), &vec![5.0, 7.0, 9.0]),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_matmul_2d() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let y = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], [2, 2]);
        let z = matmul(&x, &y);

        let guard = get_cpu_shard(z.storage_id().shard_idx as usize)
            .lock()
            .unwrap();
        match &guard.slots[z.storage_id().slot_idx as usize]
            .as_ref()
            .unwrap()
            .storage
        {
            CpuStorage::F32(vec) => assert_eq!(vec.as_ref(), &vec![19.0, 22.0, 43.0, 50.0]),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_add_broadcasting() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], [1, 3]);
        let y = Tensor::from_f32(&[10.0, 20.0], [2, 1]);
        let z = add(&x, &y);

        assert_eq!(z.shape().dims(), &[2, 3]);

        let guard = get_cpu_shard(z.storage_id().shard_idx as usize)
            .lock()
            .unwrap();
        match &guard.slots[z.storage_id().slot_idx as usize]
            .as_ref()
            .unwrap()
            .storage
        {
            CpuStorage::F32(vec) => assert_eq!(
                vec.as_ref(),
                &vec![
                    11.0, 12.0, 13.0, // y = 10
                    21.0, 22.0, 23.0 // y = 20
                ]
            ),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_add_transposed() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let x_t = x.transpose(0, 1);

        let y = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [3, 2]);
        let z = add(&x_t, &y);

        let guard = get_cpu_shard(z.storage_id().shard_idx as usize)
            .lock()
            .unwrap();
        match &guard.slots[z.storage_id().slot_idx as usize]
            .as_ref()
            .unwrap()
            .storage
        {
            CpuStorage::F32(vec) => assert_eq!(
                vec.as_ref(),
                &vec![
                    11.0, 24.0, // x_t[0] = [1, 4] + [10, 20]
                    32.0, 45.0, // x_t[1] = [2, 5] + [30, 40]
                    53.0, 66.0 // x_t[2] = [3, 6] + [50, 60]
                ]
            ),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_matmul_batched() {
        let x = Tensor::from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 0
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 1
            ],
            [2, 2, 3],
        );
        let y = Tensor::from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 0
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 1
            ],
            [2, 3, 2],
        );
        let z = matmul(&x, &y);

        assert_eq!(z.shape().dims(), &[2, 2, 2]);

        let guard = get_cpu_shard(z.storage_id().shard_idx as usize)
            .lock()
            .unwrap();
        match &guard.slots[z.storage_id().slot_idx as usize]
            .as_ref()
            .unwrap()
            .storage
        {
            CpuStorage::F32(vec) => assert_eq!(
                vec.as_ref(),
                &vec![
                    22.0, 28.0, 49.0, 64.0, // Batch 0 matmul
                    220.0, 244.0, 301.0, 334.0 // Batch 1 matmul
                ]
            ),
            _ => unreachable!(),
        }
    }

    // Exercises the public dispatch path: Tensor::add -> BACKEND_OPS registry ->
    // this crate's `add`. The other tests call the free functions directly and
    // never touch registration, so this is the only guard on the wiring.
    #[test]
    fn test_dispatch_through_registry() {
        init();
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], [3]);
        let y = Tensor::from_f32(&[4.0, 5.0, 6.0], [3]);
        let z = x.add(&y); // goes through the registry, not the free fn

        let guard = get_cpu_shard(z.storage_id().shard_idx as usize)
            .lock()
            .unwrap();
        match &guard.slots[z.storage_id().slot_idx as usize]
            .as_ref()
            .unwrap()
            .storage
        {
            CpuStorage::F32(vec) => assert_eq!(vec.as_ref(), &vec![5.0, 7.0, 9.0]),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_relu() {
        let x = Tensor::from_f32(&[1.0, -2.0, 3.0, -0.5], [4]);
        assert_eq!(*read_f32(&relu(&x)), vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_sum_dim() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        let s1 = sum(&x, 1, false); // over columns -> [6, 15]
        assert_eq!(s1.shape().dims(), &[2]);
        assert_eq!(*read_f32(&s1), vec![6.0, 15.0]);

        let s0 = sum(&x, 0, false); // over rows -> [5, 7, 9]
        assert_eq!(s0.shape().dims(), &[3]);
        assert_eq!(*read_f32(&s0), vec![5.0, 7.0, 9.0]);

        let sk = sum(&x, 1, true); // keep_dim -> [2, 1]
        assert_eq!(sk.shape().dims(), &[2, 1]);
        assert_eq!(*read_f32(&sk), vec![6.0, 15.0]);
    }

    #[test]
    fn test_mean_dim() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        assert_eq!(*read_f32(&mean(&x, 1, false)), vec![2.0, 5.0]);
        assert_eq!(*read_f32(&mean(&x, 0, false)), vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_sum_transposed_input() {
        // Reducing a non-contiguous (transposed) tensor must still be correct.
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let xt = x.transpose(0, 1); // logical [3, 2] = [[1,4],[2,5],[3,6]]
        let s = sum(&xt, 1, false); // row sums -> [5, 7, 9]
        assert_eq!(s.shape().dims(), &[3]);
        assert_eq!(*read_f32(&s), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_empty_tensor_operations() {
        let x = Tensor::zeros([2, 0, 3], vearo_core::DType::F32);
        let x_t = x.transpose(0, 1);
        let r = x_t.reshape([0, 6]);
        assert_eq!(r.shape().dims(), &[0, 6]);
        assert_eq!(r.shape().numel(), 0);

        let y = Tensor::zeros([0, 6], vearo_core::DType::F32);
        let z = add(&r, &y);
        assert_eq!(z.shape().dims(), &[0, 6]);
        assert_eq!(z.shape().numel(), 0);
    }

    #[test]
    fn test_layernorm_non_contiguous_and_empty() {
        init();
        // 1. Empty input
        let x_empty = Tensor::zeros([2, 0, 4], vearo_core::DType::F32);
        let weight = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], [4]);
        let bias = Tensor::from_f32(&[0.0, 0.0, 0.0, 0.0], [4]);
        let out_empty = layernorm(&x_empty, &weight, &bias, 1e-5);
        assert_eq!(out_empty.shape().dims(), &[2, 0, 4]);

        // 2. Non-contiguous input
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4]);
        let x_transposed = x.transpose(0, 1); // shape [4, 2]
        let weight_2 = Tensor::from_f32(&[1.0, 1.0], [2]);
        let bias_2 = Tensor::from_f32(&[0.0, 0.0], [2]);
        let out = layernorm(&x_transposed, &weight_2, &bias_2, 1e-5);
        assert_eq!(out.shape().dims(), &[4, 2]);
    }

    #[test]
    #[should_panic(expected = "Token ID cannot be negative")]
    fn test_embedding_negative_panic() {
        init();
        let x = Tensor::from_f32(&[-1.0, 0.0], [2]);
        let weight = Tensor::zeros([2, 3], vearo_core::DType::F32);
        let _ = embedding(&x, &weight);
    }

    #[test]
    #[should_panic(expected = "out of vocabulary bound")]
    fn test_embedding_oob_panic() {
        init();
        let x = Tensor::from_f32(&[2.0, 0.0], [2]);
        let weight = Tensor::zeros([2, 3], vearo_core::DType::F32);
        let _ = embedding(&x, &weight);
    }

    #[test]
    #[should_panic(expected = "Target class cannot be negative")]
    fn test_cross_entropy_negative_panic() {
        init();
        let logits = Tensor::zeros([2, 3], vearo_core::DType::F32);
        let targets = Tensor::from_f32(&[-1.0, 0.0], [2]);
        let _ = cross_entropy(&logits, &targets);
    }

    #[test]
    #[should_panic(expected = "out of vocab bounds")]
    fn test_cross_entropy_oob_panic() {
        init();
        let logits = Tensor::zeros([2, 3], vearo_core::DType::F32);
        let targets = Tensor::from_f32(&[3.0, 0.0], [2]);
        let _ = cross_entropy(&logits, &targets);
    }

    #[test]
    fn test_cross_entropy_numerical_stability() {
        init();
        // Very large logits that would overflow if not stabilized with max subtraction.
        // Target is class 0, which has the large logit 1000.0. Class 1 has -1000.0.
        let logits = Tensor::from_f32(&[1000.0, -1000.0], [1, 2]);
        let targets = Tensor::from_f32(&[0.0], [1]);
        let loss = cross_entropy(&logits, &targets);
        let loss_val = read_f32(&loss)[0];
        // log(exp(1000-1000) + exp(-1000-1000)) = log(1 + exp(-2000)) ~= 0
        // loss = - (1000 - 1000 - 0) = 0
        assert!(loss_val.abs() < 1e-4);
    }

    #[test]
    fn test_conv2d_forward() {
        // input [1,1,3,3] = 1..9, weight [1,1,2,2] = [[1,0],[0,1]] (picks corners),
        // bias 0, stride 1, padding 0 -> out [1,1,2,2].
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 3, 3]);
        let weight = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], [1, 1, 2, 2]);
        let bias = Tensor::from_f32(&[0.0], [1]);

        let out = conv2d(&x, &weight, &bias, 1, 0);
        assert_eq!(out.shape().dims(), &[1, 1, 2, 2]);
        // out[i,j] = in[i,j] + in[i+1,j+1]: 1+5, 2+6, 4+8, 5+9
        assert_eq!(read_f32(&out).to_vec(), vec![6.0, 8.0, 12.0, 14.0]);
    }
}
