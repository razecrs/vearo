//! CPU reference backend implementations.

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

/// Clone out the F32 backing buffer of a tensor under its shard lock.
///
/// Copying out lets the caller compute with **no locks held**, so an op never
/// pins a shard for the duration of its work.
fn read_f32(t: &Tensor) -> Vec<f32> {
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
    match &mut guard.slots[t.storage_id().slot_idx as usize]
        .as_mut()
        .expect("Output slot was empty")
        .storage
    {
        CpuStorage::F32(v) => *v = data,
        _ => panic!("Only F32 supported"),
    }
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

/// Matrix multiplication of two CPU tensors.
///
/// # Panics
/// Panics if ranks are less than 2, trailing dimensions do not match, or dtypes are not F32.
#[must_use]
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

    // Copy operands out (each under its own brief lock), then compute unlocked.
    let lhs_data = read_f32(lhs);
    let rhs_data = read_f32(rhs);
    let mut out_data = vec![0.0f32; out_shape.numel()];

    // Strides / batch metadata are loop-invariant - compute them once.
    let l_stride_m = lhs.strides()[rank_l - 2];
    let l_stride_k = lhs.strides()[rank_l - 1];
    let r_stride_k = rhs.strides()[rank_r - 2];
    let r_stride_n = rhs.strides()[rank_r - 1];
    let out_stride_m = out_tensor.strides()[out_tensor.strides().rank() - 2];
    let out_stride_n = out_tensor.strides()[out_tensor.strides().rank() - 1];
    let batch_strides_l = Shape::new(&lhs.strides().dims()[..rank_l - 2]);
    let batch_strides_r = Shape::new(&rhs.strides().dims()[..rank_r - 2]);
    let out_batch_dims = Shape::new(&out_shape.dims()[..out_shape.rank() - 2]);
    let out_batch_strides =
        Shape::new(&out_tensor.strides().dims()[..out_tensor.strides().rank() - 2]);

    let mut iter = NdIterator::new(out_batch_shape);
    loop {
        let batch_coord = iter.coord();
        let batch_offset_l = get_offset(batch_coord, &batch_shape_l, &batch_strides_l);
        let batch_offset_r = get_offset(batch_coord, &batch_shape_r, &batch_strides_r);
        let batch_offset_out = get_offset(batch_coord, &out_batch_dims, &out_batch_strides);

        for r_m in 0..m {
            for r_n in 0..n {
                let mut sum = 0.0;
                for r_k in 0..k_l {
                    let l_idx = batch_offset_l + r_m * l_stride_m + r_k * l_stride_k;
                    let r_idx = batch_offset_r + r_k * r_stride_k + r_n * r_stride_n;
                    sum = lhs_data[l_idx].mul_add(rhs_data[r_idx], sum);
                }
                let out_idx = batch_offset_out + r_m * out_stride_m + r_n * out_stride_n;
                out_data[out_idx] = sum;
            }
        }

        if !iter.step() {
            break;
        }
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
            CpuStorage::F32(vec) => assert_eq!(vec, &vec![5.0, 7.0, 9.0]),
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
            CpuStorage::F32(vec) => assert_eq!(vec, &vec![19.0, 22.0, 43.0, 50.0]),
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
                vec,
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
                vec,
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
                vec,
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
            CpuStorage::F32(vec) => assert_eq!(vec, &vec![5.0, 7.0, 9.0]),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_relu() {
        let x = Tensor::from_f32(&[1.0, -2.0, 3.0, -0.5], [4]);
        assert_eq!(read_f32(&relu(&x)), vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_sum_dim() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        let s1 = sum(&x, 1, false); // over columns -> [6, 15]
        assert_eq!(s1.shape().dims(), &[2]);
        assert_eq!(read_f32(&s1), vec![6.0, 15.0]);

        let s0 = sum(&x, 0, false); // over rows -> [5, 7, 9]
        assert_eq!(s0.shape().dims(), &[3]);
        assert_eq!(read_f32(&s0), vec![5.0, 7.0, 9.0]);

        let sk = sum(&x, 1, true); // keep_dim -> [2, 1]
        assert_eq!(sk.shape().dims(), &[2, 1]);
        assert_eq!(read_f32(&sk), vec![6.0, 15.0]);
    }

    #[test]
    fn test_mean_dim() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        assert_eq!(read_f32(&mean(&x, 1, false)), vec![2.0, 5.0]);
        assert_eq!(read_f32(&mean(&x, 0, false)), vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_sum_transposed_input() {
        // Reducing a non-contiguous (transposed) tensor must still be correct.
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let xt = x.transpose(0, 1); // logical [3, 2] = [[1,4],[2,5],[3,6]]
        let s = sum(&xt, 1, false); // row sums -> [5, 7, 9]
        assert_eq!(s.shape().dims(), &[3]);
        assert_eq!(read_f32(&s), vec![5.0, 7.0, 9.0]);
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
}
