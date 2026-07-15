//! CUDA backend implementations for Vearo.
#![allow(
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::{LazyLock, Mutex, OnceLock};
use vearo_core::{
    BackendOps, DType, Device, Shape, StorageId, Tensor, register_backend_ops, register_cuda_hooks,
    register_refcount_dec, register_refcount_inc,
};

/// The global CudaDevice instance.
pub static CUDA_DEVICE: OnceLock<std::sync::Arc<CudaDevice>> = OnceLock::new();

/// Get or initialize the global CudaDevice.
pub fn get_cuda_device() -> std::sync::Arc<CudaDevice> {
    CUDA_DEVICE
        .get_or_init(|| CudaDevice::new(0).expect("Failed to initialize CUDA device 0"))
        .clone()
}

pub struct CudaSlot {
    pub slice: CudaSlice<f32>,
    pub ref_count: usize,
}

pub static CUDA_SLOTS: LazyLock<Mutex<Vec<Option<CudaSlot>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
pub static FREE_CUDA_SLOTS: LazyLock<Mutex<Vec<u32>>> = LazyLock::new(|| Mutex::new(Vec::new()));

pub fn cuda_alloc(numel: usize) -> StorageId {
    let dev = get_cuda_device();
    let slice = dev
        .alloc_zeros::<f32>(numel)
        .expect("Failed to allocate CUDA memory");

    let mut slots = CUDA_SLOTS.lock().unwrap();
    let mut free = FREE_CUDA_SLOTS.lock().unwrap();

    let slot = CudaSlot {
        slice,
        ref_count: 1,
    };

    if let Some(idx) = free.pop() {
        slots[idx as usize] = Some(slot);
        StorageId {
            shard_idx: 0,
            slot_idx: idx,
        }
    } else {
        let idx = slots.len() as u32;
        slots.push(Some(slot));
        StorageId {
            shard_idx: 0,
            slot_idx: idx,
        }
    }
}

pub fn cuda_write(storage_id: StorageId, data: &[f32]) {
    let dev = get_cuda_device();
    let mut slots = CUDA_SLOTS.lock().unwrap();
    let slot = slots[storage_id.slot_idx as usize]
        .as_mut()
        .expect("Slot was empty");
    dev.htod_copy_into(data.to_vec(), &mut slot.slice)
        .expect("Failed to copy host to device");
}

pub fn cuda_read(storage_id: StorageId) -> Vec<f32> {
    let dev = get_cuda_device();
    let slots = CUDA_SLOTS.lock().unwrap();
    let slot = slots[storage_id.slot_idx as usize]
        .as_ref()
        .expect("Slot was empty");
    dev.dtoh_sync_copy(&slot.slice)
        .expect("Failed to copy device to host")
}

pub fn cuda_refcount_inc(storage_id: StorageId, device: Device) {
    if device.is_cuda() {
        let mut slots = CUDA_SLOTS.lock().unwrap();
        if let Some(ref mut slot) = slots[storage_id.slot_idx as usize] {
            slot.ref_count += 1;
        }
    }
}

pub fn cuda_refcount_dec(storage_id: StorageId, device: Device) -> bool {
    if device.is_cuda() {
        let mut slots = CUDA_SLOTS.lock().unwrap();
        let mut free = false;
        if let Some(ref mut slot) = slots[storage_id.slot_idx as usize] {
            assert!(slot.ref_count > 0, "Reference count underflow");
            slot.ref_count -= 1;
            if slot.ref_count == 0 {
                free = true;
            }
        }
        if free {
            slots[storage_id.slot_idx as usize] = None;
            FREE_CUDA_SLOTS.lock().unwrap().push(storage_id.slot_idx);
        }
        free
    } else {
        false
    }
}

pub fn init() {
    register_refcount_inc(cuda_refcount_inc);
    register_refcount_dec(cuda_refcount_dec);
    register_cuda_hooks(cuda_read, cuda_write, cuda_alloc);

    let dev = get_cuda_device();
    if !dev.has_func("vearo_kernels", "add_broadcast_kernel") {
        let ptx_content = include_str!("kernels.ptx");
        dev.load_ptx(
            ptx_content.into(),
            "vearo_kernels",
            &[
                "add_broadcast_kernel",
                "sub_broadcast_kernel",
                "mul_broadcast_kernel",
                "div_broadcast_kernel",
                "relu_forward",
                "relu_backward",
                "gelu_forward",
                "gelu_backward",
                "sum_kernel",
                "mean_kernel",
                "softmax_forward",
                "softmax_backward",
                "layernorm_forward",
                "layernorm_backward",
                "embedding_forward",
                "embedding_backward",
                "cross_entropy_forward",
                "cross_entropy_backward",
                "matmul_kernel",
            ],
        )
        .expect("Failed to load Vearo CUDA kernels");
    }

    register_backend_ops(
        Device::Cuda(0),
        BackendOps {
            add,
            sub,
            mul,
            div,
            matmul,
            sum,
            mean,
            relu,
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
        },
    );
}

/// Convolution is not yet implemented on the CUDA backend (run conv on CPU).
pub fn conv2d(
    _input: &Tensor,
    _weight: &Tensor,
    _bias: &Tensor,
    _stride: usize,
    _padding: usize,
) -> Tensor {
    unimplemented!("conv2d is not yet implemented on the CUDA backend; use the CPU backend")
}

/// Convolution backward is not yet implemented on the CUDA backend.
pub fn conv2d_backward(
    _input: &Tensor,
    _weight: &Tensor,
    _grad_out: &Tensor,
    _stride: usize,
    _padding: usize,
) -> (Tensor, Tensor, Tensor) {
    unimplemented!(
        "conv2d_backward is not yet implemented on the CUDA backend; use the CPU backend"
    )
}

fn binary_op(lhs: &Tensor, rhs: &Tensor, kernel_name: &str) -> Tensor {
    let dev = get_cuda_device();
    let out_shape = lhs
        .shape()
        .broadcast(rhs.shape())
        .expect("Shapes not broadcastable");
    let out_numel = out_shape.numel();
    let out_storage = cuda_alloc(out_numel);
    let out_tensor = Tensor::from_components(
        out_storage,
        out_shape,
        out_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if out_numel == 0 {
        return out_tensor;
    }

    let slots = CUDA_SLOTS.lock().unwrap();
    let lhs_slice = &slots[lhs.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let rhs_slice = &slots[rhs.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let mut info = vec![0i32; 51];
    info[0] = out_shape.rank() as i32;
    info[1] = lhs.shape().rank() as i32;
    info[2] = rhs.shape().rank() as i32;

    for (i, &d) in out_shape.dims().iter().enumerate() {
        info[3 + i] = d as i32;
    }
    for (i, &s) in out_tensor.strides().dims().iter().enumerate() {
        info[11 + i] = s as i32;
    }
    for (i, &d) in lhs.shape().dims().iter().enumerate() {
        info[19 + i] = d as i32;
    }
    for (i, &s) in lhs.strides().dims().iter().enumerate() {
        info[27 + i] = s as i32;
    }
    for (i, &d) in rhs.shape().dims().iter().enumerate() {
        info[35 + i] = d as i32;
    }
    for (i, &s) in rhs.strides().dims().iter().enumerate() {
        info[43 + i] = s as i32;
    }

    let info_dev = dev.htod_copy(info).unwrap();

    let func = dev.get_func("vearo_kernels", kernel_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(out_numel as u32);
    unsafe {
        func.launch(
            cfg,
            (lhs_slice, rhs_slice, out_slice, &info_dev, out_numel as i32),
        )
        .unwrap();
    }
    out_tensor
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    binary_op(lhs, rhs, "add_broadcast_kernel")
}
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    binary_op(lhs, rhs, "sub_broadcast_kernel")
}
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    binary_op(lhs, rhs, "mul_broadcast_kernel")
}
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    binary_op(lhs, rhs, "div_broadcast_kernel")
}

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let dev = get_cuda_device();
    // The kernel assumes contiguous row-major inputs; transposed tensors (from
    // backward, and from Linear's weight transpose) must be materialized first.
    let lhs = lhs.contiguous();
    let rhs = rhs.contiguous();
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
    let batch_size = out_batch_shape.numel();

    let mut out_dims = out_batch_shape.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::new(out_dims);

    let out_storage = cuda_alloc(out_shape.numel());
    let out_tensor = Tensor::from_components(
        out_storage,
        out_shape,
        out_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if out_shape.numel() == 0 {
        return out_tensor;
    }

    let slots = CUDA_SLOTS.lock().unwrap();
    let lhs_slice = &slots[lhs.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let rhs_slice = &slots[rhs.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let lhs_batch_stride = if batch_shape_l.numel() > 1 {
        m * k_l
    } else {
        0
    };
    let rhs_batch_stride = if batch_shape_r.numel() > 1 {
        k_l * n
    } else {
        0
    };
    let out_batch_stride = m * n;

    let block_dim = (16, 16, 1);
    let grid_dim = (
        ((n + 15) / 16) as u32,
        ((m + 15) / 16) as u32,
        batch_size as u32,
    );
    let cfg = LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    };

    let func = dev.get_func("vearo_kernels", "matmul_kernel").unwrap();
    unsafe {
        func.launch(
            cfg,
            (
                lhs_slice,
                rhs_slice,
                out_slice,
                m as i32,
                k_l as i32,
                n as i32,
                batch_size as i32,
                lhs_batch_stride as i32,
                rhs_batch_stride as i32,
                out_batch_stride as i32,
            ),
        )
        .unwrap();
    }
    out_tensor
}

fn reduction_op(x: &Tensor, dim: usize, keep_dim: bool, kernel_name: &str) -> Tensor {
    let dev = get_cuda_device();
    let x_shape = x.shape();
    let rank = x_shape.rank();
    assert!(dim < rank, "Reduction dim out of bounds");

    let mut out_dims = x_shape.dims().to_vec();
    let reduce_size = out_dims[dim];
    if keep_dim {
        out_dims[dim] = 1;
    } else {
        out_dims.remove(dim);
    }
    let out_shape = Shape::new(out_dims);

    let out_storage = cuda_alloc(out_shape.numel());
    let out_tensor = Tensor::from_components(
        out_storage,
        out_shape,
        out_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    let numel_out = out_shape.numel();
    if numel_out == 0 {
        return out_tensor;
    }

    let mut match_dims = x_shape.dims().to_vec();
    match_dims[dim] = 1;
    let match_shape = Shape::new(match_dims);

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let mut info = vec![0i32; 35];
    info[0] = match_shape.rank() as i32;
    info[1] = x_shape.rank() as i32;
    info[2] = dim as i32;

    for (i, &d) in match_shape.dims().iter().enumerate() {
        info[3 + i] = d as i32;
    }
    for (i, &s) in match_shape.contiguous_strides().dims().iter().enumerate() {
        info[11 + i] = s as i32;
    }
    for (i, &d) in x_shape.dims().iter().enumerate() {
        info[19 + i] = d as i32;
    }
    for (i, &s) in x.strides().dims().iter().enumerate() {
        info[27 + i] = s as i32;
    }

    let info_dev = dev.htod_copy(info).unwrap();

    let func = dev.get_func("vearo_kernels", kernel_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel_out as u32);

    unsafe {
        func.launch(
            cfg,
            (
                x_slice,
                out_slice,
                &info_dev,
                reduce_size as i32,
                numel_out as i32,
            ),
        )
        .unwrap();
    }

    out_tensor
}

pub fn sum(x: &Tensor, dim: usize, keep_dim: bool) -> Tensor {
    reduction_op(x, dim, keep_dim, "sum_kernel")
}

pub fn mean(x: &Tensor, dim: usize, keep_dim: bool) -> Tensor {
    reduction_op(x, dim, keep_dim, "mean_kernel")
}

pub fn relu(x: &Tensor) -> Tensor {
    unary_op(x, "relu_forward")
}

pub fn gelu(x: &Tensor) -> Tensor {
    unary_op(x, "gelu_forward")
}

fn unary_op(x: &Tensor, kernel_name: &str) -> Tensor {
    let dev = get_cuda_device();
    let numel = x.shape().numel();
    let out_storage = cuda_alloc(numel);
    let out_tensor = Tensor::from_components(
        out_storage,
        *x.shape(),
        x.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if numel == 0 {
        return out_tensor;
    }

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", kernel_name).unwrap();
    let cfg = LaunchConfig::for_num_elems(numel as u32);
    unsafe {
        func.launch(cfg, (x_slice, out_slice, numel as i32))
            .unwrap();
    }
    out_tensor
}

pub fn softmax(x: &Tensor, dim: usize) -> Tensor {
    let dev = get_cuda_device();
    let x_shape = x.shape();
    let rank = x_shape.rank();
    assert!(dim < rank, "Softmax dim out of bounds");

    let out_storage = cuda_alloc(x_shape.numel());
    let out_tensor = Tensor::from_components(
        out_storage,
        *x_shape,
        x_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if x_shape.numel() == 0 {
        return out_tensor;
    }

    let reduce_size = x_shape[dim];
    let outer_numel = x_shape.numel() / reduce_size;

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let mut info = vec![0i32; 18];
    info[0] = rank as i32;
    for (i, &d) in x_shape.dims().iter().enumerate() {
        info[2 + i] = d as i32;
    }
    for (i, &s) in x.strides().dims().iter().enumerate() {
        info[10 + i] = s as i32;
    }

    let info_dev = dev.htod_copy(info).unwrap();

    let func = dev.get_func("vearo_kernels", "softmax_forward").unwrap();
    let cfg = LaunchConfig::for_num_elems(outer_numel as u32);

    unsafe {
        func.launch(
            cfg,
            (
                x_slice,
                out_slice,
                &info_dev,
                dim as i32,
                reduce_size as i32,
                outer_numel as i32,
            ),
        )
        .unwrap();
    }
    out_tensor
}

pub fn layernorm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let dev = get_cuda_device();
    let x_shape = x.shape();
    let out_storage = cuda_alloc(x_shape.numel());
    let out_tensor = Tensor::from_components(
        out_storage,
        *x_shape,
        x_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if x_shape.numel() == 0 {
        return out_tensor;
    }

    let rank = x_shape.rank();
    let norm_dim = x_shape[rank - 1];
    let outer_numel = x_shape.numel() / norm_dim;

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let w_slice = &slots[weight.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let b_slice = &slots[bias.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "layernorm_forward").unwrap();
    let cfg = LaunchConfig::for_num_elems(outer_numel as u32);

    unsafe {
        func.launch(
            cfg,
            (
                x_slice,
                w_slice,
                b_slice,
                out_slice,
                norm_dim as i32,
                eps,
                outer_numel as i32,
            ),
        )
        .unwrap();
    }
    out_tensor
}

pub fn layernorm_backward(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    grad_out: &Tensor,
    eps: f32,
) -> (Tensor, Tensor, Tensor) {
    let dev = get_cuda_device();
    let x_shape = x.shape();

    let gx_storage = cuda_alloc(x_shape.numel());
    let gw_storage = cuda_alloc(weight.shape().numel());
    let gb_storage = cuda_alloc(bias.shape().numel());

    let grad_x = Tensor::from_components(
        gx_storage,
        *x_shape,
        x_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );
    let grad_w = Tensor::from_components(
        gw_storage,
        *weight.shape(),
        weight.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );
    let grad_b = Tensor::from_components(
        gb_storage,
        *bias.shape(),
        bias.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if x_shape.numel() == 0 {
        return (grad_x, grad_w, grad_b);
    }

    let rank = x_shape.rank();
    let norm_dim = x_shape[rank - 1];
    let outer_numel = x_shape.numel() / norm_dim;

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let w_slice = &slots[weight.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let b_slice = &slots[bias.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let go_slice = &slots[grad_out.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let gx_slice = &slots[gx_storage.slot_idx as usize].as_ref().unwrap().slice;
    let gw_slice = &slots[gw_storage.slot_idx as usize].as_ref().unwrap().slice;
    let gb_slice = &slots[gb_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "layernorm_backward").unwrap();
    let cfg = LaunchConfig::for_num_elems(outer_numel as u32);

    unsafe {
        func.launch(
            cfg,
            (
                x_slice,
                w_slice,
                b_slice,
                go_slice,
                gx_slice,
                gw_slice,
                gb_slice,
                norm_dim as i32,
                eps,
                outer_numel as i32,
            ),
        )
        .unwrap();
    }
    (grad_x, grad_w, grad_b)
}

pub fn embedding(x: &Tensor, weight: &Tensor) -> Tensor {
    let dev = get_cuda_device();
    let x_shape = x.shape();

    let vocab_size = weight.shape()[0];
    let embedding_dim = weight.shape()[1];

    let mut out_dims = x_shape.dims().to_vec();
    out_dims.push(embedding_dim);
    let out_shape = Shape::new(out_dims);

    let out_storage = cuda_alloc(out_shape.numel());
    let out_tensor = Tensor::from_components(
        out_storage,
        out_shape,
        out_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if x_shape.numel() == 0 {
        return out_tensor;
    }

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let w_slice = &slots[weight.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "embedding_forward").unwrap();
    let cfg = LaunchConfig::for_num_elems(x_shape.numel() as u32);

    unsafe {
        func.launch(
            cfg,
            (
                x_slice,
                w_slice,
                out_slice,
                vocab_size as i32,
                embedding_dim as i32,
                x_shape.numel() as i32,
            ),
        )
        .unwrap();
    }
    out_tensor
}

pub fn embedding_backward(x: &Tensor, weight: &Tensor, grad_out: &Tensor) -> Tensor {
    let dev = get_cuda_device();
    let x_shape = x.shape();

    let vocab_size = weight.shape()[0];
    let embedding_dim = weight.shape()[1];

    let gw_storage = cuda_alloc(weight.shape().numel());
    let grad_w = Tensor::from_components(
        gw_storage,
        *weight.shape(),
        weight.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if x_shape.numel() == 0 {
        return grad_w;
    }

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let go_slice = &slots[grad_out.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let gw_slice = &slots[gw_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "embedding_backward").unwrap();
    let cfg = LaunchConfig::for_num_elems(x_shape.numel() as u32);

    unsafe {
        func.launch(
            cfg,
            (
                x_slice,
                go_slice,
                gw_slice,
                vocab_size as i32,
                embedding_dim as i32,
                x_shape.numel() as i32,
            ),
        )
        .unwrap();
    }
    grad_w
}

pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    let dev = get_cuda_device();
    let batch_size = logits.shape()[0];
    let vocab_size = logits.shape()[1];

    if batch_size == 0 {
        let out_storage = cuda_alloc(1);
        return Tensor::from_components(
            out_storage,
            Shape::new(vec![1]),
            Shape::new(vec![1]),
            DType::F32,
            Device::Cuda(0),
        );
    }

    let temp_storage = cuda_alloc(batch_size);
    let temp_tensor = Tensor::from_components(
        temp_storage,
        Shape::new(vec![batch_size]),
        Shape::new(vec![1]),
        DType::F32,
        Device::Cuda(0),
    );

    let slots = CUDA_SLOTS.lock().unwrap();
    let logits_slice = &slots[logits.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let targets_slice = &slots[targets.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let temp_slice = &slots[temp_storage.slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;

    let func = dev
        .get_func("vearo_kernels", "cross_entropy_forward")
        .unwrap();
    let cfg = LaunchConfig::for_num_elems(batch_size as u32);

    unsafe {
        func.launch(
            cfg,
            (
                logits_slice,
                targets_slice,
                temp_slice,
                batch_size as i32,
                vocab_size as i32,
            ),
        )
        .unwrap();
    }

    let was_enabled = vearo_core::is_autograd_enabled();
    vearo_core::set_autograd_enabled(false);
    let out = mean(&temp_tensor, 0, false);
    vearo_core::set_autograd_enabled(was_enabled);
    out
}

pub fn cross_entropy_backward(logits: &Tensor, targets: &Tensor, grad_out: &Tensor) -> Tensor {
    let dev = get_cuda_device();
    let batch_size = logits.shape()[0];
    let vocab_size = logits.shape()[1];

    let gl_storage = cuda_alloc(logits.shape().numel());
    let grad_l = Tensor::from_components(
        gl_storage,
        *logits.shape(),
        logits.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if batch_size == 0 {
        return grad_l;
    }

    let slots = CUDA_SLOTS.lock().unwrap();
    let logits_slice = &slots[logits.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let targets_slice = &slots[targets.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let go_slice = &slots[grad_out.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let gl_slice = &slots[gl_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev
        .get_func("vearo_kernels", "cross_entropy_backward")
        .unwrap();
    let cfg = LaunchConfig::for_num_elems(batch_size as u32);

    unsafe {
        func.launch(
            cfg,
            (
                logits_slice,
                targets_slice,
                go_slice,
                gl_slice,
                batch_size as i32,
                vocab_size as i32,
            ),
        )
        .unwrap();
    }
    grad_l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_parity() {
        vearo_backend_cpu::init();
        vearo_core::register_refcount_inc(cuda_refcount_inc);
        vearo_core::register_refcount_dec(cuda_refcount_dec);
        vearo_core::register_cuda_hooks(cuda_read, cuda_write, cuda_alloc);
        init();

        // 1. Test elementwise add
        let a_cpu = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b_cpu = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], [2, 2]);
        let c_cpu = a_cpu.add(&b_cpu);

        let a_gpu = a_cpu.to(Device::Cuda(0));
        let b_gpu = b_cpu.to(Device::Cuda(0));
        let c_gpu = a_gpu.add(&b_gpu);

        let c_gpu_host = c_gpu.to(Device::Cpu);
        assert_eq!(c_cpu.to_vec_f32(), c_gpu_host.to_vec_f32());

        // 2. Test matmul
        let a_cpu = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b_cpu = Tensor::from_f32(&[2.0, 0.0, 1.0, 3.0], [2, 2]);
        let c_cpu = a_cpu.matmul(&b_cpu);

        let a_gpu = a_cpu.to(Device::Cuda(0));
        let b_gpu = b_cpu.to(Device::Cuda(0));
        let c_gpu = a_gpu.matmul(&b_gpu);

        let c_gpu_host = c_gpu.to(Device::Cpu);
        assert_eq!(c_cpu.to_vec_f32(), c_gpu_host.to_vec_f32());
    }

    #[test]
    fn test_cuda_matmul_transposed() {
        vearo_backend_cpu::init();
        vearo_core::register_refcount_inc(cuda_refcount_inc);
        vearo_core::register_refcount_dec(cuda_refcount_dec);
        vearo_core::register_cuda_hooks(cuda_read, cuda_write, cuda_alloc);
        init();

        // matmul with a transposed (non-contiguous) rhs - exactly what backward feeds it.
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let b = Tensor::from_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [2, 3]);
        let bt_cpu = b.transpose(0, 1); // [3, 2], non-contiguous
        let c_cpu = a.matmul(&bt_cpu); // [2,3] @ [3,2] = [2,2]

        let a_g = a.to(Device::Cuda(0));
        let b_g = b.to(Device::Cuda(0));
        let bt_g = b_g.transpose(0, 1);
        let c_g = a_g.matmul(&bt_g).to(Device::Cpu);

        assert_eq!(
            c_cpu.to_vec_f32(),
            c_g.to_vec_f32(),
            "CUDA matmul must handle transposed inputs"
        );
    }
}
