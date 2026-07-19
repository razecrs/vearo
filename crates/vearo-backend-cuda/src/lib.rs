//! CUDA backend implementations for Vearo.
#![allow(
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
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
pub static FREE_CUDA_SLOTS: LazyLock<Mutex<Vec<(usize, u32)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
pub static PEAK_CUDA_MEM_BYTES: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

pub fn get_peak_memory() -> usize {
    *PEAK_CUDA_MEM_BYTES.lock().unwrap()
}

pub fn cuda_alloc(numel: usize) -> StorageId {
    let dev = get_cuda_device();

    let mut slots = CUDA_SLOTS.lock().unwrap();
    let mut free = FREE_CUDA_SLOTS.lock().unwrap();

    let mut found_idx = None;
    for (i, &(size, slot_idx)) in free.iter().enumerate() {
        if size == numel {
            found_idx = Some((i, slot_idx));
            break;
        }
    }

    if let Some((free_list_idx, slot_idx)) = found_idx {
        free.swap_remove(free_list_idx);
        let slot = slots[slot_idx as usize].as_mut().unwrap();
        slot.ref_count = 1;

        // Zero out the reused slice asynchronously on the current stream.
        dev.memset_zeros(&mut slot.slice)
            .expect("Failed to zero reused CUDA memory");

        StorageId {
            shard_idx: 0,
            slot_idx,
        }
    } else {
        let slice = dev
            .alloc_zeros::<f32>(numel)
            .expect("Failed to allocate CUDA memory");

        let mut current_total = 0;
        for s in slots.iter().flatten() {
            current_total += s.slice.len() * 4;
        }
        current_total += numel * 4;

        let mut peak = PEAK_CUDA_MEM_BYTES.lock().unwrap();
        if current_total > *peak {
            *peak = current_total;
        }

        let slot = CudaSlot {
            slice,
            ref_count: 1,
        };

        let mut empty_idx = None;
        for (idx, s) in slots.iter().enumerate() {
            if s.is_none() {
                empty_idx = Some(idx as u32);
                break;
            }
        }

        if let Some(idx) = empty_idx {
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
        let mut size = 0;
        if let Some(ref mut slot) = slots[storage_id.slot_idx as usize] {
            assert!(slot.ref_count > 0, "Reference count underflow");
            slot.ref_count -= 1;
            if slot.ref_count == 0 {
                free = true;
                size = slot.slice.len();
            }
        }
        if free {
            FREE_CUDA_SLOTS.lock().unwrap().push((size, storage_id.slot_idx));
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
                "conv2d_forward",
                "conv2d_backward_bias",
                "conv2d_backward_weight",
                "conv2d_backward_input",
                "maxpool2d_forward",
                "maxpool2d_backward",
                "avgpool2d_forward",
                "avgpool2d_backward",
                "batchnorm_forward",
                "batchnorm_backward",
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

/// 2D convolution (NCHW input, OIHW weight) on CUDA. Bit-matches the CPU backend.
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> Tensor {
    let dev = get_cuda_device();
    let input = input.contiguous();
    let weight = weight.contiguous();
    let bias = bias.contiguous();
    let id = input.shape().dims();
    let wd = weight.shape().dims();
    let (n, cin, h, w) = (id[0], id[1], id[2], id[3]);
    let (cout, kh, kw) = (wd[0], wd[2], wd[3]);

    let oh = (h + 2 * padding - kh) / stride + 1;
    let ow = (w + 2 * padding - kw) / stride + 1;
    let out_shape = Shape::new([n, cout, oh, ow]);
    let out_storage = cuda_alloc(out_shape.numel());
    let out = Tensor::from_components(
        out_storage,
        out_shape,
        out_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );
    if out_shape.numel() == 0 {
        return out;
    }

    let p = conv_params(n, cin, h, w, cout, kh, kw, oh, ow, stride, padding);
    let p_dev = dev.htod_copy(p).unwrap();

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[input.storage_id().slot_idx as usize]
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

    let func = dev.get_func("vearo_kernels", "conv2d_forward").unwrap();
    let cfg = LaunchConfig::for_num_elems(out_shape.numel() as u32);
    unsafe {
        func.launch(cfg, (x_slice, w_slice, b_slice, out_slice, &p_dev))
            .unwrap();
    }
    out
}

/// Pack conv shape parameters into the layout the kernels expect.
fn conv_params(
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    stride: usize,
    padding: usize,
) -> Vec<i32> {
    vec![
        n as i32,
        cin as i32,
        h as i32,
        w as i32,
        cout as i32,
        kh as i32,
        kw as i32,
        oh as i32,
        ow as i32,
        stride as i32,
        padding as i32,
    ]
}

/// Backward for [`conv2d`] on CUDA: returns `(grad_input, grad_weight, grad_bias)`.
/// Bit-matches the CPU backend (gather kernels, no atomics -> deterministic order).
pub fn conv2d_backward(
    input: &Tensor,
    weight: &Tensor,
    grad_out: &Tensor,
    stride: usize,
    padding: usize,
) -> (Tensor, Tensor, Tensor) {
    let dev = get_cuda_device();
    let input = input.contiguous();
    let weight = weight.contiguous();
    let grad_out = grad_out.contiguous();
    let id = input.shape().dims();
    let wd = weight.shape().dims();
    let (n, cin, h, w) = (id[0], id[1], id[2], id[3]);
    let (cout, kh, kw) = (wd[0], wd[2], wd[3]);
    let gd = grad_out.shape().dims();
    let (oh, ow) = (gd[2], gd[3]);

    let gi_storage = cuda_alloc(input.shape().numel());
    let gw_storage = cuda_alloc(weight.shape().numel());
    let gb_storage = cuda_alloc(cout);
    let grad_in = Tensor::from_components(
        gi_storage,
        *input.shape(),
        input.shape().contiguous_strides(),
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
        Shape::new([cout]),
        Shape::new([cout]).contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    if input.shape().numel() == 0 {
        return (grad_in, grad_w, grad_b);
    }

    let p = conv_params(n, cin, h, w, cout, kh, kw, oh, ow, stride, padding);
    let p_dev = dev.htod_copy(p).unwrap();

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[input.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let w_slice = &slots[weight.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let go_slice = &slots[grad_out.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let gi_slice = &slots[gi_storage.slot_idx as usize].as_ref().unwrap().slice;
    let gw_slice = &slots[gw_storage.slot_idx as usize].as_ref().unwrap().slice;
    let gb_slice = &slots[gb_storage.slot_idx as usize].as_ref().unwrap().slice;

    let f_bias = dev
        .get_func("vearo_kernels", "conv2d_backward_bias")
        .unwrap();
    unsafe {
        f_bias
            .launch(
                LaunchConfig::for_num_elems(cout as u32),
                (go_slice, gb_slice, &p_dev),
            )
            .unwrap();
    }

    let f_weight = dev
        .get_func("vearo_kernels", "conv2d_backward_weight")
        .unwrap();
    unsafe {
        f_weight
            .launch(
                LaunchConfig::for_num_elems(weight.shape().numel() as u32),
                (x_slice, go_slice, gw_slice, &p_dev),
            )
            .unwrap();
    }

    let f_input = dev
        .get_func("vearo_kernels", "conv2d_backward_input")
        .unwrap();
    unsafe {
        f_input
            .launch(
                LaunchConfig::for_num_elems(input.shape().numel() as u32),
                (w_slice, go_slice, gi_slice, &p_dev),
            )
            .unwrap();
    }

    (grad_in, grad_w, grad_b)
}

/// Pack max-pool shape parameters into the layout the kernels expect.
fn maxpool_params(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    oh: usize,
    ow: usize,
    stride: usize,
    padding: usize,
) -> Vec<i32> {
    vec![
        n as i32,
        c as i32,
        h as i32,
        w as i32,
        k as i32,
        oh as i32,
        ow as i32,
        stride as i32,
        padding as i32,
    ]
}

/// 2D max pooling (NCHW) on CUDA. Bit-matches the CPU backend.
pub fn maxpool2d(input: &Tensor, kernel_size: usize, stride: usize, padding: usize) -> Tensor {
    let dev = get_cuda_device();
    let input = input.contiguous();
    let id = input.shape().dims();
    let (n, c, h, w) = (id[0], id[1], id[2], id[3]);
    let oh = (h + 2 * padding - kernel_size) / stride + 1;
    let ow = (w + 2 * padding - kernel_size) / stride + 1;
    let out_shape = Shape::new([n, c, oh, ow]);
    let out_storage = cuda_alloc(out_shape.numel());
    let out = Tensor::from_components(
        out_storage,
        out_shape,
        out_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );
    if out_shape.numel() == 0 {
        return out;
    }

    let p = maxpool_params(n, c, h, w, kernel_size, oh, ow, stride, padding);
    let p_dev = dev.htod_copy(p).unwrap();

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[input.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "maxpool2d_forward").unwrap();
    let cfg = LaunchConfig::for_num_elems(out_shape.numel() as u32);
    unsafe {
        func.launch(cfg, (x_slice, out_slice, &p_dev)).unwrap();
    }
    out
}

/// Backward for [`maxpool2d`] on CUDA: returns grad input. Bit-matches the CPU backend.
pub fn maxpool2d_backward(
    input: &Tensor,
    grad_out: &Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor {
    let dev = get_cuda_device();
    let input = input.contiguous();
    let grad_out = grad_out.contiguous();
    let id = input.shape().dims();
    let (n, c, h, w) = (id[0], id[1], id[2], id[3]);
    let gd = grad_out.shape().dims();
    let (oh, ow) = (gd[2], gd[3]);

    let gi_storage = cuda_alloc(input.shape().numel());
    let grad_in = Tensor::from_components(
        gi_storage,
        *input.shape(),
        input.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );
    if input.shape().numel() == 0 {
        return grad_in;
    }

    let p = maxpool_params(n, c, h, w, kernel_size, oh, ow, stride, padding);
    let p_dev = dev.htod_copy(p).unwrap();

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[input.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let go_slice = &slots[grad_out.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let gi_slice = &slots[gi_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "maxpool2d_backward").unwrap();
    let cfg = LaunchConfig::for_num_elems(input.shape().numel() as u32);
    unsafe {
        func.launch(cfg, (x_slice, go_slice, gi_slice, &p_dev))
            .unwrap();
    }
    grad_in
}

/// 2D average pooling (NCHW) on CUDA. Bit-matches the CPU backend.
pub fn avgpool2d(input: &Tensor, kernel_size: usize, stride: usize, padding: usize) -> Tensor {
    let dev = get_cuda_device();
    let input = input.contiguous();
    let id = input.shape().dims();
    let (n, c, h, w) = (id[0], id[1], id[2], id[3]);
    let oh = (h + 2 * padding - kernel_size) / stride + 1;
    let ow = (w + 2 * padding - kernel_size) / stride + 1;
    let out_shape = Shape::new([n, c, oh, ow]);
    let out_storage = cuda_alloc(out_shape.numel());
    let out = Tensor::from_components(
        out_storage,
        out_shape,
        out_shape.contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );
    if out_shape.numel() == 0 {
        return out;
    }

    let p = maxpool_params(n, c, h, w, kernel_size, oh, ow, stride, padding);
    let p_dev = dev.htod_copy(p).unwrap();

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[input.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "avgpool2d_forward").unwrap();
    let cfg = LaunchConfig::for_num_elems(out_shape.numel() as u32);
    unsafe {
        func.launch(cfg, (x_slice, out_slice, &p_dev)).unwrap();
    }
    out
}

/// Backward for [`avgpool2d`] on CUDA: returns grad input. Bit-matches the CPU backend.
pub fn avgpool2d_backward(
    input: &Tensor,
    grad_out: &Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor {
    let dev = get_cuda_device();
    let grad_out = grad_out.contiguous();
    let id = input.shape().dims();
    let (n, c, h, w) = (id[0], id[1], id[2], id[3]);
    let gd = grad_out.shape().dims();
    let (oh, ow) = (gd[2], gd[3]);

    let gi_storage = cuda_alloc(input.shape().numel());
    let grad_in = Tensor::from_components(
        gi_storage,
        *input.shape(),
        input.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );
    if input.shape().numel() == 0 {
        return grad_in;
    }

    let p = maxpool_params(n, c, h, w, kernel_size, oh, ow, stride, padding);
    let p_dev = dev.htod_copy(p).unwrap();

    let slots = CUDA_SLOTS.lock().unwrap();
    let go_slice = &slots[grad_out.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let gi_slice = &slots[gi_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "avgpool2d_backward").unwrap();
    let cfg = LaunchConfig::for_num_elems(input.shape().numel() as u32);
    unsafe {
        func.launch(cfg, (go_slice, gi_slice, &p_dev)).unwrap();
    }
    grad_in
}

/// 2D batch normalization (NCHW) on CUDA. Bit-matches the CPU backend.
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
    let dev = get_cuda_device();
    let x = x.contiguous();
    let dims = x.shape().dims();
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

    let out_storage = cuda_alloc(x.shape().numel());
    let out = Tensor::from_components(
        out_storage,
        *x.shape(),
        x.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );
    if x.shape().numel() == 0 {
        return out;
    }

    let p = vec![
        n as f32,
        c as f32,
        h as f32,
        w as f32,
        if training { 1.0f32 } else { 0.0f32 },
        momentum,
        eps,
    ];
    let p_dev = dev.htod_copy(p).unwrap();

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let gamma_slice = &slots[gamma.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let beta_slice = &slots[beta.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let rm_slice = &slots[running_mean.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let rv_slice = &slots[running_var.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let out_slice = &slots[out_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "batchnorm_forward").unwrap();
    let cfg = LaunchConfig::for_num_elems(c as u32);
    unsafe {
        func.launch(
            cfg,
            (
                x_slice,
                gamma_slice,
                beta_slice,
                rm_slice,
                rv_slice,
                out_slice,
                &p_dev,
            ),
        )
        .unwrap();
    }
    out
}

/// Backward for [`batchnorm`] on CUDA: returns (grad input, grad weight, grad bias). Bit-matches the CPU backend.
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
    let dev = get_cuda_device();
    let x = x.contiguous();
    let grad_out = grad_out.contiguous();
    let dims = x.shape().dims();
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

    let gi_storage = cuda_alloc(x.shape().numel());
    let grad_in = Tensor::from_components(
        gi_storage,
        *x.shape(),
        x.shape().contiguous_strides(),
        DType::F32,
        Device::Cuda(0),
    );

    let gw_storage = cuda_alloc(c);
    let grad_w = Tensor::from_components(
        gw_storage,
        Shape::new([c]),
        Shape::new([1]),
        DType::F32,
        Device::Cuda(0),
    );

    let gb_storage = cuda_alloc(c);
    let grad_b = Tensor::from_components(
        gb_storage,
        Shape::new([c]),
        Shape::new([1]),
        DType::F32,
        Device::Cuda(0),
    );

    if x.shape().numel() == 0 {
        return (grad_in, grad_w, grad_b);
    }

    let p = vec![
        n as f32,
        c as f32,
        h as f32,
        w as f32,
        if training { 1.0f32 } else { 0.0f32 },
        eps,
    ];
    let p_dev = dev.htod_copy(p).unwrap();

    let slots = CUDA_SLOTS.lock().unwrap();
    let x_slice = &slots[x.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let gamma_slice = &slots[gamma.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let beta_slice = &slots[beta.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let rm_slice = &slots[running_mean.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let rv_slice = &slots[running_var.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;
    let go_slice = &slots[grad_out.storage_id().slot_idx as usize]
        .as_ref()
        .unwrap()
        .slice;

    let gi_slice = &slots[gi_storage.slot_idx as usize].as_ref().unwrap().slice;
    let gw_slice = &slots[gw_storage.slot_idx as usize].as_ref().unwrap().slice;
    let gb_slice = &slots[gb_storage.slot_idx as usize].as_ref().unwrap().slice;

    let func = dev.get_func("vearo_kernels", "batchnorm_backward").unwrap();
    let cfg = LaunchConfig::for_num_elems(c as u32);
    unsafe {
        func.launch(
            cfg,
            (
                x_slice,
                gamma_slice,
                beta_slice,
                rm_slice,
                rv_slice,
                go_slice,
                gi_slice,
                gw_slice,
                gb_slice,
                &p_dev,
            ),
        )
        .unwrap();
    }
    (grad_in, grad_w, grad_b)
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
        n.div_ceil(16) as u32,
        m.div_ceil(16) as u32,
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
    // Kernel reads memory in storage order; materialize non-contiguous inputs first.
    let x = x.contiguous();
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
    // Kernel writes out[x_off] using the input's strides; a non-contiguous input
    // would scatter the result into the contiguous output buffer.
    let x = x.contiguous();
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
    // Kernel indexes rows as idx*norm_dim, assuming contiguous storage.
    let x = x.contiguous();
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
    // Kernel assumes contiguous storage for both x and grad_out.
    let x = x.contiguous();
    let grad_out = grad_out.contiguous();
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
    // Kernel reads index x[idx] in storage order.
    let x = x.contiguous();
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
    // Kernel assumes contiguous storage for the index tensor and grad_out.
    let x = x.contiguous();
    let grad_out = grad_out.contiguous();
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
    // Kernel indexes logits as idx*vocab_size, assuming contiguous storage.
    let logits = logits.contiguous();
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
    // Release the CUDA_SLOTS lock BEFORE mean(), which re-locks it. Holding it across
    // the mean() call self-deadlocks (std Mutex is not reentrant).
    drop(slots);

    let was_enabled = vearo_core::is_autograd_enabled();
    vearo_core::set_autograd_enabled(false);
    let out = mean(&temp_tensor, 0, false);
    vearo_core::set_autograd_enabled(was_enabled);
    out
}

pub fn cross_entropy_backward(logits: &Tensor, targets: &Tensor, grad_out: &Tensor) -> Tensor {
    let dev = get_cuda_device();
    // Kernel assumes contiguous storage for logits and grad_out.
    let logits = logits.contiguous();
    let grad_out = grad_out.contiguous();
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

/// Fused attention forward on CUDA: copies to CPU, runs CPU kernel, copies back.
pub fn fused_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
    let q_cpu = q.to(Device::Cpu);
    let k_cpu = k.to(Device::Cpu);
    let v_cpu = v.to(Device::Cpu);
    let mask_cpu = mask.map(|m| m.to(Device::Cpu));

    let cpu_ops = vearo_core::get_backend_ops(Device::Cpu)
        .expect("CPU backend not initialized");
    let out_cpu = (cpu_ops.fused_attention)(&q_cpu, &k_cpu, &v_cpu, mask_cpu.as_ref());
    out_cpu.to(Device::Cuda(0))
}

/// Fused attention backward on CUDA: copies to CPU, runs CPU kernel, copies back.
pub fn fused_attention_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    grad_out: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let q_cpu = q.to(Device::Cpu);
    let k_cpu = k.to(Device::Cpu);
    let v_cpu = v.to(Device::Cpu);
    let mask_cpu = mask.map(|m| m.to(Device::Cpu));
    let go_cpu = grad_out.to(Device::Cpu);

    let cpu_ops = vearo_core::get_backend_ops(Device::Cpu)
        .expect("CPU backend not initialized");
    let (dq_cpu, dk_cpu, dv_cpu) = (cpu_ops.fused_attention_backward)(
        &q_cpu,
        &k_cpu,
        &v_cpu,
        mask_cpu.as_ref(),
        &go_cpu,
    );

    (
        dq_cpu.to(Device::Cuda(0)),
        dk_cpu.to(Device::Cuda(0)),
        dv_cpu.to(Device::Cuda(0)),
    )
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

    fn setup() {
        vearo_backend_cpu::init();
        vearo_core::register_refcount_inc(cuda_refcount_inc);
        vearo_core::register_refcount_dec(cuda_refcount_dec);
        vearo_core::register_cuda_hooks(cuda_read, cuda_write, cuda_alloc);
        init();
    }

    /// Audit every CUDA op for the "assumes contiguous" bug that once broke matmul.
    /// For each op we run it on a non-contiguous input AND on `input.contiguous()`,
    /// both on the GPU. A stride-correct kernel yields the same output; a kernel that
    /// reads raw device memory in storage order diverges (wrong elements -> huge diff).
    #[test]
    #[allow(clippy::too_many_lines, clippy::float_cmp)]
    fn test_cuda_noncontiguous_parity() {
        setup();
        vearo_core::set_autograd_enabled(false);

        let cuda = Device::Cuda(0);
        let mut fails: Vec<String> = Vec::new();
        let mut check = |name: &str, nc: &Tensor, c: &Tensor| {
            let a = nc.to(Device::Cpu).to_vec_f32();
            let b = c.to(Device::Cpu).to_vec_f32();
            let bad = a.iter().any(|v| !v.is_finite());
            let maxdiff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            if a.len() != b.len() || bad || maxdiff > 1e-3 {
                fails.push(format!("{name} (maxdiff={maxdiff}, nonfinite={bad})"));
            }
        };

        // binary elementwise: transposed rhs (the backward case), and transposed lhs
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]).to(cuda);
        let b = Tensor::from_f32(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [3, 3]).to(cuda);
        let bt = b.transpose(0, 1);
        check("add(rhs_nc)", &a.add(&bt), &a.add(&bt.contiguous()));
        check("sub(rhs_nc)", &a.sub(&bt), &a.sub(&bt.contiguous()));
        check("mul(rhs_nc)", &a.mul(&bt), &a.mul(&bt.contiguous()));
        check("div(rhs_nc)", &a.div(&bt), &a.div(&bt.contiguous()));
        let at = a.transpose(0, 1);
        check("add(lhs_nc)", &at.add(&b), &at.contiguous().add(&b));

        // unary: transposed input
        let x = Tensor::from_f32(
            &[
                -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, -0.5, 4.0, -3.0, 0.0, 1.5, -1.5,
            ],
            [3, 4],
        )
        .to(cuda);
        let xt = x.transpose(0, 1); // [4,3] non-contiguous
        check("relu(nc)", &xt.relu(), &xt.contiguous().relu());
        check("gelu(nc)", &xt.gelu(), &xt.contiguous().gelu());

        // reductions over each axis
        check(
            "sum(nc,d0)",
            &xt.sum(0, false),
            &xt.contiguous().sum(0, false),
        );
        check(
            "sum(nc,d1)",
            &xt.sum(1, true),
            &xt.contiguous().sum(1, true),
        );
        check(
            "mean(nc,d0)",
            &xt.mean(0, false),
            &xt.contiguous().mean(0, false),
        );

        // softmax over each axis
        check(
            "softmax(nc,d0)",
            &xt.softmax(0),
            &xt.contiguous().softmax(0),
        );
        check(
            "softmax(nc,d1)",
            &xt.softmax(1),
            &xt.contiguous().softmax(1),
        );

        // layernorm: permuted so it stays non-contiguous but the normalized last dim is intact
        let ln = Tensor::from_f32(
            &(0..24).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            [2, 3, 4],
        )
        .to(cuda);
        let lnp = ln.permute([1, 0, 2]); // [3,2,4] non-contiguous, last dim = 4
        let w = Tensor::from_f32(&[1.0, 0.5, 2.0, 1.5], [4]).to(cuda);
        let bb = Tensor::from_f32(&[0.1, -0.1, 0.2, 0.0], [4]).to(cuda);
        check(
            "layernorm(nc)",
            &lnp.layernorm(&w, &bb, 1e-5),
            &lnp.contiguous().layernorm(&w, &bb, 1e-5),
        );

        // embedding: non-contiguous index tensor
        let idx = Tensor::from_f32(&[0.0, 2.0, 1.0, 1.0, 0.0, 2.0], [2, 3]).to(cuda);
        let idxt = idx.transpose(0, 1); // [3,2] non-contiguous
        let emb_w = Tensor::from_f32(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], [3, 2]).to(cuda);
        check(
            "embedding(nc)",
            &idxt.embedding(&emb_w),
            &idxt.contiguous().embedding(&emb_w),
        );

        // cross_entropy: non-contiguous logits [batch, vocab]
        let logits_base = Tensor::from_f32(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], [3, 2]).to(cuda);
        let logits_nc = logits_base.transpose(0, 1); // [2,3] non-contiguous
        let tgt = Tensor::from_f32(&[0.0, 2.0], [2]).to(cuda);
        check(
            "cross_entropy(nc)",
            &logits_nc.cross_entropy(&tgt),
            &logits_nc.contiguous().cross_entropy(&tgt),
        );

        assert!(
            fails.is_empty(),
            "CUDA ops that mishandle non-contiguous inputs: {fails:?}"
        );
    }

    /// conv2d forward + all three gradients must match the CPU backend.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_cuda_conv2d_parity() {
        setup();
        vearo_core::set_autograd_enabled(false);
        let cuda = Device::Cuda(0);

        fn assert_close(name: &str, cpu: &Tensor, gpu: &Tensor) {
            let a = cpu.to_vec_f32();
            let b = gpu.to(Device::Cpu).to_vec_f32();
            assert_eq!(a.len(), b.len(), "{name}: length mismatch");
            let maxdiff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(maxdiff < 1e-4, "{name}: max abs diff {maxdiff}");
        }

        let mk = |numel: usize, seed: f32| -> Vec<f32> {
            (0..numel).map(|i| (i as f32 * seed + seed).sin()).collect()
        };

        for &(stride, padding) in &[(1usize, 1usize), (2usize, 1usize)] {
            let (n, cin, h, w) = (2usize, 3usize, 5usize, 5usize);
            let (cout, kh, kw) = (4usize, 3usize, 3usize);
            let oh = (h + 2 * padding - kh) / stride + 1;
            let ow = (w + 2 * padding - kw) / stride + 1;

            let x = Tensor::from_f32(&mk(n * cin * h * w, 0.3), [n, cin, h, w]);
            let wt = Tensor::from_f32(&mk(cout * cin * kh * kw, 0.7), [cout, cin, kh, kw]);
            let b = Tensor::from_f32(&mk(cout, 1.1), [cout]);
            let go = Tensor::from_f32(&mk(n * cout * oh * ow, 0.5), [n, cout, oh, ow]);

            let (xg, wg, bg, gog) = (x.to(cuda), wt.to(cuda), b.to(cuda), go.to(cuda));

            // forward
            assert_close(
                &format!("fwd s{stride}p{padding}"),
                &x.conv2d(&wt, &b, stride, padding),
                &xg.conv2d(&wg, &bg, stride, padding),
            );

            // backward: grad_input, grad_weight, grad_bias
            let (gi_c, gw_c, gb_c) = x.conv2d_backward(&wt, &go, stride, padding);
            let (gi_g, gw_g, gb_g) = xg.conv2d_backward(&wg, &gog, stride, padding);
            assert_close(&format!("grad_input s{stride}p{padding}"), &gi_c, &gi_g);
            assert_close(&format!("grad_weight s{stride}p{padding}"), &gw_c, &gw_g);
            assert_close(&format!("grad_bias s{stride}p{padding}"), &gb_c, &gb_g);
        }
    }

    /// maxpool2d forward + backward must match the CPU backend.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_cuda_maxpool2d_parity() {
        setup();
        vearo_core::set_autograd_enabled(false);
        let cuda = Device::Cuda(0);

        let mk = |numel: usize, seed: f32| -> Vec<f32> {
            (0..numel).map(|i| (i as f32 * seed + seed).sin()).collect()
        };
        let maxdiff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        for &(k, stride, padding) in &[(2usize, 2usize, 0usize), (3usize, 2usize, 1usize)] {
            let (n, c, h, w) = (2usize, 3usize, 6usize, 6usize);
            let oh = (h + 2 * padding - k) / stride + 1;
            let ow = (w + 2 * padding - k) / stride + 1;

            let x = Tensor::from_f32(&mk(n * c * h * w, 0.3), [n, c, h, w]);
            let go = Tensor::from_f32(&mk(n * c * oh * ow, 0.5), [n, c, oh, ow]);
            let xg = x.to(cuda);
            let gog = go.to(cuda);

            let fwd_c = x.maxpool2d(k, stride, padding).to_vec_f32();
            let fwd_g = xg
                .maxpool2d(k, stride, padding)
                .to(Device::Cpu)
                .to_vec_f32();
            assert!(
                maxdiff(&fwd_c, &fwd_g) < 1e-4,
                "maxpool fwd k{k}s{stride}p{padding}"
            );

            let gi_c = x.maxpool2d_backward(&go, k, stride, padding).to_vec_f32();
            let gi_g = xg
                .maxpool2d_backward(&gog, k, stride, padding)
                .to(Device::Cpu)
                .to_vec_f32();
            assert!(
                maxdiff(&gi_c, &gi_g) < 1e-4,
                "maxpool bwd k{k}s{stride}p{padding}"
            );
        }
    }

    /// avgpool2d forward + backward must match the CPU backend.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_cuda_avgpool2d_parity() {
        setup();
        vearo_core::set_autograd_enabled(false);
        let cuda = Device::Cuda(0);

        let mk = |numel: usize, seed: f32| -> Vec<f32> {
            (0..numel).map(|i| (i as f32 * seed + seed).sin()).collect()
        };
        let maxdiff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        for &(k, stride, padding) in &[(2usize, 2usize, 0usize), (3usize, 2usize, 1usize)] {
            let (n, c, h, w) = (2usize, 3usize, 6usize, 6usize);
            let oh = (h + 2 * padding - k) / stride + 1;
            let ow = (w + 2 * padding - k) / stride + 1;

            let x = Tensor::from_f32(&mk(n * c * h * w, 0.3), [n, c, h, w]);
            let go = Tensor::from_f32(&mk(n * c * oh * ow, 0.5), [n, c, oh, ow]);
            let xg = x.to(cuda);
            let gog = go.to(cuda);

            let fwd_c = x.avgpool2d(k, stride, padding).to_vec_f32();
            let fwd_g = xg
                .avgpool2d(k, stride, padding)
                .to(Device::Cpu)
                .to_vec_f32();
            assert!(
                maxdiff(&fwd_c, &fwd_g) < 1e-4,
                "avgpool fwd k{k}s{stride}p{padding}"
            );

            let gi_c = x.avgpool2d_backward(&go, k, stride, padding).to_vec_f32();
            let gi_g = xg
                .avgpool2d_backward(&gog, k, stride, padding)
                .to(Device::Cpu)
                .to_vec_f32();
            assert!(
                maxdiff(&gi_c, &gi_g) < 1e-4,
                "avgpool bwd k{k}s{stride}p{padding}"
            );
        }
    }
}
