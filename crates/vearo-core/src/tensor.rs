//! Tensor struct and metadata views.

use crate::device::Device;
use crate::dtype::DType;
use crate::shape::Shape;
use crate::storage::{CpuStorage, StorageId, current_thread_shard_idx, get_cpu_shard};

/// Tensor representation with metadata and storage index.
#[derive(Debug)]
pub struct Tensor {
    storage_id: StorageId,
    shape: Shape,
    strides: Shape,
    dtype: DType,
    device: Device,
    node_id: std::cell::Cell<Option<u64>>,
    requires_grad: std::cell::Cell<bool>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        if self.device.is_cpu() {
            get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .increment(self.storage_id.slot_idx);
        }
        Self {
            storage_id: self.storage_id,
            shape: self.shape,
            strides: self.strides,
            dtype: self.dtype,
            device: self.device,
            node_id: std::cell::Cell::new(self.node_id.get()),
            requires_grad: std::cell::Cell::new(self.requires_grad.get()),
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.device.is_cpu() {
            get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .decrement(self.storage_id.slot_idx);
        }
    }
}

impl Tensor {
    /// Creates a zero filled CPU tensor.
    ///
    /// # Panics
    /// Panics if the CPU Arena lock is poisoned.
    #[must_use]
    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        let strides = shape.contiguous_strides();
        let shard_idx = current_thread_shard_idx();
        let storage_id = get_cpu_shard(shard_idx as usize)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .alloc(numel, dtype, shard_idx);
        Self {
            storage_id,
            shape,
            strides,
            dtype,
            device: Device::Cpu,
            node_id: std::cell::Cell::new(None),
            requires_grad: std::cell::Cell::new(false),
        }
    }

    /// Creates a CPU tensor from an f32 slice.
    ///
    /// # Panics
    /// Panics if data len does not match shape numel, or if the CPU Arena lock is poisoned.
    #[must_use]
    pub fn from_f32(data: &[f32], shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "Data len must match shape numel");
        let strides = shape.contiguous_strides();
        let shard_idx = current_thread_shard_idx();
        let storage_id = get_cpu_shard(shard_idx as usize)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .alloc_raw(CpuStorage::F32(data.to_vec()), shard_idx);
        Self {
            storage_id,
            shape,
            strides,
            dtype: DType::F32,
            device: Device::Cpu,
            node_id: std::cell::Cell::new(None),
            requires_grad: std::cell::Cell::new(false),
        }
    }

    /// Creates a CPU tensor from an i32 slice.
    ///
    /// # Panics
    /// Panics if data len does not match shape numel, or if the CPU Arena lock is poisoned.
    #[must_use]
    pub fn from_i32(data: &[i32], shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "Data len must match shape numel");
        let strides = shape.contiguous_strides();
        let shard_idx = current_thread_shard_idx();
        let storage_id = get_cpu_shard(shard_idx as usize)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .alloc_raw(CpuStorage::I32(data.to_vec()), shard_idx);
        Self {
            storage_id,
            shape,
            strides,
            dtype: DType::I32,
            device: Device::Cpu,
            node_id: std::cell::Cell::new(None),
            requires_grad: std::cell::Cell::new(false),
        }
    }

    /// Creates a CPU tensor from an i64 slice.
    ///
    /// # Panics
    /// Panics if data len does not match shape numel, or if the CPU Arena lock is poisoned.
    #[must_use]
    pub fn from_i64(data: &[i64], shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "Data len must match shape numel");
        let strides = shape.contiguous_strides();
        let shard_idx = current_thread_shard_idx();
        let storage_id = get_cpu_shard(shard_idx as usize)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .alloc_raw(CpuStorage::I64(data.to_vec()), shard_idx);
        Self {
            storage_id,
            shape,
            strides,
            dtype: DType::I64,
            device: Device::Cpu,
            node_id: std::cell::Cell::new(None),
            requires_grad: std::cell::Cell::new(false),
        }
    }

    /// Creates a CPU tensor from a bool slice.
    ///
    /// # Panics
    /// Panics if data len does not match shape numel, or if the CPU Arena lock is poisoned.
    #[must_use]
    pub fn from_bool(data: &[bool], shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "Data len must match shape numel");
        let strides = shape.contiguous_strides();
        let shard_idx = current_thread_shard_idx();
        let storage_id = get_cpu_shard(shard_idx as usize)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .alloc_raw(CpuStorage::Bool(data.to_vec()), shard_idx);
        Self {
            storage_id,
            shape,
            strides,
            dtype: DType::Bool,
            device: Device::Cpu,
            node_id: std::cell::Cell::new(None),
            requires_grad: std::cell::Cell::new(false),
        }
    }

    /// Checks if the tensor is contiguous in row-major order.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.contiguous_strides()
    }

    /// Returns a contiguous copy of the tensor's data.
    ///
    /// # Panics
    /// Panics if the CPU Arena lock is poisoned.
    #[must_use]
    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        if self.shape.numel() == 0 {
            return Self::zeros(self.shape, self.dtype);
        }

        // 1. Lock only the source shard and copy elements out to build a fresh contiguous CpuStorage
        let copied_storage = {
            let guard = get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let slot = guard.slots[self.storage_id.slot_idx as usize]
                .as_ref()
                .expect("Source slot was empty");

            match &slot.storage {
                CpuStorage::F32(src_vec) => {
                    let mut dest_vec = vec![0.0; self.shape.numel()];
                    let mut iter = crate::shape::NdIterator::new(self.shape);
                    let mut dest_offset = 0;
                    loop {
                        let coord = iter.coord();
                        let src_offset =
                            crate::shape::get_offset(coord, &self.shape, &self.strides);
                        dest_vec[dest_offset] = src_vec[src_offset];
                        dest_offset += 1;
                        if !iter.step() {
                            break;
                        }
                    }
                    CpuStorage::F32(dest_vec)
                }
                CpuStorage::I32(src_vec) => {
                    let mut dest_vec = vec![0; self.shape.numel()];
                    let mut iter = crate::shape::NdIterator::new(self.shape);
                    let mut dest_offset = 0;
                    loop {
                        let coord = iter.coord();
                        let src_offset =
                            crate::shape::get_offset(coord, &self.shape, &self.strides);
                        dest_vec[dest_offset] = src_vec[src_offset];
                        dest_offset += 1;
                        if !iter.step() {
                            break;
                        }
                    }
                    CpuStorage::I32(dest_vec)
                }
                CpuStorage::I64(src_vec) => {
                    let mut dest_vec = vec![0; self.shape.numel()];
                    let mut iter = crate::shape::NdIterator::new(self.shape);
                    let mut dest_offset = 0;
                    loop {
                        let coord = iter.coord();
                        let src_offset =
                            crate::shape::get_offset(coord, &self.shape, &self.strides);
                        dest_vec[dest_offset] = src_vec[src_offset];
                        dest_offset += 1;
                        if !iter.step() {
                            break;
                        }
                    }
                    CpuStorage::I64(dest_vec)
                }
                CpuStorage::Bool(src_vec) => {
                    let mut dest_vec = vec![false; self.shape.numel()];
                    let mut iter = crate::shape::NdIterator::new(self.shape);
                    let mut dest_offset = 0;
                    loop {
                        let coord = iter.coord();
                        let src_offset =
                            crate::shape::get_offset(coord, &self.shape, &self.strides);
                        dest_vec[dest_offset] = src_vec[src_offset];
                        dest_offset += 1;
                        if !iter.step() {
                            break;
                        }
                    }
                    CpuStorage::Bool(dest_vec)
                }
                CpuStorage::F16(src_vec) | CpuStorage::BF16(src_vec) => {
                    let mut dest_vec = vec![0; self.shape.numel()];
                    let mut iter = crate::shape::NdIterator::new(self.shape);
                    let mut dest_offset = 0;
                    loop {
                        let coord = iter.coord();
                        let src_offset =
                            crate::shape::get_offset(coord, &self.shape, &self.strides);
                        dest_vec[dest_offset] = src_vec[src_offset];
                        dest_offset += 1;
                        if !iter.step() {
                            break;
                        }
                    }
                    if matches!(slot.storage, CpuStorage::F16(_)) {
                        CpuStorage::F16(dest_vec)
                    } else {
                        CpuStorage::BF16(dest_vec)
                    }
                }
            }
        };

        // 2. Allocate a new slot in the current thread's shard using the pre-populated storage
        let shard_idx = current_thread_shard_idx();
        let storage_id = get_cpu_shard(shard_idx as usize)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .alloc_raw(copied_storage, shard_idx);

        Self {
            storage_id,
            shape: self.shape,
            strides: self.shape.contiguous_strides(),
            dtype: self.dtype,
            device: self.device,
            node_id: std::cell::Cell::new(self.node_id.get()),
            requires_grad: std::cell::Cell::new(self.requires_grad.get()),
        }
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// # Panics
    /// Panics if the new shape's numel does not match current shape's numel, or if the CPU Arena lock is poisoned.
    #[must_use]
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Self {
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape.numel(),
            new_shape.numel(),
            "Reshape numel must match"
        );
        let contiguous_self = self.contiguous();
        let new_strides = new_shape.contiguous_strides();
        if contiguous_self.device.is_cpu() {
            get_cpu_shard(contiguous_self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .increment(contiguous_self.storage_id.slot_idx);
        }
        let mut out = Self {
            storage_id: contiguous_self.storage_id,
            shape: new_shape,
            strides: new_strides,
            dtype: contiguous_self.dtype,
            device: contiguous_self.device,
            node_id: std::cell::Cell::new(contiguous_self.node_id.get()),
            requires_grad: std::cell::Cell::new(contiguous_self.requires_grad.get()),
        };

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("reshape", std::slice::from_ref(&self), &mut out);
            }
        }
        out
    }

    /// Swaps two dimensions of the tensor on the stack.
    ///
    /// # Panics
    /// Panics if dimensions are out of bounds, or if the CPU Arena lock is poisoned.
    #[must_use]
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let shape = self.shape.swapped(dim0, dim1);
        let strides = self.strides.swapped(dim0, dim1);
        if self.device.is_cpu() {
            get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .increment(self.storage_id.slot_idx);
        }
        let mut out = Self {
            storage_id: self.storage_id,
            shape,
            strides,
            dtype: self.dtype,
            device: self.device,
            node_id: std::cell::Cell::new(self.node_id.get()),
            requires_grad: std::cell::Cell::new(self.requires_grad.get()),
        };

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record(
                    &format!("transpose_{dim0}_{dim1}"),
                    std::slice::from_ref(&self),
                    &mut out,
                );
            }
        }
        out
    }

    /// Permutes the dimensions of the tensor on the stack.
    ///
    /// # Panics
    /// Panics if the number of axes does not match the rank, or if axes are out of bounds or duplicate.
    #[must_use]
    pub fn permute(&self, axes: impl AsRef<[usize]>) -> Self {
        let axes = axes.as_ref();
        let rank = self.shape.rank();
        assert_eq!(
            axes.len(),
            rank,
            "Number of permutation axes must match rank"
        );

        let mut seen = 0u8;
        let mut new_dims = [0; 8];
        let mut new_strides = [0; 8];

        for (i, &ax) in axes.iter().enumerate() {
            assert!(ax < rank, "Axis index out of bounds");
            let mask = 1 << ax;
            assert_eq!(seen & mask, 0, "Duplicate permutation axes are not allowed");
            seen |= mask;
            new_dims[i] = self.shape[ax];
            new_strides[i] = self.strides[ax];
        }

        let new_shape = Shape::new(&new_dims[..rank]);
        let new_strides_shape = Shape::new(&new_strides[..rank]);

        if self.device.is_cpu() {
            get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .increment(self.storage_id.slot_idx);
        }

        let mut out = Self {
            storage_id: self.storage_id,
            shape: new_shape,
            strides: new_strides_shape,
            dtype: self.dtype,
            device: self.device,
            node_id: std::cell::Cell::new(self.node_id.get()),
            requires_grad: std::cell::Cell::new(self.requires_grad.get()),
        };

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                let axes_str = axes
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                record(
                    &format!("permute_{axes_str}"),
                    std::slice::from_ref(&self),
                    &mut out,
                );
            }
        }
        out
    }

    /// Get shape ref.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get strides ref.
    #[must_use]
    pub const fn strides(&self) -> &Shape {
        &self.strides
    }

    /// Get `DType`.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get `Device`.
    #[must_use]
    pub const fn device(&self) -> Device {
        self.device
    }

    /// Get `StorageId`.
    #[must_use]
    pub const fn storage_id(&self) -> StorageId {
        self.storage_id
    }

    /// Sets whether this tensor requires gradient computation.
    pub fn set_requires_grad(&self, requires_grad: bool) {
        self.requires_grad.set(requires_grad);
    }

    /// Check if this tensor requires gradient.
    #[must_use]
    pub const fn requires_grad(&self) -> bool {
        self.requires_grad.get()
    }

    /// Access the tape node token of this tensor.
    ///
    /// The token is opaque to core - the autograd crate packs a `(generation,
    /// index)` pair into it so tokens from a previous tape are self-invalidating.
    #[must_use]
    pub const fn node_id(&self) -> Option<u64> {
        self.node_id.get()
    }

    /// Sets the tape node token of this tensor.
    pub fn set_node_id(&self, node_id: Option<u64>) {
        self.node_id.set(node_id);
    }

    /// Compute gradients of this tensor's scalar output with respect to all leaf variables.
    ///
    /// # Panics
    /// Panics if this tensor is not scalar (numel != 1) or if backward hook is not registered.
    pub fn backward(&self) {
        assert_eq!(
            self.shape().numel(),
            1,
            "backward can only be called on scalar tensors"
        );
        if let Some(backward) = BACKWARD_HOOK.get() {
            backward(self);
        } else {
            panic!("Autograd backward hook is not registered. Did you call vearo::init()?");
        }
    }

    /// Retrieve the accumulated gradient tensor for this tensor.
    #[must_use]
    pub fn grad(&self) -> Option<Self> {
        GRAD_HOOK.get().and_then(|grad_hook| grad_hook(self))
    }

    /// Read a value at a flat index in the storage of a contiguous F32 tensor.
    ///
    /// # Panics
    /// Panics if the tensor is non-contiguous, not F32, or index is out of bounds.
    #[must_use]
    pub fn get_f32(&self, index: usize) -> f32 {
        assert!(self.is_contiguous(), "get_f32 requires contiguous tensor");
        let guard = get_cpu_shard(self.storage_id.shard_idx as usize)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let slot = guard.slots[self.storage_id.slot_idx as usize]
            .as_ref()
            .unwrap();
        let val = match &slot.storage {
            CpuStorage::F32(vec) => {
                assert!(index < vec.len(), "Index out of bounds");
                vec[index]
            }
            _ => panic!("Expected F32 storage"),
        };
        drop(guard);
        val
    }

    /// Modify a value at a flat index in the storage of a contiguous F32 tensor.
    ///
    /// # Panics
    /// Panics if the tensor is non-contiguous, not F32, or index is out of bounds.
    pub fn set_f32(&self, index: usize, value: f32) {
        assert!(self.is_contiguous(), "set_f32 requires contiguous tensor");
        let mut guard = get_cpu_shard(self.storage_id.shard_idx as usize)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let slot = guard.slots[self.storage_id.slot_idx as usize]
            .as_mut()
            .unwrap();
        match &mut slot.storage {
            CpuStorage::F32(vec) => {
                assert!(index < vec.len(), "Index out of bounds");
                vec[index] = value;
            }
            _ => panic!("Expected F32 storage"),
        }
        drop(guard);
    }
}

/// Function pointer signature for binary tensor operations.
pub type BinaryOpFn = fn(&Tensor, &Tensor) -> Tensor;

/// Function pointer signature for unary elementwise operations.
pub type UnaryOpFn = fn(&Tensor) -> Tensor;

/// Function pointer signature for single-axis reductions: `(tensor, dim, keep_dim)`.
pub type ReduceOpFn = fn(&Tensor, usize, bool) -> Tensor;

/// Set of backend implementations for basic tensor operations.
#[derive(Clone, Copy)]
pub struct BackendOps {
    /// Elementwise addition function.
    pub add: BinaryOpFn,
    /// Elementwise subtraction function.
    pub sub: BinaryOpFn,
    /// Elementwise multiplication function.
    pub mul: BinaryOpFn,
    /// Elementwise division function.
    pub div: BinaryOpFn,
    /// Matrix multiplication function.
    pub matmul: BinaryOpFn,
    /// Elementwise `ReLU` function.
    pub relu: UnaryOpFn,
    /// Sum reduction over a single axis.
    pub sum: ReduceOpFn,
    /// Mean reduction over a single axis.
    pub mean: ReduceOpFn,
}

/// Per-backend registry of op implementations, indexed by [`Device::backend_idx`].
///
/// One slot per backend so CPU / CUDA / ... can coexist and a tensor dispatches to
/// the implementation for *its own* device.
static BACKEND_OPS: [std::sync::OnceLock<BackendOps>; crate::NUM_BACKENDS] =
    [const { std::sync::OnceLock::new() }; crate::NUM_BACKENDS];

/// Registers the op implementations for a device's backend.
///
/// Idempotent: registering an already-registered backend is a no-op, so it is
/// safe to call from every test and at application startup without coordination.
pub fn register_backend_ops(device: Device, ops: BackendOps) {
    let _ = BACKEND_OPS[device.backend_idx()].set(ops);
}

thread_local! {
    /// Thread-local flag to enable/disable autograd tape recording.
    static AUTOGRAD_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Sets whether autograd tape recording is enabled for the current thread.
pub fn set_autograd_enabled(enabled: bool) {
    AUTOGRAD_ENABLED.with(|e| e.set(enabled));
}

/// Check if autograd tape recording is enabled for the current thread.
#[must_use]
pub fn is_autograd_enabled() -> bool {
    AUTOGRAD_ENABLED.with(std::cell::Cell::get)
}

/// Function pointer signature for autograd tape recording hook.
pub type RecordOpFn = fn(op_name: &str, inputs: &[&Tensor], output: &mut Tensor);

/// Global registry for the autograd tape recording hook.
pub static RECORD_OP: std::sync::OnceLock<RecordOpFn> = std::sync::OnceLock::new();

/// Registers the autograd tape recording hook.
pub fn register_record_op(f: RecordOpFn) {
    let _ = RECORD_OP.set(f);
}

/// Function pointer signature for autograd backward pass hook.
pub type BackwardFn = fn(&Tensor);

/// Global registry for the autograd backward pass hook.
pub static BACKWARD_HOOK: std::sync::OnceLock<BackwardFn> = std::sync::OnceLock::new();

/// Registers the autograd backward pass hook.
pub fn register_backward_hook(f: BackwardFn) {
    let _ = BACKWARD_HOOK.set(f);
}

/// Function pointer signature for retrieving a tensor's gradient.
pub type GradFn = fn(&Tensor) -> Option<Tensor>;

/// Global registry for retrieving a tensor's gradient.
pub static GRAD_HOOK: std::sync::OnceLock<GradFn> = std::sync::OnceLock::new();

/// Registers the autograd gradient lookup hook.
pub fn register_grad_hook(f: GradFn) {
    let _ = GRAD_HOOK.set(f);
}

impl Tensor {
    /// Elementwise addition.
    ///
    /// # Panics
    /// Panics if backend operations are not registered, or if shapes are not broadcastable.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.device(),
            other.device(),
            "Tensors must reside on the same device"
        );
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.add)(self, other);

        if is_autograd_enabled() && (self.requires_grad() || other.requires_grad()) {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("add", &[self, other], &mut out);
            }
        }

        out
    }

    /// Elementwise subtraction.
    ///
    /// # Panics
    /// Panics if backend operations are not registered, or if shapes are not broadcastable.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(
            self.device(),
            other.device(),
            "Tensors must reside on the same device"
        );
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.sub)(self, other);

        if is_autograd_enabled() && (self.requires_grad() || other.requires_grad()) {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("sub", &[self, other], &mut out);
            }
        }

        out
    }

    /// Elementwise multiplication.
    ///
    /// # Panics
    /// Panics if backend operations are not registered, or if shapes are not broadcastable.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(
            self.device(),
            other.device(),
            "Tensors must reside on the same device"
        );
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.mul)(self, other);

        if is_autograd_enabled() && (self.requires_grad() || other.requires_grad()) {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("mul", &[self, other], &mut out);
            }
        }

        out
    }

    /// Elementwise division.
    ///
    /// # Panics
    /// Panics if backend operations are not registered, or if shapes are not broadcastable.
    #[must_use]
    pub fn div(&self, other: &Self) -> Self {
        assert_eq!(
            self.device(),
            other.device(),
            "Tensors must reside on the same device"
        );
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.div)(self, other);

        if is_autograd_enabled() && (self.requires_grad() || other.requires_grad()) {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("div", &[self, other], &mut out);
            }
        }

        out
    }

    /// Matrix multiplication.
    ///
    /// # Panics
    /// Panics if backend operations are not registered, or if dimensions are incompatible.
    #[must_use]
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.device(),
            other.device(),
            "Tensors must reside on the same device"
        );
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.matmul)(self, other);

        if is_autograd_enabled() && (self.requires_grad() || other.requires_grad()) {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("matmul", &[self, other], &mut out);
            }
        }

        out
    }

    /// Elementwise `ReLU`: `max(0, x)`.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn relu(&self) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.relu)(self);

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("relu", &[self], &mut out);
            }
        }

        out
    }

    /// Sum over a single axis. With `keep_dim`, the reduced axis is kept as size 1.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device, or `dim` is out of range.
    #[must_use]
    pub fn sum(&self, dim: usize, keep_dim: bool) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.sum)(self, dim, keep_dim);

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record(
                    &format!("sum_{dim}_{}", u8::from(keep_dim)),
                    &[self],
                    &mut out,
                );
            }
        }

        out
    }

    /// Mean over a single axis. With `keep_dim`, the reduced axis is kept as size 1.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device, or `dim` is out of range.
    #[must_use]
    pub fn mean(&self, dim: usize, keep_dim: bool) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.mean)(self, dim, keep_dim);

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record(
                    &format!("mean_{dim}_{}", u8::from(keep_dim)),
                    &[self],
                    &mut out,
                );
            }
        }

        out
    }
}

// -----------------------------------------------------------------------------
// Operator Overloading (Add, Sub, Mul, Div)
// -----------------------------------------------------------------------------

// Add implementations
#[allow(clippy::use_self)]
impl std::ops::Add for Tensor {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Tensor::add(&self, &other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Add<&Tensor> for Tensor {
    type Output = Self;
    fn add(self, other: &Tensor) -> Self {
        Tensor::add(&self, other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        Tensor::add(self, &other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: Self) -> Tensor {
        Tensor::add(self, other)
    }
}

// Sub implementations
#[allow(clippy::use_self)]
impl std::ops::Sub for Tensor {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Tensor::sub(&self, &other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Self;
    fn sub(self, other: &Tensor) -> Self {
        Tensor::sub(&self, other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Tensor {
        Tensor::sub(self, &other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: Self) -> Tensor {
        Tensor::sub(self, other)
    }
}

// Mul implementations
#[allow(clippy::use_self)]
impl std::ops::Mul for Tensor {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Tensor::mul(&self, &other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Self;
    fn mul(self, other: &Tensor) -> Self {
        Tensor::mul(&self, other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        Tensor::mul(self, &other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: Self) -> Tensor {
        Tensor::mul(self, other)
    }
}

// Div implementations
#[allow(clippy::use_self)]
impl std::ops::Div for Tensor {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Tensor::div(&self, &other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Div<&Tensor> for Tensor {
    type Output = Self;
    fn div(self, other: &Tensor) -> Self {
        Tensor::div(&self, other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, other: Tensor) -> Tensor {
        Tensor::div(self, &other)
    }
}

#[allow(clippy::use_self)]
impl std::ops::Div for &Tensor {
    type Output = Tensor;
    fn div(self, other: Self) -> Tensor {
        Tensor::div(self, other)
    }
}
