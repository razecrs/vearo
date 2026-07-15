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
        if let Some(inc) = REFCOUNT_INC.get() {
            inc(self.storage_id, self.device);
        } else if self.device.is_cpu() {
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
        let is_last = if let Some(dec) = REFCOUNT_DEC.get() {
            dec(self.storage_id, self.device)
        } else if self.device.is_cpu() {
            let mut shard = get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            let last = if let Some(Some(slot)) = shard.slots.get(self.storage_id.slot_idx as usize)
            {
                slot.ref_count == 1
            } else {
                false
            };

            shard.decrement(self.storage_id.slot_idx);
            last
        } else {
            false
        };

        if is_last && let Some(drop_hook) = DROP_HOOK.get() {
            drop_hook(self.storage_id);
        }
    }
}

impl Tensor {
    /// Create a zero-filled tensor on a specific device.
    ///
    /// # Panics
    /// Panics if device is not CPU or CUDA, or if CUDA hooks are not registered.
    #[must_use]
    pub fn zeros_on(shape: impl Into<Shape>, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        let strides = shape.contiguous_strides();

        let storage_id = if device.is_cpu() {
            let shard_idx = current_thread_shard_idx();
            get_cpu_shard(shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .alloc(numel, dtype, shard_idx)
        } else if device.is_cuda() {
            let alloc = CUDA_ALLOC_HOOK
                .get()
                .expect("CUDA alloc hook not registered");
            alloc(numel)
        } else {
            panic!("Unsupported device: {device:?}");
        };

        Self {
            storage_id,
            shape,
            strides,
            dtype,
            device,
            node_id: std::cell::Cell::new(None),
            requires_grad: std::cell::Cell::new(false),
        }
    }

    /// Creates a zero filled CPU tensor.
    ///
    /// # Panics
    /// Panics if the CPU Arena lock is poisoned.
    #[must_use]
    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> Self {
        Self::zeros_on(shape, dtype, Device::Cpu)
    }

    /// Create a CUDA tensor from CPU data.
    ///
    /// # Panics
    /// Panics if CUDA hooks are not registered.
    #[must_use]
    pub fn from_f32_cuda(data: &[f32], shape: impl Into<Shape>, device: Device) -> Self {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "Data len must match shape numel");
        let strides = shape.contiguous_strides();

        let alloc = CUDA_ALLOC_HOOK
            .get()
            .expect("CUDA alloc hook not registered");
        let write = CUDA_WRITE_HOOK
            .get()
            .expect("CUDA write hook not registered");

        let storage_id = alloc(shape.numel());
        write(storage_id, data);

        Self {
            storage_id,
            shape,
            strides,
            dtype: DType::F32,
            device,
            node_id: std::cell::Cell::new(None),
            requires_grad: std::cell::Cell::new(false),
        }
    }

    /// Create a tensor directly from components.
    #[must_use]
    pub const fn from_components(
        storage_id: StorageId,
        shape: Shape,
        strides: Shape,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            storage_id,
            shape,
            strides,
            dtype,
            device,
            node_id: std::cell::Cell::new(None),
            requires_grad: std::cell::Cell::new(false),
        }
    }

    /// Move tensor to a different device.
    ///
    /// # Panics
    /// Panics if device is not CPU or CUDA, or if CUDA hooks are not registered.
    #[must_use]
    pub fn to(&self, device: Device) -> Self {
        if self.device == device {
            return self.clone();
        }

        let numel = self.shape().numel();
        let data = self.to_vec_f32();

        if device.is_cpu() {
            let mut out = Self::from_f32(&data, *self.shape());
            out.requires_grad = self.requires_grad.clone();
            out
        } else if device.is_cuda() {
            let alloc = CUDA_ALLOC_HOOK
                .get()
                .expect("CUDA alloc hook not registered");
            let write = CUDA_WRITE_HOOK
                .get()
                .expect("CUDA write hook not registered");

            let storage_id = alloc(numel);
            write(storage_id, &data);

            Self {
                storage_id,
                shape: self.shape,
                strides: self.strides,
                dtype: self.dtype,
                device,
                node_id: std::cell::Cell::new(None),
                requires_grad: std::cell::Cell::new(self.requires_grad.get()),
            }
        } else {
            panic!("Unsupported device: {device:?}");
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
            .alloc_raw(
                CpuStorage::F32(std::sync::Arc::new(data.to_vec())),
                shard_idx,
            );
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
            .alloc_raw(
                CpuStorage::I32(std::sync::Arc::new(data.to_vec())),
                shard_idx,
            );
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
            .alloc_raw(
                CpuStorage::I64(std::sync::Arc::new(data.to_vec())),
                shard_idx,
            );
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
            .alloc_raw(
                CpuStorage::Bool(std::sync::Arc::new(data.to_vec())),
                shard_idx,
            );
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
        if self.device.is_cuda() {
            let read = CUDA_READ_HOOK.get().expect("CUDA read hook not registered");
            let raw_data = read(self.storage_id);
            let shard_idx = current_thread_shard_idx();
            let cpu_storage_id = get_cpu_shard(shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .alloc_raw(CpuStorage::F32(std::sync::Arc::new(raw_data)), shard_idx);
            let cpu_view = Self {
                storage_id: cpu_storage_id,
                shape: self.shape,
                strides: self.strides,
                dtype: self.dtype,
                device: Device::Cpu,
                node_id: std::cell::Cell::new(None),
                requires_grad: std::cell::Cell::new(false),
            };
            let cpu_contiguous = cpu_view.contiguous();
            return cpu_contiguous.to(self.device);
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
                    CpuStorage::F32(std::sync::Arc::new(dest_vec))
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
                    CpuStorage::I32(std::sync::Arc::new(dest_vec))
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
                    CpuStorage::I64(std::sync::Arc::new(dest_vec))
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
                    CpuStorage::Bool(std::sync::Arc::new(dest_vec))
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
                        CpuStorage::F16(std::sync::Arc::new(dest_vec))
                    } else {
                        CpuStorage::BF16(std::sync::Arc::new(dest_vec))
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
        if let Some(inc) = REFCOUNT_INC.get() {
            inc(contiguous_self.storage_id, contiguous_self.device);
        } else if contiguous_self.device.is_cpu() {
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
        if let Some(inc) = REFCOUNT_INC.get() {
            inc(self.storage_id, self.device);
        } else if self.device.is_cpu() {
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

        if let Some(inc) = REFCOUNT_INC.get() {
            inc(self.storage_id, self.device);
        } else if self.device.is_cpu() {
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

    /// Read all values of a contiguous F32 tensor as a Vec.
    ///
    /// # Panics
    /// Panics if the tensor is non-contiguous or not F32.
    #[must_use]
    pub fn to_vec_f32(&self) -> Vec<f32> {
        assert!(
            self.is_contiguous(),
            "to_vec_f32 requires contiguous tensor"
        );
        if self.device.is_cpu() {
            let guard = get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let slot = guard.slots[self.storage_id.slot_idx as usize]
                .as_ref()
                .expect("Storage slot was empty");
            let val = match &slot.storage {
                CpuStorage::F32(vec) => vec.as_ref().clone(),
                _ => panic!("Expected F32 storage"),
            };
            drop(guard);
            val
        } else if self.device.is_cuda() {
            let read = CUDA_READ_HOOK.get().expect("CUDA read hook not registered");
            read(self.storage_id)
        } else {
            panic!("Unsupported device: {:?}", self.device);
        }
    }

    /// Modify a value at a flat index in the storage of a contiguous F32 tensor.
    ///
    /// # Panics
    /// Panics if the tensor is non-contiguous, not F32, or index is out of bounds.
    pub fn set_f32(&self, index: usize, value: f32) {
        assert!(self.is_contiguous(), "set_f32 requires contiguous tensor");
        if self.device.is_cpu() {
            let mut guard = get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let slot = guard.slots[self.storage_id.slot_idx as usize]
                .as_mut()
                .unwrap();
            match &mut slot.storage {
                CpuStorage::F32(vec) => {
                    let vec_mut = std::sync::Arc::make_mut(vec);
                    assert!(index < vec_mut.len(), "Index out of bounds");
                    vec_mut[index] = value;
                }
                _ => panic!("Expected F32 storage"),
            }
            drop(guard);
        } else if self.device.is_cuda() {
            let mut data = self.to_vec_f32();
            assert!(index < data.len(), "Index out of bounds");
            data[index] = value;
            let write = CUDA_WRITE_HOOK
                .get()
                .expect("CUDA write hook not registered");
            write(self.storage_id, &data);
        } else {
            panic!("Unsupported device: {:?}", self.device);
        }
    }

    /// In-place scaled addition: `self += scale * other`.
    ///
    /// # Panics
    /// Panics if shapes do not match, either tensor is non-contiguous, or dtypes are not F32.
    #[allow(clippy::significant_drop_tightening, clippy::suboptimal_flops)]
    pub fn add_assign_scaled(&self, other: &Self, scale: f32) {
        assert_eq!(
            self.shape, other.shape,
            "add_assign_scaled shapes must match"
        );
        assert!(
            self.is_contiguous(),
            "add_assign_scaled self must be contiguous"
        );
        assert!(
            other.is_contiguous(),
            "add_assign_scaled other must be contiguous"
        );
        assert_eq!(self.dtype, DType::F32, "add_assign_scaled self must be F32");
        assert_eq!(
            other.dtype,
            DType::F32,
            "add_assign_scaled other must be F32"
        );

        if self.device.is_cpu() {
            let other_data = {
                let guard = get_cpu_shard(other.storage_id.shard_idx as usize)
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let slot = guard.slots[other.storage_id.slot_idx as usize]
                    .as_ref()
                    .expect("Other slot was empty");
                match &slot.storage {
                    CpuStorage::F32(vec) => vec.clone(),
                    _ => panic!("Expected F32 storage for other"),
                }
            };

            let mut guard = get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let slot = guard.slots[self.storage_id.slot_idx as usize]
                .as_mut()
                .expect("Self slot was empty");
            match &mut slot.storage {
                CpuStorage::F32(vec) => {
                    let vec_mut = std::sync::Arc::make_mut(vec);
                    for (dest, &src) in vec_mut.iter_mut().zip(other_data.iter()) {
                        *dest += scale * src;
                    }
                }
                _ => panic!("Expected F32 storage for self"),
            }
        } else if self.device.is_cuda() {
            let mut self_data = self.to_vec_f32();
            let other_data = other.to_vec_f32();
            for (dest, &src) in self_data.iter_mut().zip(other_data.iter()) {
                *dest += scale * src;
            }
            let write = CUDA_WRITE_HOOK
                .get()
                .expect("CUDA write hook not registered");
            write(self.storage_id, &self_data);
        } else {
            panic!("Unsupported device: {:?}", self.device);
        }
    }

    /// Perform an in-place AdamW step on this tensor's CPU storage.
    ///
    /// # Panics
    /// Panics if shapes do not match, either tensor is non-contiguous, or dtypes are not F32.
    #[allow(
        clippy::too_many_arguments,
        clippy::significant_drop_tightening,
        clippy::suboptimal_flops,
        clippy::doc_markdown
    )]
    pub fn adamw_step(
        &self,
        grad: &Self,
        m: &mut [f32],
        v: &mut [f32],
        t: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) {
        assert_eq!(self.shape, grad.shape, "adamw_step shapes must match");
        assert!(self.is_contiguous(), "adamw_step self must be contiguous");
        assert!(grad.is_contiguous(), "adamw_step grad must be contiguous");
        assert_eq!(self.dtype, DType::F32, "adamw_step self must be F32");
        assert_eq!(grad.dtype, DType::F32, "adamw_step grad must be F32");

        let grad_data = grad.to_vec_f32();

        if self.device.is_cpu() {
            let mut guard = get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let slot = guard.slots[self.storage_id.slot_idx as usize]
                .as_mut()
                .expect("Self slot was empty");
            match &mut slot.storage {
                CpuStorage::F32(vec) => {
                    let vec_mut = std::sync::Arc::make_mut(vec);
                    assert_eq!(
                        vec_mut.len(),
                        m.len(),
                        "Momentum vector m must match parameter length"
                    );
                    assert_eq!(
                        vec_mut.len(),
                        v.len(),
                        "Variance vector v must match parameter length"
                    );

                    let bias_correction1 = 1.0 - beta1.powi(i32::try_from(t).unwrap());
                    let bias_correction2 = 1.0 - beta2.powi(i32::try_from(t).unwrap());

                    for i in 0..vec_mut.len() {
                        let g_i = grad_data[i];
                        m[i] = beta1 * m[i] + (1.0 - beta1) * g_i;
                        v[i] = beta2 * v[i] + (1.0 - beta2) * g_i * g_i;

                        let m_hat = m[i] / bias_correction1;
                        let v_hat = v[i] / bias_correction2;

                        vec_mut[i] -= lr * weight_decay * vec_mut[i];
                        vec_mut[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                    }
                }
                _ => panic!("Expected F32 storage for self"),
            }
        } else if self.device.is_cuda() {
            let mut self_data = self.to_vec_f32();
            assert_eq!(
                self_data.len(),
                m.len(),
                "Momentum vector m must match parameter length"
            );
            assert_eq!(
                self_data.len(),
                v.len(),
                "Variance vector v must match parameter length"
            );

            let bias_correction1 = 1.0 - beta1.powi(i32::try_from(t).unwrap());
            let bias_correction2 = 1.0 - beta2.powi(i32::try_from(t).unwrap());

            for i in 0..self_data.len() {
                let g_i = grad_data[i];
                m[i] = beta1 * m[i] + (1.0 - beta1) * g_i;
                v[i] = beta2 * v[i] + (1.0 - beta2) * g_i * g_i;

                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;

                self_data[i] -= lr * weight_decay * self_data[i];
                self_data[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
            }
            let write = CUDA_WRITE_HOOK
                .get()
                .expect("CUDA write hook not registered");
            write(self.storage_id, &self_data);
        } else {
            panic!("Unsupported device: {:?}", self.device);
        }
    }

    /// Perform an in-place SGD step with momentum and weight decay on this tensor's CPU storage.
    ///
    /// # Panics
    /// Panics if shapes do not match, either tensor is non-contiguous, or dtypes are not F32.
    #[allow(clippy::significant_drop_tightening, clippy::suboptimal_flops)]
    pub fn sgd_step(
        &self,
        grad: &Self,
        velocity: &mut [f32],
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    ) {
        assert_eq!(self.shape, grad.shape, "sgd_step shapes must match");
        assert!(self.is_contiguous(), "sgd_step self must be contiguous");
        assert!(grad.is_contiguous(), "sgd_step grad must be contiguous");
        assert_eq!(self.dtype, DType::F32, "sgd_step self must be F32");
        assert_eq!(grad.dtype, DType::F32, "sgd_step grad must be F32");

        let grad_data = grad.to_vec_f32();

        if self.device.is_cpu() {
            let mut guard = get_cpu_shard(self.storage_id.shard_idx as usize)
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let slot = guard.slots[self.storage_id.slot_idx as usize]
                .as_mut()
                .expect("Self slot was empty");
            match &mut slot.storage {
                CpuStorage::F32(vec) => {
                    let vec_mut = std::sync::Arc::make_mut(vec);
                    assert_eq!(
                        vec_mut.len(),
                        velocity.len(),
                        "Velocity vector must match parameter length"
                    );

                    for i in 0..vec_mut.len() {
                        let mut g_i = grad_data[i];
                        if weight_decay != 0.0 {
                            g_i += weight_decay * vec_mut[i];
                        }
                        velocity[i] = momentum * velocity[i] + g_i;
                        vec_mut[i] -= lr * velocity[i];
                    }
                }
                _ => panic!("Expected F32 storage for self"),
            }
        } else if self.device.is_cuda() {
            let mut self_data = self.to_vec_f32();
            assert_eq!(
                self_data.len(),
                velocity.len(),
                "Velocity vector must match parameter length"
            );

            for i in 0..self_data.len() {
                let mut g_i = grad_data[i];
                if weight_decay != 0.0 {
                    g_i += weight_decay * self_data[i];
                }
                velocity[i] = momentum * velocity[i] + g_i;
                self_data[i] -= lr * velocity[i];
            }
            let write = CUDA_WRITE_HOOK
                .get()
                .expect("CUDA write hook not registered");
            write(self.storage_id, &self_data);
        } else {
            panic!("Unsupported device: {:?}", self.device);
        }
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
#[allow(clippy::type_complexity)]
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
    /// Elementwise `GELU` function.
    pub gelu: UnaryOpFn,
    /// Softmax reduction function.
    pub softmax: fn(&Tensor, usize) -> Tensor,
    /// Layer normalization function.
    pub layernorm: fn(&Tensor, &Tensor, &Tensor, f32) -> Tensor,
    /// Layer normalization backward function.
    pub layernorm_backward: fn(&Tensor, &Tensor, &Tensor, &Tensor, f32) -> (Tensor, Tensor, Tensor),
    /// Embedding lookup function.
    pub embedding: fn(&Tensor, &Tensor) -> Tensor,
    /// Embedding lookup backward function.
    pub embedding_backward: fn(&Tensor, &Tensor, &Tensor) -> Tensor,
    /// Categorical cross-entropy loss function.
    pub cross_entropy: fn(&Tensor, &Tensor) -> Tensor,
    /// Categorical cross-entropy loss backward function.
    pub cross_entropy_backward: fn(&Tensor, &Tensor, &Tensor) -> Tensor,
    /// Two-dimensional convolution (input, weight, bias, stride, padding).
    pub conv2d: fn(&Tensor, &Tensor, &Tensor, usize, usize) -> Tensor,
    /// Backward for the convolution: returns (grad input, grad weight, grad bias).
    pub conv2d_backward: fn(&Tensor, &Tensor, &Tensor, usize, usize) -> (Tensor, Tensor, Tensor),
    /// Two-dimensional max pooling (input, window, stride, padding).
    pub maxpool2d: fn(&Tensor, usize, usize, usize) -> Tensor,
    /// Backward for max pooling; returns the gradient with respect to the input.
    pub maxpool2d_backward: fn(&Tensor, &Tensor, usize, usize, usize) -> Tensor,
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

/// Global registry for the autograd drop cleanup hook.
pub static DROP_HOOK: std::sync::OnceLock<fn(StorageId)> = std::sync::OnceLock::new();

/// Registers the autograd drop cleanup hook.
pub fn register_drop_hook(f: fn(StorageId)) {
    let _ = DROP_HOOK.set(f);
}

/// Hook for incrementing reference count of a tensor's storage.
pub static REFCOUNT_INC: std::sync::OnceLock<fn(StorageId, Device)> = std::sync::OnceLock::new();

/// Registers the refcount increment hook.
pub fn register_refcount_inc(f: fn(StorageId, Device)) {
    let _ = REFCOUNT_INC.set(f);
}

/// Hook for decrementing reference count of a tensor's storage. Returns true if freed.
pub static REFCOUNT_DEC: std::sync::OnceLock<fn(StorageId, Device) -> bool> =
    std::sync::OnceLock::new();

/// Registers the refcount decrement hook.
pub fn register_refcount_dec(f: fn(StorageId, Device) -> bool) {
    let _ = REFCOUNT_DEC.set(f);
}

/// Hook for reading CUDA device memory.
pub static CUDA_READ_HOOK: std::sync::OnceLock<fn(StorageId) -> Vec<f32>> =
    std::sync::OnceLock::new();
/// Hook for writing CUDA device memory.
pub static CUDA_WRITE_HOOK: std::sync::OnceLock<fn(StorageId, &[f32])> = std::sync::OnceLock::new();
/// Hook for allocating CUDA device memory.
pub static CUDA_ALLOC_HOOK: std::sync::OnceLock<fn(usize) -> StorageId> =
    std::sync::OnceLock::new();

/// Registers the CUDA hooks.
pub fn register_cuda_hooks(
    read: fn(StorageId) -> Vec<f32>,
    write: fn(StorageId, &[f32]),
    alloc: fn(usize) -> StorageId,
) {
    let _ = CUDA_READ_HOOK.set(read);
    let _ = CUDA_WRITE_HOOK.set(write);
    let _ = CUDA_ALLOC_HOOK.set(alloc);
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

    /// Elementwise `GELU` activation.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn gelu(&self) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.gelu)(self);

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("gelu", &[self], &mut out);
            }
        }

        out
    }

    /// Softmax along a single axis.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device, or `dim` is out of range.
    #[must_use]
    pub fn softmax(&self, dim: usize) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.softmax)(self, dim);

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record(&format!("softmax_{dim}"), &[self], &mut out);
            }
        }

        out
    }

    /// Layer normalization.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn layernorm(&self, weight: &Self, bias: &Self, eps: f32) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.layernorm)(self, weight, bias, eps);

        if is_autograd_enabled()
            && (self.requires_grad() || weight.requires_grad() || bias.requires_grad())
        {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record(&format!("layernorm_{eps}"), &[self, weight, bias], &mut out);
            }
        }

        out
    }

    /// Layer normalization backward. Used internally by autograd.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn layernorm_backward(
        &self,
        weight: &Self,
        bias: &Self,
        grad_out: &Self,
        eps: f32,
    ) -> (Self, Self, Self) {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        (ops.layernorm_backward)(self, weight, bias, grad_out, eps)
    }

    /// Embedding lookup.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn embedding(&self, weight: &Self) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.embedding)(self, weight);

        if is_autograd_enabled() && weight.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("embedding", &[self, weight], &mut out);
            }
        }

        out
    }

    /// Embedding lookup backward. Used internally by autograd.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn embedding_backward(&self, weight: &Self, grad_out: &Self) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        (ops.embedding_backward)(self, weight, grad_out)
    }

    /// Categorical cross-entropy loss.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn cross_entropy(&self, targets: &Self) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.cross_entropy)(self, targets);

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record("cross_entropy", &[self, targets], &mut out);
            }
        }

        out
    }

    /// Categorical cross-entropy loss backward. Used internally by autograd.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn cross_entropy_backward(&self, targets: &Self, grad_out: &Self) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        (ops.cross_entropy_backward)(self, targets, grad_out)
    }

    /// Two-dimensional convolution (self=input, plus weight, bias, stride, padding).
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn conv2d(&self, weight: &Self, bias: &Self, stride: usize, padding: usize) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.conv2d)(self, weight, bias, stride, padding);

        if is_autograd_enabled()
            && (self.requires_grad() || weight.requires_grad() || bias.requires_grad())
        {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record(
                    &format!("conv2d_{stride}_{padding}"),
                    &[self, weight, bias],
                    &mut out,
                );
            }
        }

        out
    }

    /// Backward for the convolution: returns (grad input, grad weight, grad bias).
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn conv2d_backward(
        &self,
        weight: &Self,
        grad_out: &Self,
        stride: usize,
        padding: usize,
    ) -> (Self, Self, Self) {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        (ops.conv2d_backward)(self, weight, grad_out, stride, padding)
    }

    /// Two-dimensional max pooling over the spatial dimensions of a 4D tensor.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn maxpool2d(&self, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        let mut out = (ops.maxpool2d)(self, kernel_size, stride, padding);

        if is_autograd_enabled() && self.requires_grad() {
            out.set_requires_grad(true);
            if let Some(record) = RECORD_OP.get() {
                record(
                    &format!("maxpool2d_{kernel_size}_{stride}_{padding}"),
                    &[self],
                    &mut out,
                );
            }
        }

        out
    }

    /// Backward for [`maxpool2d`](Self::maxpool2d): returns the gradient w.r.t. the input.
    ///
    /// # Panics
    /// Panics if no backend is registered for this device.
    #[must_use]
    pub fn maxpool2d_backward(
        &self,
        grad_out: &Self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let ops = BACKEND_OPS[self.device().backend_idx()]
            .get()
            .expect("No backend registered for this device. Did you call the backend's init()?");
        (ops.maxpool2d_backward)(self, grad_out, kernel_size, stride, padding)
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
