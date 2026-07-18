//! Core tensor primitives for Vearo.
//!
//! This crate is super tiny. It just holds the vocabulary (dtypes, shapes, devices)
//! that everything else uses. No computation happens here.
//! The real tensor type will land in Phase 1.

mod device;
mod dtype;
mod shape;
mod storage;
mod tensor;

pub use device::{Device, NUM_BACKENDS};
pub use dtype::DType;
pub use shape::{NdIterator, Shape, get_offset};
pub use storage::{
    CpuArenaShard, CpuStorage, NUM_SHARDS, StorageId, current_thread_shard_idx, get_cpu_shard,
};
pub use tensor::{
    BackendOps, BackwardFn, GradFn, RecordOpFn, Tensor, is_autograd_enabled, is_training,
    register_backend_ops, register_backward_hook, register_cuda_hooks, register_drop_hook,
    register_grad_hook, register_record_op, register_refcount_dec, register_refcount_inc,
    set_autograd_enabled, set_training,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes_and_float_flag() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::I64.size_bytes(), 8);
        assert_eq!(DType::Bool.size_bytes(), 1);

        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::I64.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[test]
    fn device_checks() {
        let cpu = Device::Cpu;
        let cuda = Device::Cuda(3);

        assert!(cpu.is_cpu());
        assert!(!cpu.is_cuda());
        assert!(cuda.is_cuda());
        assert!(!cuda.is_cpu());
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn shape_rank_and_numel() {
        let s = Shape::from([2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dims(), &[2, 3, 4]);
    }

    #[test]
    fn scalar_shape_has_numel_one() {
        let s = Shape::from([]);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.rank(), 0);
        assert_eq!(s.dims(), &[]);
    }

    #[test]
    fn shape_max_rank_works() {
        let dims = [2; 8];
        let s = Shape::new(dims);
        assert_eq!(s.rank(), 8);
        assert_eq!(s.numel(), 256);
    }

    #[test]
    #[should_panic(expected = "Vearo only supports shapes up to rank 8")]
    fn shape_exceeding_max_rank_panics() {
        let dims = [1; 9];
        let _ = Shape::new(dims);
    }

    #[test]
    fn shape_indexing() {
        let s = Shape::from([5, 6, 7]);
        assert_eq!(s[0], 5);
        assert_eq!(s[1], 6);
        assert_eq!(s[2], 7);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn shape_index_out_of_bounds_panics() {
        let s = Shape::from([5, 6, 7]);
        let _ = s[3];
    }

    #[test]
    fn shape_iterator_and_ref_into_iterator() {
        let s = Shape::from([2, 4, 8]);
        let mut idx = 0;
        let expected = [2, 4, 8];

        for &dim in &s {
            assert_eq!(dim, expected[idx]);
            idx += 1;
        }
        assert_eq!(idx, 3);

        let collected: Vec<usize> = s.iter().copied().collect();
        assert_eq!(collected, vec![2, 4, 8]);
    }

    #[test]
    fn shape_contiguous_strides() {
        // Scalar strides
        let s_scalar = Shape::from([]);
        assert_eq!(s_scalar.contiguous_strides().dims(), &[]);

        // 1D strides
        let s_1d = Shape::from([7]);
        assert_eq!(s_1d.contiguous_strides().dims(), &[1]);

        // 2D strides
        let s_2d = Shape::from([3, 5]);
        assert_eq!(s_2d.contiguous_strides().dims(), &[5, 1]);

        // 3D strides
        let s_3d = Shape::from([2, 3, 4]);
        assert_eq!(s_3d.contiguous_strides().dims(), &[12, 4, 1]);

        // 4D strides
        let s_4d = Shape::from([2, 5, 3, 4]);
        assert_eq!(s_4d.contiguous_strides().dims(), &[60, 12, 4, 1]);
    }

    #[test]
    #[should_panic(expected = "Strides overflowed")]
    fn shape_strides_overflow_panics() {
        let s = Shape::from([usize::MAX, 2]);
        let _ = s.contiguous_strides();
    }

    #[test]
    fn test_reshape_non_contiguous() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let tt = t.transpose(0, 1); // shape [3, 2], strides [1, 3] (non-contiguous)
        let r = tt.reshape([6]); // should copy and read transposed logical order: 1, 4, 2, 5, 3, 6

        let guard = get_cpu_shard(r.storage_id().shard_idx as usize)
            .lock()
            .unwrap();
        match &guard.slots[r.storage_id().slot_idx as usize]
            .as_ref()
            .unwrap()
            .storage
        {
            CpuStorage::F32(vec) => assert_eq!(vec.as_ref(), &vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
            _ => unreachable!(),
        }
    }

    // Regression: an empty tensor (zero-size dim) taken through the
    // non-contiguous reshape -> contiguous() path must not panic and must stay
    // empty. Guards the `numel == 0` early return in contiguous().
    #[test]
    fn test_reshape_empty_tensor() {
        let t = Tensor::zeros([2, 0, 3], DType::F32);
        let r = t.transpose(0, 1).reshape([0, 6]);
        assert_eq!(r.shape().numel(), 0);
        assert_eq!(r.shape().dims(), &[0, 6]);
    }

    #[test]
    fn test_permute_correctness() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]);
        let p = t.permute([2, 0, 1]);
        assert_eq!(p.shape().dims(), &[3, 1, 2]);
        assert_eq!(p.strides().dims(), &[1, 6, 3]);
    }

    // Refcount *wiring* (clone/drop/view) tested on the global arena. Only
    // asserts the ref_count of a slot this test exclusively owns, so it is safe
    // under parallel execution. Free-list reuse lives in storage.rs on a local
    // arena (it races here because tests share the global CPU_ARENA_SHARDS).
    #[test]
    fn test_tensor_refcount_lifecycle() {
        let ref_count = |id: StorageId| {
            get_cpu_shard(id.shard_idx as usize).lock().unwrap().slots[id.slot_idx as usize]
                .as_ref()
                .map(|s| s.ref_count)
        };

        // Initial allocation -> refcount 1.
        let t1 = Tensor::zeros([3], DType::F32);
        let id1 = t1.storage_id();
        assert_eq!(ref_count(id1), Some(1));

        // Clone increases refcount.
        let t2 = t1.clone();
        assert_eq!(ref_count(id1), Some(2));

        // Drop decreases refcount.
        drop(t2);
        assert_eq!(ref_count(id1), Some(1));

        // A view (transpose) shares storage and increments refcount.
        let t3 = t1.transpose(0, 0);
        assert_eq!(t3.storage_id(), id1);
        assert_eq!(ref_count(id1), Some(2));

        // Dropping the parent keeps storage alive because the view still holds it.
        drop(t1);
        assert_eq!(ref_count(id1), Some(1));

        // Dropping the last owner releases this test's ref.
        drop(t3);
    }
}
