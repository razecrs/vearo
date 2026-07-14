//! CPU storage arena and allocation logic.

use crate::dtype::DType;
use std::sync::Mutex;
use std::sync::OnceLock;

/// The number of shards in the sharded CPU Arena.
pub const NUM_SHARDS: usize = 32;

/// Unique identifier for a storage block in the sharded arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageId {
    /// The shard index this storage belongs to.
    pub shard_idx: u8,
    /// The slot index within that shard.
    pub slot_idx: u32,
}

/// Backing memory for CPU tensors.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    /// 32-bit float storage.
    F32(Vec<f32>),
    /// 16-bit float storage.
    F16(Vec<u16>),
    /// 16-bit brain float storage.
    BF16(Vec<u16>),
    /// 32-bit integer storage.
    I32(Vec<i32>),
    /// 64-bit integer storage.
    I64(Vec<i64>),
    /// Boolean storage.
    Bool(Vec<bool>),
}

/// A slot in a CPU storage arena shard.
#[derive(Debug, Clone)]
pub struct CpuStorageSlot {
    /// The actual data storage.
    pub storage: CpuStorage,
    /// Reference count of tensors pointing to this slot.
    pub ref_count: usize,
}

/// A single shard of the CPU Arena.
#[derive(Debug, Default)]
pub struct CpuArenaShard {
    /// Storage slots in the shard.
    pub slots: Vec<Option<CpuStorageSlot>>,
    /// Reusable slot indices.
    pub free_indices: Vec<u32>,
}

impl CpuArenaShard {
    /// Creates an empty `CpuArenaShard`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_indices: Vec::new(),
        }
    }

    /// Allocates a new zero-filled storage block in this shard.
    pub fn alloc(&mut self, numel: usize, dtype: DType, shard_idx: u8) -> StorageId {
        let storage = match dtype {
            DType::F32 => CpuStorage::F32(vec![0.0; numel]),
            DType::F16 => CpuStorage::F16(vec![0; numel]),
            DType::BF16 => CpuStorage::BF16(vec![0; numel]),
            DType::I32 => CpuStorage::I32(vec![0; numel]),
            DType::I64 => CpuStorage::I64(vec![0; numel]),
            DType::Bool => CpuStorage::Bool(vec![false; numel]),
        };
        self.alloc_raw(storage, shard_idx)
    }

    /// Inserts a pre-populated storage block in this shard.
    #[allow(clippy::cast_possible_truncation)]
    pub fn alloc_raw(&mut self, storage: CpuStorage, shard_idx: u8) -> StorageId {
        let slot = CpuStorageSlot {
            storage,
            ref_count: 1,
        };
        if let Some(idx) = self.free_indices.pop() {
            self.slots[idx as usize] = Some(slot);
            StorageId {
                shard_idx,
                slot_idx: idx,
            }
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Some(slot));
            StorageId {
                shard_idx,
                slot_idx: idx,
            }
        }
    }

    /// Increments reference count of a storage slot.
    pub fn increment(&mut self, slot_idx: u32) {
        if let Some(Some(slot)) = self.slots.get_mut(slot_idx as usize) {
            slot.ref_count += 1;
        }
    }

    /// Decrements reference count of a storage slot, freeing it if 0.
    pub fn decrement(&mut self, slot_idx: u32) {
        let mut free = false;
        if let Some(Some(slot)) = self.slots.get_mut(slot_idx as usize) {
            debug_assert!(slot.ref_count > 0, "Reference count underflow");
            slot.ref_count -= 1;
            if slot.ref_count == 0 {
                free = true;
            }
        }
        if free {
            self.slots[slot_idx as usize] = None;
            self.free_indices.push(slot_idx);
        }
    }
}

/// Global sharded CPU arena instance.
pub static CPU_ARENA_SHARDS: OnceLock<[Mutex<CpuArenaShard>; NUM_SHARDS]> = OnceLock::new();

/// Get a reference to a specific CPU arena shard.
#[must_use]
pub fn get_cpu_shard(shard_idx: usize) -> &'static Mutex<CpuArenaShard> {
    let shards =
        CPU_ARENA_SHARDS.get_or_init(|| std::array::from_fn(|_| Mutex::new(CpuArenaShard::new())));
    &shards[shard_idx]
}

/// Helper to get the shard index for the current thread.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn current_thread_shard_idx() -> u8 {
    thread_local! {
        static SHARD_IDX: u8 = {
            static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let val = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (val % NUM_SHARDS) as u8
        };
    }
    SHARD_IDX.with(|&idx| idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_free_and_slot_reuse() {
        let mut arena = CpuArenaShard::new();

        let a = arena.alloc(3, DType::F32, 0);
        let b = arena.alloc(2, DType::F32, 0);
        assert_eq!(
            a,
            StorageId {
                shard_idx: 0,
                slot_idx: 0
            }
        );
        assert_eq!(
            b,
            StorageId {
                shard_idx: 0,
                slot_idx: 1
            }
        );
        assert_eq!(
            arena.slots[a.slot_idx as usize].as_ref().unwrap().ref_count,
            1
        );

        arena.increment(a.slot_idx);
        assert_eq!(
            arena.slots[a.slot_idx as usize].as_ref().unwrap().ref_count,
            2
        );
        arena.decrement(a.slot_idx);
        assert_eq!(
            arena.slots[a.slot_idx as usize].as_ref().unwrap().ref_count,
            1
        );

        arena.decrement(a.slot_idx);
        assert!(arena.slots[a.slot_idx as usize].is_none());
        assert_eq!(arena.free_indices, vec![a.slot_idx]);

        let c = arena.alloc(5, DType::F32, 0);
        assert_eq!(c, a);
        assert!(arena.free_indices.is_empty());

        assert_eq!(
            arena.slots[b.slot_idx as usize].as_ref().unwrap().ref_count,
            1
        );
    }

    #[test]
    fn arena_alloc_raw_preserves_data() {
        let mut arena = CpuArenaShard::new();
        let id = arena.alloc_raw(CpuStorage::F32(vec![1.0, 2.0, 3.0]), 0);
        match &arena.slots[id.slot_idx as usize].as_ref().unwrap().storage {
            CpuStorage::F32(v) => assert_eq!(v, &[1.0, 2.0, 3.0]),
            _ => unreachable!(),
        }
    }
}
