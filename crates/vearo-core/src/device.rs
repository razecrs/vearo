/// Number of distinct compute backends Vearo dispatches over.
///
/// Registry index per backend: 0 = CPU, 1 = CUDA, 2 = Vulkan (reserved),
/// 3 = oneAPI (reserved). See [`Device::backend_idx`].
pub const NUM_BACKENDS: usize = 4;

/// Where the tensor memory actually lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Just the normal CPU.
    #[default]
    Cpu,
    /// A CUDA GPU with its index.
    Cuda(usize),
}

impl Device {
    /// Is this a CPU?
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Is this a CUDA GPU?
    #[must_use]
    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Index into the per-backend op registry for this device's backend.
    ///
    /// The CUDA ordinal is irrelevant here - every CUDA device shares one
    /// backend implementation.
    #[must_use]
    pub const fn backend_idx(self) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Cuda(_) => 1,
        }
    }
}
