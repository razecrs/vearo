/// What kind of numbers are inside the tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// standard float
    F32,
    /// half float
    F16,
    /// bfloat16
    BF16,
    /// standard int
    I32,
    /// big int
    I64,
    /// boolean
    Bool,
}

impl DType {
    /// How many bytes one element takes up.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::Bool => 1,
        }
    }

    /// Checks if it's a float type that can hold gradients.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}
