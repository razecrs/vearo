/// Max dimensions we support before we panic.
pub const MAX_RANK: usize = 8;

/// Tensor dimensions, outermost first.
///
/// Stack-allocated up to rank 8 to avoid heap allocations.
/// Empty shape `[]` is just a scalar with 1 element.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    dims: [usize; MAX_RANK],
    rank: usize,
}

impl Default for Shape {
    fn default() -> Self {
        Self {
            dims: [0; MAX_RANK],
            rank: 0,
        }
    }
}

impl Shape {
    /// Make a new shape from a slice of dims.
    ///
    /// # Panics
    /// Panics if the rank is greater than 8.
    #[must_use]
    pub fn new(dims: impl AsRef<[usize]>) -> Self {
        let dims_slice = dims.as_ref();
        assert!(
            dims_slice.len() <= MAX_RANK,
            "Vearo only supports shapes up to rank {}, but got rank {}",
            MAX_RANK,
            dims_slice.len()
        );
        let mut array = [0; MAX_RANK];
        array[..dims_slice.len()].copy_from_slice(dims_slice);
        Self {
            dims: array,
            rank: dims_slice.len(),
        }
    }

    /// Get dims as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }

    /// How many dimensions.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims().iter().product()
    }

    /// Computes row-major strides.
    ///
    /// # Panics
    /// Panics if stride calculation overflows.
    #[must_use]
    pub fn contiguous_strides(&self) -> Self {
        let mut strides = [0; MAX_RANK];
        let mut acc: usize = 1;
        for i in (0..self.rank).rev() {
            strides[i] = acc;
            acc = acc.checked_mul(self.dims[i]).expect("Strides overflowed");
        }
        Self {
            dims: strides,
            rank: self.rank,
        }
    }

    /// Checks if this is a scalar shape (rank 0).
    #[must_use]
    pub const fn is_scalar(&self) -> bool {
        self.rank == 0
    }

    /// Broadcasts two shapes together to find their common broadcasted shape.
    ///
    /// Follows standard NumPy/PyTorch broadcasting rules. Returns `None` if incompatible.
    #[must_use]
    pub fn broadcast(&self, other: &Self) -> Option<Self> {
        let max_rank = std::cmp::max(self.rank, other.rank);
        if max_rank > MAX_RANK {
            return None;
        }
        let mut result_dims = [0; MAX_RANK];
        for i in 0..max_rank {
            let self_dim = if i < self.rank {
                self.dims[self.rank - 1 - i]
            } else {
                1
            };
            let other_dim = if i < other.rank {
                other.dims[other.rank - 1 - i]
            } else {
                1
            };

            let out_dim = if self_dim == other_dim {
                self_dim
            } else if self_dim == 1 {
                other_dim
            } else if other_dim == 1 {
                self_dim
            } else {
                return None;
            };
            result_dims[max_rank - 1 - i] = out_dim;
        }
        Some(Self {
            dims: result_dims,
            rank: max_rank,
        })
    }

    /// Checks if this shape can be broadcast to the target shape.
    ///
    /// Returns `true` if it can be broadcast to `target`.
    #[must_use]
    pub fn can_broadcast_to(&self, target: &Self) -> bool {
        if self.rank > target.rank {
            return false;
        }
        for i in 0..self.rank {
            let self_dim = self.dims[self.rank - 1 - i];
            let target_dim = target.dims[target.rank - 1 - i];
            if self_dim != target_dim && self_dim != 1 {
                return false;
            }
        }
        true
    }

    /// Swaps dimensions i and j on the stack.
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    #[must_use]
    pub fn swapped(&self, i: usize, j: usize) -> Self {
        assert!(i < self.rank && j < self.rank, "Swap index out of bounds");
        let mut new_shape = *self;
        new_shape.dims.swap(i, j);
        new_shape
    }

    /// Iterates over the dimensions.
    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.dims().iter()
    }
}

impl<'a> IntoIterator for &'a Shape {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self::new(v)
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(a: [usize; N]) -> Self {
        Self::new(a)
    }
}

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Self {
        Self::new(s)
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        self.dims()
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims()[index]
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.dims() == other.dims()
    }
}

impl Eq for Shape {}

impl std::hash::Hash for Shape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dims().hash(state);
    }
}

/// Iterator over N-dimensional coordinate space.
pub struct NdIterator {
    shape: Shape,
    coord: [usize; 8],
    done: bool,
}

impl NdIterator {
    /// Creates a new iterator for the given shape.
    #[must_use]
    pub fn new(shape: Shape) -> Self {
        let done = shape.numel() == 0;
        Self {
            shape,
            coord: [0; 8],
            done,
        }
    }

    /// Access the current coordinate slice.
    #[must_use]
    pub fn coord(&self) -> &[usize] {
        &self.coord[..self.shape.rank()]
    }

    /// Advances the iterator. Returns `true` if advanced successfully, `false` if finished.
    pub fn step(&mut self) -> bool {
        if self.done {
            return false;
        }
        let rank = self.shape.rank();
        if rank == 0 {
            self.done = true;
            return false;
        }
        let mut idx = rank - 1;
        loop {
            self.coord[idx] += 1;
            if self.coord[idx] < self.shape[idx] {
                break;
            }
            self.coord[idx] = 0;
            if idx == 0 {
                self.done = true;
                return false;
            }
            idx -= 1;
        }
        true
    }
}

/// Map a high-dimensional output coordinate to a flat storage offset in a tensor.
#[must_use]
pub fn get_offset(coord: &[usize], shape: &Shape, strides: &Shape) -> usize {
    let mut offset = 0;
    let rank = shape.rank();
    let coord_offset = coord.len().saturating_sub(rank);
    for i in 0..rank {
        let dim_in = shape[i];
        let c = coord[coord_offset + i];
        let coord_in = if dim_in == 1 { 0 } else { c };
        offset += coord_in * strides[i];
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn test_custom_eq_and_hash_ignores_tail_garbage() {
        let s1 = Shape {
            dims: [2, 3, 0, 0, 0, 0, 0, 0],
            rank: 2,
        };
        let s2 = Shape {
            dims: [2, 3, 99, 99, 99, 99, 99, 99],
            rank: 2,
        };

        assert_eq!(s1, s2);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        s1.hash(&mut hasher1);
        s2.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_is_scalar() {
        assert!(Shape::from([]).is_scalar());
        assert!(!Shape::from([1]).is_scalar());
        assert!(!Shape::from([2, 3]).is_scalar());
    }

    #[test]
    fn test_broadcasting() {
        let s1 = Shape::from([3, 1]);
        let s2 = Shape::from([2, 3, 4]);
        let b = s1.broadcast(&s2).unwrap();
        assert_eq!(b.dims(), &[2, 3, 4]);

        let s3 = Shape::from([2, 1]);
        let s4 = Shape::from([3, 4]);
        assert!(s3.broadcast(&s4).is_none());

        let scalar = Shape::from([]);
        let s5 = Shape::from([5, 6]);
        assert_eq!(scalar.broadcast(&s5).unwrap().dims(), &[5, 6]);
        assert_eq!(s5.broadcast(&scalar).unwrap().dims(), &[5, 6]);
    }

    #[test]
    fn test_can_broadcast_to() {
        let s1 = Shape::from([3, 1]);
        let target = Shape::from([2, 3, 4]);
        assert!(s1.can_broadcast_to(&target));

        let s2 = Shape::from([2, 3]);
        assert!(!s2.can_broadcast_to(&target));
    }

    #[test]
    fn test_randomized_broadcasting_properties() {
        use std::num::Wrapping;
        let mut seed = Wrapping(123_456_789usize);
        let mut next_rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed.0
        };

        for _ in 0..5000 {
            let rank_a = next_rand() % 9;
            let mut dims_a = Vec::with_capacity(rank_a);
            for _ in 0..rank_a {
                let dim = if next_rand() % 3 == 1 {
                    2 + (next_rand() % 10)
                } else {
                    1
                };
                dims_a.push(dim);
            }
            let s_a = Shape::new(&dims_a);

            let rank_b = next_rand() % 9;
            let mut dims_b = Vec::with_capacity(rank_b);
            for _ in 0..rank_b {
                let dim = if next_rand() % 3 == 1 {
                    2 + (next_rand() % 10)
                } else {
                    1
                };
                dims_b.push(dim);
            }
            let s_b = Shape::new(&dims_b);

            if let Some(s_c) = s_a.broadcast(&s_b) {
                assert!(s_c.rank() >= s_a.rank());
                assert!(s_c.rank() >= s_b.rank());
                assert!(s_c.rank() <= MAX_RANK);
                assert!(s_a.can_broadcast_to(&s_c));
                assert!(s_b.can_broadcast_to(&s_c));

                let s_c_alt = s_b.broadcast(&s_a).unwrap();
                assert_eq!(s_c, s_c_alt);

                assert_eq!(s_c.numel() % s_a.numel(), 0);
                assert_eq!(s_c.numel() % s_b.numel(), 0);
            } else {
                let mut conflict = false;
                let min_rank = std::cmp::min(s_a.rank(), s_b.rank());
                for i in 0..min_rank {
                    let lhs = s_a.dims()[s_a.rank() - 1 - i];
                    let rhs = s_b.dims()[s_b.rank() - 1 - i];
                    if lhs != rhs && lhs != 1 && rhs != 1 {
                        conflict = true;
                        break;
                    }
                }
                assert!(conflict);
            }
        }
    }

    #[test]
    fn test_randomized_strides_correctness() {
        use std::num::Wrapping;
        let mut seed = Wrapping(987_654_321usize);
        let mut next_rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed.0
        };

        for _ in 0..1000 {
            let rank = next_rand() % 9;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(1 + (next_rand() % 10));
            }
            let s = Shape::new(&dims);
            let strides = s.contiguous_strides();

            assert_eq!(strides.rank(), s.rank());

            let mut expected_strides = vec![0; rank];
            let mut current = 1;
            for i in (0..rank).rev() {
                expected_strides[i] = current;
                current *= dims[i];
            }
            assert_eq!(strides.dims(), &expected_strides[..]);
        }
    }
}
