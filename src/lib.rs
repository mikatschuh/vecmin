//! Provides a `VecMin` and `VecOne` newtype wrapper around `Vec` that enforces a minimum length at compile time.
//! Also provides the `ArrVecMin` wrapper that has the same guaranties as `VecMin` but stores the guarantied elements on the stack.

// This file was modified by mikatschuh to add a corresponding `arrvecmin` macro

#![no_std]

pub mod inline;
pub mod vec;

extern crate alloc;

#[doc(hidden)]
pub extern crate alloc as __alloc;

use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::ops::{Bound, Range, RangeBounds, RangeTo};

pub use inline::ArrVecMin;
pub use vec::{VecMin, VecOne};

/// An error indicating that an operation would reduce the length of a vector below its minimum required length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModifyError<const M: usize>;

impl<const M: usize> Display for ModifyError<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "operation would reduce length below minimum required {}",
            M
        )
    }
}

impl<const M: usize> Error for ModifyError<M> {}

#[inline]
#[track_caller]
fn slice_range<R>(range: &R, bounds: RangeTo<usize>) -> Range<usize>
where
    R: RangeBounds<usize>,
{
    let len = bounds.end;

    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start
            .checked_add(1)
            .expect("attempted to index slice from after maximum usize"),
        Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        Bound::Included(&end) => end
            .checked_add(1)
            .expect("attempted to index slice up to maximum usize"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };

    assert!(
        start <= end,
        "slice index starts at {start} but ends at {end}"
    );
    assert!(
        end <= len,
        "range end index {end} out of range for slice of length {len}"
    );

    Range { start, end }
}

/// Creates a [`VecOne`] containing the arguments.
///
/// `vecone!` allows `VecOne`s to be defined with similar syntax to `vec!`, but with a minimum length of 1.
/// The length must be constant, for non-constant length use checked constructors.
#[macro_export]
macro_rules! vecone {
    ($($x:expr),+ $(,)?) => {
        $crate::vecmin![1; [$($x),+]]
    };
    ($elem:expr; $n:expr) => {
        $crate::vecmin![1; [$elem; $n]]
    };
}

/// Creates a [`VecMin`] containing the arguments.
///
/// There is a minimum length argument preceding the list, if not included the minimum is inferred as the length.
/// The length must be constant, for non-constant length use checked constructors.
///
/// - Create a [`VecMin`] with a fixed minimum length:
/// ```
/// use vecmin::vecmin;
///
/// let v = vecmin![5; [1; 6]];
/// assert!(!v.is_minimum());
/// assert_eq!(v.minimum(), 5);
///
/// let v = vecmin![2; [1, 2, 3]];
/// assert!(!v.is_minimum());
/// assert_eq!(v.minimum(), 2);
/// ```
///
/// - Create a [`VecMin`] with a minimum length inferred from the number of elements:
/// ```
/// use vecmin::vecmin;
///
/// let v = vecmin![1; 6];
/// assert!(v.is_minimum());
/// assert_eq!(v.minimum(), 6);
///
/// let v = vecmin![1, 2, 3];
/// assert!(v.is_minimum());
/// assert_eq!(v.minimum(), 3);
/// ```
#[macro_export]
macro_rules! vecmin {
    ($min:expr; [$x:expr; $n:expr]) => {{
        let _: [(); $n - $min];
        let vec = $crate::__alloc::vec![$x; $n];
        // Safety: The compile-time length check guarantees at least `$min` elements.
        unsafe { $crate::VecMin::<_, $min>::from_vec_unchecked(vec) }
    }};
    ($min:expr; [$($x:expr),+ $(,)?]) => {{
        const N: usize = <[()]>::len(&[$( { let _ = &$x; }),+]);
        let _: [(); N - $min];
        let vec = $crate::__alloc::vec![$($x),+];
        // Safety: The compile-time length check guarantees at least `$min` elements.
        unsafe { $crate::VecMin::<_, $min>::from_vec_unchecked(vec) }
    }};
    ($x:expr; $n:expr) => {
        $crate::VecMin::from_array([$x; $n])
    };
    ($($x:expr),+ $(,)?) => {
        $crate::VecMin::from_array([$($x),+])
    };
}

/// Creates an [`ArrVecMin`] containing the arguments.
///
/// Uses the same syntax as [`vecmin!`]: the minimum is inferred from the number
/// of elements unless an explicit minimum precedes a bracketed list. Lengths
/// and explicit minima must be constant. Elements are evaluated once, in order.
/// Only elements beyond the minimum require a heap allocation.
///
/// ```
/// use vecmin::{ArrVecMin, arrvecmin};
///
/// let inferred = arrvecmin![1, 2, 3];
/// assert_eq!(inferred.minimum(), 3);
/// assert!(inferred.is_minimum());
///
/// let explicit = arrvecmin![2; [1, 2, 3]];
/// assert_eq!(explicit.split_at_min(), (&[1, 2], &[3][..]));
///
/// let repeated = arrvecmin![2; [String::from("hello"); 4]];
/// assert_eq!(repeated.len(), 4);
///
/// const INLINE: ArrVecMin<i32, 3> = arrvecmin![7; 3];
/// assert_eq!(INLINE.min_slice(), &[7, 7, 7]);
///
/// let empty: ArrVecMin<i32, 0> = arrvecmin![];
/// assert!(empty.is_empty());
/// ```
///
/// A length below the minimum is rejected at compile time:
///
/// ```compile_fail
/// use vecmin::arrvecmin;
/// let _ = arrvecmin![3; [1, 2]];
/// ```
///
/// ```compile_fail
/// use vecmin::arrvecmin;
/// let _ = arrvecmin![3; [1; 2]];
/// ```
#[macro_export]
macro_rules! arrvecmin {
    ($min:expr; [$x:expr; $n:expr]) => {{
        let _: [(); $n - $min];
        $crate::ArrVecMin::<_, { $min }>::collect(::core::iter::repeat_n($x, $n))
            .unwrap_or_else(|_| ::core::unreachable!("length is checked at compile time"))
    }};
    ($min:expr; [$($x:expr),* $(,)?]) => {{
        let _: [(); <[()]>::len(&[$({ let _ = ::core::stringify!($x); }),*]) - $min];
        $crate::ArrVecMin::<_, { $min }>::collect([$($x),*])
            .unwrap_or_else(|_| ::core::unreachable!("length is checked at compile time"))
    }};
    ($x:expr; $n:expr) => {
        $crate::ArrVecMin::new([$x; $n])
    };
    ($($x:expr),* $(,)?) => {
        $crate::ArrVecMin::new([$($x),*])
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn slice_range() {
        use super::slice_range;

        assert_eq!(slice_range(&(..), ..5), 0..5);
        assert_eq!(slice_range(&(1..), ..5), 1..5);
        assert_eq!(slice_range(&(..3), ..5), 0..3);
        assert_eq!(slice_range(&(1..3), ..5), 1..3);
        assert_eq!(slice_range(&(1..=3), ..5), 1..4);
    }

    #[test]
    fn vecone() {
        let v = vecone![1, 1, 1];

        assert_eq!(v.minimum(), 1);
        assert_eq!(v.len(), 3);

        let v = vecone![1; 3];

        assert_eq!(v.minimum(), 1);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn vecmin_explicit() {
        let v = vecmin![2; [1, 1, 1]];

        assert_eq!(v.minimum(), 2);
        assert_eq!(v.len(), 3);

        let v = vecmin![2; [1; 3]];

        assert_eq!(v.minimum(), 2);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn vecmin_implicit() {
        let v = vecmin![1, 1, 1];

        assert_eq!(v.minimum(), 3);
        assert_eq!(v.len(), 3);

        let v = vecmin![1; 3];

        assert_eq!(v.minimum(), 3);
        assert_eq!(v.len(), 3);
    }

    // ---- compile errors ----
    // fn lt_min() {
    //     let v = vecone![];
    //     let v = vecone![1; 0];
    //     let v = vecmin![5; [1, 1, 1, 1]];
    //     let v = vecmin![5; [1; 4]];
    // }

    // fn nonconst() {
    //     let n = 0;
    //     let v = vecone![2; n];
    //     let v = vecmin![2; [1; n]];
    // }
}
