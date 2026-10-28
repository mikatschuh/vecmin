//! Implementation of the [`ArrVecMin`] type.

// This code was added by mikatschuh

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::collections::TryReserveError;
use alloc::vec::{self, Vec};
use core::array;
use core::cmp::Ordering;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::Chain;
use core::mem::MaybeUninit;
use core::ops::{Index, IndexMut, RangeBounds};
use core::slice;

use crate::{ModifyError, slice_range};

/// A vector with a minimum length of `M`.
///
/// Most methods of `Vec` are available on `ArrVecMin` except those that reduce the length of the vector an unknown amount.
/// Methods that reduce the length of the vector by a known amount (e.g. `remove`, `truncate`) are available on `ArrVecMin`
/// but return an error if the operation would reduce the length of the vector below `M`.
///
/// Unlike [`crate::VecMin`], this implementation stores its first `M` elements inline.
/// The remaining elements are stored in a separate heap allocation. Consequently,
/// it cannot expose a single slice of all elements; use [`Self::split_at_min`],
/// [`Self::iter`], or [`Self::iter_mut`] instead.
#[derive(Clone, Debug)]
pub struct ArrVecMin<T, const M: usize> {
    arr: [T; M],
    vec: Vec<T>,
}

// --- Custom ---
impl<T, const M: usize> ArrVecMin<T, M> {
    /// Returns the minimum length of the vector.
    #[inline]
    pub const fn minimum(&self) -> usize {
        M
    }

    /// Returns `true` if the length of the vector is equal to the minimum length `M`.
    #[inline]
    pub const fn is_minimum(&self) -> bool {
        self.vec.is_empty()
    }

    /// Returns a slice to the first `M` elements of the vector, which are guaranteed to exist.
    #[inline]
    pub const fn min_slice(&self) -> &[T; M] {
        &self.arr
    }

    /// Returns a mutable slice to the first `M` elements of the vector, which are guaranteed to exist.
    #[inline]
    pub const fn min_slice_mut(&mut self) -> &mut [T; M] {
        &mut self.arr
    }

    /// Returns the element at `I` within the guaranteed minimum.
    ///
    /// `I < M` is checked at compile time when this method is instantiated.
    ///
    /// ```
    /// use vecmin::ArrVecMin;
    /// let values = ArrVecMin::<_, 2>::from_array([10, 20, 30]);
    /// assert_eq!(*values.get_min::<1>(), 20);
    /// ```
    ///
    /// ```compile_fail
    /// use vecmin::ArrVecMin;
    /// let values = ArrVecMin::<_, 2>::from_array([10, 20, 30]);
    /// let _ = values.get_min::<2>();
    /// ```
    #[inline]
    pub const fn get_min<const I: usize>(&self) -> &T {
        const { assert!(I < M, "index must be within the guaranteed minimum") };
        &self.min_slice()[I]
    }

    /// Returns a mutable element at `I` within the guaranteed minimum.
    ///
    /// `I < M` is checked at compile time when this method is instantiated.
    ///
    /// ```compile_fail
    /// use vecmin::ArrVecMin;
    /// let mut values = ArrVecMin::<_, 2>::from_array([10, 20, 30]);
    /// let _ = values.get_min_mut::<2>();
    /// ```
    #[inline]
    pub const fn get_min_mut<const I: usize>(&mut self) -> &mut T {
        const { assert!(I < M, "index must be within the guaranteed minimum") };
        &mut self.min_slice_mut()[I]
    }

    /// Returns a tuple of a slice to the first `M` elements of the vector, which are guaranteed to exist, and a slice to the remaining elements of the vector.
    #[inline]
    pub const fn split_at_min(&self) -> (&[T; M], &[T]) {
        (&self.arr, self.vec.as_slice())
    }

    /// Returns a tuple of a mutable slice to the first `M` elements of the vector, which are guaranteed to exist, and a mutable slice to the remaining elements of the vector.
    #[inline]
    pub const fn split_at_min_mut(&mut self) -> (&mut [T; M], &mut [T]) {
        (&mut self.arr, self.vec.as_mut_slice())
    }
}

// --- Constructors, Convertors, and Destructors ---
/// A vector that is too short to be a valid `ArrVecMin` returned as an error in a constructor.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstructError<T, const M: usize>(pub Vec<T>);

impl<T: Debug, const M: usize> Display for ConstructError<T, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Length {} of {:?} is less than the minimum {}",
            self.0.len(),
            self.0,
            M
        )
    }
}

impl<T: Debug, const M: usize> Error for ConstructError<T, M> {}

impl<T, const M: usize> ArrVecMin<T, M> {
    /// Creates a new `ArrVecMin` from an array containing the minimum elements.
    #[inline]
    pub const fn new(min: [T; M]) -> Self {
        Self {
            arr: min,
            vec: Vec::new(),
        }
    }

    /// Creates an `ArrVecMin` from an array with at least `M` elements.
    ///
    /// `N >= M` is checked at compile time when this constructor is instantiated.
    /// Use [`Self::try_new`] for a conversion that returns an error instead.
    ///
    /// ```
    /// use vecmin::ArrVecMin;
    /// let values = ArrVecMin::<_, 2>::from_array([1, 2, 3]);
    /// assert_eq!(values.len(), 3);
    /// assert_eq!(values.minimum(), 2);
    /// ```
    ///
    /// ```compile_fail
    /// use vecmin::ArrVecMin;
    /// let _ = ArrVecMin::<_, 3>::from_array([1, 2]);
    /// ```
    #[inline]
    pub fn from_array<const N: usize>(arr: [T; N]) -> Self {
        const { assert!(N >= M, "array is shorter than the minimum") };
        let mut iter = arr.into_iter();
        let arr = array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() });
        let vec = iter.collect();
        Self { arr, vec }
    }

    /// Creates a new `ArrVecMin` from a `Vec`
    ///
    /// # Safety
    /// - Given a `Vec` as input with length less than `M` is undefined behavior
    #[inline]
    pub unsafe fn from_vec_unchecked(mut vec: Vec<T>) -> Self {
        let arr = {
            let mut prefix = vec.drain(..M);
            array::from_fn(|_| unsafe { prefix.next().unwrap_unchecked() })
        };
        Self { arr, vec }
    }

    /// Creates a new `ArrVecMin` from a `Vec`, returning an error if the length of the provided `Vec` is less than `M`.
    #[inline]
    pub fn try_from_vec(vec: Vec<T>) -> Result<Self, ConstructError<T, M>> {
        if vec.len() >= M {
            // Safety: We just checked that the length was at least `M`.
            Ok(unsafe { Self::from_vec_unchecked(vec) })
        } else {
            Err(ConstructError(vec))
        }
    }

    /// Creates a new `ArrVecMin` from anything that can be converted into a `Vec`, returning an error if the length of the provided `Vec` is less than `M`.
    #[inline]
    pub fn try_new(vec: impl Into<Vec<T>>) -> Result<Self, ConstructError<T, M>> {
        Self::try_from_vec(vec.into())
    }

    /// Creates a new `ArrVecMin` from an iterator, returning an error if the length of the collected `Vec` is less than `M`.
    #[inline]
    pub fn collect(iter: impl IntoIterator<Item = T>) -> Result<Self, ConstructError<T, M>> {
        let iter = iter.into_iter();
        let (low, high) = iter.size_hint();

        Self::collect_with_capacity(iter, high.unwrap_or(low).max(M))
    }

    /// Creates a new `ArrVecMin` from an iterator, returning an error if the length of the collected `Vec` is less than `M`.
    /// Reserves space for at least `capacity` total elements, including the inline minimum.
    #[inline]
    pub fn collect_with_capacity(
        iter: impl IntoIterator<Item = T>,
        capacity: usize,
    ) -> Result<Self, ConstructError<T, M>> {
        let mut iter = iter.into_iter().fuse();
        let prefix: [_; M] = array::from_fn(|_| iter.next());
        if prefix.iter().any(Option::is_none) {
            return Err(ConstructError(prefix.into_iter().flatten().collect()));
        }
        let arr = prefix.map(|item| unsafe { item.unwrap_unchecked() });
        let mut vec = Vec::with_capacity(capacity.saturating_sub(M));
        vec.extend(iter);
        Ok(Self { arr, vec })
    }

    /// Collects all elements into a contiguous `Vec`, consuming the `ArrVecMin`.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.into_iter().collect()
    }

    /// See [`Vec::into_boxed_slice`].
    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.into_vec().into_boxed_slice()
    }

    #[inline]
    pub fn clone_take_into_iter(
        &mut self,
    ) -> Chain<core::array::IntoIter<T, M>, alloc::vec::IntoIter<T>>
    where
        T: Clone,
    {
        self.arr
            .clone()
            .into_iter()
            .chain(core::mem::take(&mut self.vec))
    }

    /// See [`Vec::leak`].
    #[inline]
    pub fn leak(self) -> &'static mut [T] {
        self.into_vec().leak()
    }
}

impl<T: Default, const M: usize> Default for ArrVecMin<T, M> {
    #[inline]
    fn default() -> Self {
        Self::new(array::from_fn(|_| T::default()))
    }
}

impl<T, const M: usize> TryFrom<Vec<T>> for ArrVecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        Self::try_new(vec)
    }
}

impl<T, const M: usize> From<ArrVecMin<T, M>> for Vec<T> {
    #[inline]
    fn from(vec_min: ArrVecMin<T, M>) -> Self {
        vec_min.into_vec()
    }
}

impl<T, const M: usize> TryFrom<Box<[T]>> for ArrVecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(boxed_slice: Box<[T]>) -> Result<Self, Self::Error> {
        Self::try_new(boxed_slice)
    }
}

impl<T, const M: usize> From<ArrVecMin<T, M>> for Box<[T]> {
    #[inline]
    fn from(vec_min: ArrVecMin<T, M>) -> Self {
        vec_min.into_boxed_slice()
    }
}

impl<T: Clone, const M: usize> TryFrom<&[T]> for ArrVecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        Self::try_new(slice)
    }
}

impl<T: Clone, const M: usize> TryFrom<&mut [T]> for ArrVecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(slice: &mut [T]) -> Result<Self, Self::Error> {
        Self::try_new(slice)
    }
}

impl<T: Clone, const M: usize> TryFrom<Cow<'_, [T]>> for ArrVecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(cow: Cow<'_, [T]>) -> Result<Self, Self::Error> {
        Self::try_new(cow)
    }
}

impl<T, const N: usize, const M: usize> TryFrom<[T; N]> for ArrVecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(array: [T; N]) -> Result<Self, Self::Error> {
        Self::try_new(array)
    }
}

impl<T: Clone, const N: usize, const M: usize> TryFrom<&[T; N]> for ArrVecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(array: &[T; N]) -> Result<Self, Self::Error> {
        Self::try_new(array)
    }
}

impl<T: Clone, const N: usize, const M: usize> TryFrom<&mut [T; N]> for ArrVecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(array: &mut [T; N]) -> Result<Self, Self::Error> {
        Self::try_new(array)
    }
}

impl<T, const N: usize, const M: usize> TryFrom<ArrVecMin<T, M>> for [T; N] {
    type Error = ArrVecMin<T, M>;

    #[inline]
    fn try_from(vec_min: ArrVecMin<T, M>) -> Result<[T; N], Self::Error> {
        if vec_min.len() != N {
            return Err(vec_min);
        }
        Ok(array::from_fn({
            let mut iter = vec_min.into_iter();
            move |_| unsafe { iter.next().unwrap_unchecked() }
        }))
    }
}

// --- View ---
impl<T, const M: usize> ArrVecMin<T, M> {
    /// Returns an element at a logical index spanning both storage regions.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < M {
            Some(unsafe { self.arr.get_unchecked(index) })
        } else {
            self.vec.get(index - M)
        }
    }

    /// Returns a mutable element at a logical index spanning both storage regions.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < M {
            Some(unsafe { self.arr.get_unchecked_mut(index) })
        } else {
            self.vec.get_mut(index - M)
        }
    }

    /// Returns a reference to the first element.
    ///
    /// Requires `M > 0`, checked at compile time when this method is instantiated.
    ///
    /// ```compile_fail
    /// use vecmin::ArrVecMin;
    /// let values = ArrVecMin::<_, 0>::from_array([1, 2]);
    /// let _ = values.first();
    /// ```
    #[inline]
    pub const fn first(&self) -> &T {
        const { assert!(M != 0, "first requires a nonzero minimum") };
        &self.min_slice()[0]
    }

    /// Returns a mutable reference to the first element.
    ///
    /// Requires `M > 0`, checked at compile time when this method is instantiated.
    ///
    /// ```compile_fail
    /// use vecmin::ArrVecMin;
    /// let mut values = ArrVecMin::<_, 0>::from_array([1, 2]);
    /// let _ = values.first_mut();
    /// ```
    #[inline]
    pub const fn first_mut(&mut self) -> &mut T {
        const { assert!(M != 0, "first_mut requires a nonzero minimum") };
        &mut self.min_slice_mut()[0]
    }

    /// Returns a reference to the last element.
    ///
    /// Requires `M` to not be zero.
    ///
    /// ```compile_fail
    /// use vecmin::ArrVecMin;
    /// let values = ArrVecMin::<i32, 0>::new([]);
    /// let _ = values.last();
    /// ```
    #[inline]
    pub fn last(&self) -> &T {
        const {
            assert!(M != 0, "last requires a nonzero minimum");
        }
        self.vec
            .last()
            .unwrap_or_else(|| unsafe { self.arr.last().unwrap_unchecked() })
    }

    /// Returns a mutable reference to the last element.
    ///
    /// Requires `M` to not be zero.
    ///
    /// ```compile_fail
    /// use vecmin::ArrVecMin;
    /// let mut values = ArrVecMin::<i32, 0>::new([]);
    /// let _ = values.last_mut();
    /// ```
    #[inline]
    pub fn last_mut(&mut self) -> &mut T {
        const {
            assert!(M != 0, "last requires a nonzero minimum");
        }
        self.vec
            .last_mut()
            .unwrap_or_else(|| unsafe { self.arr.last_mut().unwrap_unchecked() })
    }

    /// Returns the uninitialized spare capacity of the heap tail.
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.vec.spare_capacity_mut()
    }

    /// Swaps two elements, allowing indices in either storage region.
    pub const fn swap(&mut self, a: usize, b: usize) {
        assert!(a < self.len() && b < self.len(), "index out of bounds");
        match (a < M, b < M) {
            (true, true) => self.arr.swap(a, b),
            (false, false) => self.split_at_min_mut().1.swap(a - M, b - M),
            (true, false) => {
                let (arr, vec) = self.split_at_min_mut();
                core::mem::swap(&mut arr[a], &mut vec[b - M])
            }
            (false, true) => {
                let (arr, vec) = self.split_at_min_mut();
                core::mem::swap(&mut vec[a - M], &mut arr[b])
            }
        }
    }

    const fn reverse_range(&mut self, start: usize, end: usize) {
        let mut offset = 0;
        while offset < (end - start) / 2 {
            self.swap(start + offset, end - 1 - offset);
            offset += 1;
        }
    }
}

impl<T, const M: usize> Index<usize> for ArrVecMin<T, M> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.get(index).expect("index out of bounds")
    }
}

impl<T, const M: usize> IndexMut<usize> for ArrVecMin<T, M> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).expect("index out of bounds")
    }
}

// --- Iterators ---
impl<T, const M: usize> IntoIterator for ArrVecMin<T, M> {
    type Item = T;
    type IntoIter = Chain<array::IntoIter<T, M>, vec::IntoIter<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.arr.into_iter().chain(self.vec)
    }
}

impl<'a, T: 'a, const M: usize> IntoIterator for &'a ArrVecMin<T, M> {
    type Item = &'a T;
    type IntoIter = Chain<slice::Iter<'a, T>, slice::Iter<'a, T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: 'a, const M: usize> IntoIterator for &'a mut ArrVecMin<T, M> {
    type Item = &'a mut T;
    type IntoIter = Chain<slice::IterMut<'a, T>, slice::IterMut<'a, T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const M: usize> ArrVecMin<T, M> {
    /// Iterates over the inline elements followed by the heap tail.
    pub fn iter(&self) -> Chain<slice::Iter<'_, T>, slice::Iter<'_, T>> {
        self.arr.iter().chain(self.vec.iter())
    }

    /// Mutably iterates over the inline elements followed by the heap tail.
    pub fn iter_mut(&mut self) -> Chain<slice::IterMut<'_, T>, slice::IterMut<'_, T>> {
        self.arr.iter_mut().chain(self.vec.iter_mut())
    }

    /// Returns whether there are no elements (possible only when `M` is zero).
    pub const fn is_empty(&self) -> bool {
        M == 0 && self.vec.is_empty()
    }
}

// --- Immutable Access ---
impl<T, const M: usize> ArrVecMin<T, M> {
    /// Returns the total capacity, including the `M` inline elements.
    /// Saturates at `usize::MAX` for zero-sized elements.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.vec.capacity().saturating_add(M)
    }

    /// See [`Vec::len`].
    #[inline]
    pub const fn len(&self) -> usize {
        self.vec.len() + M
    }
}

// --- Searching and order ---
impl<T, const M: usize> ArrVecMin<T, M> {
    /// See [`slice::contains`]. Searches both storage regions.
    #[inline]
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.arr.contains(value) || self.vec.contains(value)
    }

    /// See [`slice::starts_with`]. The prefix may cross the storage boundary.
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        needle.len() <= self.len() && self.iter().take(needle.len()).eq(needle.iter())
    }

    /// See [`slice::ends_with`]. The suffix may cross the storage boundary.
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        needle.len() <= self.len()
            && self
                .iter()
                .skip(self.len() - needle.len())
                .eq(needle.iter())
    }

    /// See [`slice::binary_search`]. The entire logical sequence must be sorted.
    /// Returns a logical index spanning both storage regions.
    pub fn binary_search(&self, value: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|probe| probe.cmp(value))
    }

    /// See [`slice::binary_search_by`]. The entire sequence must be sorted
    /// consistently with the comparator. Any matching index may be returned.
    pub fn binary_search_by<'a, F>(&'a self, mut compare: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> Ordering,
    {
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let mid = left + (right - left) / 2;
            match compare(&self[mid]) {
                Ordering::Less => left = mid + 1,
                Ordering::Greater => right = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        Err(left)
    }

    /// See [`slice::binary_search_by_key`]. The entire sequence must be sorted
    /// by the extracted key. Returns a logical index spanning both regions.
    pub fn binary_search_by_key<'a, B, F>(&'a self, key: &B, mut f: F) -> Result<usize, usize>
    where
        B: Ord,
        F: FnMut(&'a T) -> B,
    {
        self.binary_search_by(|probe| f(probe).cmp(key))
    }

    /// See [`slice::partition_point`]. The entire sequence must be partitioned
    /// with matching elements before nonmatching elements.
    pub fn partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let mid = left + (right - left) / 2;
            if pred(&self[mid]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left
    }

    /// See [`slice::is_sorted`]. Includes the adjacent pair at the storage boundary.
    #[inline]
    pub fn is_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.iter().is_sorted()
    }

    /// See [`slice::is_sorted_by`]. Includes the adjacent pair at the storage boundary.
    pub fn is_sorted_by<'a, F>(&'a self, mut compare: F) -> bool
    where
        F: FnMut(&'a T, &'a T) -> bool,
    {
        self.iter().is_sorted_by(|a, b| compare(a, b))
    }

    /// See [`slice::is_sorted_by_key`]. Checks keys across both storage regions.
    #[inline]
    pub fn is_sorted_by_key<'a, F, K>(&'a self, f: F) -> bool
    where
        F: FnMut(&'a T) -> K,
        K: PartialOrd,
    {
        self.iter().is_sorted_by_key(f)
    }
}

// --- Rearranging and filling ---
impl<T, const M: usize> ArrVecMin<T, M> {
    /// See [`slice::reverse`]. Reverses the entire logical sequence in place.
    pub const fn reverse(&mut self) {
        self.reverse_range(0, self.len());
    }

    /// See [`slice::rotate_left`]. Rotates across both storage regions in place.
    /// Panics if `mid` exceeds the total length.
    pub const fn rotate_left(&mut self, mid: usize) {
        let len = self.len();
        assert!(mid <= len, "rotation amount exceeds length");
        if mid == 0 || mid == len {
            return;
        }
        self.reverse_range(0, mid);
        self.reverse_range(mid, len);
        self.reverse_range(0, len);
    }

    /// See [`slice::rotate_right`]. Rotates across both storage regions in place.
    /// Panics if `k` exceeds the total length.
    pub const fn rotate_right(&mut self, k: usize) {
        let len = self.len();
        assert!(k <= len, "rotation amount exceeds length");
        self.rotate_left(len - k);
    }

    /// See [`slice::fill`]. Replaces elements in both storage regions.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        let mut iter = self.iter_mut();
        if let Some(last) = iter.next_back() {
            for item in iter {
                item.clone_from(&value);
            }
            *last = value;
        }
    }

    /// See [`slice::fill_with`]. Calls the generator in logical element order.
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        for item in self.iter_mut() {
            *item = f();
        }
    }
}

// --- Mutable Access ---

// -- Not Len Decreasing --

// - Capacity -
impl<T, const M: usize> ArrVecMin<T, M> {
    /// See [`Vec::reserve`].
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.vec.reserve(additional);
    }

    /// See [`Vec::reserve_exact`].
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.vec.reserve_exact(additional);
    }

    /// See [`Vec::try_reserve`].
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.vec.try_reserve(additional)
    }

    /// See [`Vec::try_reserve_exact`].
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.vec.try_reserve_exact(additional)
    }

    /// See [`Vec::shrink_to_fit`].
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.vec.shrink_to_fit();
    }

    /// See [`Vec::shrink_to`].
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.vec.shrink_to(min_capacity.saturating_sub(M));
    }
}

// - Growth -
impl<T, const M: usize> ArrVecMin<T, M> {
    /// See [`Vec::push`].
    #[inline]
    pub fn push(&mut self, item: T) {
        self.vec.push(item);
    }

    /// See [`Vec::insert`].
    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(index <= self.len(), "insertion index out of bounds");
        if index < M {
            self.vec.reserve(1);
            let mut carry = element;
            for slot in &mut self.arr[index..] {
                core::mem::swap(slot, &mut carry);
            }
            self.vec.insert(0, carry);
        } else {
            self.vec.insert(index - M, element);
        }
    }

    /// See [`Vec::append`].
    #[inline]
    pub fn append(&mut self, other: &mut Vec<T>) {
        self.vec.append(other);
    }

    /// See [`Vec::extend_from_slice`].
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.vec.extend_from_slice(other);
    }

    /// See [`Vec::extend_from_within`].
    #[inline]
    pub fn extend_from_within<R>(&mut self, range: R)
    where
        R: RangeBounds<usize>,
        T: Clone,
    {
        let range = slice_range(&range, ..self.len());
        self.vec.reserve(range.len());
        let inline_end = range.end.min(M);
        if range.start < inline_end {
            self.vec
                .extend_from_slice(&self.arr[range.start..inline_end]);
        }
        if range.end > M {
            self.vec
                .extend_from_within(range.start.saturating_sub(M)..range.end - M);
        }
    }
}

impl<T, const M: usize> Extend<T> for ArrVecMin<T, M> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.vec.extend(iter);
    }
}

impl<'a, T: Copy, const M: usize> Extend<&'a T> for ArrVecMin<T, M> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.vec.extend(iter);
    }
}

// -- Len Decreasing --
impl<T, const M: usize> ArrVecMin<T, M> {
    /// See [`Vec::pop`]. Pops an element from the vector if the length of the vector is greater than `M`, otherwise does nothing and returns `None`.
    #[inline]
    pub fn pop_to_min(&mut self) -> Option<T> {
        self.vec.pop()
    }

    /// See [`Vec::pop_if`]. Pops an element from the vector if the length of the vector is greater than `M` and the provided predicate returns `true`, otherwise does nothing and returns `None`.
    #[inline]
    pub fn pop_to_min_if(&mut self, pred: impl FnOnce(&mut T) -> bool) -> Option<T> {
        self.vec.pop_if(pred)
    }

    /// See [`Vec::remove`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn remove(&mut self, index: usize) -> Result<T, ModifyError<M>> {
        if !self.vec.is_empty() {
            assert!(index < self.len(), "removal index out of bounds");
            if index < M {
                let replacement = self.vec.remove(0);
                let removed = core::mem::replace(&mut self.arr[index], replacement);
                self.arr[index..].rotate_left(1);
                Ok(removed)
            } else {
                Ok(self.vec.remove(index - M))
            }
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::swap_remove`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn swap_remove(&mut self, index: usize) -> Result<T, ModifyError<M>> {
        if !self.vec.is_empty() {
            assert!(index < self.len(), "removal index out of bounds");
            if index < M {
                let replacement = unsafe { self.vec.pop().unwrap_unchecked() };
                Ok(core::mem::replace(&mut self.arr[index], replacement))
            } else {
                Ok(self.vec.swap_remove(index - M))
            }
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::truncate`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn truncate(&mut self, len: usize) -> Result<(), ModifyError<M>> {
        if len >= M {
            self.vec.truncate(len - M);
            Ok(())
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::truncate`]. Truncates the vector to `len` if `len` is greater than or equal to `M`, otherwise truncates the vector to `M`.
    #[inline]
    pub fn truncate_or_min(&mut self, len: usize) {
        self.vec.truncate(len.saturating_sub(M))
    }

    /// See [`Vec::truncate`]. Truncates the vector to `M`.
    #[inline]
    pub fn truncate_to_min(&mut self) {
        self.vec.clear();
    }

    /// See [`Vec::resize`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn resize(&mut self, new_len: usize, value: T) -> Result<(), ModifyError<M>>
    where
        T: Clone,
    {
        if new_len >= M {
            self.vec.resize(new_len - M, value);
            Ok(())
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::resize`]. Resizes the vector to `new_len` if `new_len` is greater than or equal to `M`, otherwise resizes the vector to `M`.
    #[inline]
    pub fn resize_or_min(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.vec.resize(new_len.saturating_sub(M), value);
    }

    /// See [`Vec::resize_with`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn resize_with<F>(&mut self, new_len: usize, generator: F) -> Result<(), ModifyError<M>>
    where
        F: FnMut() -> T,
    {
        if new_len >= M {
            self.vec.resize_with(new_len - M, generator);
            Ok(())
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::resize_with`]. Resizes the vector to `new_len` if `new_len` is greater than or equal to `M`, otherwise resizes the vector to `M`.
    #[inline]
    pub fn resize_or_min_with<F>(&mut self, new_len: usize, generator: F)
    where
        F: FnMut() -> T,
    {
        self.vec.resize_with(new_len.saturating_sub(M), generator);
    }

    /// See [`Vec::drain`]. Returns an error if the operation would reduce the length of the vector below `M`.
    /// Ranges touching the inline array move the retained elements into place
    /// before returning an iterator over the removed elements in the heap tail.
    #[must_use = "this operation may fail"]
    pub fn drain<R>(&mut self, range: R) -> Result<vec::Drain<'_, T>, ModifyError<M>>
    where
        R: RangeBounds<usize>,
    {
        let range = slice_range(&range, ..self.len());
        let drain_len = range.len();

        if drain_len <= self.vec.len() {
            if range.start >= M {
                return Ok(self.vec.drain(range.start - M..range.end - M));
            }
            // Rotate the removed values to the end of the heap tail. This leaves
            // the inline array fully initialized even if the drain is forgotten.
            let len = self.len();
            self.reverse_range(range.start, range.end);
            self.reverse_range(range.end, len);
            self.reverse_range(range.start, len);
            let tail_start = self.vec.len() - drain_len;
            Ok(self.vec.drain(tail_start..))
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::split_off`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn split_off(&mut self, at: usize) -> Result<Vec<T>, ModifyError<M>> {
        if at >= M {
            Ok(self.vec.split_off(at - M))
        } else {
            Err(ModifyError)
        }
    }
}

// --- Equality & Ordering ---
impl<T: PartialEq, const M1: usize, const M2: usize> PartialEq<ArrVecMin<T, M2>>
    for ArrVecMin<T, M1>
{
    #[inline]
    fn eq(&self, other: &ArrVecMin<T, M2>) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T: Eq, const M: usize> Eq for ArrVecMin<T, M> {}

impl<T: PartialEq, const M: usize> PartialEq<Vec<T>> for ArrVecMin<T, M> {
    #[inline]
    fn eq(&self, other: &Vec<T>) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T: PartialEq, const M: usize> PartialEq<ArrVecMin<T, M>> for Vec<T> {
    #[inline]
    fn eq(&self, other: &ArrVecMin<T, M>) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T: PartialOrd, const M1: usize, const M2: usize> PartialOrd<ArrVecMin<T, M2>>
    for ArrVecMin<T, M1>
{
    #[inline]
    fn partial_cmp(&self, other: &ArrVecMin<T, M2>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, const M: usize> Ord for ArrVecMin<T, M> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<T: PartialOrd, const M: usize> PartialOrd<Vec<T>> for ArrVecMin<T, M> {
    #[inline]
    fn partial_cmp(&self, other: &Vec<T>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: PartialOrd, const M: usize> PartialOrd<ArrVecMin<T, M>> for Vec<T> {
    #[inline]
    fn partial_cmp(&self, other: &ArrVecMin<T, M>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

// Hash the logical sequence so equal values with different minima hash equally.
impl<T: Hash, const M: usize> Hash for ArrVecMin<T, M> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for item in self {
            item.hash(state);
        }
    }
}

#[cfg(feature = "serde")]
mod __serde {
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    impl<T: Serialize, const M: usize> Serialize for ArrVecMin<T, M> {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serializer.collect_seq(self.iter())
        }
    }

    impl<'de, T: Deserialize<'de> + Debug, const M: usize> Deserialize<'de> for ArrVecMin<T, M> {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let vec = Vec::deserialize(deserializer)?;
            Self::try_from_vec(vec).map_err(Error::custom)
        }
    }
}
