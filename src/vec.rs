//! Implementation of the [`VecMin`] and [`VecOne`] newtypes.

// The documentation comment on `VecMin` has been modified by mikatschuh to include a reference to `ArrVecMin`

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::collections::TryReserveError;
use alloc::vec::{self, Vec};
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::repeat_with;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
use core::slice::{self, SliceIndex};

use crate::{ModifyError, slice_range};

/// A [`VecMin`] with a minimum length of 1.
pub type VecOne<T> = VecMin<T, 1>;

/// A vector with a minimum length of `M`.
///
/// Most methods of `Vec` are available on `VecMin` except those that reduce the length of the vector an unknown amount.
/// Methods that reduce the length of the vector by a known amount (e.g. `remove`, `truncate`) are available on `VecMin`
/// but return an error if the operation would reduce the length of the vector below `M`.
///
/// Other than ArrVecMin, this implementation stores all of it's elements on the heap in a single allocation,
/// i.e. it's just a `Vec` newtype.
#[repr(transparent)]
#[derive(Clone, Debug, Hash)]
pub struct VecMin<T, const M: usize> {
    vec: Vec<T>,
}

// --- Custom ---
impl<T, const M: usize> VecMin<T, M> {
    /// Assertion that that the length of the vector is at least `M`.
    #[inline]
    #[track_caller]
    pub const fn assert_invariant(&self) {
        assert!(self.vec.len() >= M);
    }

    /// Debug assertion that that the length of the vector is at least `M`.
    #[inline]
    #[track_caller]
    pub const fn debug_assert_invariant(&self) {
        debug_assert!(self.vec.len() >= M);
    }

    /// Returns the minimum length of the vector.
    #[inline]
    pub const fn minimum(&self) -> usize {
        M
    }

    /// Returns `true` if the length of the vector is equal to the minimum length `M`.
    #[inline]
    pub const fn is_minimum(&self) -> bool {
        self.vec.len() == M
    }

    /// Returns a slice to the first `M` elements of the vector, which are guaranteed to exist.
    #[inline]
    pub const fn min_slice(&self) -> &[T; M] {
        self.split_at_min().0
    }

    /// Returns a mutable slice to the first `M` elements of the vector, which are guaranteed to exist.
    #[inline]
    pub const fn min_slice_mut(&mut self) -> &mut [T; M] {
        self.split_at_min_mut().0
    }

    /// Returns the element at `I` within the guaranteed minimum.
    ///
    /// `I < M` is checked at compile time when this method is instantiated.
    ///
    /// ```
    /// use vecmin::VecMin;
    /// let values = VecMin::<_, 2>::from_array([10, 20, 30]);
    /// assert_eq!(*values.get_min::<1>(), 20);
    /// ```
    ///
    /// ```compile_fail
    /// use vecmin::VecMin;
    /// let values = VecMin::<_, 2>::from_array([10, 20, 30]);
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
    /// use vecmin::VecMin;
    /// let mut values = VecMin::<_, 2>::from_array([10, 20, 30]);
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
        self.debug_assert_invariant();

        let (min, extra) = unsafe { self.vec.as_slice().split_at_unchecked(M) };
        let min = unsafe { &*(min.as_ptr() as *const [T; M]) };
        (min, extra)
    }

    /// Returns a tuple of a mutable slice to the first `M` elements of the vector, which are guaranteed to exist, and a mutable slice to the remaining elements of the vector.
    #[inline]
    pub const fn split_at_min_mut(&mut self) -> (&mut [T; M], &mut [T]) {
        self.debug_assert_invariant();

        let (min, extra) = unsafe { self.vec.as_mut_slice().split_at_mut_unchecked(M) };
        let min = unsafe { &mut *(min.as_mut_ptr() as *mut [T; M]) };
        (min, extra)
    }
}

// --- Constructors, Convertors, and Destructors ---
/// A vector that is too short to be a valid `VecMin` returned as an error in a constructor.
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

impl<T, const M: usize> VecMin<T, M> {
    /// Creates a new `VecMin` from a `Vec`
    ///
    /// # Safety
    /// - The length of the `Vec` must be at least `M`.
    #[inline]
    pub const unsafe fn from_vec_unchecked(vec: Vec<T>) -> Self {
        Self { vec }
    }

    /// Creates a new `VecMin` from a `Vec`, returning an error if the length of the provided `Vec` is less than `M`.
    #[inline]
    pub const fn try_from_vec(vec: Vec<T>) -> Result<Self, ConstructError<T, M>> {
        if vec.len() >= M {
            // Safety: We just checked that the length was at least `M`.
            Ok(unsafe { Self::from_vec_unchecked(vec) })
        } else {
            Err(ConstructError(vec))
        }
    }

    /// Creates a new `VecMin` from anything that can be converted into a `Vec`, returning an error if the length of the provided `Vec` is less than `M`.
    #[inline]
    pub fn try_new(vec: impl Into<Vec<T>>) -> Result<Self, ConstructError<T, M>> {
        Self::try_from_vec(vec.into())
    }

    /// Creates a `VecMin` from an array with at least `M` elements.
    ///
    /// `N >= M` is checked at compile time when this constructor is instantiated.
    /// Use [`Self::try_new`] for a conversion that returns an error instead.
    ///
    /// ```
    /// use vecmin::VecMin;
    /// let values = VecMin::<_, 2>::from_array([1, 2, 3]);
    /// assert_eq!(values.len(), 3);
    /// assert_eq!(values.minimum(), 2);
    /// ```
    ///
    /// ```compile_fail
    /// use vecmin::VecMin;
    /// let _ = VecMin::<_, 3>::from_array([1, 2]);
    /// ```
    #[inline]
    pub fn from_array<const N: usize>(arr: [T; N]) -> Self {
        const { assert!(N >= M, "array is shorter than the minimum") };
        // Safety: The compile-time assertion guarantees at least `M` elements.
        unsafe { Self::from_vec_unchecked(arr.into()) }
    }

    /// Creates a new `VecMin` from an iterator, returning an error if the length of the collected `Vec` is less than `M`.
    #[inline]
    pub fn collect(iter: impl IntoIterator<Item = T>) -> Result<Self, ConstructError<T, M>> {
        let iter = iter.into_iter();
        let (low, high) = iter.size_hint();

        Self::collect_with_capacity(iter, high.unwrap_or(low).max(M))
    }

    /// Creates a new `VecMin` from an iterator, returning an error if the length of the collected `Vec` is less than `M`.
    /// The provided `capacity` is preallocated into the `Vec`.
    #[inline]
    pub fn collect_with_capacity(
        iter: impl IntoIterator<Item = T>,
        capacity: usize,
    ) -> Result<Self, ConstructError<T, M>> {
        let mut vec = Vec::with_capacity(capacity);
        vec.extend(iter);

        Self::try_new(vec)
    }

    /// Returns the inner `Vec`, consuming the `VecMin`.
    #[inline]
    pub fn into_inner(self) -> Vec<T> {
        self.vec
    }

    /// Returns a reference to the inner `Vec`.
    #[inline]
    pub fn vec(&self) -> &Vec<T> {
        &self.vec
    }

    /// See [`Vec::into_boxed_slice`].
    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.vec.into_boxed_slice()
    }

    /// See [`Vec::leak`].
    #[inline]
    pub fn leak(self) -> &'static mut [T] {
        self.vec.leak()
    }
}

impl<T: Default, const M: usize> Default for VecMin<T, M> {
    #[inline]
    fn default() -> Self {
        // Safety: We collect `M` elements satisfying the minimum length requirement.
        unsafe { Self::from_vec_unchecked(repeat_with(T::default).take(M).collect()) }
    }
}

impl<T, const M: usize> TryFrom<Vec<T>> for VecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        Self::try_new(vec)
    }
}

impl<T, const M: usize> From<VecMin<T, M>> for Vec<T> {
    #[inline]
    fn from(vec_min: VecMin<T, M>) -> Self {
        vec_min.vec
    }
}

impl<T, const M: usize> TryFrom<Box<[T]>> for VecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(boxed_slice: Box<[T]>) -> Result<Self, Self::Error> {
        Self::try_new(boxed_slice)
    }
}

impl<T, const M: usize> From<VecMin<T, M>> for Box<[T]> {
    #[inline]
    fn from(vec_min: VecMin<T, M>) -> Self {
        vec_min.vec.into_boxed_slice()
    }
}

impl<T: Clone, const M: usize> TryFrom<&[T]> for VecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        Self::try_new(slice)
    }
}

impl<T: Clone, const M: usize> TryFrom<&mut [T]> for VecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(slice: &mut [T]) -> Result<Self, Self::Error> {
        Self::try_new(slice)
    }
}

impl<T: Clone, const M: usize> TryFrom<Cow<'_, [T]>> for VecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(cow: Cow<'_, [T]>) -> Result<Self, Self::Error> {
        Self::try_new(cow)
    }
}

impl<T, const N: usize, const M: usize> TryFrom<[T; N]> for VecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(array: [T; N]) -> Result<Self, Self::Error> {
        Self::try_new(array)
    }
}

impl<T: Clone, const N: usize, const M: usize> TryFrom<&[T; N]> for VecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(array: &[T; N]) -> Result<Self, Self::Error> {
        Self::try_new(array)
    }
}

impl<T: Clone, const N: usize, const M: usize> TryFrom<&mut [T; N]> for VecMin<T, M> {
    type Error = ConstructError<T, M>;

    #[inline]
    fn try_from(array: &mut [T; N]) -> Result<Self, Self::Error> {
        Self::try_new(array)
    }
}

impl<T, const N: usize, const M: usize> TryFrom<VecMin<T, M>> for [T; N] {
    type Error = VecMin<T, M>;

    #[inline]
    fn try_from(vec_min: VecMin<T, M>) -> Result<[T; N], Self::Error> {
        // Safety: We obtained the original `Vec` from a valid `VecMin`.
        vec_min
            .vec
            .try_into()
            .map_err(|vec| unsafe { VecMin::from_vec_unchecked(vec) })
    }
}

// --- View ---
impl<T, const M: usize> VecMin<T, M> {
    /// Returns an element or subslice, or `None` if the index is out of bounds.
    #[inline]
    pub fn get<I: SliceIndex<[T]>>(&self, index: I) -> Option<&I::Output> {
        self.vec.get(index)
    }

    /// Returns a mutable element or subslice, or `None` if the index is out of bounds.
    #[inline]
    pub fn get_mut<I: SliceIndex<[T]>>(&mut self, index: I) -> Option<&mut I::Output> {
        self.vec.get_mut(index)
    }

    /// Returns a reference to the first element.
    ///
    /// Requires `M > 0`, checked at compile time when this method is instantiated.
    ///
    /// ```compile_fail
    /// use vecmin::VecMin;
    /// let values = VecMin::<_, 0>::from_array([1, 2]);
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
    /// use vecmin::VecMin;
    /// let mut values = VecMin::<_, 0>::from_array([1, 2]);
    /// let _ = values.first_mut();
    /// ```
    #[inline]
    pub const fn first_mut(&mut self) -> &mut T {
        const { assert!(M != 0, "first_mut requires a nonzero minimum") };
        &mut self.min_slice_mut()[0]
    }

    /// Returns a reference to the last element.
    ///
    /// Requires `M > 0`, checked at compile time when this method is instantiated.
    ///
    /// ```compile_fail
    /// use vecmin::VecMin;
    /// let values = VecMin::<_, 0>::from_array([1, 2]);
    /// let _ = values.last();
    /// ```
    #[inline]
    pub const fn last(&self) -> &T {
        const { assert!(M != 0, "last requires a nonzero minimum") };
        self.vec
            .as_slice()
            .last()
            .expect("nonzero minimum guarantees an element")
    }

    /// Returns a mutable reference to the last element.
    ///
    /// Requires `M > 0`, checked at compile time when this method is instantiated.
    ///
    /// ```compile_fail
    /// use vecmin::VecMin;
    /// let mut values = VecMin::<_, 0>::from_array([1, 2]);
    /// let _ = values.last_mut();
    /// ```
    #[inline]
    pub const fn last_mut(&mut self) -> &mut T {
        const { assert!(M != 0, "last_mut requires a nonzero minimum") };
        self.vec
            .as_mut_slice()
            .last_mut()
            .expect("nonzero minimum guarantees an element")
    }

    #[inline]
    /// See [`Vec::as_slice`].
    pub const fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }

    /// See [`Vec::as_mut_slice`].
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        self.vec.as_mut_slice()
    }

    #[inline]
    /// See [`Vec::as_ptr`].
    pub const fn as_ptr(&self) -> *const T {
        self.vec.as_ptr()
    }

    /// See [`Vec::as_mut_ptr`].
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.vec.as_mut_ptr()
    }

    /// See [`Vec::spare_capacity_mut`].
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.vec.spare_capacity_mut()
    }
}

impl<T, const M: usize, I: SliceIndex<[T]>> Index<I> for VecMin<T, M> {
    type Output = I::Output;

    #[inline]
    #[track_caller]
    fn index(&self, index: I) -> &Self::Output {
        &self.vec[index]
    }
}

impl<T, const M: usize, I: SliceIndex<[T]>> IndexMut<I> for VecMin<T, M> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.vec[index]
    }
}

impl<T, const M: usize> Deref for VecMin<T, M> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.vec.deref()
    }
}

impl<T, const M: usize> DerefMut for VecMin<T, M> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vec.deref_mut()
    }
}

impl<T, const M: usize> AsRef<[T]> for VecMin<T, M> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.vec.as_ref()
    }
}

impl<T, const M: usize> AsMut<[T]> for VecMin<T, M> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.vec.as_mut()
    }
}

impl<T, const M: usize> Borrow<[T]> for VecMin<T, M> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.vec.borrow()
    }
}

impl<T, const M: usize> BorrowMut<[T]> for VecMin<T, M> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.vec.borrow_mut()
    }
}

// --- Iterators ---
impl<T, const M: usize> IntoIterator for VecMin<T, M> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<'a, T: 'a, const M: usize> IntoIterator for &'a VecMin<T, M> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.vec.as_slice().iter()
    }
}

impl<'a, T: 'a, const M: usize> IntoIterator for &'a mut VecMin<T, M> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.vec.as_mut_slice().iter_mut()
    }
}

// --- Immutable Access ---
impl<T, const M: usize> VecMin<T, M> {
    /// See [`Vec::capacity`].
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    /// See [`Vec::len`].
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.vec.len()
    }
}

// --- Mutable Access ---

// -- Not Len Decreasing --

// - Capacity -
impl<T, const M: usize> VecMin<T, M> {
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
        self.vec.shrink_to(min_capacity);
    }
}

// - Growth -
impl<T, const M: usize> VecMin<T, M> {
    /// See [`Vec::push`].
    #[inline]
    pub fn push(&mut self, item: T) {
        self.vec.push(item);
    }

    /// See [`Vec::insert`].
    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        self.vec.insert(index, element);
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
        self.vec.extend_from_within(range);
    }
}

impl<T, const M: usize> Extend<T> for VecMin<T, M> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.vec.extend(iter);
    }
}

impl<'a, T: Copy, const M: usize> Extend<&'a T> for VecMin<T, M> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.vec.extend(iter);
    }
}

// -- Len Decreasing --
impl<T, const M: usize> VecMin<T, M> {
    /// See [`Vec::pop`]. Pops an element from the vector if the length of the vector is greater than `M`, otherwise does nothing and returns `None`.
    #[inline]
    pub fn pop_to_min(&mut self) -> Option<T> {
        if self.vec.len() > M {
            self.vec.pop()
        } else {
            None
        }
    }

    /// See [`Vec::pop_if`]. Pops an element from the vector if the length of the vector is greater than `M` and the provided predicate returns `true`, otherwise does nothing and returns `None`.
    #[inline]
    pub fn pop_to_min_if(&mut self, pred: impl FnOnce(&mut T) -> bool) -> Option<T> {
        if self.vec.len() > M {
            self.vec.pop_if(pred)
        } else {
            None
        }
    }

    /// See [`Vec::remove`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn remove(&mut self, index: usize) -> Result<T, ModifyError<M>> {
        if self.vec.len() > M {
            Ok(self.vec.remove(index))
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::swap_remove`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn swap_remove(&mut self, index: usize) -> Result<T, ModifyError<M>> {
        if self.vec.len() > M {
            Ok(self.vec.swap_remove(index))
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::truncate`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn truncate(&mut self, len: usize) -> Result<(), ModifyError<M>> {
        if len >= M {
            self.vec.truncate(len);
            Ok(())
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::truncate`]. Truncates the vector to `len` if `len` is greater than or equal to `M`, otherwise truncates the vector to `M`.
    #[inline]
    pub fn truncate_or_min(&mut self, len: usize) {
        self.vec.truncate(len.max(M))
    }

    /// See [`Vec::truncate`]. Truncates the vector to `M`.
    #[inline]
    pub fn truncate_to_min(&mut self) {
        self.vec.truncate(M);
    }

    /// See [`Vec::resize`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn resize(&mut self, new_len: usize, value: T) -> Result<(), ModifyError<M>>
    where
        T: Clone,
    {
        if new_len >= M {
            self.vec.resize(new_len, value);
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
        self.vec.resize(new_len.max(M), value);
    }

    /// See [`Vec::resize_with`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn resize_with<F>(&mut self, new_len: usize, generator: F) -> Result<(), ModifyError<M>>
    where
        F: FnMut() -> T,
    {
        if new_len >= M {
            self.vec.resize_with(new_len, generator);
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
        self.vec.resize_with(new_len.max(M), generator);
    }

    /// See [`Vec::drain`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[must_use = "this operation may fail"]
    pub fn drain<R>(&mut self, range: R) -> Result<vec::Drain<'_, T>, ModifyError<M>>
    where
        R: RangeBounds<usize>,
    {
        let drain_len = slice_range(&range, ..self.vec.len()).len();
        let final_len = self.vec.len() - drain_len;

        if final_len >= M {
            Ok(self.vec.drain(range))
        } else {
            Err(ModifyError)
        }
    }

    /// See [`Vec::split_off`]. Returns an error if the operation would reduce the length of the vector below `M`.
    #[inline]
    #[must_use = "this operation may fail"]
    pub fn split_off(&mut self, at: usize) -> Result<Vec<T>, ModifyError<M>> {
        if at >= M {
            Ok(self.vec.split_off(at))
        } else {
            Err(ModifyError)
        }
    }
}

// --- Equality & Ordering ---
impl<T: PartialEq, const M1: usize, const M2: usize> PartialEq<VecMin<T, M2>> for VecMin<T, M1> {
    #[inline]
    fn eq(&self, other: &VecMin<T, M2>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq, const M: usize> Eq for VecMin<T, M> {}

impl<T: PartialEq, const M: usize> PartialEq<Vec<T>> for VecMin<T, M> {
    #[inline]
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq, const M: usize> PartialEq<VecMin<T, M>> for Vec<T> {
    #[inline]
    fn eq(&self, other: &VecMin<T, M>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialOrd, const M1: usize, const M2: usize> PartialOrd<VecMin<T, M2>> for VecMin<T, M1> {
    #[inline]
    fn partial_cmp(&self, other: &VecMin<T, M2>) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Ord, const M: usize> Ord for VecMin<T, M> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: PartialOrd, const M: usize> PartialOrd<Vec<T>> for VecMin<T, M> {
    #[inline]
    fn partial_cmp(&self, other: &Vec<T>) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: PartialOrd, const M: usize> PartialOrd<VecMin<T, M>> for Vec<T> {
    #[inline]
    fn partial_cmp(&self, other: &VecMin<T, M>) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

#[cfg(feature = "serde")]
mod __serde {
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    impl<T: Serialize, const M: usize> Serialize for VecMin<T, M> {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.vec.serialize(serializer)
        }
    }

    impl<'de, T: Deserialize<'de> + Debug, const M: usize> Deserialize<'de> for VecMin<T, M> {
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
