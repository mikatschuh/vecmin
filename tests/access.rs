use std::ops::{Index, IndexMut};
use vecmin::{ArrVecMin, VecMin};

#[test]
fn vec_get_and_index_access_head_and_tail() {
    let mut values = VecMin::<_, 2>::from_array([10, 20, 30, 40]);
    for (index, expected) in [10, 20, 30, 40].into_iter().enumerate() {
        assert_eq!(VecMin::get(&values, index), Some(&expected));
        assert_eq!(*Index::index(&values, index), expected);
    }
    assert_eq!(values.get(4), None);
    assert_eq!(values.get(usize::MAX), None);
    assert_eq!(values.get_mut(4), None);
    assert_eq!(values.get_mut(usize::MAX), None);
    *VecMin::get_mut(&mut values, 0).unwrap() = 11;
    *IndexMut::index_mut(&mut values, 1) = 22;
    values[2] = 33;
    *values.get_mut(3).unwrap() = 44;
    assert_eq!(values.as_slice(), &[11, 22, 33, 44]);
}

#[test]
fn vec_access_preserves_slice_range_support() {
    let mut values = VecMin::<_, 2>::from_array([10, 20, 30, 40]);
    assert_eq!(values.get(1..3), Some(&[20, 30][..]));
    assert_eq!(values.get(..), Some(&[10, 20, 30, 40][..]));
    assert_eq!(values.get(4..4), Some(&[][..]));
    assert_eq!(values.get(..5), None);
    assert_eq!(values.get_mut(..5), None);
    assert_eq!(&values[1..=2], &[20, 30]);
    values.get_mut(1..3).unwrap().copy_from_slice(&[21, 31]);
    values[2..].copy_from_slice(&[32, 42]);
    assert_eq!(&values[..], &[10, 21, 32, 42]);
}

#[test]
fn zero_minimum_get_and_index_work_with_actual_length() {
    let mut empty = VecMin::<i32, 0>::from_array([]);
    assert_eq!(empty.get(0), None);
    assert_eq!(empty.get_mut(0), None);
    assert_eq!(empty.get(..), Some(&[][..]));
    empty.push(7);
    assert_eq!(empty.get(0), Some(&7));
    empty[0] = 8;
    assert_eq!(empty[0], 8);
}

#[test]
#[should_panic]
fn vec_index_panics_out_of_bounds() {
    let values = VecMin::<_, 2>::from_array([10, 20]);
    let _ = values[2];
}

#[test]
#[should_panic]
fn vec_index_mut_panics_out_of_bounds() {
    let mut values = VecMin::<_, 2>::from_array([10, 20]);
    values[2] = 30;
}

#[test]
fn first_and_last_follow_tail_growth_and_removal() {
    let mut heap = VecMin::<_, 2>::from_array([10, 20]);
    let mut inline = ArrVecMin::new([10, 20]);
    assert_eq!(*heap.first(), 10);
    assert_eq!(*inline.first(), 10);
    assert_eq!(*heap.last(), 20);
    assert_eq!(*inline.last(), 20);
    *heap.first_mut() = 11;
    *inline.first_mut() = 11;
    *heap.last_mut() = 21;
    *inline.last_mut() = 21;
    assert_eq!(heap.as_slice(), &[11, 21]);
    assert_eq!(inline.min_slice(), &[11, 21]);
    heap.push(30);
    inline.push(30);
    *heap.last_mut() = 31;
    *inline.last_mut() = 31;
    assert_eq!(*heap.first(), 11);
    assert_eq!(*inline.first(), 11);
    assert_eq!(*heap.last(), 31);
    assert_eq!(*inline.last(), 31);
    assert_eq!(heap.pop_to_min(), Some(31));
    assert_eq!(inline.pop_to_min(), Some(31));
    assert_eq!(*heap.last(), 21);
    assert_eq!(*inline.last(), 21);
}

#[test]
fn singleton_first_and_last_refer_to_same_element() {
    let mut heap = VecMin::<_, 1>::from_array([String::from("initial")]);
    let mut inline = ArrVecMin::new([String::from("initial")]);
    heap.first_mut().push_str(" first");
    inline.first_mut().push_str(" first");
    assert_eq!(heap.last(), "initial first");
    assert_eq!(inline.last(), "initial first");
    heap.last_mut().push_str(" last");
    inline.last_mut().push_str(" last");
    assert_eq!(heap.first(), "initial first last");
    assert_eq!(inline.first(), "initial first last");
}
