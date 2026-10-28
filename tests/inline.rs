// These tests have been added by mikatschuh

use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use vecmin::{ArrVecMin, ModifyError};

fn exercise_indices<const M: usize>() {
    for len in M..=M + 5 {
        let original: Vec<_> = (0..len).collect();
        for index in 0..=len {
            let mut actual = ArrVecMin::<_, M>::try_from(original.clone()).unwrap();
            let mut expected = original.clone();
            actual.insert(index, 99);
            expected.insert(index, 99);
            assert_eq!(actual, expected);
            if len > M && index < len {
                let mut actual = ArrVecMin::<_, M>::try_from(original.clone()).unwrap();
                let mut expected = original.clone();
                assert_eq!(actual.remove(index).unwrap(), expected.remove(index));
                assert_eq!(actual, expected);
                let mut actual = ArrVecMin::<_, M>::try_from(original.clone()).unwrap();
                let mut expected = original.clone();
                assert_eq!(
                    actual.swap_remove(index).unwrap(),
                    expected.swap_remove(index)
                );
                assert_eq!(actual, expected);
            }
        }
        for start in 0..=len {
            for end in start..=len {
                let mut actual = ArrVecMin::<_, M>::try_from(original.clone()).unwrap();
                let mut expected = original.clone();
                actual.extend_from_within(start..end);
                expected.extend_from_within(start..end);
                assert_eq!(actual, expected);

                let mut actual = ArrVecMin::<_, M>::try_from(original.clone()).unwrap();
                if len - (end - start) >= M {
                    let mut expected = original.clone();
                    let removed: Vec<_> = expected.drain(start..end).collect();
                    assert_eq!(
                        actual.drain(start..end).unwrap().collect::<Vec<_>>(),
                        removed
                    );
                    assert_eq!(actual, expected);
                } else {
                    assert!(actual.drain(start..end).is_err());
                    assert_eq!(actual, original);
                }
            }
        }
    }
}

#[test]
fn operations_match_vec_across_storage_boundary() {
    exercise_indices::<0>();
    exercise_indices::<1>();
    exercise_indices::<3>();
    exercise_indices::<6>();
}

#[test]
fn constructors_conversions_and_views() {
    const MIN: ArrVecMin<i32, 2> = ArrVecMin::new([1, 2]);
    assert_eq!(MIN.len(), 2);
    assert_eq!(MIN.capacity(), 2);
    assert!(MIN.is_minimum());
    let mut v = ArrVecMin::<_, 2>::try_from(vec![1, 2, 3, 4]).unwrap();
    assert_eq!(v.split_at_min(), (&[1, 2], &[3, 4][..]));
    assert_eq!(v.split_at_min().1, &[3, 4]);
    assert_eq!(v.get(4), None);
    v[0] = 5;
    *v.get_mut(2).unwrap() = 6;
    for item in &mut v {
        *item *= 2;
    }
    assert_eq!(
        (&v).into_iter().copied().collect::<Vec<_>>(),
        [10, 4, 12, 8]
    );
    assert_eq!(v.iter().rev().copied().collect::<Vec<_>>(), [8, 12, 4, 10]);
    let v = <[i32; 3]>::try_from(v).unwrap_err();
    assert_eq!(<[i32; 4]>::try_from(v).unwrap(), [10, 4, 12, 8]);
    assert_eq!(
        ArrVecMin::<_, 3>::try_from(vec![1, 2]).unwrap_err().0,
        [1, 2]
    );
    assert_eq!(Vec::from(MIN.clone()), [1, 2]);
    assert_eq!(&*Box::<[i32]>::from(MIN.clone()), &[1, 2]);
    assert_eq!(MIN.leak(), &[1, 2]);
    assert_eq!(ArrVecMin::<i32, 3>::default(), vec![0, 0, 0]);
    assert_eq!(ArrVecMin::<i32, 3>::default().capacity(), 3);
    assert_eq!(ArrVecMin::<_, 2>::collect(0..4).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(ArrVecMin::<_, 2>::collect(0..1).unwrap_err().0, [0]);
    assert_eq!(ArrVecMin::<_, 2>::collect(0..2).unwrap().capacity(), 2);
    let v = ArrVecMin::<_, 2>::collect_with_capacity(0..4, 10).unwrap();
    assert!(v.capacity() >= 10);
    assert_eq!(v, vec![0, 1, 2, 3]);
}

#[test]
fn lengths_capacity_and_minimum_guards() {
    let mut v = ArrVecMin::new([1, 2]);
    assert_eq!(v.remove(0), Err(ModifyError));
    assert_eq!(v.swap_remove(0), Err(ModifyError));
    assert_eq!(v.truncate(1), Err(ModifyError));
    assert_eq!(v.resize(1, 0), Err(ModifyError));
    assert_eq!(
        v.resize_with(1, || panic!("must not run")),
        Err(ModifyError)
    );
    assert_eq!(v.pop_to_min(), None);
    assert_eq!(v.pop_to_min_if(|_| panic!("must not run")), None);
    assert!(v.split_off(1).is_err());
    v.reserve_exact(5);
    assert!(v.capacity() >= 7);
    v.try_reserve(2).unwrap();
    v.try_reserve_exact(2).unwrap();
    v.push(3);
    assert_eq!(
        v.pop_to_min_if(|x| {
            *x += 1;
            false
        }),
        None
    );
    assert_eq!(v.pop_to_min(), Some(4));
    v.resize(5, 7).unwrap();
    assert_eq!(v, vec![1, 2, 7, 7, 7]);
    v.truncate(4).unwrap();
    assert_eq!(v.split_off(3).unwrap(), [7]);
    v.resize_with(5, || 8).unwrap();
    assert_eq!(v, vec![1, 2, 7, 8, 8]);
    v.truncate_or_min(0);
    assert!(v.is_minimum());
    v.resize_or_min(4, 9);
    v.resize_or_min_with(0, || panic!("must not run"));
    assert!(v.is_minimum());
    v.extend([3, 4]);
    v.truncate_to_min();
    v.shrink_to(2);
    assert_eq!(v.capacity(), 2);
    v.extend_from_slice(&[3, 4]);
    let mut tail = vec![5, 6];
    v.append(&mut tail);
    assert!(tail.is_empty());
    assert_eq!(v, vec![1, 2, 3, 4, 5, 6]);
    v.shrink_to_fit();
    assert_eq!(v.capacity(), 6);
    let mut empty = ArrVecMin::<i32, 0>::new([]);
    assert!(empty.is_empty());
    empty.push(1);
    assert!(!empty.is_empty());
    let zst = ArrVecMin::new([(); 3]);
    assert_eq!(zst.capacity(), usize::MAX);
}

#[test]
fn comparison_and_hash_ignore_storage_split() {
    let a = ArrVecMin::<_, 1>::collect(0..4).unwrap();
    let b = ArrVecMin::<_, 3>::collect(0..4).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Equal));
    let hash = |v: &dyn Fn(&mut DefaultHasher)| {
        let mut state = DefaultHasher::new();
        v(&mut state);
        state.finish()
    };
    assert_eq!(hash(&|h| a.hash(h)), hash(&|h| b.hash(h)));
    assert!(a < vec![0, 1, 2, 5]);
    assert!(vec![0, 1, 2, 3] == a);
    assert!(vec![0, 1, 2, 5] > a);
}

#[derive(Debug)]
struct Tracked(usize, Rc<Vec<Cell<usize>>>);
impl Drop for Tracked {
    fn drop(&mut self) {
        self.1[self.0].set(self.1[self.0].get() + 1);
    }
}

#[test]
fn non_copy_values_are_moved_and_dropped_once() {
    let drops = Rc::new((0..8).map(|_| Cell::new(0)).collect::<Vec<_>>());
    {
        let values = (0..8).map(|id| Tracked(id, drops.clone())).collect();
        let mut v = unsafe { ArrVecMin::<_, 3>::from_vec_unchecked(values) };
        let item = v.remove(1).unwrap();
        v.insert(2, item);
        drop(v.swap_remove(0).unwrap());
        let mut drain = v.drain(1..4).unwrap();
        assert_eq!(drain.next().unwrap().0, 2);
        drop(drain); // Partially consumed drain drops all remaining removed values.
        drop(v.into_vec());
    }
    assert!(drops.iter().all(|n| n.get() == 1));
}

#[test]
fn forgetting_drain_preserves_inline_minimum() {
    let mut v = ArrVecMin::<_, 3>::collect(0..8).unwrap();
    std::mem::forget(v.drain(1..4).unwrap());
    assert_eq!(v, vec![0, 4, 5, 6, 7]);
    assert_eq!(v.min_slice(), &[0, 4, 5]);
}

#[test]
#[should_panic]
fn invalid_insert_panics() {
    ArrVecMin::new([1, 2]).insert(3, 9);
}

#[test]
#[should_panic]
fn invalid_drain_range_panics() {
    let mut v = ArrVecMin::new([1, 2]);
    let _ = v.drain(..3);
}

#[cfg(feature = "serde")]
#[test]
fn deserialize_checks_minimum_and_preserves_all_elements() {
    use serde::Deserialize;
    use serde::de::value::{Error, SeqDeserializer};
    let de = SeqDeserializer::<_, Error>::new([1, 2, 3, 4].into_iter());
    let v = ArrVecMin::<i32, 2>::deserialize(de).unwrap();
    assert_eq!(v, vec![1, 2, 3, 4]);
    let de = SeqDeserializer::<_, Error>::new([1].into_iter());
    assert!(ArrVecMin::<i32, 2>::deserialize(de).is_err());
}

#[test]
fn macro_inferred_and_explicit_minima() {
    use vecmin::arrvecmin;

    const INLINE: ArrVecMin<i32, 3> = arrvecmin![1, 2, 3,];
    assert_eq!(INLINE.minimum(), 3);
    assert_eq!(INLINE.capacity(), 3);
    let repeated = arrvecmin![7; 3];
    assert_eq!(repeated.min_slice(), &[7, 7, 7]);
    let explicit = arrvecmin![2; [1, 2, 3,]];
    assert_eq!(explicit.split_at_min(), (&[1, 2], &[3][..]));
    let repeated = arrvecmin![2; [String::from("hello"); 4]];
    assert_eq!(repeated.minimum(), 2);
    assert_eq!(repeated.into_vec(), vec!["hello"; 4]);
    let inline = arrvecmin![3; [1, 2, 3]];
    assert_eq!(inline.capacity(), 3);
    let inline = arrvecmin![3; [7; 3]];
    assert_eq!(inline.capacity(), 3);
    let empty: ArrVecMin<i32, 0> = arrvecmin![];
    assert!(empty.is_empty());
    let empty: ArrVecMin<i32, 0> = arrvecmin![0; []];
    assert!(empty.is_empty());
    let empty = arrvecmin![0; [1; 0]];
    assert!(empty.is_empty());
    let heap = arrvecmin![0; [1, 2]];
    assert_eq!(heap.split_at_min(), (&[], &[1, 2][..]));
}

#[test]
fn macro_moves_elements_and_evaluates_once_in_order() {
    use vecmin::arrvecmin;

    let mut next = 0;
    let mut element = || {
        next += 1;
        next.to_string()
    };
    let v = arrvecmin![1; [element(), element(), element()]];
    assert_eq!(v.into_vec(), vec!["1", "2", "3"]);
    assert_eq!(next, 3);
    let first = String::from("first");
    let last = String::from("last");
    assert_eq!(
        arrvecmin![1; [first, last]].into_vec(),
        vec!["first", "last"]
    );
    let first = String::from("first");
    assert_eq!(arrvecmin![first].into_vec(), vec!["first"]);
    let mut calls = 0;
    let v = arrvecmin![1; [{ calls += 1; String::from("value") }; 3]];
    assert_eq!(calls, 1);
    assert_eq!(v.into_vec(), vec!["value"; 3]);
}
