use vecmin::ArrVecMin;

fn check_against_slice<const M: usize>() {
    for len in M..M + 7 {
        let expected: Vec<_> = (0..len as i32).map(|i| i * 2).collect();
        let values = ArrVecMin::<_, M>::collect(expected.clone()).unwrap();
        for key in -1..=len as i32 * 2 + 1 {
            assert_eq!(values.contains(&key), expected.contains(&key));
            assert_eq!(values.binary_search(&key), expected.binary_search(&key));
            assert_eq!(
                values.binary_search_by(|x| x.cmp(&key)),
                expected.binary_search(&key)
            );
            assert_eq!(
                values.binary_search_by_key(&key, |x| *x),
                expected.binary_search(&key)
            );
            assert_eq!(
                values.partition_point(|x| *x < key),
                expected.partition_point(|x| *x < key)
            );
        }
        for start in 0..=len {
            for end in start..=len {
                let needle = &expected[start..end];
                assert_eq!(values.starts_with(needle), expected.starts_with(needle));
                assert_eq!(values.ends_with(needle), expected.ends_with(needle));
            }
        }
        assert!(!values.starts_with(&vec![0; len + 1]));
        assert!(!values.ends_with(&vec![0; len + 1]));
        assert!(values.is_sorted());
        assert!(values.is_sorted_by(|a, b| a <= b));
        assert!(values.is_sorted_by_key(|x| *x));
        let mut reversed = values.clone();
        reversed.reverse();
        let mut expected_reversed = expected.clone();
        expected_reversed.reverse();
        assert_eq!(reversed, expected_reversed);
        assert_eq!(reversed.is_sorted(), expected_reversed.is_sorted());
        assert!(reversed.is_sorted_by(|a, b| a >= b));
        assert!(reversed.is_sorted_by_key(|x| -x));
        for amount in 0..=len {
            let mut actual = values.clone();
            let mut rotated = expected.clone();
            actual.rotate_left(amount);
            rotated.rotate_left(amount);
            assert_eq!(actual, rotated);
            actual.rotate_right(amount);
            assert_eq!(actual, expected);
            actual.rotate_right(amount);
            rotated = expected.clone();
            rotated.rotate_right(amount);
            assert_eq!(actual, rotated);
            assert_eq!(actual.capacity(), values.capacity());
        }
    }
}

#[test]
fn searches_checks_and_rotations_match_slices() {
    check_against_slice::<0>();
    check_against_slice::<1>();
    check_against_slice::<3>();
    check_against_slice::<6>();
}

#[test]
fn order_checks_detect_boundary_and_unordered_pairs() {
    // Each region is sorted; only the pair spanning them is out of order.
    let values = ArrVecMin::<_, 2>::from_array([1, 4, 2, 5]);
    assert!(!values.is_sorted());
    assert!(!values.is_sorted_by(|a, b| a <= b));
    assert!(!values.is_sorted_by_key(|x| *x));
    let values = ArrVecMin::<_, 2>::from_array([1, 2, 2, 3]);
    assert!(values.is_sorted());
    assert!(!values.is_sorted_by(|a, b| a < b));
    let values = ArrVecMin::<_, 1>::from_array([0.0, f64::NAN]);
    assert!(!values.is_sorted());
    assert!(!values.is_sorted_by_key(|x| *x));
}

#[test]
fn search_duplicates_returns_matching_index_and_correct_insertions() {
    let values = ArrVecMin::<_, 3>::from_array([1, 2, 2, 2, 4]);
    let index = values.binary_search(&2).unwrap();
    assert_eq!(values[index], 2);
    assert_eq!(values.binary_search(&0), Err(0));
    assert_eq!(values.binary_search(&3), Err(4));
    assert_eq!(values.binary_search(&5), Err(5));
    assert_eq!(values.partition_point(|x| *x < 2), 1);
    assert_eq!(values.partition_point(|x| *x <= 2), 4);
}

#[test]
fn borrowed_keys_and_descending_comparators() {
    let values =
        ArrVecMin::<_, 1>::from_array([String::from("a"), String::from("b"), String::from("c")]);
    assert_eq!(values.binary_search_by_key(&"b", |s| s.as_str()), Ok(1));
    assert!(values.is_sorted_by_key(|s| s.as_str()));
    let descending = ArrVecMin::<_, 2>::from_array([9, 7, 5, 3]);
    assert_eq!(descending.binary_search_by(|x| x.cmp(&5).reverse()), Ok(2));
    assert_eq!(descending.binary_search_by(|x| x.cmp(&6).reverse()), Err(2));
}

#[test]
fn fill_handles_both_regions_and_generator_order() {
    let mut values =
        ArrVecMin::<_, 2>::from_array([String::new(), String::new(), String::new(), String::new()]);
    values.fill(String::from("filled"));
    assert!(values.iter().all(|s| s == "filled"));
    let mut next = 0;
    values.fill_with(|| {
        next += 1;
        next.to_string()
    });
    assert_eq!(next, 4);
    assert_eq!(values.into_vec(), ["1", "2", "3", "4"]);
    let mut inline = ArrVecMin::new([String::new()]);
    inline.fill(String::from("inline"));
    assert_eq!(inline.first(), "inline");
    let mut heap = ArrVecMin::<_, 0>::from_array([String::new(), String::new()]);
    heap.fill(String::from("heap"));
    assert_eq!(heap.into_vec(), ["heap", "heap"]);
    let mut empty = ArrVecMin::<String, 0>::new([]);
    empty.fill(String::from("unused"));
    empty.fill_with(|| panic!("empty vector must not call generator"));
    assert!(empty.is_empty());
}

#[test]
fn generator_panic_leaves_all_elements_initialized() {
    let mut values = ArrVecMin::<_, 2>::from_array([String::new(), String::new(), String::new()]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut calls = 0;
        values.fill_with(|| {
            calls += 1;
            assert!(calls < 3, "generator failed");
            calls.to_string()
        });
    }));
    assert!(result.is_err());
    assert_eq!(values.into_vec(), ["1", "2", ""]);
}

#[test]
fn invalid_rotations_panic_before_modifying_values() {
    for amount in [4, usize::MAX] {
        let mut values = ArrVecMin::<_, 2>::from_array([1, 2, 3]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| values.rotate_left(amount)))
                .is_err()
        );
        assert_eq!(values, vec![1, 2, 3]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| values.rotate_right(amount)))
                .is_err()
        );
        assert_eq!(values, vec![1, 2, 3]);
    }
}
