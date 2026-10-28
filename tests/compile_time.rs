use vecmin::{ArrVecMin, VecMin, arrvecmin, vecmin};

#[test]
fn array_constructors_keep_all_elements() {
    #[derive(Debug, PartialEq)]
    struct Item(usize);

    let inline = ArrVecMin::<_, 2>::from_array([Item(0), Item(1), Item(2)]);
    assert_eq!(inline.split_at_min(), (&[Item(0), Item(1)], &[Item(2)][..]));
    let heap = VecMin::<_, 2>::from_array([Item(0), Item(1), Item(2)]);
    assert_eq!(heap.into_inner(), inline.into_vec());

    let inline = ArrVecMin::<_, 2>::from_array([1, 2]);
    assert_eq!(inline.capacity(), 2);
    assert!(inline.is_minimum());
    assert!(VecMin::<_, 2>::from_array([1, 2]).is_minimum());
    assert!(ArrVecMin::<i32, 0>::from_array([]).is_empty());
    assert!(VecMin::<i32, 0>::from_array([]).is_empty());
    assert_eq!(
        ArrVecMin::<_, 0>::from_array([1, 2]).split_at_min().1,
        &[1, 2]
    );
    assert_eq!(VecMin::<_, 0>::from_array([1, 2]).as_slice(), &[1, 2]);
}

#[test]
fn indices_access_and_mutate_guaranteed_elements() {
    fn inline_at_one(values: &ArrVecMin<i32, 2>) -> &i32 {
        values.get_min::<1>()
    }
    fn heap_at_one(values: &VecMin<i32, 2>) -> &i32 {
        values.get_min::<1>()
    }

    let mut inline = ArrVecMin::<_, 2>::from_array([10, 20, 30]);
    let mut heap = VecMin::<_, 2>::from_array([10, 20, 30]);
    assert_eq!(*inline.get_min::<0>(), 10);
    assert_eq!(*heap.get_min::<0>(), 10);
    *inline.get_min_mut::<1>() = 99;
    *heap.get_min_mut::<1>() = 99;
    assert_eq!(*inline_at_one(&inline), 99);
    assert_eq!(*heap_at_one(&heap), 99);
    assert_eq!(inline.split_at_min().1, &[30]);
    assert_eq!(heap.as_slice(), &[10, 99, 30]);
}

#[test]
fn try_from_arrays_still_returns_errors() {
    assert_eq!(ArrVecMin::<_, 3>::try_from([1, 2]).unwrap_err().0, [1, 2]);
    assert_eq!(VecMin::<_, 3>::try_from([1, 2]).unwrap_err().0, [1, 2]);
}

#[test]
fn macro_assertions_accept_enclosing_const_generics() {
    fn repeated<const M: usize, const N: usize>() -> (ArrVecMin<i32, M>, VecMin<i32, M>) {
        (arrvecmin![M; [7; N]], vecmin![M; [7; N]])
    }
    fn listed<const M: usize>(a: String, b: String) -> (ArrVecMin<i32, M>, VecMin<String, M>) {
        (arrvecmin![M; [1, 2]], vecmin![M; [a, b]])
    }
    let (inline, heap) = repeated::<2, 4>();
    assert_eq!(inline.into_vec(), heap.into_inner());
    let (inline, heap) = listed::<1>(String::from("a"), String::from("b"));
    assert_eq!(inline.into_vec(), [1, 2]);
    assert_eq!(heap.into_inner(), ["a", "b"]);
}

#[test]
fn macros_infer_minimum_without_type_annotations() {
    let heap_list = vecmin![String::from("first"), String::from("last"),];
    let inline_list = arrvecmin![String::from("first"), String::from("last"),];
    assert_eq!(heap_list.minimum(), 2);
    assert_eq!(inline_list.minimum(), 2);
    assert!(heap_list.is_minimum());
    assert!(inline_list.is_minimum());
    assert_eq!(heap_list.into_inner(), inline_list.into_vec());

    let heap_repeated = vecmin![7; 3];
    let inline_repeated = arrvecmin![7; 3];
    assert_eq!(heap_repeated.minimum(), 3);
    assert_eq!(inline_repeated.minimum(), 3);
    assert!(heap_repeated.is_minimum());
    assert!(inline_repeated.is_minimum());

    let heap_empty = vecmin![7; 0];
    let inline_empty = arrvecmin![7; 0];
    assert_eq!(heap_empty.minimum(), 0);
    assert_eq!(inline_empty.minimum(), 0);
    assert!(heap_empty.is_empty());
    assert!(inline_empty.is_empty());
}

#[test]
fn inferred_repetition_accepts_enclosing_const_generic() {
    fn make<const N: usize>() -> (VecMin<i32, N>, ArrVecMin<i32, N>) {
        // Inspect before returning so the return type cannot resolve the minimum.
        let heap = vecmin![7; N];
        let inline = arrvecmin![7; N];
        assert_eq!(heap.minimum(), N);
        assert_eq!(inline.minimum(), N);
        (heap, inline)
    }
    let (heap, inline) = make::<4>();
    assert_eq!(heap.into_inner(), inline.into_vec());
}
