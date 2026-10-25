# VecMin

Provides a `VecMin` and `VecOne` newtype wrapper around `Vec` that enforces a minimum length at compile time.

This crate requires `alloc` or `std`.

# Example
```rs
use vecmin::{VecMin, VecOne, vecmin, vecone};

fn main() {
    let mut one = vecone![1, 2, 3];

    // there are some custom methods using min.
    while let Some(x) = one.pop_to_min() {
        println!("{x}");
    }
    assert_eq!(one.len(), 1);

    one.remove(0).unwrap_err();

    // vecmin! sets the minimum to the length unless another length is provided
    let five = vecmin![2; 5];
    assert_eq!(five.minimum(), 5);

    // the minimum is provided first
    let two = vecmin![2; [2; 5]];
    assert_eq!(two.minimum(), 2);

    assert_eq!(five, two);

    assert_eq!(two.min_slice(), &[2, 2]);
}
```

`ArrVecMin` stores its first `M` elements inline and the rest in a `Vec`.
Since the two parts are separate, use iterators or `split_at_min()` to view them.

```rs
use vecmin::ArrVecMin;

let mut values = ArrVecMin::new([1, 2]);
values.push(3);
values.insert(1, 4);
assert_eq!(values.split_at_min(), (&[1, 4], &[2, 3][..]));
assert_eq!(values.into_vec(), vec![1, 4, 2, 3]);
```

# License
Licensed under Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)

This file has been modified by mikatschuh
