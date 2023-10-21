#![allow(dead_code)]
use std::cmp::min;
/// Returns and calcuates the new x,y corrdinates from the polar corrdinate system
///
/// # Arguments
///
/// * `magnitude` - A f64 number that represents the magnitude of R in the polar corrdinate system
/// * `angle_degrees` - A f64 number that represents theta in the polar corrdinate system
///
#[must_use]
pub fn polar_to_xy(magnitude: f64, theta: f64) -> (f64, f64) {
    let (dy, dx) = theta.to_radians().sin_cos();
    (dx * magnitude, dy * magnitude)
}

pub(crate) fn binom(n: usize, k: usize) -> usize {
    let mut result = 1;
    for i in 0..k {
        result = (result * (n - i)) / (i + 1);
    }
    result
}

// pub(crate) fn factorial(n: usize) -> usize {
//     (1..=n).product()
// }

// fn recbinom(n: usize, k: usize) -> usize {
//     if n == k || k == 0 {
//         1
//     } else {
//         recbinom(n - 1, k) + recbinom(n - 1, k - 1)
//     }
// }

pub(crate) fn dpbinom(n: usize, k: usize) -> usize {
    let mut dp = vec![0; k + 1];
    dp[0] = 1;
    for i in 1..=n {
        let mut j = min(i, k);
        while j > 0 {
            dp[j] += dp[j - 1];
            j -= 1;
        }
    }
    dp[k]
}

#[test]
fn binom_test() {
    assert_eq!(1, dpbinom(6, 6));
    assert_eq!(10, binom(5, 3));
    assert_eq!(210, binom(10, 6));
}

pub(crate) fn mapper(instart: f64, inend: f64, outstart: f64, outend: f64) -> impl Fn(f64) -> f64 {
    let slope = (outend - outstart) / (inend - instart);
    // move values into closure so they are captured by value, not ref
    move |x| outstart + slope * (x - instart)
}
