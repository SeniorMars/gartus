/// Returns and calcuates the new x,y corrdinates from the polar corrdinate system
///
/// # Arguments
///
/// * `magnitude` - A f64 number that represents the magnitude of R in the polar corrdinate system
/// * `angle_degrees` - A f64 number that represents theta in the polar corrdinate system
///
#[must_use]
pub fn polar_to_xy(magnitude: f64, theta: f64) -> (f64, f64) {
    let (sin, cos) = theta.to_radians().sin_cos();
    (cos * magnitude, sin * magnitude)
}

pub(crate) fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }

    let mut result = 1;
    for i in 0..k.min(n - k) {
        result = (result * (n - i)) / (i + 1);
    }
    result
}

#[test]
fn binom_test() {
    assert_eq!(10, binom(5, 3));
    assert_eq!(210, binom(10, 6));
}
