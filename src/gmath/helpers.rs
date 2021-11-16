#[allow(dead_code)]
/// Returns and calcuates the new x,y corrdinates from the polar corrdinate system
///
/// # Arguments
///
/// * `magnitude` - A f64 number that represents the magnitude of R in the polar corrdinate system
/// * `angle_degrees` - A f64 number that represents theta in the polar corrdinate system
///
pub fn polar_to_xy(magnitude: f64, theta: f64) -> (f64, f64) {
    let (dy, dx) = theta.to_radians().sin_cos();
    (dx * magnitude, dy * magnitude)
}

// These are the numbers you get when you multiply by the Inverse Hermite Matrix
pub(crate) fn hermite_curve_coeffs(p0: f64, p1: f64, r0: f64, r1: f64) -> (f64, f64, f64, f64) {
    (
        // Take advantage that p1 is greater than p0...so
        // another way to write this is:
        // 2.0 * p0 + -2 * p1 + ro + r1, but since the first temrs will cancel out
        //   technically this is fine
        2.0 * (p0 - p1) + r0 + r1,
        3.0 * (-p0 + p1) - 2.0 * r0 - r1,
        r0,
        p0,
    )
}

pub(crate) fn bezier_curve_coeffs(p0: f64, p1: f64, p2: f64, p3: f64) -> (f64, f64, f64, f64) {
    todo!()
}

pub(crate) fn mapper(instart: f64, inend: f64, outstart: f64, outend: f64) -> impl Fn(f64) -> f64 {
    let slope = (outend - outstart) / (inend - instart);
    // move values into closure so they are captured by value, not ref
    move |x| outstart + slope * (x - instart)
}
