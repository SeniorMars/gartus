#![allow(dead_code)]
#![allow(unused_variables)]
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

pub(crate) fn mapper(instart: f64, inend: f64, outstart: f64, outend: f64) -> impl Fn(f64) -> f64 {
    let slope = (outend - outstart) / (inend - instart);
    // move values into closure so they are captured by value, not ref
    move |x| outstart + slope * (x - instart)
}
