use std::f64::consts::PI;

use super::matrix::Matrix;

impl Matrix {
    /// Adds a new circle centered at (cx, cy, cz) with radius r, and a precision of the circle
    ///
    /// # Arguments
    ///
    /// * `cx` - The x corrdinate of the center of the circle
    /// * `cy` - The y corrdinate of the center of the circle
    /// * `cz` - The z corrdinate of the center of the circle
    /// * `r` - The radius of the circle
    /// * `step` - The precision of the circle
    pub fn add_circle(&mut self, cx: f64, cy: f64, cz: f64, r: f64, step: f64) {
        self.add_parametric_curve(
            |t: f64| r * (t * 2.0 * PI).cos() + cx,
            |t: f64| r * (t * 2.0 * PI).sin() + cy,
            cz,
            step,
        );
    }

    // pub fn add_triange(&mut self, cx: f64)
}
