use crate::graphics::matrix::Matrix;

// transformations
#[allow(dead_code)]
impl Matrix {
    // reflection over y-axis
    pub fn reflect_yz() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, -1.0);
        t
    }

    // reflection over x-axis
    pub fn reflect_xz() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(1, 1, -1.0);
        t
    }

    // reflect over z
    pub fn reflect_xy() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(2, 2, -1.0);
        t
    }

    pub fn reflect_45() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, 0.0);
        t.set(1, 0, 1.0);
        t.set(0, 1, 1.0);
        t.set(1, 1, 0.0);
        t
    }

    pub fn reflect_neg45() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, 0.0);
        t.set(1, 0, -1.0);
        t.set(0, 1, -1.0);
        t.set(1, 1, 0.0);
        t
    }

    pub fn reflect_origin() -> Self {
        let mut t = Self::new(4, 4, vec![]);
        t.set(0, 0, -1.0);
        t.set(1, 1, -1.0);
        t
    }

    pub fn translate(x: f64, y: f64, z: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(3, 0, x);
        t.set(3, 1, y);
        t.set(3, 2, z);
        t
    }

    pub fn scale(x: f64, y: f64, z: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, x);
        t.set(1, 1, y);
        t.set(2, 2, z);
        t
    }

    pub fn rotate_point(theta: f64, x: f64, y: f64, z: f64) -> Self {
        let mut m = Self::identity_matrix(4);
        let angle = theta.to_radians();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        m.set(0, 0, (t * x * x) + c);
        m.set(0, 1, (t * x * y) - (s * z));
        m.set(0, 2, (t * x * z) + (s * z));
        m.set(1, 0, (t * x * y) + (s * z));
        m.set(1, 1, (t * y * y) + c);
        m.set(1, 2, (t * y * z) - (s * x));
        m.set(2, 0, (t * x * z) - (s * y));
        m.set(2, 1, (t * y * z) + (s * x));
        m.set(2, 2, (t * z * z) + c);
        m
    }

    pub fn rotate_x(theta: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        let angle = theta.to_radians();
        t.set(1, 1, angle.cos());
        t.set(2, 1, -angle.sin());
        t.set(1, 2, angle.sin());
        t.set(2, 2, angle.cos());
        t
    }

    pub fn rotate_y(theta: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        let angle = theta.to_radians();
        t.set(0, 0, angle.cos());
        t.set(0, 2, -angle.sin());
        t.set(2, 0, angle.sin());
        t.set(2, 2, angle.cos());
        t
    }

    pub fn rotate_z(theta: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        let angle = theta.to_radians();
        t.set(0, 0, angle.cos());
        t.set(1, 0, -angle.sin());
        t.set(0, 1, angle.sin());
        t.set(1, 1, angle.cos());
        t
    }

    pub fn shearing_x(sh_y: f64, sh_z: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 1, sh_y);
        t.set(0, 2, sh_z);
        t
    }

    pub fn shearing_y(sh_x: f64, sh_z: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(1, 0, sh_x);
        t.set(1, 2, sh_z);
        t
    }

    pub fn shearing_z(sh_x: f64, sh_y: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(2, 0, sh_x);
        t.set(2, 1, sh_y);
        t
    }
}
