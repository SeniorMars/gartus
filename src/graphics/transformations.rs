use crate::graphics::matrix::Matrix;
#[allow(dead_code)]
impl Matrix {
    /// Returns a reflection over the yz (y) axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let reflect = Matrix::reflect_yz();
    /// ```
    pub fn reflect_yz() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, -1.0);
        t
    }

    /// Returns a reflection over the xz (x) axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let reflect = Matrix::reflect_xz();
    /// ```
    pub fn reflect_xz() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(1, 1, -1.0);
        t
    }

    /// Returns a reflection over the xy (y) axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let reflect = Matrix::reflect_xy();
    /// ```
    pub fn reflect_xy() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(2, 2, -1.0);
        t
    }

    /// Returns a reflection over the y=x axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let reflect = Matrix::reflect_45();
    /// ```
    pub fn reflect_45() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, 0.0);
        t.set(1, 0, 1.0);
        t.set(0, 1, 1.0);
        t.set(1, 1, 0.0);
        t
    }

    /// Returns a reflection over the y=-x axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let reflect = Matrix::reflect_neg45();
    /// ```
    pub fn reflect_neg45() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, 0.0);
        t.set(1, 0, -1.0);
        t.set(0, 1, -1.0);
        t.set(1, 1, 0.0);
        t
    }

    /// Returns a reflection over the origin transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let reflect = Matrix::reflect_origin();
    /// ```
    pub fn reflect_origin() -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, -1.0);
        t.set(1, 1, -1.0);
        t
    }


    /// Returns a translation transformation [Matrix].
    ///
    /// # Arguments
    ///
    /// * `x` - A f64 float specifying the x direction to move.
    /// * `y` - A f64 float specifying the y direction to move.
    /// * `z` - A f64 float specifying the z direction to move.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let translate = Matrix::translate(50.0, -100.0, 0.0);
    /// ```
    pub fn translate(x: f64, y: f64, z: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(3, 0, x);
        t.set(3, 1, y);
        t.set(3, 2, z);
        t
    }

    /// Returns a dilation transformation [Matrix].
    ///
    /// # Arguments
    ///
    /// * `x` - A f64 float specifying the x scale factor.
    /// * `y` - A f64 float specifying the y scale factor.
    /// * `z` - A f64 float specifying the z scale factor.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let dilation = Matrix::scale(0.5, 0.25, 0.1);
    /// ```
    pub fn scale(x: f64, y: f64, z: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 0, x);
        t.set(1, 1, y);
        t.set(2, 2, z);
        t
    }

    #[allow(clippy::many_single_char_names)]
    /// Returns a transformation [Matrix] to allow a vector
    /// to be rotated around any axis.
    ///
    /// # Arguments
    ///
    /// * `theta` - A f64 float representing the angle of rotation.
    /// * `x` - A f64 float specifying the x componet of the specified axis
    /// * `y` - A f64 float specifying the y componet of the specified axis
    /// * `z` - A f64 float specifying the z componet of the specified axis
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let rotate = Matrix::rotate_point(180.0, 100.0, 100.0, 100.0);
    /// ```
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

    /// Returns a rotation over the x axis transformation [Matrix].
    ///
    /// # Arguments
    ///
    /// * `theta` - A f64 float representing the angle of rotation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let rotate = Matrix::rotate_x(45.0);
    /// ```
    pub fn rotate_x(theta: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        let angle = theta.to_radians();
        t.set(1, 1, angle.cos());
        t.set(2, 1, -angle.sin());
        t.set(1, 2, angle.sin());
        t.set(2, 2, angle.cos());
        t
    }

    /// Returns a rotation over the y axis transformation [Matrix].
    ///
    /// # Arguments
    ///
    /// * `theta` - A f64 float representing the angle of rotation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let rotate = Matrix::rotate_y(45.0);
    /// ```
    pub fn rotate_y(theta: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        let angle = theta.to_radians();
        t.set(0, 0, angle.cos());
        t.set(0, 2, -angle.sin());
        t.set(2, 0, angle.sin());
        t.set(2, 2, angle.cos());
        t
    }

    /// Returns a rotation over the z axis transformation [Matrix].
    ///
    /// # Arguments
    ///
    /// * `theta` - A f64 float representing the angle of rotation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let rotate = Matrix::rotate_z(45.0);
    /// ```
    pub fn rotate_z(theta: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        let angle = theta.to_radians();
        t.set(0, 0, angle.cos());
        t.set(1, 0, -angle.sin());
        t.set(0, 1, angle.sin());
        t.set(1, 1, angle.cos());
        t
    }

    /// Returns a shearing over the x axis transformation [Matrix].
    ///
    /// # Arguments
    ///
    /// * `sh_y` - The factor for which the y axis will shear
    /// * `sh_z` - The factor for which the z axis will shear
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let rotate = Matrix::shearing_x(1.3, 0.5);
    /// ```
    pub fn shearing_x(sh_y: f64, sh_z: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(0, 1, sh_y);
        t.set(0, 2, sh_z);
        t
    }

    /// Returns a shearing over the y axis transformation [Matrix].
    ///
    /// # Arguments
    ///
    /// * `sh_x` - The factor for which the x axis will shear
    /// * `sh_z` - The factor for which the z axis will shear
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let rotate = Matrix::shearing_y(1.3, 0.5);
    /// ```
    pub fn shearing_y(sh_x: f64, sh_z: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(1, 0, sh_x);
        t.set(1, 2, sh_z);
        t
    }

    /// Returns a shearing over the z axis transformation [Matrix].
    ///
    /// # Arguments
    ///
    /// * `sh_x` - The factor for which the x axis will shear
    /// * `sh_y` - The factor for which the y axis will shear
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::graphics::matrix::Matrix;
    /// let rotate = Matrix::shearing_z(1.3, 0.5);
    /// ```
    pub fn shearing_z(sh_x: f64, sh_y: f64) -> Self {
        let mut t = Self::identity_matrix(4);
        t.set(2, 0, sh_x);
        t.set(2, 1, sh_y);
        t
    }
}
