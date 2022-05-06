use super::matrix::Matrix;
#[allow(dead_code)]
#[rustfmt::skip]
impl Matrix {
    /// Returns a hermite [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let h = Matrix::hermite();
    /// ```
    #[must_use] pub fn hermite() -> Self {
        Matrix::new(4, 4, vec![0.0, 1.0, 0.0, 3.0,
                               0.0, 1.0, 0.0, 2.0,
                               0.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 0.0, 0.0])
    }

    /// Returns an inverse of the hermite [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let h_inverse = Matrix::inverse_hermite();
    /// ```
    #[must_use] pub fn inverse_hermite() -> Self {
        Matrix::new(4, 4, vec![2.0, -3.0, 0.0, 1.0,
                               -2.0, 3.0, 0.0, 0.0,
                               1.0, -2.0, 1.0, 0.0,
                               1.0, -1.0, 0.0, 0.0])
    }

    /// Returns an inverse of the bezier [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let h_inverse = Matrix::inverse_bezier();
    /// ```
    #[must_use] pub fn inverse_bezier() -> Self {
        Matrix::new(4, 4, vec![-1.0, 3.0, -3.0, 1.0,
                               3.0, -6.0, 3.0, 0.0,
                               -3.0, 3.0, 0.0, 0.0,
                               1.0, 0.0, 0.0, 0.0])
    }

    /// Returns a reflection over the yz (y) axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let reflect = Matrix::reflect_yz();
    /// ```
    #[must_use] pub fn reflect_yz() -> Self {
        Matrix::new(4, 4, vec![-1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
    }

    /// Returns a reflection over the xz (x) axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let reflect = Matrix::reflect_xz();
    /// ```
    #[must_use] pub fn reflect_xz() -> Self {
        Matrix::new(4, 4, vec![1.0, 0.0, 0.0, 0.0,
                               0.0, -1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
    }

    /// Returns a reflection over the xy (y) axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let reflect = Matrix::reflect_xy();
    /// ```
    #[must_use] pub fn reflect_xy() -> Self {
        Matrix::new(4, 4, vec![1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, -1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
    }

    /// Returns a reflection over the y=x axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let reflect = Matrix::reflect_45();
    /// ```
    #[must_use] pub fn reflect_45() -> Self {
        Matrix::new(4, 4, vec![0.0, 1.0, 0.0, 0.0,
                               1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
    }

    /// Returns a reflection over the y=-x axis transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let reflect = Matrix::reflect_neg45();
    /// ```
    #[must_use] pub fn reflect_neg45() -> Self {
        Matrix::new(4, 4, vec![0.0, -1.0, 0.0, 0.0,
                               -1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
    }

    /// Returns a reflection over the origin transformation [Matrix].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let reflect = Matrix::reflect_origin();
    /// ```
    #[must_use] pub fn reflect_origin() -> Self {
        Matrix::new(4, 4, vec![-1.0, 0.0, 0.0, 0.0,
                               0.0, -1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let translate = Matrix::translate(50.0, -100.0, 0.0);
    /// ```
    #[must_use] pub fn translate(x: f64, y: f64, z: f64) -> Self {
        Matrix::new(4, 4, vec![1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               x, y, z, 1.0])
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let dilation = Matrix::scale(0.5, 0.25, 0.1);
    /// ```
    #[must_use] pub fn scale(x: f64, y: f64, z: f64) -> Self {
        Matrix::new(4, 4, vec![x, 0.0, 0.0, 0.0,
                               0.0, y, 0.0, 0.0,
                               0.0, 0.0, z, 0.0,
                               0.0, 0.0, 0.0, 1.0])
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let rotate = Matrix::rotate_point(180.0, 100.0, 100.0, 100.0);
    /// ```
    #[must_use] pub fn rotate_point(theta: f64, x: f64, y: f64, z: f64) -> Self {
        let angle = theta.to_radians();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        Matrix::new(
            4,
            4,
            vec![
                (t * x * x) + c,
                (t * x * y) - (s * z),
                (t * x * z) + (s * z),
                0.0,
                (t * x * y) + (s * z),
                (t * y * y) + c,
                (t * y * z) - (s * x),
                0.0,
                (t * x * z) - (s * y),
                (t * y * z) + (s * x),
                (t * z * z) + c,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let rotate = Matrix::rotate_x(45.0);
    /// ```
    #[must_use] pub fn rotate_x(theta: f64) -> Self {
        let angle = theta.to_radians();
        Matrix::new(4, 4, vec![1.0, 0.0, 0.0, 0.0,
                               0.0, angle.cos(), angle.sin(), 0.0,
                               0.0, -angle.sin(), angle.cos(), 0.0,
                               0.0, 0.0, 0.0, 1.0])
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let rotate = Matrix::rotate_y(45.0);
    /// ```
    #[must_use] pub fn rotate_y(theta: f64) -> Self {
        let angle = theta.to_radians();
        Matrix::new(4, 4, vec![angle.cos(), 0.0, -angle.sin(), 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               angle.sin(), 0.0, angle.cos(), 0.0,
                               0.0, 0.0, 0.0, 1.0])
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let rotate = Matrix::rotate_z(45.0);
    /// ```
    #[must_use] pub fn rotate_z(theta: f64) -> Self {
        let angle = theta.to_radians();
        Matrix::new(4, 4, vec![angle.cos(), angle.sin(), 0.0, 0.0,
                               -angle.sin(), angle.cos(), 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let rotate = Matrix::shearing_x(1.3, 0.5);
    /// ```
    #[must_use] pub fn shearing_x(sh_y: f64, sh_z: f64) -> Self {
        Matrix::new(4, 4, vec![1.0, sh_y, sh_z, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let rotate = Matrix::shearing_y(1.3, 0.5);
    /// ```
    #[must_use] pub fn shearing_y(sh_x: f64, sh_z: f64) -> Self {
        Matrix::new(4, 4, vec![1.0, 0.0, 0.0, 0.0,
                               sh_x, 1.0, sh_z, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
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
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let rotate = Matrix::shearing_z(1.3, 0.5);
    /// ```
    #[must_use] pub fn shearing_z(sh_x: f64, sh_y: f64) -> Self {
        Matrix::new(4, 4, vec![1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               sh_x, sh_y, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
    }
}

// #[rustfmt::skip]
// impl<const SIZE: usize> ConstMatrix<{ SIZE }> {
//     /// Returns a hermite [ConstMatrix].
//     ///
//     /// # Examples
//     ///
//     /// Basic usage:
//     /// ```
//     /// use crate::gartus::gmath::matrix::ConstMatrix;
//     /// let h = ConstMatrix::hermite();
//     /// ```
//     pub fn hermite() -> Self {
//         ConstMatrix {
//             data: [0.0, 1.0, 0.0, 3.0,
//                 0.0, 1.0, 0.0, 2.0,
//                 0.0, 1.0, 1.0, 1.0,
//                 1.0, 1.0, 0.0, 0.0],
//         }
//     }
// }
//
// #[test]
// // #[should_panic]
// fn print_matrices() {
//     println!("{}", Matrix::identity_matrix(4));
// }
