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
    #[must_use]
    pub fn rotate_point(theta: f64, x: f64, y: f64, z: f64) -> Self {
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


    #[must_use]
    /// Creates an orthographic projection matrix for 3D graphics.
    ///
    /// The orthographic projection matrix is used to project 3D coordinates onto a 2D plane
    /// without considering perspective. It defines a viewing volume where objects inside
    /// this volume are displayed on the 2D plane.
    ///
    /// # Arguments
    ///
    /// * `left` - The left coordinate of the view volume.
    /// * `right` - The right coordinate of the view volume.
    /// * `bottom` - The bottom coordinate of the view volume.
    /// * `top` - The top coordinate of the view volume.
    /// * `near` - The near clipping plane distance.
    /// * `far` - The far clipping plane distance.
    ///
    /// # Returns
    ///
    /// A 4x4 orthographic projection matrix.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use transform_rs::graphics::matrix::Matrix;
    /// let ortho_matrix = Matrix::orthographic_projection(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
    /// ```
    pub fn orthographic_projection(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        Matrix::new(
            4,
            4,
            vec![
                2.0 / (right - left),
                0.0,
                0.0,
                -(right + left) / (right - left),
                0.0,
                2.0 / (top - bottom),
                0.0,
                -(top + bottom) / (top - bottom),
                0.0,
                0.0,
                -2.0 / (far - near),
                -(far + near) / (far - near),
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )
    }

    #[must_use]
    /// Creates a perspective projection matrix for 3D graphics.
    ///
    /// The perspective projection matrix is used to project 3D coordinates onto a 2D plane
    /// while considering perspective, simulating the effect of objects appearing smaller as
    /// they move farther from the viewer.
    ///
    /// # Arguments
    ///
    /// * `theta` - The field of view angle in degrees, which will be converted to radians.
    /// * `aspect_ratio` - The aspect ratio (width divided by height) of the view.
    /// * `near` - The near clipping plane distance.
    /// * `far` - The far clipping plane distance.
    ///
    /// # Returns
    ///
    /// A 4x4 perspective projection matrix.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let perspective_matrix = Matrix::perspective_projection(45.0, 16.0 / 9.0, 0.1, 100.0);
    /// ```
    pub fn perspective_projection(theta: f64, aspect_ratio: f64, near: f64, far: f64) -> Self {
        let angle = theta.to_radians();
        Matrix::new(
            4,
            4,
            vec![
                1.0 / (angle / 2.0).tan() / aspect_ratio,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 / (angle / 2.0).tan(),
                0.0,
                0.0,
                0.0,
                0.0,
                -(far + near) / (far - near),
                -(2.0 * far * near) / (far - near),
                0.0,
                0.0,
                -1.0,
                0.0,
            ],
        )
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    /// Creates a viewport transformation matrix for mapping normalized device coordinates to screen coordinates.
    ///
    /// The viewport transformation matrix is used to map coordinates from normalized device space (-1 to 1) to
    /// screen coordinates (pixel coordinates).
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the viewport's top-left corner in pixels.
    /// * `y` - The y-coordinate of the viewport's top-left corner in pixels.
    /// * `width` - The width of the viewport in pixels.
    /// * `height` - The height of the viewport in pixels.
    ///
    /// # Returns
    ///
    /// A 4x4 viewport transformation matrix.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let viewport_matrix = Matrix::viewport(0, 0, 800, 600);
    /// ```
    pub fn viewport(x: usize, y: usize, width: usize, height: usize) -> Self {
        Matrix::new(
            4,
            4,
            vec![
                width as f64 / 2.0,
                0.0,
                0.0,
                x as f64 + width as f64 / 2.0,
                0.0,
                -(height as f64 / 2.0), // Note the negative scaling in the Y direction.
                0.0,
                y as f64 + height as f64 / 2.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )
    }

    #[must_use]
    /// Creates a `LookAt` transformation matrix for positioning a camera in 3D space.
    ///
    /// The `LookAt` transformation matrix is used to position the camera in a 3D scene by specifying
    /// the camera's position, target point, and up direction.
    ///
    /// # Arguments
    ///
    /// * `eye` - The position of the camera (eye point).
    /// * `target` - The target point the camera is looking at.
    /// * `up` - The up direction of the camera.
    ///
    /// # Returns
    ///
    /// A 4x4 `LookAt` transformation matrix.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let camera_matrix = Matrix::look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
    /// ```
    pub fn look_at(eye: [f64; 3], target: [f64; 3], up: [f64; 3]) -> Self {
        let f = {
            let forward = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
            let length = (forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]).sqrt();
            [forward[0] / length, forward[1] / length, forward[2] / length]
        };

        let r = {
            let right = [
                up[1] * f[2] - up[2] * f[1],
                up[2] * f[0] - up[0] * f[2],
                up[0] * f[1] - up[1] * f[0],
            ];
            let length = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
            [right[0] / length, right[1] / length, right[2] / length]
        };

        let u = {
            let up_normalized = [
                up[0] / (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt(),
                up[1] / (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt(),
                up[2] / (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt(),
            ];
            [up_normalized[0], up_normalized[1], up_normalized[2]]
        };

        Matrix::new(
            4,
            4,
            vec![
                r[0], r[1], r[2], -r[0] * eye[0] - r[1] * eye[1] - r[2] * eye[2],
                u[0], u[1], u[2], -u[0] * eye[0] - u[1] * eye[1] - u[2] * eye[2],
               -f[0],-f[1],-f[2],  f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2],
                0.0,   0.0,   0.0,   1.0,
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
