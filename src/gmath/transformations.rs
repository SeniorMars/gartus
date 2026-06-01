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
    pub fn hermite() -> Self {
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
    pub fn inverse_hermite() -> Self {
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
    pub fn inverse_bezier() -> Self {
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
    pub fn reflect_yz() -> Self {
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
    pub fn reflect_xz() -> Self {
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
    pub fn reflect_xy() -> Self {
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
    pub fn reflect_45() -> Self {
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
    pub fn reflect_neg45() -> Self {
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
    pub fn reflect_origin() -> Self {
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
    pub fn translate(x: f64, y: f64, z: f64) -> Self {
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
    pub fn scale(x: f64, y: f64, z: f64) -> Self {
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
    /// # Panics
    /// Panics if the rotation axis `(x, y, z)` is a zero vector.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let rotate = Matrix::rotate_point(180.0, 100.0, 100.0, 100.0);
    /// ```
    pub fn rotate_point(theta: f64, x: f64, y: f64, z: f64) -> Self {
        let len = (x * x + y * y + z * z).sqrt();
        assert!(len > f64::EPSILON, "rotation axis must be non-zero");
        let (x, y, z) = (x / len, y / len, z / len);
        let angle = theta.to_radians();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        Matrix::new(
            4,
            4,
            vec![
                (t * x * x) + c,
                (t * x * y) + (s * z),
                (t * x * z) - (s * y),
                0.0,
                (t * x * y) - (s * z),
                (t * y * y) + c,
                (t * y * z) + (s * x),
                0.0,
                (t * x * z) + (s * y),
                (t * y * z) - (s * x),
                (t * z * z) + c,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )
    }


    /// Basic usage:
    ///
    /// ```
    /// use gartus::gmath::matrix::Matrix;
    /// let ortho_matrix = Matrix::orthographic_projection(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
    /// ```
    pub fn orthographic_projection(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        Matrix::new(
            4,
            4,
            vec![
                2.0 / (right - left), 0.0, 0.0, 0.0,
                0.0, 2.0 / (top - bottom), 0.0, 0.0,
                0.0, 0.0, -2.0 / (far - near), 0.0,
                -(right + left) / (right - left),
                -(top + bottom) / (top - bottom),
                -(far + near) / (far - near),
                1.0,
            ],
        )
    }

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
                1.0 / (angle / 2.0).tan() / aspect_ratio, 0.0, 0.0, 0.0,
                0.0, 1.0 / (angle / 2.0).tan(), 0.0, 0.0,
                0.0, 0.0, -(far + near) / (far - near), -1.0,
                0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0,
            ],
        )
    }

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
                width as f64 / 2.0, 0.0, 0.0, 0.0,
                0.0, -(height as f64 / 2.0), 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                x as f64 + width as f64 / 2.0,
                y as f64 + height as f64 / 2.0,
                0.0,
                1.0,
            ],
        )
    }

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
    /// # Panics
    /// Panics if `eye` and `target` are identical, or if `up` is parallel to the view direction.
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
        const EPS: f64 = 1e-12;

        let f = {
            let forward = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
            let length = (forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]).sqrt();
            assert!(length > EPS, "look_at requires eye and target to differ");
            [forward[0] / length, forward[1] / length, forward[2] / length]
        };

        let r = {
            let right = [
                f[1] * up[2] - f[2] * up[1],
                f[2] * up[0] - f[0] * up[2],
                f[0] * up[1] - f[1] * up[0],
            ];
            let length = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
            assert!(length > EPS, "look_at up vector must not be parallel to view direction");
            [right[0] / length, right[1] / length, right[2] / length]
        };

        let u = [
            r[1] * f[2] - r[2] * f[1],
            r[2] * f[0] - r[0] * f[2],
            r[0] * f[1] - r[1] * f[0],
        ];

        Matrix::new(
            4,
            4,
            vec![
                r[0], u[0], -f[0], 0.0,
                r[1], u[1], -f[1], 0.0,
                r[2], u[2], -f[2], 0.0,
                -r[0] * eye[0] - r[1] * eye[1] - r[2] * eye[2],
                -u[0] * eye[0] - u[1] * eye[1] - u[2] * eye[2],
                f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2],
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
    pub fn rotate_x(theta: f64) -> Self {
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
    pub fn rotate_y(theta: f64) -> Self {
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
    pub fn rotate_z(theta: f64) -> Self {
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
    pub fn shearing_x(sh_y: f64, sh_z: f64) -> Self {
        Matrix::new(4, 4, vec![1.0, 0.0, 0.0, 0.0,
                               sh_y, 1.0, 0.0, 0.0,
                               sh_z, 0.0, 1.0, 0.0,
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
    pub fn shearing_y(sh_x: f64, sh_z: f64) -> Self {
        Matrix::new(4, 4, vec![1.0, sh_x, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, sh_z, 1.0, 0.0,
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
    pub fn shearing_z(sh_x: f64, sh_y: f64) -> Self {
        Matrix::new(4, 4, vec![1.0, 0.0, sh_x, 0.0,
                               0.0, 1.0, sh_y, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmath::vector::Vector;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // 1. rotate_point axis normalization
    #[test]
    fn rotate_point_axis_normalization() {
        let unnorm = Matrix::rotate_point(45.0, 1.0, 1.0, 1.0);
        let s = 3f64.sqrt();
        let norm = Matrix::rotate_point(45.0, 1.0 / s, 1.0 / s, 1.0 / s);
        for row in 0..4 {
            for col in 0..4 {
                assert!(
                    approx_eq(unnorm[(row, col)], norm[(row, col)]),
                    "mismatch at ({row},{col}): {} vs {}",
                    unnorm[(row, col)],
                    norm[(row, col)]
                );
            }
        }
    }

    // 2. rotate_point identity: 0 degrees -> identity matrix
    #[test]
    fn rotate_point_zero_degrees_is_identity() {
        let m = Matrix::rotate_point(0.0, 0.0, 0.0, 1.0);
        let ident = Matrix::identity_matrix(4);
        for row in 0..4 {
            for col in 0..4 {
                assert!(
                    approx_eq(m[(row, col)], ident[(row, col)]),
                    "mismatch at ({row},{col}): {} vs {}",
                    m[(row, col)],
                    ident[(row, col)]
                );
            }
        }
    }

    // 3. rotate_point(90, 0, 0, 1) == rotate_z(90)
    #[test]
    fn rotate_point_z_matches_rotate_z() {
        let rp = Matrix::rotate_point(90.0, 0.0, 0.0, 1.0);
        let rz = Matrix::rotate_z(90.0);
        for row in 0..4 {
            for col in 0..4 {
                assert!(
                    approx_eq(rp[(row, col)], rz[(row, col)]),
                    "mismatch at ({row},{col}): {} vs {}",
                    rp[(row, col)],
                    rz[(row, col)]
                );
            }
        }
    }

    // 4. rotate_point(90, 1, 0, 0) == rotate_x(90)
    #[test]
    fn rotate_point_x_matches_rotate_x() {
        let rp = Matrix::rotate_point(90.0, 1.0, 0.0, 0.0);
        let rx = Matrix::rotate_x(90.0);
        for row in 0..4 {
            for col in 0..4 {
                assert!(
                    approx_eq(rp[(row, col)], rx[(row, col)]),
                    "mismatch at ({row},{col}): {} vs {}",
                    rp[(row, col)],
                    rx[(row, col)]
                );
            }
        }
    }

    // 5. look_at canonical: camera at origin looking down -Z with Y up -> identity
    #[test]
    fn look_at_canonical_is_identity() {
        let m = Matrix::look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        assert!(
            approx_eq(m[(0, 0)], 1.0),
            "M[0][0] should be 1, got {}",
            m[(0, 0)]
        );
        assert!(
            approx_eq(m[(1, 1)], 1.0),
            "M[1][1] should be 1, got {}",
            m[(1, 1)]
        );
        assert!(
            approx_eq(m[(2, 2)], 1.0),
            "M[2][2] should be 1, got {}",
            m[(2, 2)]
        );
        assert!(
            approx_eq(m[(3, 3)], 1.0),
            "M[3][3] should be 1, got {}",
            m[(3, 3)]
        );
        // off-diagonal elements in the rotation part should be zero
        assert!(
            approx_eq(m[(0, 1)], 0.0),
            "M[0][1] should be 0, got {}",
            m[(0, 1)]
        );
        assert!(
            approx_eq(m[(0, 2)], 0.0),
            "M[0][2] should be 0, got {}",
            m[(0, 2)]
        );
        assert!(
            approx_eq(m[(1, 0)], 0.0),
            "M[1][0] should be 0, got {}",
            m[(1, 0)]
        );
        assert!(
            approx_eq(m[(1, 2)], 0.0),
            "M[1][2] should be 0, got {}",
            m[(1, 2)]
        );
        assert!(
            approx_eq(m[(2, 0)], 0.0),
            "M[2][0] should be 0, got {}",
            m[(2, 0)]
        );
        assert!(
            approx_eq(m[(2, 1)], 0.0),
            "M[2][1] should be 0, got {}",
            m[(2, 1)]
        );
    }

    // 6. look_at canonical: right vector is row 0 = [1,0,0], up vector is row 1 = [0,1,0]
    #[test]
    fn look_at_canonical_right_and_up_vectors() {
        let m = Matrix::look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        // right vector: row 0, cols 0..2
        assert!(
            approx_eq(m[(0, 0)], 1.0),
            "right.x should be 1, got {}",
            m[(0, 0)]
        );
        assert!(
            approx_eq(m[(0, 1)], 0.0),
            "right.y should be 0, got {}",
            m[(0, 1)]
        );
        assert!(
            approx_eq(m[(0, 2)], 0.0),
            "right.z should be 0, got {}",
            m[(0, 2)]
        );
        // up vector: row 1, cols 0..2
        assert!(
            approx_eq(m[(1, 0)], 0.0),
            "up.x should be 0, got {}",
            m[(1, 0)]
        );
        assert!(
            approx_eq(m[(1, 1)], 1.0),
            "up.y should be 1, got {}",
            m[(1, 1)]
        );
        assert!(
            approx_eq(m[(1, 2)], 0.0),
            "up.z should be 0, got {}",
            m[(1, 2)]
        );
    }

    #[test]
    fn look_at_translates_eye_to_origin() {
        let m = Matrix::look_at([1.0, 2.0, 3.0], [1.0, 2.0, 2.0], [0.0, 1.0, 0.0]);
        let eye = m.mult_vector(Vector::new(1.0, 2.0, 3.0));
        let target = m.mult_vector(Vector::new(1.0, 2.0, 2.0));

        assert!(approx_eq(eye[0], 0.0), "eye.x should be 0, got {}", eye[0]);
        assert!(approx_eq(eye[1], 0.0), "eye.y should be 0, got {}", eye[1]);
        assert!(approx_eq(eye[2], 0.0), "eye.z should be 0, got {}", eye[2]);
        assert!(
            approx_eq(target[2], -1.0),
            "target.z should be -1, got {}",
            target[2]
        );
    }

    #[test]
    #[should_panic(expected = "look_at requires eye and target to differ")]
    fn look_at_rejects_equal_eye_and_target() {
        let _ = Matrix::look_at([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "look_at up vector must not be parallel")]
    fn look_at_rejects_parallel_up() {
        let _ = Matrix::look_at([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]);
    }

    #[test]
    fn mult_vector_applies_column_major_translation() {
        let m = Matrix::translate(5.0, -2.0, 3.0).mult_matrix(&Matrix::scale(2.0, 3.0, 4.0));
        let result = m.mult_vector(Vector::new(1.0, 2.0, 3.0));

        assert!(
            approx_eq(result[0], 7.0),
            "x should be 7, got {}",
            result[0]
        );
        assert!(
            approx_eq(result[1], 4.0),
            "y should be 4, got {}",
            result[1]
        );
        assert!(
            approx_eq(result[2], 15.0),
            "z should be 15, got {}",
            result[2]
        );
    }

    // 7. orthographic_projection maps bounds
    #[test]
    fn orthographic_projection_maps_bounds() {
        let m = Matrix::orthographic_projection(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        // center (0,0,0) -> (0,0,0)
        let center = m.mult_vector(Vector::new(0.0, 0.0, 0.0));
        assert!(
            approx_eq(center[0], 0.0),
            "center.x should be 0, got {}",
            center[0]
        );
        assert!(
            approx_eq(center[1], 0.0),
            "center.y should be 0, got {}",
            center[1]
        );
        assert!(
            approx_eq(center[2], 0.0),
            "center.z should be 0, got {}",
            center[2]
        );
        // corner (1,1,1) -> (1,1,-1) in NDC
        let corner = m.mult_vector(Vector::new(1.0, 1.0, 1.0));
        assert!(
            approx_eq(corner[0], 1.0),
            "corner.x should be 1, got {}",
            corner[0]
        );
        assert!(
            approx_eq(corner[1], 1.0),
            "corner.y should be 1, got {}",
            corner[1]
        );
        assert!(
            approx_eq(corner[2], -1.0),
            "corner.z should be -1, got {}",
            corner[2]
        );
    }

    // 8. viewport center mapping: NDC (0,0) -> screen center (400, 300)
    #[test]
    fn viewport_center_mapping() {
        let m = Matrix::viewport(0, 0, 800, 600);
        let ndc_center = m.mult_vector(Vector::new(0.0, 0.0, 0.0));
        assert!(
            approx_eq(ndc_center[0], 400.0),
            "screen x should be 400, got {}",
            ndc_center[0]
        );
        assert!(
            approx_eq(ndc_center[1], 300.0),
            "screen y should be 300, got {}",
            ndc_center[1]
        );
    }

    // 9. shearing_x correct direction: [0,1,0] with factor 2 -> x gets +2
    #[test]
    fn shearing_x_correct_direction() {
        let m = Matrix::shearing_x(2.0, 0.0);
        let result = m.mult_vector(Vector::new(0.0, 1.0, 0.0));
        assert!(
            approx_eq(result[0], 2.0),
            "x should be 2, got {}",
            result[0]
        );
        assert!(
            approx_eq(result[1], 1.0),
            "y should be 1, got {}",
            result[1]
        );
        assert!(
            approx_eq(result[2], 0.0),
            "z should be 0, got {}",
            result[2]
        );
    }

    // 10. shearing_y correct direction: [1,0,0] with factor 3 -> y gets +3
    #[test]
    fn shearing_y_correct_direction() {
        let m = Matrix::shearing_y(3.0, 0.0);
        let result = m.mult_vector(Vector::new(1.0, 0.0, 0.0));
        assert!(
            approx_eq(result[0], 1.0),
            "x should be 1, got {}",
            result[0]
        );
        assert!(
            approx_eq(result[1], 3.0),
            "y should be 3, got {}",
            result[1]
        );
        assert!(
            approx_eq(result[2], 0.0),
            "z should be 0, got {}",
            result[2]
        );
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
