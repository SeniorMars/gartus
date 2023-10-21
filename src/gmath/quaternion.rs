/// Represents a quaternion for 3D rotations.
#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
    /// The real component of the quaternion.
    pub w: f64,
    /// The first imaginary component of the quaternion.
    pub x: f64,
    /// The second imaginary component of the quaternion.
    pub y: f64,
    /// The third imaginary component of the quaternion.
    pub z: f64,
}

impl Quaternion {
    #[must_use]
    /// Creates a new quaternion from an angle in radians and an axis of rotation.
    ///
    /// # Arguments
    ///
    /// * `angle` - The angle of rotation in radians.
    /// * `axis` - The axis of rotation as a 3D vector.
    ///
    /// # Returns
    ///
    /// A quaternion representing the rotation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use crate::gartus::gmath::Quaternion;
    /// let quat = Quaternion::from_axis_angle(std::f64::consts::PI / 4.0, [0.0, 1.0, 0.0]);
    /// ```
    pub fn from_axis_angle(angle: f64, axis: [f64; 3]) -> Self {
        let half_angle = angle / 2.0;
        let sin_half_angle = half_angle.sin();
        Quaternion {
            w: half_angle.cos(),
            x: axis[0] * sin_half_angle,
            y: axis[1] * sin_half_angle,
            z: axis[2] * sin_half_angle,
        }
    }

    #[must_use]
    /// Returns the conjugate of the quaternion.
    pub fn conjugate(&self) -> Self {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    #[must_use]
    /// Returns the magnitude (norm) of the quaternion.
    pub fn magnitude(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalizes the quaternion to have a magnitude of 1.
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        self.w /= mag;
        self.x /= mag;
        self.y /= mag;
        self.z /= mag;
    }

    #[must_use]
    /// Rotates a 3D vector by the quaternion.
    ///
    /// # Arguments
    ///
    /// * `v` - The 3D vector to be rotated.
    ///
    /// # Returns
    ///
    /// The rotated 3D vector.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use crate::gartus::gmath::Quaternion;
    /// let quat = Quaternion::from_axis_angle(std::f64::consts::PI / 2.0, [0.0, 1.0, 0.0]);
    /// let vec = [1.0, 0.0, 0.0];
    /// let rotated_vec = quat.rotate_vector(vec);
    /// ```
    pub fn rotate_vector(&self, v: [f64; 3]) -> [f64; 3] {
        let qv = Quaternion {
            w: 0.0,
            x: v[0],
            y: v[1],
            z: v[2],
        };
        let rotated_qv = *self * qv * self.conjugate();
        [rotated_qv.x, rotated_qv.y, rotated_qv.z]
    }
}

// Define quaternion multiplication.
impl std::ops::Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}
