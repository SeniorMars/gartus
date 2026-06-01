use std::{
    fmt::{self, Display},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// Represents a point in 3D space.
///
/// Semantically, a point describes a location.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    /// Data of the point
    pub data: [f64; 3],
}

impl Point {
    /// Creates a new 3D point.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { data: [x, y, z] }
    }
}

impl Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({:.2}, {:.2}, {:.2})",
            self.data[0], self.data[1], self.data[2]
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
/// A 3D geometric vector.
///
/// Semantically, a vector describes a relationship between two points (direction and magnitude).
pub struct Vector {
    /// Data of the vector
    pub data: [f64; 3],
}

impl Vector {
    /// Creates a new 3D geometric vector.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { data: [x, y, z] }
    }

    /// Calculates the vector between two points (p1 - p0).
    #[must_use]
    pub fn between(p0: Point, p1: Point) -> Self {
        Self::new(
            p1.data[0] - p0.data[0],
            p1.data[1] - p0.data[1],
            p1.data[2] - p0.data[2],
        )
    }

    /// Produces a new vector based on the cross product of two vectors.
    #[must_use]
    pub fn cross(self, other: Vector) -> Self {
        Self {
            data: [
                self[1] * other[2] - self[2] * other[1],
                self[2] * other[0] - self[0] * other[2],
                self[0] * other[1] - self[1] * other[0],
            ],
        }
    }

    /// Does dot multiplication between two vectors.
    #[must_use]
    pub fn dot(&self, other: Vector) -> f64 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }

    /// Returns the mathematical length (magnitude) of a vector.
    #[must_use]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Returns a normalized vector.
    #[must_use]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < f64::EPSILON {
            Self::default()
        } else {
            self / len
        }
    }
}

impl Index<usize> for Vector {
    type Output = f64;
    fn index(&self, index: usize) -> &f64 {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.data[index]
    }
}

impl Add for Vector {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            data: [self[0] + other[0], self[1] + other[1], self[2] + other[2]],
        }
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, other: Self) {
        self.data[0] += other[0];
        self.data[1] += other[1];
        self.data[2] += other[2];
    }
}

impl Sub for Vector {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self {
            data: [self[0] - other[0], self[1] - other[1], self[2] - other[2]],
        }
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, other: Self) {
        self.data[0] -= other[0];
        self.data[1] -= other[1];
        self.data[2] -= other[2];
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;
    fn mul(self, other: f64) -> Self::Output {
        Self {
            data: [self[0] * other, self[1] * other, self[2] * other],
        }
    }
}

impl MulAssign<f64> for Vector {
    fn mul_assign(&mut self, other: f64) {
        self.data[0] *= other;
        self.data[1] *= other;
        self.data[2] *= other;
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;
    fn mul(self, other: Vector) -> Self::Output {
        other * self
    }
}

impl Div<f64> for Vector {
    type Output = Self;
    fn div(self, other: f64) -> Self {
        Self {
            data: [self[0] / other, self[1] / other, self[2] / other],
        }
    }
}

impl DivAssign<f64> for Vector {
    fn div_assign(&mut self, other: f64) {
        self.data[0] /= other;
        self.data[1] /= other;
        self.data[2] /= other;
    }
}

impl Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{:.2}, {:.2}, {:.2}>", self[0], self[1], self[2])
    }
}

// Interop between Point and Vector
impl Sub<Point> for Point {
    type Output = Vector;
    fn sub(self, other: Point) -> Vector {
        Vector::between(other, self)
    }
}

impl Add<Vector> for Point {
    type Output = Point;
    fn add(self, other: Vector) -> Point {
        Point::new(
            self.data[0] + other[0],
            self.data[1] + other[1],
            self.data[2] + other[2],
        )
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod test {
    use super::*;

    #[test]
    fn test_vector_between_points() {
        let p0 = Point::new(4.0, 10.0, 0.0);
        let p1 = Point::new(6.0, 5.0, 23.0);

        let v = Vector::between(p0, p1);
        assert_eq!(v.data, [2.0, -5.0, 23.0]);

        let v_rev = Vector::between(p1, p0);
        assert_eq!(v_rev.data, [-2.0, 5.0, -23.0]);

        // Test subtraction operator
        let v_op = p1 - p0;
        assert_eq!(v_op.data, [2.0, -5.0, 23.0]);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector::new(1.0, 2.0, 3.0);
        let v2 = Vector::new(4.0, 5.0, 6.0);
        assert_eq!(v1.dot(v2), 4.0 + 10.0 + 18.0);
    }

    #[test]
    fn test_cross_product() {
        let v1 = Vector::new(1.0, 0.0, 0.0);
        let v2 = Vector::new(0.0, 1.0, 0.0);
        let cross = v1.cross(v2);
        assert_eq!(cross.data, [0.0, 0.0, 1.0]);

        let v3 = Vector::new(1.0, 2.0, 3.0);
        let v4 = Vector::new(4.0, 5.0, 6.0);
        let cross2 = v3.cross(v4);
        // [2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4] = [12-15, 12-6, 5-8] = [-3, 6, -3]
        assert_eq!(cross2.data, [-3.0, 6.0, -3.0]);
    }

    #[test]
    fn test_magnitude_and_normalize() {
        let v = Vector::new(3.0, 4.0, 0.0);
        assert_eq!(v.length(), 5.0);

        let norm = v.normalized();
        assert_eq!(norm.data, [0.6, 0.8, 0.0]);
        assert!((norm.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_display_formats() {
        let p = Point::new(1.0, 2.0, 3.0);
        let v = Vector::new(1.0, 2.0, 3.0);

        assert_eq!(format!("{p}"), "(1.00, 2.00, 3.00)");
        assert_eq!(format!("{v}"), "<1.00, 2.00, 3.00>");
    }

    #[test]
    fn test_point_vector_addition() {
        let p = Point::new(1.0, 2.0, 3.0);
        let v = Vector::new(10.0, 20.0, 30.0);
        let p2 = p + v;
        assert_eq!(p2.data, [11.0, 22.0, 33.0]);
    }
}
