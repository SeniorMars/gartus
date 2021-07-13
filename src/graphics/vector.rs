use std::{
    fmt::{self, Display},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// Representing a ray color
pub type Color = Vector;

#[derive(Debug, Clone, Copy)]
/// A 3D geometric vector
pub struct Vector {
    /// Data of the vector
    pub data: [f64; 3],
}

#[allow(dead_code)]
impl Vector {
    /// Creates a new 3D geometric vector
    ///
    /// # Arguments
    ///
    /// * `x` - The x corrdinate of the vector
    /// * `y` - The y corrdinate of the vector
    /// * `z` - The z corrdinate of the vector
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::vector::Vector;
    /// let vec3 = Vector::new(0.0, 0.0, 0.0);
    /// ```
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { data: [x, y, z] }
    }

    /// Produces a new vector based on the cross product of two vectors
    ///
    /// # Arguments
    ///
    /// * `other` - A vector to be multipled on
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::vector::Vector;
    /// let one = Vector::new(1.0, 1.0, 1.0);
    /// let two = Vector::new(1.0, 2.0, 3.0);
    /// let cross = one.cross(two);
    /// ```
    pub fn cross(self, other: Vector) -> Self {
        Self {
            data: [
                self[1] * other[2] - self[2] * other[1],
                self[2] * other[0] - self[0] * other[2],
                self[0] * other[1] - self[1] * other[0],
            ],
        }
    }

    /// Does dot multiplation between two vector
    ///
    /// # Arguments
    ///
    /// * `other` - A vector to be multipled on
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::vector::Vector;
    /// let one = Vector::new(1.0, 1.0, 1.0);
    /// let two = Vector::new(1.0, 2.0, 3.0);
    /// let dot = one.dot(two);
    /// ```
    pub fn dot(&self, other: Vector) -> f64 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }

    /// Returns the mathematical length of a vector
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::vector::Vector;
    /// let one = Vector::new(1.0, 1.0, 1.0);
    /// let length = one.length();
    /// ```
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Returns a normalized vector
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::vector::Vector;
    /// let one = Vector::new(2.0, 2.0, 2.0);
    /// let normal = one.normalized();
    /// ```
    pub fn normalized(self) -> Self {
        self / self.length()
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
        *self = Self {
            data: [self[0] + other[0], self[1] + other[1], self[2] + other[2]],
        }
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
        *self = Self {
            data: [self[0] - other[0], self[1] - other[1], self[2] - other[2]],
        }
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
        *self = Self {
            data: [self[0] * other, self[1] * other, self[2] * other],
        }
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, other: Vector) -> Self::Output {
        Vector {
            data: [self * other[0], self * other[1], self * other[2]],
        }
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
        *self = Self {
            data: [self[0] / other, self[1] / other, self[2] / other],
        };
    }
}

impl Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6}, {:.6}, {:.6})", self[0], self[1], self[2])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn print_vector() {
        let one = Vector::new(1.0, 1.0, 1.0);
        let two = Vector::new(1.0, 2.0, 3.0);
        let cross = one.cross(two);
        println!("{}", cross);
        println!("{:.6}", cross.length())
    }
}
