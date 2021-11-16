use super::vector::Vector;

#[derive(Clone, Debug, Copy)]
/// A ray that composes of an origin vector and direction vector
pub struct Ray {
    origin: Vector,
    direction: Vector,
}

#[allow(dead_code)]
impl Ray {
    /// Returns a new Ray consisting of an origin and a direction
    ///
    /// # Arguments
    ///
    /// * `origin` - The origin vector
    /// * `direction` - The direction of the ray
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::curves_rs::gmath::vector::Vector;
    /// use crate::curves_rs::gmath::ray::Ray;
    /// let one = Vector::new(1.0, 1.0, 1.0);
    /// let two = Vector::new(1.0, 1.0, 1.0);
    /// let ray = Ray::new(one, two);
    /// ```
    pub fn new(origin: Vector, direction: Vector) -> Self {
        Self { origin, direction }
    }

    /// Get a reference to the ray's direction.
    pub fn direction(&self) -> &Vector {
        &self.direction
    }

    /// Get a reference to the ray's orgin.
    pub fn orgin(&self) -> &Vector {
        &self.origin
    }

    /// Returns the position a ray will be located given a real number
    ///
    /// # Arguments
    ///
    /// * `t` - A real number that will determines where
    /// the ray will be located
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::curves_rs::gmath::vector::Vector;
    /// use crate::curves_rs::gmath::ray::Ray;
    /// let one = Vector::new(1.0, 1.0, 1.0);
    /// let two = Vector::new(1.0, 1.0, 1.0);
    /// let ray = Ray::new(one, two);
    /// let new_loc = ray.at(10.00);
    pub fn at(&self, t: f64) -> Vector {
        self.origin + t * self.direction
    }
}
