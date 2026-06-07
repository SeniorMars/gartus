use super::vector::{Point, Vector};

#[derive(Clone, Debug, Copy, PartialEq)]
/// A ray composed of an origin point and a direction vector.
pub struct Ray {
    origin: Point,
    direction: Vector,
    time: f64,
}

impl Ray {
    /// Returns a new ray consisting of an origin and a direction.
    ///
    /// # Arguments
    ///
    /// * `origin` - The point where the ray starts
    /// * `direction` - The direction of the ray
    ///
    /// # Examples
    ///
    /// ```
    /// use gartus::gmath::{ray::Ray, vector::{Point, Vector}};
    /// let origin = Point::new(0.0, 0.0, 0.0);
    /// let direction = Vector::new(1.0, 1.0, 1.0);
    /// let ray = Ray::new(origin, direction);
    /// ```
    #[must_use]
    pub fn new(origin: Point, direction: Vector) -> Self {
        Self::with_time(origin, direction, 0.0)
    }

    /// Returns a new ray with an explicit shutter time.
    ///
    /// # Panics
    /// Panics if `time` is not finite. Debug builds also check that origin and direction
    /// components are finite.
    #[must_use]
    pub fn with_time(origin: Point, direction: Vector, time: f64) -> Self {
        assert!(time.is_finite(), "ray time must be finite");
        debug_assert!(origin.is_finite(), "ray origin should be finite");
        debug_assert!(direction.is_finite(), "ray direction should be finite");
        Self {
            origin,
            direction,
            time,
        }
    }

    /// Returns a checked ray consisting of an origin and a direction.
    ///
    /// Zero-length directions are accepted; only non-finite values are rejected.
    #[must_use]
    pub fn try_new(origin: Point, direction: Vector) -> Option<Self> {
        Self::try_with_time(origin, direction, 0.0)
    }

    /// Returns a checked ray with an explicit shutter time.
    ///
    /// Zero-length directions are accepted; only non-finite origin, direction, or time values are
    /// rejected.
    #[must_use]
    pub fn try_with_time(origin: Point, direction: Vector, time: f64) -> Option<Self> {
        (origin.is_finite() && direction.is_finite() && time.is_finite()).then_some(Self {
            origin,
            direction,
            time,
        })
    }

    /// Get a reference to the ray's direction.
    #[must_use]
    pub fn direction(&self) -> &Vector {
        &self.direction
    }

    /// Get a reference to the ray's origin.
    #[must_use]
    pub fn origin(&self) -> &Point {
        &self.origin
    }

    /// Returns the shutter time carried by this ray.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get a reference to the ray's origin.
    #[deprecated(note = "use origin instead")]
    #[doc(hidden)]
    #[must_use]
    pub fn orgin(&self) -> &Point {
        self.origin()
    }

    /// Returns the position along the ray at parameter `t`.
    ///
    /// # Arguments
    ///
    /// * `t` - A real number that determines where along the ray to sample
    ///
    /// # Examples
    ///
    /// ```
    /// use gartus::gmath::{ray::Ray, vector::{Point, Vector}};
    /// let origin = Point::new(1.0, 1.0, 1.0);
    /// let direction = Vector::new(1.0, 1.0, 1.0);
    /// let ray = Ray::new(origin, direction);
    /// let new_loc = ray.at(10.00);
    /// ```
    #[must_use]
    pub fn at(&self, t: f64) -> Point {
        self.origin + t * self.direction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_defaults_to_zero_time_and_can_store_time() {
        let origin = Point::new(0.0, 0.0, 0.0);
        let direction = Vector::new(1.0, 0.0, 0.0);

        assert!((Ray::new(origin, direction).time() - 0.0).abs() < 1e-10);
        assert!((Ray::with_time(origin, direction, 0.375).time() - 0.375).abs() < 1e-10);
    }

    #[test]
    fn checked_ray_constructors_reject_non_finite_values() {
        let origin = Point::new(0.0, 0.0, 0.0);
        let direction = Vector::new(1.0, 0.0, 0.0);

        assert!(Ray::try_new(origin, direction).is_some());
        assert_eq!(Ray::try_with_time(origin, direction, f64::NAN), None);
        assert_eq!(
            Ray::try_new(
                origin,
                Vector {
                    data: [1.0, f64::NAN, 0.0]
                }
            ),
            None
        );
    }
}
