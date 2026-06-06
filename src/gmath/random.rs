//! Deterministic random sampling helpers for graphics algorithms.

use super::vector::Vector;
use rand::{Rng, SeedableRng, rngs::SmallRng};

/// Small deterministic random-number generator for graphics samples.
#[derive(Clone, Debug)]
pub struct SampleRng {
    inner: SmallRng,
}

impl SampleRng {
    /// Creates a sample RNG from a seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            inner: SmallRng::seed_from_u64(seed),
        }
    }

    /// Returns a random real in `[0, 1)`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn random_double(&mut self) -> f64 {
        let bits = self.inner.next_u64() >> 11;
        bits as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Returns a random real in `[min, max)`.
    #[must_use]
    pub fn random_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.random_double()
    }

    /// Returns a random vector with each component in `[0, 1)`.
    #[must_use]
    pub fn random_vector(&mut self) -> Vector {
        Vector::new(
            self.random_double(),
            self.random_double(),
            self.random_double(),
        )
    }

    /// Returns a random vector with each component in `[min, max)`.
    #[must_use]
    pub fn random_vector_range(&mut self, min: f64, max: f64) -> Vector {
        Vector::new(
            self.random_range(min, max),
            self.random_range(min, max),
            self.random_range(min, max),
        )
    }

    /// Returns a uniformly random unit vector using rejection sampling.
    #[must_use]
    pub fn random_unit_vector(&mut self) -> Vector {
        loop {
            let point = self.random_vector_range(-1.0, 1.0);
            let length_squared = point.length_squared();
            if 1e-160 < length_squared && length_squared <= 1.0 {
                return point / length_squared.sqrt();
            }
        }
    }

    /// Returns a random unit vector on the same hemisphere as `normal`.
    #[must_use]
    pub fn random_on_hemisphere(&mut self, normal: Vector) -> Vector {
        let on_unit_sphere = self.random_unit_vector();
        if on_unit_sphere.dot(normal) > 0.0 {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }

    /// Returns a random point inside the unit disk in the xy plane.
    #[must_use]
    pub fn random_in_unit_disk(&mut self) -> Vector {
        loop {
            let point = Vector::new(
                self.random_range(-1.0, 1.0),
                self.random_range(-1.0, 1.0),
                0.0,
            );
            if point.length_squared() < 1.0 {
                return point;
            }
        }
    }
}

impl Default for SampleRng {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn random_double_returns_values_in_half_open_range() {
        let mut rng = SampleRng::new(7);

        for _ in 0..100 {
            let value = rng.random_double();
            assert!((0.0..1.0).contains(&value));
            let ranged = rng.random_range(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&ranged));
        }
    }

    #[test]
    fn random_unit_vector_is_unit_length() {
        let mut rng = SampleRng::new(11);

        for _ in 0..20 {
            let vector = rng.random_unit_vector();
            assert!((vector.length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn random_on_hemisphere_matches_normal_side() {
        let mut rng = SampleRng::new(13);
        let normal = Vector::new(0.0, 1.0, 0.0);

        for _ in 0..20 {
            assert!(rng.random_on_hemisphere(normal).dot(normal) > 0.0);
        }
    }

    #[test]
    fn random_in_unit_disk_stays_in_xy_unit_disk() {
        let mut rng = SampleRng::new(17);

        for _ in 0..20 {
            let point = rng.random_in_unit_disk();
            assert!(point.length_squared() < 1.0);
            assert_close(point.z(), 0.0);
        }
    }
}
