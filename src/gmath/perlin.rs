//! Deterministic 3D Perlin noise for procedural textures and terrain.

use super::{random::SampleRng, vector::Point, vector::Vector};

const POINT_COUNT: usize = 256;

/// Repeatable gradient Perlin noise with 256-entry permutation tables.
#[derive(Clone, Debug)]
pub struct Perlin {
    gradients: [Vector; POINT_COUNT],
    perm_x: [usize; POINT_COUNT],
    perm_y: [usize; POINT_COUNT],
    perm_z: [usize; POINT_COUNT],
}

impl Perlin {
    /// Creates deterministic Perlin noise from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = SampleRng::new(seed);
        let mut gradients = [Vector::default(); POINT_COUNT];
        for gradient in &mut gradients {
            *gradient = rng.random_unit_vector();
        }

        Self {
            gradients,
            perm_x: generate_permutation(&mut rng),
            perm_y: generate_permutation(&mut rng),
            perm_z: generate_permutation(&mut rng),
        }
    }

    /// Returns smooth Perlin noise at `point`, usually in roughly `-1.0..=1.0`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn noise(&self, point: Point) -> f64 {
        let frac_x = point.x() - point.x().floor();
        let frac_y = point.y() - point.y().floor();
        let frac_z = point.z() - point.z().floor();

        let base_x = point.x().floor() as i32;
        let base_y = point.y().floor() as i32;
        let base_z = point.z().floor() as i32;
        let mut corners = [[[Vector::default(); 2]; 2]; 2];

        for (offset_x, plane) in [0_i32, 1].into_iter().zip(corners.iter_mut()) {
            for (offset_y, row) in [0_i32, 1].into_iter().zip(plane.iter_mut()) {
                for (offset_z, corner) in [0_i32, 1].into_iter().zip(row.iter_mut()) {
                    let gradient_index = self.perm_x[((base_x + offset_x) & 255) as usize]
                        ^ self.perm_y[((base_y + offset_y) & 255) as usize]
                        ^ self.perm_z[((base_z + offset_z) & 255) as usize];
                    *corner = self.gradients[gradient_index];
                }
            }
        }

        perlin_interp(corners, frac_x, frac_y, frac_z)
    }

    /// Returns absolute summed octave noise.
    #[must_use]
    pub fn turbulence(&self, point: Point, depth: usize) -> f64 {
        let mut accum = 0.0;
        let mut temp_point = point;
        let mut weight = 1.0;

        for _ in 0..depth {
            accum += weight * self.noise(temp_point);
            weight *= 0.5;
            temp_point = scale_point(temp_point, 2.0);
        }

        accum.abs()
    }
}

impl Default for Perlin {
    fn default() -> Self {
        Self::new(1)
    }
}

fn generate_permutation(rng: &mut SampleRng) -> [usize; POINT_COUNT] {
    let mut permutation = [0; POINT_COUNT];
    for (index, value) in permutation.iter_mut().enumerate() {
        *value = index;
    }
    permute(&mut permutation, rng);
    permutation
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn permute(permutation: &mut [usize; POINT_COUNT], rng: &mut SampleRng) {
    for index in (1..POINT_COUNT).rev() {
        let target = (rng.random_double() * (index as f64 + 1.0)) as usize;
        permutation.swap(index, target);
    }
}

fn perlin_interp(corners: [[[Vector; 2]; 2]; 2], u: f64, v: f64, w: f64) -> f64 {
    let uu = smoothstep(u);
    let vv = smoothstep(v);
    let ww = smoothstep(w);
    let mut accum = 0.0;

    for (offset_x, plane) in [0.0, 1.0].into_iter().zip(corners.iter()) {
        for (offset_y, row) in [0.0, 1.0].into_iter().zip(plane.iter()) {
            for (offset_z, gradient) in [0.0, 1.0].into_iter().zip(row.iter()) {
                let weight = Vector::new(u - offset_x, v - offset_y, w - offset_z);
                accum += blend(offset_x, uu)
                    * blend(offset_y, vv)
                    * blend(offset_z, ww)
                    * gradient.dot(weight);
            }
        }
    }

    accum
}

fn smoothstep(value: f64) -> f64 {
    value * value * (3.0 - 2.0 * value)
}

fn blend(offset: f64, smoothed: f64) -> f64 {
    if offset > 0.5 {
        smoothed
    } else {
        1.0 - smoothed
    }
}

/// Scales a point's coordinates from the origin.
#[must_use]
pub fn scale_point(point: Point, scale: f64) -> Point {
    Point::new(point.x() * scale, point.y() * scale, point.z() * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn perlin_noise_is_repeatable_for_same_seed() {
        let first = Perlin::new(7);
        let second = Perlin::new(7);
        let point = Point::new(1.25, -0.5, 3.75);

        assert_close(first.noise(point), second.noise(point));
        assert_close(first.turbulence(point, 7), second.turbulence(point, 7));
    }

    #[test]
    fn perlin_noise_varies_with_seed_and_position() {
        let first = Perlin::new(7);
        let second = Perlin::new(8);
        let point = Point::new(1.25, -0.5, 3.75);

        assert!((first.noise(point) - second.noise(point)).abs() > f64::EPSILON);
        assert!(
            (first.noise(point) - first.noise(Point::new(1.5, -0.5, 3.75))).abs() > f64::EPSILON
        );
    }

    #[test]
    fn turbulence_is_non_negative() {
        let noise = Perlin::new(11);

        assert!(noise.turbulence(Point::new(0.2, 0.4, 0.6), 7) >= 0.0);
    }
}
