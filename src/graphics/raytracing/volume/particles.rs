use super::field::DensityField;
use crate::gmath::vector::{Point, Vector};
use std::collections::HashMap;

/// One density-carrying particle for [`ParticleSplatField`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidParticle {
    /// Particle center in world space.
    pub position: Point,
    /// Radius of influence in world-space units.
    pub radius: f64,
    /// Density contribution at the particle center before kernel falloff.
    pub density: f64,
}

impl FluidParticle {
    /// Creates a finite particle with positive radius and non-negative density.
    ///
    /// # Panics
    ///
    /// Panics if `position` is not finite, `radius` is not positive and finite, or `density` is
    /// not finite and non-negative.
    #[must_use]
    pub fn new(position: Point, radius: f64, density: f64) -> Self {
        assert!(
            position.is_finite(),
            "fluid particle position must be finite"
        );
        assert!(
            radius.is_finite() && radius > 0.0,
            "fluid particle radius must be positive and finite"
        );
        assert!(
            density.is_finite() && density >= 0.0,
            "fluid particle density must be finite and non-negative"
        );
        Self {
            position,
            radius,
            density,
        }
    }

    fn contribution(self, point: Point, kernel: SplatKernel) -> f64 {
        if !point.is_finite() || self.density <= 0.0 {
            return 0.0;
        }

        let distance = (point - self.position).length();
        if distance >= self.radius {
            return 0.0;
        }

        self.density * kernel.evaluate(distance / self.radius)
    }
}

/// Radial kernel used to turn particles into a smooth density field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplatKernel {
    /// Truncated Gaussian falloff normalized to be zero at the particle radius.
    Gaussian,
    /// Smoothstep falloff from center to radius.
    Smooth,
    /// Compact SPH-style poly6 falloff.
    Poly6,
}

impl SplatKernel {
    /// Evaluates the kernel at normalized radius `r / particle_radius`.
    ///
    /// Returns one at the center and zero at or beyond the particle radius. Non-finite inputs
    /// return zero.
    #[must_use]
    pub fn evaluate(self, normalized_radius: f64) -> f64 {
        if !normalized_radius.is_finite() {
            return 0.0;
        }

        let radius = normalized_radius.abs();
        if radius >= 1.0 {
            return 0.0;
        }

        match self {
            Self::Gaussian => gaussian_kernel(radius),
            Self::Smooth => smooth_kernel(radius),
            Self::Poly6 => poly6_kernel(radius),
        }
    }
}

/// Particle-backed density field for blobs, spray, foam, and puff-like volumes.
///
/// Particles are inserted into a spatial hash grid over their radius of influence. Density queries
/// only inspect particles in the sampled cell, then clamp the accumulated density to
/// `max_density`. The default maximum density is conservative: the sum of all particle center
/// densities.
#[derive(Clone, Debug)]
pub struct ParticleSplatField {
    particles: Vec<FluidParticle>,
    acceleration: ParticleGrid,
    kernel: SplatKernel,
    max_density: f64,
}

impl ParticleSplatField {
    /// Creates a particle density field using [`SplatKernel::Poly6`].
    #[must_use]
    pub fn new(particles: Vec<FluidParticle>) -> Self {
        let cell_size = default_cell_size(&particles);
        Self::from_parts(
            particles,
            SplatKernel::Poly6,
            None,
            cell_size,
            MajorantMode::Conservative,
        )
    }

    fn from_parts(
        particles: Vec<FluidParticle>,
        kernel: SplatKernel,
        max_density: Option<f64>,
        cell_size: f64,
        majorant_mode: MajorantMode,
    ) -> Self {
        validate_cell_size(cell_size);
        let conservative = conservative_majorant(&particles);
        let max_density = match (max_density, majorant_mode) {
            (Some(max_density), MajorantMode::Explicit) => validate_explicit_majorant(max_density),
            (Some(max_density), MajorantMode::Conservative) => {
                validate_explicit_majorant(max_density).max(conservative)
            }
            (None, _) => conservative,
        };

        let acceleration = ParticleGrid::new(&particles, cell_size);
        Self {
            particles,
            acceleration,
            kernel,
            max_density,
        }
    }

    /// Returns the particles backing the density field.
    #[must_use]
    pub fn particles(&self) -> &[FluidParticle] {
        &self.particles
    }

    /// Returns the active kernel.
    #[must_use]
    pub const fn kernel(&self) -> SplatKernel {
        self.kernel
    }

    /// Returns the configured density majorant.
    #[must_use]
    pub const fn maximum_density(&self) -> f64 {
        self.max_density
    }

    /// Returns the spatial hash cell size.
    #[must_use]
    pub const fn cell_size(&self) -> f64 {
        self.acceleration.cell_size()
    }

    /// Returns the number of occupied spatial hash buckets.
    #[must_use]
    pub fn bucket_count(&self) -> usize {
        self.acceleration.bucket_count()
    }

    /// Returns a copy with a different radial kernel.
    #[must_use]
    pub fn with_kernel(mut self, kernel: SplatKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Returns a copy with an explicit density majorant.
    ///
    /// This method trusts the caller's bound. Use the default conservative majorant when particles
    /// may overlap heavily and no tighter scene-specific bound is known.
    ///
    /// # Panics
    ///
    /// Panics if `max_density` is not positive and finite.
    #[must_use]
    pub fn with_max_density(mut self, max_density: f64) -> Self {
        self.max_density = validate_explicit_majorant(max_density);
        self
    }

    /// Rebuilds the spatial hash with a custom cell size.
    ///
    /// Smaller cells reduce candidate counts for small particles but insert large particles into
    /// more buckets. Larger cells build faster but may check more candidates per density query.
    ///
    /// # Panics
    ///
    /// Panics if `cell_size` is not positive and finite.
    #[must_use]
    pub fn with_cell_size(self, cell_size: f64) -> Self {
        Self::from_parts(
            self.particles,
            self.kernel,
            Some(self.max_density),
            cell_size,
            MajorantMode::Explicit,
        )
    }

    fn density_unclamped(&self, point: Point) -> f64 {
        if !point.is_finite() {
            return 0.0;
        }

        self.acceleration
            .candidate_indices(point)
            .iter()
            .map(|index| self.particles[*index].contribution(point, self.kernel))
            .sum()
    }

    #[cfg(test)]
    fn density_naive_unclamped(&self, point: Point) -> f64 {
        self.particles
            .iter()
            .map(|particle| particle.contribution(point, self.kernel))
            .sum()
    }
}

impl DensityField for ParticleSplatField {
    fn density(&self, point: Point, _time: f64) -> f64 {
        self.density_unclamped(point).clamp(0.0, self.max_density)
    }

    fn max_density(&self) -> f64 {
        self.max_density
    }
}

#[derive(Clone, Copy)]
enum MajorantMode {
    Conservative,
    Explicit,
}

#[derive(Clone, Debug)]
struct ParticleGrid {
    cell_size: f64,
    buckets: HashMap<[i32; 3], Vec<usize>>,
}

impl ParticleGrid {
    fn new(particles: &[FluidParticle], cell_size: f64) -> Self {
        validate_cell_size(cell_size);
        let mut buckets: HashMap<[i32; 3], Vec<usize>> = HashMap::with_capacity(particles.len());

        for (index, particle) in particles.iter().enumerate() {
            if particle.density <= 0.0 {
                continue;
            }

            let radius = Vector::new(particle.radius, particle.radius, particle.radius);
            let min_key = cell_key(particle.position - radius, cell_size)
                .expect("particle grid minimum key should be representable");
            let max_key = cell_key(particle.position + radius, cell_size)
                .expect("particle grid maximum key should be representable");

            for z in min_key[2]..=max_key[2] {
                for y in min_key[1]..=max_key[1] {
                    for x in min_key[0]..=max_key[0] {
                        buckets.entry([x, y, z]).or_default().push(index);
                    }
                }
            }
        }

        Self { cell_size, buckets }
    }

    const fn cell_size(&self) -> f64 {
        self.cell_size
    }

    fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    fn candidate_indices(&self, point: Point) -> &[usize] {
        let Some(key) = cell_key(point, self.cell_size) else {
            return &[];
        };
        self.buckets.get(&key).map_or(&[], Vec::as_slice)
    }
}

fn gaussian_kernel(radius: f64) -> f64 {
    let edge = (-4.0_f64).exp();
    (((-4.0 * radius * radius).exp() - edge) / (1.0 - edge)).clamp(0.0, 1.0)
}

fn smooth_kernel(radius: f64) -> f64 {
    let smooth = radius * radius * (3.0 - 2.0 * radius);
    (1.0 - smooth).clamp(0.0, 1.0)
}

fn poly6_kernel(radius: f64) -> f64 {
    let falloff = 1.0 - radius * radius;
    (falloff * falloff * falloff).clamp(0.0, 1.0)
}

fn default_cell_size(particles: &[FluidParticle]) -> f64 {
    let (radius_sum, count) = particles
        .iter()
        .filter(|particle| particle.density > 0.0)
        .fold((0.0, 0_usize), |(radius_sum, count), particle| {
            (radius_sum + particle.radius, count + 1)
        });

    if count == 0 {
        1.0
    } else {
        let count = f64::from(u32::try_from(count).expect("particle count should fit in u32"));
        (radius_sum / count).max(f64::MIN_POSITIVE)
    }
}

fn conservative_majorant(particles: &[FluidParticle]) -> f64 {
    let max_density = particles
        .iter()
        .map(|particle| particle.density)
        .sum::<f64>();
    assert!(
        max_density.is_finite(),
        "particle density conservative majorant must be finite"
    );
    positive_majorant(max_density)
}

fn validate_explicit_majorant(max_density: f64) -> f64 {
    assert!(
        max_density.is_finite() && max_density > 0.0,
        "particle splat maximum density must be positive and finite"
    );
    max_density
}

fn validate_cell_size(cell_size: f64) {
    assert!(
        cell_size.is_finite() && cell_size > 0.0,
        "particle splat cell size must be positive and finite"
    );
}

fn positive_majorant(max_density: f64) -> f64 {
    if max_density.is_finite() && max_density > 0.0 {
        max_density
    } else {
        f64::MIN_POSITIVE
    }
}

fn cell_key(point: Point, cell_size: f64) -> Option<[i32; 3]> {
    if !point.is_finite() {
        return None;
    }
    Some([
        floor_to_i32(point.x() / cell_size)?,
        floor_to_i32(point.y() / cell_size)?,
        floor_to_i32(point.z() / cell_size)?,
    ])
}

#[allow(clippy::cast_possible_truncation)]
fn floor_to_i32(value: f64) -> Option<i32> {
    let value = value.floor();
    (value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX)).then_some(value as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }

    #[test]
    fn splat_kernels_are_bounded() {
        for kernel in [
            SplatKernel::Gaussian,
            SplatKernel::Smooth,
            SplatKernel::Poly6,
        ] {
            assert_close(kernel.evaluate(0.0), 1.0);
            assert_close(kernel.evaluate(1.0), 0.0);
            assert_close(kernel.evaluate(f64::NAN), 0.0);
            assert!((0.0..=1.0).contains(&kernel.evaluate(0.5)));
        }
    }

    #[test]
    fn particle_splat_density_is_highest_at_particle_center() {
        let field = ParticleSplatField::new(vec![FluidParticle::new(
            Point::new(0.0, 0.0, 0.0),
            1.0,
            2.0,
        )])
        .with_kernel(SplatKernel::Poly6);

        let center = field.density(Point::new(0.0, 0.0, 0.0), 0.0);
        let shoulder = field.density(Point::new(0.5, 0.0, 0.0), 0.0);

        assert_close(center, 2.0);
        assert!(center > shoulder);
    }

    #[test]
    fn particle_splat_density_zero_outside_radius() {
        let field = ParticleSplatField::new(vec![FluidParticle::new(
            Point::new(0.0, 0.0, 0.0),
            1.0,
            2.0,
        )]);

        assert_close(field.density(Point::new(1.01, 0.0, 0.0), 0.0), 0.0);
        assert_close(field.density(Point::new(0.0, -1.01, 0.0), 0.0), 0.0);
    }

    #[test]
    fn particle_grid_matches_naive_density() {
        let field = ParticleSplatField::new(vec![
            FluidParticle::new(Point::new(0.0, 0.0, 0.0), 0.8, 1.0),
            FluidParticle::new(Point::new(0.35, 0.0, 0.0), 0.6, 0.7),
            FluidParticle::new(Point::new(2.0, 0.5, -0.25), 0.4, 1.2),
        ])
        .with_kernel(SplatKernel::Smooth)
        .with_cell_size(0.25);

        for point in [
            Point::new(0.0, 0.0, 0.0),
            Point::new(0.25, 0.0, 0.0),
            Point::new(2.0, 0.5, -0.25),
            Point::new(3.0, 0.0, 0.0),
        ] {
            assert_close(
                field.density_unclamped(point),
                field.density_naive_unclamped(point),
            );
        }
    }

    #[test]
    fn particle_grid_avoids_full_scan_for_distributed_particles() {
        let particles = (0..32)
            .map(|index| FluidParticle::new(Point::new(f64::from(index), 0.0, 0.0), 0.1, 1.0))
            .collect();
        let field = ParticleSplatField::new(particles).with_cell_size(0.2);
        let candidates = field
            .acceleration
            .candidate_indices(Point::new(0.0, 0.0, 0.0));

        assert!(!candidates.is_empty());
        assert!(candidates.len() < field.particles().len());
    }

    #[test]
    fn particle_splat_majorant_can_be_overridden_and_clamps() {
        let field = ParticleSplatField::new(vec![
            FluidParticle::new(Point::new(0.0, 0.0, 0.0), 1.0, 2.0),
            FluidParticle::new(Point::new(0.0, 0.0, 0.0), 1.0, 2.0),
        ]);

        assert_close(field.max_density(), 4.0);
        let clamped = field.with_max_density(1.5);
        assert_close(clamped.maximum_density(), 1.5);
        assert_close(clamped.density(Point::new(0.0, 0.0, 0.0), 0.0), 1.5);
    }

    #[test]
    fn empty_particle_splat_field_is_empty_with_positive_majorant() {
        let field = ParticleSplatField::new(Vec::new());

        assert_close(field.density(Point::new(0.0, 0.0, 0.0), 0.0), 0.0);
        assert!(field.max_density() > 0.0);
        assert_eq!(field.bucket_count(), 0);
        assert_close(field.cell_size(), 1.0);
    }
}
