use super::field::DensityField;
use crate::gmath::{
    perlin::{Perlin, scale_point},
    vector::{Point, Vector},
};

const DEFAULT_WARP_STRENGTH: f64 = 1.0;
const DEFAULT_WARP_SCALE: f64 = 1.0;
const DEFAULT_WARP_SPEED: f64 = 0.25;
const DEFAULT_CURL_EPSILON: f64 = 0.01;

/// Procedural curl-noise vector field for domain-warping density fields.
///
/// This is not a fluid solver. It builds a deterministic 3D vector field from three Perlin noise
/// fields, then approximates its curl with central differences. Sampling the curl gives a
/// divergence-reduced swirling offset suitable for smoke and current-like volumetric motion.
#[derive(Clone, Debug)]
pub struct CurlNoiseField {
    noise_x: Perlin,
    noise_y: Perlin,
    noise_z: Perlin,
    seed: u64,
    scale: f64,
    speed: f64,
    epsilon: f64,
}

impl CurlNoiseField {
    /// Creates a curl-noise field from a deterministic seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            noise_x: Perlin::new(seed),
            noise_y: Perlin::new(seed.wrapping_add(0x9E37_79B9_7F4A_7C15)),
            noise_z: Perlin::new(seed.wrapping_add(0xBF58_476D_1CE4_E5B9)),
            seed,
            scale: DEFAULT_WARP_SCALE,
            speed: DEFAULT_WARP_SPEED,
            epsilon: DEFAULT_CURL_EPSILON,
        }
    }

    /// Returns the deterministic seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the spatial scale used before noise sampling.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the time-advection speed.
    #[must_use]
    pub const fn speed(&self) -> f64 {
        self.speed
    }

    /// Returns the finite-difference step used to approximate curl.
    #[must_use]
    pub const fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns a copy with a different deterministic seed.
    #[must_use]
    pub fn with_seed(self, seed: u64) -> Self {
        Self {
            scale: self.scale,
            speed: self.speed,
            epsilon: self.epsilon,
            ..Self::new(seed)
        }
    }

    /// Returns a copy with a different spatial scale.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is not positive and finite.
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        assert!(
            scale.is_finite() && scale > 0.0,
            "curl noise scale must be positive and finite"
        );
        self.scale = scale;
        self
    }

    /// Returns a copy with a different time-advection speed.
    ///
    /// # Panics
    ///
    /// Panics if `speed` is not finite.
    #[must_use]
    pub fn with_speed(mut self, speed: f64) -> Self {
        assert!(speed.is_finite(), "curl noise speed must be finite");
        self.speed = speed;
        self
    }

    /// Returns a copy with a different finite-difference step.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` is not positive and finite.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon > 0.0,
            "curl noise epsilon must be positive and finite"
        );
        self.epsilon = epsilon;
        self
    }

    /// Samples the curl vector at `point` and `time`.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn sample(&self, point: Point, time: f64) -> Vector {
        if !point.is_finite() || !time.is_finite() {
            return Vector::default();
        }

        let step = self.epsilon;
        let inv_step = 0.5 / step;
        let dx = Vector::new(step, 0.0, 0.0);
        let dy = Vector::new(0.0, step, 0.0);
        let dz = Vector::new(0.0, 0.0, step);

        let fx_plus = self.vector_field(point + dx, time);
        let fx_minus = self.vector_field(point - dx, time);
        let fy_plus = self.vector_field(point + dy, time);
        let fy_minus = self.vector_field(point - dy, time);
        let fz_plus = self.vector_field(point + dz, time);
        let fz_minus = self.vector_field(point - dz, time);

        let d_fz_dy = (fy_plus.z() - fy_minus.z()) * inv_step;
        let d_fy_dz = (fz_plus.y() - fz_minus.y()) * inv_step;
        let d_fx_dz = (fz_plus.x() - fz_minus.x()) * inv_step;
        let d_fz_dx = (fx_plus.z() - fx_minus.z()) * inv_step;
        let d_fy_dx = (fx_plus.y() - fx_minus.y()) * inv_step;
        let d_fx_dy = (fy_plus.x() - fy_minus.x()) * inv_step;

        Vector::new(d_fz_dy - d_fy_dz, d_fx_dz - d_fz_dx, d_fy_dx - d_fx_dy)
    }

    fn vector_field(&self, point: Point, time: f64) -> Vector {
        let flow = time * self.speed;
        let sample_point =
            scale_point(point, self.scale) + Vector::new(0.31 * flow, -0.17 * flow, 0.23 * flow);
        Vector::new(
            self.noise_x.noise(sample_point),
            self.noise_y
                .noise(sample_point + Vector::new(19.1, 7.3, -11.4)),
            self.noise_z
                .noise(sample_point + Vector::new(-5.7, 13.8, 23.2)),
        )
    }
}

impl Default for CurlNoiseField {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Density field adapter that samples a base field at curl-noise-warped coordinates.
#[derive(Clone, Debug)]
pub struct DomainWarpedDensityField<D> {
    base: D,
    warp: CurlNoiseField,
    strength: f64,
}

impl<D> DomainWarpedDensityField<D> {
    /// Creates a domain-warped density field with default curl-noise settings.
    #[must_use]
    pub fn new(base: D) -> Self {
        Self {
            base,
            warp: CurlNoiseField::default(),
            strength: DEFAULT_WARP_STRENGTH,
        }
    }

    /// Creates a domain-warped density field from an explicit curl-noise field.
    #[must_use]
    pub fn with_warp(base: D, warp: CurlNoiseField) -> Self {
        Self {
            base,
            warp,
            strength: DEFAULT_WARP_STRENGTH,
        }
    }

    /// Returns the wrapped base density field.
    #[must_use]
    pub const fn base(&self) -> &D {
        &self.base
    }

    /// Returns the curl-noise field.
    #[must_use]
    pub const fn warp(&self) -> &CurlNoiseField {
        &self.warp
    }

    /// Returns the warp strength multiplier.
    #[must_use]
    pub const fn warp_strength(&self) -> f64 {
        self.strength
    }

    /// Consumes this adapter and returns the wrapped base density field.
    #[must_use]
    pub fn into_base(self) -> D {
        self.base
    }

    /// Returns a copy with a different curl-noise field.
    #[must_use]
    pub fn with_warp_field(mut self, warp: CurlNoiseField) -> Self {
        self.warp = warp;
        self
    }

    /// Returns a copy with a different curl-noise seed.
    #[must_use]
    pub fn with_warp_seed(mut self, seed: u64) -> Self {
        self.warp = self.warp.with_seed(seed);
        self
    }

    /// Returns a copy with a different warp strength.
    ///
    /// # Panics
    ///
    /// Panics if `strength` is not finite. Negative strength is clamped to zero.
    #[must_use]
    pub fn with_warp_strength(mut self, strength: f64) -> Self {
        assert!(strength.is_finite(), "domain warp strength must be finite");
        self.strength = strength.max(0.0);
        self
    }

    /// Returns a copy with a different curl-noise scale.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is not positive and finite.
    #[must_use]
    pub fn with_warp_scale(mut self, scale: f64) -> Self {
        self.warp = self.warp.with_scale(scale);
        self
    }

    /// Returns a copy with a different curl-noise time-advection speed.
    ///
    /// # Panics
    ///
    /// Panics if `speed` is not finite.
    #[must_use]
    pub fn with_warp_speed(mut self, speed: f64) -> Self {
        self.warp = self.warp.with_speed(speed);
        self
    }

    /// Returns a copy with a different curl finite-difference step.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` is not positive and finite.
    #[must_use]
    pub fn with_warp_epsilon(mut self, epsilon: f64) -> Self {
        self.warp = self.warp.with_epsilon(epsilon);
        self
    }
}

impl<D: DensityField> DensityField for DomainWarpedDensityField<D> {
    fn density(&self, point: Point, time: f64) -> f64 {
        if !point.is_finite() || !time.is_finite() {
            return 0.0;
        }

        let offset = self.warp.sample(point, time) * self.strength;
        if !offset.is_finite() {
            return 0.0;
        }
        self.base.density(point + offset, time)
    }

    fn max_density(&self) -> f64 {
        self.base.max_density()
    }
}
