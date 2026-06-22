use super::{
    field::DensityField,
    warp::{CurlNoiseField, DomainWarpedDensityField},
};
use crate::gmath::{
    perlin::{Perlin, scale_point},
    procedural::smoothstep,
    vector::{Point, Vector},
};

/// Built-in animated procedural density styles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProceduralDensityPreset {
    /// Turbulent, puffy density suitable for smoke and low clouds.
    Smoke,
    /// Soft low-frequency density suitable for fog banks.
    Mist,
    /// Animated wave-like density suitable for sci-fi or magical volumes.
    Plasma,
    /// Layered turbulence suitable for space clouds and colored fog shells.
    Nebula,
    /// Directional bands suitable for underwater currents or heat haze.
    Underwater,
}

/// Animated procedural density field for fog, smoke, plasma, and nebula-like volumes.
///
/// The generated density is deterministic for a given seed, bounded by `max_density`, and varies
/// over time. Use it with [`super::NonUniformMedium`] when a scene needs authored volumetric motion
/// without a fluid simulation:
///
/// ```
/// # use gartus::prelude::*;
/// let medium = NonUniformMedium::new(
///     Sphere::new(Point::new(0.0, 1.0, 0.0), 3.0),
///     ProceduralDensityField::smoke().with_seed(42),
///     LinearColor::new(0.8, 0.85, 0.9),
/// );
/// ```
#[derive(Clone, Debug)]
pub struct ProceduralDensityField {
    preset: ProceduralDensityPreset,
    max_density: f64,
    scale: f64,
    speed: f64,
    turbulence: f64,
    contrast: f64,
    seed: u64,
    noise: Perlin,
}

impl ProceduralDensityField {
    /// Creates a procedural field from one built-in preset.
    #[must_use]
    pub fn new(preset: ProceduralDensityPreset) -> Self {
        match preset {
            ProceduralDensityPreset::Smoke => {
                Self::from_parts(preset, 0.8, 1.35, 0.35, 0.9, 1.15, 1)
            }
            ProceduralDensityPreset::Mist => {
                Self::from_parts(preset, 0.35, 0.55, 0.12, 0.35, 0.75, 1)
            }
            ProceduralDensityPreset::Plasma => {
                Self::from_parts(preset, 1.0, 3.2, 1.25, 1.8, 1.2, 1)
            }
            ProceduralDensityPreset::Nebula => {
                Self::from_parts(preset, 0.65, 0.75, 0.18, 1.4, 1.45, 1)
            }
            ProceduralDensityPreset::Underwater => {
                Self::from_parts(preset, 0.55, 2.1, 0.85, 0.7, 0.9, 1)
            }
        }
    }

    fn from_parts(
        preset: ProceduralDensityPreset,
        max_density: f64,
        scale: f64,
        flow_speed: f64,
        turbulence: f64,
        contrast: f64,
        noise_seed: u64,
    ) -> Self {
        assert!(
            max_density.is_finite() && max_density > 0.0,
            "density field maximum must be positive and finite"
        );
        assert!(
            scale.is_finite() && scale > 0.0,
            "procedural density scale must be positive and finite"
        );
        assert!(
            flow_speed.is_finite(),
            "procedural density speed must be finite"
        );
        assert!(
            turbulence.is_finite(),
            "procedural density turbulence must be finite"
        );
        assert!(
            contrast.is_finite() && contrast >= 0.0,
            "procedural density contrast must be finite and non-negative"
        );

        Self {
            preset,
            max_density,
            scale,
            speed: flow_speed,
            turbulence: turbulence.max(0.0),
            contrast,
            seed: noise_seed,
            noise: Perlin::new(noise_seed),
        }
    }

    /// Wispy density suitable for smoke or low cloud volumes.
    #[must_use]
    pub fn smoke() -> Self {
        Self::new(ProceduralDensityPreset::Smoke)
    }

    /// Soft, low-frequency density suitable for mist and fog banks.
    #[must_use]
    pub fn mist() -> Self {
        Self::new(ProceduralDensityPreset::Mist)
    }

    /// High-frequency animated density suitable for plasma and magical-energy volumes.
    #[must_use]
    pub fn plasma() -> Self {
        Self::new(ProceduralDensityPreset::Plasma)
    }

    /// Soft, slow-moving density suitable for space nebulae or colored fog shells.
    #[must_use]
    pub fn nebula() -> Self {
        Self::new(ProceduralDensityPreset::Nebula)
    }

    /// Elongated animated density suitable for underwater-current or heat-haze volumes.
    #[must_use]
    pub fn underwater() -> Self {
        Self::new(ProceduralDensityPreset::Underwater)
    }

    /// Compatibility alias for [`Self::underwater`].
    #[must_use]
    pub fn underwater_current() -> Self {
        Self::underwater()
    }

    /// Returns the selected preset.
    #[must_use]
    pub const fn preset(&self) -> ProceduralDensityPreset {
        self.preset
    }

    /// Returns the configured density majorant.
    #[must_use]
    pub const fn maximum_density(&self) -> f64 {
        self.max_density
    }

    /// Returns the spatial frequency scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the time-advection speed.
    #[must_use]
    pub const fn speed(&self) -> f64 {
        self.speed
    }

    /// Returns the domain-warp strength.
    #[must_use]
    pub const fn turbulence(&self) -> f64 {
        self.turbulence
    }

    /// Returns the output contrast.
    #[must_use]
    pub const fn contrast(&self) -> f64 {
        self.contrast
    }

    /// Returns the deterministic noise seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns a copy with a different deterministic seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.noise = Perlin::new(seed);
        self
    }

    /// Returns a copy with a different density majorant.
    ///
    /// # Panics
    ///
    /// Panics if `max_density` is not positive and finite.
    #[must_use]
    pub fn with_max_density(self, max_density: f64) -> Self {
        Self::from_parts(
            self.preset,
            max_density,
            self.scale,
            self.speed,
            self.turbulence,
            self.contrast,
            self.seed,
        )
    }

    /// Returns a copy with a different spatial frequency scale.
    ///
    /// # Panics
    ///
    /// Panics if `scale` is not positive and finite.
    #[must_use]
    pub fn with_scale(self, scale: f64) -> Self {
        Self::from_parts(
            self.preset,
            self.max_density,
            scale,
            self.speed,
            self.turbulence,
            self.contrast,
            self.seed,
        )
    }

    /// Returns a copy with a different time-advection speed.
    ///
    /// # Panics
    ///
    /// Panics if `speed` is not finite.
    #[must_use]
    pub fn with_speed(self, speed: f64) -> Self {
        Self::from_parts(
            self.preset,
            self.max_density,
            self.scale,
            speed,
            self.turbulence,
            self.contrast,
            self.seed,
        )
    }

    /// Returns a copy with a different domain-warp strength.
    ///
    /// # Panics
    ///
    /// Panics if `turbulence` is not finite.
    #[must_use]
    pub fn with_turbulence(self, turbulence: f64) -> Self {
        Self::from_parts(
            self.preset,
            self.max_density,
            self.scale,
            self.speed,
            turbulence,
            self.contrast,
            self.seed,
        )
    }

    /// Returns a copy with different output contrast.
    ///
    /// # Panics
    ///
    /// Panics if `contrast` is not finite or is negative.
    #[must_use]
    pub fn with_contrast(self, contrast: f64) -> Self {
        Self::from_parts(
            self.preset,
            self.max_density,
            self.scale,
            self.speed,
            self.turbulence,
            contrast,
            self.seed,
        )
    }

    /// Wraps this field in a default curl-noise domain warp.
    #[must_use]
    pub fn domain_warped(self) -> DomainWarpedDensityField<Self> {
        DomainWarpedDensityField::new(self)
    }

    /// Wraps this field in an explicit curl-noise domain warp.
    #[must_use]
    pub fn with_domain_warp(self, warp: CurlNoiseField) -> DomainWarpedDensityField<Self> {
        DomainWarpedDensityField::with_warp(self, warp)
    }

    fn warped_point(&self, point: Point, time: f64) -> Point {
        let advect = time * self.speed;
        let flow = Vector::new(0.37 * advect, 0.19 * advect, -0.29 * advect);
        let point = scale_point(point, self.scale) + flow;
        let warp_strength = 0.35 * self.turbulence;
        let warp = warp_strength
            * Vector::new(
                self.noise.noise(point + Vector::new(17.0, 0.0, 0.0)),
                self.noise.noise(point + Vector::new(0.0, 31.0, 0.0)),
                self.noise.noise(point + Vector::new(0.0, 0.0, 47.0)),
            );
        point + warp
    }

    fn preset_density(&self, point: Point, time: f64) -> f64 {
        match self.preset {
            ProceduralDensityPreset::Smoke => {
                let base = self.noise.turbulence(point, 5).clamp(0.0, 1.0);
                let low = (0.5 + 0.5 * self.noise.noise(scale_point(point, 0.55))).clamp(0.0, 1.0);
                smoothstep(0.08, 1.0, 0.65 * base + 0.35 * low).powf(1.35)
            }
            ProceduralDensityPreset::Mist => {
                let low = (0.5 + 0.5 * self.noise.noise(scale_point(point, 0.35))).clamp(0.0, 1.0);
                let detail = self
                    .noise
                    .turbulence(scale_point(point, 0.7), 3)
                    .clamp(0.0, 1.0);
                0.25 + 0.75 * (0.8 * low + 0.2 * detail)
            }
            ProceduralDensityPreset::Plasma => {
                let wave_a = (point.x() * 2.0 + point.y() * 1.5 + time * self.speed).sin();
                let wave_b = (point.z() * 1.7 - point.x() * 0.7 - time * self.speed * 0.7).sin();
                let noise = self.noise.turbulence(point, 4).clamp(0.0, 1.0);
                (0.5 + 0.25 * wave_a + 0.20 * wave_b + 0.25 * noise).clamp(0.0, 1.0)
            }
            ProceduralDensityPreset::Nebula => {
                let large = self
                    .noise
                    .turbulence(scale_point(point, 0.45), 4)
                    .clamp(0.0, 1.0);
                let fine = self
                    .noise
                    .turbulence(scale_point(point, 1.5), 6)
                    .clamp(0.0, 1.0);
                smoothstep(0.15, 0.95, 0.7 * large + 0.3 * fine).powf(1.8)
            }
            ProceduralDensityPreset::Underwater => {
                let bands =
                    0.5 + 0.5 * (point.x() * 1.7 + point.z() * 0.4 + time * self.speed).sin();
                let noise = self
                    .noise
                    .turbulence(scale_point(point, 0.9), 3)
                    .clamp(0.0, 1.0);
                0.35 + 0.65 * (0.65 * bands + 0.35 * noise)
            }
        }
    }
}

impl DensityField for ProceduralDensityField {
    fn density(&self, point: Point, time: f64) -> f64 {
        if !point.is_finite() || !time.is_finite() {
            return 0.0;
        }

        let warped = self.warped_point(point, time);
        let density = self.preset_density(warped, time);
        let shaped = (density * self.contrast).clamp(0.0, 1.0);
        self.max_density * shaped
    }

    fn max_density(&self) -> f64 {
        self.max_density
    }
}
