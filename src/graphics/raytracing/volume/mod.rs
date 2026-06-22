//! Participating media and density fields for path tracing.

mod field;
mod grid;
mod marching_cubes;
mod medium;
mod particles;
mod procedural;
mod solver;
mod warp;

pub use field::{ConstantDensity, DensityField, DensityFieldRef, FnDensityField};
pub use grid::{GridBounds, GridDensityField, GridDensityMetadata, GridInterpolation};
pub use marching_cubes::{ExtractedSurface, LiquidSurface, MarchingCubes};
pub use medium::{ConstantMedium, NonUniformMedium};
pub use particles::{FluidParticle, ParticleSplatField, SplatKernel};
pub use procedural::{ProceduralDensityField, ProceduralDensityPreset};
pub use solver::{
    MacCellFlags, MacFluidEmitter, MacFluidGrid2, MacFluidGrid3, MacProjectionStats,
    MacScalarAdvection, MacScalarGrid3, MacStepStats, StableFluidEmitter, StableFluidGrid2,
};
pub use warp::{CurlNoiseField, DomainWarpedDensityField};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gmath::{
            random::SampleRng,
            ray::Ray,
            vector::{Point, Vector},
        },
        graphics::raytracing::{Hittable, INFINITY, Interval, LinearColor, Sphere},
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn function_density_field_reports_explicit_majorant() {
        let field = FnDensityField::new(3.5, |point: Point, time| point.x() + time);

        assert_close(field.maximum_density(), 3.5);
        assert_close(field.max_density(), 3.5);
        assert_close(field.density(Point::new(2.0, 0.0, 0.0), 0.25), 2.25);
    }

    #[test]
    fn procedural_density_field_is_bounded_and_repeatable() {
        let field = ProceduralDensityField::smoke().with_seed(42);
        let same = ProceduralDensityField::smoke().with_seed(42);
        let point = Point::new(1.25, -0.5, 3.75);
        let density = field.density(point, 0.5);

        assert!((0.0..=field.max_density()).contains(&density));
        assert_close(density, same.density(point, 0.5));
        assert_close(field.maximum_density(), 0.8);
        assert_eq!(field.preset(), ProceduralDensityPreset::Smoke);
        assert_eq!(field.seed(), 42);
    }

    #[test]
    fn procedural_density_field_animates_over_time() {
        let field = ProceduralDensityField::plasma().with_seed(17);
        let point = Point::new(0.35, 0.8, -1.1);
        let first = field.density(point, 0.0);
        let second = field.density(point, 1.0);

        assert!((0.0..=field.max_density()).contains(&first));
        assert!((0.0..=field.max_density()).contains(&second));
        assert!((first - second).abs() > f64::EPSILON);
    }

    #[test]
    fn procedural_density_field_tuners_preserve_seed() {
        let field = ProceduralDensityField::nebula()
            .with_seed(9)
            .with_max_density(0.4)
            .with_scale(1.25)
            .with_speed(0.5)
            .with_turbulence(0.25)
            .with_contrast(0.75);

        assert_close(field.max_density(), 0.4);
        assert_close(field.scale(), 1.25);
        assert_close(field.speed(), 0.5);
        assert_close(field.turbulence(), 0.25);
        assert_close(field.contrast(), 0.75);
        assert_eq!(field.seed(), 9);
    }

    #[test]
    fn procedural_density_presets_are_distinct_and_bounded() {
        let point = Point::new(0.35, -0.2, 1.1);
        for preset in [
            ProceduralDensityPreset::Smoke,
            ProceduralDensityPreset::Mist,
            ProceduralDensityPreset::Plasma,
            ProceduralDensityPreset::Nebula,
            ProceduralDensityPreset::Underwater,
        ] {
            let field = ProceduralDensityField::new(preset);
            let density = field.density(point, 0.25);
            assert!((0.0..=field.max_density()).contains(&density));
        }
    }

    #[test]
    fn curl_noise_returns_finite_repeatable_vectors() {
        let curl = CurlNoiseField::new(99)
            .with_scale(1.4)
            .with_speed(0.5)
            .with_epsilon(0.02);
        let point = Point::new(0.7, -0.25, 1.5);

        let first = curl.sample(point, 0.75);
        let second = curl.sample(point, 0.75);

        assert!(first.is_finite());
        assert_eq!(first, second);
        assert_eq!(curl.seed(), 99);
        assert_close(curl.scale(), 1.4);
        assert_close(curl.speed(), 0.5);
        assert_close(curl.epsilon(), 0.02);
    }

    #[test]
    fn domain_warp_preserves_base_majorant() {
        let base = ProceduralDensityField::smoke()
            .with_seed(10)
            .with_max_density(0.4);
        let density = DomainWarpedDensityField::new(base)
            .with_warp_seed(99)
            .with_warp_strength(0.8)
            .with_warp_scale(1.4)
            .with_warp_speed(0.5);
        let sample = density.density(Point::new(0.35, 0.8, -1.1), 0.75);

        assert_close(density.max_density(), 0.4);
        assert!((0.0..=density.max_density()).contains(&sample));
        assert_close(density.warp_strength(), 0.8);
        assert_eq!(density.warp().seed(), 99);
    }

    #[test]
    fn domain_warp_zero_strength_matches_base_density() {
        let base = ProceduralDensityField::nebula()
            .with_seed(12)
            .with_max_density(0.5);
        let point = Point::new(0.2, 0.4, -0.8);
        let time = 0.65;
        let expected = base.density(point, time);
        let warped = base.clone().domain_warped().with_warp_strength(0.0);

        assert_close(warped.density(point, time), expected);
    }

    #[test]
    fn warped_density_is_repeatable_for_seed() {
        let point = Point::new(1.2, 0.3, -0.7);
        let first = ProceduralDensityField::smoke()
            .with_seed(7)
            .with_domain_warp(CurlNoiseField::new(101).with_scale(1.2))
            .with_warp_strength(0.6)
            .density(point, 0.5);
        let second = ProceduralDensityField::smoke()
            .with_seed(7)
            .with_domain_warp(CurlNoiseField::new(101).with_scale(1.2))
            .with_warp_strength(0.6)
            .density(point, 0.5);

        assert_close(first, second);
    }

    #[test]
    fn non_uniform_medium_rejects_empty_density() {
        let boundary = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let field = FnDensityField::new(1.0, |_point: Point, _time| 0.0);
        let medium = NonUniformMedium::new(boundary, field, LinearColor::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let mut rng = SampleRng::new(47);

        assert!(
            medium
                .hit_with_rng(&ray, Interval::new(0.0, INFINITY), &mut rng)
                .is_none()
        );
        assert!(medium.bounding_box().is_some());
    }

    #[test]
    fn non_uniform_medium_samples_dense_region() {
        let boundary = Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5);
        let field = FnDensityField::new(16.0, |point: Point, time| {
            if point.z() < -0.75 && time > 0.5 {
                16.0
            } else {
                0.0
            }
        });
        let medium = NonUniformMedium::new(boundary, field, LinearColor::new(1.0, 1.0, 1.0));
        let ray = Ray::with_time(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0), 0.75);
        let mut rng = SampleRng::new(47);

        let record = medium
            .hit_with_rng(&ray, Interval::new(0.0, INFINITY), &mut rng)
            .expect("dense region should scatter");

        assert!(record.t > 0.75);
        assert!(record.t < 1.5);
        assert_eq!(record.normal, Vector::new(1.0, 0.0, 0.0));
        assert!(record.front_face);
    }
}
