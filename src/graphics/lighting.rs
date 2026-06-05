//! Phong reflection lighting for polygon fills.

use crate::{gmath::vector::Vector, graphics::colors::Rgb};

/// Default specular exponent from the course lighting source.
pub const DEFAULT_SPECULAR_EXPONENT: u32 = 4;

/// Per-channel reflection constants used by the Phong reflection model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectionConstants {
    /// Red channel reflection coefficient.
    pub red: f64,
    /// Green channel reflection coefficient.
    pub green: f64,
    /// Blue channel reflection coefficient.
    pub blue: f64,
}

impl ReflectionConstants {
    /// Creates new per-channel reflection constants.
    #[must_use]
    pub const fn new(red: f64, green: f64, blue: f64) -> Self {
        Self { red, green, blue }
    }

    fn values(self) -> [f64; 3] {
        [self.red, self.green, self.blue]
    }
}

/// A point light source with a direction/location vector and RGB color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// Vector from the surface to the light source.
    pub location: Vector,
    /// RGB color/intensity of the light source.
    pub color: Rgb,
}

impl PointLight {
    /// Creates a point light.
    #[must_use]
    pub const fn new(location: Vector, color: Rgb) -> Self {
        Self { location, color }
    }
}

/// Lighting inputs for Phong reflection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// View vector from the surface to the viewer.
    pub view: Vector,
    /// Ambient light color.
    pub ambient: Rgb,
    /// Point light source.
    pub point_light: PointLight,
    /// Ambient reflection constants.
    pub ambient_reflection: ReflectionConstants,
    /// Diffuse reflection constants.
    pub diffuse_reflection: ReflectionConstants,
    /// Specular reflection constants.
    pub specular_reflection: ReflectionConstants,
    /// Specular exponent controlling highlight falloff.
    pub specular_exponent: u32,
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            view: Vector::new(0.0, 0.0, 1.0),
            ambient: Rgb::new(50, 50, 50),
            point_light: PointLight::new(Vector::new(0.75, 0.75, 1.0), Rgb::WHITE),
            ambient_reflection: ReflectionConstants::new(0.1, 0.1, 0.1),
            diffuse_reflection: ReflectionConstants::new(0.75, 0.25, 0.25),
            specular_reflection: ReflectionConstants::new(0.25, 0.25, 0.75),
            specular_exponent: DEFAULT_SPECULAR_EXPONENT,
        }
    }
}

impl Lighting {
    /// Calculates one flat-shaded color for a polygon surface normal.
    #[must_use]
    pub fn illuminate(self, normal: Vector) -> Rgb {
        self.prepare().illuminate(normal)
    }

    pub(crate) fn prepare(self) -> PreparedLighting {
        let ambient = rgb_values(self.ambient);
        let point = rgb_values(self.point_light.color);
        let ambient_reflection = self.ambient_reflection.values();
        let diffuse_reflection = self.diffuse_reflection.values();
        let specular_reflection = self.specular_reflection.values();

        PreparedLighting {
            view: self.view.normalized(),
            light: self.point_light.location.normalized(),
            ambient: [
                ambient[0] * ambient_reflection[0],
                ambient[1] * ambient_reflection[1],
                ambient[2] * ambient_reflection[2],
            ],
            diffuse: [
                point[0] * diffuse_reflection[0],
                point[1] * diffuse_reflection[1],
                point[2] * diffuse_reflection[2],
            ],
            specular: [
                point[0] * specular_reflection[0],
                point[1] * specular_reflection[1],
                point[2] * specular_reflection[2],
            ],
            specular_exponent: i32::try_from(self.specular_exponent).unwrap_or(i32::MAX),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedLighting {
    view: Vector,
    light: Vector,
    ambient: [f64; 3],
    diffuse: [f64; 3],
    specular: [f64; 3],
    specular_exponent: i32,
}

impl PreparedLighting {
    pub(crate) fn illuminate(self, normal: Vector) -> Rgb {
        self.illuminate_unit(normal.normalized())
    }

    pub(crate) fn illuminate_unit(self, normal: Vector) -> Rgb {
        let (diffuse_factor, specular_factor) = self.reflection_factors_unit(normal);

        Rgb::new(
            self.channel_intensity(0, diffuse_factor, specular_factor),
            self.channel_intensity(1, diffuse_factor, specular_factor),
            self.channel_intensity(2, diffuse_factor, specular_factor),
        )
    }

    pub(crate) fn illuminate_toon(self, normal: Vector) -> Rgb {
        let (diffuse_factor, specular_factor) = self.reflection_factors_unit(normal.normalized());
        let diffuse_factor = quantize_diffuse(diffuse_factor);
        let specular_factor = if specular_factor >= 0.45 { 1.0 } else { 0.0 };

        Rgb::new(
            self.channel_intensity(0, diffuse_factor, specular_factor),
            self.channel_intensity(1, diffuse_factor, specular_factor),
            self.channel_intensity(2, diffuse_factor, specular_factor),
        )
    }

    fn reflection_factors_unit(self, normal: Vector) -> (f64, f64) {
        let normal_dot_light = normal.dot(self.light).max(0.0);

        let reflection = normal * (2.0 * normal_dot_light) - self.light;
        let specular_factor = if normal_dot_light > 0.0 {
            reflection
                .dot(self.view)
                .max(0.0)
                .powi(self.specular_exponent)
        } else {
            0.0
        };

        (normal_dot_light, specular_factor)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn channel_intensity(self, channel: usize, diffuse_factor: f64, specular_factor: f64) -> u8 {
        let ambient = self.ambient[channel];
        let diffuse = self.diffuse[channel] * diffuse_factor;
        let specular = self.specular[channel] * specular_factor;

        (ambient + diffuse + specular).round().clamp(0.0, 255.0) as u8
    }
}

fn quantize_diffuse(value: f64) -> f64 {
    if value >= 0.9 {
        1.0
    } else if value >= 0.55 {
        0.72
    } else if value >= 0.25 {
        0.38
    } else {
        0.12
    }
}

fn rgb_values(color: Rgb) -> [f64; 3] {
    [
        f64::from(color.red),
        f64::from(color.green),
        f64::from(color.blue),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lighting_matches_assignment_constants() {
        let lighting = Lighting::default();

        assert_eq!(lighting.view, Vector::new(0.0, 0.0, 1.0));
        assert_eq!(lighting.ambient, Rgb::new(50, 50, 50));
        assert_eq!(
            lighting.point_light,
            PointLight::new(Vector::new(0.75, 0.75, 1.0), Rgb::WHITE)
        );
        assert_eq!(
            lighting.ambient_reflection,
            ReflectionConstants::new(0.1, 0.1, 0.1)
        );
        assert_eq!(
            lighting.diffuse_reflection,
            ReflectionConstants::new(0.75, 0.25, 0.25)
        );
        assert_eq!(
            lighting.specular_reflection,
            ReflectionConstants::new(0.25, 0.25, 0.75)
        );
        assert_eq!(lighting.specular_exponent, DEFAULT_SPECULAR_EXPONENT);
    }

    #[test]
    fn illuminate_adds_ambient_diffuse_and_specular() {
        let color = Lighting::default().illuminate(Vector::new(0.0, 0.0, 1.0));

        assert_eq!(color, Rgb::new(150, 63, 91));
    }

    #[test]
    fn illuminate_limits_channels_to_rgb_range() {
        let lighting = Lighting {
            ambient: Rgb::WHITE,
            ambient_reflection: ReflectionConstants::new(10.0, 10.0, 10.0),
            diffuse_reflection: ReflectionConstants::new(10.0, 10.0, 10.0),
            specular_reflection: ReflectionConstants::new(10.0, 10.0, 10.0),
            ..Lighting::default()
        };

        assert_eq!(lighting.illuminate(Vector::new(0.0, 0.0, 1.0)), Rgb::WHITE);
    }

    #[test]
    fn illuminate_toon_quantizes_smooth_lighting() {
        let lighting = Lighting::default().prepare();
        let normal = Vector::new(0.0, 0.0, 1.0);

        assert_eq!(lighting.illuminate(normal), Rgb::new(150, 63, 91));
        assert_eq!(lighting.illuminate_toon(normal), Rgb::new(143, 51, 51));
    }
}
