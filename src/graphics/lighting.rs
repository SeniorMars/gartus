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

/// Whether a light is positional or a legacy direction vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightKind {
    /// `location` is a rendered-space point and light direction varies per surface point.
    Positional,
    /// `location` is a direction vector, matching the original course lighting model.
    Directional,
}

/// Distance falloff model for positional lights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightAttenuation {
    /// No distance falloff.
    None,
    /// Fall off as `radius / (radius + distance)`.
    InverseLinear {
        /// Distance scale where the light reaches half intensity.
        radius: f64,
    },
    /// Fall off as `radius^2 / (radius^2 + distance^2)`.
    InverseSquare {
        /// Distance scale where the light reaches half intensity.
        radius: f64,
    },
}

impl LightAttenuation {
    fn factor(self, distance: f64) -> f64 {
        match self {
            Self::None => 1.0,
            Self::InverseLinear { radius } => {
                let radius = radius.max(f64::EPSILON);
                radius / (radius + distance.max(0.0))
            }
            Self::InverseSquare { radius } => {
                let radius = radius.max(f64::EPSILON);
                let radius_squared = radius * radius;
                radius_squared / (radius_squared + distance.max(0.0).powi(2))
            }
        }
    }
}

/// A light source with a position/direction and RGB color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// Light position or direction.
    pub location: Vector,
    /// RGB color/intensity of the light source.
    pub color: Rgb,
    /// Light interpretation.
    pub kind: LightKind,
    /// Distance falloff for positional lights.
    pub attenuation: LightAttenuation,
}

impl PointLight {
    /// Creates a legacy directional light.
    #[must_use]
    pub const fn new(location: Vector, color: Rgb) -> Self {
        Self::directional(location, color)
    }

    /// Creates a positional light.
    #[must_use]
    pub const fn positional(location: Vector, color: Rgb) -> Self {
        Self {
            location,
            color,
            kind: LightKind::Positional,
            attenuation: LightAttenuation::None,
        }
    }

    /// Creates a directional light.
    #[must_use]
    pub const fn directional(location: Vector, color: Rgb) -> Self {
        Self {
            location,
            color,
            kind: LightKind::Directional,
            attenuation: LightAttenuation::None,
        }
    }

    /// Returns this light with inverse-linear distance falloff.
    #[must_use]
    pub const fn with_inverse_linear_attenuation(mut self, radius: f64) -> Self {
        self.attenuation = LightAttenuation::InverseLinear { radius };
        self
    }

    /// Returns this light with inverse-square distance falloff.
    #[must_use]
    pub const fn with_inverse_square_attenuation(mut self, radius: f64) -> Self {
        self.attenuation = LightAttenuation::InverseSquare { radius };
        self
    }

    /// Returns this light with explicit attenuation.
    #[must_use]
    pub const fn with_attenuation(mut self, attenuation: LightAttenuation) -> Self {
        self.attenuation = attenuation;
        self
    }
}

/// Lighting inputs for Phong reflection.
#[derive(Clone, Debug, PartialEq)]
pub struct Lighting {
    /// View vector from the surface to the viewer.
    pub view: Vector,
    /// Ambient light color.
    pub ambient: Rgb,
    /// Point light source used when `point_lights` is empty.
    pub point_light: PointLight,
    /// Explicit point light sources. When empty, `point_light` is used for compatibility.
    pub point_lights: Vec<PointLight>,
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
            point_light: PointLight::new(Vector::new(0.5, 0.75, 1.0), Rgb::WHITE),
            point_lights: Vec::new(),
            ambient_reflection: ReflectionConstants::new(0.1, 0.1, 0.1),
            diffuse_reflection: ReflectionConstants::new(0.5, 0.5, 0.5),
            specular_reflection: ReflectionConstants::new(0.5, 0.5, 0.5),
            specular_exponent: DEFAULT_SPECULAR_EXPONENT,
        }
    }
}

impl Lighting {
    /// Calculates one flat-shaded color for a polygon surface normal.
    #[must_use]
    pub fn illuminate(&self, normal: Vector) -> Rgb {
        self.prepare().illuminate(normal)
    }

    /// Calculates one color for a surface normal at a 3D point.
    #[must_use]
    pub fn illuminate_at(&self, normal: Vector, point: Vector) -> Rgb {
        self.prepare().illuminate_at(normal, point)
    }

    pub(crate) fn prepare(&self) -> PreparedLighting {
        let ambient = rgb_values(self.ambient);
        let ambient_reflection = self.ambient_reflection.values();
        let diffuse_reflection = self.diffuse_reflection.values();
        let specular_reflection = self.specular_reflection.values();
        let source_lights = if self.point_lights.is_empty() {
            std::slice::from_ref(&self.point_light)
        } else {
            &self.point_lights
        };
        let point_lights = source_lights
            .iter()
            .copied()
            .map(|point_light| {
                let point = rgb_values(point_light.color);
                PreparedPointLight {
                    position: point_light.location,
                    kind: point_light.kind,
                    attenuation: point_light.attenuation,
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
                }
            })
            .collect();

        PreparedLighting {
            view: self.view.normalized(),
            ambient: [
                ambient[0] * ambient_reflection[0],
                ambient[1] * ambient_reflection[1],
                ambient[2] * ambient_reflection[2],
            ],
            point_lights,
            specular_exponent: i32::try_from(self.specular_exponent).unwrap_or(i32::MAX),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedPointLight {
    position: Vector,
    kind: LightKind,
    attenuation: LightAttenuation,
    diffuse: [f64; 3],
    specular: [f64; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedLighting {
    view: Vector,
    ambient: [f64; 3],
    point_lights: Vec<PreparedPointLight>,
    specular_exponent: i32,
}

impl PreparedLighting {
    pub(crate) fn illuminate(&self, normal: Vector) -> Rgb {
        self.illuminate_at(normal, Vector::default())
    }

    pub(crate) fn illuminate_at(&self, normal: Vector, point: Vector) -> Rgb {
        self.illuminate_unit_at(normal.normalized(), point)
    }

    pub(crate) fn illuminate_unit_at(&self, normal: Vector, point: Vector) -> Rgb {
        self.illuminate_unit_with(normal, point, false)
    }

    pub(crate) fn illuminate_toon_at(&self, normal: Vector, point: Vector) -> Rgb {
        self.illuminate_unit_with(normal.normalized(), point, true)
    }

    fn illuminate_unit_with(&self, normal: Vector, point: Vector, toon: bool) -> Rgb {
        let mut channels = self.ambient;

        for point_light in &self.point_lights {
            let (light, attenuation) = match point_light.kind {
                LightKind::Positional => {
                    let light_vector = point_light.position - point;
                    (
                        light_vector.normalized(),
                        point_light.attenuation.factor(light_vector.length()),
                    )
                }
                LightKind::Directional => (point_light.position.normalized(), 1.0),
            };
            let (diffuse_factor, specular_factor) = self.reflection_factors_unit(normal, light);
            let diffuse_factor = if toon {
                quantize_diffuse(diffuse_factor)
            } else {
                diffuse_factor
            };
            let specular_factor = if toon && specular_factor < 0.45 {
                0.0
            } else if toon {
                1.0
            } else {
                specular_factor
            };
            for (channel, value) in channels.iter_mut().enumerate() {
                *value += attenuation
                    * (point_light.diffuse[channel] * diffuse_factor
                        + point_light.specular[channel] * specular_factor);
            }
        }

        Rgb::new(
            channel_intensity(channels[0]),
            channel_intensity(channels[1]),
            channel_intensity(channels[2]),
        )
    }

    fn reflection_factors_unit(&self, normal: Vector, light: Vector) -> (f64, f64) {
        let normal_dot_light = normal.dot(light).max(0.0);

        let reflection = normal * (2.0 * normal_dot_light) - light;
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

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn channel_intensity(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
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
            PointLight::new(Vector::new(0.5, 0.75, 1.0), Rgb::WHITE)
        );
        assert!(lighting.point_lights.is_empty());
        assert_eq!(
            lighting.ambient_reflection,
            ReflectionConstants::new(0.1, 0.1, 0.1)
        );
        assert_eq!(
            lighting.diffuse_reflection,
            ReflectionConstants::new(0.5, 0.5, 0.5)
        );
        assert_eq!(
            lighting.specular_reflection,
            ReflectionConstants::new(0.5, 0.5, 0.5)
        );
        assert_eq!(lighting.specular_exponent, DEFAULT_SPECULAR_EXPONENT);
    }

    #[test]
    fn illuminate_adds_ambient_diffuse_and_specular() {
        let color = Lighting::default().illuminate(Vector::new(0.0, 0.0, 1.0));

        assert_eq!(color, Rgb::new(139, 139, 139));
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
    fn zero_vector_light_normalizes_safely() {
        let lighting = Lighting {
            point_lights: vec![PointLight::new(Vector::default(), Rgb::WHITE)],
            ..Lighting::default()
        };

        assert_eq!(
            lighting.illuminate(Vector::new(0.0, 0.0, 1.0)),
            Rgb::new(5, 5, 5)
        );
    }

    #[test]
    fn positional_light_direction_depends_on_surface_point() {
        let lighting = Lighting {
            ambient: Rgb::BLACK,
            point_lights: vec![PointLight::positional(
                Vector::new(0.0, 0.0, 10.0),
                Rgb::WHITE,
            )],
            ambient_reflection: ReflectionConstants::new(0.0, 0.0, 0.0),
            diffuse_reflection: ReflectionConstants::new(1.0, 1.0, 1.0),
            specular_reflection: ReflectionConstants::new(0.0, 0.0, 0.0),
            ..Lighting::default()
        };

        assert_eq!(
            lighting.illuminate_at(Vector::new(0.0, 0.0, 1.0), Vector::default()),
            Rgb::WHITE
        );
        assert_eq!(
            lighting.illuminate_at(Vector::new(0.0, 0.0, 1.0), Vector::new(0.0, 0.0, 20.0)),
            Rgb::BLACK
        );
    }

    #[test]
    fn positional_light_inverse_linear_attenuation_reduces_distance_intensity() {
        let lighting = Lighting {
            ambient: Rgb::BLACK,
            point_lights: vec![
                PointLight::positional(Vector::new(0.0, 0.0, 10.0), Rgb::WHITE)
                    .with_inverse_linear_attenuation(10.0),
            ],
            ambient_reflection: ReflectionConstants::new(0.0, 0.0, 0.0),
            diffuse_reflection: ReflectionConstants::new(1.0, 1.0, 1.0),
            specular_reflection: ReflectionConstants::new(0.0, 0.0, 0.0),
            ..Lighting::default()
        };

        assert_eq!(
            lighting.illuminate_at(Vector::new(0.0, 0.0, 1.0), Vector::default()),
            Rgb::new(128, 128, 128)
        );
    }

    #[test]
    fn positional_light_inverse_square_attenuation_reduces_distance_intensity() {
        let lighting = Lighting {
            ambient: Rgb::BLACK,
            point_lights: vec![
                PointLight::positional(Vector::new(0.0, 0.0, 10.0), Rgb::WHITE)
                    .with_inverse_square_attenuation(10.0),
            ],
            ambient_reflection: ReflectionConstants::new(0.0, 0.0, 0.0),
            diffuse_reflection: ReflectionConstants::new(1.0, 1.0, 1.0),
            specular_reflection: ReflectionConstants::new(0.0, 0.0, 0.0),
            ..Lighting::default()
        };

        assert_eq!(
            lighting.illuminate_at(Vector::new(0.0, 0.0, 1.0), Vector::default()),
            Rgb::new(128, 128, 128)
        );
    }

    #[test]
    fn illuminate_toon_quantizes_smooth_lighting() {
        let lighting = Lighting::default().prepare();
        let normal = Vector::new(0.0, 0.0, 1.0);

        assert_eq!(lighting.illuminate(normal), Rgb::new(139, 139, 139));
        assert_eq!(
            lighting.illuminate_toon_at(normal, Vector::default()),
            Rgb::new(97, 97, 97)
        );
    }

    #[test]
    fn illuminate_accumulates_multiple_point_lights() {
        let lighting = Lighting {
            ambient: Rgb::BLACK,
            point_lights: vec![
                PointLight::new(Vector::new(0.0, 0.0, 1.0), Rgb::new(80, 0, 0)),
                PointLight::new(Vector::new(0.0, 0.0, 1.0), Rgb::new(0, 60, 0)),
            ],
            ambient_reflection: ReflectionConstants::new(0.0, 0.0, 0.0),
            diffuse_reflection: ReflectionConstants::new(1.0, 1.0, 1.0),
            specular_reflection: ReflectionConstants::new(0.0, 0.0, 0.0),
            ..Lighting::default()
        };

        assert_eq!(
            lighting.illuminate(Vector::new(0.0, 0.0, 1.0)),
            Rgb::new(80, 60, 0)
        );
    }
}
