//! Phong reflection lighting for polygon fills.

pub use crate::graphics::material::SurfaceMaterial;
use crate::{
    gmath::vector::Vector,
    graphics::colors::{LinearRgb, Rgb},
};

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

/// Classic Phong material coefficients.
///
/// These presets use the common OpenGL material table values collected at
/// <http://www.barradeau.com/nicoptere/dump/materials.html>. Alpha is preserved for callers that
/// want material metadata, but the current polygon renderer uses only the ambient, diffuse,
/// specular, and shininess fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhongMaterial {
    /// Ambient reflection constants.
    pub ambient: ReflectionConstants,
    /// Diffuse reflection constants.
    pub diffuse: ReflectionConstants,
    /// Specular reflection constants.
    pub specular: ReflectionConstants,
    /// Phong specular exponent.
    pub shininess: f64,
    /// Source alpha value from RGBA material tables.
    pub alpha: f64,
}

impl PhongMaterial {
    /// Creates a material with opaque alpha.
    #[must_use]
    pub const fn new(
        ambient: ReflectionConstants,
        diffuse: ReflectionConstants,
        specular: ReflectionConstants,
        shininess: f64,
    ) -> Self {
        Self::new_with_alpha(ambient, diffuse, specular, shininess, 1.0)
    }

    /// Creates a material with explicit alpha.
    #[must_use]
    pub const fn new_with_alpha(
        ambient: ReflectionConstants,
        diffuse: ReflectionConstants,
        specular: ReflectionConstants,
        shininess: f64,
        alpha: f64,
    ) -> Self {
        Self {
            ambient,
            diffuse,
            specular,
            shininess,
            alpha,
        }
    }

    /// Returns `shininess` rounded for [`Lighting::specular_exponent`].
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn specular_exponent(self) -> u32 {
        self.shininess.round().clamp(0.0, f64::from(u32::MAX)) as u32
    }

    /// Brass material preset.
    pub const BRASS: Self = Self::new(
        ReflectionConstants::new(0.329_412, 0.223_529, 0.027_451),
        ReflectionConstants::new(0.780_392, 0.568_627, 0.113_725),
        ReflectionConstants::new(0.992_157, 0.941_176, 0.807_843),
        27.8974,
    );

    /// Bronze material preset.
    pub const BRONZE: Self = Self::new(
        ReflectionConstants::new(0.2125, 0.1275, 0.054),
        ReflectionConstants::new(0.714, 0.4284, 0.181_44),
        ReflectionConstants::new(0.393_548, 0.271_906, 0.166_721),
        25.6,
    );

    /// Polished bronze material preset.
    pub const POLISHED_BRONZE: Self = Self::new(
        ReflectionConstants::new(0.25, 0.148, 0.064_75),
        ReflectionConstants::new(0.4, 0.2368, 0.1036),
        ReflectionConstants::new(0.774_597, 0.458_561, 0.200_621),
        76.8,
    );

    /// Chrome material preset.
    pub const CHROME: Self = Self::new(
        ReflectionConstants::new(0.25, 0.25, 0.25),
        ReflectionConstants::new(0.4, 0.4, 0.4),
        ReflectionConstants::new(0.774_597, 0.774_597, 0.774_597),
        76.8,
    );

    /// Copper material preset.
    pub const COPPER: Self = Self::new(
        ReflectionConstants::new(0.191_25, 0.0735, 0.0225),
        ReflectionConstants::new(0.7038, 0.270_48, 0.0828),
        ReflectionConstants::new(0.256_777, 0.137_622, 0.086_014),
        12.8,
    );

    /// Polished copper material preset.
    pub const POLISHED_COPPER: Self = Self::new(
        ReflectionConstants::new(0.2295, 0.088_25, 0.0275),
        ReflectionConstants::new(0.5508, 0.2118, 0.066),
        ReflectionConstants::new(0.580_594, 0.223_257, 0.069_570_1),
        51.2,
    );

    /// Gold material preset.
    pub const GOLD: Self = Self::new(
        ReflectionConstants::new(0.247_25, 0.1995, 0.0745),
        ReflectionConstants::new(0.751_64, 0.606_48, 0.226_48),
        ReflectionConstants::new(0.628_281, 0.555_802, 0.366_065),
        51.2,
    );

    /// Polished gold material preset.
    pub const POLISHED_GOLD: Self = Self::new(
        ReflectionConstants::new(0.247_25, 0.2245, 0.0645),
        ReflectionConstants::new(0.346_15, 0.3143, 0.0903),
        ReflectionConstants::new(0.797_357, 0.723_991, 0.208_006),
        83.2,
    );

    /// Pewter material preset.
    pub const PEWTER: Self = Self::new(
        ReflectionConstants::new(0.105_882, 0.058_824, 0.113_725),
        ReflectionConstants::new(0.427_451, 0.470_588, 0.541_176),
        ReflectionConstants::new(0.333_333, 0.333_333, 0.521_569),
        9.846_15,
    );

    /// Silver material preset.
    pub const SILVER: Self = Self::new(
        ReflectionConstants::new(0.192_25, 0.192_25, 0.192_25),
        ReflectionConstants::new(0.507_54, 0.507_54, 0.507_54),
        ReflectionConstants::new(0.508_273, 0.508_273, 0.508_273),
        51.2,
    );

    /// Polished silver material preset.
    pub const POLISHED_SILVER: Self = Self::new(
        ReflectionConstants::new(0.231_25, 0.231_25, 0.231_25),
        ReflectionConstants::new(0.2775, 0.2775, 0.2775),
        ReflectionConstants::new(0.773_911, 0.773_911, 0.773_911),
        89.6,
    );

    /// Emerald material preset.
    pub const EMERALD: Self = Self::new_with_alpha(
        ReflectionConstants::new(0.0215, 0.1745, 0.0215),
        ReflectionConstants::new(0.075_68, 0.614_24, 0.075_68),
        ReflectionConstants::new(0.633, 0.727_811, 0.633),
        76.8,
        0.55,
    );

    /// Jade material preset.
    pub const JADE: Self = Self::new_with_alpha(
        ReflectionConstants::new(0.135, 0.2225, 0.1575),
        ReflectionConstants::new(0.54, 0.89, 0.63),
        ReflectionConstants::new(0.316_228, 0.316_228, 0.316_228),
        12.8,
        0.95,
    );

    /// Obsidian material preset.
    pub const OBSIDIAN: Self = Self::new_with_alpha(
        ReflectionConstants::new(0.053_75, 0.05, 0.066_25),
        ReflectionConstants::new(0.182_75, 0.17, 0.225_25),
        ReflectionConstants::new(0.332_741, 0.328_634, 0.346_435),
        38.4,
        0.82,
    );

    /// Pearl material preset.
    pub const PEARL: Self = Self::new_with_alpha(
        ReflectionConstants::new(0.25, 0.207_25, 0.207_25),
        ReflectionConstants::new(1.0, 0.829, 0.829),
        ReflectionConstants::new(0.296_648, 0.296_648, 0.296_648),
        11.264,
        0.922,
    );

    /// Ruby material preset.
    pub const RUBY: Self = Self::new_with_alpha(
        ReflectionConstants::new(0.1745, 0.011_75, 0.011_75),
        ReflectionConstants::new(0.614_24, 0.041_36, 0.041_36),
        ReflectionConstants::new(0.727_811, 0.626_959, 0.626_959),
        76.8,
        0.55,
    );

    /// Turquoise material preset.
    pub const TURQUOISE: Self = Self::new_with_alpha(
        ReflectionConstants::new(0.1, 0.187_25, 0.1745),
        ReflectionConstants::new(0.396, 0.741_51, 0.691_02),
        ReflectionConstants::new(0.297_254, 0.308_29, 0.306_678),
        12.8,
        0.8,
    );

    /// Black plastic material preset.
    pub const BLACK_PLASTIC: Self = Self::new(
        ReflectionConstants::new(0.0, 0.0, 0.0),
        ReflectionConstants::new(0.01, 0.01, 0.01),
        ReflectionConstants::new(0.5, 0.5, 0.5),
        32.0,
    );

    /// Black rubber material preset.
    pub const BLACK_RUBBER: Self = Self::new(
        ReflectionConstants::new(0.02, 0.02, 0.02),
        ReflectionConstants::new(0.01, 0.01, 0.01),
        ReflectionConstants::new(0.4, 0.4, 0.4),
        10.0,
    );
}

impl From<ReflectionConstants> for LinearRgb {
    fn from(reflectance: ReflectionConstants) -> Self {
        Self::new(reflectance.red, reflectance.green, reflectance.blue)
    }
}

/// Index of refraction for a transparent medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefractiveIndex(pub f64);

impl RefractiveIndex {
    /// Creates a refractive index value.
    #[must_use]
    ///
    /// # Panics
    ///
    /// Panics if `index` is not positive and finite.
    pub fn new(index: f64) -> Self {
        assert!(
            index.is_finite() && index > 0.0,
            "refractive index must be positive and finite"
        );
        Self(index)
    }

    /// Creates a refractive index value, returning `None` for invalid indices.
    #[must_use]
    pub fn try_new(index: f64) -> Option<Self> {
        (index.is_finite() && index > 0.0).then_some(Self(index))
    }

    /// Vacuum index of refraction.
    pub const VACUUM: Self = Self(1.0);
    /// Approximate air index of refraction.
    pub const AIR: Self = Self(1.000_29);
    /// Common glass index of refraction.
    pub const GLASS: Self = Self(1.5);
    /// Ice index of refraction.
    pub const ICE: Self = Self(1.3);
    /// Diamond index of refraction.
    pub const DIAMOND: Self = Self(2.42);
    /// Water index of refraction.
    pub const WATER: Self = Self(1.33);
    /// Ruby index of refraction.
    pub const RUBY: Self = Self(1.77);
    /// Emerald index of refraction.
    pub const EMERALD: Self = Self(1.57);

    /// Returns this index relative to an enclosing medium.
    #[must_use]
    pub fn relative_to(self, enclosing: Self) -> Self {
        Self::new(self.0 / enclosing.0)
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

impl From<&SurfaceMaterial> for PhongMaterial {
    fn from(material: &SurfaceMaterial) -> Self {
        Self::new(
            reflection_from_linear_rgb(material.ambient_color),
            reflection_from_linear_rgb(material.base_color),
            reflection_from_linear_rgb(material.specular_color),
            material.shininess,
        )
    }
}

impl From<SurfaceMaterial> for PhongMaterial {
    fn from(material: SurfaceMaterial) -> Self {
        Self::from(&material)
    }
}

fn reflection_from_linear_rgb(color: LinearRgb) -> ReflectionConstants {
    ReflectionConstants::new(color.red, color.green, color.blue)
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
    /// Returns this lighting configuration with Phong reflection constants from `material`.
    #[must_use]
    pub fn with_material(mut self, material: PhongMaterial) -> Self {
        self.set_material(material);
        self
    }

    /// Applies Phong reflection constants from `material`.
    pub fn set_material(&mut self, material: PhongMaterial) {
        self.ambient_reflection = material.ambient;
        self.diffuse_reflection = material.diffuse;
        self.specular_reflection = material.specular;
        self.specular_exponent = material.specular_exponent();
    }

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
                let position = match point_light.kind {
                    LightKind::Directional => point_light.location.normalized(),
                    LightKind::Positional => point_light.location,
                };
                PreparedPointLight {
                    position,
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
                    let light_length = light_vector.length();
                    let light = if light_length < f64::EPSILON {
                        Vector::default()
                    } else {
                        light_vector / light_length
                    };
                    (light, point_light.attenuation.factor(light_length))
                }
                LightKind::Directional => (point_light.position, 1.0),
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
    fn material_presets_keep_source_values() {
        assert_eq!(
            PhongMaterial::GOLD.ambient,
            ReflectionConstants::new(0.247_25, 0.1995, 0.0745)
        );
        assert!((PhongMaterial::RUBY.alpha - 0.55).abs() < f64::EPSILON);
        assert_eq!(RefractiveIndex::DIAMOND, RefractiveIndex(2.42));
        assert_eq!(
            RefractiveIndex::AIR.relative_to(RefractiveIndex::GLASS),
            RefractiveIndex::new(RefractiveIndex::AIR.0 / RefractiveIndex::GLASS.0)
        );
    }

    #[test]
    fn refractive_index_rejects_invalid_values() {
        assert_eq!(RefractiveIndex::try_new(1.5), Some(RefractiveIndex(1.5)));
        assert_eq!(RefractiveIndex::try_new(0.0), None);
        assert_eq!(RefractiveIndex::try_new(f64::NAN), None);
    }

    #[test]
    #[should_panic(expected = "refractive index must be positive and finite")]
    fn refractive_index_new_panics_on_zero() {
        let _ = RefractiveIndex::new(0.0);
    }

    #[test]
    fn lighting_with_material_applies_phong_coefficients() {
        let lighting = Lighting::default().with_material(PhongMaterial::POLISHED_GOLD);

        assert_eq!(
            lighting.ambient_reflection,
            PhongMaterial::POLISHED_GOLD.ambient
        );
        assert_eq!(
            lighting.diffuse_reflection,
            PhongMaterial::POLISHED_GOLD.diffuse
        );
        assert_eq!(
            lighting.specular_reflection,
            PhongMaterial::POLISHED_GOLD.specular
        );
        assert_eq!(lighting.specular_exponent, 83);
    }

    #[test]
    fn surface_material_converts_to_phong_material() {
        let surface = SurfaceMaterial::new(
            LinearRgb::new(0.1, 0.2, 0.3),
            LinearRgb::new(0.4, 0.5, 0.6),
            LinearRgb::new(0.7, 0.8, 0.9),
            32.0,
        );

        let phong = PhongMaterial::from(&surface);

        assert_eq!(phong.ambient, ReflectionConstants::new(0.1, 0.2, 0.3));
        assert_eq!(phong.diffuse, ReflectionConstants::new(0.4, 0.5, 0.6));
        assert_eq!(phong.specular, ReflectionConstants::new(0.7, 0.8, 0.9));
        assert!((phong.shininess - 32.0).abs() < f64::EPSILON);
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
