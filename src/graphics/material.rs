//! Renderer-neutral material data.

use std::path::PathBuf;

use super::{
    colors::LinearRgb,
    lighting::{PhongMaterial, RefractiveIndex},
};

/// Renderer-neutral surface material data.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMaterial {
    /// Ambient/base occlusion color used by local lighting renderers.
    pub ambient_color: LinearRgb,
    /// Diffuse/base surface color.
    pub base_color: LinearRgb,
    /// Specular reflection color.
    pub specular_color: LinearRgb,
    /// Specular exponent or roughness hint, depending on renderer.
    pub shininess: f64,
    /// Optional refractive index for transparent materials.
    pub refractive_index: Option<RefractiveIndex>,
    /// Optional diffuse texture path or cache key.
    ///
    /// This is renderer-neutral metadata. High-level [`SurfaceScene`](crate::graphics::scene::SurfaceScene)
    /// rasterization and [`PathTracer::render_scene`](crate::graphics::raytracing::PathTracer::render_scene)
    /// do not resolve or sample this path; use the lower-level textured raster or ray APIs when a
    /// render needs texture sampling.
    pub diffuse_texture: Option<PathBuf>,
}

impl SurfaceMaterial {
    /// Creates an opaque surface material.
    ///
    /// # Panics
    ///
    /// Panics if any color channel or `shininess` is not finite.
    #[must_use]
    pub fn new(
        ambient_color: LinearRgb,
        base_color: LinearRgb,
        specular_color: LinearRgb,
        shininess: f64,
    ) -> Self {
        assert!(
            ambient_color.is_finite()
                && base_color.is_finite()
                && specular_color.is_finite()
                && shininess.is_finite(),
            "surface material values must be finite"
        );
        Self {
            ambient_color,
            base_color,
            specular_color,
            shininess,
            refractive_index: None,
            diffuse_texture: None,
        }
    }

    /// Creates an opaque surface material only when all color channels and `shininess` are finite.
    #[must_use]
    pub fn try_new(
        ambient_color: LinearRgb,
        base_color: LinearRgb,
        specular_color: LinearRgb,
        shininess: f64,
    ) -> Option<Self> {
        (ambient_color.is_finite()
            && base_color.is_finite()
            && specular_color.is_finite()
            && shininess.is_finite())
        .then_some(Self {
            ambient_color,
            base_color,
            specular_color,
            shininess,
            refractive_index: None,
            diffuse_texture: None,
        })
    }

    /// Adds a refractive index hint.
    #[must_use]
    pub fn with_refractive_index(mut self, refractive_index: RefractiveIndex) -> Self {
        self.refractive_index = Some(refractive_index);
        self
    }

    /// Adds diffuse texture metadata.
    ///
    /// High-level `SurfaceScene` render helpers keep this as metadata and shade with
    /// [`Self::base_color`]. Use lower-level textured raster or ray APIs to load and sample the
    /// texture.
    #[must_use]
    pub fn with_diffuse_texture(mut self, diffuse_texture: impl Into<PathBuf>) -> Self {
        self.diffuse_texture = Some(diffuse_texture.into());
        self
    }
}

impl Default for SurfaceMaterial {
    fn default() -> Self {
        Self::new(
            LinearRgb::new(0.1, 0.1, 0.1),
            LinearRgb::new(0.5, 0.5, 0.5),
            LinearRgb::new(0.5, 0.5, 0.5),
            f64::from(super::lighting::DEFAULT_SPECULAR_EXPONENT),
        )
    }
}

impl From<PhongMaterial> for SurfaceMaterial {
    fn from(material: PhongMaterial) -> Self {
        Self::new(
            material.ambient.into(),
            material.diffuse.into(),
            material.specular.into(),
            material.shininess,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_surface_material_constructor_rejects_non_finite_values() {
        let finite = LinearRgb::new(0.1, 0.2, 0.3);
        let invalid = LinearRgb::new(0.1, f64::NAN, 0.3);

        assert!(SurfaceMaterial::try_new(finite, finite, finite, 4.0).is_some());
        assert!(SurfaceMaterial::try_new(invalid, finite, finite, 4.0).is_none());
        assert!(SurfaceMaterial::try_new(finite, finite, finite, f64::INFINITY).is_none());
    }

    #[test]
    #[should_panic(expected = "surface material values must be finite")]
    fn surface_material_constructor_rejects_non_finite_values() {
        let finite = LinearRgb::new(0.1, 0.2, 0.3);
        let invalid = LinearRgb::new(0.1, f64::NAN, 0.3);

        let _ = SurfaceMaterial::new(finite, invalid, finite, 4.0);
    }
}
