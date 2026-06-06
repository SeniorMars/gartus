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
    /// Optional diffuse texture path or key.
    pub diffuse_texture: Option<PathBuf>,
}

impl SurfaceMaterial {
    /// Creates an opaque surface material.
    #[must_use]
    pub fn new(
        ambient_color: LinearRgb,
        base_color: LinearRgb,
        specular_color: LinearRgb,
        shininess: f64,
    ) -> Self {
        Self {
            ambient_color,
            base_color,
            specular_color,
            shininess,
            refractive_index: None,
            diffuse_texture: None,
        }
    }

    /// Adds a refractive index hint.
    #[must_use]
    pub fn with_refractive_index(mut self, refractive_index: RefractiveIndex) -> Self {
        self.refractive_index = Some(refractive_index);
        self
    }

    /// Adds a diffuse texture path or cache key.
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
