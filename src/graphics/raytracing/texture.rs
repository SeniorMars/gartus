//! Ray-tracing texture types built on the renderer-neutral texture trait.

use super::{LinearColor, rgb_to_linear_color};
use crate::{
    gmath::perlin::{Perlin, scale_point},
    graphics::{
        colors::Rgb,
        display::Canvas,
        texture::{SurfaceTexture, SurfaceTextureRef, Texture as BitmapTexture, TextureSample},
    },
};
use std::sync::Arc;

/// Shared renderer-neutral surface texture handle used by ray-tracing materials.
pub type TextureRef = SurfaceTextureRef;

/// A texture that always returns one linear color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidColor {
    /// Constant texture color.
    pub color: LinearColor,
}

impl SolidColor {
    /// Creates a constant linear-color texture.
    #[must_use]
    pub const fn new(color: LinearColor) -> Self {
        Self { color }
    }

    /// Creates a constant texture from display RGB bytes treated as linear unit values.
    #[must_use]
    pub fn from_rgb(color: Rgb) -> Self {
        Self::new(rgb_to_linear_color(color))
    }
}

impl From<LinearColor> for SolidColor {
    fn from(color: LinearColor) -> Self {
        Self::new(color)
    }
}

impl From<Rgb> for SolidColor {
    fn from(color: Rgb) -> Self {
        Self::from_rgb(color)
    }
}

impl SurfaceTexture for SolidColor {
    fn sample_linear(&self, _sample: TextureSample) -> LinearColor {
        self.color
    }
}

/// A 3D checker texture that alternates between two child textures in world space.
#[derive(Clone, Debug)]
pub struct CheckerTexture {
    inv_scale: f64,
    even: TextureRef,
    odd: TextureRef,
}

impl CheckerTexture {
    /// Creates a checker texture from two child textures.
    #[must_use]
    pub fn new(scale: f64, even: TextureRef, odd: TextureRef) -> Self {
        let inv_scale = if scale.is_finite() && scale.abs() > f64::EPSILON {
            1.0 / scale.abs()
        } else {
            1.0
        };
        Self {
            inv_scale,
            even,
            odd,
        }
    }

    /// Creates a checker texture from two constant colors.
    #[must_use]
    pub fn from_colors(scale: f64, even: LinearColor, odd: LinearColor) -> Self {
        Self::new(
            scale,
            Arc::new(SolidColor::new(even)),
            Arc::new(SolidColor::new(odd)),
        )
    }
}

impl SurfaceTexture for CheckerTexture {
    #[allow(clippy::cast_possible_truncation)]
    fn sample_linear(&self, sample: TextureSample) -> LinearColor {
        let point = sample.point;
        let x_integer = (self.inv_scale * point.x()).floor() as i64;
        let y_integer = (self.inv_scale * point.y()).floor() as i64;
        let z_integer = (self.inv_scale * point.z()).floor() as i64;
        if (x_integer + y_integer + z_integer) % 2 == 0 {
            self.even.sample_linear(sample)
        } else {
            self.odd.sample_linear(sample)
        }
    }
}

/// A ray-tracing texture backed by the library's 2D bitmap texture sampler.
#[derive(Clone, Debug)]
pub struct ImageTexture {
    texture: BitmapTexture,
}

impl ImageTexture {
    /// Creates an image texture from an existing bitmap texture.
    #[must_use]
    pub const fn new(texture: BitmapTexture) -> Self {
        Self { texture }
    }

    /// Creates an image texture from an existing canvas.
    #[must_use]
    pub const fn from_canvas(canvas: Canvas) -> Self {
        Self::new(BitmapTexture::from_canvas(canvas))
    }

    /// Returns the underlying bitmap texture.
    #[must_use]
    pub const fn texture(&self) -> &BitmapTexture {
        &self.texture
    }

    /// Loads an image file through the library's external image loader.
    ///
    /// # Errors
    ///
    /// Returns an error if the image cannot be loaded or converted into a canvas.
    #[cfg(feature = "external")]
    pub fn from_file(path: impl AsRef<str>) -> Result<Self, Box<dyn std::error::Error>> {
        let canvas = crate::external::ppmify(path.as_ref(), false)?;
        Ok(Self::from_canvas(canvas))
    }
}

impl SurfaceTexture for ImageTexture {
    fn sample_linear(&self, sample: TextureSample) -> LinearColor {
        if self.texture.image().is_empty() {
            return rgb_to_linear_color(Rgb::CYAN);
        }
        self.texture.sample_linear(sample)
    }
}

/// Procedural Perlin-noise texture.
#[derive(Clone, Debug)]
pub struct NoiseTexture {
    noise: Perlin,
    scale: f64,
    kind: NoiseTextureKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NoiseTextureKind {
    Noise,
    Turbulence { depth: usize },
    Marble { depth: usize },
}

impl NoiseTexture {
    /// Creates a shifted Perlin noise texture from `seed`.
    #[must_use]
    pub fn new(scale: f64, seed: u64) -> Self {
        Self {
            noise: Perlin::new(seed),
            scale,
            kind: NoiseTextureKind::Noise,
        }
    }

    /// Creates a turbulence texture from `seed`.
    #[must_use]
    pub fn turbulence(scale: f64, depth: usize, seed: u64) -> Self {
        Self {
            noise: Perlin::new(seed),
            scale,
            kind: NoiseTextureKind::Turbulence { depth },
        }
    }

    /// Creates a marble-like texture from `seed`.
    #[must_use]
    pub fn marble(scale: f64, depth: usize, seed: u64) -> Self {
        Self {
            noise: Perlin::new(seed),
            scale,
            kind: NoiseTextureKind::Marble { depth },
        }
    }

    /// Returns the coordinate scale applied by this texture.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }
}

impl Default for NoiseTexture {
    fn default() -> Self {
        Self::new(1.0, 1)
    }
}

impl SurfaceTexture for NoiseTexture {
    fn sample_linear(&self, sample: TextureSample) -> LinearColor {
        let point = sample.point;
        let intensity = match self.kind {
            NoiseTextureKind::Noise => {
                0.5 * (1.0 + self.noise.noise(scale_point(point, self.scale)))
            }
            NoiseTextureKind::Turbulence { depth } => {
                self.noise.turbulence(scale_point(point, self.scale), depth)
            }
            NoiseTextureKind::Marble { depth } => {
                let point = scale_point(point, self.scale);
                0.5 * (1.0 + (point.z() + 10.0 * self.noise.turbulence(point, depth)).sin())
            }
        };
        LinearColor::new(intensity, intensity, intensity)
    }
}
