//! Ray-tracing texture types and adapters.

use super::{LinearColor, rgb_to_linear_color};
use crate::{
    gmath::{
        perlin::{Perlin, scale_point},
        vector::Point,
    },
    graphics::{
        colors::Rgb,
        display::Canvas,
        texture::{SurfaceTexture, Texture as BitmapTexture, TextureSample},
    },
};
use std::{fmt, sync::Arc};

/// Procedural or image-backed texture sampled by ray-tracing materials.
pub trait RayTexture: fmt::Debug + Send + Sync {
    /// Returns the linear color at texture coordinate `(u, v)` and surface point `point`.
    fn value(&self, u: f64, v: f64, point: Point) -> LinearColor;
}

/// Shared ray-texture handle.
pub type TextureRef = Arc<dyn RayTexture>;

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

impl RayTexture for SolidColor {
    fn value(&self, _u: f64, _v: f64, _point: Point) -> LinearColor {
        self.color
    }
}

impl<T: RayTexture + ?Sized> SurfaceTexture for T {
    fn sample_linear(&self, sample: TextureSample) -> LinearColor {
        self.value(sample.u, sample.v, sample.point)
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

impl RayTexture for CheckerTexture {
    #[allow(clippy::cast_possible_truncation)]
    fn value(&self, texture_u: f64, texture_v: f64, point: Point) -> LinearColor {
        let x_integer = (self.inv_scale * point.x()).floor() as i64;
        let y_integer = (self.inv_scale * point.y()).floor() as i64;
        let z_integer = (self.inv_scale * point.z()).floor() as i64;
        if (x_integer + y_integer + z_integer) % 2 == 0 {
            self.even.value(texture_u, texture_v, point)
        } else {
            self.odd.value(texture_u, texture_v, point)
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

impl RayTexture for ImageTexture {
    fn value(&self, u: f64, v: f64, _point: Point) -> LinearColor {
        if self.texture.image().is_empty() {
            return rgb_to_linear_color(Rgb::CYAN);
        }
        rgb_to_linear_color(self.texture.sample(u, v))
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

impl RayTexture for NoiseTexture {
    fn value(&self, _u: f64, _v: f64, point: Point) -> LinearColor {
        let intensity = match self.kind {
            NoiseTextureKind::Noise => {
                0.5 * (1.0 + self.noise.noise(scale_point(point, self.scale)))
            }
            NoiseTextureKind::Turbulence { depth } => {
                self.noise.turbulence(scale_point(point, self.scale), depth)
            }
            NoiseTextureKind::Marble { depth } => {
                0.5 * (1.0
                    + (self.scale * point.z() + 10.0 * self.noise.turbulence(point, depth)).sin())
            }
        };
        LinearColor::new(intensity, intensity, intensity)
    }
}
