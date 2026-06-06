use std::{fmt, sync::Arc};

use super::{
    colors::{LinearRgb, Rgb},
    display::Canvas,
};
use crate::gmath::vector::Point;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Controls how texture coordinates outside `0..=1` are handled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TextureWrap {
    /// Clamp out-of-range coordinates to the nearest texture edge.
    #[default]
    Clamp,
    /// Repeat the texture by wrapping coordinates with modulo arithmetic.
    Repeat,
}

/// Controls how texels are sampled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TextureFilter {
    /// Pick the nearest texel.
    #[default]
    Nearest,
    /// Bilinearly interpolate neighboring texels; mipmapped textures also blend adjacent levels.
    Linear,
}

/// Inputs for renderer-neutral surface texture sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureSample {
    /// Horizontal surface texture coordinate.
    pub u: f64,
    /// Vertical surface texture coordinate.
    pub v: f64,
    /// Surface point in world or object space, depending on the texture.
    pub point: Point,
}

impl TextureSample {
    /// Creates a surface texture sample.
    #[must_use]
    pub const fn new(u: f64, v: f64, point: Point) -> Self {
        Self { u, v, point }
    }
}

/// Texture data that can be sampled as linear RGB by any renderer.
pub trait SurfaceTexture: fmt::Debug + Send + Sync {
    /// Returns the linear color for a surface sample.
    fn sample_linear(&self, sample: TextureSample) -> LinearRgb;
}

/// Shared renderer-neutral surface texture handle.
pub type SurfaceTextureRef = Arc<dyn SurfaceTexture>;

/// A 2D RGB texture sampled with normalized `(s, t)` coordinates.
#[derive(Clone, Debug)]
pub struct Texture {
    image: Canvas,
    mipmaps: Vec<Canvas>,
    wrap_s: TextureWrap,
    wrap_t: TextureWrap,
    filter: TextureFilter,
}

/// A texture sampler selected once for a draw call.
pub(crate) enum ActiveTextureSampler<'a> {
    NearestBase {
        image: &'a Canvas,
        wrap_s: TextureWrap,
        wrap_t: TextureWrap,
    },
    LinearBase {
        image: &'a Canvas,
        wrap_s: TextureWrap,
        wrap_t: TextureWrap,
    },
    NearestMip {
        texture: &'a Texture,
        wrap_s: TextureWrap,
        wrap_t: TextureWrap,
        max_level: usize,
    },
    LinearMip {
        texture: &'a Texture,
        wrap_s: TextureWrap,
        wrap_t: TextureWrap,
        max_level: usize,
    },
}

impl Texture {
    /// Creates a texture from an existing canvas.
    #[must_use]
    pub const fn from_canvas(image: Canvas) -> Self {
        Self {
            image,
            mipmaps: Vec::new(),
            wrap_s: TextureWrap::Clamp,
            wrap_t: TextureWrap::Clamp,
            filter: TextureFilter::Nearest,
        }
    }

    /// Sets both texture-coordinate wrap modes.
    #[must_use]
    pub const fn wrap(mut self, wrap_s: TextureWrap, wrap_t: TextureWrap) -> Self {
        self.wrap_s = wrap_s;
        self.wrap_t = wrap_t;
        self
    }

    /// Sets the texture filter.
    #[must_use]
    pub const fn filter(mut self, filter: TextureFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Generates and stores downsampled mipmap levels.
    #[must_use]
    pub fn mipmapped(mut self) -> Self {
        self.mipmaps = build_mipmaps(&self.image);
        self
    }

    /// Returns the underlying image.
    pub const fn image(&self) -> &Canvas {
        &self.image
    }

    /// Returns the number of available texture levels, including the base image.
    #[must_use]
    pub fn level_count(&self) -> usize {
        1 + self.mipmaps.len()
    }

    /// Samples the texture at normalized texture coordinate `(s, t)`.
    ///
    /// The `t` axis uses graphics convention: `t = 0` samples the bottom row, and `t = 1`
    /// samples the top row.
    #[must_use]
    pub fn sample(&self, s: f64, t: f64) -> Rgb {
        if self.image.is_empty() || !s.is_finite() || !t.is_finite() {
            return Rgb::BLACK;
        }

        sample_canvas(&self.image, s, t, self.filter, self.wrap_s, self.wrap_t)
    }

    /// Samples a mipmap level selected by `lod`, where `0.0` is the base image.
    ///
    /// [`TextureFilter::Nearest`] picks the nearest mip level. [`TextureFilter::Linear`] samples
    /// the two adjacent mip levels and blends them by the fractional part of `lod`.
    #[must_use]
    pub fn sample_lod(&self, s: f64, t: f64, lod: f64) -> Rgb {
        if self.image.is_empty() || !s.is_finite() || !t.is_finite() || !lod.is_finite() {
            return Rgb::BLACK;
        }

        if self.filter == TextureFilter::Linear && self.level_count() > 1 {
            let (lower, upper, blend) = mip_level_pair_from_lod(lod, self.level_count() - 1);
            let lower_sample = sample_canvas(
                self.level_image(lower),
                s,
                t,
                TextureFilter::Linear,
                self.wrap_s,
                self.wrap_t,
            );
            let upper_sample = sample_canvas(
                self.level_image(upper),
                s,
                t,
                TextureFilter::Linear,
                self.wrap_s,
                self.wrap_t,
            );
            lower_sample.lerp(upper_sample, blend)
        } else {
            let level = mip_level_from_lod(lod, self.level_count() - 1);
            sample_canvas(
                self.level_image(level),
                s,
                t,
                self.filter,
                self.wrap_s,
                self.wrap_t,
            )
        }
    }

    /// Estimates a mipmap level from texture-coordinate derivatives in screen pixels.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    pub fn lod_from_derivatives(
        &self,
        slope_s_x: f64,
        slope_t_x: f64,
        slope_s_y: f64,
        slope_t_y: f64,
    ) -> f64 {
        if self.mipmaps.is_empty()
            || !slope_s_x.is_finite()
            || !slope_t_x.is_finite()
            || !slope_s_y.is_finite()
            || !slope_t_y.is_finite()
        {
            return 0.0;
        }

        let width = f64::from(self.image.width());
        let height = f64::from(self.image.height());
        let x_footprint = (slope_s_x * width).hypot(slope_t_x * height);
        let y_footprint = (slope_s_y * width).hypot(slope_t_y * height);
        x_footprint
            .max(y_footprint)
            .max(1.0)
            .log2()
            .clamp(0.0, self.mipmaps.len() as f64)
    }

    fn level_image(&self, level: usize) -> &Canvas {
        if level == 0 {
            &self.image
        } else {
            &self.mipmaps[level - 1]
        }
    }

    pub(crate) fn active_sampler(&self) -> ActiveTextureSampler<'_> {
        match (self.filter, self.mipmaps.is_empty()) {
            (TextureFilter::Nearest, true) => ActiveTextureSampler::NearestBase {
                image: &self.image,
                wrap_s: self.wrap_s,
                wrap_t: self.wrap_t,
            },
            (TextureFilter::Linear, true) => ActiveTextureSampler::LinearBase {
                image: &self.image,
                wrap_s: self.wrap_s,
                wrap_t: self.wrap_t,
            },
            (TextureFilter::Nearest, false) => ActiveTextureSampler::NearestMip {
                texture: self,
                wrap_s: self.wrap_s,
                wrap_t: self.wrap_t,
                max_level: self.mipmaps.len(),
            },
            (TextureFilter::Linear, false) => ActiveTextureSampler::LinearMip {
                texture: self,
                wrap_s: self.wrap_s,
                wrap_t: self.wrap_t,
                max_level: self.mipmaps.len(),
            },
        }
    }
}

impl SurfaceTexture for Texture {
    fn sample_linear(&self, sample: TextureSample) -> LinearRgb {
        LinearRgb::from_rgb_linear_units(self.sample(sample.u, sample.v))
    }
}

impl ActiveTextureSampler<'_> {
    pub(crate) const fn uses_mips(&self) -> bool {
        matches!(self, Self::NearestMip { .. } | Self::LinearMip { .. })
    }

    pub(crate) fn sample(&self, s: f64, t: f64, lod: f64) -> Rgb {
        if !s.is_finite() || !t.is_finite() || !lod.is_finite() {
            return Rgb::BLACK;
        }

        match self {
            Self::NearestBase {
                image,
                wrap_s,
                wrap_t,
            } => {
                if image.is_empty() {
                    return Rgb::BLACK;
                }
                sample_nearest(image, apply_wrap(s, *wrap_s), apply_wrap(t, *wrap_t))
            }
            Self::LinearBase {
                image,
                wrap_s,
                wrap_t,
            } => {
                if image.is_empty() {
                    return Rgb::BLACK;
                }
                sample_linear(image, s, t, *wrap_s, *wrap_t)
            }
            Self::NearestMip {
                texture,
                wrap_s,
                wrap_t,
                max_level,
            } => {
                if texture.image.is_empty() {
                    return Rgb::BLACK;
                }
                let level = mip_level_from_lod(lod, *max_level);
                sample_nearest(
                    texture.level_image(level),
                    apply_wrap(s, *wrap_s),
                    apply_wrap(t, *wrap_t),
                )
            }
            Self::LinearMip {
                texture,
                wrap_s,
                wrap_t,
                max_level,
            } => {
                if texture.image.is_empty() {
                    return Rgb::BLACK;
                }
                let (lower, upper, blend) = mip_level_pair_from_lod(lod, *max_level);
                let lower_sample =
                    sample_linear(texture.level_image(lower), s, t, *wrap_s, *wrap_t);
                let upper_sample =
                    sample_linear(texture.level_image(upper), s, t, *wrap_s, *wrap_t);
                lower_sample.lerp(upper_sample, blend)
            }
        }
    }
}

fn sample_canvas(
    image: &Canvas,
    s: f64,
    t: f64,
    filter: TextureFilter,
    wrap_s: TextureWrap,
    wrap_t: TextureWrap,
) -> Rgb {
    match filter {
        TextureFilter::Nearest => {
            sample_nearest(image, apply_wrap(s, wrap_s), apply_wrap(t, wrap_t))
        }
        TextureFilter::Linear => sample_linear(image, s, t, wrap_s, wrap_t),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sample_nearest(image: &Canvas, s: f64, t: f64) -> Rgb {
    let width = image.width();
    let height = image.height();
    let x = (s * f64::from(width.saturating_sub(1))).round() as u32;
    let y = ((1.0 - t) * f64::from(height.saturating_sub(1))).round() as u32;
    pixel_at_storage(image, x, y)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sample_linear(image: &Canvas, s: f64, t: f64, wrap_s: TextureWrap, wrap_t: TextureWrap) -> Rgb {
    let width = image.width();
    let height = image.height();
    let (x, x0, x1) = linear_axis(s, width, wrap_s);
    let (y, y0, y1) = linear_axis(1.0 - t, height, wrap_t);
    let tx = x - f64::from(x0);
    let ty = y - f64::from(y0);

    let top = pixel_at_storage(image, x0, y0).lerp(pixel_at_storage(image, x1, y0), tx);
    let bottom = pixel_at_storage(image, x0, y1).lerp(pixel_at_storage(image, x1, y1), tx);
    top.lerp(bottom, ty)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn linear_axis(value: f64, size: u32, wrap: TextureWrap) -> (f64, u32, u32) {
    match wrap {
        TextureWrap::Clamp => {
            let coord = value.clamp(0.0, 1.0) * f64::from(size.saturating_sub(1));
            let low = coord.floor() as u32;
            (coord, low, (low + 1).min(size - 1))
        }
        TextureWrap::Repeat => {
            let coord = value.rem_euclid(1.0) * f64::from(size);
            let low = (coord.floor() as u32).min(size - 1);
            let high = if low + 1 == size { 0 } else { low + 1 };
            (coord, low, high)
        }
    }
}

fn pixel_at_storage(image: &Canvas, x: u32, y: u32) -> Rgb {
    let index = y as usize * image.width() as usize + x as usize;
    image.pixels()[index]
}

fn build_mipmaps(image: &Canvas) -> Vec<Canvas> {
    let mut levels = Vec::new();
    let mut source = image;
    while source.width() > 1 || source.height() > 1 {
        levels.push(downsample_mipmap_level(source));
        source = levels.last().expect("just pushed generated mipmap level");
    }
    levels
}

#[allow(clippy::cast_possible_truncation)]
fn downsample_mipmap_level(image: &Canvas) -> Canvas {
    let next_width = (image.width() / 2).max(1);
    let next_height = (image.height() / 2).max(1);
    let pixel_count = next_width as usize * next_height as usize;
    let pixels = {
        #[cfg(feature = "rayon")]
        {
            (0..pixel_count)
                .into_par_iter()
                .map(|idx| {
                    let x = idx % next_width as usize;
                    let y = idx / next_width as usize;
                    average_source_texels(image, x as u32 * 2, y as u32 * 2)
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            (0..pixel_count)
                .map(|idx| {
                    let x = idx % next_width as usize;
                    let y = idx / next_width as usize;
                    average_source_texels(image, x as u32 * 2, y as u32 * 2)
                })
                .collect()
        }
    };

    Canvas::from_pixels(next_width, next_height, pixels)
}

fn average_source_texels(image: &Canvas, x: u32, y: u32) -> Rgb {
    let mut red = 0_u32;
    let mut green = 0_u32;
    let mut blue = 0_u32;
    let mut count = 0_u32;

    for sample_y in y..=(y + 1).min(image.height() - 1) {
        for sample_x in x..=(x + 1).min(image.width() - 1) {
            let pixel = pixel_at_storage(image, sample_x, sample_y);
            red += u32::from(pixel.red);
            green += u32::from(pixel.green);
            blue += u32::from(pixel.blue);
            count += 1;
        }
    }

    Rgb::new(
        u8::try_from((red + count / 2) / count).unwrap_or(255),
        u8::try_from((green + count / 2) / count).unwrap_or(255),
        u8::try_from((blue + count / 2) / count).unwrap_or(255),
    )
}

fn apply_wrap(value: f64, wrap: TextureWrap) -> f64 {
    match wrap {
        TextureWrap::Clamp => value.clamp(0.0, 1.0),
        TextureWrap::Repeat => value.rem_euclid(1.0),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn mip_level_from_lod(lod: f64, max_level: usize) -> usize {
    (lod.round().max(0.0) as usize).min(max_level)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn mip_level_pair_from_lod(lod: f64, max_level: usize) -> (usize, usize, f64) {
    let lod = lod.max(0.0);
    let lower = (lod.floor() as usize).min(max_level);
    let upper = (lower + 1).min(max_level);
    let blend = if lower == upper {
        0.0
    } else {
        lod - lod.floor()
    };
    (lower, upper, blend)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_texture() -> Texture {
        Texture::from_canvas(Canvas::from_pixels(
            2,
            2,
            vec![Rgb::RED, Rgb::GREEN, Rgb::BLUE, Rgb::WHITE],
        ))
    }

    #[test]
    fn nearest_sampling_uses_bottom_left_texture_origin() {
        let texture = test_texture();

        assert_eq!(texture.sample(0.0, 0.0), Rgb::BLUE);
        assert_eq!(texture.sample(1.0, 1.0), Rgb::GREEN);
    }

    #[test]
    fn repeated_sampling_wraps_coordinates() {
        let texture = test_texture().wrap(TextureWrap::Repeat, TextureWrap::Repeat);

        assert_eq!(texture.sample(1.25, -0.25), texture.sample(0.25, 0.75));
    }

    #[test]
    fn linear_sampling_blends_neighboring_texels() {
        let texture = Texture::from_canvas(Canvas::from_pixels(
            2,
            2,
            vec![
                Rgb::new(0, 0, 0),
                Rgb::new(100, 0, 0),
                Rgb::new(0, 100, 0),
                Rgb::new(100, 100, 0),
            ],
        ))
        .filter(TextureFilter::Linear);

        assert_eq!(texture.sample(0.5, 0.5), Rgb::new(50, 50, 0));
    }

    #[test]
    fn linear_repeat_wraps_neighbor_texels_across_seam() {
        let texture = Texture::from_canvas(Canvas::from_pixels(2, 1, vec![Rgb::RED, Rgb::GREEN]))
            .wrap(TextureWrap::Repeat, TextureWrap::Repeat)
            .filter(TextureFilter::Linear);

        let sample = texture.sample(0.99, 0.0);

        assert!(sample.red > sample.green);
    }

    #[test]
    fn mipmapped_texture_samples_downsampled_levels() {
        let texture = test_texture().mipmapped();

        assert_eq!(texture.level_count(), 2);
        assert_eq!(texture.sample_lod(0.0, 0.0, 1.0), Rgb::new(128, 128, 128));
    }

    #[test]
    fn linear_mipmap_sampling_blends_adjacent_levels() {
        let texture = test_texture().filter(TextureFilter::Linear).mipmapped();

        assert_eq!(
            texture.sample_lod(0.0, 0.0, 0.5),
            Rgb::BLUE.lerp(Rgb::new(128, 128, 128), 0.5)
        );
    }

    #[test]
    fn nearest_mipmap_sampling_rounds_to_one_level() {
        let texture = test_texture().mipmapped();

        assert_eq!(texture.sample_lod(0.0, 0.0, 0.49), Rgb::BLUE);
        assert_eq!(texture.sample_lod(0.0, 0.0, 0.51), Rgb::new(128, 128, 128));
    }

    #[test]
    fn texture_lod_uses_largest_texel_footprint() {
        let texture = Texture::from_canvas(Canvas::new(8, 8, Rgb::WHITE)).mipmapped();

        assert!(texture.lod_from_derivatives(0.0, 0.0, 0.0, 0.0).abs() < f64::EPSILON);
        assert!(texture.lod_from_derivatives(0.5, 0.0, 0.0, 0.5) >= 2.0);
    }
}
