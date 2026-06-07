use std::{
    collections::HashMap,
    error::Error,
    fmt, io,
    path::{Path, PathBuf},
    sync::Arc,
};
#[cfg(not(feature = "external"))]
use std::{fs, str::FromStr};

use super::{
    colors::{LinearRgb, Rgb},
    display::Canvas,
    material::SurfaceMaterial,
    scene::SurfaceScene,
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
///
/// This is the canonical texture trait for both raster and ray-tracing code. Bitmap-backed
/// textures decode display RGB bytes into linear color by default; callers that need data textures
/// should keep those values in an explicit raw-linear representation.
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

/// Cache for bitmap textures loaded from filesystem paths.
#[derive(Clone, Debug, Default)]
pub struct TextureCache {
    textures: HashMap<PathBuf, Arc<Texture>>,
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

    /// Loads a texture from an image path.
    ///
    /// With the `external` feature enabled, this delegates to the external image conversion loader.
    /// Without that feature, it supports PPM input through the built-in parser.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, converted, or parsed.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let path = path.as_ref();
        #[cfg(feature = "external")]
        let canvas = {
            let path = path.to_str().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "texture path is not valid UTF-8",
                )
            })?;
            crate::external::ppmify(path, false)?
        };
        #[cfg(not(feature = "external"))]
        let canvas = load_ppm_canvas(path)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(Self::from_canvas(canvas))
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

    /// Generates and stores at most `level_count` texture levels, including the base image.
    ///
    /// Passing `0` or `1` stores no additional mipmaps. Larger values generate only the requested
    /// number of downsampled levels, stopping earlier if the texture reaches `1x1`.
    #[must_use]
    pub fn with_mip_levels(mut self, level_count: usize) -> Self {
        self.mipmaps = build_mipmaps_with_limit(&self.image, level_count.saturating_sub(1));
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

impl TextureCache {
    /// Creates an empty texture cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of cached textures.
    #[must_use]
    pub fn len(&self) -> usize {
        self.textures.len()
    }

    /// Returns true when the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }

    /// Removes all cached textures.
    pub fn clear(&mut self) {
        self.textures.clear();
    }

    /// Returns a cached texture by path.
    #[must_use]
    pub fn get(&self, path: impl AsRef<Path>) -> Option<Arc<Texture>> {
        self.textures.get(path.as_ref()).map(Arc::clone)
    }

    /// Inserts or replaces a texture under `path`.
    pub fn insert(&mut self, path: impl Into<PathBuf>, texture: Texture) -> Arc<Texture> {
        let texture = Arc::new(texture);
        self.textures.insert(path.into(), Arc::clone(&texture));
        texture
    }

    /// Loads `path` if needed and returns the cached texture.
    ///
    /// # Errors
    ///
    /// Returns an error if the texture cannot be loaded.
    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<Arc<Texture>, Box<dyn Error>> {
        let path = path.as_ref();
        if let Some(texture) = self.textures.get(path) {
            return Ok(Arc::clone(texture));
        }

        let texture = Arc::new(Texture::from_path(path)?);
        self.textures
            .insert(path.to_path_buf(), Arc::clone(&texture));
        Ok(texture)
    }

    /// Loads the diffuse texture referenced by `material`, if present.
    ///
    /// # Errors
    ///
    /// Returns an error if the referenced texture cannot be loaded.
    pub fn load_material_texture(
        &mut self,
        material: &SurfaceMaterial,
    ) -> Result<Option<Arc<Texture>>, Box<dyn Error>> {
        material
            .diffuse_texture
            .as_ref()
            .map_or(Ok(None), |path| self.load(path).map(Some))
    }

    /// Loads all diffuse textures referenced by a surface scene.
    ///
    /// # Errors
    ///
    /// Returns the first texture loading error.
    pub fn load_surface_scene_materials(
        &mut self,
        scene: &SurfaceScene,
    ) -> Result<(), Box<dyn Error>> {
        for mesh in scene.meshes() {
            self.load_material_texture(&mesh.material)?;
        }
        Ok(())
    }
}

#[cfg(not(feature = "external"))]
pub(crate) fn load_ppm_canvas(path: &Path) -> Result<Canvas, String> {
    let buffer = fs::read(path).map_err(|error| error.to_string())?;
    let mut cursor = 0;

    let magic = next_ppm_token(&buffer, &mut cursor).ok_or("Invalid PPM file: missing magic")?;
    let width = parse_ppm_token::<u32>(
        next_ppm_token(&buffer, &mut cursor).ok_or("Invalid PPM file: missing width")?,
    )?;
    let height = parse_ppm_token::<u32>(
        next_ppm_token(&buffer, &mut cursor).ok_or("Invalid PPM file: missing height")?,
    )?;
    let maxval = parse_ppm_token::<u16>(
        next_ppm_token(&buffer, &mut cursor).ok_or("Invalid PPM file: missing maxval")?,
    )?;

    if maxval == 0 {
        return Err("unsupported PPM maxval 0; maxval must be 1..=65535".to_string());
    }

    let pixel_count = u64::from(width) * u64::from(height);
    let pixel_count = usize::try_from(pixel_count).map_err(|_| "PPM image too large")?;

    let pixels = match magic {
        b"P3" => parse_p3_pixels(&buffer, &mut cursor, pixel_count, maxval)?,
        b"P6" => parse_p6_pixels(&buffer, &mut cursor, pixel_count, maxval)?,
        other => {
            return Err(format!(
                "Invalid PPM file: unsupported magic {}",
                String::from_utf8_lossy(other)
            ));
        }
    };

    Ok(Canvas::from_pixels(width, height, pixels))
}

#[cfg(not(feature = "external"))]
fn next_ppm_token<'a>(buffer: &'a [u8], cursor: &mut usize) -> Option<&'a [u8]> {
    loop {
        while *cursor < buffer.len() && buffer[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        }

        if *cursor < buffer.len() && buffer[*cursor] == b'#' {
            while *cursor < buffer.len() && buffer[*cursor] != b'\n' {
                *cursor += 1;
            }
            continue;
        }

        break;
    }

    if *cursor >= buffer.len() {
        return None;
    }

    let start = *cursor;
    while *cursor < buffer.len()
        && !buffer[*cursor].is_ascii_whitespace()
        && buffer[*cursor] != b'#'
    {
        *cursor += 1;
    }

    Some(&buffer[start..*cursor])
}

#[cfg(not(feature = "external"))]
fn parse_ppm_token<T>(token: &[u8]) -> Result<T, String>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    let token = std::str::from_utf8(token).map_err(|error| error.to_string())?;
    token.parse::<T>().map_err(|error| error.to_string())
}

#[cfg(not(feature = "external"))]
fn scale_ppm_channel(value: u16, maxval: u16) -> Result<u8, String> {
    if value > maxval {
        return Err(format!("PPM channel value {value} exceeds maxval {maxval}"));
    }

    Ok(
        u8::try_from((u32::from(value) * 255 + u32::from(maxval) / 2) / u32::from(maxval))
            .unwrap_or(255),
    )
}

#[cfg(not(feature = "external"))]
fn parse_p3_pixels(
    buffer: &[u8],
    cursor: &mut usize,
    pixel_count: usize,
    maxval: u16,
) -> Result<Vec<Rgb>, String> {
    let remaining = buffer.len().saturating_sub(*cursor);
    let estimated_token_capacity = remaining.div_ceil(2);
    let mut pixels = Vec::with_capacity(pixel_count.min(estimated_token_capacity / 3));
    for _ in 0..pixel_count {
        let red = parse_ppm_token::<u16>(
            next_ppm_token(buffer, cursor).ok_or("Invalid PPM file: missing red channel")?,
        )?;
        let green = parse_ppm_token::<u16>(
            next_ppm_token(buffer, cursor).ok_or("Invalid PPM file: missing green channel")?,
        )?;
        let blue = parse_ppm_token::<u16>(
            next_ppm_token(buffer, cursor).ok_or("Invalid PPM file: missing blue channel")?,
        )?;

        pixels.push(Rgb::new(
            scale_ppm_channel(red, maxval)?,
            scale_ppm_channel(green, maxval)?,
            scale_ppm_channel(blue, maxval)?,
        ));
    }
    Ok(pixels)
}

#[cfg(not(feature = "external"))]
fn parse_p6_pixels(
    buffer: &[u8],
    cursor: &mut usize,
    pixel_count: usize,
    maxval: u16,
) -> Result<Vec<Rgb>, String> {
    consume_p6_separator(buffer, cursor)?;

    let bytes_per_sample = if maxval < 256 { 1 } else { 2 };
    let needed = pixel_count
        .checked_mul(3)
        .and_then(|count| count.checked_mul(bytes_per_sample))
        .ok_or("PPM image data is too large")?;
    if buffer.len().saturating_sub(*cursor) < needed {
        return Err(format!(
            "Invalid PPM file: expected {needed} bytes of pixel data, found {}",
            buffer.len().saturating_sub(*cursor)
        ));
    }

    let mut pixels = Vec::with_capacity(pixel_count);
    if bytes_per_sample == 1 {
        for chunk in buffer[*cursor..*cursor + needed].chunks_exact(3) {
            pixels.push(Rgb::new(
                scale_ppm_channel(u16::from(chunk[0]), maxval)?,
                scale_ppm_channel(u16::from(chunk[1]), maxval)?,
                scale_ppm_channel(u16::from(chunk[2]), maxval)?,
            ));
        }
    } else {
        for chunk in buffer[*cursor..*cursor + needed].chunks_exact(6) {
            let red = u16::from_be_bytes([chunk[0], chunk[1]]);
            let green = u16::from_be_bytes([chunk[2], chunk[3]]);
            let blue = u16::from_be_bytes([chunk[4], chunk[5]]);
            pixels.push(Rgb::new(
                scale_ppm_channel(red, maxval)?,
                scale_ppm_channel(green, maxval)?,
                scale_ppm_channel(blue, maxval)?,
            ));
        }
    }

    Ok(pixels)
}

#[cfg(not(feature = "external"))]
fn consume_p6_separator(buffer: &[u8], cursor: &mut usize) -> Result<(), String> {
    if *cursor >= buffer.len() || !buffer[*cursor].is_ascii_whitespace() {
        return Err("Invalid PPM file: missing binary data separator".to_string());
    }

    let separator = buffer[*cursor];
    *cursor += 1;
    if separator == b'\r' && *cursor < buffer.len() && buffer[*cursor] == b'\n' {
        *cursor += 1;
    }
    Ok(())
}

impl SurfaceTexture for Texture {
    fn sample_linear(&self, sample: TextureSample) -> LinearRgb {
        LinearRgb::from_rgb_srgb(self.sample(sample.u, sample.v))
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
    build_mipmaps_with_limit(image, usize::MAX)
}

fn build_mipmaps_with_limit(image: &Canvas, additional_levels: usize) -> Vec<Canvas> {
    let mut levels = Vec::new();
    let mut source = image;
    while levels.len() < additional_levels && (source.width() > 1 || source.height() > 1) {
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
    #[cfg(not(feature = "external"))]
    use crate::gmath::polygon_matrix::PolygonMatrix;
    #[cfg(not(feature = "external"))]
    use crate::graphics::raytracing::Lambertian;
    use std::path::PathBuf;

    fn test_texture() -> Texture {
        Texture::from_canvas(Canvas::from_pixels(
            2,
            2,
            vec![Rgb::RED, Rgb::GREEN, Rgb::BLUE, Rgb::WHITE],
        ))
    }

    #[test]
    fn texture_cache_can_bind_material_texture_metadata() {
        let path = PathBuf::from("inline.ppm");
        let material = SurfaceMaterial::default().with_diffuse_texture(&path);
        let mut cache = TextureCache::new();

        let inserted = cache.insert(path.clone(), test_texture());
        let loaded = cache
            .load_material_texture(&material)
            .expect("inserted texture should load")
            .expect("material should reference a texture");

        assert_eq!(cache.len(), 1);
        assert!(Arc::ptr_eq(&inserted, &loaded));
        assert!(cache.get(&path).is_some());
    }

    #[cfg(not(feature = "external"))]
    fn temp_ppm_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("gartus-texture-{name}-{}.ppm", std::process::id()))
    }

    #[cfg(not(feature = "external"))]
    #[test]
    fn ppm_loader_rejects_huge_p3_header_without_large_preallocation() {
        let path = temp_ppm_path("huge-p3");
        std::fs::write(&path, b"P3\n1000000 1000000\n255\n").unwrap();

        let error = load_ppm_canvas(&path).expect_err("truncated huge image should fail");

        assert!(error.contains("missing red channel"));
        let _ = std::fs::remove_file(path);
    }

    #[cfg(not(feature = "external"))]
    #[test]
    fn ppm_loader_accepts_p6_crlf_separator() {
        let path = temp_ppm_path("p6-crlf");
        std::fs::write(&path, b"P6\r\n1 1\r\n255\r\n\xff\x00\x80").unwrap();

        let canvas = load_ppm_canvas(&path).expect("parse p6 ppm");

        assert_eq!(canvas.pixels(), &[Rgb::new(255, 0, 128)]);
        let _ = std::fs::remove_file(path);
    }

    #[cfg(not(feature = "external"))]
    #[test]
    fn texture_cache_loads_surface_scene_material_paths_for_ray_materials() {
        let path = temp_ppm_path("surface-scene-material");
        std::fs::write(&path, b"P3\n2 1\n255\n255 0 0\n0 255 0\n").unwrap();
        let material = SurfaceMaterial::default().with_diffuse_texture(&path);
        let mut polygons = PolygonMatrix::new();
        polygons.add_polygon((0.0, 0.0, -1.0), (1.0, 0.0, -1.0), (0.0, 1.0, -1.0));
        let mut scene = SurfaceScene::new();
        scene.add_mesh(polygons, material);
        let mut cache = TextureCache::new();

        cache
            .load_surface_scene_materials(&scene)
            .expect("surface scene texture path should load");
        let texture = cache.get(&path).expect("texture should be cached");
        let texture: SurfaceTextureRef = texture;
        let lambertian = Lambertian::from_shared_texture(texture);

        assert_eq!(cache.len(), 1);
        assert_eq!(
            lambertian
                .texture()
                .sample_linear(TextureSample::new(0.0, 0.0, Point::default())),
            LinearRgb::from_rgb_srgb(Rgb::RED)
        );
        assert_eq!(
            lambertian
                .texture()
                .sample_linear(TextureSample::new(1.0, 0.0, Point::default())),
            LinearRgb::from_rgb_srgb(Rgb::GREEN)
        );
        let _ = std::fs::remove_file(path);
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
    fn mip_level_count_can_be_limited() {
        let full = Texture::from_canvas(Canvas::new(8, 8, Rgb::WHITE)).mipmapped();
        let limited = Texture::from_canvas(Canvas::new(8, 8, Rgb::WHITE)).with_mip_levels(2);
        let base_only = Texture::from_canvas(Canvas::new(8, 8, Rgb::WHITE)).with_mip_levels(1);

        assert_eq!(full.level_count(), 4);
        assert_eq!(limited.level_count(), 2);
        assert_eq!(base_only.level_count(), 1);
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
