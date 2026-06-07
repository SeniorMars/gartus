use crate::graphics::{colors::Rgb, lighting::Lighting};

use core::slice;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{
    error::Error,
    fmt,
    fs::File,
    io::{self, BufWriter, Write},
    ops::{Index, IndexMut},
    process::{Command, Stdio},
};

/// Controls how filled polygon triangles choose their draw color.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PolygonColorMode {
    /// Use the canvas line color for every triangle.
    #[default]
    LineColor,
    /// Calculate one flat Phong reflection color per triangle.
    PhongReflection,
    /// Generate a stable pseudo-random color from each triangle index.
    DeterministicRandom,
    /// Generate stable color variation blended from the canvas line color.
    TintedFromLine,
}

/// Controls how polygon surfaces are rendered.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ShadingMode {
    /// Draw only triangle edges.
    Wireframe,
    /// Calculate one color per polygon and fill the whole polygon with it.
    ///
    /// Use [`PolygonColorMode::PhongReflection`] for lit flat shading.
    #[default]
    Flat,
    /// Calculate one lit color per vertex and interpolate colors across each polygon.
    Gouraud,
    /// Interpolate vertex normals and calculate lighting at each plotted pixel.
    Phong,
    /// Interpolate vertex normals and calculate quantized banded lighting per pixel.
    Toon,
}

/// Inclusive 2D coordinate bounds for mapping pixels into a mathematical domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain2D {
    /// Minimum x coordinate.
    pub x_min: f64,
    /// Maximum x coordinate.
    pub x_max: f64,
    /// Minimum y coordinate.
    pub y_min: f64,
    /// Maximum y coordinate.
    pub y_max: f64,
}

impl Domain2D {
    /// Creates a 2D domain.
    ///
    /// # Panics
    /// Panics if any bound is non-finite, or if either axis has identical endpoints.
    #[must_use]
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        assert!(
            [x_min, x_max, y_min, y_max]
                .iter()
                .all(|value| value.is_finite()),
            "domain bounds must be finite"
        );
        assert!(
            (x_max - x_min).abs() > f64::EPSILON,
            "domain x bounds must differ"
        );
        assert!(
            (y_max - y_min).abs() > f64::EPSILON,
            "domain y bounds must differ"
        );
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ZSpan {
    pub x0: i64,
    pub x1: i64,
    pub y: i64,
    pub z: f64,
    pub dz: f64,
}

#[derive(Clone, Debug)]
/// An art [Canvas] / computer screen is represented here.
#[must_use]
pub struct Canvas {
    width: u32,
    height: u32,
    pixels: Vec<Rgb>,
    zbuffer: Vec<f64>,
    upper_left_origin: bool,
    wrapped: bool,
    line: Rgb,
    /// Width of drawn lines in pixels. Default 1.0.
    line_width: f64,
    polygon_color_mode: PolygonColorMode,
    shading_mode: ShadingMode,
    lighting: Lighting,
}

/// Error returned by checked [`Canvas`] constructors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanvasBuildError {
    /// `width * height` overflowed while computing the pixel count.
    DimensionsOverflow,
    /// The canvas has more pixels than can be indexed on this platform.
    TooLargeForPlatform,
    /// The supplied pixel data length did not match `width * height`.
    PixelDataLengthMismatch {
        /// Number of pixels required by the canvas dimensions.
        expected: usize,
        /// Number of pixels supplied by the caller.
        actual: usize,
    },
    /// The backing pixel or z-buffer allocation could not be reserved.
    AllocationFailed,
}

impl fmt::Display for CanvasBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionsOverflow => formatter.write_str("canvas dimensions overflow"),
            Self::TooLargeForPlatform => formatter.write_str("canvas too large for this platform"),
            Self::PixelDataLengthMismatch { expected, actual } => write!(
                formatter,
                "pixel data length mismatch: expected {expected}, got {actual}"
            ),
            Self::AllocationFailed => formatter.write_str("canvas allocation failed"),
        }
    }
}

impl Error for CanvasBuildError {}

/// Lightweight RGB image buffer without canvas drawing state or z-buffer data.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<Rgb>,
}

impl RgbImage {
    /// Creates an image filled with `pixel`.
    ///
    /// # Panics
    ///
    /// Panics if `width * height` overflows or if allocation fails.
    #[must_use]
    pub fn new(width: u32, height: u32, pixel: Rgb) -> Self {
        Self::try_new(width, height, pixel).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Creates a checked image filled with `pixel`.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` cannot be represented or if allocation fails.
    pub fn try_new(width: u32, height: u32, pixel: Rgb) -> Result<Self, CanvasBuildError> {
        let pixel_count = Canvas::try_pixel_count(width, height)?;
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(pixel_count)
            .map_err(|_| CanvasBuildError::AllocationFailed)?;
        pixels.resize(pixel_count, pixel);
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Creates an image from exact pixel data.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len()` is not `width * height`.
    #[must_use]
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<Rgb>) -> Self {
        Self::try_from_pixels(width, height, pixels).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Creates a checked image from exact pixel data.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` cannot be represented or if `pixels.len()` does not
    /// match that count.
    pub fn try_from_pixels(
        width: u32,
        height: u32,
        pixels: Vec<Rgb>,
    ) -> Result<Self, CanvasBuildError> {
        let expected = Canvas::try_pixel_count(width, height)?;
        if pixels.len() != expected {
            return Err(CanvasBuildError::PixelDataLengthMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Returns the image width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the number of pixels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Returns true when this image has no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Returns immutable pixel data.
    #[must_use]
    pub fn pixels(&self) -> &[Rgb] {
        &self.pixels
    }

    /// Returns mutable pixel data.
    #[must_use]
    pub fn pixels_mut(&mut self) -> &mut [Rgb] {
        &mut self.pixels
    }

    /// Consumes the image and returns its pixel data.
    #[must_use]
    pub fn into_pixels(self) -> Vec<Rgb> {
        self.pixels
    }

    /// Converts this image into a drawing canvas.
    pub fn into_canvas(self) -> Canvas {
        self.into()
    }
}

impl From<RgbImage> for Canvas {
    fn from(image: RgbImage) -> Self {
        Self::from_pixels(image.width, image.height, image.pixels)
    }
}

impl From<Canvas> for RgbImage {
    fn from(canvas: Canvas) -> Self {
        Self {
            width: canvas.width,
            height: canvas.height,
            pixels: canvas.pixels,
        }
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixels: Vec::new(),
            zbuffer: Vec::new(),
            upper_left_origin: false,
            wrapped: true,
            line: Rgb::default(),
            line_width: 1.0,
            polygon_color_mode: PolygonColorMode::default(),
            shading_mode: ShadingMode::default(),
            lighting: Lighting::default(),
        }
    }
}

impl Canvas {
    /// Returns a [`CanvasBuilder`] to configure a new [Canvas].
    pub fn builder(width: u32, height: u32) -> CanvasBuilder {
        CanvasBuilder::new(width, height)
    }

    pub(crate) fn pixel_count(width: u32, height: u32) -> usize {
        Self::try_pixel_count(width, height).unwrap_or_else(|err| panic!("{err}"))
    }

    fn try_pixel_count(width: u32, height: u32) -> Result<usize, CanvasBuildError> {
        let count = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or(CanvasBuildError::DimensionsOverflow)?;
        usize::try_from(count).map_err(|_| CanvasBuildError::TooLargeForPlatform)
    }

    fn zbuffer_for_pixel_count(pixel_count: usize) -> Result<Vec<f64>, CanvasBuildError> {
        let mut zbuffer = Vec::new();
        zbuffer
            .try_reserve_exact(pixel_count)
            .map_err(|_| CanvasBuildError::AllocationFailed)?;
        zbuffer.resize(pixel_count, f64::NEG_INFINITY);
        Ok(zbuffer)
    }

    /// Returns a new blank [Canvas] to be drawn on.
    ///
    /// # Arguments
    ///
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `line_color` - An RGB or HSL value that will also represent the default color for the
    ///   drawing line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// ```
    pub fn new(width: u32, height: u32, line_color: Rgb) -> Self {
        Self::builder(width, height).line_color(line_color).build()
    }

    /// Returns a checked new blank [`Canvas`].
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` cannot be represented or if allocation fails.
    pub fn try_new(width: u32, height: u32, line_color: Rgb) -> Result<Self, CanvasBuildError> {
        Self::builder(width, height)
            .line_color(line_color)
            .try_build()
    }

    /// Returns a new [Canvas] to be drawn on with a specific background color.
    ///
    /// # Arguments
    ///
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `background_color` - A RGB or HSL value that will represent the color of the
    ///   background color that will fill the [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let background_color = Rgb::new(1, 2, 3);
    /// let image = Canvas::new_with_bg(500, 500, background_color);
    /// ```
    pub fn new_with_bg(width: u32, height: u32, background_color: Rgb) -> Self {
        Self::builder(width, height)
            .background(background_color)
            .build()
    }

    /// Returns a checked new [`Canvas`] with a specific background color.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` cannot be represented or if allocation fails.
    pub fn try_new_with_bg(
        width: u32,
        height: u32,
        background_color: Rgb,
    ) -> Result<Self, CanvasBuildError> {
        Self::builder(width, height)
            .background(background_color)
            .try_build()
    }

    /// Returns a new [`Canvas`] initialized with exact pixel data.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len()` is not `width * height`.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<Rgb>) -> Self {
        Self::from_pixels_with_options(width, height, pixels, false, true)
    }

    /// Returns a checked [`Canvas`] initialized with exact pixel data.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` cannot be represented, if `pixels.len()` does not match
    /// that count, or if z-buffer allocation fails.
    pub fn try_from_pixels(
        width: u32,
        height: u32,
        pixels: Vec<Rgb>,
    ) -> Result<Self, CanvasBuildError> {
        Self::try_from_pixels_with_options(width, height, pixels, false, true)
    }

    /// Returns a new [`Canvas`] initialized with exact pixel data and coordinate options.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len()` is not `width * height`.
    pub fn from_pixels_with_options(
        width: u32,
        height: u32,
        pixels: Vec<Rgb>,
        upper_left_origin: bool,
        wrapped: bool,
    ) -> Self {
        Self::try_from_pixels_with_options(width, height, pixels, upper_left_origin, wrapped)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Returns a checked [`Canvas`] initialized with exact pixel data and coordinate options.
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` cannot be represented, if `pixels.len()` does not match
    /// that count, or if z-buffer allocation fails.
    pub fn try_from_pixels_with_options(
        width: u32,
        height: u32,
        pixels: Vec<Rgb>,
        upper_left_origin: bool,
        wrapped: bool,
    ) -> Result<Self, CanvasBuildError> {
        let pixel_count = Self::try_pixel_count(width, height)?;
        if pixels.len() != pixel_count {
            return Err(CanvasBuildError::PixelDataLengthMismatch {
                expected: pixel_count,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
            zbuffer: Self::zbuffer_for_pixel_count(pixel_count)?,
            upper_left_origin,
            wrapped,
            line: Rgb::default(),
            line_width: 1.0,
            polygon_color_mode: PolygonColorMode::default(),
            shading_mode: ShadingMode::default(),
            lighting: Lighting::default(),
        })
    }

    /// Returns a new [`Canvas`] initialized by evaluating `pixel` for every storage coordinate.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_fn<F>(width: u32, height: u32, mut pixel: F) -> Self
    where
        F: FnMut(u32, u32) -> Rgb,
    {
        let mut pixels = Vec::with_capacity(Self::pixel_count(width, height));
        for y in 0..height {
            for x in 0..width {
                pixels.push(pixel(x, y));
            }
        }
        Self::from_pixels(width, height, pixels)
    }

    /// Returns a new [`Canvas`] initialized by independently evaluating `pixel` for every
    /// storage coordinate.
    ///
    /// When the `rayon` feature is enabled, pixels are evaluated in parallel. Use this for
    /// renderers where each pixel is deterministic and does not depend on traversal order.
    pub fn from_fn_independent<F>(width: u32, height: u32, pixel: F) -> Self
    where
        F: Fn(u32, u32) -> Rgb + Send + Sync,
    {
        Self::from_fn_independent_with_options(width, height, pixel, false, true)
    }

    /// Returns a new [`Canvas`] initialized by independently evaluating `pixel`, with explicit
    /// coordinate origin and wrapping options.
    ///
    /// # Panics
    ///
    /// Panics if the canvas dimensions exceed addressable `u32` pixel coordinates.
    pub fn from_fn_independent_with_options<F>(
        width: u32,
        height: u32,
        pixel: F,
        upper_left_origin: bool,
        wrapped: bool,
    ) -> Self
    where
        F: Fn(u32, u32) -> Rgb + Send + Sync,
    {
        let pixel_count = Self::pixel_count(width, height);
        let width_usize = width as usize;
        let pixels = {
            #[cfg(feature = "rayon")]
            {
                (0..pixel_count)
                    .into_par_iter()
                    .map(|idx| {
                        let x = u32::try_from(idx % width_usize).expect("pixel x fits u32");
                        let y = u32::try_from(idx / width_usize).expect("pixel y fits u32");
                        pixel(x, y)
                    })
                    .collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                (0..pixel_count)
                    .map(|idx| {
                        let x = u32::try_from(idx % width_usize).expect("pixel x fits u32");
                        let y = u32::try_from(idx / width_usize).expect("pixel y fits u32");
                        pixel(x, y)
                    })
                    .collect()
            }
        };

        Self::from_pixels_with_options(width, height, pixels, upper_left_origin, wrapped)
    }

    /// Returns a new [`Canvas`] by mapping each pixel into a 2D coordinate domain.
    ///
    /// The first storage row maps to `domain.y_max` and the last row approaches `domain.y_min`,
    /// which matches the common top-to-bottom scan order used by image and fractal renderers.
    /// Zero width or height returns an empty canvas without evaluating `pixel`.
    pub fn from_domain<F>(width: u32, height: u32, domain: Domain2D, mut pixel: F) -> Self
    where
        F: FnMut(f64, f64) -> Rgb,
    {
        if width == 0 || height == 0 {
            return Self::from_pixels(width, height, Vec::new());
        }

        let scale_x = (domain.x_max - domain.x_min) / f64::from(width);
        let scale_y = (domain.y_max - domain.y_min) / f64::from(height);
        Self::from_fn(width, height, |x, y| {
            let px = domain.x_min + f64::from(x) * scale_x;
            let py = domain.y_max - f64::from(y) * scale_y;
            pixel(px, py)
        })
    }

    /// Returns a lower-resolution canvas by averaging `factor` by `factor` pixel blocks.
    ///
    /// This is the final step of supersampling: render into a canvas whose dimensions are
    /// multiplied by `factor`, then downsample it to smooth jagged edges.
    ///
    /// # Panics
    /// Panics if `factor` is zero or if the canvas dimensions are not divisible by `factor`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn downsample(&self, factor: u32) -> Self {
        assert!(factor > 0, "downsample factor must be positive");
        assert_eq!(
            self.width % factor,
            0,
            "canvas width must be divisible by downsample factor"
        );
        assert_eq!(
            self.height % factor,
            0,
            "canvas height must be divisible by downsample factor"
        );

        let width = self.width / factor;
        let height = self.height / factor;
        let samples = u64::from(factor) * u64::from(factor);
        #[allow(clippy::cast_possible_truncation)]
        let source_width = self.width as usize;
        let pixel_count = Self::pixel_count(width, height);
        let pixels = {
            #[cfg(feature = "rayon")]
            {
                (0..pixel_count)
                    .into_par_iter()
                    .map(|idx| {
                        self.downsample_pixel(
                            idx,
                            width as usize,
                            factor as usize,
                            source_width,
                            samples,
                        )
                    })
                    .collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                (0..pixel_count)
                    .map(|idx| {
                        self.downsample_pixel(
                            idx,
                            width as usize,
                            factor as usize,
                            source_width,
                            samples,
                        )
                    })
                    .collect()
            }
        };

        Self {
            width,
            height,
            pixels,
            zbuffer: vec![f64::NEG_INFINITY; Self::pixel_count(width, height)],
            upper_left_origin: self.upper_left_origin,
            wrapped: self.wrapped,
            line: self.line,
            line_width: self.line_width,
            polygon_color_mode: self.polygon_color_mode,
            shading_mode: self.shading_mode,
            lighting: self.lighting.clone(),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn downsample_pixel(
        &self,
        idx: usize,
        width: usize,
        factor: usize,
        source_width: usize,
        samples: u64,
    ) -> Rgb {
        let x = idx % width;
        let y = idx / width;
        let mut red = 0_u64;
        let mut green = 0_u64;
        let mut blue = 0_u64;
        for sample_y in 0..factor {
            for sample_x in 0..factor {
                let source_x = x * factor + sample_x;
                let source_y = y * factor + sample_y;
                let pixel = self.pixels[source_y * source_width + source_x];
                red += u64::from(pixel.red);
                green += u64::from(pixel.green);
                blue += u64::from(pixel.blue);
            }
        }
        Rgb::new(
            ((red + samples / 2) / samples) as u8,
            ((green + samples / 2) / samples) as u8,
            ((blue + samples / 2) / samples) as u8,
        )
    }

    pub(crate) fn with_pixels_like(&self, pixels: Vec<Rgb>) -> Self {
        assert_eq!(
            pixels.len(),
            self.pixels.len(),
            "new pixel data must match canvas size"
        );
        Self {
            width: self.width,
            height: self.height,
            pixels,
            zbuffer: vec![f64::NEG_INFINITY; self.pixels.len()],
            upper_left_origin: self.upper_left_origin,
            wrapped: self.wrapped,
            line: self.line,
            line_width: self.line_width,
            polygon_color_mode: self.polygon_color_mode,
            shading_mode: self.shading_mode,
            lighting: self.lighting.clone(),
        }
    }

    /// Returns the width of a [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// let width = image.width();
    /// ```
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height of a [Canvas].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// let height = image.height();
    /// ```
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the total size of of a [Canvas].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// let size = image.len();
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Returns true if the [Canvas] has no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Returns true when external `(0, 0)` coordinates map to the top-left pixel.
    #[must_use]
    pub fn upper_left_origin(&self) -> bool {
        self.upper_left_origin
    }

    /// Sets whether external `(0, 0)` coordinates map to the top-left pixel.
    pub fn set_upper_left_origin(&mut self, upper_left_origin: bool) {
        self.upper_left_origin = upper_left_origin;
    }

    /// Returns true when out-of-bounds coordinates wrap around canvas edges.
    #[must_use]
    pub fn wrapped(&self) -> bool {
        self.wrapped
    }

    /// Sets whether out-of-bounds coordinates wrap around canvas edges.
    pub fn set_wrapped(&mut self, wrapped: bool) {
        self.wrapped = wrapped;
    }

    /// Returns an iterator over the pixels of the [Canvas].
    pub fn iter(&self) -> slice::Iter<'_, Rgb> {
        self.pixels.iter()
    }

    /// Returns a mutable iterator over the pixels of the [Canvas].
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, Rgb> {
        self.pixels.iter_mut()
    }

    /// Returns an iterator that iterates over the rows of the [Canvas].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// let iter = image.iter_row();
    /// ```
    pub fn iter_row(&self) -> slice::ChunksExact<'_, Rgb> {
        self.pixels.chunks_exact((self.width as usize).max(1))
    }

    /// Returns a mutable iterator that iterates over a specific row.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// let mut iter = image.iter_row_mut();
    /// ```
    pub fn iter_row_mut(&mut self) -> slice::ChunksExactMut<'_, Rgb> {
        self.pixels.chunks_exact_mut((self.width as usize).max(1))
    }

    /// Returns a reference to the `(x, y)` pixel in the canvas.
    ///
    /// Returns `None` if out of bounds.
    #[must_use]
    pub fn get_pixel(&self, x: i64, y: i64) -> Option<&Rgb> {
        let (x, y) = self.normalize_coords(x, y)?;
        Some(&self.pixels[y as usize * self.width as usize + x as usize])
    }

    /// Returns a mutable reference to the `(x, y)` pixel in the canvas.
    ///
    /// Returns `None` if out of bounds.
    pub fn get_pixel_mut(&mut self, x: i64, y: i64) -> Option<&mut Rgb> {
        let (x, y) = self.normalize_coords(x, y)?;
        let width = self.width;
        Some(&mut self.pixels[y as usize * width as usize + x as usize])
    }

    /// Returns the z-buffer value at `(x, y)`, or `None` if out of bounds.
    #[must_use]
    pub fn get_zbuffer(&self, x: i64, y: i64) -> Option<f64> {
        let (x, y) = self.normalize_coords(x, y)?;
        Some(self.zbuffer[y as usize * self.width as usize + x as usize])
    }

    /// Maps external coordinates (potentially negative or out-of-bounds) to
    /// internal canvas coordinates based on wrapping and origin settings.
    ///
    /// Returns `None` if the point is clipped.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[must_use]
    pub fn normalize_coords(&self, x: i64, y: i64) -> Option<(u32, u32)> {
        let (width, height) = (i64::from(self.width), i64::from(self.height));
        if width == 0 || height == 0 {
            return None;
        }

        let (x, y) = if self.wrapped {
            (x.rem_euclid(width), y.rem_euclid(height))
        } else if x >= 0 && x < width && y >= 0 && y < height {
            (x, y)
        } else {
            return None;
        };

        let y = if self.upper_left_origin {
            y
        } else {
            height - 1 - y
        };

        Some((x as u32, y as u32))
    }

    /// Sets the pixel at (x, y) to `pixel`.
    ///
    /// # Arguments
    ///
    /// * `pixel` - RGB value that will be plotted at `(x, y)`
    /// * `x` - An signed int that represents horizontal location
    /// * `y` - An signed int that represents vertical location
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// let pixel = Rgb::new(255, 255, 255);
    /// image.plot(&pixel, 250, 250);
    /// ```
    pub fn plot(&mut self, pixel: &Rgb, x: i64, y: i64) {
        self.plot_z(pixel, x, y, 0.0);
    }

    /// Sets the pixel at `(x, y)` if `z` is closer than the current z-buffer value.
    pub fn plot_z(&mut self, pixel: &Rgb, x: i64, y: i64, z: f64) {
        if !z.is_finite() {
            return;
        }

        if let Some((x, y)) = self.normalize_coords(x, y) {
            let index = y as usize * self.width as usize + x as usize;
            if z > self.zbuffer[index] {
                self.pixels[index] = *pixel;
                self.zbuffer[index] = z;
            }
        }
    }

    /// Returns the storage index for a visible z-buffered pixel.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn visible_pixel_index(&self, x: i64, y: i64, z: f64) -> Option<usize> {
        if !z.is_finite() {
            return None;
        }

        let (x, y) = self.normalize_coords(x, y)?;
        let index = y as usize * self.width as usize + x as usize;
        (z > self.zbuffer[index]).then_some(index)
    }

    /// Returns the storage index for a visible non-wrapping, already-clipped screen pixel.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn visible_pixel_index_clipped_unchecked(
        &self,
        x: i64,
        y: i64,
        z: f64,
    ) -> Option<usize> {
        if !z.is_finite() {
            return None;
        }

        debug_assert!(!self.wrapped);
        debug_assert!(x >= 0 && x < i64::from(self.width));
        debug_assert!(y >= 0 && y < i64::from(self.height));

        let storage_y = if self.upper_left_origin {
            y
        } else {
            i64::from(self.height) - 1 - y
        };
        let index = storage_y as usize * self.width as usize + x as usize;
        (z > self.zbuffer[index]).then_some(index)
    }

    /// Sets a z-buffered pixel by storage index.
    pub(crate) fn plot_z_index_unchecked(&mut self, index: usize, pixel: Rgb, z: f64) {
        debug_assert!(index < self.pixels.len());
        self.pixels[index] = pixel;
        self.zbuffer[index] = z;
    }

    /// Draws a clipped, non-wrapping horizontal z-buffered span.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub(crate) fn plot_z_span_clipped(
        &mut self,
        pixel: Rgb,
        mut x0: i64,
        mut x1: i64,
        y: i64,
        mut z: f64,
        dz: f64,
    ) {
        let width = i64::from(self.width);
        let height = i64::from(self.height);
        if width == 0 || height == 0 || y < 0 || y >= height || x0 > x1 || !z.is_finite() {
            return;
        }

        if x1 < 0 || x0 >= width {
            return;
        }

        if x0 < 0 {
            z += dz * (-x0) as f64;
            x0 = 0;
        }
        x1 = x1.min(width - 1);

        let storage_y = if self.upper_left_origin {
            y
        } else {
            height - 1 - y
        };
        let start = storage_y as usize * self.width as usize + x0 as usize;
        for (index, _) in (start..).zip(x0..=x1) {
            if z > self.zbuffer[index] {
                self.pixels[index] = pixel;
                self.zbuffer[index] = z;
            }
            z += dz;
        }
    }

    /// Draws a clipped, non-wrapping horizontal z-buffered span with per-pixel state.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub(crate) fn plot_z_span_clipped_with<State, Advance, Color>(
        &mut self,
        span: ZSpan,
        mut state: State,
        mut advance: Advance,
        mut color: Color,
    ) where
        Advance: FnMut(&mut State, f64),
        Color: FnMut(&State) -> Rgb,
    {
        let (mut x0, mut x1, y, mut z, dz) = (span.x0, span.x1, span.y, span.z, span.dz);
        let width = i64::from(self.width);
        let height = i64::from(self.height);
        if width == 0 || height == 0 || y < 0 || y >= height || x0 > x1 || !z.is_finite() {
            return;
        }

        if x1 < 0 || x0 >= width {
            return;
        }

        if x0 < 0 {
            let skipped = (-x0) as f64;
            z += dz * skipped;
            advance(&mut state, skipped);
            x0 = 0;
        }
        x1 = x1.min(width - 1);

        let storage_y = if self.upper_left_origin {
            y
        } else {
            height - 1 - y
        };
        let start = storage_y as usize * self.width as usize + x0 as usize;
        for (index, _) in (start..).zip(x0..=x1) {
            if z > self.zbuffer[index] {
                self.pixels[index] = color(&state);
                self.zbuffer[index] = z;
            }
            z += dz;
            advance(&mut state, 1.0);
        }
    }

    /// Returns a flat representation of all the pixels in the [Canvas].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// let pixels = image.pixels();
    /// ```
    #[must_use]
    pub fn pixels(&self) -> &[Rgb] {
        self.pixels.as_ref()
    }

    /// Returns the canvas z-buffer.
    #[must_use]
    pub fn zbuffer(&self) -> &[f64] {
        self.zbuffer.as_ref()
    }

    /// Overwrites all pixels in the canvas with the given pixel data.
    ///
    /// # Panics
    /// Panics if the length of `pixels` does not match the canvas size.
    pub fn fill_canvas(&mut self, pixels: Vec<Rgb>) {
        assert_eq!(
            pixels.len(),
            self.pixels.len(),
            "new pixel data must match canvas size"
        );
        self.pixels = pixels;
        self.clear_zbuffer();
    }

    /// Restores canvas pixels from a same-sized baseline without reallocating.
    pub(crate) fn restore_pixels(&mut self, pixels: &[Rgb]) {
        assert_eq!(
            pixels.len(),
            self.pixels.len(),
            "baseline pixel data must match canvas size"
        );
        self.pixels.copy_from_slice(pixels);
        self.clear_zbuffer();
    }

    /// Returns a new canvas with every pixel transformed by `f`.
    pub fn map_pixels<F>(&self, mut f: F) -> Self
    where
        F: FnMut(Rgb) -> Rgb,
    {
        self.with_pixels_like(self.pixels.iter().copied().map(&mut f).collect())
    }

    #[cfg(feature = "filters")]
    pub(crate) fn map_pixels_independent<F>(&self, f: F) -> Self
    where
        F: Fn(Rgb) -> Rgb + Send + Sync,
    {
        let pixels = {
            #[cfg(feature = "rayon")]
            {
                self.pixels.par_iter().copied().map(f).collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                self.pixels.iter().copied().map(f).collect()
            }
        };
        self.with_pixels_like(pixels)
    }

    /// Returns a new canvas with every pixel transformed by `f`.
    ///
    /// The callback receives `(x, y, pixel)` in storage coordinates.
    #[allow(clippy::cast_possible_truncation)]
    pub fn map_pixels_with_position<F>(&self, mut f: F) -> Self
    where
        F: FnMut(u32, u32, Rgb) -> Rgb,
    {
        let width = self.width as usize;
        let pixels = self
            .pixels
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, pixel)| f((idx % width) as u32, (idx / width) as u32, pixel))
            .collect();
        self.with_pixels_like(pixels)
    }

    #[cfg(feature = "filters")]
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn map_pixels_with_position_independent<F>(&self, f: F) -> Self
    where
        F: Fn(u32, u32, Rgb) -> Rgb + Send + Sync,
    {
        let width = self.width as usize;
        let pixels = {
            #[cfg(feature = "rayon")]
            {
                self.pixels
                    .par_iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, pixel)| f((idx % width) as u32, (idx / width) as u32, pixel))
                    .collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                self.pixels
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, pixel)| f((idx % width) as u32, (idx / width) as u32, pixel))
                    .collect()
            }
        };
        self.with_pixels_like(pixels)
    }
}

impl Canvas {
    /// Sets the color of the drawing line to a different color given three ints.
    ///
    /// # Arguments
    ///
    /// * `red` - An unsigned u8 int that represents red light
    /// * `green` - An unsigned u8 int that represents green light
    /// * `blue` - An unsigned u8 int that represents blue light
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// image.set_line_rgb_values(255, 255, 255);
    /// ```
    pub fn set_line_rgb_values(&mut self, red: u8, green: u8, blue: u8) {
        self.line = Rgb::new(red, green, blue);
    }

    /// Sets the color of the drawing line to a different color given a [Rgb] value.
    ///
    /// # Arguments
    ///
    /// * `color` - A [Rgb] value that will be the new color of the drawing line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// let color = Rgb::new(255, 255, 255);
    /// image.set_line_rgb(color);
    /// ```
    pub fn set_line_rgb(&mut self, color: Rgb) {
        self.line = color;
    }

    /// Sets the current drawing line color.
    pub fn set_line_color(&mut self, color: Rgb) {
        self.line = color;
    }

    /// Sets the color of the drawing line to a different color given a [Rgb] value.
    ///
    /// # Arguments
    ///
    /// * `color` - A [Rgb] value that will be the new color of the drawing line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// let color = Rgb::new(255, 255, 255);
    /// image.set_line_pixel(color);
    /// ```
    pub fn set_line_pixel(&mut self, color: Rgb) {
        self.line = color;
    }

    /// Clears the current [Canvas].
    ///
    /// Re-fills the canvas with its default background color (all black).
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// image.clear_canvas();
    /// ```
    pub fn clear_canvas(&mut self) {
        self.pixels.fill(Rgb::default());
        self.clear_zbuffer();
    }

    /// Resets every z-buffer entry to negative infinity.
    pub fn clear_zbuffer(&mut self) {
        self.zbuffer.fill(f64::NEG_INFINITY);
    }

    /// Returns the current drawing line color.
    #[must_use]
    pub fn line_color(&self) -> Rgb {
        self.line
    }

    /// Sets how filled polygon triangles choose colors.
    pub fn set_polygon_color_mode(&mut self, mode: PolygonColorMode) {
        self.polygon_color_mode = mode;
    }

    /// Returns how filled polygon triangles choose colors.
    #[must_use]
    pub fn polygon_color_mode(&self) -> PolygonColorMode {
        self.polygon_color_mode
    }

    /// Sets how polygon surfaces are shaded.
    pub fn set_shading_mode(&mut self, mode: ShadingMode) {
        self.shading_mode = mode;
    }

    /// Returns how polygon surfaces are shaded.
    #[must_use]
    pub fn shading_mode(&self) -> ShadingMode {
        self.shading_mode
    }

    /// Sets the Phong reflection lighting configuration.
    pub fn set_lighting(&mut self, lighting: Lighting) {
        self.lighting = lighting;
    }

    /// Returns the Phong reflection lighting configuration.
    #[must_use]
    pub fn lighting(&self) -> Lighting {
        self.lighting.clone()
    }

    /// Returns the Phong reflection lighting configuration by reference.
    pub(crate) fn lighting_ref(&self) -> &Lighting {
        &self.lighting
    }

    /// Returns the Phong reflection lighting configuration mutably.
    pub fn lighting_mut(&mut self) -> &mut Lighting {
        &mut self.lighting
    }

    /// Sets the current drawing line width.
    ///
    /// # Panics
    ///
    /// Panics if `width` is not positive and finite.
    pub fn set_line_width(&mut self, width: f64) {
        assert!(
            width.is_finite() && width > 0.0,
            "line width must be positive and finite"
        );
        self.line_width = width;
    }

    /// Returns the current drawing line width.
    #[must_use]
    pub fn line_width(&self) -> f64 {
        self.line_width
    }
}

impl Index<usize> for Canvas {
    type Output = Rgb;

    fn index(&self, index: usize) -> &Self::Output {
        &self.pixels[index]
    }
}

impl IndexMut<usize> for Canvas {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.pixels[index]
    }
}

impl Canvas {
    /// Saves the current state of an image as a binary ppm file.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    ///   Should end in ".ppm".
    ///
    /// # Errors
    /// Returns `Err` if the underlying I/O fails.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// image.save_binary("test.ppm").expect("Could not save file");
    /// ```
    pub fn save_binary(&self, file_name: &str) -> io::Result<()> {
        let file = File::create(file_name)?;
        let out = BufWriter::new(file);
        self.write_binary_ppm(out)
    }

    fn write_binary_ppm<W: Write>(&self, mut out: W) -> io::Result<()> {
        writeln!(out, "P6\n{} {}\n255", self.width, self.height)?;

        // SAFETY: Rgb is #[repr(C)] and contains three u8 fields with no padding.
        // It is safe to view a slice of Rgb as a slice of u8 for binary output.
        let bytes = unsafe {
            std::slice::from_raw_parts(self.pixels.as_ptr().cast::<u8>(), self.pixels.len() * 3)
        };
        out.write_all(bytes)?;
        Ok(())
    }

    /// Saves the current state of an image as an ascii ppm file.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    ///   Should end in ".ppm".
    ///
    /// # Errors
    /// Returns `Err` if the underlying I/O fails.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// image.save_ascii("test.ppm").expect("Could not save file");
    /// ```
    pub fn save_ascii(&self, file_name: &str) -> io::Result<()> {
        let file = File::create(file_name)?;
        let mut out = BufWriter::new(file);
        writeln!(out, "P3\n{} {}\n255", self.width, self.height)?;
        for pixel in self {
            writeln!(out, "{} {} {}", pixel.red, pixel.green, pixel.blue)?;
        }
        Ok(())
    }

    /// Saves the current state of an image as any format supported by `ImageMagick`.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    ///
    /// # Errors
    /// Returns `Err` if `magick` is not installed or the output extension is unsupported.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// image.save_extension("pics/test.png").expect("Could not save file")
    /// ```
    pub fn save_extension(&self, file_name: &str) -> io::Result<()> {
        let mut child = Command::new("magick")
            .args(["-", file_name])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Failed to execute ImageMagick `magick`: {e}. Is it installed?"),
                )
            })?;

        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::other("Failed to open stdin for ImageMagick"))?;

        self.write_binary_ppm(stdin)?;

        let status = child.wait()?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "ImageMagick `magick` failed with status {status}; check that the output extension is supported"
            )));
        }
        Ok(())
    }

    /// Display the current state of the [Canvas].
    ///
    /// # Errors
    /// Returns `Err` if `magick` is not installed.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// image.display().expect("Could not display file");
    /// ```
    pub fn display(&self) -> io::Result<()> {
        let mut child = Command::new("magick")
            .args(["display", "-"])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Failed to execute ImageMagick `magick`: {e}. Is it installed?"),
                )
            })?;

        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::other("Failed to open stdin for ImageMagick"))?;

        self.write_binary_ppm(stdin)?;

        let _ = child.wait()?;
        Ok(())
    }
}

impl<'a> IntoIterator for &'a Canvas {
    type Item = &'a Rgb;
    type IntoIter = slice::Iter<'a, Rgb>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut Canvas {
    type Item = &'a mut Rgb;
    type IntoIter = slice::IterMut<'a, Rgb>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// A builder for creating a [`Canvas`] with custom configuration.
#[derive(Debug, Clone)]
#[must_use]
pub struct CanvasBuilder {
    width: u32,
    height: u32,
    background: Rgb,
    line_color: Rgb,
    line_width: f64,
    polygon_color_mode: PolygonColorMode,
    shading_mode: ShadingMode,
    lighting: Lighting,
    upper_left_origin: bool,
    wrapped: bool,
}

impl CanvasBuilder {
    /// Creates a new builder for a canvas of the given size.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            background: Rgb::default(),
            line_color: Rgb::default(),
            line_width: 1.0,
            polygon_color_mode: PolygonColorMode::default(),
            shading_mode: ShadingMode::default(),
            lighting: Lighting::default(),
            upper_left_origin: false,
            wrapped: true,
        }
    }

    /// Sets the background color of the canvas.
    pub fn background(mut self, color: Rgb) -> Self {
        self.background = color;
        self
    }

    /// Sets the initial drawing line color.
    pub fn line_color(mut self, color: Rgb) -> Self {
        self.line_color = color;
        self
    }

    /// Sets the initial drawing line width.
    ///
    /// # Panics
    ///
    /// Panics if `width` is not positive and finite.
    pub fn line_width(mut self, width: f64) -> Self {
        assert!(
            width.is_finite() && width > 0.0,
            "line width must be positive and finite"
        );
        self.line_width = width;
        self
    }

    /// Sets how filled polygon triangles choose colors.
    pub fn polygon_color_mode(mut self, mode: PolygonColorMode) -> Self {
        self.polygon_color_mode = mode;
        self
    }

    /// Sets how polygon surfaces are shaded.
    pub fn shading_mode(mut self, mode: ShadingMode) -> Self {
        self.shading_mode = mode;
        self
    }

    /// Sets the Phong reflection lighting configuration.
    pub fn lighting(mut self, lighting: Lighting) -> Self {
        self.lighting = lighting;
        self
    }

    /// Sets whether the origin is at the top-left (true) or bottom-left (false).
    pub fn upper_left_origin(mut self, upper_left: bool) -> Self {
        self.upper_left_origin = upper_left;
        self
    }

    /// Sets whether coordinates wrap around the canvas edges.
    pub fn wrapped(mut self, wrapped: bool) -> Self {
        self.wrapped = wrapped;
        self
    }

    /// Consumes the builder and returns a new [Canvas].
    ///
    /// # Panics
    ///
    /// Panics if `width * height` cannot be represented or if allocation fails.
    pub fn build(self) -> Canvas {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }

    /// Consumes the builder and returns a checked new [`Canvas`].
    ///
    /// # Errors
    ///
    /// Returns an error if `width * height` cannot be represented or if allocation fails.
    pub fn try_build(self) -> Result<Canvas, CanvasBuildError> {
        let pixel_count = Canvas::try_pixel_count(self.width, self.height)?;
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(pixel_count)
            .map_err(|_| CanvasBuildError::AllocationFailed)?;
        pixels.resize(pixel_count, self.background);
        let zbuffer = Canvas::zbuffer_for_pixel_count(pixel_count)?;
        Ok(Canvas {
            width: self.width,
            height: self.height,
            pixels,
            zbuffer,
            upper_left_origin: self.upper_left_origin,
            wrapped: self.wrapped,
            line: self.line_color,
            line_width: self.line_width,
            polygon_color_mode: self.polygon_color_mode,
            shading_mode: self.shading_mode,
            lighting: self.lighting,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_builds_canvas_with_checked_dimensions() {
        let canvas = Canvas::try_new_with_bg(2, 3, Rgb::new(1, 2, 3)).expect("valid canvas");

        assert_eq!(canvas.width(), 2);
        assert_eq!(canvas.height(), 3);
        assert_eq!(canvas.pixels(), &[Rgb::new(1, 2, 3); 6]);
    }

    #[test]
    fn zero_size_canvas_accessors_are_empty_and_non_panicking() {
        let mut canvas = Canvas::new(0, 0, Rgb::WHITE);

        assert_eq!(canvas.width(), 0);
        assert_eq!(canvas.height(), 0);
        assert_eq!(canvas.len(), 0);
        assert!(canvas.is_empty());
        assert_eq!(canvas.get_pixel(0, 0), None);
        assert_eq!(canvas.get_pixel(-1, -1), None);
        assert_eq!(canvas.get_zbuffer(0, 0), None);
        assert_eq!(canvas.iter_row().count(), 0);
        assert_eq!(canvas.iter_row_mut().count(), 0);

        canvas.plot(&Rgb::RED, 0, 0);
        assert!(canvas.pixels().is_empty());
    }

    #[test]
    fn zero_width_canvas_rows_and_domain_mapping_are_empty() {
        let mut calls = 0;
        let canvas = Canvas::from_domain(0, 3, Domain2D::new(-1.0, 1.0, -1.0, 1.0), |_, _| {
            calls += 1;
            Rgb::RED
        });

        assert_eq!(canvas.width(), 0);
        assert_eq!(canvas.height(), 3);
        assert_eq!(canvas.len(), 0);
        assert_eq!(calls, 0);
        assert_eq!(canvas.iter_row().count(), 0);
    }

    #[test]
    fn zero_height_domain_mapping_is_empty() {
        let mut calls = 0;
        let canvas = Canvas::from_domain(3, 0, Domain2D::new(-1.0, 1.0, -1.0, 1.0), |_, _| {
            calls += 1;
            Rgb::RED
        });

        assert_eq!(canvas.width(), 3);
        assert_eq!(canvas.height(), 0);
        assert_eq!(canvas.len(), 0);
        assert_eq!(calls, 0);
        assert_eq!(canvas.iter_row().count(), 0);
    }

    #[test]
    fn huge_canvas_dimensions_return_checked_error() {
        let error = Canvas::builder(u32::MAX, u32::MAX)
            .try_build()
            .expect_err("huge canvas should not allocate");

        assert!(matches!(
            error,
            CanvasBuildError::AllocationFailed | CanvasBuildError::TooLargeForPlatform
        ));
    }

    #[test]
    fn canvas_state_accessors_update_origin_wrapping_and_line_color() {
        let mut canvas = Canvas::new(2, 2, Rgb::BLACK);

        canvas.set_upper_left_origin(true);
        canvas.set_wrapped(false);
        canvas.set_line_color(Rgb::new(1, 2, 3));

        assert!(canvas.upper_left_origin());
        assert!(!canvas.wrapped());
        assert_eq!(canvas.line_color(), Rgb::new(1, 2, 3));
    }

    #[test]
    #[should_panic(expected = "line width must be positive and finite")]
    fn canvas_rejects_non_finite_line_width() {
        let mut canvas = Canvas::new(2, 2, Rgb::BLACK);

        canvas.set_line_width(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "line width must be positive and finite")]
    fn canvas_builder_rejects_negative_line_width() {
        let _ = Canvas::builder(2, 2).line_width(-1.0);
    }

    #[test]
    fn try_from_pixels_rejects_wrong_pixel_count() {
        let err = Canvas::try_from_pixels(2, 2, vec![Rgb::BLACK; 3]).expect_err("wrong length");

        assert_eq!(
            err,
            CanvasBuildError::PixelDataLengthMismatch {
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn rgb_image_is_lightweight_pixel_storage_convertible_to_canvas() {
        let image = RgbImage::try_new(2, 2, Rgb::BLUE).expect("valid image");
        let canvas = image.clone().into_canvas();
        let round_trip = RgbImage::from(canvas);

        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.pixels(), &[Rgb::BLUE; 4]);
        assert_eq!(round_trip, image);
        assert_eq!(
            RgbImage::try_from_pixels(2, 2, vec![Rgb::BLACK; 3]),
            Err(CanvasBuildError::PixelDataLengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn from_fn_visits_pixels_in_storage_order() {
        let canvas = Canvas::from_fn(2, 2, |x, y| match (x, y) {
            (0, 0) => Rgb::new(0, 0, 0),
            (1, 0) => Rgb::new(1, 0, 0),
            (0, 1) => Rgb::new(0, 1, 0),
            (1, 1) => Rgb::new(1, 1, 0),
            _ => unreachable!("test canvas is 2x2"),
        });

        assert_eq!(
            canvas.pixels(),
            &[
                Rgb::new(0, 0, 0),
                Rgb::new(1, 0, 0),
                Rgb::new(0, 1, 0),
                Rgb::new(1, 1, 0),
            ]
        );
    }

    #[test]
    fn from_domain_maps_top_row_to_y_max() {
        let domain = Domain2D::new(-1.0, 1.0, -1.0, 1.0);
        let canvas = Canvas::from_domain(2, 4, domain, |x, y| match (x >= 0.0, y >= 0.0) {
            (false, true) => Rgb::new(0, 200, 0),
            (true, true) => Rgb::new(100, 200, 0),
            (false, false) => Rgb::new(0, 100, 0),
            (true, false) => Rgb::new(100, 100, 0),
        });

        assert_eq!(canvas.pixels()[0], Rgb::new(0, 200, 0));
        assert_eq!(canvas.pixels()[7], Rgb::new(100, 100, 0));
    }

    #[test]
    fn downsample_averages_pixel_blocks() {
        let canvas = Canvas::from_pixels(
            2,
            2,
            vec![
                Rgb::new(0, 0, 0),
                Rgb::new(10, 20, 30),
                Rgb::new(20, 40, 60),
                Rgb::new(30, 60, 90),
            ],
        );

        let downsampled = canvas.downsample(2);

        assert_eq!(downsampled.width(), 1);
        assert_eq!(downsampled.height(), 1);
        assert_eq!(downsampled.pixels(), &[Rgb::new(15, 30, 45)]);
    }
}
