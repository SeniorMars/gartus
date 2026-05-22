use crate::graphics::colors::Rgb;

use core::slice;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    ops::{Index, IndexMut},
    process::{Command, Stdio},
};

#[derive(Clone, Debug)]
/// An art [Canvas] / computer screen is represented here.
#[must_use]
pub struct Canvas {
    width: u32,
    height: u32,
    // always 255 — field kept for internal use only
    pixels: Vec<Rgb>,
    /// When true, (0,0) is top-left. When false (default), (0,0) is bottom-left.
    pub upper_left_origin: bool,
    /// When true (default), coordinates wrap around canvas edges. When false, out-of-bounds plots are clipped.
    pub wrapped: bool,
    /// A `PixelColor` that represents the color that will be used to draw lines.
    pub line: Rgb,
    /// Width of drawn lines in pixels. Default 1.0.
    line_width: f64,
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixels: Vec::new(),
            upper_left_origin: false,
            wrapped: true,
            line: Rgb::default(),
            line_width: 1.0,
        }
    }
}

impl Canvas {
    /// Returns a [`CanvasBuilder`] to configure a new [Canvas].
    pub fn builder(width: u32, height: u32) -> CanvasBuilder {
        CanvasBuilder::new(width, height)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn pixel_count(width: u32, height: u32) -> usize {
        width as usize * height as usize
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
        Self::builder(width, height)
            .line_color(line_color)
            .build()
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
        self.pixels.chunks_exact(self.width as usize)
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
        self.pixels.chunks_exact_mut(self.width as usize)
    }

    /// Returns a reference to the (x, y) [Pixel] of body of in the
    /// [Canvas].
    ///
    /// Returns `None` if out of bounds.
    #[must_use]
    pub fn get_pixel(&self, x: i64, y: i64) -> Option<&Rgb> {
        let (x, y) = self.normalize_coords(x, y)?;
        Some(&self.pixels[y as usize * self.width as usize + x as usize])
    }

    /// Returns a mutable reference to the (x, y) [Pixel] of body of in the
    /// [Canvas].
    ///
    /// Returns `None` if out of bounds.
    pub fn get_pixel_mut(&mut self, x: i64, y: i64) -> Option<&mut Rgb> {
        let (x, y) = self.normalize_coords(x, y)?;
        let width = self.width;
        Some(&mut self.pixels[y as usize * width as usize + x as usize])
    }

    /// Maps external coordinates (potentially negative or out-of-bounds) to
    /// internal canvas coordinates based on wrapping and origin settings.
    ///
    /// Returns `None` if the point is clipped.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[must_use]
    pub fn normalize_coords(&self, x: i64, y: i64) -> Option<(u32, u32)> {
        let (width, height) = (i64::from(self.width), i64::from(self.height));

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
    /// * `pixel` - A [Pixel] that will be plotted at (x, y)
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
        if let Some(target) = self.get_pixel_mut(x, y) {
            *target = *pixel;
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
    }

    /// Returns the current drawing line color.
    #[must_use]
    pub fn line_color(&self) -> Rgb {
        self.line
    }

    /// Sets the current drawing line width.
    pub fn set_line_width(&mut self, width: f64) {
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
            std::slice::from_raw_parts(
                self.pixels.as_ptr().cast::<u8>(),
                self.pixels.len() * 3,
            )
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

        let stdin = child.stdin.as_mut().ok_or_else(|| {
            io::Error::other("Failed to open stdin for ImageMagick")
        })?;

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
            .args(["-", "display:"])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Failed to execute ImageMagick `magick`: {e}. Is it installed?"),
                )
            })?;

        let stdin = child.stdin.as_mut().ok_or_else(|| {
            io::Error::other("Failed to open stdin for ImageMagick")
        })?;

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
    pub fn line_width(mut self, width: f64) -> Self {
        self.line_width = width;
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
    pub fn build(self) -> Canvas {
        let pixels = vec![self.background; Canvas::pixel_count(self.width, self.height)];
        Canvas {
            width: self.width,
            height: self.height,
            pixels,
            upper_left_origin: self.upper_left_origin,
            wrapped: self.wrapped,
            line: self.line_color,
            line_width: self.line_width,
        }
    }
}
