use crate::graphics::colors::Rgb;

use core::{fmt, slice};
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    ops::{Index, IndexMut},
    process::{Command, Stdio},
};

#[derive(Clone, Debug)]
/// An art [Canvas] / computer screen is represented here.
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
    #[must_use] 
    pub fn new(width: u32, height: u32, line_color: Rgb) -> Self {
        let pixels: Vec<Rgb> = vec![Rgb::default(); Self::pixel_count(width, height)];
        Self {
            height,
            width,
            pixels,
            upper_left_origin: false,
            wrapped: true,
            line: line_color,
            line_width: 1.0,
        }
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
    #[must_use] 
    pub fn new_with_bg(width: u32, height: u32, background_color: Rgb) -> Self {
        let line = Rgb::default();
        Self {
            height,
            width,
            pixels: vec![background_color; Self::pixel_count(width, height)],
            upper_left_origin: false,
            wrapped: true,
            line,
            line_width: 1.0,
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
        Self::pixel_count(self.width, self.height)
    }

    /// Returns if [Canvas] is empty
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
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Sets the color of the drawing line to a different color given a [Pixel].
    ///
    /// # Arguments
    ///
    /// * `new_color` - A `pixel` that represent the new drawing color
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// let new_color = Rgb::new(12, 20, 30);
    /// image.set_line_pixel(new_color);
    /// ```
    pub fn set_line_pixel(&mut self, new_color: Rgb) {
        self.line = new_color;
    }

    /// Fills in an empty canvas
    ///
    /// # Arguments
    ///
    /// * `new_pixels` - A vector of pixels that represents new data
    ///   to append to an empty canvas
    ///
    /// # Errors
    /// Returns an error if `new_pixels` does not exactly fill the canvas.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(1, 1, Rgb::default());
    /// let data = vec![Rgb::default()];
    /// image.fill_canvas(data).expect("data should match canvas size")
    /// ```
    pub fn fill_canvas(&mut self, new_pixels: Vec<Rgb>) -> Result<(), &'static str> {
        if new_pixels.len() != self.len() {
            return Err("new data must exactly fill canvas");
        }
        self.pixels = new_pixels;
        Ok(())
    }

    /// Sets an (X, Y) pair to the proper spot in Pixels
    #[allow(clippy::cast_possible_truncation)]
    pub(in crate::graphics) fn index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    /// Returns an iterator over the canvas pixels.
    pub fn iter(&self) -> impl Iterator<Item = &Rgb> + '_ {
        self.pixels.iter()
    }

    /// Returns a mutable iterator on pixels
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Rgb> + '_ {
        self.pixels.iter_mut()
    }

    /// Returns a iterator that iterates over a specific row.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(1, 1, Rgb::default());
    /// let iter = image.iter_row();
    /// ```
    #[allow(clippy::cast_possible_truncation)]
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
    /// let mut image = Canvas::new(1, 1, Rgb::default());
    /// let iter = image.iter_row_mut();
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn iter_row_mut(&mut self) -> slice::ChunksExactMut<'_, Rgb> {
        self.pixels.chunks_exact_mut(self.width as usize)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn normalize_coords(&self, x: i64, y: i64) -> Option<(u32, u32)> {
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

    /// Returns a reference to the (x, y) [Pixel] of body of in the
    /// [Canvas].
    ///
    /// # Arguments
    ///
    /// * `x` - A signed i32 int that represents the x coordinate that will access the "body"
    /// * `y` - A signed i32 int that represents the y coordinate that will access the "body"
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// let color = image.get_pixel(250, 250);
    /// ```
    #[must_use] 
    pub fn get_pixel(&self, x: i64, y: i64) -> Option<&Rgb> {
        let (x, y) = self.normalize_coords(x, y)?;
        Some(&self.pixels[self.index(x, y)])
    }

    /// Returns a reference to the pixel at `(x, y)`, panicking if out of bounds.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate is out of bounds and wrapping is disabled.
    #[must_use]
    pub fn pixel(&self, x: i64, y: i64) -> &Rgb {
        self.get_pixel(x, y)
            .expect("pixel coordinate out of bounds")
    }

    /// Plots `new_color` to the (X, Y) coordinate pair corresponding to the [Canvas] body.
    ///
    /// # Arguments
    ///
    /// * `new_color` - The [Pixel] that will be "drawn" in the [Canvas] "body"
    /// * `x` - A signed i32 int that represents the x coordinate that will access the "body"
    /// * `y` - A signed i32 int that represents the y coordinate that will access the "body"
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// let color = Rgb::new(1, 1, 1);
    /// image.plot(&color, 100, 100);
    /// ```
    pub fn plot(&mut self, new_color: &Rgb, x: i64, y: i64) {
        if let Some((x, y)) = self.normalize_coords(x, y) {
            let index = self.index(x, y);
            self.pixels[index] = *new_color;
        }
    }

    /// Clears the [Canvas] to `Pixel::default`()
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let background_color = Rgb::new(1, 2, 3);
    /// let mut image = Canvas::new_with_bg(500, 500, background_color);
    /// image.clear_canvas()
    /// ```
    pub fn clear_canvas(&mut self) {
        self.pixels.fill(Rgb::default());
    }

    /// Fills the entire [Canvas] with one [Pixel]
    ///
    /// # Arguments
    ///
    /// * `background_color` - A [Pixel] that will represent the color of the background color that will fill the [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let background_color = Rgb::new(1, 2, 3);
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// image.fill_color(&background_color)
    /// ```
    pub fn fill_color(&mut self, bg: &Rgb) {
        self.pixels.fill(*bg);
    }

    /// Get a reference to the canvas's pixels.
    #[must_use] 
    pub fn pixels(&self) -> &[Rgb] {
        self.pixels.as_ref()
    }
}

impl Canvas {
    /// Sets the color of the drawing line to a different color given three ints.
    ///
    /// # Arguments
    ///
    /// * `red` - An unsigned u8 int that represents the new red of the drawing color
    /// * `green` - An unsigned u8 int that represents the new green of the drawing color
    /// * `blue` - An unsigned u8 int that represents blue of the drawing color
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// image.set_line_color_rgb(55, 95, 100);
    /// ```
    pub fn set_line_color_rgb(&mut self, red: u8, green: u8, blue: u8) {
        self.line = Rgb::new(red, green, blue);
    }

    /// Sets the color of the drawing line to a different color given an RGB Pixel
    ///
    /// # Arguments
    ///
    /// * `Pixel` - A rgb pixel
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let mut image = Canvas::new(500, 500, Rgb::default());
    /// image.set_line_rgb(Rgb::default());
    /// ```
    pub fn set_line_rgb(&mut self, color: Rgb) {
        self.line = color;
    }

    /// Returns the current line width in pixels.
    #[must_use]
    pub fn line_width(&self) -> f64 {
        self.line_width
    }

    /// Sets the line width in pixels.
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
}

impl IntoIterator for Canvas {
    type Item = Rgb;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.pixels.into_iter()
    }
}

impl Index<usize> for Canvas {
    type Output = Rgb;
    fn index(&self, index: usize) -> &Self::Output {
        &self.pixels[index]
    }
}

impl IndexMut<usize> for Canvas {
    fn index_mut(&mut self, index: usize) -> &mut Rgb {
        &mut self.pixels[index]
    }
}

impl fmt::Display for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "P3\n{} {}\n255", self.width, self.height)?;
        for pixel in self.iter() {
            writeln!(f, "{} {} {}", pixel.red, pixel.green, pixel.blue)?;
        }
        Ok(())
    }
}

// saving
impl Canvas {
    fn write_ascii_ppm<W: Write>(&self, mut out: W) -> io::Result<()> {
        writeln!(out, "P3\n{} {}\n255", self.width, self.height)?;
        for pixel in self.iter() {
            writeln!(out, "{} {} {}", pixel.red, pixel.green, pixel.blue)?;
        }
        Ok(())
    }

    fn write_binary_ppm<W: Write>(&self, mut out: W) -> io::Result<()> {
        writeln!(out, "P6\n{} {}\n255", self.width, self.height)?;
        for pixel in self.iter() {
            out.write_all(&pixel.to_be_bytes())?;
        }
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
    /// image.save_ascii("pics/test.ppm").expect("Could not save file")
    /// ```
    pub fn save_ascii(&self, file_name: &str) -> io::Result<()> {
        let file = File::create(file_name)?;
        self.write_ascii_ppm(BufWriter::new(file))
    }

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
    /// image.save_binary("pics/test.ppm").expect("Could not save file")
    /// ```
    pub fn save_binary(&self, file_name: &str) -> io::Result<()> {
        let file = File::create(file_name)?;
        self.write_binary_ppm(BufWriter::new(file))
    }

    /// Saves the current state of an image to any extension.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
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
    /// image.save_extension("pics/test.png").expect("Could not save file")
    /// ```
    pub fn save_extension(&self, file_name: &str) -> io::Result<()> {
        let mut child = Command::new("magick")
            .arg("-")
            .arg(file_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|err| {
                io::Error::new(
                    err.kind(),
                    format!("failed to run ImageMagick `magick`; is ImageMagick installed and in PATH? {err}"),
                )
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "failed to open magick stdin")
        })?;
        self.write_binary_ppm(BufWriter::new(stdin))?;

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
    /// Returns `Err` if the underlying I/O fails.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::prelude::{Canvas, Rgb};
    /// let image = Canvas::new(500, 500, Rgb::default());
    /// image.display().expect("Could not display image")
    /// ```
    pub fn display(&self) -> io::Result<()> {
        let mut child = Command::new("magick")
            .arg("display")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|err| {
                io::Error::new(
                    err.kind(),
                    format!("failed to run ImageMagick `magick display`; is ImageMagick installed and in PATH? {err}"),
                )
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "failed to open display stdin")
        })?;
        self.write_binary_ppm(BufWriter::new(stdin))?;

        let status = child.wait()?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "ImageMagick `display` failed with status {status}; check that ImageMagick display is available"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_canvas_has_full_pixel_buffer() {
        let canvas = Canvas::new(3, 2, Rgb::WHITE);
        assert_eq!(canvas.len(), 6);
        assert_eq!(canvas.pixels().len(), 6);
        assert!(!canvas.is_empty());
    }

    #[test]
    fn fill_canvas_replaces_existing_pixels() {
        let mut canvas = Canvas::new(2, 2, Rgb::WHITE);
        canvas
            .fill_canvas(vec![Rgb::BLACK; 4])
            .expect("pixel data should match canvas size");
        canvas
            .fill_canvas(vec![Rgb::WHITE; 4])
            .expect("pixel data should match canvas size");

        assert_eq!(canvas.pixels().len(), 4);
        assert!(canvas.pixels().iter().all(|pixel| *pixel == Rgb::WHITE));
    }

    #[test]
    fn fill_canvas_rejects_wrong_size() {
        let mut canvas = Canvas::new(2, 2, Rgb::WHITE);
        assert_eq!(
            canvas.fill_canvas(vec![Rgb::BLACK; 3]),
            Err("new data must exactly fill canvas")
        );
    }

    #[test]
    fn wrapped_plot_wraps_width_and_height_edges() {
        let mut canvas = Canvas::new(2, 2, Rgb::WHITE);
        canvas.upper_left_origin = true;
        canvas.plot(&Rgb::BLACK, 2, 2);

        assert_eq!(canvas.get_pixel(0, 0), Some(&Rgb::BLACK));
    }

    #[test]
    fn unclipped_get_pixel_returns_none_out_of_bounds() {
        let mut canvas = Canvas::new(2, 2, Rgb::WHITE);
        canvas.wrapped = false;

        assert_eq!(canvas.get_pixel(-1, 0), None);
        assert_eq!(canvas.get_pixel(2, 0), None);
        assert_eq!(canvas.get_pixel(0, 2), None);
    }

    #[test]
    fn bottom_left_origin_maps_y_to_internal_row() {
        let mut canvas = Canvas::new(2, 2, Rgb::WHITE);
        canvas.wrapped = false;
        canvas.plot(&Rgb::BLACK, 0, 0);

        assert_eq!(canvas.pixels()[canvas.index(0, 1)], Rgb::BLACK);
        assert_eq!(canvas.get_pixel(0, 0), Some(&Rgb::BLACK));
    }

    #[test]
    fn set_line_width_accepts_positive_finite_values() {
        let mut canvas = Canvas::new(2, 2, Rgb::WHITE);
        canvas.set_line_width(2.5);
        assert!((canvas.line_width() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "line width must be positive and finite")]
    fn set_line_width_rejects_invalid_values() {
        let mut canvas = Canvas::new(2, 2, Rgb::WHITE);
        canvas.set_line_width(f64::NAN);
    }
}
