use crate::graphics::colors::{ColorSpace, Hsl, Rgb};
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    ops::{Index, IndexMut},
    process::{Command, Stdio},
};

#[derive(Default, Debug, Clone)]
/// An art [Canvas] / computer screen is represented here.
pub struct Canvas<C: ColorSpace>
where
    Rgb: From<C>,
{
    /// The height of the canvas
    height: u32,
    /// The width of the canvas
    width: u32,
    /// The maximum depth of the canvas
    range: u8,
    /// The "body" of the canvas that holds all the [Pixel]s that will be displayed
    pixels: Vec<C>,
    /// A counter that will be used when saving images for animations
    pub(in crate::graphics) anim_index: u32,
    /// A boolean that will determine where "(0, 0)" - the start of the canvas - is located
    pub upper_left_system: bool,
    /// A [PixelColor] that represents the color that will be used to draw lines.
    pub line: C,
}

#[allow(dead_code)]
impl<C: ColorSpace> Canvas<C>
where
    Rgb: From<C>,
{
    /// Returns a new blank [Canvas] to be drawn on.
    ///
    /// # Arguments
    ///
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `range` - An unsigned int that will represent maximum depth of colors in the [Canvas]
    /// * `line_color` - An RGB or HSL value that will also represent the default color for the
    /// drawing line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// ```
    pub fn new(height: u32, width: u32, range: u8, line_color: C) -> Self {
        let pixels: Vec<C> = vec![C::default(); (height * width) as usize];
        Self {
            height,
            width,
            range,
            pixels,
            anim_index: 0,
            upper_left_system: false,
            line: line_color,
        }
    }

    /// Returns a new [Canvas] to be drawn on with a specific background color.
    ///
    /// # Arguments
    ///
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `range` - An unsigned int that will represent maximum depth
    /// of colors in the [Canvas]
    /// * `background_color` - A RGB or HSL value that will represent the color of the
    /// background color that will fill the [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let background_color = Pixel::RGB(RGB::new(1, 2, 3));
    /// let image = Canvas::new_with_bg(500, 500, 255, background_color);
    /// ```
    pub fn new_with_bg(height: u32, width: u32, range: u8, background_color: C) -> Self {
        let line = C::default();
        Self {
            height,
            width,
            range,
            pixels: vec![background_color; (height * width) as usize],
            anim_index: 0,
            upper_left_system: false,
            line,
        }
    }

    /// Returns a blank [Canvas] that can be filled
    ///
    ///
    /// # Arguments
    ///
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `range` - An unsigned int that will represent maximum depth
    /// of colors in the [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::with_capacity(500, 500, 255, Pixel::RGB(RGB::default()));
    /// ```
    pub fn with_capacity(height: u32, width: u32, range: u8, line_color: C) -> Self {
        let line = line_color;
        Self {
            height,
            width,
            range,
            pixels: Vec::with_capacity((height * width) as usize),
            anim_index: 0,
            upper_left_system: false,
            line,
        }
    }

    /// Returns the width of a [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// let width = image.width();
    /// ```
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height of a [Canvas].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// let height = image.height();
    /// ```
    pub fn height(&self) -> u32 {
        self.height
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
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let mut image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// let new_color = Pixel::RGB(RGB::new(12, 20, 30));
    /// image.set_line_pixel(&new_color);
    /// ```
    pub fn set_line_pixel(&mut self, new_color: &C) {
        self.line = *new_color
    }

    /// Fills in an empty canvas
    ///
    /// # Arguments
    ///
    /// * `new_pixels` - A vector of pixels that represents new data
    /// to append to an empty canvas
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let mut image = Canvas::with_capacity(1, 1, 255, Pixel::RGB(RGB::default()));
    /// let mut data = vec![Pixel::RGB(RGB::default())];
    /// image.fill_canvas(data)
    /// ```
    pub fn fill_canvas(&mut self, mut new_pixels: Vec<C>) {
        assert!(
            new_pixels.len() == (self.width * self.height) as usize,
            "New data must fill canvas"
        );
        self.pixels.append(&mut new_pixels)
    }

    /// Sets an (X, Y) pair to the proper spot in Pixels
    fn index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    fn iter(&self) -> impl Iterator<Item = &C> + '_ {
        self.pixels.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut C> + '_ {
        self.pixels.iter_mut()
    }

    /// Deals with negative numbers by wrapping the [Canvas]
    fn deal_with_negs(&self, x: i32, y: i32) -> (i32, i32, i32, i32) {
        let (width, height) = (self.width as i32, self.height as i32);
        let x = if x > width {
            x % width
        } else if x < 0 {
            let r = x % width;
            if r != 0 {
                r + width
            } else {
                r
            }
        } else {
            x
        };

        let y = if y > height {
            y % height
        } else if y < 0 {
            let r = y % height;
            if r != 0 {
                r + height
            } else {
                r
            }
        } else {
            y
        };
        (x, y, width, height)
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
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// let color = image.get_pixel(250, 250);
    /// ```
    pub fn get_pixel(&self, x: i32, y: i32) -> &C {
        let (x, y, width, height) = self.deal_with_negs(x, y);
        // println!("i32:{} as {}", x, x as u32);
        if self.upper_left_system {
            let index = self.index(x as u32, y as u32);
            &self.pixels[index]
        } else {
            let new_y = height - 1 - y;
            if x >= 0 && x < width && new_y >= 0 && new_y < height {
                let index = self.index(x as u32, new_y as u32);
                &self.pixels[index]
            } else {
                panic!("Wrong input and reference can not be retrieved")
            }
        }
    }

    /// Plots new_color to the (X, Y) coordinate pair corresponding to the [Canvas] body.
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
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let mut image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// let color = Pixel::RGB(RGB::new(1, 1, 1));
    /// image.plot(&color, 100, 100);
    /// ```
    pub fn plot(&mut self, new_color: &C, x: i32, y: i32) {
        let (x, y, width, height) = self.deal_with_negs(x, y);
        if self.upper_left_system {
            let index = self.index(x as u32, y as u32);
            self.pixels[index] = *new_color
        } else {
            let new_y = height - 1 - y;
            if x >= 0 && x < width && new_y >= 0 && new_y < height {
                let index = self.index(x as u32, new_y as u32);
                self.pixels[index] = *new_color
            }
        }
    }

    /// Clears the [Canvas] to Pixel::default()
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let background_color = Pixel::RGB(RGB::new(1, 2, 3));
    /// let mut image = Canvas::new_with_bg(500, 500, 255, background_color);
    /// image.clear_canvas()
    /// ```
    pub fn clear_canvas(&mut self) {
        self.iter_mut().for_each(|i| *i = C::default())
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
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let background_color = Pixel::RGB(RGB::new(1, 2, 3));
    /// let mut image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// image.fill_color(&background_color)
    /// ```
    pub fn fill_color(&mut self, bg: &C) {
        self.iter_mut().for_each(|i| *i = *bg);
    }
}

impl Canvas<Hsl> {
    /// Sets the color of the drawing line to a different color given three ints.
    ///
    /// # Arguments
    ///
    /// * `hue` - A u16 that represents hue -- should be a number from [0, 360)
    /// * `saturation` - A u8 that represents saturation percentage -- should be a number from [0, 100]
    /// * `light` - A u8 that represent light percentage -- should be a number from [0, 100]
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, HSL};
    /// let mut image = Canvas::new(500, 500, 255, Pixel::HSL(HSL::default()));
    /// image.set_line_color_hsl(55, 95, 100);
    /// ```
    pub fn set_line_color_hsl(&mut self, hue: u16, saturation: u16, light: u16) {
        self.line = Hsl::new(hue, saturation, light);
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
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let mut image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// image.set_line_rgb(RGB::default());
    /// ```
    pub fn set_line_hsl(&mut self, color: Hsl) {
        self.line = color
    }
}

impl Canvas<Rgb> {
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
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let mut image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// image.set_line_color_rgb(55, 95, 100);
    /// ```
    pub fn set_line_color_rgb(&mut self, red: u16, green: u16, blue: u16) {
        self.line = Rgb::new(red, green, blue)
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
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let mut image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// image.set_line_rgb(RGB::default());
    /// ```
    pub fn set_line_rgb(&mut self, color: Rgb) {
        self.line = color
    }
}

impl<C: ColorSpace> IntoIterator for Canvas<C>
where
    Rgb: From<C>,
{
    type Item = C;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.pixels.into_iter()
    }
}

impl<C: ColorSpace> Index<usize> for Canvas<C>
where
    Rgb: From<C>,
{
    type Output = C;
    fn index(&self, index: usize) -> &Self::Output {
        &self.pixels[index]
    }
}

impl<C: ColorSpace> IndexMut<usize> for Canvas<C>
where
    Rgb: From<C>,
{
    fn index_mut(&mut self, index: usize) -> &mut C {
        &mut self.pixels[index]
    }
}

// saving
#[allow(dead_code)]
impl<C: ColorSpace> Canvas<C>
where
    Rgb: From<C>,
{
    /// Saves the current state of an image as an ascii ppm file.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    /// Should end in ".ppm".
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// image.save_binary("pics/test.ppm").expect("Could not save file")
    /// ```
    pub fn save_ascii(&self, file_name: &str) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(file_name)?);
        writeln!(
            &mut file,
            "P3 {} {} {}",
            self.height, self.width, self.range
        )?;

        self.iter().for_each(|pixel| {
            let rgb = Rgb::from(*pixel);
            write!(file, "{} {} {} ", rgb.red, rgb.green, rgb.blue)
                .expect("File should always be written to");
        });
        Ok(())
    }

    /// Saves the current state of an image as a binary ppm file.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    /// Should end in ".ppm".
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// image.save_binary("pics/test.ppm").expect("Could not save file")
    /// ```
    pub fn save_binary(&self, file_name: &str) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(file_name)?);

        writeln!(
            &mut file,
            "P6 {} {} {}",
            self.height, self.width, self.range
        )?;

        self.iter().for_each(|pixel| {
            let rgb = Rgb::from(*pixel);
            file.write_all(&rgb.red.to_be_bytes())
                .expect("Could not write as binary");
            file.write_all(&rgb.green.to_be_bytes())
                .expect("Could not write as binary");
            file.write_all(&rgb.blue.to_be_bytes())
                .expect("Could not write as binary");
        });

        Ok(())
    }

    /// Saves the current state of an image to any extension.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// image.save_extension("pics/test.png").expect("Could not save file")
    /// ```
    pub fn save_extension(&self, file_name: &str) -> io::Result<()> {
        let mut child = Command::new("convert")
            .arg("-")
            .arg(file_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let mut stdin = BufWriter::new(child.stdin.as_mut().unwrap());
        writeln!(stdin, "P3 {} {} {}", self.height, self.width, self.range)?;

        self.iter().for_each(|pixel| {
            let rgb = Rgb::from(*pixel);
            write!(stdin, "{} {} {} ", rgb.red, rgb.green, rgb.blue)
                .expect("Cannot write to stdin");
        });
        stdin.flush()
    }

    /// Display the current state of the [Canvas].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::{Pixel, RGB};
    /// let image = Canvas::new(500, 500, 255, Pixel::RGB(RGB::default()));
    /// image.display().expect("Could not display image")
    /// ```
    pub fn display(&self) -> io::Result<()> {
        // let command = if cfg!(target_os = "linux") {
        //     "display"
        // } else if cfg!(target_os = "windows") {
        //     "windows"
        // } else {
        //     "display"
        // };
        let mut child = Command::new("display")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let mut stdin = BufWriter::new(child.stdin.as_mut().unwrap());
        writeln!(stdin, "P3 {} {} {}", self.height, self.width, self.range)?;
        self.iter().for_each(|pixel| {
            let rgb = Rgb::from(*pixel);
            write!(stdin, "{} {} {} ", rgb.red, rgb.green, rgb.blue)
                .expect("Cannot write to stdin");
        });
        stdin.flush()
    }
}
