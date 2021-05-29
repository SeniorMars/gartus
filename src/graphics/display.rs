use std::{
    fs::File,
    io::{self, BufWriter, Write},
    ops::{Index, IndexMut},
    process::{Command, Stdio},
};

#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A computer pixel struct is represented by its red, green, blue values
pub struct Pixel {
    /// The first byte that represents red light
    pub red: u8,
    /// The second byte that represents green light
    pub green: u8,
    /// The final byte that represents blue light
    pub blue: u8,
}

#[allow(dead_code)]
impl Pixel {
    /// Returns a pixel that will be used in [Canvas]
    ///
    /// # Arguments
    ///
    /// * `red` - An unsigned u8 int that represents red
    /// * `green` - An unsigned u8 int that represents green
    /// * `blue` - An unsigned u8 int that represents blue
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Pixel;
    /// let color = Pixel::new(0, 64, 255);
    /// ```
    pub fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
/// A convention that represents a Pixel based on hue, saturation, and light
pub struct HSL {
    /// Hue
    pub hue: f64,
    /// saturation
    pub saturation: f64,
    /// light
    pub light: f64,
}

#[allow(dead_code)]
impl HSL {
    /// Returns a HSL that can be used in [Canvas]
    ///
    /// # Arguments
    ///
    /// * `hue` - A f64 that represents hue
    /// * `saturation` - A f64 that represents
    /// * `light` - A f64 that represent light
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::HSL;
    /// let color = HSL::new(0.0, 0.0, 0.0);
    /// ```
    pub fn new(hue: f64, saturation: f64, light: f64) -> Self {
        Self {
            hue,
            saturation,
            light,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
enum Color {
    HSL(HSL),
    Pixel(Pixel),
}

// pub struct Matrix<const rows:usize, const cols:usize> {
//     data: Vec<f64>, not going to do generics
// }

#[derive(Default, Debug, Clone)]
/// An art [Canvas] / computer screen is represented here.
pub struct Canvas {
    /// The height of the canvas
    height: u32,
    /// The width of the canvas
    width: u32,
    /// The maximum depth of the canvas
    range: u8,
    /// The "body" of the canvas that holds all the [Pixel]s that will be displayed
    pixels: Vec<Pixel>,
    /// A counter that will be used when saving images for animations
    pub(in crate::graphics) anim_index: u32,
    /// A boolean that will determine where "(0, 0)" - the start of the canvas - is located
    pub upper_left_system: bool,
    /// A [Pixel] that represents the color that will be used to draw lines.
    pub line: Pixel,
}

#[allow(dead_code)]
impl Canvas {
    /// Returns a new blank [Canvas] to be drawn on.
    ///
    /// # Arguments
    ///
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `range` - An unsigned int that will represent maximum depth of colors in the [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// let image = Canvas::new(500, 500, 255);
    /// ```
    pub fn new(height: u32, width: u32, range: u8) -> Self {
        Self {
            height,
            width,
            range,
            pixels: vec![Pixel::default(); (height * width) as usize],
            anim_index: 0,
            upper_left_system: false,
            line: Pixel::default(),
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
    /// * `background_color` - A [Pixel] that will represent the color of the
    /// background color that will fill the [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::display::Pixel;
    /// let background_color = Pixel::new(1, 2, 3);
    /// let image = Canvas::new_with_bg(500, 500, 255, &background_color);
    /// ```
    pub fn new_with_bg(height: u32, width: u32, range: u8, background_color: &Pixel) -> Self {
        Self {
            height,
            width,
            range,
            pixels: vec![*background_color; (height * width) as usize],
            anim_index: 0,
            upper_left_system: false,
            line: Pixel::default(),
        }
    }

    /// Returns a blank [Canvas] that can be filled
    ///
    /// # Arguments
    ///
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `range` - An unsigned int that will represent maximum depth
    /// of colors in the [Canvas]
    /// background color that will fill the [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// let image = Canvas::empty(500, 500, 255);
    /// ```
    pub fn empty(height: u32, width: u32, range: u8) -> Self {
        Self {
            height,
            width,
            range,
            pixels: Vec::with_capacity((height * width) as usize),
            anim_index: 0,
            upper_left_system: false,
            line: Pixel::default(),
        }
    }

    /// Returns the width of a [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// let image = Canvas::new(500, 500, 255);
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
    /// let image = Canvas::new(500, 500, 255);
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
    /// use crate::curves_rs::graphics::display::Pixel;
    /// let mut image = Canvas::new(500, 500, 255);
    /// let new_color = Pixel::new(12, 20, 30);
    /// image.set_line_pixel(&new_color);
    /// ```
    pub fn set_line_pixel(&mut self, new_color: &Pixel) {
        self.line = *new_color
    }

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
    /// let mut image = Canvas::new(500, 500, 255);
    /// image.set_line_color(55, 95, 100);
    /// ```
    pub fn set_line_color(&mut self, red: u8, green: u8, blue: u8) {
        self.line.red = red;
        self.line.green = green;
        self.line.blue = blue
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
    /// use crate::curves_rs::graphics::display::Pixel;
    /// let mut image = Canvas::empty(1, 1, 255);
    /// let mut data = vec![Pixel::default()];
    /// image.fill_canvas(data)
    /// ```
    pub fn fill_canvas(&mut self, mut new_pixels: Vec<Pixel>) {
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

    fn iter(&self) -> impl Iterator<Item = &Pixel> + '_ {
        self.pixels.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Pixel> + '_ {
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
    /// use crate::curves_rs::graphics::display::Pixel;
    /// let image = Canvas::new(500, 500, 255);
    /// let color = image.get_pixel(250, 250);
    /// ```
    pub fn get_pixel(&self, x: i32, y: i32) -> &Pixel {
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
    /// use crate::curves_rs::graphics::display::Pixel;
    /// let mut image = Canvas::new(500, 500, 255);
    /// let color = Pixel::new(1, 1, 1);
    /// image.plot(&color, 100, 100);
    /// ```
    pub fn plot(&mut self, new_color: &Pixel, x: i32, y: i32) {
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
    /// use crate::curves_rs::graphics::display::Pixel;
    /// let background_color = Pixel::new(1, 2, 3);
    /// let mut image = Canvas::new_with_bg(500, 500, 255, &background_color);
    /// image.clear_canvas()
    /// ```
    pub fn clear_canvas(&mut self) {
        self.iter_mut().for_each(|i| *i = Pixel::default());
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
    /// use crate::curves_rs::graphics::display::Pixel;
    /// let background_color = Pixel::new(1, 2, 3);
    /// let mut image = Canvas::new(500, 500, 255);
    /// image.fill_color(&background_color)
    /// ```
    pub fn fill_color(&mut self, bg: &Pixel) {
        self.iter_mut().for_each(|i| *i = *bg);
    }
}

impl IntoIterator for Canvas {
    type Item = Pixel;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.pixels.into_iter()
    }
}

impl Index<usize> for Canvas {
    type Output = Pixel;
    fn index(&self, index: usize) -> &Self::Output {
        &self.pixels[index]
    }
}

impl IndexMut<usize> for Canvas {
    fn index_mut(&mut self, index: usize) -> &mut Pixel {
        &mut self.pixels[index]
    }
}

// saving
#[allow(dead_code)]
impl Canvas {
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
    /// let image = Canvas::new(500, 500, 255);
    /// image.save_binary("pics/test.ppm").expect("Could not save file")
    /// ```
    pub fn save_ascii(&self, file_name: &str) -> io::Result<()> {
        let mut file = File::create(file_name)?;
        let mut writer = BufWriter::new(&mut file);
        writeln!(writer, "P3 {} {} {}", self.height, self.width, self.range)?;
        self.iter().for_each(|pixel| {
            write!(writer, "{} {} {} ", pixel.red, pixel.green, pixel.blue)
                .expect("Could not write into file")
        });
        writer.flush()
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
    /// let image = Canvas::new(500, 500, 255);
    /// image.save_binary("pics/test.ppm").expect("Could not save file")
    /// ```
    pub fn save_binary(&self, file_name: &str) -> io::Result<()> {
        let mut file = File::create(file_name)?;
        let mut writer = BufWriter::new(&mut file);
        writer
            .write_all(format!("P6 {} {} {}\n", self.height, self.width, self.range).as_bytes())?;
        self.iter().for_each(|pixel| {
            writer
                .write_all(&pixel.red.to_be_bytes())
                .expect("Could not write as binary");
            writer
                .write_all(&pixel.green.to_be_bytes())
                .expect("Could not write as binary");
            writer
                .write_all(&pixel.blue.to_be_bytes())
                .expect("Could not write as binary");
        });
        writer.flush()
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
    /// let image = Canvas::new(500, 500, 255);
    /// image.save_extension("pics/test.png").expect("Could not save file")
    /// ```
    pub fn save_extension(&self, file_name: &str) -> io::Result<()> {
        let mut content: String = format!("P3 {} {} {}\n", self.height, self.width, self.range);
        self.iter().for_each(|pixel| {
            content.push_str(&format!("{} {} {} ", &pixel.red, &pixel.green, &pixel.blue))
        });
        let mut child = Command::new("convert")
            .arg("-")
            .arg(file_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        child.stdin.as_mut().unwrap().write_all(&content.as_bytes())
    }

    /// Display the current state of the [Canvas].
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// let image = Canvas::new(500, 500, 255);
    /// image.display().expect("Could not display image")
    /// ```
    pub fn display(&self) -> io::Result<()> {
        let mut content: String = format!("P3 {} {} {}\n", self.height, self.width, self.range);
        self.iter().for_each(|pixel| {
            content.push_str(&format!("{} {} {} ", &pixel.red, &pixel.green, &pixel.blue))
        });
        let mut child = Command::new("display")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        child.stdin.as_mut().unwrap().write_all(&content.as_bytes())
    }
}
