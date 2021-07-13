use crate::graphics::vector::Vector;
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
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// let color = Pixel::new(0, 64, 255);
    /// ```
    pub fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }
}

impl From<Vector> for Pixel {
    fn from(color: Vector) -> Self {
        Self {
            red: (255.99 * color[0] as f64) as u8,
            green: (255.99 * color[1] as f64) as u8,
            blue: (255.99 * color[2] as f64) as u8,
        }
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
    /// use crate::curves_rs::graphics::colors::HSL;
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
