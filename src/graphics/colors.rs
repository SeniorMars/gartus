use crate::gmath::vector::Vector;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul};

/// A trait that is meant to bound [Display]
pub trait ColorSpace: Copy + Default + PartialEq + Debug + Into<Rgb> {}

/// Linear RGB color components, before display gamma encoding.
#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct LinearRgb {
    /// Red channel in linear color space.
    pub red: f64,
    /// Green channel in linear color space.
    pub green: f64,
    /// Blue channel in linear color space.
    pub blue: f64,
}

impl LinearRgb {
    /// Creates a linear RGB color.
    #[must_use]
    pub const fn new(red: f64, green: f64, blue: f64) -> Self {
        Self { red, green, blue }
    }

    /// Returns the red component. Kept as an alias for vector-like raytracing code.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.red
    }

    /// Returns the green component. Kept as an alias for vector-like raytracing code.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.green
    }

    /// Returns the blue component. Kept as an alias for vector-like raytracing code.
    #[must_use]
    pub const fn z(self) -> f64 {
        self.blue
    }

    /// Converts display RGB bytes treated as already-linear unit values.
    #[must_use]
    pub fn from_rgb_linear_units(rgb: Rgb) -> Self {
        Self::new(
            f64::from(rgb.red) / 255.0,
            f64::from(rgb.green) / 255.0,
            f64::from(rgb.blue) / 255.0,
        )
    }

    /// Decodes display RGB bytes using the library's gamma-2 approximation.
    #[must_use]
    pub fn from_rgb_srgb(rgb: Rgb) -> Self {
        let decode = |channel: u8| {
            let value = f64::from(channel) / 255.0;
            value * value
        };
        Self::new(decode(rgb.red), decode(rgb.green), decode(rgb.blue))
    }

    /// Encodes this linear color to display RGB using gamma correction.
    #[must_use]
    pub fn gamma_encode(self) -> Rgb {
        Rgb::from_linear_color(self)
    }

    /// Encodes this linear color to display RGB without gamma correction.
    #[must_use]
    pub fn raw_encode(self) -> Rgb {
        Rgb::from_raw_linear_color(self)
    }

    /// Multiplies each channel by the corresponding channel in `rhs`.
    #[must_use]
    pub const fn component_mul(self, rhs: Self) -> Self {
        Self::new(
            self.red * rhs.red,
            self.green * rhs.green,
            self.blue * rhs.blue,
        )
    }
}

impl Add for LinearRgb {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.red + rhs.red,
            self.green + rhs.green,
            self.blue + rhs.blue,
        )
    }
}

impl AddAssign for LinearRgb {
    fn add_assign(&mut self, rhs: Self) {
        self.red += rhs.red;
        self.green += rhs.green;
        self.blue += rhs.blue;
    }
}

impl Mul<f64> for LinearRgb {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.red * rhs, self.green * rhs, self.blue * rhs)
    }
}

impl Mul<LinearRgb> for f64 {
    type Output = LinearRgb;

    fn mul(self, rhs: LinearRgb) -> Self::Output {
        rhs * self
    }
}

impl Div<f64> for LinearRgb {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        (1.0 / rhs) * self
    }
}

impl Index<usize> for LinearRgb {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.red,
            1 => &self.green,
            2 => &self.blue,
            _ => panic!("linear RGB channel index out of bounds"),
        }
    }
}

impl IndexMut<usize> for LinearRgb {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.red,
            1 => &mut self.green,
            2 => &mut self.blue,
            _ => panic!("linear RGB channel index out of bounds"),
        }
    }
}

impl From<Vector> for LinearRgb {
    fn from(color: Vector) -> Self {
        Self::new(color.x(), color.y(), color.z())
    }
}

impl From<LinearRgb> for Vector {
    fn from(color: LinearRgb) -> Self {
        Self::new(color.red, color.green, color.blue)
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[repr(C)]
/// A computer pixel struct is represented by its red, green, blue values
pub struct Rgb {
    /// The first byte that represents red light
    pub red: u8,
    /// The second byte that represents green light
    pub green: u8,
    /// The final byte that represents blue light
    pub blue: u8,
}

/// A sorted list of color stops that can be sampled with linear interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorRamp {
    stops: Vec<(f64, Rgb)>,
}

impl Rgb {
    /// A black Pixel
    pub const BLACK: Rgb = Rgb {
        red: 0,
        green: 0,
        blue: 0,
    };

    /// A red Pixel
    pub const RED: Rgb = Rgb {
        red: 255,
        green: 0,
        blue: 0,
    };

    /// A green Pixel
    pub const GREEN: Rgb = Rgb {
        red: 0,
        green: 255,
        blue: 0,
    };

    /// A blue Pixel
    pub const BLUE: Rgb = Rgb {
        red: 0,
        green: 0,
        blue: 255,
    };

    /// A magenta Pixel
    pub const MAGENTA: Rgb = Rgb {
        red: 255,
        green: 0,
        blue: 255,
    };

    /// A white Pixel
    pub const WHITE: Rgb = Rgb {
        red: 255,
        green: 255,
        blue: 255,
    };

    /// A yellow Pixel
    pub const YELLOW: Rgb = Rgb {
        red: 255,
        green: 255,
        blue: 0,
    };

    /// A cyan Pixel
    pub const CYAN: Rgb = Rgb {
        red: 0,
        green: 255,
        blue: 255,
    };

    /// An orange Pixel
    pub const ORANGE: Rgb = Rgb {
        red: 255,
        green: 165,
        blue: 0,
    };

    /// A purple Pixel
    pub const PURPLE: Rgb = Rgb {
        red: 128,
        green: 0,
        blue: 128,
    };

    /// A gray Pixel
    pub const GRAY: Rgb = Rgb {
        red: 128,
        green: 128,
        blue: 128,
    };

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
    /// use crate::gartus::graphics::colors::Rgb;
    /// let color = Rgb::new(0, 64, 255);
    /// ```
    #[must_use]
    pub fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }

    /// Returns the values of a pixel
    #[must_use]
    pub fn values(&self) -> (u8, u8, u8) {
        (self.red, self.green, self.blue)
    }

    /// Returns the values of a pixel in an array to be bytes
    #[must_use]
    pub fn to_be_bytes(&self) -> [u8; 3] {
        [self.red, self.green, self.blue]
    }

    /// Converts a linear color component to gamma space using gamma 2.
    #[must_use]
    pub fn linear_to_gamma_component(linear_component: f64) -> f64 {
        if linear_component.is_nan() {
            0.0
        } else if linear_component > 0.0 {
            linear_component.sqrt()
        } else {
            0.0
        }
    }

    /// Converts linear RGB components in `0.0..=1.0` to display RGB with gamma correction.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn from_linear_color(color: impl Into<LinearRgb>) -> Self {
        let color = color.into();
        let channel = |component: f64| {
            (256.0 * Self::linear_to_gamma_component(component).clamp(0.0, 0.999)) as u8
        };
        Self::new(channel(color.x()), channel(color.y()), channel(color.z()))
    }

    /// Converts linear RGB components in `0.0..=1.0` to display RGB without gamma correction.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn from_raw_linear_color(color: impl Into<LinearRgb>) -> Self {
        let color = color.into();
        let sanitize = |component: f64| {
            if component.is_nan() { 0.0 } else { component }
        };
        Self {
            red: (255.999 * sanitize(color[0]).clamp(0.0, 1.0)) as u8,
            green: (255.999 * sanitize(color[1]).clamp(0.0, 1.0)) as u8,
            blue: (255.999 * sanitize(color[2]).clamp(0.0, 1.0)) as u8,
        }
    }

    pub(crate) fn name_to_const(color: &str) -> Option<Rgb> {
        match color {
            "black" => Some(Rgb::BLACK),
            "red" => Some(Rgb::RED),
            "green" => Some(Rgb::GREEN),
            "blue" => Some(Rgb::BLUE),
            "magenta" => Some(Rgb::MAGENTA),
            "white" => Some(Rgb::WHITE),
            "yellow" => Some(Rgb::YELLOW),
            "cyan" => Some(Rgb::CYAN),
            "orange" => Some(Rgb::ORANGE),
            "purple" => Some(Rgb::PURPLE),
            "gray" => Some(Rgb::GRAY),
            _ => None,
        }
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    /// Returns the luminance of a pixel
    ///
    /// * `rgb`: A [Rgb] that represents the pixel
    pub fn luminance(self) -> u8 {
        let red = self.red as f32;
        let green = self.green as f32;
        let blue = self.blue as f32;

        (0.299 * red + 0.587 * green + 0.114 * blue).round() as u8
    }

    /// Linearly interpolates between two colors
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            red: (f64::from(self.red) + f64::from(i32::from(other.red) - i32::from(self.red)) * t)
                .round() as u8,
            green: (f64::from(self.green)
                + f64::from(i32::from(other.green) - i32::from(self.green)) * t)
                .round() as u8,
            blue: (f64::from(self.blue)
                + f64::from(i32::from(other.blue) - i32::from(self.blue)) * t)
                .round() as u8,
        }
    }

    /// Multiplies each channel by `factor`, clamping the result to `0..=255`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn scale(self, factor: f64) -> Self {
        let scale_channel =
            |channel: u8| (f64::from(channel) * factor).round().clamp(0.0, 255.0) as u8;
        Self::new(
            scale_channel(self.red),
            scale_channel(self.green),
            scale_channel(self.blue),
        )
    }
}

impl ColorRamp {
    /// Creates a color ramp from `(position, color)` stops.
    ///
    /// Stop positions are sorted ascending. Sampling before the first stop returns the first
    /// color, and sampling after the last stop returns the last color.
    ///
    /// # Panics
    /// Panics if no stops are supplied or any stop position is non-finite.
    #[must_use]
    pub fn new(mut stops: Vec<(f64, Rgb)>) -> Self {
        assert!(!stops.is_empty(), "color ramp must have at least one stop");
        assert!(
            stops.iter().all(|(position, _)| position.is_finite()),
            "color ramp stop positions must be finite"
        );
        stops.sort_by(|(a, _), (b, _)| a.partial_cmp(b).expect("positions should be finite"));
        Self { stops }
    }

    /// Samples the ramp at `position`.
    #[must_use]
    pub fn sample(&self, position: f64) -> Rgb {
        if position <= self.stops[0].0 {
            return self.stops[0].1;
        }
        for window in self.stops.windows(2) {
            let (low_position, low_color) = window[0];
            let (high_position, high_color) = window[1];
            if position <= high_position {
                let span = high_position - low_position;
                let t = if span.abs() <= f64::EPSILON {
                    0.0
                } else {
                    (position - low_position) / span
                };
                return low_color.lerp(high_color, t);
            }
        }
        self.stops[self.stops.len() - 1].1
    }

    /// Returns the sorted color stops.
    #[must_use]
    pub fn stops(&self) -> &[(f64, Rgb)] {
        &self.stops
    }
}

impl ColorSpace for Rgb {}

impl From<Vector> for Rgb {
    fn from(color: Vector) -> Self {
        Self::from_linear_color(color)
    }
}

impl From<LinearRgb> for Rgb {
    fn from(color: LinearRgb) -> Self {
        Self::from_linear_color(color)
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A convention that represents a Pixel based on hue, saturation, and light
pub struct Hsl {
    /// Hue [0, 360)
    pub hue: u16,
    /// Saturation [0, 100]
    pub saturation: u16,
    /// Light [0, 100]
    pub light: u16,
}

impl ColorSpace for Hsl {}

impl Hsl {
    /// Returns a HSL color.
    ///
    /// Hue is normalized with modulo 360. Saturation and lightness are clamped to `0..=100`.
    #[must_use]
    pub fn new(hue: u16, saturation: u16, light: u16) -> Self {
        Self {
            hue: hue % 360,
            saturation: saturation.min(100),
            light: light.min(100),
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]
impl From<Hsl> for Rgb {
    fn from(hsl: Hsl) -> Self {
        let hue = hsl.hue as f32 / 360.0;
        let saturation = hsl.saturation as f32 / 100.0;
        let light = hsl.light as f32 / 100.0;

        let (r, g, b) = if saturation == 0.0 {
            (light, light, light)
        } else {
            let q = if light < 0.5 {
                light * (1.0 + saturation)
            } else {
                light + saturation - light * saturation
            };
            let p = 2.0 * light - q;

            let hue_to_rgb = |p: f32, q: f32, mut t: f32| {
                if t < 0.0 {
                    t += 1.0;
                }
                if t > 1.0 {
                    t -= 1.0;
                }
                if t < 1.0 / 6.0 {
                    return p + (q - p) * 6.0 * t;
                }
                if t < 1.0 / 2.0 {
                    return q;
                }
                if t < 2.0 / 3.0 {
                    return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
                }
                p
            };

            (
                hue_to_rgb(p, q, hue + 1.0 / 3.0),
                hue_to_rgb(p, q, hue),
                hue_to_rgb(p, q, hue - 1.0 / 3.0),
            )
        };

        Rgb {
            red: (r * 255.999) as u8,
            green: (g * 255.999) as u8,
            blue: (b * 255.999) as u8,
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]
impl From<Rgb> for Hsl {
    fn from(rgb: Rgb) -> Self {
        let r = rgb.red as f32 / 255.0;
        let g = rgb.green as f32 / 255.0;
        let b = rgb.blue as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = f32::midpoint(max, min);

        let (h, s) = if max == min {
            (0.0, 0.0)
        } else {
            let d = max - min;
            let s = if l > 0.5 {
                d / (2.0 - max - min)
            } else {
                d / (max + min)
            };
            let h = if max == r {
                (g - b) / d + (if g < b { 6.0 } else { 0.0 })
            } else if max == g {
                (b - r) / d + 2.0
            } else {
                (r - g) / d + 4.0
            };
            (h / 6.0, s)
        };

        Hsl {
            hue: (h * 360.0).round() as u16,
            saturation: (s * 100.0).round() as u16,
            light: (l * 100.0).round() as u16,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A convention that represents a Pixel based on Hue, Saturation, and Value
pub struct Hsv {
    /// Hue [0, 360)
    pub hue: u16,
    /// Saturation [0, 100]
    pub saturation: u16,
    /// Value [0, 100]
    pub value: u16,
}

impl ColorSpace for Hsv {}

impl Hsv {
    /// Returns a HSV color.
    ///
    /// Hue is normalized with modulo 360. Saturation and value are clamped to `0..=100`.
    #[must_use]
    pub fn new(hue: u16, saturation: u16, value: u16) -> Self {
        Self {
            hue: hue % 360,
            saturation: saturation.min(100),
            value: value.min(100),
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
impl From<Hsv> for Rgb {
    fn from(hsv: Hsv) -> Self {
        let h = hsv.hue as f32 / 360.0;
        let s = hsv.saturation as f32 / 100.0;
        let v = hsv.value as f32 / 100.0;

        let i = (h * 6.0).floor();
        let f = h * 6.0 - i;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);

        let (r, g, b) = match (i as i32) % 6 {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };

        Rgb::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }
}

#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]
impl From<Rgb> for Hsv {
    fn from(rgb: Rgb) -> Self {
        let r = rgb.red as f32 / 255.0;
        let g = rgb.green as f32 / 255.0;
        let b = rgb.blue as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;

        let h = if max == min {
            0.0
        } else if max == r {
            (g - b) / d + (if g < b { 6.0 } else { 0.0 })
        } else if max == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };

        let s = if max == 0.0 { 0.0 } else { d / max };
        let v = max;

        Hsv {
            hue: ((h / 6.0) * 360.0).round() as u16,
            saturation: (s * 100.0).round() as u16,
            value: (v * 100.0).round() as u16,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A convention that represents a Pixel based on Cyan, Magenta, and Yellow
pub struct Cmy {
    /// Cyan [0, 100]
    pub cyan: u16,
    /// Magenta [0, 100]
    pub magenta: u16,
    /// Yellow [0, 100]
    pub yellow: u16,
}

impl ColorSpace for Cmy {}

impl Cmy {
    /// Returns a CMY color
    #[must_use]
    pub fn new(cyan: u16, magenta: u16, yellow: u16) -> Self {
        Self {
            cyan: cyan.min(100),
            magenta: magenta.min(100),
            yellow: yellow.min(100),
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
impl From<Cmy> for Rgb {
    fn from(cmy: Cmy) -> Self {
        let cyan = cmy.cyan as f32 / 100.0;
        let magenta = cmy.magenta as f32 / 100.0;
        let yellow = cmy.yellow as f32 / 100.0;

        Rgb {
            red: ((1.0 - cyan) * 255.999) as u8,
            green: ((1.0 - magenta) * 255.999) as u8,
            blue: ((1.0 - yellow) * 255.999) as u8,
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
impl From<Rgb> for Cmy {
    fn from(rgb: Rgb) -> Self {
        Self {
            cyan: ((1.0 - (rgb.red as f32 / 255.0)) * 100.0).round() as u16,
            magenta: ((1.0 - (rgb.green as f32 / 255.0)) * 100.0).round() as u16,
            yellow: ((1.0 - (rgb.blue as f32 / 255.0)) * 100.0).round() as u16,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A convention that represents a Pixel based on Cyan, Magenta, Yellow, and Key (Black)
pub struct Cmyk {
    /// Cyan [0, 100]
    pub cyan: u16,
    /// Magenta [0, 100]
    pub magenta: u16,
    /// Yellow [0, 100]
    pub yellow: u16,
    /// Key (Black) [0, 100]
    pub key: u16,
}

impl ColorSpace for Cmyk {}

impl Cmyk {
    /// Returns a CMYK color
    #[must_use]
    pub fn new(cyan: u16, magenta: u16, yellow: u16, key: u16) -> Self {
        Self {
            cyan: cyan.min(100),
            magenta: magenta.min(100),
            yellow: yellow.min(100),
            key: key.min(100),
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
impl From<Cmyk> for Rgb {
    fn from(cmyk: Cmyk) -> Self {
        let c = cmyk.cyan as f32 / 100.0;
        let m = cmyk.magenta as f32 / 100.0;
        let y = cmyk.yellow as f32 / 100.0;
        let k = cmyk.key as f32 / 100.0;

        Rgb {
            red: (255.0 * (1.0 - c) * (1.0 - k)) as u8,
            green: (255.0 * (1.0 - m) * (1.0 - k)) as u8,
            blue: (255.0 * (1.0 - y) * (1.0 - k)) as u8,
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]
impl From<Rgb> for Cmyk {
    fn from(rgb: Rgb) -> Self {
        let r = rgb.red as f32 / 255.0;
        let g = rgb.green as f32 / 255.0;
        let b = rgb.blue as f32 / 255.0;

        let k = 1.0 - r.max(g).max(b);
        if k == 1.0 {
            return Cmyk {
                cyan: 0,
                magenta: 0,
                yellow: 0,
                key: 100,
            };
        }

        let c = (1.0 - r - k) / (1.0 - k);
        let m = (1.0 - g - k) / (1.0 - k);
        let y = (1.0 - b - k) / (1.0 - k);

        Cmyk {
            cyan: (c * 100.0).round() as u16,
            magenta: (m * 100.0).round() as u16,
            yellow: (y * 100.0).round() as u16,
            key: (k * 100.0).round() as u16,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn vector_to_rgb_applies_gamma_correction() {
        assert_eq!(
            Rgb::from(Vector::new(0.25, 0.0, 1.0)),
            Rgb::new(128, 0, 255)
        );
    }

    #[test]
    fn raw_linear_color_conversion_skips_gamma_correction() {
        assert_eq!(
            Rgb::from_raw_linear_color(Vector::new(0.25, 0.0, 1.0)),
            Rgb::new(63, 0, 255)
        );
    }

    #[test]
    fn linear_color_conversion_replaces_nan_channels() {
        let color = LinearRgb::new(f64::NAN, f64::INFINITY, -1.0);

        assert_eq!(Rgb::from_linear_color(color), Rgb::new(0, 255, 0));
        assert_eq!(Rgb::from_raw_linear_color(color), Rgb::new(0, 255, 0));
    }

    #[test]
    fn hsl_rgb() {
        let hsl = Hsl::new(1, 100, 50);
        let rgb = Rgb::from(hsl);
        assert_eq!(
            rgb,
            Rgb {
                red: 255,
                green: 4,
                blue: 0
            }
        );
    }

    #[test]
    fn rgb_hsl() {
        let rgb = Rgb::new(255, 4, 0);
        let hsl = Hsl::from(rgb);
        assert_eq!(
            hsl,
            Hsl {
                hue: 1,
                saturation: 100,
                light: 50
            }
        );
    }

    #[test]
    fn hsv_rgb() {
        let hsv = Hsv::new(0, 100, 100);
        let rgb = Rgb::from(hsv);
        assert_eq!(rgb, Rgb::RED);
    }

    #[test]
    fn rgb_hsv() {
        let rgb = Rgb::RED;
        let hsv = Hsv::from(rgb);
        assert_eq!(hsv, Hsv::new(0, 100, 100));
    }

    #[test]
    fn cmyk_rgb() {
        let cmyk = Cmyk::new(0, 0, 0, 100);
        let rgb = Rgb::from(cmyk);
        assert_eq!(rgb, Rgb::BLACK);
    }

    #[test]
    fn rgb_cmyk() {
        let rgb = Rgb::BLACK;
        let cmyk = Cmyk::from(rgb);
        assert_eq!(cmyk, Cmyk::new(0, 0, 0, 100));
    }

    #[test]
    fn lerp_test() {
        let red = Rgb::RED;
        let blue = Rgb::BLUE;
        let purple = red.lerp(blue, 0.5);
        assert_eq!(purple.red, 128);
        assert_eq!(purple.blue, 128);
    }

    #[test]
    fn scale_clamps_channels() {
        assert_eq!(Rgb::new(100, 120, 140).scale(2.0), Rgb::new(200, 240, 255));
    }

    #[test]
    fn color_ramp_sorts_and_samples_stops() {
        let ramp = ColorRamp::new(vec![(1.0, Rgb::WHITE), (0.0, Rgb::BLACK)]);

        assert_eq!(ramp.sample(-1.0), Rgb::BLACK);
        assert_eq!(ramp.sample(0.5), Rgb::new(128, 128, 128));
        assert_eq!(ramp.sample(2.0), Rgb::WHITE);
    }
}
