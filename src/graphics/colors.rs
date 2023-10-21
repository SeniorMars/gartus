use crate::gmath::vector::Vector;
use std::fmt::Debug;

/// A trait that is meant to bound [Display]
pub trait ColorSpace: Copy + Default + PartialEq + Debug + Into<Rgb> {}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A computer pixel struct is represented by its red, green, blue values
pub struct Rgb {
    /// The first byte that represents red light
    pub red: u8,
    /// The second byte that represents green light
    pub green: u8,
    /// The final byte that represents blue light
    pub blue: u8,
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
            red: (f64::from(self.red) + f64::from(i32::from(other.red) - i32::from(self.red)) * t).round()
                as u8,
            green: (f64::from(self.green) + f64::from(i32::from(other.green) - i32::from(self.green)) * t)
                .round() as u8,
            blue: (f64::from(self.blue) + f64::from(i32::from(other.blue) - i32::from(self.blue)) * t)
                .round() as u8,
        }
    }
}

impl ColorSpace for Rgb {}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
impl From<Vector> for Rgb {
    fn from(color: Vector) -> Self {
        Self {
            red: (255.999 * color[0].clamp(0.0, 1.0)) as u8,
            green: (255.999 * color[1].clamp(0.0, 1.0)) as u8,
            blue: (255.999 * color[2].clamp(0.0, 1.0)) as u8,
        }
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
}
