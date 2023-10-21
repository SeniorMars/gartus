use crate::gmath::vector::Vector;
use std::fmt::Debug;

#[cfg(feature = "colors")]
use std::cmp::{max, min};

/// A trait that is meant to bound [Display]
pub trait ColorSpace: Copy + Default + PartialEq + Debug
where
    Rgb: From<Self>,
{
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A computer pixel struct is represented by its red, green, blue values
// TODO: in ppm spec, Rgb is actually u16 max. Reimplment this to make sure it follows the spec correctly
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
        [
            self.red.to_be_bytes()[0], // Convert red to 1-byte array and take the first byte
            self.green.to_be_bytes()[0], // Convert green to 1-byte array and take the first byte
            self.blue.to_be_bytes()[0], // Convert blue to 1-byte array and take the first byte
        ]
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
}

impl ColorSpace for Rgb {}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
impl From<Vector> for Rgb {
    fn from(color: Vector) -> Self {
        Self {
            red: (255.999 * color[0]) as u8,
            green: (255.999 * color[1]) as u8,
            blue: (255.999 * color[2]) as u8,
        }
    }
}

#[cfg(feature = "colors")]
#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
impl From<Hsl> for Rgb {
    fn from(hsl: Hsl) -> Self {
        let (r, g, b);
        let hue = hsl.hue as f32 / 360.0;
        let saturation = hsl.saturation as f32 / 100.0;
        let light = hsl.light as f32 / 100.0;

        if saturation == 0.0 {
            r = light;
            g = light;
            b = light;
        } else {
            let hue_conversion = |p: f32, q: f32, mut t: f32| {
                if t < 0.0 {
                    t += 1.0;
                }
                if t > 1.0 {
                    t -= 1.0;
                }
                if t < (1.0 / 6.0) {
                    return p + (q - p) * 6.0 * t;
                }
                if t < (1.0 / 2.0) {
                    return q;
                }
                if t < (2.0 / 3.0) {
                    return p + (q - p) * ((2.0 / 3.0) - t) * 6.0;
                }
                p
            };
            let q = if light < 0.5 {
                light * (1.0 + saturation)
            } else {
                light + saturation - light * saturation
            };
            let p = 2.0 * light - q;
            r = hue_conversion(p, q, hue + (1.0 / 3.0));
            g = hue_conversion(p, q, hue);
            b = hue_conversion(p, q, hue - (1.0 / 3.0));
        }
        Rgb {
            red: (r * 255.999) as u8,
            green: (g * 255.999) as u8,
            blue: (b * 255.999) as u8,
        }
    }
}

#[cfg(feature = "colors")]
#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A convention that represents a Pixel based on hue, saturation, and light
pub struct Hsl {
    /// Hue
    pub hue: u16,
    /// saturation
    pub saturation: u16,
    /// light
    pub light: u16,
}

#[cfg(feature = "colors")]
impl ColorSpace for Hsl {}

#[cfg(feature = "colors")]
#[allow(dead_code)]
impl Hsl {
    /// Returns a HSL that can be used in [Canvas]
    ///
    /// # Arguments
    ///
    /// * `hue` - A u16 that represents hue -- should be a number from [0, 360)
    /// * `saturation` - A u8 that represents saturation percentage -- should be a number from [0, 100]
    /// * `light` - A u8 that represent light percentage -- should be a number from [0, 100]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::colors::Hsl;
    /// let color = Hsl::new(10, 50, 0);
    /// ```
    #[must_use]
    pub fn new(hue: u16, saturation: u16, light: u16) -> Self {
        Self {
            hue: hue.clamp(0, 359),
            saturation: saturation.clamp(0, 100),
            light: light.clamp(0, 100),
        }
    }
}

#[cfg(feature = "colors")]
#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
impl From<Rgb> for Hsl {
    fn from(rgb: Rgb) -> Self {
        let (mut h, s, l);
        let r = rgb.red as f32 / 255.0;
        let g = rgb.green as f32 / 255.0;
        let b = rgb.blue as f32 / 255.0;
        let (max, min) = (
            max(rgb.red, max(rgb.green, rgb.blue)) as f32 / 255.0,
            min(rgb.red, min(rgb.green, rgb.blue)) as f32 / 255.0,
        );
        l = (max + min) / 2.0;
        if (max - min).abs() < 0.00001 {
            h = 0.0;
            s = 0.0;
        } else {
            let delta = max - min;
            s = if l > 0.5 {
                delta / (2.0 - max - min)
            } else {
                delta / (max + min)
            };

            h = if r > g && r > b {
                let float = if g < b { 6.0 } else { 0.0 };
                (g - b) / delta + float
            } else if g > b {
                (b - r) / delta + 2.0
            } else {
                (r - g) / delta + 4.0
            };
            h /= 6.0;
        }

        Hsl {
            hue: (h * 360.0).round() as u16,
            saturation: (s * 100.0).round() as u16,
            light: (l * 100.0).round() as u16,
        }
    }
}

#[cfg(feature = "colors")]
#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
impl From<Cmy> for Rgb {
    fn from(cmy: Cmy) -> Self {
        let (r, g, b);
        let cyan = cmy.cyan as f32 / 100.0;
        let magenta = cmy.magenta as f32 / 100.0;
        let yellow = cmy.yellow as f32 / 100.0;

        r = (1.0 - cyan) * 255.999;
        g = (1.0 - magenta) * 255.999;
        b = (1.0 - yellow) * 255.999;

        Rgb {
            red: r as u8,
            green: g as u8,
            blue: b as u8,
        }
    }
}

#[cfg(feature = "colors")]
#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A convention that represents a Pixel based on Cyan, Magenta, and Yellow
pub struct Cmy {
    /// Cyan
    pub cyan: u16,
    /// Magenta
    pub magenta: u16,
    /// Yellow
    pub yellow: u16,
}

#[cfg(feature = "colors")]
impl ColorSpace for Cmy {}

#[allow(clippy::many_single_char_names)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
#[cfg(feature = "colors")]
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
    fn conversion() {
        let rgb = Rgb::new(255, 4, 0);
        let hsl = Hsl::from(rgb);
        let new_rgb = Rgb::from(hsl);
        assert_eq!(
            new_rgb,
            Rgb {
                red: 255,
                green: 4,
                blue: 0
            }
        );
    }
}
