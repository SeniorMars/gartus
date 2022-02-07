use crate::gmath::vector::Vector;
use std::cmp::{max, min};

/// A trait that is meant to bound [Display]
pub trait ColorSpace: Default + Copy + PartialEq
where
    Rgb: From<Self>,
{
    fn new(_: u16, _: u16, _: u16) -> Self;
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
/// A computer pixel struct is represented by its red, green, blue values
pub struct Rgb {
    /// The first byte that represents red light
    pub red: u16,
    /// The second byte that represents green light
    pub green: u16,
    /// The final byte that represents blue light
    pub blue: u16,
}

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

// impl Pixel for RGB {}

#[allow(dead_code)]
impl ColorSpace for Rgb {
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
    /// use crate::curves_rs::graphics::colors::RGB;
    /// let color = RGB::new(0, 64, 255);
    /// ```
    fn new(red: u16, green: u16, blue: u16) -> Self {
        Self { red, green, blue }
    }
}

impl From<Vector> for Rgb {
    fn from(color: Vector) -> Self {
        Self {
            red: (255.00 * color[0] as f64) as u16,
            green: (255.00 * color[1] as f64) as u16,
            blue: (255.00 * color[2] as f64) as u16,
        }
    }
}

#[allow(clippy::many_single_char_names)]
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
                    t += 1.0
                }
                if t > 1.0 {
                    t -= 1.0
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
            red: (r * 255.00) as u16,
            green: (g * 255.00) as u16,
            blue: (b * 255.00) as u16,
        }
    }
}

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

#[allow(dead_code)]
impl ColorSpace for Hsl {
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
    /// use crate::curves_rs::graphics::colors::HSL;
    /// let color = HSL::new(10, 50, 0);
    /// ```
    fn new(hue: u16, saturation: u16, light: u16) -> Self {
        Self {
            hue: hue.clamp(0, 359),
            saturation: saturation.clamp(0, 100),
            light: light.clamp(0, 100),
        }
    }
}

#[allow(clippy::many_single_char_names)]
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

// #[derive(Debug, Clone, Copy, PartialEq)]
// /// A type that represents a Pixel on a [Canvas] that can either be a RGB or HSL value
// pub enum Pixel {
//     /// A pixel defined in terms of HSL
//     Hsl(Hsl),
//     /// A pixel defined in terms of RGB
//     Rgb(Rgb),
// }
//
// impl Pixel {
//     /// Returns `true` if the pixel color is [`HSL`].
//     ///
//     ///
//     /// [`HSL`]: PixelColor::HSL
//     pub fn is_hsl(&self) -> bool {
//         matches!(self, Self::Hsl(..))
//     }
//
//     /// Returns `true` if the pixel color is [`RGB`].
//     ///
//     /// [`RGB`]: PixelColor::RGB
//     pub fn is_rgb(&self) -> bool {
//         matches!(self, Self::Rgb(..))
//     }
// }
//
// impl Default for Pixel {
//     fn default() -> Self {
//         Self::Rgb(Rgb::default())
//     }
// }
//
// impl From<Pixel> for Rgb {
//     fn from(pixel: Pixel) -> Self {
//         match pixel {
//             Pixel::Hsl(hsl) => Rgb::from(hsl),
//             Pixel::Rgb(rgb) => rgb,
//         }
//     }
// }
//
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
        )
    }
}
