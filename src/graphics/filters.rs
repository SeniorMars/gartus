use super::{colors::Rgb, display::Canvas};

impl Canvas<Rgb> {
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    /// Applies a grayscale filter to the current canvas and results in a new canvas
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::gartus::graphics::{display::Canvas, colors::Rgb};
    /// let colors = [Rgb::YELLOW, Rgb::CYAN, Rgb::RED, Rgb::BLUE];
    /// let mut canvas = Canvas::with_capacity(2, 2, 255, Rgb::new(0, 0, 0));
    /// canvas.fill_canvas(colors.to_vec());
    /// let gray = canvas.grayscale();
    /// ```
    pub fn grayscale(&self) -> Canvas<Rgb> {
        let mut filtered_image = self.clone();
        filtered_image.iter_mut().for_each(|pixel| {
            let (r, g, b) = pixel.values();
            let average = ((r as f32 + g as f32 + b as f32) / 3.0).round() as u8;
            *pixel = Rgb::new(average, average, average);
        });
        filtered_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    /// Applies the sepia filter to the current canvas
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::gartus::graphics::{display::Canvas, colors::Rgb};
    /// let colors = [Rgb::YELLOW, Rgb::CYAN, Rgb::RED, Rgb::BLUE];
    /// let mut canvas = Canvas::with_capacity(2, 2, 255, Rgb::new(0, 0, 0));
    /// canvas.fill_canvas(colors.to_vec());
    /// let sepia = canvas.sepia();
    /// ```
    pub fn sepia(&self) -> Canvas<Rgb> {
        let mut filtered_image = self.clone();
        filtered_image.iter_mut().for_each(|pixel| {
            let (r, g, b) = pixel.values();
            let sepia_red = (0.393 * r as f32 + 0.769 * g as f32 + 0.198 * b as f32).round() as u8;
            let sepia_green =
                (0.349 * r as f32 + 0.686 * g as f32 + 0.168 * b as f32).round() as u8;
            let sepia_blue = (0.272 * r as f32 + 0.534 * g as f32 + 0.131 * b as f32).round() as u8;
            *pixel = Rgb::new(sepia_red, sepia_green, sepia_blue);
        });
        filtered_image
    }

    #[must_use]
    /// Applies the reflect filter to the current canvas and results in a new canvas
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::gartus::graphics::{display::Canvas, colors::Rgb};
    /// let colors = [Rgb::YELLOW, Rgb::CYAN, Rgb::RED, Rgb::BLUE];
    /// let mut canvas = Canvas::with_capacity(2, 2, 255, Rgb::new(0, 0, 0));
    /// canvas.fill_canvas(colors.to_vec());
    /// let reflect = canvas.reflect();
    /// ```
    pub fn reflect(&self) -> Canvas<Rgb> {
        let mut filtered_image = self.clone();
        filtered_image.iter_row_mut().for_each(|row| {
            let len = row.len();
            (0..row.len() / 2).for_each(|i| {
                row.swap(i, len - i - 1);
            });
        });
        filtered_image
    }

    #[allow(clippy::cast_possible_wrap)]
    fn grid(i: usize, width: isize) -> [isize; 9] {
        // it could be that we get negative numbers, so we must be sure to
        // adjust for that. The algorithm would function differently if don't adjust for this,
        // and this would be undefined behavior
        let i = i as isize;
        // Here is how these numbers I would get therse numbers
        // [   (i - width - 1), (i - width), (i - width + 1)
        //     i - 1, i, i + 1
        //     (i+ width - 1), (i + width), (i + width + 1)
        // ]
        [
            (i - width - 1),
            (i - width),
            (i - width + 1),
            (i - 1),
            i,
            (i + 1),
            (i + width - 1),
            (i + width),
            (i + width + 1),
        ]
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless,
        clippy::cast_possible_wrap,
        clippy::similar_names
    )]
    /// Applies a blur filter to the current canvas and results in a new canvas
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::gartus::graphics::{display::Canvas, colors::Rgb};
    /// let colors = [Rgb::YELLOW, Rgb::CYAN, Rgb::RED, Rgb::BLUE];
    /// let mut canvas = Canvas::with_capacity(2, 2, 255, Rgb::new(0, 0, 0));
    /// canvas.fill_canvas(colors.to_vec());
    /// let blur = canvas.blur();
    /// ```
    pub fn blur(&self) -> Canvas<Rgb> {
        let width = self.width() as isize;
        let size = self.len() as isize;
        let mut filtered_image = self.clone();

        let blur = |i: usize| -> (u8, u8, u8) {
            let mut counter = 0f32;
            let (mut red_sum, mut green_sum, mut blue_sum) = (0u16, 0u16, 0u16);
            let blur_grid = Canvas::grid(i, width);
            for element in &blur_grid {
                let index = *element;
                if index >= 0 && index < size {
                    let pixel = self[index as usize];
                    red_sum += pixel.red as u16;
                    green_sum += pixel.green as u16;
                    blue_sum += pixel.blue as u16;
                    counter += 1.0;
                }
            }
            (
                (red_sum as f32 / counter).round() as u8,
                (green_sum as f32 / counter).round() as u8,
                (blue_sum as f32 / counter).round() as u8,
            )
        };

        filtered_image
            .iter_mut()
            .enumerate()
            .for_each(|(i, pixel)| {
                let (red, green, blue) = blur(i);
                *pixel = Rgb::new(red, blue, green);
            });
        filtered_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless,
        clippy::cast_possible_wrap,
        clippy::similar_names
    )]
    /// Applies a sobel filter to the current canvas and results in a new canvas
    ///
    /// ```
    /// use crate::gartus::graphics::{display::Canvas, colors::Rgb};
    /// let colors = [Rgb::YELLOW, Rgb::CYAN, Rgb::RED, Rgb::BLUE];
    /// let mut canvas = Canvas::with_capacity(2, 2, 255, Rgb::new(0, 0, 0));
    /// canvas.fill_canvas(colors.to_vec());
    /// let sobel = canvas.sobel();
    /// ```
    pub fn sobel(&self) -> Canvas<Rgb> {
        let width = self.width() as isize;
        let size = self.len() as isize;
        let mut filtered_image = self.clone();

        let gx_kernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        let gy_kernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
        let g_filter = |gx: i32, gy: i32| -> u8 {
            let color = (((gx * gx + gy * gy) as f64).sqrt()).round();
            if color > 255f64 {
                255
            } else {
                color as u8
            }
        };

        let sobel = |i: usize| -> (u8, u8, u8) {
            let (mut red_x, mut green_x, mut blue_x) = (0i16, 0i16, 0i16);
            let (mut red_y, mut green_y, mut blue_y) = (0i16, 0i16, 0i16);
            let grid = Canvas::grid(i, width);
            grid.iter().enumerate().for_each(|(g_index, element)| {
                let copy_index = *element;
                if copy_index >= 0 && copy_index < size {
                    let pixel = self[copy_index as usize];
                    let (r, g, b) = (pixel.red as i16, pixel.green as i16, pixel.blue as i16);

                    red_x += r * gx_kernel[g_index];
                    red_y += r * gy_kernel[g_index];

                    green_x += g * gx_kernel[g_index];
                    green_y += g * gy_kernel[g_index];

                    blue_x += b * gx_kernel[g_index];
                    blue_y += b * gy_kernel[g_index];
                }
            });
            (
                g_filter(red_x as i32, red_y as i32),
                g_filter(green_x as i32, green_y as i32),
                g_filter(blue_x as i32, blue_y as i32),
            )
        };

        filtered_image
            .iter_mut()
            .enumerate()
            .for_each(|(i, pixel)| {
                let (red, green, blue) = sobel(i);
                *pixel = Rgb::new(red, blue, green);
            });
        filtered_image
    }
}

#[test]
fn blur_test() {
    let colors = vec![
        Rgb::GREEN,
        Rgb::BLUE,
        Rgb::RED,
        Rgb::GREEN,
        Rgb::BLUE,
        Rgb::RED,
        Rgb::GREEN,
        Rgb::BLUE,
        Rgb::RED,
    ];
    let mut canvas = Canvas::with_capacity(3, 3, 255, Rgb::BLACK);
    canvas.fill_canvas(colors);
    println!("{}", canvas.blur());
}
