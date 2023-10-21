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
                *pixel = Rgb::new(red, green, blue);
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

    #[must_use]
    /// Applies an invert colors filter to the current canvas and results in a new canvas
    pub fn invert(&self) -> Canvas<Rgb> {
        let mut inverted_image = self.clone();
        inverted_image.iter_mut().for_each(|pixel| {
            let (r, g, b) = pixel.values();
            let inverted_red = 255 - r;
            let inverted_green = 255 - g;
            let inverted_blue = 255 - b;
            *pixel = Rgb::new(inverted_red, inverted_green, inverted_blue);
        });
        inverted_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    /// Applies a black and white filter to the current canvas and results in a new canvas
    ///
    /// * `threshold`: The threshold value to use to determine if a pixel is black or white
    pub fn black_and_white(&self, threshold: u8) -> Canvas<Rgb> {
        let mut bw_image = self.clone();
        bw_image.iter_mut().for_each(|pixel| {
            let (r, g, b) = pixel.values();
            let grayscale_value =
                ((f32::from(r) + f32::from(g) + f32::from(b)) / 3.0).round() as u8;
            if grayscale_value >= threshold {
                *pixel = Rgb::new(255, 255, 255); // White
            } else {
                *pixel = Rgb::new(0, 0, 0); // Black
            }
        });
        bw_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    /// Applies a brightness filter to the current canvas and results in a new canvas
    ///
    /// * `brightness`: The brightness value to use to adjust the brightness of the image
    pub fn adjust_brightness(&self, brightness: u8) -> Canvas<Rgb> {
        let mut adjusted_image = self.clone();
        adjusted_image.iter_mut().for_each(|pixel| {
            let (r, g, b) = pixel.values();

            // Adjust brightness by adding the constant value
            let new_r = (r + brightness).max(0).min(255);
            let new_g = (g + brightness).max(0).min(255);
            let new_b = (b + brightness).max(0).min(255);

            *pixel = Rgb::new(new_r, new_g, new_b);
        });
        adjusted_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    /// Applies a contrast filter to the current canvas and results in a new canvas
    pub fn adjust_contrast(&self, contrast: f32) -> Canvas<Rgb> {
        let mut adjusted_image = self.clone();
        adjusted_image.iter_mut().for_each(|pixel| {
            let (r, g, b) = pixel.values();

            // Adjust contrast by multiplying each color component by the contrast factor
            let new_r = (((r as f32 - 127.5) * contrast) + 127.5)
                .max(0.0)
                .min(255.0) as u8;
            let new_g = (((g as f32 - 127.5) * contrast) + 127.5)
                .max(0.0)
                .min(255.0) as u8;
            let new_b = (((b as f32 - 127.5) * contrast) + 127.5)
                .max(0.0)
                .min(255.0) as u8;

            *pixel = Rgb::new(new_r, new_g, new_b);
        });
        adjusted_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless
    )]
    /// Laplacian edge detection filter
    pub fn laplacian_edge_detection(&self) -> Canvas<Rgb> {
        let width = self.width() as isize;
        let height = self.height() as isize;
        let mut filtered_image = self.clone();

        let kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]];

        let apply_kernel = |i: isize| -> (u8, u8, u8) {
            let (mut red_acc, mut green_acc, mut blue_acc) = (0i32, 0i32, 0i32);

            for y in 0..3 {
                for x in 0..3 {
                    let neighbor_x = (i % width) + (x - 1);
                    let neighbor_y = (i / width) + (y - 1);

                    if neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height
                    {
                        let neighbor_index = neighbor_y * width + neighbor_x;
                        let pixel = self[neighbor_index as usize];
                        let (r, g, b) = (pixel.red as i32, pixel.green as i32, pixel.blue as i32);

                        let weight = kernel[y as usize][x as usize];
                        red_acc += r * weight;
                        green_acc += g * weight;
                        blue_acc += b * weight;
                    }
                }
            }

            (
                (red_acc.max(0).min(255)) as u8,
                (green_acc.max(0).min(255)) as u8,
                (blue_acc.max(0).min(255)) as u8,
            )
        };

        filtered_image
            .iter_mut()
            .enumerate()
            .for_each(|(i, pixel)| {
                let (red, green, blue) = apply_kernel(i as isize);
                *pixel = Rgb::new(red, green, blue);
            });

        filtered_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless,
        clippy::cast_precision_loss
    )]
    /// Applies Gaussian blur to the current canvas and returns a new canvas with the blurred effect.
    ///
    /// # Arguments
    ///
    /// * `radius` - The radius of the Gaussian blur. A larger radius results in stronger blur.
    ///
    /// # Returns
    ///
    /// A new `Canvas<Rgb>` with the Gaussian blur applied.
    pub fn gaussian_blur(&self, radius: f32) -> Canvas<Rgb> {
        let width = self.width() as isize;
        let height = self.len() as isize;
        let mut blurred_image = self.clone();

        // Gaussian kernel
        let kernel_size = (2 * radius as isize + 1) as usize;
        let mut kernel = vec![vec![0.0; kernel_size]; kernel_size];
        let kernel_center = radius as isize;
        let sigma = radius / 2.0;
        let two_sigma_squared = 2.0 * sigma * sigma;
        let kernel_sum: f32 = (0..kernel_size)
            .map(|i| {
                (0..kernel_size)
                    .map(|j| {
                        let x = (i as isize - kernel_center) as f32;
                        let y = (j as isize - kernel_center) as f32;
                        let exponent = -(x * x + y * y) / two_sigma_squared;
                        let weight = (-exponent).exp();
                        kernel[i][j] = weight;
                        weight
                    })
                    .sum::<f32>()
            })
            .sum();

        // Normalize the kernel
        (0..kernel_size).for_each(|i| {
            for j in 0..kernel_size {
                kernel[i][j] /= kernel_sum;
            }
        });

        let apply_kernel = |x: isize, y: isize| -> (u8, u8, u8) {
            let mut red_sum = 0.0;
            let mut green_sum = 0.0;
            let mut blue_sum = 0.0;
            (0..kernel_size).for_each(|i| {
                for j in 0..kernel_size {
                    let dx = i as isize - kernel_center;
                    let dy = j as isize - kernel_center;
                    let px = x + dx;
                    let py = y + dy;
                    if px >= 0 && px < width && py >= 0 && py < height {
                        let pixel = self[(py * width + px) as usize];
                        let weight = kernel[i][j];
                        red_sum += weight * pixel.red as f32;
                        green_sum += weight * pixel.green as f32;
                        blue_sum += weight * pixel.blue as f32;
                    }
                }
            });
            (
                red_sum.round() as u8,
                green_sum.round() as u8,
                blue_sum.round() as u8,
            )
        };

        for y in 0..self.height() as isize {
            for x in 0..width {
                let (red, green, blue) = apply_kernel(x, y);
                blurred_image[(y * width + x) as usize] = Rgb::new(red, green, blue);
            }
        }

        blurred_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless,
        clippy::cast_precision_loss
    )]
    /// Applies the Emboss Filter to the current canvas and returns a new canvas with the effect.
    ///
    /// # Returns
    ///
    /// A new `Canvas<Rgb>` with the Emboss Filter effect applied.
    pub fn emboss(&self) -> Canvas<Rgb> {
        let mut embossed_image = self.clone();

        let width = self.width() as isize;
        let height = self.height() as isize;

        let emboss = |i: usize| -> (u8, u8, u8) {
            let (mut red, mut green, mut blue) = (0i16, 0i16, 0i16);
            let grid = Canvas::grid(i, width);
            let mut prev_color = self[i];
            for &index in &grid {
                if index >= 0 && index < height {
                    let current_color = self[index as usize];
                    red += current_color.red as i16 - prev_color.red as i16;
                    green += current_color.green as i16 - prev_color.green as i16;
                    blue += current_color.blue as i16 - prev_color.blue as i16;
                    prev_color = current_color;
                }
            }
            // Adjust values to fit within [0, 255] range
            red = red.clamp(-128, 127) + 128;
            green = green.clamp(-128, 127) + 128;
            blue = blue.clamp(-128, 127) + 128;
            (red as u8, green as u8, blue as u8)
        };

        embossed_image
            .iter_mut()
            .enumerate()
            .for_each(|(i, pixel)| {
                let (red, green, blue) = emboss(i);
                *pixel = Rgb::new(red, green, blue);
            });
        embossed_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless,
        clippy::cast_precision_loss
    )]
    /// Applies an Oil Painting Filter to the current canvas and returns a new canvas with the effect.
    ///
    /// # Returns
    ///
    /// A new `Canvas<Rgb>` with the Oil Painting Filter effect applied.
    pub fn oil_painting(&self) -> Canvas<Rgb> {
        let mut oiled_image = self.clone();

        let width = self.width() as isize;
        let height = self.height() as isize;

        let oil_painting = |i: usize| -> (u8, u8, u8) {
            let grid = Canvas::grid(i, width);
            let mut color_counts = vec![0; 256];
            let mut max_color = 0;
            let mut max_count = 0;
            for &index in &grid {
                if index >= 0 && index < width * height {
                    let pixel = self[index as usize];
                    let luminance = pixel.luminance();
                    color_counts[luminance as usize] += 1;
                    if color_counts[luminance as usize] > max_count {
                        max_count = color_counts[luminance as usize];
                        max_color = luminance;
                    }
                }
            }
            (max_color, max_color, max_color)
        };

        oiled_image.iter_mut().enumerate().for_each(|(i, pixel)| {
            let (red, green, blue) = oil_painting(i);
            *pixel = Rgb::new(red, green, blue);
        });
        oiled_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless,
        clippy::cast_precision_loss
    )]
    /// Applies a posterize filter to the current canvas and results in a new canvas
    ///
    /// * `levels`: The number of levels to posterize the image to
    pub fn posterize(&self, levels: u8) -> Canvas<Rgb> {
        let mut posterized_image = self.clone();

        let levels = levels as f32;

        posterized_image.iter_mut().for_each(|pixel| {
            let red = ((pixel.red as f32 / 255.0) * levels).round();
            let green = ((pixel.green as f32 / 255.0) * levels).round();
            let blue = ((pixel.blue as f32 / 255.0) * levels).round();

            pixel.red = ((red / levels) * 255.0).round() as u8;
            pixel.green = ((green / levels) * 255.0).round() as u8;
            pixel.blue = ((blue / levels) * 255.0).round() as u8;
        });

        posterized_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless,
        clippy::cast_precision_loss
    )]
    /// Applies a solarize filter to the current canvas and results in a new canvas
    ///
    /// * `threshold`: The threshold value to use to determine if a pixel is solarized
    pub fn solarize(&self, threshold: u8) -> Canvas<Rgb> {
        let mut solarized_image = self.clone();

        solarized_image.iter_mut().for_each(|pixel| {
            if pixel.red < threshold {
                pixel.red = 255 - pixel.red;
            }
            if pixel.green < threshold {
                pixel.green = 255 - pixel.green;
            }
            if pixel.blue < threshold {
                pixel.blue = 255 - pixel.blue;
            }
        });

        solarized_image
    }

    #[must_use]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_lossless,
        clippy::cast_precision_loss
    )]
    /// Applies a basic Watercolor Effect to the current canvas and returns a new canvas with the effect.
    ///
    /// # Returns
    ///
    /// A new `Canvas<Rgb>` with the Watercolor Effect applied.
    pub fn watercolor(&self) -> Canvas<Rgb> {
        let mut watercolor_image = self.clone();
        let width = self.width() as isize;
        let height = self.height() as isize;

        let radius = 3; // Adjust the radius to control the blending area

        let watercolor = |i: usize| -> Rgb {
            let mut red_sum = 0u32;
            let mut green_sum = 0u32;
            let mut blue_sum = 0u32;
            let mut count = 0u32;

            let (x, y) = (i as isize % width, i as isize / width);

            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    let nx = x + dx;
                    let ny = y + dy;

                    if nx >= 0 && ny >= 0 && nx < width && ny < height {
                        let index = ny * width + nx;
                        let pixel = self[index as usize];
                        red_sum += pixel.red as u32;
                        green_sum += pixel.green as u32;
                        blue_sum += pixel.blue as u32;
                        count += 1;
                    }
                }
            }

            let red_avg = (red_sum / count) as u8;
            let green_avg = (green_sum / count) as u8;
            let blue_avg = (blue_sum / count) as u8;
            Rgb::new(red_avg, green_avg, blue_avg)
        };

        watercolor_image
            .iter_mut()
            .enumerate()
            .for_each(|(i, pixel)| {
                *pixel = watercolor(i);
            });
        watercolor_image
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
