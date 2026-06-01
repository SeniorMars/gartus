use super::{
    colors::{Hsv, Rgb},
    display::Canvas,
};
use std::cmp::min;

impl Canvas {
    /// Helper to clamp a float to u8 range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn clamp_u8(val: f32) -> u8 {
        val.clamp(0.0, 255.0).round() as u8
    }

    fn idx(width: usize, x: usize, y: usize) -> usize {
        y * width + x
    }

    fn grayscale_values(&self) -> Vec<f32> {
        self.iter()
            .map(|pixel| f32::from(pixel.luminance()))
            .collect()
    }

    fn blank_like(&self) -> Canvas {
        self.with_pixels_like(vec![Rgb::default(); self.len()])
    }

    /// Generic convolution with a 3x3 kernel and additive bias.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn convolve_3x3(&self, kernel: [f32; 9], bias: f32) -> Canvas {
        let width = self.width() as isize;
        let height = self.height() as isize;
        let mut filtered = self.blank_like();

        for y in 0..height {
            for x in 0..width {
                let mut r_acc = 0.0;
                let mut g_acc = 0.0;
                let mut b_acc = 0.0;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let nx = x + (kx - 1);
                        let ny = y + (ky - 1);

                        // Edge handling: clamping to nearest pixel
                        let nx = nx.clamp(0, width - 1);
                        let ny = ny.clamp(0, height - 1);

                        let pixel = self[(ny * width + nx) as usize];
                        let weight = kernel[(ky * 3 + kx) as usize];

                        r_acc += f32::from(pixel.red) * weight;
                        g_acc += f32::from(pixel.green) * weight;
                        b_acc += f32::from(pixel.blue) * weight;
                    }
                }

                filtered[(y * width + x) as usize] = Rgb::new(
                    Self::clamp_u8(r_acc + bias),
                    Self::clamp_u8(g_acc + bias),
                    Self::clamp_u8(b_acc + bias),
                );
            }
        }
        filtered
    }

    /// Applies a sharpen filter.
    pub fn sharpen(&self) -> Canvas {
        let kernel = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
        self.convolve_3x3(kernel, 0.0)
    }

    #[allow(clippy::similar_names)]
    /// Prewitt edge detection.
    pub fn prewitt(&self) -> Canvas {
        let gx_kernel = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
        let gy_kernel = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        self.apply_edge_kernels(gx_kernel, gy_kernel)
    }

    #[allow(clippy::similar_names)]
    /// Scharr edge detection (more rotationally invariant than Sobel).
    pub fn scharr(&self) -> Canvas {
        let gx_kernel = [-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0];
        let gy_kernel = [-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0];
        self.apply_edge_kernels(gx_kernel, gy_kernel)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn apply_edge_kernels(&self, gx: [f32; 9], gy: [f32; 9]) -> Canvas {
        let x_img = self.convolve_3x3(gx, 0.0);
        let y_img = self.convolve_3x3(gy, 0.0);
        let mut result = self.blank_like();

        for i in 0..self.len() {
            let px = x_img[i];
            let py = y_img[i];
            result[i] = Rgb::new(
                (f32::from(px.red).powi(2) + f32::from(py.red).powi(2))
                    .sqrt()
                    .min(255.0) as u8,
                (f32::from(px.green).powi(2) + f32::from(py.green).powi(2))
                    .sqrt()
                    .min(255.0) as u8,
                (f32::from(px.blue).powi(2) + f32::from(py.blue).powi(2))
                    .sqrt()
                    .min(255.0) as u8,
            );
        }
        result
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    /// Gamma correction.
    pub fn gamma(&self, gamma: f32) -> Canvas {
        let inv_gamma = 1.0 / gamma;
        self.map_pixels(|pixel| {
            Rgb::new(
                (255.0 * (f32::from(pixel.red) / 255.0).powf(inv_gamma)).min(255.0) as u8,
                (255.0 * (f32::from(pixel.green) / 255.0).powf(inv_gamma)).min(255.0) as u8,
                (255.0 * (f32::from(pixel.blue) / 255.0).powf(inv_gamma)).min(255.0) as u8,
            )
        })
    }

    /// Adjust saturation.
    pub fn adjust_saturation(&self, factor: f32) -> Canvas {
        self.map_pixels(|pixel| {
            let gray = f32::from(pixel.luminance());
            Rgb::new(
                Self::clamp_u8(gray + factor * (f32::from(pixel.red) - gray)),
                Self::clamp_u8(gray + factor * (f32::from(pixel.green) - gray)),
                Self::clamp_u8(gray + factor * (f32::from(pixel.blue) - gray)),
            )
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    /// Rotates hue by given degrees.
    pub fn hue_rotate(&self, degrees: f32) -> Canvas {
        self.map_pixels(|pixel| {
            let mut hsv = Hsv::from(pixel);
            let new_hue = (f32::from(hsv.hue) + degrees).round() as i32;
            hsv.hue = ((new_hue % 360 + 360) % 360) as u16;
            Rgb::from(hsv)
        })
    }

    /// Adjust color temperature (positive = warmer, negative = cooler).
    pub fn adjust_temperature(&self, amount: i16) -> Canvas {
        self.map_pixels(|pixel| {
            Rgb::new(
                Self::clamp_u8(f32::from(pixel.red) + f32::from(amount)),
                pixel.green,
                Self::clamp_u8(f32::from(pixel.blue) - f32::from(amount)),
            )
        })
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    /// Vignette effect.
    pub fn vignette(&self, strength: f32) -> Canvas {
        let w = self.width() as f32;
        let h = self.height() as f32;
        let cx = w / 2.0;
        let cy = h / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();

        self.map_pixels_with_position(|x, y, pixel| {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt() / max_dist;
            let factor = 1.0 - (dist * strength).min(1.0);
            Rgb::new(
                (f32::from(pixel.red) * factor) as u8,
                (f32::from(pixel.green) * factor) as u8,
                (f32::from(pixel.blue) * factor) as u8,
            )
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    /// Pixelate / Mosaic effect.
    pub fn pixelate(&self, block_size: usize) -> Canvas {
        let mut result = self.blank_like();
        let w = self.width() as usize;
        let h = self.height() as usize;

        for by in (0..h).step_by(block_size) {
            for bx in (0..w).step_by(block_size) {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for y in by..min(by + block_size, h) {
                    for x in bx..min(bx + block_size, w) {
                        let p = self[y * w + x];
                        r_sum += u32::from(p.red);
                        g_sum += u32::from(p.green);
                        b_sum += u32::from(p.blue);
                        count += 1;
                    }
                }

                let avg = Rgb::new(
                    (r_sum / count) as u8,
                    (g_sum / count) as u8,
                    (b_sum / count) as u8,
                );

                for y in by..min(by + block_size, h) {
                    for x in bx..min(bx + block_size, w) {
                        result[y * w + x] = avg;
                    }
                }
            }
        }
        result
    }

    #[allow(clippy::cast_precision_loss)]
    /// Ordered dithering using a 4x4 Bayer matrix.
    pub fn ordered_dither(&self) -> Canvas {
        let bayer = [0, 12, 3, 15, 8, 4, 11, 7, 2, 14, 1, 13, 10, 6, 9, 5];
        self.map_pixels_with_position(|x, y, pixel| {
            let threshold =
                ((bayer[((y as usize % 4) * 4) + (x as usize % 4)] as f32 + 0.5) / 16.0) * 255.0;
            Rgb::new(
                if f32::from(pixel.red) > threshold {
                    255
                } else {
                    0
                },
                if f32::from(pixel.green) > threshold {
                    255
                } else {
                    0
                },
                if f32::from(pixel.blue) > threshold {
                    255
                } else {
                    0
                },
            )
        })
    }

    /// Floyd-Steinberg error-diffusion dithering.
    pub fn floyd_steinberg_dither(&self) -> Canvas {
        let width = self.width() as usize;
        let height = self.height() as usize;
        let mut values = self.grayscale_values();

        for y in 0..height {
            for x in 0..width {
                let idx = Self::idx(width, x, y);
                let old = values[idx];
                let new = if old >= 128.0 { 255.0 } else { 0.0 };
                let error = old - new;
                values[idx] = new;

                for (dx, dy, weight) in [
                    (1isize, 0isize, 7.0 / 16.0),
                    (-1, 1, 3.0 / 16.0),
                    (0, 1, 5.0 / 16.0),
                    (1, 1, 1.0 / 16.0),
                ] {
                    let nx = x.cast_signed() + dx;
                    let ny = y.cast_signed() + dy;
                    if nx >= 0 && nx < width.cast_signed() && ny >= 0 && ny < height.cast_signed() {
                        #[allow(clippy::cast_sign_loss)]
                        let nidx = Self::idx(width, nx as usize, ny as usize);
                        values[nidx] = (values[nidx] + error * weight).clamp(0.0, 255.0);
                    }
                }
            }
        }

        let pixels = values
            .into_iter()
            .map(|value| {
                let value = Self::clamp_u8(value);
                Rgb::new(value, value, value)
            })
            .collect();
        self.with_pixels_like(pixels)
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    /// Median filter for noise reduction.
    pub fn median_filter(&self, radius: usize) -> Canvas {
        let mut result = self.blank_like();
        let w = self.width() as isize;
        let h = self.height() as isize;
        let window_len = (2 * radius + 1).pow(2);
        let mut rs = Vec::with_capacity(window_len);
        let mut gs = Vec::with_capacity(window_len);
        let mut bs = Vec::with_capacity(window_len);

        for y in 0..h {
            for x in 0..w {
                rs.clear();
                gs.clear();
                bs.clear();

                for dy in -(radius as isize)..=(radius as isize) {
                    for dx in -(radius as isize)..=(radius as isize) {
                        let nx = (x + dx).clamp(0, w - 1);
                        let ny = (y + dy).clamp(0, h - 1);
                        let p = self[(ny * w + nx) as usize];
                        rs.push(p.red);
                        gs.push(p.green);
                        bs.push(p.blue);
                    }
                }

                rs.sort_unstable();
                gs.sort_unstable();
                bs.sort_unstable();

                let mid = rs.len() / 2;
                result[(y * w + x) as usize] = Rgb::new(rs[mid], gs[mid], bs[mid]);
            }
        }
        result
    }

    /// Applies an Oil Painting Filter.
    pub fn oil_painting(&self) -> Canvas {
        self.oil_painting_custom(3, 32)
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    /// Applies an oil-painting effect using local intensity buckets.
    ///
    /// # Panics
    ///
    /// Panics if `levels` is 0.
    pub fn oil_painting_custom(&self, radius: usize, levels: usize) -> Canvas {
        assert!(levels > 0, "oil painting levels must be positive");
        let mut result = self.blank_like();
        let w = self.width() as isize;
        let h = self.height() as isize;
        let radius = radius as isize;
        let mut counts = vec![0u32; levels];
        let mut red_sums = vec![0u32; levels];
        let mut green_sums = vec![0u32; levels];
        let mut blue_sums = vec![0u32; levels];

        for y in 0..h {
            for x in 0..w {
                counts.fill(0);
                red_sums.fill(0);
                green_sums.fill(0);
                blue_sums.fill(0);

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = (x + dx).clamp(0, w - 1);
                        let ny = (y + dy).clamp(0, h - 1);
                        let p = self[(ny * w + nx) as usize];
                        let bucket = (usize::from(p.luminance()) * levels / 256).min(levels - 1);
                        counts[bucket] += 1;
                        red_sums[bucket] += u32::from(p.red);
                        green_sums[bucket] += u32::from(p.green);
                        blue_sums[bucket] += u32::from(p.blue);
                    }
                }
                let bucket = counts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, count)| *count)
                    .map_or(0, |(bucket, _)| bucket);
                let count = counts[bucket].max(1);
                result[(y * w + x) as usize] = Rgb::new(
                    (red_sums[bucket] / count) as u8,
                    (green_sums[bucket] / count) as u8,
                    (blue_sums[bucket] / count) as u8,
                );
            }
        }
        result
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    /// Applies a Watercolor Effect.
    pub fn watercolor(&self) -> Canvas {
        let mut result = self.blank_like();
        let w = self.width() as isize;
        let h = self.height() as isize;
        let radius = 3;

        for y in 0..h {
            for x in 0..w {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = (x + dx).clamp(0, w - 1);
                        let ny = (y + dy).clamp(0, h - 1);
                        let p = self[(ny * w + nx) as usize];
                        r_sum += u32::from(p.red);
                        g_sum += u32::from(p.green);
                        b_sum += u32::from(p.blue);
                        count += 1;
                    }
                }
                result[(y * w + x) as usize] = Rgb::new(
                    (r_sum / count) as u8,
                    (g_sum / count) as u8,
                    (b_sum / count) as u8,
                );
            }
        }
        result
    }

    // --- REFACTORED EXISTING FILTERS ---

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    /// Converts the image to grayscale.
    pub fn grayscale(&self) -> Canvas {
        self.map_pixels(|pixel| {
            let (r, g, b) = pixel.values();
            let average = ((f32::from(r) + f32::from(g) + f32::from(b)) / 3.0).round() as u8;
            Rgb::new(average, average, average)
        })
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    /// Applies a sepia tone filter.
    pub fn sepia(&self) -> Canvas {
        self.map_pixels(|pixel| {
            let (r, g, b) = pixel.values();
            let sepia_red = (0.393 * f32::from(r) + 0.769 * f32::from(g) + 0.198 * f32::from(b))
                .min(255.0) as u8;
            let sepia_green = (0.349 * f32::from(r) + 0.686 * f32::from(g) + 0.168 * f32::from(b))
                .min(255.0) as u8;
            let sepia_blue = (0.272 * f32::from(r) + 0.534 * f32::from(g) + 0.131 * f32::from(b))
                .min(255.0) as u8;
            Rgb::new(sepia_red, sepia_green, sepia_blue)
        })
    }

    /// Reflects the image horizontally.
    pub fn reflect(&self) -> Canvas {
        let mut pixels = Vec::with_capacity(self.len());
        for row in self.iter_row() {
            pixels.extend(row.iter().rev().copied());
        }
        self.with_pixels_like(pixels)
    }

    /// Applies a box blur filter.
    pub fn blur(&self) -> Canvas {
        let kernel = [1.0 / 9.0; 9];
        self.convolve_3x3(kernel, 0.0)
    }

    #[allow(clippy::similar_names)]
    /// Sobel edge detection.
    pub fn sobel(&self) -> Canvas {
        let gx_kernel = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let gy_kernel = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
        self.apply_edge_kernels(gx_kernel, gy_kernel)
    }

    /// Inverts all pixel colors.
    pub fn invert(&self) -> Canvas {
        self.map_pixels(|pixel| {
            let (r, g, b) = pixel.values();
            Rgb::new(255 - r, 255 - g, 255 - b)
        })
    }

    /// Converts to black and white using a luminance threshold.
    pub fn black_and_white(&self, threshold: u8) -> Canvas {
        self.map_pixels(|pixel| {
            if pixel.luminance() >= threshold {
                Rgb::WHITE
            } else {
                Rgb::BLACK
            }
        })
    }

    /// Adjusts brightness by adding a signed offset to each channel.
    pub fn adjust_brightness(&self, brightness: i16) -> Canvas {
        self.map_pixels(|pixel| {
            Rgb::new(
                Self::clamp_u8(f32::from(pixel.red) + f32::from(brightness)),
                Self::clamp_u8(f32::from(pixel.green) + f32::from(brightness)),
                Self::clamp_u8(f32::from(pixel.blue) + f32::from(brightness)),
            )
        })
    }

    /// Adjusts contrast by scaling each channel around the midpoint.
    pub fn adjust_contrast(&self, contrast: f32) -> Canvas {
        self.map_pixels(|pixel| {
            Rgb::new(
                Self::clamp_u8((f32::from(pixel.red) - 127.5) * contrast + 127.5),
                Self::clamp_u8((f32::from(pixel.green) - 127.5) * contrast + 127.5),
                Self::clamp_u8((f32::from(pixel.blue) - 127.5) * contrast + 127.5),
            )
        })
    }

    /// Laplacian edge detection.
    pub fn laplacian_edge_detection(&self) -> Canvas {
        let kernel = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];
        self.convolve_3x3(kernel, 0.0)
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    /// Gaussian blur with the given radius.
    ///
    /// # Panics
    ///
    /// Panics if `radius` is not finite or is less than 1.0.
    pub fn gaussian_blur(&self, radius: f32) -> Canvas {
        assert!(
            radius.is_finite() && radius >= 1.0,
            "gaussian blur radius must be finite and at least 1.0"
        );
        if self.is_empty() {
            return self.blank_like();
        }

        let radius = radius.round() as isize;
        let width = self.width() as usize;
        let height = self.height() as usize;
        let sigma = (radius as f32 / 2.0).max(1.0);
        let two_sigma_squared = 2.0 * sigma * sigma;
        let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
        let mut kernel_sum = 0.0;
        for i in -radius..=radius {
            let distance = i as f32;
            let weight = (-(distance * distance) / two_sigma_squared).exp();
            kernel.push(weight);
            kernel_sum += weight;
        }
        for weight in &mut kernel {
            *weight /= kernel_sum;
        }

        let mut horizontal = vec![[0.0f32; 3]; self.len()];
        for y in 0..height {
            for x in 0..width {
                let mut r_sum = 0.0;
                let mut g_sum = 0.0;
                let mut b_sum = 0.0;
                for (kernel_idx, weight) in kernel.iter().enumerate() {
                    let dx = kernel_idx as isize - radius;
                    let px = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                    let pixel = self[Self::idx(width, px, y)];
                    r_sum += weight * f32::from(pixel.red);
                    g_sum += weight * f32::from(pixel.green);
                    b_sum += weight * f32::from(pixel.blue);
                }
                horizontal[Self::idx(width, x, y)] = [r_sum, g_sum, b_sum];
            }
        }

        let mut pixels = Vec::with_capacity(self.len());
        for y in 0..height {
            for x in 0..width {
                let mut r_sum = 0.0;
                let mut g_sum = 0.0;
                let mut b_sum = 0.0;
                for (kernel_idx, weight) in kernel.iter().enumerate() {
                    let dy = kernel_idx as isize - radius;
                    let py = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                    let pixel = horizontal[Self::idx(width, x, py)];
                    r_sum += weight * pixel[0];
                    g_sum += weight * pixel[1];
                    b_sum += weight * pixel[2];
                }
                pixels.push(Rgb::new(
                    Self::clamp_u8(r_sum),
                    Self::clamp_u8(g_sum),
                    Self::clamp_u8(b_sum),
                ));
            }
        }

        self.with_pixels_like(pixels)
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    /// Bilateral filter that smooths noise while preserving strong color edges.
    ///
    /// # Panics
    ///
    /// Panics if `radius` is 0, `sigma_space` is not positive and finite, or
    /// `sigma_color` is not positive and finite.
    pub fn bilateral_filter(&self, radius: usize, sigma_space: f32, sigma_color: f32) -> Canvas {
        assert!(radius > 0, "bilateral radius must be positive");
        assert!(
            sigma_space.is_finite() && sigma_space > 0.0,
            "bilateral sigma_space must be positive and finite"
        );
        assert!(
            sigma_color.is_finite() && sigma_color > 0.0,
            "bilateral sigma_color must be positive and finite"
        );

        let width = self.width() as isize;
        let height = self.height() as isize;
        let radius = radius as isize;
        let two_sigma_space_squared = 2.0 * sigma_space * sigma_space;
        let two_sigma_color_squared = 2.0 * sigma_color * sigma_color;
        let mut result = self.blank_like();

        for y in 0..height {
            for x in 0..width {
                let center = self[(y * width + x) as usize];
                let mut r_sum = 0.0;
                let mut g_sum = 0.0;
                let mut b_sum = 0.0;
                let mut weight_sum = 0.0;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = (x + dx).clamp(0, width - 1);
                        let ny = (y + dy).clamp(0, height - 1);
                        let pixel = self[(ny * width + nx) as usize];
                        let spatial_distance = (dx * dx + dy * dy) as f32;
                        let color_distance = (f32::from(pixel.red) - f32::from(center.red)).powi(2)
                            + (f32::from(pixel.green) - f32::from(center.green)).powi(2)
                            + (f32::from(pixel.blue) - f32::from(center.blue)).powi(2);
                        let weight = (-spatial_distance / two_sigma_space_squared).exp()
                            * (-color_distance / two_sigma_color_squared).exp();

                        r_sum += f32::from(pixel.red) * weight;
                        g_sum += f32::from(pixel.green) * weight;
                        b_sum += f32::from(pixel.blue) * weight;
                        weight_sum += weight;
                    }
                }

                result[(y * width + x) as usize] = Rgb::new(
                    Self::clamp_u8(r_sum / weight_sum),
                    Self::clamp_u8(g_sum / weight_sum),
                    Self::clamp_u8(b_sum / weight_sum),
                );
            }
        }
        result
    }

    /// Unsharp-mask sharpening: original + amount * (original - blurred).
    ///
    /// # Panics
    ///
    /// Panics if `amount` is not finite or is negative.
    pub fn unsharp_mask(&self, radius: f32, amount: f32) -> Canvas {
        assert!(
            amount.is_finite() && amount >= 0.0,
            "unsharp amount must be finite and non-negative"
        );
        let blurred = self.gaussian_blur(radius);
        let mut pixels = Vec::with_capacity(self.len());
        for idx in 0..self.len() {
            let original = self[idx];
            let blur = blurred[idx];
            pixels.push(Rgb::new(
                Self::clamp_u8(
                    f32::from(original.red)
                        + amount * (f32::from(original.red) - f32::from(blur.red)),
                ),
                Self::clamp_u8(
                    f32::from(original.green)
                        + amount * (f32::from(original.green) - f32::from(blur.green)),
                ),
                Self::clamp_u8(
                    f32::from(original.blue)
                        + amount * (f32::from(original.blue) - f32::from(blur.blue)),
                ),
            ));
        }
        self.with_pixels_like(pixels)
    }

    #[allow(clippy::cast_precision_loss)]
    /// Global luminance histogram equalization.
    pub fn histogram_equalization(&self) -> Canvas {
        let mut hist = [0usize; 256];
        for pixel in self {
            hist[usize::from(pixel.luminance())] += 1;
        }
        let total = self.len();
        let cdf_min = hist.iter().copied().find(|count| *count > 0).unwrap_or(0);
        if total == cdf_min {
            return self.clone();
        }

        let mut cumulative = 0usize;
        let mut lut = [0u8; 256];
        for (idx, count) in hist.iter().enumerate() {
            cumulative += count;
            lut[idx] = Self::clamp_u8(
                ((cumulative.saturating_sub(cdf_min)) as f32 / (total - cdf_min) as f32) * 255.0,
            );
        }

        self.apply_luminance_lut(&lut)
    }

    #[allow(clippy::similar_names)]
    /// Contrast-limited adaptive histogram equalization over square tiles.
    ///
    /// # Panics
    ///
    /// Panics if `tile_size` is 0.
    pub fn clahe(&self, tile_size: usize, clip_limit: usize) -> Canvas {
        assert!(tile_size > 0, "CLAHE tile size must be positive");
        let width = self.width() as usize;
        let height = self.height() as usize;
        let tiles_x = width.div_ceil(tile_size);
        let tiles_y = height.div_ceil(tile_size);
        let mut luts = vec![[0u8; 256]; tiles_x * tiles_y];

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let x0 = tile_x * tile_size;
                let y0 = tile_y * tile_size;
                let x1 = min(x0 + tile_size, width);
                let y1 = min(y0 + tile_size, height);
                luts[Self::idx(tiles_x, tile_x, tile_y)] =
                    self.clahe_tile_lut(x0, y0, x1, y1, clip_limit);
            }
        }

        let mut result = self.blank_like();
        for y in 0..height {
            for x in 0..width {
                let tile_x = (x / tile_size).min(tiles_x - 1);
                let tile_y = (y / tile_size).min(tiles_y - 1);
                let lut = &luts[Self::idx(tiles_x, tile_x, tile_y)];
                result[Self::idx(width, x, y)] =
                    Self::equalized_luminance_pixel(self[Self::idx(width, x, y)], lut);
            }
        }
        result
    }

    #[allow(clippy::cast_precision_loss)]
    fn clahe_tile_lut(
        &self,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        clip_limit: usize,
    ) -> [u8; 256] {
        let width = self.width() as usize;
        let mut hist = [0usize; 256];
        for y in y0..y1 {
            for x in x0..x1 {
                hist[usize::from(self[Self::idx(width, x, y)].luminance())] += 1;
            }
        }

        if clip_limit > 0 {
            let mut clipped = 0usize;
            for count in &mut hist {
                if *count > clip_limit {
                    clipped += *count - clip_limit;
                    *count = clip_limit;
                }
            }
            let redistribute = clipped / 256;
            let mut remainder = clipped % 256;
            for count in &mut hist {
                *count += redistribute;
                if remainder > 0 {
                    *count += 1;
                    remainder -= 1;
                }
            }
        }

        let area = (x1 - x0) * (y1 - y0);
        let cdf_min = hist.iter().copied().find(|count| *count > 0).unwrap_or(0);
        let mut cumulative = 0usize;
        let mut lut = [0u8; 256];
        if area == cdf_min {
            return lut;
        }
        for (idx, count) in hist.iter().enumerate() {
            cumulative += count;
            lut[idx] = Self::clamp_u8(
                ((cumulative.saturating_sub(cdf_min)) as f32 / (area - cdf_min) as f32) * 255.0,
            );
        }
        lut
    }

    fn apply_luminance_lut(&self, lut: &[u8; 256]) -> Canvas {
        self.map_pixels(|pixel| Self::equalized_luminance_pixel(pixel, lut))
    }

    fn equalized_luminance_pixel(pixel: Rgb, lut: &[u8; 256]) -> Rgb {
        let old_luma = pixel.luminance().max(1);
        let new_luma = lut[usize::from(old_luma)];
        let scale = f32::from(new_luma) / f32::from(old_luma);
        Rgb::new(
            Self::clamp_u8(f32::from(pixel.red) * scale),
            Self::clamp_u8(f32::from(pixel.green) * scale),
            Self::clamp_u8(f32::from(pixel.blue) * scale),
        )
    }

    /// Canny-style edge detector with Gaussian blur, non-maximum suppression, and hysteresis.
    ///
    /// # Panics
    ///
    /// Panics if `low_threshold` is greater than `high_threshold`.
    pub fn canny(&self, low_threshold: u8, high_threshold: u8) -> Canvas {
        assert!(
            low_threshold <= high_threshold,
            "Canny low threshold must be <= high threshold"
        );
        let blurred = self.gaussian_blur(1.0);
        let width = self.width() as usize;
        let height = self.height() as usize;
        let gray = blurred.grayscale_values();
        let (magnitude, direction) = sobel_magnitude_direction(&gray, width, height);
        let suppressed = non_maximum_suppression(&magnitude, &direction, width, height);
        let edges = hysteresis(
            &suppressed,
            width,
            height,
            f32::from(low_threshold),
            f32::from(high_threshold),
        );

        self.with_pixels_like(
            edges
                .into_iter()
                .map(|edge| if edge { Rgb::WHITE } else { Rgb::BLACK })
                .collect(),
        )
    }

    /// Emboss filter.
    pub fn emboss(&self) -> Canvas {
        let kernel = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        self.convolve_3x3(kernel, 128.0)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    /// Posterize to a given number of color levels.
    pub fn posterize(&self, levels: u8) -> Canvas {
        let levels = f32::from(levels.max(1));
        self.map_pixels(|pixel| {
            Rgb::new(
                ((f32::from(pixel.red) / 255.0 * levels).round() / levels * 255.0) as u8,
                ((f32::from(pixel.green) / 255.0 * levels).round() / levels * 255.0) as u8,
                ((f32::from(pixel.blue) / 255.0 * levels).round() / levels * 255.0) as u8,
            )
        })
    }

    /// Solarize channels above the given threshold.
    pub fn solarize(&self, threshold: u8) -> Canvas {
        self.map_pixels(|pixel| {
            Rgb::new(
                if pixel.red > threshold {
                    255 - pixel.red
                } else {
                    pixel.red
                },
                if pixel.green > threshold {
                    255 - pixel.green
                } else {
                    pixel.green
                },
                if pixel.blue > threshold {
                    255 - pixel.blue
                } else {
                    pixel.blue
                },
            )
        })
    }
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
fn sobel_magnitude_direction(gray: &[f32], width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
    let mut magnitude = vec![0.0; gray.len()];
    let mut direction = vec![0.0; gray.len()];
    let gx_kernel = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let gy_kernel = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    for y in 0..height {
        for x in 0..width {
            let mut gx = 0.0;
            let mut gy = 0.0;
            for ky in 0..3 {
                for kx in 0..3 {
                    let nx = (x as isize + kx as isize - 1).clamp(0, width as isize - 1) as usize;
                    let ny = (y as isize + ky as isize - 1).clamp(0, height as isize - 1) as usize;
                    let value = gray[Canvas::idx(width, nx, ny)];
                    gx += value * gx_kernel[ky * 3 + kx];
                    gy += value * gy_kernel[ky * 3 + kx];
                }
            }
            let idx = Canvas::idx(width, x, y);
            magnitude[idx] = (gx * gx + gy * gy).sqrt();
            direction[idx] = gy.atan2(gx).to_degrees();
        }
    }

    (magnitude, direction)
}

fn non_maximum_suppression(
    magnitude: &[f32],
    direction: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; magnitude.len()];
    if width < 3 || height < 3 {
        return out;
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = Canvas::idx(width, x, y);
            let mut angle = direction[idx];
            if angle < 0.0 {
                angle += 180.0;
            }

            let (q, r) = if (0.0..22.5).contains(&angle) || (157.5..=180.0).contains(&angle) {
                (
                    magnitude[Canvas::idx(width, x + 1, y)],
                    magnitude[Canvas::idx(width, x - 1, y)],
                )
            } else if (22.5..67.5).contains(&angle) {
                (
                    magnitude[Canvas::idx(width, x + 1, y - 1)],
                    magnitude[Canvas::idx(width, x - 1, y + 1)],
                )
            } else if (67.5..112.5).contains(&angle) {
                (
                    magnitude[Canvas::idx(width, x, y - 1)],
                    magnitude[Canvas::idx(width, x, y + 1)],
                )
            } else {
                (
                    magnitude[Canvas::idx(width, x - 1, y - 1)],
                    magnitude[Canvas::idx(width, x + 1, y + 1)],
                )
            };

            if magnitude[idx] >= q && magnitude[idx] >= r {
                out[idx] = magnitude[idx];
            }
        }
    }
    out
}

fn hysteresis(
    magnitude: &[f32],
    width: usize,
    height: usize,
    low_threshold: f32,
    high_threshold: f32,
) -> Vec<bool> {
    let mut edges = vec![false; magnitude.len()];
    let mut stack = Vec::new();

    for (idx, value) in magnitude.iter().enumerate() {
        if *value >= high_threshold {
            edges[idx] = true;
            stack.push(idx);
        }
    }

    while let Some(idx) = stack.pop() {
        let x = idx % width;
        let y = idx / width;
        let x0 = x.saturating_sub(1);
        let y0 = y.saturating_sub(1);
        let x1 = min(x + 1, width - 1);
        let y1 = min(y + 1, height - 1);

        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let nidx = Canvas::idx(width, nx, ny);
                if !edges[nidx] && magnitude[nidx] >= low_threshold {
                    edges[nidx] = true;
                    stack.push(nidx);
                }
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use crate::graphics::{colors::Rgb, display::Canvas};

    #[test]
    fn solarize_inverts_channels_above_threshold() {
        let mut canvas = Canvas::new(1, 1, Rgb::BLACK);
        canvas.fill_canvas(vec![Rgb::new(100, 150, 200)]);

        let solarized = canvas.solarize(128);

        assert_eq!(solarized.pixels(), &[Rgb::new(100, 105, 55)]);
    }

    #[test]
    fn ordered_dither_uses_lowest_bayer_cell_for_dark_pixels() {
        let mut canvas = Canvas::new(1, 1, Rgb::BLACK);
        canvas.fill_canvas(vec![Rgb::new(7, 8, 9)]);

        let dithered = canvas.ordered_dither();

        assert_eq!(dithered.pixels(), &[Rgb::new(0, 255, 255)]);
    }

    #[test]
    #[should_panic(expected = "gaussian blur radius must be finite and at least 1.0")]
    fn gaussian_blur_rejects_subpixel_radius() {
        let _ = Canvas::new(1, 1, Rgb::BLACK).gaussian_blur(0.5);
    }

    #[test]
    fn histogram_equalization_stretches_luminance_range() {
        let mut canvas = Canvas::new(2, 1, Rgb::BLACK);
        canvas.fill_canvas(vec![Rgb::new(10, 10, 10), Rgb::new(20, 20, 20)]);

        let equalized = canvas.histogram_equalization();

        assert_eq!(equalized.pixels(), &[Rgb::BLACK, Rgb::WHITE]);
    }

    #[test]
    fn bilateral_filter_preserves_hard_edges_better_than_gaussian() {
        let mut canvas = Canvas::new(3, 1, Rgb::BLACK);
        canvas.fill_canvas(vec![Rgb::BLACK, Rgb::BLACK, Rgb::WHITE]);

        let bilateral = canvas.bilateral_filter(1, 1.0, 8.0);

        assert_eq!(bilateral.pixels()[1], Rgb::BLACK);
    }

    #[test]
    fn unsharp_mask_preserves_flat_image() {
        let mut canvas = Canvas::new(3, 1, Rgb::BLACK);
        canvas.fill_canvas(vec![Rgb::new(120, 120, 120); 3]);

        let sharpened = canvas.unsharp_mask(1.0, 1.0);

        assert_eq!(sharpened.pixels(), canvas.pixels());
    }

    #[test]
    fn canny_returns_binary_edges() {
        let mut canvas = Canvas::new(3, 3, Rgb::BLACK);
        canvas.fill_canvas(vec![
            Rgb::BLACK,
            Rgb::WHITE,
            Rgb::WHITE,
            Rgb::BLACK,
            Rgb::WHITE,
            Rgb::WHITE,
            Rgb::BLACK,
            Rgb::WHITE,
            Rgb::WHITE,
        ]);

        let edges = canvas.canny(10, 20);

        assert!(
            edges
                .pixels()
                .iter()
                .all(|pixel| *pixel == Rgb::BLACK || *pixel == Rgb::WHITE)
        );
    }
}
