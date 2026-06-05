//! Small 2D drawing primitives and bitmap text.

use crate::graphics::{colors::Rgb, display::Canvas};

const GLYPH_WIDTH: i64 = 5;
const GLYPH_HEIGHT: i64 = 7;
const GLYPH_ADVANCE: i64 = 6;

impl Canvas {
    /// Fills an axis-aligned rectangle.
    pub fn fill_rect(&mut self, x: i64, y: i64, width: i64, height: i64, color: Rgb) {
        if width <= 0 || height <= 0 {
            return;
        }

        let Some(x_end) = x.checked_add(width) else {
            return;
        };
        let Some(y_end) = y.checked_add(height) else {
            return;
        };

        for py in y..y_end {
            for px in x..x_end {
                self.plot(&color, px, py);
            }
        }
    }

    /// Draws an axis-aligned rectangle outline.
    pub fn draw_rect(&mut self, left: f64, bottom: f64, right: f64, top: f64, color: Rgb) {
        self.draw_line(color, left, bottom, right, bottom);
        self.draw_line(color, right, bottom, right, top);
        self.draw_line(color, right, top, left, top);
        self.draw_line(color, left, top, left, bottom);
    }

    /// Fills a 2D disc using the current canvas coordinate system.
    pub fn fill_disc(&mut self, cx: i64, cy: i64, radius: i64, color: Rgb) {
        if radius < 0 {
            return;
        }

        let radius_sq = i128::from(radius) * i128::from(radius);
        for y in cy - radius..=cy + radius {
            for x in cx - radius..=cx + radius {
                let dx = i128::from(x - cx);
                let dy = i128::from(y - cy);
                if dx * dx + dy * dy <= radius_sq {
                    self.plot(&color, x, y);
                }
            }
        }
    }

    /// Returns the bitmap text dimensions for `text` at `scale`.
    ///
    /// Glyphs are 5x7 pixels with a one-pixel advance gap between glyphs.
    #[must_use]
    pub fn text_dimensions(text: &str, scale: u32) -> (i64, i64) {
        let scale = i64::from(scale);
        if scale == 0 {
            return (0, 0);
        }

        let glyph_count = i64::try_from(text.chars().count()).unwrap_or(i64::MAX / GLYPH_ADVANCE);
        if glyph_count == 0 {
            return (0, 0);
        }

        (
            (glyph_count * GLYPH_ADVANCE - 1) * scale,
            GLYPH_HEIGHT * scale,
        )
    }

    /// Draws 5x7 bitmap text with `(x, y)` as the lower-left corner.
    ///
    /// Lowercase ASCII letters are rendered as uppercase. Unsupported characters render as blank
    /// spaces so layout remains stable.
    pub fn draw_text(&mut self, text: &str, x: i64, y: i64, scale: u32, color: Rgb) {
        let scale = i64::from(scale);
        if scale == 0 {
            return;
        }

        for (index, ch) in text.chars().enumerate() {
            let Some(pattern) = glyph_pattern(ch) else {
                continue;
            };
            let x_offset = i64::try_from(index)
                .unwrap_or(i64::MAX / GLYPH_ADVANCE)
                .saturating_mul(GLYPH_ADVANCE * scale);
            for (row, bits) in pattern.iter().enumerate() {
                let Ok(row) = i64::try_from(row) else {
                    continue;
                };
                for col in 0..GLYPH_WIDTH {
                    if bits & (1 << (GLYPH_WIDTH - 1 - col)) != 0 {
                        self.fill_rect(
                            x + x_offset + col * scale,
                            y + (GLYPH_HEIGHT - 1 - row) * scale,
                            scale,
                            scale,
                            color,
                        );
                    }
                }
            }
        }
    }

    /// Draws 5x7 bitmap text centered around `(cx, cy)`.
    pub fn draw_text_centered(&mut self, text: &str, cx: i64, cy: i64, scale: u32, color: Rgb) {
        let (width, height) = Self::text_dimensions(text, scale);
        self.draw_text(text, cx - width / 2, cy - height / 2, scale, color);
    }
}

fn glyph_pattern(ch: char) -> Option<[u8; 7]> {
    punctuation_pattern(ch)
        .or_else(|| digit_pattern(ch))
        .or_else(|| letter_pattern(ch))
}

fn punctuation_pattern(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        ' ' => [0; 7],
        '!' => [
            0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110,
        ],
        ':' => [
            0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00110, 0b00000,
        ],
        '/' => [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
        '_' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
        ],
        _ => return None,
    })
}

fn digit_pattern(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        _ => return None,
    })
}

fn letter_pattern(ch: char) -> Option<[u8; 7]> {
    Some(match ch.to_ascii_uppercase() {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_dimensions_use_advance_spacing() {
        assert_eq!(Canvas::text_dimensions("", 3), (0, 0));
        assert_eq!(Canvas::text_dimensions("A", 2), (10, 14));
        assert_eq!(Canvas::text_dimensions("AB", 2), (22, 14));
    }

    #[test]
    fn draw_text_renders_lowercase_as_uppercase() {
        let mut upper = Canvas::new_with_bg(8, 8, Rgb::WHITE);
        let mut lower = Canvas::new_with_bg(8, 8, Rgb::WHITE);

        upper.draw_text("A", 1, 0, 1, Rgb::BLACK);
        lower.draw_text("a", 1, 0, 1, Rgb::BLACK);

        assert_eq!(upper.pixels(), lower.pixels());
        assert_eq!(upper.get_pixel(2, 6), Some(&Rgb::BLACK));
        assert_eq!(upper.get_pixel(1, 0), Some(&Rgb::BLACK));
    }

    #[test]
    fn fill_rect_ignores_nonpositive_dimensions() {
        let mut canvas = Canvas::new_with_bg(4, 4, Rgb::WHITE);

        canvas.fill_rect(0, 0, 0, 2, Rgb::BLACK);
        canvas.fill_rect(0, 0, 2, -1, Rgb::BLACK);

        assert!(canvas.pixels().iter().all(|pixel| *pixel == Rgb::WHITE));
    }

    #[test]
    fn fill_disc_handles_zero_radius() {
        let mut canvas = Canvas::new_with_bg(5, 5, Rgb::WHITE);

        canvas.fill_disc(2, 2, 0, Rgb::BLACK);

        assert_eq!(canvas.get_pixel(2, 2), Some(&Rgb::BLACK));
        assert_eq!(
            canvas
                .pixels()
                .iter()
                .filter(|pixel| **pixel == Rgb::BLACK)
                .count(),
            1
        );
    }
}
