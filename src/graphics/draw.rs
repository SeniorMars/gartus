use crate::graphics::display::Canvas;
use crate::graphics::display::Pixel;
use crate::graphics::matrix::Matrix;
use std::io;

// I'm unsure how I want to structure my project
#[allow(dead_code)]
impl Canvas {
    pub fn fill(&mut self, x: i32, y: i32, fill_color: Pixel, boundary_color: Pixel) {
        let current = self.get_pixel(x, y);
        if current != boundary_color && current != fill_color {
            self.plot(fill_color, x as i32, y as i32);
            self.fill(x + 1, y, fill_color, boundary_color);
            self.fill(x, y + 1, fill_color, boundary_color);
            self.fill(x - 1, y, fill_color, boundary_color);
            self.fill(x, y - 1, fill_color, boundary_color);
            // self.fill(x + 1, y, fill_color, boundary_color);
            // self.fill(x, y + 1, fill_color, boundary_color);
            // self.fill(x - 1, y, fill_color, boundary_color);
            // self.fill(x, y - 1, fill_color, boundary_color);
            // self.fill(x - 1, y - 1, fill_color, boundary_color);
            // self.fill(x - 1, y + 1, fill_color, boundary_color);
            // self.fill(x + 1, y - 1, fill_color, boundary_color);
            // self.fill(x + 1, y + 1, fill_color, boundary_color);
        }
    }

    pub fn draw_lines(&mut self, matrix: &Matrix) {
        let mut iter = matrix.iter_by_point();
        while let Some(point) = iter.next() {
            let (x0, y0, _z0) = (point[0], point[1], point[3]);
            let (x1, y1, _z1) = match iter.next() {
                Some(p1) => (p1[0], p1[1], p1[2]),
                None => panic!("Need at least 2 points to draw"),
            };

            self.draw_line(self.line, x0, y0, x1, y1);
        }
    }

    pub fn draw_lines_for_animation(&mut self, matrix: &Matrix, filename: &str) -> io::Result<()> {
        let mut iter = matrix.iter_by_point();
        while let Some(point) = iter.next() {
            let (x0, y0, _z0) = (point[0], point[1], point[3]);
            let (x1, y1, _z1) = match iter.next() {
                Some(p1) => (p1[0], p1[1], p1[2]),
                None => panic!("Need at least 2 points to draw"),
            };

            self.save_binary(&format!("anim/{}{:08}.ppm", filename, self.anim_index))?;
            self.draw_line(self.line, x0, y0, x1, y1);
        }
        self.save_binary(&format!("anim/{}{:08}.ppm", filename, self.anim_index))
    }

    pub fn draw_line(&mut self, color: Pixel, x0: f64, y0: f64, x1: f64, y1: f64) {
        self.anim_index += 1;
        let (x0, y0, x1, y1) = if x0 > x1 {
            (x1, y1, x0, y0)
        } else {
            (x0, y0, x1, y1)
        };
        let (mut x0, mut y0, x1, y1) = (
            x0.round() as i32,
            y0.round() as i32,
            x1.round() as i32,
            y1.round() as i32,
        );
        let (delta_y, delta_x) = (2 * (y1 - y0), -2 * (x1 - x0));

        if (x1 - x0).abs() >= (y1 - y0).abs() {
            if delta_y > 0 {
                // octant 1
                let mut d = delta_y + delta_x / 2;
                for x in x0..=x1 {
                    self.plot(color, x, y0);
                    if d > 0 {
                        y0 += 1;
                        d += delta_x;
                    }
                    d += delta_y;
                }
            } else {
                // octant 8
                let mut d = delta_y - delta_x / 2;
                for x in x0..=x1 {
                    self.plot(color, x, y0);
                    if d < 0 {
                        y0 -= 1;
                        d -= delta_x;
                    }
                    d += delta_y;
                }
            }
        } else {
            if delta_y > 0 {
                // octant 2
                let mut d = delta_y / 2 + delta_x;
                for y in y0..=y1 {
                    self.plot(color, x0, y);
                    if d < 0 {
                        x0 += 1;
                        d += delta_y;
                    }
                    d += delta_x;
                }
            } else {
                // octant 7
                let mut d = delta_y / 2 - delta_x;
                for y in (y1..=y0).rev() {
                    self.plot(color, x0, y);
                    if d > 0 {
                        x0 += 1;
                        d += delta_y;
                    }
                    d -= delta_x;
                }
            }
        }
    }
}
