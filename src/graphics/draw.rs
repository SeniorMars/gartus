use super::colors::{ColorSpace, Rgb};
use crate::gmath::matrix::Matrix;
use crate::graphics::display::Canvas;

#[allow(dead_code)]
impl<C: ColorSpace> Canvas<C>
where
    Rgb: From<C>,
{
    /// Fills in the area of a 2D figure given a random point inside the figure.
    ///
    /// # Arguments
    ///
    /// * `x` - A signed i32 int that represents the x of the random point
    /// * `y` - A signed i32 int that represents the y of the random point
    /// * `fill_color` - A [Pixel] will be the color the polygon will be filled in
    /// * `boundary_color` - A [Pixel] that is the represents the outline of the shape
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::colors::Rgb;
    /// use crate::gartus::graphics::display::Canvas;
    /// let background_color = Rgb::new(0, 0, 0);
    /// let mut image = Canvas::new(25, 25, 255, background_color);
    /// let color = Rgb::new(0, 64, 255);
    /// image.fill(10, 10, &color, &background_color)
    /// ```
    pub fn fill(&mut self, x: i64, y: i64, fill_color: &C, boundary_color: &C) {
        let mut points = vec![(x, y)];
        while let Some((x, y)) = points.pop() {
            let pixel = self.get_pixel(x, y);
            if pixel == boundary_color || pixel == fill_color {
                continue;
            }
            // Terrible idea
            // if self.config.animation() {
            //     self.save_binary(&format!(
            //         "anim/{}{:08}.ppm",
            //         self.config.file_prefix(),
            //         self.config.anim_index(),
            //     ))
            //     .expect("Could not save to file");
            //     self.config.increase_anim_index()
            // }
            self.plot(fill_color, x, y);
            points.push((x + 1, y));
            points.push((x, y + 1));
            points.push((x - 1, y));
            points.push((x, y - 1));
            // points.push((x - 1, y - 1));
            // points.push((x - 1, y + 1));
            // points.push((x + 1, y - 1));
            // points.push((x + 1, y + 1));
        }
    }

    /// Draws all lines in provided in a given [Matrix] onto the [Canvas]
    ///
    /// # Arguments
    ///
    /// * `matrix` - A [Matrix] reference that has at least two points
    /// (2 by 4) to draw onto the [Canvas]
    ///
    /// # Panics
    /// * If Matrix does not have two points to draw
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::display::Canvas;
    /// use crate::gartus::graphics::colors::Rgb;
    /// use crate::gartus::gmath::matrix::Matrix;
    /// let mut image = Canvas::new(25, 25, 255, Rgb::default());
    /// let color = Rgb::new(0, 64, 255);
    /// image.set_line_pixel(&color);
    /// let matrix = Matrix::identity_matrix(4);
    /// image.draw_lines(&matrix)
    /// ```
    pub fn draw_lines(&mut self, matrix: &Matrix) {
        let mut iter = matrix.iter_by_point();
        while let Some(point) = iter.next() {
            let (x0, y0, _z0) = (point[0], point[1], point[3]);
            let (x1, y1, _z1) = match iter.next() {
                Some(p1) => (p1[0], p1[1], p1[2]),
                None => panic!("Need at least 2 points to draw"),
            };
            if self.config.animation() {
                self.save_binary(&format!(
                    "anim/{}{:08}.ppm",
                    self.config.file_prefix(),
                    self.config.anim_index(),
                ))
                .expect("Could not save to file");
            }
            self.draw_line(self.line, x0, y0, x1, y1);
        }
        if self.config.animation() {
            self.save_binary(&format!(
                "anim/{}{:08}.ppm",
                self.config.file_prefix(),
                self.config.anim_index(),
            ))
            .expect("Could not save to file");
        }
    }

    /// Draws a line onto the [Canvas] provided two sets of points.
    ///
    /// # Arguments
    ///
    /// * `color` - A [Pixel] that will will represent the color of the new line
    /// * `x0` - A f64 float that represents the start x coordinate of the line
    /// * `y0` - A f64 float that represents the start y coordinate of the line
    /// * `x1` - A f64 float that represents the end x coordinate of the line
    /// * `y1` - A f64 float that represents the end y coordinate of the line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::display::Canvas;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let mut image = Canvas::new(25, 25, 255, Rgb::default());
    /// let color = Rgb::new(0, 64, 255);
    /// image.draw_line(color, 0.0, 0.0, 24.0, 24.0)
    /// ```
    pub fn draw_line(&mut self, color: C, x0: f64, y0: f64, x1: f64, y1: f64) {
        if self.config.animation() {
            {
                let this = &mut self.config;
                this.increase_anim_index();
            }
        }
        let (x0, y0, x1, y1) = if x0 > x1 {
            (x1, y1, x0, y0)
        } else {
            (x0, y0, x1, y1)
        };
        #[allow(clippy::cast_possible_truncation)]
        let (mut x0, mut y0, x1, y1) = (
            x0.round() as i64,
            y0.round() as i64,
            x1.round() as i64,
            y1.round() as i64,
        );
        let (delta_y, delta_x) = (2 * (y1 - y0), -2 * (x1 - x0));

        if (x1 - x0).abs() >= (y1 - y0).abs() {
            if delta_y > 0 {
                // octant 1
                let mut d = delta_y + delta_x / 2;
                for x in x0..=x1 {
                    self.plot(&color, x, y0);
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
                    self.plot(&color, x, y0);
                    if d < 0 {
                        y0 -= 1;
                        d -= delta_x;
                    }
                    d += delta_y;
                }
            }
        } else if delta_y > 0 {
            // octant 2
            let mut d = delta_y / 2 + delta_x;
            for y in y0..=y1 {
                self.plot(&color, x0, y);
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
                self.plot(&color, x0, y);
                if d > 0 {
                    x0 += 1;
                    d += delta_y;
                }
                d -= delta_x;
            }
        }
    }
}
