use crate::graphics::colors::Pixel;
use crate::graphics::display::Canvas;
use crate::graphics::matrix::Matrix;

#[allow(dead_code)]
impl Canvas {
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
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// use crate::curves_rs::graphics::display::Canvas;
    /// let mut image = Canvas::new(25, 25, 255);
    /// let color = Pixel::new(0, 64, 255);
    /// let background_color = Pixel::new(0, 0, 0);
    /// image.fill(10, 10, &color, &background_color)
    /// ```
    pub fn fill(&mut self, x: i32, y: i32, fill_color: &Pixel, boundary_color: &Pixel) {
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

    /// Fills in the area of a 2D figure given a random point
    /// inside the figure that is meant for animation.
    ///
    /// # Arguments
    ///
    /// * `x` - A signed i32 int that represents the x of the random point
    /// * `y` - A signed i32 int that represents the y of the random point
    /// * `fill_color` - A [Pixel] will be the color the polygon will be filled in
    /// * `boundary_color` - A [Pixel] that is the represents the outline of the shape
    /// * `filename` - The prefix of the name the animation will belong to
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// use crate::curves_rs::graphics::display::Canvas;
    /// let mut image = Canvas::new(25, 25, 255);
    /// let color = Pixel::new(0, 64, 255);
    /// let background_color = Pixel::new(0, 0, 0);
    /// image.fill_with_animation(10, 10, &color, &background_color, "image")
    /// ```
    pub fn fill_with_animation(
        &mut self,
        x: i32,
        y: i32,
        fill_color: &Pixel,
        boundary_color: &Pixel,
        filename: &str,
    ) {
        let current = self.get_pixel(x, y);
        if current != boundary_color && current != fill_color {
            self.plot(fill_color, x as i32, y as i32);
            self.save_binary(&format!("anim/{}{:08}.ppm", filename, self.anim_index))
                .expect("Could not save to file");
            self.anim_index += 1;
            self.fill_with_animation(x + 1, y, fill_color, boundary_color, filename);
            self.fill_with_animation(x, y + 1, fill_color, boundary_color, filename);
            self.fill_with_animation(x - 1, y, fill_color, boundary_color, filename);
            self.fill_with_animation(x, y - 1, fill_color, boundary_color, filename);
        }
    }

    /// Draws all lines in provided in a given [Matrix] onto the [Canvas]
    ///
    /// # Arguments
    ///
    /// * `matrix` - A [Matrix] reference that has at least two points
    /// (2 by 4) to draw onto the [Canvas]
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// use crate::curves_rs::graphics::matrix::Matrix;
    /// let mut image = Canvas::new(25, 25, 255);
    /// let color = Pixel::new(0, 64, 255);
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

            self.draw_line(self.line, x0, y0, x1, y1);
        }
    }

    /// Draws all lines in provided in a given [Matrix] onto the [Canvas] for an animation
    ///
    /// # Arguments
    ///
    /// * `matrix` - A [Matrix] reference that has at least two points
    /// (2 by 4) to draw onto the [Canvas]
    /// * `filename` - The prefix of the name the animation will belong to
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// use crate::curves_rs::graphics::matrix::Matrix;
    /// let mut image = Canvas::new(25, 25, 255);
    /// let color = Pixel::new(0, 64, 255);
    /// image.set_line_pixel(&color);
    /// let matrix = Matrix::identity_matrix(4);
    /// image.draw_lines_for_animation(&matrix, "cool_picture")
    /// ```
    pub fn draw_lines_for_animation(&mut self, matrix: &Matrix, filename: &str) {
        let mut iter = matrix.iter_by_point();
        while let Some(point) = iter.next() {
            let (x0, y0, _z0) = (point[0], point[1], point[3]);
            let (x1, y1, _z1) = match iter.next() {
                Some(p1) => (p1[0], p1[1], p1[2]),
                None => panic!("Need at least 2 points to draw"),
            };

            self.save_binary(&format!("anim/{}{:08}.ppm", filename, self.anim_index))
                .expect("Could not save to file");
            self.draw_line(self.line, x0, y0, x1, y1);
        }
        self.save_binary(&format!("anim/{}{:08}.ppm", filename, self.anim_index))
            .expect("Could not save to file");
    }

    // def add_circle( points, cx, cy, cz, r, step ):
    //     pass

    // def add_curve( points, x0, y0, x1, y1, x2, y2, x3, y3, step, curve_type ):
    //     pass

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
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// let mut image = Canvas::new(25, 25, 255);
    /// let color = Pixel::new(0, 64, 255);
    /// image.draw_line(color, 0.0, 0.0, 24.0, 24.0)
    /// ```
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
