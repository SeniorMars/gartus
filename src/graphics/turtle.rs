use crate::graphics::colors::Pixel;
use crate::graphics::display::Canvas;
use crate::utils::polar_to_xy;

#[derive(Debug, Clone, Default)]
/// A turle is an agent that can be controlled to draw on the [Canvas]
pub struct Turtle {
    /// Your drawing space
    /// TODO: refactor this so Turtles are owned by Canvas.
    canvas: Box<Canvas>,
    /// The color your agent will draw on
    color: Pixel,
    /// A boolean that dictacts weather the turtle will draw on the Canvas
    pub pen_mode: bool,
    /// The direction your agent will move forward or backwards. This is an angle in degrees
    direction_angle: f64,
    /// X corrdinate of where the Turtle is located
    x: u32,
    /// Y corrdinate of where the Turtle is located
    y: u32,
}

#[allow(dead_code)]
impl Turtle {
    /// Returns a new turtle that will be can be used to draw in [Canvas]
    ///
    /// # Notes
    ///
    /// Due to the fact that the Turtle will modify your canvas, you cannot have more than one
    /// Turtle per Canvas
    ///
    /// # Arguments
    ///
    /// * `canvas` - Your drawing Canvas
    /// * `line_color` - A pixel that will be the color your Turtle will use to draw
    /// * `direction_angle` - A f32 that represents that angle your turtle will move
    /// * `x` - A u32 that represents that will be the x corrdinate of where the turtle spawns in
    /// * `y` - A u32 that represents that will be the y corrdinate of where the turtle spawns in
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::turtle::Turtle;
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// let drawing = Box::new(Canvas::new(50, 50, 255));
    /// let red = Pixel::new(255, 0, 0);
    /// let turle = Turtle::new(drawing, red, 0.0, 25, 25);
    /// ```
    pub fn new(canvas: Box<Canvas>, color: Pixel, direction_angle: f64, x: u32, y: u32) -> Self {
        Self {
            canvas,
            color,
            direction_angle,
            x,
            y,
            pen_mode: false,
        }
    }

    /// Set the turtle's line color.
    ///
    /// # Arguments
    ///
    /// * `line_color` - A pixel that will be the new color your Turtle will use to draw
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::turtle::Turtle;
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// let drawing = Box::new(Canvas::new(50, 50, 255));
    /// let red = Pixel::new(255, 0, 0);
    /// let green = Pixel::new(0, 255, 0);
    /// let mut turle = Turtle::new(drawing, red, 0.0, 25, 25);
    /// turle.set_color(green)
    /// ```
    pub fn set_color(&mut self, new_color: Pixel) {
        self.color = new_color;
    }

    /// Set the turtle's direction angle.
    ///
    /// # Arguments
    ///
    /// * `direction_angle` - A f32 that represents the new angle your turtle will move towards
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::turtle::Turtle;
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// let drawing = Box::new(Canvas::new(50, 50, 255));
    /// let red = Pixel::new(255, 0, 0);
    /// let mut turle = Turtle::new(drawing, red, 0.0, 25, 25);
    /// turle.set_heading(90.0);
    /// ```
    pub fn set_heading(&mut self, direction_angle: f64) {
        self.direction_angle = (self.direction_angle + direction_angle) % 360.0;
    }

    /// Move the turtle forwards or backwards
    ///
    /// # Notes
    ///
    /// If is_drawing is true, then it will also draw a line from the old corrdinates and the new
    /// corrdinates.
    ///
    /// # Arguments
    ///
    /// * `step` - An i32 that represents direction and magnitude of where the turtle will move
    /// towards
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::turtle::Turtle;
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// let drawing = Box::new(Canvas::new(50, 50, 255));
    /// let red = Pixel::new(255, 0, 0);
    /// let mut turle = Turtle::new(drawing, red, 0.0, 25, 25);
    /// turle.move_turtle(-3);
    /// ```
    pub fn move_turtle(&mut self, step: i32) {
        let (dx, dy) = polar_to_xy(step.into(), self.direction_angle);
        let (new_x, new_y) = (self.x as f64 + dx, self.y as f64 + dy);
        if self.pen_mode {
            self.canvas
                .draw_line(self.color, self.x as f64, self.y as f64, new_x, new_y)
        }
        self.x = new_x.round() as u32;
        self.y = new_y.round() as u32;
        assert!(self.x < self.canvas.width());
        assert!(self.y < self.canvas.height());
        assert!(self.x * self.y < self.canvas.height() * self.canvas.width());
    }

    /// Set new corrdinate for turtle
    ///
    /// # Notes
    ///
    /// If is_drawing is true, then it will also draw a line from the old corrdinates and the new
    /// corrdinates
    ///
    /// # Arguments
    ///
    /// * `new_x` - A u32 that represents the new x corrdinate of the turtle
    /// * `new_y` - A u32 that represents the new y corrdinate of the turtle
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::graphics::turtle::Turtle;
    /// use crate::curves_rs::graphics::display::Canvas;
    /// use crate::curves_rs::graphics::colors::Pixel;
    /// let drawing = Box::new(Canvas::new(50, 50, 255));
    /// let red = Pixel::new(255, 0, 0);
    /// let mut turle = Turtle::new(drawing, red, 0.0, 25, 25);
    /// turle.goto(49, 49);
    /// ```
    pub fn goto(&mut self, new_x: u32, new_y: u32) {
        assert!(new_x < self.canvas.width());
        assert!(new_y < self.canvas.height());
        assert!(new_x * new_y < self.canvas.height() * self.canvas.width());
        if self.pen_mode {
            self.canvas.draw_line(
                self.color,
                self.x as f64,
                self.y as f64,
                new_x as f64,
                new_y as f64,
            )
        }
        self.x = new_x;
        self.y = new_y;
    }

    /// Get a reference to the turtle's canvas.
    pub fn canvas(&self) -> &Canvas {
        self.canvas.as_ref()
    }

    /// Set the turtle's is drawing.
    /// False is off, True is On
    pub fn set_draw_mode(&mut self, bool: bool) {
        self.pen_mode = bool;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_turtle() {
        let start_x = 50;
        let start_y = 50;
        let mut turtle = Turtle::new(
            Box::new(Canvas::new(100, 100, 255)),
            Pixel::new(150, 50, 65),
            90.0,
            start_x,
            start_y,
        );
        turtle.set_draw_mode(true);
        for _ in 0..4 {
            turtle.set_heading(90.0);
            turtle.move_turtle(10);
        }
        turtle.canvas.display().expect("Could not render image")
    }

    #[test]
    fn spiral() {
        let start_x = 149;
        let start_y = 149;
        let mut turtle = Turtle::new(
            Box::new(Canvas::new_with_bg(
                start_x * 2 + 1,
                start_y * 2 + 1,
                255,
                &Pixel::new(235, 235, 235),
            )),
            Pixel::new(150, 50, 65),
            90.0,
            start_x,
            start_y,
        );
        turtle.set_draw_mode(true);
        let mut distance = 1;
        let mut flag = 175;
        while flag > 0 {
            turtle.move_turtle(distance);
            turtle.set_heading(120.0);
            turtle.set_heading(1.0);
            distance += 1;
            flag -= 1;
        }
        turtle
            .canvas
            .save_binary("pics/spiral.png")
            .expect("Image is writeable");
        turtle.canvas.display().expect("Could not render image")
    }
}
