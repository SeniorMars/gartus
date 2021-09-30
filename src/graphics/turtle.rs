use crate::graphics::colors::Pixel;
use crate::graphics::display::Canvas;
use crate::utils::polar_to_xy;

#[derive(Debug, Clone, Default)]
/// A turle is an agent that can be controlled to draw on the [Canvas]
pub struct Turtle {
    /// Your drawing space
    canvas: Box<Canvas>,
    /// The color your agent will draw on
    line_color: Pixel,
    /// A boolean that dictacts weather the turtle will draw on the Canvas
    pub is_drawing: bool,
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
    pub fn new(
        canvas: Box<Canvas>,
        line_color: Pixel,
        direction_angle: f64,
        x: u32,
        y: u32,
    ) -> Self {
        Self {
            canvas,
            line_color,
            direction_angle,
            x,
            y,
            is_drawing: false,
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
    /// turle.set_line_color(green)
    /// ```
    pub fn set_line_color(&mut self, line_color: Pixel) {
        self.line_color = line_color;
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
        if self.is_drawing {
            self.canvas
                .draw_line(self.line_color, self.x as f64, self.y as f64, new_x, new_y)
        }
        self.x = new_x.round() as u32;
        self.y = new_y.round() as u32;
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
        if self.is_drawing {
            self.canvas.draw_line(
                self.line_color,
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
    pub fn set_is_drawing(&mut self, is_drawing: bool) {
        self.is_drawing = is_drawing;
    }
}
