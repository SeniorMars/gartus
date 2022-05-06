use crate::{gmath::helpers::polar_to_xy, graphics::display::Canvas};

use super::colors::{ColorSpace, Rgb};

#[derive(Debug, Clone, Default)]
/// A turle is an agent that can be controlled to draw on the [Canvas]
pub struct Turtle<C: ColorSpace>
where
    Rgb: From<C>,
{
    /// The color your agent will draw on
    color: C,
    corrdinates: Vec<(u32, u32)>,
    /// The direction your agent will move forward or backwards. This is an angle in degrees
    direction_angle: f64,
    /// A boolean that dictacts weather the turtle will draw on the Canvas
    pub pen_mode: bool,
    /// X corrdinate of where the Turtle is located
    x: u32,
    /// Y corrdinate of where the Turtle is located
    y: u32,
}

#[allow(dead_code)]
impl<C: ColorSpace> Turtle<C>
where
    Rgb: From<C>,
{
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
    /// use crate::gartus::graphics::turtle::Turtle;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let red = Rgb::new(255, 0, 0);
    /// let turle = Turtle::new(red, 0.0, 25, 25);
    /// ```
    pub fn new(color: C, direction_angle: f64, x: u32, y: u32) -> Self {
        let corrdinates = vec![(x, y)];
        Self {
            color,
            direction_angle,
            x,
            y,
            pen_mode: false,
            corrdinates,
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
    /// use crate::gartus::graphics::turtle::Turtle;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let red = Rgb::new(255, 0, 0);
    /// let green = Rgb::new(0, 255, 0);
    /// let mut turle = Turtle::new(red, 0.0, 25, 25);
    /// turle.set_color(green)
    /// ```
    pub fn set_color(&mut self, new_color: C) {
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
    /// use crate::gartus::graphics::turtle::Turtle;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let red = Rgb::new(255, 0, 0);
    /// let mut turle = Turtle::new(red, 0.0, 25, 25);
    /// turle.set_heading(90.0);
    /// ```
    pub fn set_heading(&mut self, direction_angle: f64) {
        self.direction_angle = (self.direction_angle + direction_angle) % 360.0;
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_lossless,
        clippy::cast_sign_loss
    )]
    /// Move the turtle forwards or backwards
    ///
    /// # Notes
    ///
    /// If `is_drawing` is true, then it will also draw a line from the old corrdinates and the new
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
    /// use crate::gartus::graphics::turtle::Turtle;
    /// use crate::gartus::graphics::display::Canvas;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let mut drawing = Canvas::new(50, 50, 255, Rgb::default());
    /// let red = Rgb::new(255, 0, 0);
    /// let mut turle = Turtle::new(red, 0.0, 25, 25);
    /// turle.move_turtle(&mut drawing, -3);
    /// ```
    pub fn move_turtle(&mut self, canvas: &mut Canvas<C>, step: i32) {
        let (dx, dy) = polar_to_xy(step.into(), self.direction_angle);
        let (new_x, new_y) = (f64::from(self.x) + dx, f64::from(self.y) + dy);
        if self.pen_mode {
            canvas.draw_line(
                self.color,
                f64::from(self.x),
                f64::from(self.y),
                new_x,
                new_y,
            );
        }

        self.x = new_x.round() as u32;
        self.y = new_y.round() as u32;
        self.corrdinates.push((self.x, self.y));
    }

    /// Set new corrdinate for turtle
    ///
    /// # Notes
    ///
    /// If `is_drawing` is true, then it will also draw a line from the old corrdinates and the new
    /// corrdinates
    ///
    /// # Arguments
    ///
    /// * `new_x` - A u32 that represents the new x corrdinate of the turtle
    /// * `new_y` - A u32 that represents the new y corrdinate of the turtle
    ///
    /// # Panics
    /// * If the arguments are greater than the canvas dimensions
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::turtle::Turtle;
    /// use crate::gartus::graphics::display::Canvas;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let mut drawing = Canvas::new(50, 50, 255, Rgb::default());
    /// let red = Rgb::new(255, 0, 0);
    /// let mut turle = Turtle::new(red, 0.0, 25, 25);
    /// turle.goto(&mut drawing, 49, 49);
    /// ```
    pub fn goto(&mut self, canvas: &mut Canvas<C>, new_x: u32, new_y: u32) {
        assert!(new_x < canvas.width());
        assert!(new_y < canvas.height());
        if self.pen_mode {
            canvas.draw_line(
                self.color,
                f64::from(self.x),
                f64::from(self.y),
                f64::from(new_x),
                f64::from(new_y),
            );
        }

        self.corrdinates.push((new_x, new_y));
        self.x = new_x;
        self.y = new_y;
    }

    // /// Get a reference to the turtle's canvas.
    // pub fn canvas(&self) -> &Canvas<C> {
    //     self.canvas.as_ref()
    // }

    /// Set the turtle's is drawing.
    /// False is off, True is On
    pub fn set_draw_mode(&mut self, bool: bool) {
        self.pen_mode = bool;
    }

    /// Get a reference to the turtle's corrdinates.
    pub fn corrdinates(&self) -> &[(u32, u32)] {
        self.corrdinates.as_ref()
    }

    /// Get a mutable reference to the turtle's corrdinates.
    pub fn corrdinates_mut(&mut self) -> &mut Vec<(u32, u32)> {
        &mut self.corrdinates
    }
}
#[cfg(test)]
mod test {
    use crate::graphics::colors::Rgb;

    use super::*;

    #[test]
    fn test_turtle() {
        let start_x = 50;
        let start_y = 50;
        let mut canvas = Canvas::new(100, 100, 255, Rgb::default());
        let mut turtle = Turtle::new(Rgb::new(150, 50, 65), 90.0, start_x, start_y);
        turtle.set_draw_mode(true);
        for _ in 0..4 {
            turtle.set_heading(90.0);
            turtle.move_turtle(&mut canvas, 10);
        }
        // println!("{:?}", turtle.corrdinates());
        // canvas.display().expect("Could not render image")
    }

    #[test]
    fn spiral() {
        let start_x = 149;
        let start_y = 149;
        let mut canvas = Canvas::new_with_bg(
            start_x * 2 + 1,
            start_y * 2 + 1,
            255,
            Rgb::new(235, 235, 235),
        );
        let mut turtle = Turtle::new(Rgb::new(150, 50, 65), 90.0, start_x, start_y);
        turtle.set_draw_mode(true);
        let mut distance = 1;
        let mut flag = 175;
        while flag > 0 {
            turtle.move_turtle(&mut canvas, distance);
            turtle.set_heading(120.0);
            turtle.set_heading(1.0);
            distance += 1;
            flag -= 1;
        }
        canvas
            .save_binary("./pics/spiral.png")
            .expect("Image is writeable");
        // canvas.display().expect("Could not render image")
    }
}
