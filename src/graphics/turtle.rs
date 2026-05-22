use crate::{gmath::helpers::polar_to_xy, graphics::display::Canvas};

use super::colors::Rgb;

#[derive(Debug, Clone, Default)]
/// A turtle is an agent that can be controlled to draw on a [`Canvas`].
pub struct Turtle {
    /// The color your agent will draw on
    color: Rgb,
    /// The direction your agent will move forward or backward. This is an angle in degrees.
    direction_angle: f64,
    /// A boolean that dictates whether the turtle will draw on the Canvas.
    pub pen_mode: bool,
    /// X coordinate of where the Turtle is located
    x: f64,
    /// Y coordinate of where the Turtle is located
    y: f64,
    /// A stack of saved states of the turtle. Each state includes the position, direction, pen mode, and color of the turtle at the time it was saved.
    state_stack: Vec<TurtleState>,
}

/// A struct representing the state of a [`Turtle`] at a given point in time. This is used for saving and restoring the turtle's state with the push and pop operations.
#[derive(Debug, Clone, Default)]
pub struct TurtleState {
    x: f64,
    y: f64,
    direction_angle: f64,
    pen_mode: bool,
    color: Rgb,
}

impl TurtleState {
    /// Returns the `(x, y)` position of this state.
    #[must_use]
    pub fn position(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    /// Returns the direction angle of this state in degrees.
    #[must_use]
    pub fn heading(&self) -> f64 {
        self.direction_angle
    }

    /// Returns the draw color of this state.
    #[must_use]
    pub fn color(&self) -> Rgb {
        self.color
    }

    /// Returns whether the pen was down in this state.
    #[must_use]
    pub fn pen_mode(&self) -> bool {
        self.pen_mode
    }
}

#[allow(dead_code)]
impl Turtle {
    /// Returns a new turtle that can be used as a geometric cursor or to draw on a [`Canvas`].
    ///
    /// # Arguments
    ///
    /// * `line_color` - A pixel that will be the color your Turtle will use to draw
    /// * `direction_angle` - A f64 that represents the angle your turtle will move
    /// * `x` - The x coordinate where the turtle starts
    /// * `y` - The y coordinate where the turtle starts
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::turtle::Turtle;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let red = Rgb::new(255, 0, 0);
    /// let turtle = Turtle::new(red, 0.0, 25.0, 25.0);
    /// ```
    #[must_use] 
    pub fn new(color: Rgb, direction_angle: f64, x: f64, y: f64) -> Self {
        Self {
            color,
            direction_angle: Self::normalize_angle(direction_angle),
            x,
            y,
            pen_mode: false,
            state_stack: Vec::new(),
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
    /// let mut turtle = Turtle::new(red, 0.0, 25.0, 25.0);
    /// turtle.set_color(green)
    /// ```
    pub fn set_color(&mut self, new_color: Rgb) {
        self.color = new_color;
    }

    fn normalize_angle(angle: f64) -> f64 {
        angle.rem_euclid(360.0)
    }

    /// Set the turtle's direction angle absolutely.
    pub fn set_heading(&mut self, direction_angle: f64) {
        self.direction_angle = Self::normalize_angle(direction_angle);
    }

    /// Rotate the turtle by adding to its current direction angle.
    pub fn rotate(&mut self, delta: f64) {
        self.direction_angle = Self::normalize_angle(self.direction_angle + delta);
    }

    /// Returns the current `(x, y)` position of the turtle.
    #[must_use]
    pub fn position(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    /// Returns the current direction angle of the turtle in degrees.
    #[must_use]
    pub fn heading(&self) -> f64 {
        self.direction_angle
    }

    /// Returns the current draw color of the turtle.
    #[must_use]
    pub fn color(&self) -> Rgb {
        self.color
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_lossless,
        clippy::cast_sign_loss
    )]
    /// Move the turtle forward or backward without drawing.
    ///
    /// # Notes
    ///
    /// Returns the old and new positions.
    ///
    /// # Arguments
    ///
    /// * `step` - Direction and magnitude of where the turtle will move
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::turtle::Turtle;
    /// use crate::gartus::graphics::display::Canvas;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let red = Rgb::new(255, 0, 0);
    /// let mut turtle = Turtle::new(red, 0.0, 25.0, 25.0);
    /// turtle.forward(-3.0);
    /// ```
    pub fn forward(&mut self, step: f64) -> ((f64, f64), (f64, f64)) {
        let start = self.position();
        let (dx, dy) = polar_to_xy(step, self.direction_angle);
        let (new_x, new_y) = (self.x + dx, self.y + dy);
        self.x = new_x;
        self.y = new_y;
        (start, self.position())
    }

    /// Move the turtle forward or backward and draw when the pen is down.
    pub fn draw_forward(&mut self, canvas: &mut Canvas, step: f64) {
        let (start, end) = self.forward(step);
        if self.pen_mode {
            canvas.draw_line(self.color, start.0, start.1, end.0, end.1);
        }
    }

    /// Compatibility wrapper for [`Turtle::draw_forward`].
    pub fn move_turtle(&mut self, canvas: &mut Canvas, step: f64) {
        self.draw_forward(canvas, step);
    }

    /// Set a new coordinate for the turtle without drawing.
    ///
    /// # Notes
    ///
    /// Returns the old and new positions.
    ///
    /// # Arguments
    ///
    /// * `new_x` - The new x coordinate of the turtle
    /// * `new_y` - The new y coordinate of the turtle
    ///
    /// # Panics
    /// Panics if `new_x` or `new_y` is not finite.
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::gartus::graphics::turtle::Turtle;
    /// use crate::gartus::graphics::display::Canvas;
    /// use crate::gartus::graphics::colors::Rgb;
    /// let red = Rgb::new(255, 0, 0);
    /// let mut turtle = Turtle::new(red, 0.0, 25.0, 25.0);
    /// turtle.goto(49.0, 49.0);
    /// ```
    pub fn goto(&mut self, new_x: f64, new_y: f64) -> ((f64, f64), (f64, f64)) {
        assert!(
            new_x.is_finite() && new_y.is_finite(),
            "turtle coordinates must be finite"
        );
        let start = self.position();
        self.x = new_x;
        self.y = new_y;
        (start, self.position())
    }

    /// Set a new coordinate and draw when the pen is down.
    pub fn draw_goto(&mut self, canvas: &mut Canvas, new_x: f64, new_y: f64) {
        let (start, end) = self.goto(new_x, new_y);
        if self.pen_mode {
            canvas.draw_line(self.color, start.0, start.1, end.0, end.1);
        }
    }

    // /// Get a reference to the turtle's canvas.
    // pub fn canvas(&self) -> &Canvas<C> {
    //     self.canvas.as_ref()
    // }

    /// Put the pen down so the turtle draws as it moves.
    pub fn pen_down(&mut self) {
        self.pen_mode = true;
    }

    /// Lift the pen so the turtle moves without drawing.
    pub fn pen_up(&mut self) {
        self.pen_mode = false;
    }

    /// Set the turtle's position.
    pub fn set_position(&mut self, x: f64, y: f64) {
        self.goto(x, y);
    }

    /// Push the current state of the turtle onto the stack.
    pub fn push_state(&mut self) {
        // Save the current state of the turtle
        let state = TurtleState {
            x: self.x,
            y: self.y,
            color: self.color,
            direction_angle: self.direction_angle,
            pen_mode: self.pen_mode,
        };
        self.state_stack.push(state);
    }

    /// Pop the last state of the turtle from the stack.
    pub fn pop_state(&mut self) -> Option<TurtleState> {
        let state = self.state_stack.pop()?;
        self.x = state.x;
        self.y = state.y;
        self.color = state.color;
        self.direction_angle = state.direction_angle;
        self.pen_mode = state.pen_mode;
        Some(state)
    }

    /// Get a reference to the turtle's state stack.
    #[must_use] 
    pub fn state_stack(&self) -> &[TurtleState] {
        &self.state_stack
    }

    /// Rotate the turtle to the right.
    pub fn rotate_right(&mut self, angle: f64) {
        self.rotate(-angle);
    }

    /// Rotate the turtle to the left.
    pub fn rotate_left(&mut self, angle: f64) {
        self.rotate(angle);
    }
}

#[cfg(test)]
mod test {
    use crate::graphics::colors::Rgb;

    use super::*;

    #[test]
    fn heading_is_normalized() {
        let mut turtle = Turtle::new(Rgb::default(), -10.0, 0.0, 0.0);
        assert!((turtle.heading() - 350.0).abs() < f64::EPSILON);

        turtle.rotate_right(90.0);
        assert!((turtle.heading() - 260.0).abs() < f64::EPSILON);

        turtle.rotate_left(190.0);
        assert!((turtle.heading() - 90.0).abs() < f64::EPSILON);

        turtle.set_heading(-30.0);
        assert!((turtle.heading() - 330.0).abs() < f64::EPSILON);
    }

    #[test]
    fn forward_updates_position_without_canvas() {
        let mut turtle = Turtle::new(Rgb::default(), 0.0, 10.0, 20.0);
        let (start, end) = turtle.forward(5.0);

        assert_eq!(start, (10.0, 20.0));
        assert!((end.0 - 15.0).abs() < f64::EPSILON);
        assert!((end.1 - 20.0).abs() < f64::EPSILON);
        assert_eq!(turtle.position(), end);
    }

    #[test]
    fn goto_updates_position_without_canvas() {
        let mut turtle = Turtle::new(Rgb::default(), 0.0, 10.0, 20.0);
        let (start, end) = turtle.goto(-5.0, 12.0);

        assert_eq!(start, (10.0, 20.0));
        assert_eq!(end, (-5.0, 12.0));
        assert_eq!(turtle.position(), (-5.0, 12.0));
    }

    #[test]
    #[should_panic(expected = "turtle coordinates must be finite")]
    fn goto_rejects_non_finite_coordinates() {
        let mut turtle = Turtle::new(Rgb::default(), 0.0, 0.0, 0.0);
        turtle.goto(f64::NAN, 0.0);
    }

    #[test]
    fn push_pop_restores_state() {
        let red = Rgb::new(255, 0, 0);
        let blue = Rgb::new(0, 0, 255);
        let mut turtle = Turtle::new(red, 45.0, 10.0, 20.0);
        turtle.pen_down();
        turtle.push_state();

        turtle.set_color(blue);
        turtle.set_heading(180.0);
        turtle.set_position(50.0, 60.0);
        turtle.pen_up();

        let restored = turtle.pop_state().expect("state should be present");
        let rpos = restored.position();
        assert!((rpos.0 - 10.0).abs() < f64::EPSILON && (rpos.1 - 20.0).abs() < f64::EPSILON);
        assert_eq!(turtle.color(), red);
        assert!((turtle.heading() - 45.0).abs() < f64::EPSILON);
        let tpos = turtle.position();
        assert!((tpos.0 - 10.0).abs() < f64::EPSILON && (tpos.1 - 20.0).abs() < f64::EPSILON);
        assert!(turtle.pen_mode);
        assert!(turtle.state_stack().is_empty());
    }

    #[test]
    fn pop_state_returns_none_when_empty() {
        let mut turtle = Turtle::new(Rgb::default(), 0.0, 0.0, 0.0);
        assert!(turtle.pop_state().is_none());
    }

    #[test]
    fn draw_forward_draws_when_pen_is_down() {
        let start_x = 50.0;
        let start_y = 50.0;
        let mut canvas = Canvas::new_with_bg(100, 100, Rgb::WHITE);
        let mut turtle = Turtle::new(Rgb::new(150, 50, 65), 90.0, start_x, start_y);
        turtle.pen_down();
        turtle.draw_forward(&mut canvas, 10.0);

        assert!(canvas.pixels().iter().any(|pixel| *pixel == turtle.color()));
    }

    #[test]
    #[ignore = "writes an example image"]
    fn spiral() {
        let start_x = 149.0;
        let start_y = 149.0;
        let mut canvas = Canvas::new_with_bg(149 * 2 + 1, 149 * 2 + 1, Rgb::new(235, 235, 235));
        let mut turtle = Turtle::new(Rgb::new(150, 50, 65), 90.0, start_x, start_y);
        turtle.pen_down();
        let mut distance = 1.0;
        let mut flag = 175;
        while flag > 0 {
            turtle.move_turtle(&mut canvas, distance);
            turtle.rotate(120.0);
            turtle.rotate(1.0);
            distance += 1.0;
            flag -= 1;
        }
        canvas
            .save_binary("./pics/spiral.png")
            .expect("Image is writeable");
        // canvas.display().expect("Could not render image")
    }
}
