use super::{
    colors::{ColorSpace, Rgb},
    display::Canvas,
};

impl<C: ColorSpace> Canvas<C>
where
    Rgb: From<C>,
{
    /// .
    ///
    /// # Examples
    ///
    /// ```
    /// use curves_rs::graphics::filters::Canvas;
    ///
    /// let mut canvas = ;
    /// canvas.grayscale();
    /// assert_eq!(canvas, );
    /// ```
    pub fn grayscale(&mut self) {}
}
