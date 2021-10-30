#[derive(Debug)]
/// A type that represents a parametric equation
pub struct Parametric<F: Fn(f64) -> f64, G: Fn(f64) -> f64> {
    /// The first part, x, of the parametric equation: f(t)
    x: F,
    /// The second part, y, of the parametric equation: g(t)
    y: G,
}

impl<F: Fn(f64) -> f64, G: Fn(f64) -> f64> Parametric<F, G> {
    /// Creates a new Parametric equation with functions that give x and y based on t
    ///
    /// # Arguments
    ///
    /// * `F` - A function that returns x (a f64) based on t -- f(t)
    /// * `G` - A function that returns y (a f64) based on t -- g(t)
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```
    /// use crate::curves_rs::gmath::parametric::Parametric;
    /// fn value(num: f64) -> f64 {
    ///    num
    /// }
    /// let equations = Parametric::new(value, value);
    /// ```
    pub fn new(x: F, y: G) -> Self {
        Self { x, y }
    }
}
