//! The graphics module hosts all the needed struts to playing
//! around with computer graphics.

/// Includes the [Pixel] and [HSL] struts, which are the basic foundation to color
pub mod colors;
/// Includes [CanvasConfig] and [AnimationConfig] to configure [Canvas]
pub mod config;
/// Includes the [Canvas] strut, which represents your "drawing board".
pub mod display;
/// Hosts all the functions needed to start drawing onto the [Canvas]
pub mod draw;
/// Some preset filters that can be applied to a Canvas
pub mod filters;
/// An agent that can move throughout the [Canvas]
pub mod turtle;
