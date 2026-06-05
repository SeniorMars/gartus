//! Minimal ray-tracing helpers following the early "Ray Tracing in One Weekend" steps.

use crate::{
    gmath::{
        ray::Ray,
        vector::{Point, Vector},
    },
    graphics::{camera::RayCamera, colors::Rgb, display::Canvas},
};

/// The 16:9 aspect ratio used by the first weekend camera setup.
pub const WIDESCREEN_ASPECT_RATIO: f64 = 16.0 / 9.0;

/// A color represented as linear floating-point RGB components in `0.0..=1.0`.
pub type LinearColor = Vector;

/// Renders the book's first red/green PPM gradient into a [`Canvas`].
#[must_use]
pub fn render_unit_gradient(width: u32, height: u32) -> Canvas {
    let denom_x = f64::from(width.saturating_sub(1)).max(1.0);
    let denom_y = f64::from(height.saturating_sub(1)).max(1.0);
    Canvas::from_fn(width, height, |x, y| {
        Rgb::from(LinearColor::new(
            f64::from(x) / denom_x,
            f64::from(y) / denom_y,
            0.0,
        ))
    })
}

/// Returns true if `ray` intersects the sphere.
#[must_use]
pub fn hit_sphere(center: Point, radius: f64, ray: &Ray) -> bool {
    let oc = center - *ray.origin();
    let a = ray.direction().length_squared();
    let b = -2.0 * ray.direction().dot(oc);
    let c = oc.length_squared() - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    discriminant >= 0.0
}

/// Computes the blue-to-white background gradient for a ray.
#[must_use]
pub fn sky_gradient(ray: &Ray) -> LinearColor {
    let unit_direction = ray.direction().normalized();
    let a = 0.5 * (unit_direction.y() + 1.0);
    (1.0 - a) * LinearColor::new(1.0, 1.0, 1.0) + a * LinearColor::new(0.5, 0.7, 1.0)
}

/// Computes the first sphere scene from the book: a red sphere over a blue sky.
#[must_use]
pub fn first_sphere_color(ray: &Ray) -> LinearColor {
    if hit_sphere(Point::new(0.0, 0.0, -1.0), 0.5, ray) {
        LinearColor::new(1.0, 0.0, 0.0)
    } else {
        sky_gradient(ray)
    }
}

/// Renders the first sphere scene from the book with a 16:9 camera.
#[must_use]
pub fn render_first_sphere(image_width: u32) -> Canvas {
    RayCamera::new(image_width, WIDESCREEN_ASPECT_RATIO).render(first_sphere_color)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn unit_gradient_matches_first_ppm_image_corners() {
        let canvas = render_unit_gradient(3, 3);
        assert_eq!(canvas.pixels()[0], Rgb::BLACK);
        assert_eq!(canvas.pixels()[2], Rgb::new(255, 0, 0));
        assert_eq!(canvas.pixels()[8], Rgb::YELLOW);
    }

    #[test]
    fn sphere_intersection_detects_direct_hit_and_miss() {
        let origin = Point::new(0.0, 0.0, 0.0);
        let center = Point::new(0.0, 0.0, -1.0);

        let hit = Ray::new(origin, Vector::new(0.0, 0.0, -1.0));
        assert!(hit_sphere(center, 0.5, &hit));

        let miss = Ray::new(origin, Vector::new(0.0, 1.0, -1.0));
        assert!(!hit_sphere(center, 0.5, &miss));
    }

    #[test]
    fn sky_gradient_blends_white_to_blue() {
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, -1.0));
        let color = sky_gradient(&ray);

        assert_close(color.x(), 0.75);
        assert_close(color.y(), 0.85);
        assert_close(color.z(), 1.0);
    }

    #[test]
    fn first_sphere_render_has_red_center() {
        let canvas = render_first_sphere(40);
        let center = canvas
            .get_pixel(20, 11)
            .expect("center pixel should be inside the canvas");

        assert_eq!(*center, Rgb::RED);
    }
}
