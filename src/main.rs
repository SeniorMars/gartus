use curves_rs::gmath::ray::Ray;
use curves_rs::gmath::vector::{Point, Vector};
use curves_rs::graphics::colors::Rgb;
use curves_rs::graphics::display::Canvas;

fn hit_sphere(center: Point, radius: f64, r: &Ray) -> bool {
    let oc = *r.orgin() - center;
    let a = r.direction().dot(*r.direction());
    let b = 2.0 * oc.dot(*r.direction());
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    discriminant > 0.0
}

pub fn ray_color(r: &Ray) -> Vector {
    if hit_sphere(Point::new(0.0, 0.0, -1.0), 0.5, r) {
        return Vector::new(1.0, 0.0, 0.0);
    }
    let unit_direction = r.direction().normalized();
    let t = 0.5 * (unit_direction[1] + 1.0);
    (1.0 - t) * Vector::new(1.0, 1.0, 1.0) + t * Vector::new(0.5, 0.7, 1.0)
}

pub fn main() {
    const ASPECT_RATIO: f64 = 16.0 / 9.0;
    const IMAGE_WIDTH: u64 = 256;
    const IMAGE_HEIGHT: u64 = (256_f64 / ASPECT_RATIO) as u64;

    let viewport_height = 2.0;
    let viewport_width = ASPECT_RATIO * viewport_height;
    let focal_length = 1.0;

    let origin = Point::new(0.0, 0.0, 0.0);
    let horizontal = Vector::new(viewport_width, 0.0, 0.0);
    let vertical = Vector::new(0.0, viewport_height, 0.0);
    let lower_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - Vector::new(0.0, 0.0, focal_length);

    let mut img =
        Canvas::with_capacity(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32, 255, Rgb::default());
    let mut data: Vec<Rgb> = Vec::with_capacity((img.width() * img.height()) as usize);

    (0..IMAGE_HEIGHT).rev().for_each(|j| {
        // eprintln!("Scanlines reminaing: {}", IMAGE_HEIGHT - j - 1);
        // stderk().flush().unwrap();
        (0..IMAGE_WIDTH).for_each(|i| {
            let u = i as f64 / ((IMAGE_WIDTH - 1) as f64);
            let v = j as f64 / ((IMAGE_HEIGHT - 1) as f64);

            let pixel_color = ray_color(&Ray::new(
                origin,
                lower_left_corner + u * horizontal + v * vertical - origin,
            ));
            data.push(Rgb::from(pixel_color))
        });
    });
    eprintln!("Done.");
    img.fill_canvas(data);
    img.display().expect("Could not render image")
}
