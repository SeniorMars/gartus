// use gartus::gmath::ray::Ray;
// use gartus::gmath::vector::{Point, Vector};
use gartus::graphics::colors::Rgb;
use gartus::graphics::config::CanvasConfig;
use gartus::graphics::display::Canvas;
use gartus::prelude::Matrix;

pub fn main() {
    let mut img = Canvas::new_with_bg(800, 800, 255, Rgb::new(24, 26, 27));
    let mut geass = Matrix::new(4, 0, Vec::with_capacity(112 * 2));
    img.set_config(CanvasConfig {
        upper_left_system: true,
        ..Default::default()
    });

    let geass_corrs = [
        170, 216, 190, 249, 190, 249, 220, 274, 220, 274, 250, 295, 250, 295, 289, 318, 289, 318,
        347, 349, 347, 349, 347, 421, 347, 421, 400, 449, 400, 449, 453, 421, 453, 421, 453, 349,
        453, 349, 511, 318, 511, 318, 550, 295, 550, 295, 580, 274, 580, 274, 606, 249, 606, 249,
        630, 216, 630, 216, 601, 285, 601, 285, 571, 323, 571, 323, 525, 358, 525, 358, 492, 388,
        492, 388, 489, 448, 489, 448, 441, 475, 441, 475, 400, 499, 400, 499, 359, 475, 359, 475,
        311, 448, 311, 448, 308, 388, 308, 388, 275, 358, 275, 358, 229, 323, 229, 323, 199, 285,
        199, 285, 170, 216,
    ];

    for corr in geass_corrs.chunks(2) {
        geass.add_point(corr[0] as f64, corr[1] as f64, 0.0)
    }

    let mut base = Matrix::translate(-400.0, -400.0, 0.0);
    let mut reflect = Matrix::reflect_xz();
    let mut half = Matrix::reflect_45();
    let mut last_half = Matrix::reflect_yz();

    base *= geass.clone();
    reflect *= base.clone();
    last_half *= half.clone();
    last_half *= base.clone();
    half *= base.clone();
    base.add_dataset(&reflect);
    base.add_dataset(&half);
    base.add_dataset(&last_half);

    let white = Rgb::new(255, 255, 255);
    img.set_line_pixel(&white);
    img.draw_lines(
        &geass
            .mult_matrix(&Matrix::scale(0.1, 0.1, 0.1))
            .mult_matrix(&Matrix::translate(360.0, 370.0, 0.0)),
    );
    img.fill(406, 413, &white, &white);
    img.set_line_pixel(&Rgb::new(191, 70, 61));
    for i in 0..180 {
        let mut copy = img.clone();
        copy.draw_lines(&base.mult_matrix(
            &Matrix::rotate_y(i as f64).mult_matrix(&Matrix::translate(400.0, 400.0, 0.0)),
        ));
        copy.save_extension(&format!("./anim/geass{:04}.png", i))
            .expect("Could not save image")
    }
    // img.display().expect("Could not display image")
    // let file_name = "./geass.gif";
    // utils::animation("geass", file_name);
    // utils::view_animation(file_name)
}

// fn hit_sphere(center: Point, radius: f64, r: &Ray) -> bool {
//     let oc = *r.orgin() - center;
//     let a = r.direction().dot(*r.direction());
//     let b = 2.0 * oc.dot(*r.direction());
//     let c = oc.dot(oc) - radius * radius;
//     let discriminant = b * b - 4.0 * a * c;
//     discriminant > 0.0
// }
//
// pub fn ray_color(r: &Ray) -> Vector {
//     if hit_sphere(Point::new(0.0, 0.0, -1.0), 0.5, r) {
//         return Vector::new(1.0, 0.0, 0.0);
//     }
//     let unit_direction = r.direction().normalized();
//     let t = 0.5 * (unit_direction[1] + 1.0);
//     (1.0 - t) * Vector::new(1.0, 1.0, 1.0) + t * Vector::new(0.5, 0.7, 1.0)
// }
//
// pub fn main() {
//     const ASPECT_RATIO: f64 = 16.0 / 9.0;
//     const IMAGE_WIDTH: u64 = 256;
//     const IMAGE_HEIGHT: u64 = (256_f64 / ASPECT_RATIO) as u64;
//
//     let viewport_height = 2.0;
//     let viewport_width = ASPECT_RATIO * viewport_height;
//     let focal_length = 1.0;
//
//     let origin = Point::new(0.0, 0.0, 0.0);
//     let horizontal = Vector::new(viewport_width, 0.0, 0.0);
//     let vertical = Vector::new(0.0, viewport_height, 0.0);
//     let lower_left_corner =
//         origin - horizontal / 2.0 - vertical / 2.0 - Vector::new(0.0, 0.0, focal_length);
//
//     let mut img =
//         Canvas::with_capacity(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32, 255, Rgb::default());
//     let mut data: Vec<Rgb> = Vec::with_capacity((img.width() * img.height()) as usize);
//
//     (0..IMAGE_HEIGHT).rev().for_each(|j| {
//         eprintln!("Scanlines reminaing: {}", IMAGE_HEIGHT - j - 1);
//         (0..IMAGE_WIDTH).for_each(|i| {
//             let u = i as f64 / ((IMAGE_WIDTH - 1) as f64);
//             let v = j as f64 / ((IMAGE_HEIGHT - 1) as f64);
//
//             let pixel_color = ray_color(&Ray::new(
//                 origin,
//                 lower_left_corner + u * horizontal + v * vertical - origin,
//             ));
//             data.push(Rgb::from(pixel_color))
//         });
//     });
//     eprintln!("Done.");
//     img.fill_canvas(data);
//     img.display().expect("Could not render image")
// }
