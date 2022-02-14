use curves_rs::graphics::colors::Rgb;
use curves_rs::graphics::display::Canvas;

fn main() {
    let mut img = Canvas::with_capacity(256, 256, 255, Rgb::default());
    let (width, height) = (img.width(), img.height());
    let mut data: Vec<Rgb> = Vec::with_capacity((width * height) as usize);
    (0..height).rev().for_each(|j| {
        eprintln!("Scanlines reminaing: {}", height - j - 1);
        (0..width).for_each(|i| {
            data.push(Rgb {
                red: (255.99 * (i as f64 / (width - 1) as f64)) as u8,
                green: (255.99 * (j as f64 / (height - 1) as f64)) as u8,
                blue: (255.99 * 0.25) as u8,
            })
        });
    });
    eprintln!("Done.");
    img.fill_canvas(data);
    img.display().expect("Could not render image")
}

#[cfg(test)]
mod test {
    use super::*;
    use curves_rs::graphics::colors::Hsl;
    use num::complex::Complex;
    use std::f32::consts::PI;

    #[test]
    fn mandelcos() {
        const HEIGHT: u32 = 800;
        const WIDTH: u32 = 800;
        let max_iterations = 256u16;
        let cxmin = -2f32;
        let cxmax = 1f32;
        let cymin = -1.5f32;
        let cymax = 1.5f32;
        let scalex = (cxmax - cxmin) / HEIGHT as f32;
        let scaley = (cymax - cymin) / WIDTH as f32;
        let mut mandelcos = Canvas::with_capacity(HEIGHT, WIDTH, 255, Rgb::default());
        let mut data: Vec<Rgb> = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        (0..WIDTH).for_each(|x| {
            (0..HEIGHT).for_each(|y| {
                let cx = cxmin + x as f32 * scalex;
                let cy = cymin + y as f32 * scaley;

                let c = Complex::new(cx, cy);
                let mut z = Complex::new(0f32, 0f32);

                let mut i = 0;
                for n in 0..max_iterations {
                    if z.norm() > 2.0 {
                        break;
                    }
                    z = (z.cos()).powu(n.into()) + c;
                    i = n;
                }
                let red = (i << 3) as u8;
                let green = (i << 5) as u8;
                let blue = (i << 4) as u8;
                data.push(Rgb { red, green, blue })
            });
        });
        mandelcos.fill_canvas(data);
        mandelcos.sobel_incorrect();
        mandelcos.display().expect("Could not render image");
        mandelcos
            .save_extension("pics/sobel_cos.png")
            .expect("Could not save image");
    }

    #[test]
    fn nfam() {
        const HEIGHT: u32 = 800;
        const WIDTH: u32 = 800;
        let max_iterations = 256u16;
        let cxmin = -2f32;
        let cxmax = 1f32;
        let cymin = -1.5f32;
        let cymax = 1.5f32;
        let scalex = (cxmax - cxmin) / HEIGHT as f32;
        let scaley = (cymax - cymin) / WIDTH as f32;
        let mut nfam = Canvas::with_capacity(HEIGHT, WIDTH, 255, Rgb::default());
        let mut data: Vec<Rgb> = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        (0..WIDTH).for_each(|x| {
            (0..HEIGHT).for_each(|y| {
                let cx = cxmin + x as f32 * scalex;
                let cy = cymin + y as f32 * scaley;

                let c = Complex::new(cx, cy);
                let mut z = Complex::new(0f32, 0f32);

                let mut i = 0;
                for n in 0..max_iterations {
                    if z.norm() > 2.0 {
                        break;
                    }
                    z = (z.inv().powi(n.into())) + c;
                    i = n;
                }
                let red = (i << 3) as u8;
                let green = (i << 5) as u8;
                let blue = (i << 4) as u8;
                data.push(Rgb { red, green, blue })
            });
        });
        nfam.fill_canvas(data);
        nfam.display().expect("Could not render image");
        nfam.save_extension("nfam.png")
            .expect("Could not save image");
    }

    #[test]
    fn mandel() {
        const HEIGHT: u32 = 800;
        const WIDTH: u32 = 800;
        let max_iterations = 256u16;
        let cxmin = -2f32;
        let cxmax = 1f32;
        let cymin = -1.5f32;
        let cymax = 1.5f32;
        let scalex = (cxmax - cxmin) / HEIGHT as f32;
        let scaley = (cymax - cymin) / WIDTH as f32;
        let mut mandel = Canvas::with_capacity(HEIGHT, WIDTH, 255, Rgb::default());
        let mut data: Vec<Rgb> = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        (0..WIDTH).for_each(|x| {
            (0..HEIGHT).for_each(|y| {
                let cx = cxmin + x as f32 * scalex;
                let cy = cymin + y as f32 * scaley;

                let c = Complex::new(cx, cy);
                let mut z = Complex::new(0f32, 0f32);

                let mut i = 0;
                for n in 0..max_iterations {
                    if z.norm() > 2.0 {
                        break;
                    }
                    z = z * z + c;
                    i = n;
                }
                let red = (i << 3) as u8;
                let green = (i << 5) as u8;
                let blue = (i << 4) as u8;
                data.push(Rgb { red, green, blue })
            });
        });
        mandel.fill_canvas(data);
        mandel.sobel_incorrect();
        mandel
            .save_extension("mandel.png")
            .expect("Could not save image")
    }

    #[test]
    fn ship_rgb() {
        const HEIGHT: u32 = 800;
        const WIDTH: u32 = 800;

        let max_iterations = 256u16;
        let cxmin = -2f32;
        let cxmax = 1f32;
        let cymin = -1.5f32;
        let cymax = 1.5f32;
        let scalex = (cxmax - cxmin) / HEIGHT as f32;
        let scaley = (cymax - cymin) / WIDTH as f32;
        let mut ship = Canvas::with_capacity(HEIGHT, WIDTH, 255, Rgb::default());
        let mut data: Vec<Rgb> = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        (0..WIDTH).for_each(|x| {
            (0..HEIGHT).for_each(|y| {
                let cx = cxmin + x as f32 * scalex;
                let cy = cymin + y as f32 * scaley;

                let c = Complex::new(cx, cy);
                let mut z = Complex::new(0f32, 0f32);

                let mut i = 0;
                for n in 0..max_iterations {
                    if z.norm() > 2.0 {
                        break;
                    }
                    let tempz = Complex::new(z.re.abs(), 0.0) + Complex::new(0.0, (z.im).abs());
                    z = (tempz.powi(2)) + c;
                    i = n;
                }
                // let red = i as u8;
                // let green = i as u8;
                // let blue = i as u8;
                let red = (i << 5) as u8;
                let green = (i << 3) as u8;
                let blue = (i << 4) as u8;
                data.push(Rgb { red, green, blue })
            });
        });
        ship.fill_canvas(data);
        ship.sobel_incorrect();
        ship.display().expect("Could not render image");
        ship.save_extension("red_ship.png")
            .expect("Could not render image")
    }

    #[test]
    fn ship_hsl() {
        const HEIGHT: u32 = 800;
        const WIDTH: u32 = 800;
        const ZOOM: f64 = 700.0;

        let max_iterations = 1000u16;
        let cxmin = -2f32;
        let cxmax = 1f32;
        let cymin = -1.8f32;
        let cymax = 1.8f32;
        let scalex = (cxmax - cxmin) / ZOOM as f32;
        let scaley = (cymax - cymin) / ZOOM as f32;
        let mut ship = Canvas::with_capacity(HEIGHT, WIDTH, 255, Hsl::default());
        ship.upper_left_system = true;
        let mut data: Vec<Hsl> = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        (0..WIDTH).for_each(|x| {
            (0..HEIGHT).for_each(|y| {
                let cx = cxmin + x as f32 * scalex;
                let cy = cymin + y as f32 * scaley;

                let c = Complex::new(cx, cy);
                let mut z = Complex::new(0f32, 0f32);

                let mut i = 0;
                for n in 0..max_iterations {
                    if z.norm() > 2.0 {
                        break;
                    }
                    let tempz = Complex::new(z.re.abs(), 0.0) + Complex::new(0.0, (z.im).abs());
                    z = (tempz.powi(2)) + c;
                    i = n;
                }
                let hue = i % 360;
                let saturation = 100;
                let light = 75;
                data.push(Hsl {
                    hue,
                    saturation,
                    light,
                })
            });
        });
        ship.fill_canvas(data);
        ship.display().expect("Could not render image")
    }

    #[test]
    fn domain_coloring_plot() {
        const HEIGHT: u32 = 800;
        const WIDTH: u32 = 800;
        const ZOOM: u16 = 800;

        let max_iterations = 16;
        let cxmin = -5f32;
        let cxmax = 5f32;
        let cymin = -5f32;
        let cymax = 5f32;
        let scalex = (cxmax - cxmin) / ZOOM as f32;
        let scaley = (cymax - cymin) / ZOOM as f32;
        let mut color_domain = Canvas::with_capacity(HEIGHT, WIDTH, 255, Hsl::default());
        let mut data: Vec<Hsl> = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        let unit = Complex::new(1.0, 0.0);
        let four = Complex::new(4.0, 0.0);
        let lattes = |z: Complex<f32>| ((z + unit).powi(2)) / ((four * z) * (z.powi(2) - unit));
        (0..WIDTH).for_each(|x| {
            (0..HEIGHT).for_each(|y| {
                let cx = cxmin + x as f32 * scalex;
                let cy = cymin + y as f32 * scaley;
                let mut z = Complex::new(cx, cy);
                for _ in 0..max_iterations {
                    z = lattes(z);
                }
                let hue = (z.arg() * 180.0 / PI).round() as u16;
                let saturation = 100;
                let light = 50;
                data.push(Hsl {
                    hue,
                    saturation,
                    light,
                })
            })
        });
        color_domain.fill_canvas(data);
        color_domain.display().expect("Could not render image");
        color_domain
            .save_extension("failed_domain20.png")
            .expect("Could not render image")
    }

    #[test]
    fn julia() {
        let width = 800;
        let height = 600;
        let mut julia = Canvas::with_capacity(height, width, 255, Rgb::default());
        let mut data: Vec<Rgb> = Vec::with_capacity((width * height) as usize);
        let cx = -0.9;
        let cy = 0.27015;
        let interations = 110;
        for x in 0..width {
            for y in 0..height {
                let mut zx = 3.0 * (x as f32 - 0.5 * width as f32) / (width as f32);
                let mut zy = 2.0 * (y as f32 - 0.5 * height as f32) / (height as f32);
                let mut i = interations;
                while zx * zx + zy * zy < 4.0 && i > 1 {
                    let temp = zx * zx - zy * zy + cx;
                    zy = 2.0 * zx * zy + cy;
                    zx = temp;
                    i -= 1;
                }
                data.push(Rgb {
                    red: (i << 3) as u8,
                    green: (i << 5) as u8,
                    blue: (i << 4) as u8,
                })
                // write!(writer, "{} {} {} ", i as u8, i as u8, i as u8)?;
            }
        }
        julia.fill_canvas(data);
        julia.sobel_incorrect();
        julia.display().expect("Could not render image");
        julia
            .save_extension("juila.png")
            .expect("Could not render image")
    }
}
