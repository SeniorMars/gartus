use gartus::{
    graphics::colors::Hsl,
    prelude::{Canvas, Rgb},
};
use num::complex::Complex;
use std::{error::Error, f32::consts::PI, fs};

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 1000;
const MAX_ITERATIONS: u16 = 420;
const BAILOUT2: f32 = 16.0;

#[derive(Clone, Copy)]
struct View {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
}

#[derive(Clone, Copy)]
struct Escape {
    smooth: f32,
    escaped: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/fractals")?;

    let renders = [
        ("mandelcos", render_mandelcos()),
        ("nfam", render_nfam()),
        ("mandelbrot_nebula", render_mandelbrot_nebula()),
        ("burning_ship_radar", render_burning_ship_radar()),
        ("julia_lantern", render_julia_lantern()),
        ("domain_lattice", render_domain_lattice()),
    ];

    for (name, canvas) in renders {
        let path = format!("final/fractals/{name}.png");
        canvas.save_extension(&path)?;
        println!("saved {path}");
    }

    Ok(())
}

fn render_mandelcos() -> Canvas {
    render_escape_fractal(
        WIDTH,
        HEIGHT,
        View {
            x_min: -1.95,
            x_max: 1.15,
            y_min: -1.55,
            y_max: 1.55,
        },
        MAX_ITERATIONS,
        |c| {
            let mut z = Complex::new(0.0, 0.0);
            for n in 0..MAX_ITERATIONS {
                if z.norm_sqr() > BAILOUT2 {
                    return escaped(n, z);
                }
                z = z.cos().powu(u32::from(n.max(1))) + c;
            }
            bounded(MAX_ITERATIONS)
        },
        classic_shift_palette,
    )
}

fn render_nfam() -> Canvas {
    render_escape_fractal(
        WIDTH,
        HEIGHT,
        View {
            x_min: -1.85,
            x_max: 1.25,
            y_min: -1.55,
            y_max: 1.55,
        },
        MAX_ITERATIONS,
        |c| {
            let mut z = c;
            for n in 0..MAX_ITERATIONS {
                if z.norm_sqr() > BAILOUT2 {
                    return escaped(n, z);
                }
                let denominator = z * z + Complex::new(0.018, -0.011);
                if denominator.norm_sqr() <= f32::EPSILON {
                    return escaped(n, z);
                }
                z = denominator.inv().powi(i32::from(n.max(1))) + c;
            }
            bounded(MAX_ITERATIONS)
        },
        classic_shift_palette,
    )
}

fn render_mandelbrot_nebula() -> Canvas {
    render_escape_fractal(
        WIDTH,
        HEIGHT,
        View {
            x_min: -0.95,
            x_max: -0.42,
            y_min: 0.37,
            y_max: 0.82,
        },
        900,
        |c| {
            let mut z = Complex::new(0.0, 0.0);
            for n in 0..900 {
                if z.norm_sqr() > BAILOUT2 {
                    return escaped(n, z);
                }
                z = z * z + c;
            }
            bounded(900)
        },
        nebula_palette,
    )
}

fn render_burning_ship_radar() -> Canvas {
    render_escape_fractal(
        WIDTH,
        HEIGHT,
        View {
            x_min: -1.9,
            x_max: -1.65,
            y_min: -0.08,
            y_max: 0.12,
        },
        650,
        |c| {
            let mut z = Complex::new(0.0, 0.0);
            for n in 0..650 {
                if z.norm_sqr() > BAILOUT2 {
                    return escaped(n, z);
                }
                z = Complex::new(z.re.abs(), z.im.abs());
                z = z * z + c;
            }
            bounded(650)
        },
        radar_palette,
    )
}

fn render_julia_lantern() -> Canvas {
    render_escape_fractal(
        WIDTH,
        HEIGHT,
        View {
            x_min: -1.65,
            x_max: 1.65,
            y_min: -1.65,
            y_max: 1.65,
        },
        520,
        |point| {
            let c = Complex::new(-0.74543, 0.11301);
            let mut z = point;
            for n in 0..520 {
                if z.norm_sqr() > BAILOUT2 {
                    return escaped(n, z);
                }
                z = z * z + c;
            }
            bounded(520)
        },
        lantern_palette,
    )
}

fn render_domain_lattice() -> Canvas {
    let unit = Complex::new(1.0, 0.0);
    let four = Complex::new(4.0, 0.0);
    render_domain(
        WIDTH,
        HEIGHT,
        View {
            x_min: -3.6,
            x_max: 3.6,
            y_min: -3.6,
            y_max: 3.6,
        },
        |mut z| {
            for _ in 0..18 {
                let denom = (four * z) * (z.powi(2) - unit);
                if denom.norm_sqr() <= f32::EPSILON {
                    break;
                }
                z = ((z + unit).powi(2)) / denom;
            }
            domain_palette(z)
        },
    )
}

fn render_escape_fractal<F, P>(
    width: u32,
    height: u32,
    view: View,
    max_iterations: u16,
    mut iterate: F,
    palette: P,
) -> Canvas
where
    F: FnMut(Complex<f32>) -> Escape,
    P: Fn(Escape, u16) -> Rgb,
{
    render_domain(width, height, view, |point| {
        palette(iterate(point), max_iterations)
    })
}

fn render_domain<F>(width: u32, height: u32, view: View, mut pixel: F) -> Canvas
where
    F: FnMut(Complex<f32>) -> Rgb,
{
    let width_usize = usize::try_from(width).expect("width fits in usize");
    let height_usize = usize::try_from(height).expect("height fits in usize");
    let mut data = Vec::with_capacity(width_usize * height_usize);
    let scale_x = (view.x_max - view.x_min) / width as f32;
    let scale_y = (view.y_max - view.y_min) / height as f32;

    for y in 0..height {
        let cy = view.y_max - y as f32 * scale_y;
        for x in 0..width {
            let cx = view.x_min + x as f32 * scale_x;
            data.push(pixel(Complex::new(cx, cy)));
        }
    }

    let mut canvas = Canvas::new(width, height, Rgb::BLACK);
    canvas.upper_left_origin = true;
    canvas.fill_canvas(data);
    canvas
}

fn escaped(iteration: u16, z: Complex<f32>) -> Escape {
    Escape {
        smooth: smooth_escape(iteration, z),
        escaped: true,
    }
}

fn bounded(max_iterations: u16) -> Escape {
    Escape {
        smooth: f32::from(max_iterations),
        escaped: false,
    }
}

fn smooth_escape(iteration: u16, z: Complex<f32>) -> f32 {
    let radius = z.norm_sqr().sqrt().max(1.000_001);
    f32::from(iteration) + 1.0 - radius.ln().ln() / 2.0_f32.ln()
}

fn classic_shift_palette(escape: Escape, _max_iterations: u16) -> Rgb {
    let i = escape.smooth.round() as u16;
    Rgb::new((i << 3) as u8, (i << 5) as u8, (i << 4) as u8)
}

fn nebula_palette(escape: Escape, max_iterations: u16) -> Rgb {
    if !escape.escaped {
        return Rgb::new(0, 1, 6);
    }
    let t = escape.smooth / f32::from(max_iterations);
    let edge = t.powf(0.22);
    hsl(
        205 + (95.0 * (1.0 - edge)).round() as u16,
        92,
        8 + (70.0 * edge).round() as u16,
    )
}

fn radar_palette(escape: Escape, max_iterations: u16) -> Rgb {
    if !escape.escaped {
        return Rgb::new(0, 7, 5);
    }
    let t = escape.smooth / f32::from(max_iterations);
    let rings = ((escape.smooth * 0.33).sin().abs() * 24.0).round() as u16;
    let glow = t.powf(0.35);
    hsl(112 + rings, 100, 8 + (68.0 * glow).round() as u16)
}

fn lantern_palette(escape: Escape, max_iterations: u16) -> Rgb {
    if !escape.escaped {
        return Rgb::new(8, 4, 2);
    }
    let t = escape.smooth / f32::from(max_iterations);
    hsl(
        18 + (58.0 * t.sqrt()).round() as u16,
        98,
        10 + (70.0 * t.powf(0.45)).round() as u16,
    )
}

fn domain_palette(z: Complex<f32>) -> Rgb {
    let angle = ((z.arg() + PI) / (2.0 * PI) * 360.0).round() as u16;
    let modulus = z.norm().ln_1p();
    let rings = (modulus * 18.0).sin().abs();
    hsl(angle, 92, 28 + (rings * 44.0).round() as u16)
}

fn hsl(hue: u16, saturation: u16, light: u16) -> Rgb {
    Rgb::from(Hsl {
        hue: hue % 360,
        saturation,
        light: light.min(90),
    })
}
