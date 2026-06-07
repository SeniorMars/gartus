use gartus::{external, prelude::*};
use std::{error::Error, fs};

const WIDTH: u32 = 900;
const HEIGHT: u32 = 700;

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render it takes two:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("pics")?;

    let bunny = external::meshify("examples/data/meshes/bunny.obj")?;
    let normalize = external::normalize_mesh_transform(&bunny, 260.0, external::MeshUpAxis::Y);

    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::new(252, 240, 229));
    canvas.set_wrapped(false);
    canvas.set_shading_mode(ShadingMode::Flat);
    canvas.set_polygon_color_mode(PolygonColorMode::LineColor);

    draw_floor(&mut canvas);
    draw_bunny_pair(&mut canvas, &bunny, &normalize);
    draw_heart(&mut canvas, 450.0, 436.0);
    draw_caption(&mut canvas);

    canvas.save_extension("pics/it_takes_two.png")?;
    println!("saved pics/it_takes_two.png");
    Ok(())
}

fn draw_floor(canvas: &mut Canvas) {
    canvas.fill_rect(0, 0, i64::from(WIDTH), 190, Rgb::new(231, 205, 177));
    canvas.draw_line(Rgb::new(191, 147, 116), 0.0, 190.0, f64::from(WIDTH), 190.0);
}

fn draw_bunny_pair(canvas: &mut Canvas, bunny: &PolygonMatrix, normalize: &Matrix) {
    let left = Matrix::translate(338.0, 196.0, 20.0)
        * Matrix::rotate_y(207.0)
        * Matrix::rotate_z(-2.0)
        * normalize.clone();
    let right = Matrix::translate(562.0, 190.0, 20.0)
        * Matrix::rotate_y(-33.0)
        * Matrix::rotate_z(2.0)
        * normalize.clone();

    canvas.set_line_pixel(Rgb::new(202, 176, 156));
    canvas.draw_polygons(&bunny.apply(&left));

    canvas.set_line_pixel(Rgb::new(184, 205, 218));
    canvas.draw_polygons(&bunny.apply(&right));
}

fn draw_heart(canvas: &mut Canvas, cx: f64, cy: f64) {
    let background = Rgb::new(252, 240, 229);
    let color = Rgb::new(218, 34, 84);
    let highlight = Rgb::new(255, 135, 169);

    fill_implicit_heart(canvas, cx, cy, color);
    carve_heart_cleft(canvas, cx, cy, background);

    for offset in 0..5 {
        canvas.draw_line(
            highlight,
            cx - 42.0 + f64::from(offset),
            cy + 8.0 - f64::from(offset) * 0.8,
            cx - 12.0 + f64::from(offset),
            cy + 22.0 - f64::from(offset) * 0.8,
        );
    }
}

fn carve_heart_cleft(canvas: &mut Canvas, cx: f64, cy: f64, color: Rgb) {
    let bottom = (cy + 26.0).round() as i64;
    let top = (cy + 64.0).round() as i64;

    for py in bottom..=top {
        let t = (py - bottom) as f64 / (top - bottom) as f64;
        let width = 2.0 + t * 15.0;
        let left = (cx - width).round() as i64;
        let right = (cx + width).round() as i64;
        canvas.fill_rect(left, py, right - left + 1, 1, color);
    }
}

fn fill_implicit_heart(canvas: &mut Canvas, cx: f64, cy: f64, color: Rgb) {
    let min_x = (cx - 86.0).round() as i64;
    let max_x = (cx + 86.0).round() as i64;
    let min_y = (cy - 78.0).round() as i64;
    let max_y = (cy + 66.0).round() as i64;

    for py in min_y..=max_y {
        let mut span_start = None;

        for px in min_x..=max_x {
            let x = (px as f64 - cx) / 66.0;
            let y = (py as f64 - cy + 10.0) / 58.0;
            let inside = (x * x + y * y - 1.0).powi(3) - x * x * y.powi(3) <= 0.0;

            match (inside, span_start) {
                (true, None) => span_start = Some(px),
                (false, Some(start)) => {
                    canvas.fill_rect(start, py, px - start, 1, color);
                    span_start = None;
                }
                _ => {}
            }
        }

        if let Some(start) = span_start {
            canvas.fill_rect(start, py, max_x - start + 1, 1, color);
        }
    }
}

fn draw_caption(canvas: &mut Canvas) {
    canvas.draw_text_centered("IT TAKES TWO", 454, 610, 7, Rgb::new(91, 54, 70));
    canvas.draw_text_centered("IT TAKES TWO", 450, 614, 7, Rgb::new(255, 74, 126));
}
