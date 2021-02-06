extern crate num;
extern crate rand;
use num::complex::Complex;
use rand::Rng;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

const WIDTH: usize = 800;
const HEIGHT: usize = 800;
const RANGE: usize = 255;

#[derive(Default)]
struct Pixel {
    red: u8,
    green: u8,
    blue: u8,
}

fn writer_image(mut writer: impl Write) -> io::Result<()> {
    let width = 8000;
    let height = 6000;
    writeln!(writer, "P3 {} {} {}\n", height, width, RANGE)?;

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
            write!(writer, "{} {} {} ", i as u8, i as u8, i as u8)?;
            // let red = (i << 3) as u8;
            // let green = (i << 5) as u8;
            // let blue = (i << 4) as u8;
            // write!(writer, "{} {} {} ", red, green, blue)?;
        }
        write!(writer, "\n")?;
    }
    Ok(())
}

fn write_blocks(mut writer: impl Write) -> io::Result<()> {
    writeln!(writer, "P3 {} {} {}\n", HEIGHT, WIDTH, RANGE)?;
    let mut rng = rand::thread_rng();
    let mut blockrow = String::new();
    for _row in 0..8 {
        blockrow.clear();
        for _col in 0..8 {
            let rgb = Pixel {
                red: rng.gen_range(100, 250),
                green: rng.gen_range(0, 200),
                blue: rng.gen_range(50, 200),
            };
            for _blockrow in 0..100 {
                blockrow += &format!("{} {} {} ", rgb.red, rgb.green, rgb.blue);
            }
        }
        for _blockscol in 0..100 {
            writeln!(writer, "{}", blockrow)?;
        }
    }
    Ok(())
}

fn writer_mandel(mut writer: impl Write) -> io::Result<()> {
    let max_iterations = 256u16;
    let cxmin = -2f32;
    let cxmax = 1f32;
    let cymin = -1.5f32;
    let cymax = 1.5f32;
    let scalex = (cxmax - cxmin) / HEIGHT as f32;
    let scaley = (cymax - cymin) / WIDTH as f32;

    writeln!(writer, "P3 {} {} {}\n", HEIGHT, WIDTH, RANGE)?;
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let cx = cxmin + x as f32 * scalex;
            let cy = cymin + y as f32 * scaley;

            let c = Complex::new(cx, cy);
            let mut z = Complex::new(0f32, 0f32);

            let mut i = 0;
            for t in 0..max_iterations {
                if z.norm() > 2.0 {
                    break;
                }
                z = z * z + c;
                i = t;
            }
            let red = (i << 3) as u8;
            let green = (i << 5) as u8;
            let blue = (i << 4) as u8;
            write!(writer, "{} {} {} ", red, green, blue)?;
        }
        write!(writer, "\n")?;
    }
    Ok(())
}

pub fn main() -> io::Result<()> {
    let mut file = File::create("image.ppm").unwrap();
    let writer = BufWriter::new(&mut file);
    // writer_mandel(writer)
    writer_image(writer)
    // write_blocks(writer)
}
