use std::{collections::VecDeque, fs::File};
use std::{
    ffi::OsStr,
    io::{BufReader, Read},
    path::Path,
    process::Command,
};

use crate::graphics::{colors::Rgb, display::Canvas};
/// ppmifizes an image so that it works with with this systems
///
/// # Arguments
/// * `file_name` - The name, an &str, of the file to be ppmifizes
/// * `pos_glitch` - A bool that turns potential glitch on.
///
/// # Note
/// Make sure to turn on [Canvas] with pos_glitch on
/// with [CanvasConfig] if `pos_glitch` is turned on
///
/// # Examples
///
/// Basic usage:
///```no_run
/// use crate::curves_rs::prelude::{Canvas, Rgb};
/// use crate::curves_rs::external;
/// let colors = vec![
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
/// ];
/// let mut canvas = Canvas::with_capacity(3, 3, 255, Rgb::BLACK);
/// canvas.fill_canvas(colors);
/// canvas.save_binary("./works.ppm").expect("Works");
/// let other = external::ppmify("./works.ppm", true).expect("Life is wrong");
/// assert_eq!(canvas.pixels(), other.pixels());
/// ```
pub fn ppmify(
    file_name: &str,
    pos_glitch: bool,
) -> Result<Canvas<Rgb>, Box<dyn std::error::Error>> {
    let path = Path::new(file_name);
    if !path.exists() {
        panic!("File Does not exit");
    }
    let ext = path
        .extension()
        .and_then(OsStr::to_str)
        .unwrap_or_else(|| panic!("Check your file input: {:?}", path));
    let correct_ext = path.with_extension("ppm");
    if ext != "ppm" {
        let converted = correct_ext.to_str().expect("Cannot get new file name");
        Command::new("convert")
            .arg(file_name)
            .arg(converted)
            .spawn()?
            .wait()?;
    };
    parse_ppm(&correct_ext, pos_glitch)
}

fn vec_to_int(vec: Vec<u8>) -> u32 {
    let mut length = vec.len() as u32;
    let mut sum = 0u32;
    vec.into_iter().for_each(|i| {
        let i_num = ascii_to_num(i) as u32;
        length -= 1;
        sum += i_num * 10_u32.pow(length)
    });
    sum
}

fn byte_vec_fill(bytes: &mut VecDeque<u8>, vec: &mut Vec<u8>) {
    loop {
        let pot_num = bytes.pop_front();
        if pot_num != Some(32) && pot_num != Some(10) {
            vec.push(pot_num.unwrap())
        } else {
            break;
        };
    }
}

fn ascii_to_num(byte: u8) -> u8 {
    match byte {
        48 => 0,
        49 => 1,
        50 => 2,
        51 => 3,
        52 => 4,
        53 => 5,
        54 => 6,
        55 => 7,
        56 => 8,
        57 => 9,
        _ => panic!("Not in the valid range to convert"),
    }
}

fn parse_ppm(path: &Path, pos_glitch: bool) -> Result<Canvas<Rgb>, Box<dyn std::error::Error>> {
    // this is a naive parser. Not 100% compatible with the spec
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut bytes = reader
        .bytes()
        .map(|pos_byte| pos_byte.expect("File Follows Spec"))
        .collect::<VecDeque<u8>>();

    let p_type = (bytes.pop_front().unwrap(), bytes.pop_front().unwrap());

    // pop off newline
    bytes.pop_front();

    // We have to loop as a Canvas's width/height can be very large
    let mut height_vec = Vec::new();
    let mut width_vec = Vec::new();
    // if pos_glitch is on, then inccorectly gathers width and height wrong. May look cool.
    if pos_glitch {
        byte_vec_fill(&mut bytes, &mut height_vec);
        byte_vec_fill(&mut bytes, &mut width_vec);
    } else {
        byte_vec_fill(&mut bytes, &mut width_vec);
        byte_vec_fill(&mut bytes, &mut height_vec);
    }

    let width = vec_to_int(width_vec);
    let height = vec_to_int(height_vec);

    let mut color_depth_vec = Vec::new();
    byte_vec_fill(&mut bytes, &mut color_depth_vec);

    // Note due to the spec, this will never overflow or other unspecified behavior
    let color_depth: u8 = vec_to_int(color_depth_vec)
        .try_into()
        .expect("File does not follow ppm spec");

    let mut canvas = Canvas::with_capacity(width, height, color_depth, Rgb::default());

    let mut pixels = Vec::with_capacity(height as usize * width as usize);
    match p_type {
        // p3
        (80, 51) => {
            while !bytes.is_empty() {
                let mut red_vec = Vec::with_capacity(3);
                let mut green_vec = Vec::with_capacity(3);
                let mut blue_vec = Vec::with_capacity(3);
                byte_vec_fill(&mut bytes, &mut red_vec);
                byte_vec_fill(&mut bytes, &mut green_vec);
                byte_vec_fill(&mut bytes, &mut blue_vec);
                let (red, green, blue) = (
                    vec_to_int(red_vec)
                        .try_into()
                        .expect("File does not follow ppm spec"),
                    vec_to_int(green_vec)
                        .try_into()
                        .expect("File does not follow ppm spec"),
                    vec_to_int(blue_vec)
                        .try_into()
                        .expect("File does not follow ppm spec"),
                );
                pixels.push(Rgb::new(red, green, blue));
            }
        }
        // p6
        (80, 54) => {
            while !bytes.is_empty() {
                let red = bytes.pop_front().expect("File does not follow ppm spec");
                let green = bytes.pop_front().expect("File does not follow ppm spec");
                let blue = bytes.pop_front().expect("File does not follow ppm spec");
                pixels.push(Rgb::new(red, green, blue));
            }
        }
        _ => panic!("Unsupported spec"),
    };
    canvas.fill_canvas(pixels);
    Ok(canvas)
}

#[test]
fn external_fun() {
    use crate::graphics::display::CanvasConfig;
    let pos_glitch = true;
    let mut canvas = ppmify("./pics/index.png", pos_glitch).expect("Implmentation is wrong");
    canvas.set_config(CanvasConfig::new(false, pos_glitch));
    canvas.display().expect("Could not display image");
    canvas.sobel();
    canvas.display().expect("Could not display image");
    canvas
        .save_extension("corro.png")
        .expect("Could not save image");
}
