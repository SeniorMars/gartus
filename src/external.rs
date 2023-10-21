use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::{ffi::OsStr, io::Read, path::Path, process::Command};

use crate::graphics::{colors::Rgb, display::Canvas};
/// ppmifizes an image so that it works with with this systems
///
/// # Arguments
/// * `file_name` - The name, an &str, of the file to be ppmifizes
/// * `pos_glitch` - A bool that turns potential glitch on.
///
/// # Note
/// Make sure to turn on [Canvas] with `pos_glitch` on
/// with [`CanvasConfig`] if `pos_glitch` is turned on
///
/// # Panics
/// If file does not exist, or it cannot be converted into a ppm file
///
/// # Errors
/// todo!()
///
/// # Examples
///
/// Basic usage:
///```no_run
/// use crate::gartus::prelude::{Canvas, Rgb};
/// use crate::gartus::external;
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
/// let other = external::ppmify("./works.ppm", false).expect("Life is wrong");
/// assert_eq!(canvas.pixels(), other.pixels());
/// ```
pub fn ppmify(
    file_name: &str,
    pos_glitch: bool,
) -> Result<Canvas<Rgb>, Box<dyn std::error::Error>> {
    let path = Path::new(file_name);
    if !path.exists() {
        return Err(format!("File does not exist: {file_name}").into());
    }

    let ext = path
        .extension()
        .and_then(OsStr::to_str)
        .ok_or("Invalid file extension")?;

    let correct_ext = path.with_extension("ppm");

    if ext != "ppm" {
        let converted = correct_ext.to_str().ok_or("Cannot get new file name")?;
        let status = Command::new("convert")
            .arg(file_name)
            .arg(converted)
            .status()?;
        if !status.success() {
            return Err("Failed to convert image to ppm".into());
        }
    }

    parse_ppm(&correct_ext, pos_glitch)
}

#[allow(dead_code)]
fn bytes_to_int(vec: Vec<u8>) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let mut length = vec.len() as u32;
    let mut sum = 0u32;
    for i in vec {
        let i_num = u32::from(ascii_to_num(i));
        length -= 1;
        sum += i_num * 10_u32.pow(length);
    }
    sum
}

#[allow(dead_code)]
fn byte_vec_fill(bytes: &mut VecDeque<u8>, vec: &mut Vec<u8>) {
    while let Some(byte) = bytes.pop_front() {
        if byte != 32 && byte != 10 {
            vec.push(byte);
        } else {
            break;
        }
    }
}

// fn ascii_to_num(byte: u8) -> u8 {
//     println!("{byte}");
//     match byte {
//         48 => 0,
//         49 => 1,
//         50 => 2,
//         51 => 3,
//         52 => 4,
//         53 => 5,
//         54 => 6,
//         55 => 7,
//         56 => 8,
//         57 => 9,
//         _ => panic!("Not in the valid range to convert"),
//     }
// }

#[allow(dead_code)]
fn ascii_to_num(byte: u8) -> u8 {
    match byte {
        b'0' => 0, // Use byte literals (e.g., b'0') for ASCII characters
        b'1' => 1,
        b'2' => 2,
        b'3' => 3,
        b'4' => 4,
        b'5' => 5,
        b'6' => 6,
        b'7' => 7,
        b'8' => 8,
        b'9' => 9,
        _ => panic!("Not in the valid range to convert"),
    }
}

fn parse_ppm(path: &Path, pos_glitch: bool) -> Result<Canvas<Rgb>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut cursor = 0;

    // Read PPM type
    let p_type = match (buffer.get(cursor), buffer.get(cursor + 1)) {
        (Some(&b'P'), Some(&b'3')) => {
            cursor += 2;
            '3'
        }
        (Some(&b'P'), Some(&b'6')) => {
            cursor += 2;
            '6'
        }
        _ => {
            dbg!(buffer[cursor + 1]);
            return Err("Invalid PPM file: Unsupported type".into());
        }
    };

    // Skip comments
    while cursor < buffer.len() && buffer[cursor] == b'#' {
        while cursor < buffer.len() && buffer[cursor] != b'\n' {
            cursor += 1;
        }
        cursor += 1; // Skip newline
    }

    // Read dimensions
    let mut dimensions = Vec::new();
    while cursor < buffer.len() && dimensions.len() < 2 {
        if buffer[cursor].is_ascii_digit() {
            let mut value = 0u32;
            while cursor < buffer.len() && buffer[cursor].is_ascii_digit() {
                value = value * 10 + u32::from(buffer[cursor] - b'0');
                cursor += 1;
            }
            dimensions.push(value);
        } else {
            cursor += 1;
        }
    }

    if dimensions.len() != 2 {
        return Err("Invalid PPM file: Invalid dimensions".into());
    }

    let (width, height) = if pos_glitch {
        (dimensions[1], dimensions[0]) // Swap dimensions for pos_glitch
    } else {
        (dimensions[0], dimensions[1])
    };

    cursor += 1; // Skip newline

    // Read color depth
    let mut color_depth = 0u16;
    while cursor < buffer.len() && buffer[cursor].is_ascii_digit() {
        color_depth = color_depth * 10 + u16::from(buffer[cursor] - b'0');
        cursor += 1;
    }

    if cursor >= buffer.len() || buffer[cursor] != b'\n' {
        return Err("Invalid PPM file: Invalid color depth".into());
    }

    cursor += 1; // Skip newline

    let mut canvas = Canvas::with_capacity(width, height, color_depth, Rgb::default());
    let mut pixels = Vec::with_capacity(height as usize * width as usize);

    // Process ASCII data (P3 format)
    if p_type == '3' {
        let mut red = 0u8;
        let mut green = 0u8;
        let mut blue = 0u8;

        while cursor < buffer.len() {
            while cursor < buffer.len() && buffer[cursor].is_ascii_digit() {
                red = red * 10 + (buffer[cursor] - b'0');
                cursor += 1;
            }

            while cursor < buffer.len() && buffer[cursor].is_ascii_digit() {
                green = green * 10 + (buffer[cursor] - b'0');
                cursor += 1;
            }

            while cursor < buffer.len() && buffer[cursor].is_ascii_digit() {
                blue = blue * 10 + (buffer[cursor] - b'0');
                cursor += 1;
            }

            // Skip whitespace
            while cursor < buffer.len() && !buffer[cursor].is_ascii_digit() {
                cursor += 1;
            }

            pixels.push(Rgb::new(red, green, blue));

            if pixels.len() == (width * height).try_into().unwrap() {
                break;
            }
        }
    }
    // Process binary data (P6 format)
    else if p_type == '6' {
        while cursor < buffer.len() {
            let red = buffer[cursor];
            let green = buffer[cursor + 1];
            let blue = buffer[cursor + 2];
            pixels.push(Rgb::new(red, green, blue));
            cursor += 3;

            if pixels.len() == width as usize * height as usize {
                break;
            }
        }
    } else {
        return Err("Unsupported PPM type".into());
    }

    canvas.fill_canvas(pixels);
    Ok(canvas)
}

// fn parse_ppm_old(path: &Path, pos_glitch: bool) -> Result<Canvas<Rgb>, Box<dyn Error>> {
//     let mut file = File::open(path)?;
//     let mut bytes = Vec::new();
//     file.read_to_end(&mut bytes)?;
//
//     let p_type = (bytes.first().copied(), bytes.get(1).copied());
//
//     let mut bytes = bytes.into_iter().skip(2).collect::<VecDeque<u8>>();
//
//     // Handle newline
//     bytes.pop_front();
//
//     let mut height_vec = Vec::new();
//     let mut width_vec = Vec::new();
//     // if pos_glitch is on, then inccorectly gathers width and height wrong. May look cool.
//     if pos_glitch {
//         byte_vec_fill(&mut bytes, &mut height_vec);
//         byte_vec_fill(&mut bytes, &mut width_vec);
//     } else {
//         byte_vec_fill(&mut bytes, &mut width_vec);
//         byte_vec_fill(&mut bytes, &mut height_vec);
//     }
//
//     let width = bytes_to_int(width_vec);
//     let height = bytes_to_int(height_vec);
//
//     let mut color_depth_vec = Vec::new();
//     byte_vec_fill(&mut bytes, &mut color_depth_vec);
//
//     let color_depth: u16 = bytes_to_int(color_depth_vec)
//         .try_into()
//         .map_err(|_| "File does not follow ppm spec")?;
//
//     dbg!(color_depth);
//
//     let mut canvas = Canvas::with_capacity(width, height, color_depth, Rgb::default());
//     let mut pixels = Vec::with_capacity(height as usize * width as usize);
//
//     match p_type {
//         (Some(80), Some(51)) => {
//             while !bytes.is_empty() {
//                 let mut red_vec = Vec::with_capacity(3);
//                 let mut green_vec = Vec::with_capacity(3);
//                 let mut blue_vec = Vec::with_capacity(3);
//                 byte_vec_fill(&mut bytes, &mut red_vec);
//                 byte_vec_fill(&mut bytes, &mut green_vec);
//                 byte_vec_fill(&mut bytes, &mut blue_vec);
//
//                 let (red, green, blue) = (
//                     bytes_to_int(red_vec).try_into()?,
//                     bytes_to_int(green_vec).try_into()?,
//                     bytes_to_int(blue_vec).try_into()?,
//                 );
//
//                 pixels.push(Rgb::new(red, green, blue));
//             }
//         }
//         (Some(80), Some(54)) => {
//             while !bytes.is_empty() {
//                 let (red, green, blue) = (
//                     bytes.pop_front().ok_or("File does not follow ppm spec")?,
//                     bytes.pop_front().ok_or("File does not follow ppm spec")?,
//                     bytes.pop_front().ok_or("File does not follow ppm spec")?,
//                 );
//
//                 pixels.push(Rgb::new(red, green, blue));
//             }
//         }
//         _ => return Err("Unsupported spec".into()),
//     };
//
//     let pixels = column_major(width, height, &pixels);
//
//     canvas.fill_canvas(pixels);
//     Ok(canvas)
// }

#[test]
fn external_fun() {
    use crate::graphics::config::CanvasConfig;
    let pos_glitch = true;
    let mut canvas = ppmify("./corro.png", pos_glitch).expect("Implmentation is wrong");
    canvas.set_config(CanvasConfig::new(false, pos_glitch, false));
    canvas.display().expect("Could not display image");
    let sobel = canvas.sobel();
    sobel.display().expect("Could not display image");
    sobel
        .save_extension("corro.png")
        .expect("Could not save image");
}

#[test]
fn command_block() {
    use crate::graphics::config::CanvasConfig;
    let pos_glitch = true;
    let mut canvas = ppmify("./CAR.png", pos_glitch).expect("Implmentation is wrong");
    canvas.set_config(CanvasConfig::new(false, pos_glitch, false));
    canvas.display().expect("Could not display image");
    let sobel = canvas.sobel();
    sobel.display().expect("Could not display image");
    sobel
        .save_extension("corro.png")
        .expect("Could not save image");
}

#[test]
fn parse_and_display() {
    let canvas = ppmify("./stop_1.ppm", false).expect("Implmentation is wrong");
    // let blur = canvas.blur();
    // let sobel = canvas.sobel();
    let edge = canvas.laplacian_edge_detection();
    // blur.display().expect("Could not display image");
    // sobel.display().expect("Could not display image");
    edge.display().expect("Could not display image");
}
