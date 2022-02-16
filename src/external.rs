use crate::graphics::display::CanvasConfig;
use std::{collections::VecDeque, fs::File};
use std::{
    ffi::OsStr,
    io::{BufReader, Read},
    path::Path,
    process::Command,
};

use crate::graphics::{colors::Rgb, display::Canvas};
/// ppmifizes an image so that it works with with this systems
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
    if ext != "ppm" {
        let new_ext = path.with_extension("ppm");
        let converted = new_ext.to_str().expect("Cannot get new file name");
        let mut child = Command::new("convert")
            .arg(file_name)
            .arg(converted)
            .spawn()?;
        child.wait()?;
    };
    parse_ppm(&path.with_extension("ppm"), pos_glitch)
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
    // this will take so long and is naive
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut bytes = reader
        .bytes()
        .map(|pos_byte| pos_byte.expect("File Follows Spec"))
        .collect::<VecDeque<u8>>();

    let p_type = (bytes.pop_front().unwrap(), bytes.pop_front().unwrap());
    let canvas_type = match p_type {
        (80, 51) => "P3",
        (80, 54) => "P6",
        (_, _) => panic!("Unsupported spec"),
    };

    // pop off newline
    bytes.pop_front();

    let mut height_vec = Vec::new();
    let mut width_vec = Vec::new();
    // We have to loop as a Canvas's width/height can be very large
    // inccorectly gathers width and height wrong. May look cool.
    // if pos_glitch is on, then use display_alt to see image.
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
    match canvas_type {
        "P3" => {
            while !bytes.is_empty() {
                let mut red = Vec::with_capacity(3);
                byte_vec_fill(&mut bytes, &mut red);
                let mut green = Vec::with_capacity(3);
                byte_vec_fill(&mut bytes, &mut green);
                let mut blue = Vec::with_capacity(3);
                byte_vec_fill(&mut bytes, &mut blue);
                let (red, green, blue) = (
                    vec_to_int(red)
                        .try_into()
                        .expect("File does not follow ppm spec"),
                    vec_to_int(green)
                        .try_into()
                        .expect("File does not follow ppm spec"),
                    vec_to_int(blue)
                        .try_into()
                        .expect("File does not follow ppm spec"),
                );
                pixels.push(Rgb::new(red, green, blue));
            }
        }
        "P6" => {
            while !bytes.is_empty() {
                let red = bytes.pop_front().expect("File does not follow ppm spec");
                let green = bytes.pop_front().expect("File does not follow ppm spec");
                let blue = bytes.pop_front().expect("File does not follow ppm spec");
                pixels.push(Rgb::new(red, green, blue));
            }
        }
        _ => unreachable!(),
    };
    canvas.fill_canvas(pixels);
    Ok(canvas)
}

#[test]
fn file_parse_test() {
    let colors = vec![
        Rgb::GREEN,
        Rgb::BLUE,
        Rgb::RED,
        Rgb::GREEN,
        Rgb::BLUE,
        Rgb::RED,
        Rgb::GREEN,
        Rgb::BLUE,
        Rgb::RED,
    ];
    let mut canvas = Canvas::with_capacity(3, 3, 255, Rgb::BLACK);
    canvas.fill_canvas(colors);
    canvas.save_binary("./pics/pleasework.ppm").expect("Works");
    let other = ppmify("./pics/pleasework.ppm", true).expect("Life is wrong");
    assert_eq!(canvas.pixels(), other.pixels());
}

#[test]
fn external_fun() {
    let pos_glitch = true;
    let mut canvas = ppmify("./pics/index.png", pos_glitch).expect("Implmentation is wrong");
    canvas.set_config(CanvasConfig::new(false, pos_glitch));
    canvas.display().expect("Could not display image");
    canvas.blur();
    canvas.display().expect("Could not display image");
    canvas
        .save_extension("corro.png")
        .expect("Could not save image");
}
