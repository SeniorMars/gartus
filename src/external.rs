use std::{
    ffi::OsStr,
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::Path,
    process::Command,
};

use crate::graphics::{colors::Rgb, display::Canvas};
/// ppmifizes an image so that it works with with this systems
// pub fn ppmify(file_name: &str) -> Result<Canvas<Rgb>, Box<dyn std::error::Error>> {
//     let path = Path::new(file_name);
//     if !path.exists() {
//         panic!("File Does not exit");
//     }
//     let ext = path
//         .extension()
//         .and_then(OsStr::to_str)
//         .unwrap_or_else(|| panic!("Check your file input: {:?}", path));
//     if ext != "ppm" {
//         let new_ext = path.with_extension("ppm");
//         let converted = new_ext.to_str().expect("Cannot get new file name");
//         let mut child = Command::new("convert")
//             .arg(file_name)
//             .arg(converted)
//             .spawn()?;
//         child.wait()?;
//     };
//     parse_ppm(path)
// }

// fn parse_ppm(path: &Path) -> Result<Canvas<Rgb>, Box<dyn std::error::Error>> {
//     let contents = fs::read_to_string(path).expect("Something went wrong reading the file");
//     let mut bytes = contents.bytes();
//     let canvas_type = bytes.next().expect("PPM alligns with specification");
//     let dimensions = bytes.next().expect("PPM alligns with specification");
//     let mut dimensions_spilt = dimensions.split_whitespace();
//     let height = dimensions_spilt
//         .next()
//         .expect("PPM alligns with specification")
//         .parse::<u32>()?;
//     dbg!("{}", height);
//     let width = dimensions_spilt
//         .next()
//         .expect("PPM alligns with specification")
//         .parse::<u32>()?;
//     dbg!("{}", width);
//     let color_depth = bytes
//         .next()
//         .expect("PPM alligns with specification")
//         .parse::<u8>()?;
//     dbg!("{}", color_depth);
//     let mut canvas = Canvas::with_capacity(height, width, color_depth, Rgb::default());
//     let pixels = match canvas_type {
//         "P6" => {
//             let rest_of_file = bytes.next().unwrap();
//             let mut buffer = rest_of_file.bytes();
//             let mut pixels = Vec::with_capacity(height as usize * height as usize);
//             while buffer.len() != 0 {
//                 let red = buffer.next().unwrap();
//                 let green = buffer.next().unwrap();
//                 let blue = buffer.next().unwrap();
//                 pixels.push(Rgb { red, green, blue })
//             }
//             pixels
//         }
//         _ => panic!("Unsupported Specification"),
//     };
//     canvas.fill_canvas(pixels);
//     Ok(canvas)
// }

#[test]
fn yea_yea() {
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
    canvas.save_ascii("pleasework.ppm").expect("Works");
}

#[test]
fn works() {
    // this will take so long.
    let file = File::open("example.ppm").unwrap();
    let reader = BufReader::new(file);
    let lines = reader.bytes();
    for i in lines {
        println!("{i:?}")
    }
    // let canvas = ppmify("pleasework.png").unwrap();
    // let contents =
    //     fs::read_to_string("pleasework.png").expect("Something went wrong reading the file");
    // println!("{}", canvas);
}
