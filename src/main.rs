extern crate rand;
use rand::Rng;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

const LENGTH: &str = "500";
const WIDTH: &str = "500";
const RANGE: &str = "255";

struct Pixel {
    red: u8,
    green: u8,
    blue: u8,
}

fn write_ppm(mut writer: impl Write) -> io::Result<()> {
    let mut rng = rand::thread_rng();
    writeln!(writer, "P3 {} {} {}\n", LENGTH, WIDTH, RANGE)?;
    for _row in 0..5 {
        let mut blockrow = "".to_string();
        for _col in 0..5 {
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

pub fn main() -> io::Result<()> {
    let mut file = File::create("image.ppm").unwrap();
    let writer = BufWriter::new(&mut file);
    write_ppm(writer)
}
