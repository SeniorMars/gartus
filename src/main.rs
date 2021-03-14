use std::{
    fs::{remove_file, File},
    io::{self, BufWriter, Write},
    process::{Command, Stdio},
};

#[derive(Default, Copy, Clone)]
pub struct Pixel {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

pub struct Canvas {
    height: u32,
    width: u32,
    range: u8,
    pixels: Vec<Pixel>,
    pub line: Pixel,
}

impl Canvas {
    pub fn new(height: u32, width: u32, range: u8) -> Self {
        Self {
            height,
            width,
            range,
            pixels: vec![Pixel::default(); (height * width) as usize],
            line: Pixel::default(),
        }
    }

    fn get_index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    pub fn plot(&mut self, new_color: Pixel, x: u32, y: u32) {
        let new_y = self.height - 1 - y;
        if x < self.width && new_y < self.height {
            let index = self.get_index(x, new_y);
            self.pixels[index] = new_color
        }
    }

    pub fn set_line_pixel(&mut self, new_color: Pixel) {
        self.line = new_color
    }

    pub fn set_line_color(&mut self, red: u8, green: u8, blue: u8) {
        self.line.red = red;
        self.line.green = green;
        self.line.blue = blue
    }

    pub fn clear_canvas(&mut self) {
        for i in self.pixels.iter_mut() {
            *i = Pixel::default()
        }
    }

    pub fn save_ascii(&self, file_name: &str) -> io::Result<()> {
        let mut file = File::create(file_name)?;
        let mut writer = BufWriter::new(&mut file);
        writeln!(writer, "P3 {} {} {}\n", self.height, self.width, self.range)?;
        for pixel in self.pixels.iter() {
            writer.write(&[pixel.red])?;
            writer.write(&[pixel.green])?;
            writer.write(&[pixel.blue])?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn save_binary(&self, file_name: &str) -> io::Result<()> {
        let mut file = File::create(file_name)?;
        let mut writer = BufWriter::new(&mut file);
        writer.write(format!("P6 {} {} {}\n", self.height, self.width, self.range).as_bytes())?;
        for pixel in self.pixels.iter() {
            writer.write(&pixel.red.to_be_bytes())?;
            writer.write(&pixel.green.to_be_bytes())?;
            writer.write(&pixel.blue.to_be_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn save_extension(&self, file_name: &str) -> io::Result<()> {
        let name = &file_name.split(".").collect::<Vec<&str>>();
        let ppm_name = format!("{}.ppm", name[0]);
        self.save_ascii(&ppm_name[..])?;
        Command::new("convert")
            .arg(&ppm_name[..])
            .arg(file_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?
            .wait_with_output()?;
        remove_file(ppm_name)
    }

    pub fn display(&self) -> io::Result<()> {
        let ppm_name = "pic.ppm";
        self.save_ascii(&ppm_name)?;
        Command::new("display")
            .arg(&ppm_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?
            .wait_with_output()?;
        remove_file(ppm_name)
    }
}

fn main() -> io::Result<()> {
    let image: Canvas = Canvas::new(500, 500, 255);
    image.display()
}
