use std::{
    fs::File,
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

    pub fn plot(&mut self, new_color: Pixel, x: i32, y: i32) {
        let new_y = self.height as i32 - 1 - y;
        if x >= 0 && x < self.width as i32 && new_y >= 0 && new_y < self.height as i32 {
            let index = self.get_index(x as u32, new_y as u32);
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
        let mut content: String = format!("P3 {} {} {}\n", self.height, self.width, self.range);
        for pixel in self.pixels.iter() {
            content.push_str(&format!("{} {} {} ", &pixel.red, &pixel.green, &pixel.blue))
        }
        let mut child = Command::new("convert")
            .arg("-")
            .arg(file_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        child.stdin.as_mut().unwrap().write_all(&content.as_bytes())
    }

    pub fn display(&self) -> io::Result<()> {
        let mut content: String = format!("P3 {} {} {}\n", self.height, self.width, self.range);
        for pixel in self.pixels.iter() {
            content.push_str(&format!("{} {} {} ", &pixel.red, &pixel.green, &pixel.blue))
        }
        let mut child = Command::new("display")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        child.stdin.as_mut().unwrap().write_all(&content.as_bytes())
    }
}

// finally the homework :(
impl Canvas {
    pub fn draw_line(&mut self, color: Pixel, x0: f64, y0: f64, x1: f64, y1: f64) {
        let (x0, y0, x1, y1) = if x0 > x1 {
            (x1, y1, x0, y0)
        } else {
            (x0, y0, x1, y1)
        };
        let (mut x0, mut y0, x1, y1) = (x0 as i32, y0 as i32, x1 as i32, y1 as i32);
        // let slope = (y1 - y0) / (x1 - x0);
        let (delta_y, delta_x) = (2 * (y1 - y0), -2 * (x1 - x0));
        if delta_x == 0 {
            for y in y0..=y1 {
                self.plot(color, x0, y)
            }
            return ();
        }

        if (x1 - x0).abs() >= (y1 - y0).abs() {
            if delta_y > 0 {
                // octant 1
                let mut d = delta_y + delta_x / 2;
                for x in x0..=x1 {
                    self.plot(color, x, y0);
                    if d > 0 {
                        y0 += 1;
                        d += delta_x;
                    }
                    d += delta_y;
                }
            } else {
                // octant 8
                let mut d = delta_y - delta_x / 2;
                for x in x0..=x1 {
                    self.plot(color, x, y0);
                    if d < 0 {
                        y0 -= 1;
                        d -= delta_x;
                    }
                    d += delta_y;
                }
            }
        } else {
            if delta_y > 0 {
                // octant 2
                let mut d = delta_y / 2 + delta_x;
                for y in y0..=y1 {
                    self.plot(color, x0, y);
                    if d < 0 {
                        x0 += 1;
                        d += delta_y;
                    }
                    d += delta_x;
                }
            } else {
                // octant 7
                let mut d = delta_y / 2 - delta_x;
                for y in (y0..=y1).rev() {
                    self.plot(color, x0, y);
                    if d > 0 {
                        x0 += 1;
                        d += delta_y;
                    }
                    d -= delta_x;
                }
            }
        }
    }
}
