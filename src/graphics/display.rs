use std::{
    fs::File,
    io::{self, BufWriter, Write},
    process::{Command, Stdio},
};

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct Pixel {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

#[allow(dead_code)]
impl Pixel {
    pub fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Canvas {
    height: u32,
    width: u32,
    range: u8,
    pixels: Vec<Pixel>,
    pub(in crate::graphics) anim_index: u32,
    pub upper_left_system: bool,
    pub line: Pixel,
}
// geass, kirby dance, bulbasuar, 3d shapes water bottle cylinder
#[allow(dead_code)]
impl Canvas {
    pub fn new(height: u32, width: u32, range: u8) -> Self {
        Self {
            height,
            width,
            range,
            pixels: vec![Pixel::default(); (height * width) as usize],
            anim_index: 0,
            upper_left_system: false,
            line: Pixel::default(),
        }
    }

    pub fn new_with_bg(height: u32, width: u32, range: u8, bg: Pixel) -> Self {
        Self {
            height,
            width,
            range,
            pixels: vec![bg; (height * width) as usize],
            anim_index: 0,
            upper_left_system: false,
            line: Pixel::default(),
        }
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }

    pub fn set_line_pixel(&mut self, new_color: Pixel) {
        self.line = new_color
    }

    pub fn set_line_color(&mut self, red: u8, green: u8, blue: u8) {
        self.line.red = red;
        self.line.green = green;
        self.line.blue = blue
    }

    fn iter(&self) -> impl Iterator<Item = &Pixel> + '_ {
        self.pixels.iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Pixel> + '_ {
        self.pixels.iter_mut()
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    fn deal_with_negs(&self, x: i32, y: i32) -> (i32, i32, i32, i32) {
        let (width, height) = (self.width as i32, self.height as i32);
        let x = if x > width {
            x % width
        } else if x < 0 {
            let r = x % width;
            if r != 0 {
                r + width
            } else {
                r
            }
        } else {
            x
        };

        let y = if y > height {
            y % height
        } else if y < 0 {
            let r = y % height;
            if r != 0 {
                r + height
            } else {
                r
            }
        } else {
            y
        };
        (x, y, width, height)
    }

    pub fn get_pixel(&self, x: i32, y: i32) -> Pixel {
        let (x, y, width, height) = self.deal_with_negs(x, y);
        // println!("i32:{} as {}", x, x as u32);
        if self.upper_left_system {
            let index = self.index(x as u32, y as u32);
            self.pixels[index]
        } else {
            let new_y = height - 1 - y;
            if x >= 0 && x < width && new_y >= 0 && new_y < height {
                let index = self.index(x as u32, new_y as u32);
                self.pixels[index]
            } else {
                // should never reach this
                return Pixel::default();
            }
        }
    }

    pub fn plot(&mut self, new_color: Pixel, x: i32, y: i32) {
        let (x, y, width, height) = self.deal_with_negs(x, y);
        if self.upper_left_system {
            let index = self.index(x as u32, y as u32);
            self.pixels[index] = new_color
        } else {
            let new_y = height - 1 - y;
            if x >= 0 && x < width && new_y >= 0 && new_y < height {
                let index = self.index(x as u32, new_y as u32);
                self.pixels[index] = new_color
            }
        }
    }

    pub fn clear_canvas(&mut self) {
        for i in self.iter_mut() {
            *i = Pixel::default()
        }
    }

    pub fn fill_color(&mut self, bg: Pixel) {
        for i in self.iter_mut() {
            *i = bg
        }
    }
}

// saving
#[allow(dead_code)]
impl Canvas {
    pub fn save_ascii(&self, file_name: &str) -> io::Result<()> {
        let mut file = File::create(file_name)?;
        let mut writer = BufWriter::new(&mut file);
        writeln!(writer, "P3 {} {} {}", self.height, self.width, self.range)?;
        for pixel in self.iter() {
            write!(writer, "{} {} {} ", pixel.red, pixel.green, pixel.blue)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn save_binary(&self, file_name: &str) -> io::Result<()> {
        let mut file = File::create(file_name)?;
        let mut writer = BufWriter::new(&mut file);
        writer
            .write_all(format!("P6 {} {} {}\n", self.height, self.width, self.range).as_bytes())?;
        for pixel in self.iter() {
            writer.write_all(&pixel.red.to_be_bytes())?;
            writer.write_all(&pixel.green.to_be_bytes())?;
            writer.write_all(&pixel.blue.to_be_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn save_extension(&self, file_name: &str) -> io::Result<()> {
        let mut content: String = format!("P3 {} {} {}\n", self.height, self.width, self.range);
        for pixel in self.iter() {
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
        for pixel in self.iter() {
            content.push_str(&format!("{} {} {} ", &pixel.red, &pixel.green, &pixel.blue))
        }
        let mut child = Command::new("display")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        child.stdin.as_mut().unwrap().write_all(&content.as_bytes())
    }
}
