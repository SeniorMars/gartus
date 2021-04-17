use crate::graphics::matrix::*;
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
        Self {red, green, blue}
    }
}

#[derive(Default, Debug, Clone)]
pub struct Canvas {
    height: u32,
    width: u32,
    range: u8,
    anim_index: u32,
    pixels: Vec<Pixel>,
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
            anim_index: 0,
            pixels: vec![Pixel::default(); (height * width) as usize],
            upper_left_system: false,
            line: Pixel::default(),
        }
    }

    pub fn new_with_bg(height: u32, width: u32, range: u8, bg: Pixel) -> Self {
        Self {
            height,
            width,
            range,
            anim_index: 0,
            pixels: vec![bg; (height * width) as usize],
            upper_left_system: false,
            line: Pixel::default(),
        }
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

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
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

    pub fn set_line_pixel(&mut self, new_color: Pixel) {
        self.line = new_color
    }

    pub fn set_line_color(&mut self, red: u8, green: u8, blue: u8) {
        self.line.red = red;
        self.line.green = green;
        self.line.blue = blue
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

// ploting / drawing
#[allow(dead_code)]
impl Canvas {
    pub fn fill(&mut self, x: i32, y: i32, fill_color: Pixel, boundary_color: Pixel) {
        let current = self.get_pixel(x, y);
        if current != boundary_color && current != fill_color {
            self.plot(fill_color, x as i32, y as i32);
            // assert!(x > self.width  as i32 || y > self.height as i32, "WTF");
            self.fill(x + 1, y, fill_color, boundary_color);
            self.fill(x, y + 1, fill_color, boundary_color);
            self.fill(x - 1, y, fill_color, boundary_color);
            self.fill(x, y - 1, fill_color, boundary_color);
            // self.fill(x + 1, y, fill_color, boundary_color);
            // self.fill(x, y + 1, fill_color, boundary_color);
            // self.fill(x - 1, y, fill_color, boundary_color);
            // self.fill(x, y - 1, fill_color, boundary_color);
            // self.fill(x - 1, y - 1, fill_color, boundary_color);
            // self.fill(x - 1, y + 1, fill_color, boundary_color);
            // self.fill(x + 1, y - 1, fill_color, boundary_color);
            // self.fill(x + 1, y + 1, fill_color, boundary_color);
        }
    }

    pub fn draw_lines(&mut self, matrix: &Matrix) {
        let mut iter = matrix.iter_by_point();
        while let Some(point) = iter.next() {
            let (x0, y0, _z0) = (point[0], point[1], point[3]);
            let (x1, y1, _z1) = match iter.next() {
                Some(p1) => (p1[0], p1[1], p1[2]),
                None => panic!("Need at least 2 points to draw"),
            };

            self.draw_line(self.line, x0, y0, x1, y1);
        }
    }

    pub fn draw_lines_for_animation(&mut self, matrix: &Matrix, filename: &str) -> io::Result<()> {
        let mut iter = matrix.iter_by_point();
        while let Some(point) = iter.next() {
            let (x0, y0, _z0) = (point[0], point[1], point[3]);
            let (x1, y1, _z1) = match iter.next() {
                Some(p1) => (p1[0], p1[1], p1[2]),
                None => panic!("Need at least 2 points to draw"),
            };

            self.save_binary(&format!("anim/{}{:08}.ppm", filename, self.anim_index))?;
            self.draw_line(self.line, x0, y0, x1, y1);
        }
        self.save_binary(&format!("anim/{}{:08}.ppm", filename, self.anim_index))
    }

    pub fn draw_line(&mut self, color: Pixel, x0: f64, y0: f64, x1: f64, y1: f64) {
        self.anim_index += 1;
        let (x0, y0, x1, y1) = if x0 > x1 {
            (x1, y1, x0, y0)
        } else {
            (x0, y0, x1, y1)
        };
        let (mut x0, mut y0, x1, y1) = (
            x0.round() as i32,
            y0.round() as i32,
            x1.round() as i32,
            y1.round() as i32,
        );
        let (delta_y, delta_x) = (2 * (y1 - y0), -2 * (x1 - x0));

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
                for y in (y1..=y0).rev() {
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
