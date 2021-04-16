use crate::graphics::matrix::*;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    process::{Child, Command, Stdio},
};

#[derive(Default, Debug, Copy, Clone)]
pub struct Pixel {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
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

    pub fn plot(&mut self, new_color: Pixel, x: i32, y: i32) {
        // deal with negative numbers
        let x = if x < 0 { self.width as i32 - 1 + x } else { x };
        let y = if y < 0 { self.height as i32 - 1 + y } else { y };
        if self.upper_left_system {
            let index = self.index(x as u32, y as u32);
            self.pixels[index] = new_color
        } else {
            let new_y = self.height as i32 - 1 - y;
            if x >= 0 && x < self.width as i32 && new_y >= 0 && new_y < self.height as i32 {
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
        writer.write_all(format!("P6 {} {} {}\n", self.height, self.width, self.range).as_bytes())?;
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

    pub fn view_animation(&self, file_name: &str) -> io::Result<Child> {
        Command::new("animate")
            .arg(file_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
    }

    // TODO: make a better version
    pub fn animation(&self, file_name: &str) -> io::Result<Child> {
        println!("Making a new animation: {}.gif", file_name);
        Command::new("convert")
            .arg("-delay")
            .arg("2.7")
            .arg(&format!("anim/{}*", file_name))
            .arg(&format!("{}.gif", file_name))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
    }
}

#[allow(dead_code)]
impl Canvas {
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
