mod graphics;
// use graphics::display::*;
use std::io;

fn main() -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::graphics::display::*;
    use std::io;

    #[test]
    fn dw_line_test() -> io::Result<()> {
        let xres: i32 = 500;
        let yres: i32 = 500;
        let mut image = Canvas::new(xres as u32, yres as u32, 255);
        let mut line_color = Pixel {
            red: 0,
            green: 255,
            blue: 0,
        };

        // octants 1 and 5
        image.draw_line(line_color, 0, 0, xres - 1, yres - 1);
        image.draw_line(line_color, 0, 0, xres - 1, yres / 2);
        image.draw_line(line_color, xres - 1, yres - 1, 0, yres / 2);

        // octants 8 and 4
        line_color.blue = 255;
        image.draw_line(line_color, 0, yres - 1, xres - 1, 0);
        image.draw_line(line_color, 0, yres - 1, xres - 1, yres / 2);
        image.draw_line(line_color, xres - 1, 0, 0, yres / 2);

        // octants 2 and 6
        line_color = Pixel {
            red: 255,
            green: 0,
            blue: 0,
        };

        image.draw_line(line_color, 0, 0, xres / 2, yres - 1);
        image.draw_line(line_color, xres - 1, yres - 1, xres / 2, 0);

        // octants 7 and 3
        line_color.blue = 255;
        image.draw_line(line_color, 0, yres - 1, xres / 2, 0);
        image.draw_line(line_color, xres - 1, 0, xres / 2, yres - 1);

        // horizontal and vertical
        line_color = Pixel {
            red: 255,
            green: 255,
            blue: 0,
        };

        image.draw_line(line_color, 0, yres / 2, xres - 1, yres / 2);
        image.draw_line(line_color, xres / 2, 0, xres / 2, yres - 1);

        // saving
        image.display()?;
        image.save_binary("binary.ppm")?;
        image.save_ascii("ascii.ppm")?;
        image.save_extension("img.png")
    }
}
