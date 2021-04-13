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
        let xres: f64 = 500.0;
        let yres: f64 = 500.0;
        let mut screen = Canvas::new(xres as u32, yres as u32, 255);
        screen.set_line_color(0, 255, 0);

        // octants 1 and 5
        screen.draw_line(screen.line, 0.0, 0.0, xres - 1.0, yres - 1.0);
        screen.draw_line(screen.line, 0.0, 0.0, xres - 1.0, yres / 2.0);
        screen.draw_line(screen.line, xres - 1.0, yres - 1.0, 0.0, yres / 2.0);

        // octants 8 and 4
        screen.line.blue = 255;
        screen.draw_line(screen.line, 0.0, yres - 1.0, xres - 1.0, 0.0);
        screen.draw_line(screen.line, 0.0, yres - 1.0, xres - 1.0, yres / 2.0);
        screen.draw_line(screen.line, xres - 1.0, 0.0, 0.0, yres / 2.0);

        // octants 2 and 6
        screen.set_line_color(255, 0, 0);
        screen.draw_line(screen.line, 0.0, 0.0, xres / 2.0, yres - 1.0);
        screen.draw_line(screen.line, xres - 1.0, yres - 1.0, xres / 2.0, 0.0);

        // octants 7 and 3
        screen.line.blue = 255;
        screen.draw_line(screen.line, 0.0, yres - 1.0, xres / 2.0, 0.0);
        screen.draw_line(screen.line, xres - 1.0, 0.0, xres / 2.0, yres - 1.0);

        // horizontal and vertical
        screen.set_line_color(255, 255, 0);
        screen.draw_line(screen.line, 0.0, yres / 2.0, xres - 1.0, yres / 2.0);
        screen.draw_line(screen.line, xres / 2.0, 0.0, xres / 2.0, yres - 1.0);

        // saving
        screen.display()?;
        screen.save_binary("binary.ppm")?;
        screen.save_ascii("ascii.ppm")?;
        screen.save_extension("img.png")
    }
}
