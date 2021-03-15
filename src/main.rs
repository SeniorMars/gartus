mod graphics;
use std::io;

fn main() -> io::Result<()> {
    let image = graphics::display::Canvas::new(500, 500, 255);
    image.display()
}
