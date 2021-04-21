mod graphics;
mod parser;
use parser::Parser;
use graphics::display::Pixel;
fn main() {
    let outline = Pixel::new(235, 219, 178);
    let purplish = Pixel::new(17, 46, 81);
    let mut porygon = Parser::new_with_bg("./porygon_script", 512, 512, outline, purplish);
    porygon.parse_file()
}
