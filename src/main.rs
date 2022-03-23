use transform_rs::graphics::display::Pixel;
use transform_rs::parser::Parser;
fn main() {
    let outline = Pixel::new(235, 219, 178);
    let purplish = Pixel::new(17, 46, 81);
    let mut porygon = Parser::new_with_bg("./porygon_script", 512, 512, 255, outline, purplish);
    porygon.parse_file()
}
