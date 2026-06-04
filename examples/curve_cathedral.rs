use gartus::graphics::colors::Rgb;
use gartus::parser::Parser;
use std::error::Error;
use std::fs;

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render curve cathedral:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;

    let line = Rgb::new(255, 255, 255);
    let background = Rgb::new(6, 8, 18);
    let mut parser =
        Parser::new_with_bg("scripts/curve_cathedral.cg", 900, 900, &line, &background);
    parser.set_display_enabled(false);
    parser.parse_file()?;

    println!("Saved final/curve_cathedral.png");
    Ok(())
}
