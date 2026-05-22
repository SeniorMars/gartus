use std::error::Error;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::graphics::{colors::Rgb, display::Canvas};
/// Converts an image to a [`Canvas`], converting non-PPM images to a sibling `.ppm` file first.
///
/// # Arguments
/// * `file_name` - The name of the file to load.
/// * `pos_glitch` - Whether to swap the parsed canvas dimensions after loading.
///
/// # Note
/// Non-PPM inputs are converted through a temporary `.ppm` file that is removed after parsing.
///
/// # Errors
/// todo!()
///
/// # Examples
///
/// Basic usage:
///```no_run
/// use crate::gartus::prelude::{Canvas, Rgb};
/// use crate::gartus::external;
/// let colors = vec![
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
///     Rgb::GREEN,
///     Rgb::BLUE,
///     Rgb::RED,
/// ];
/// let mut canvas = Canvas::new(3, 3, Rgb::BLACK);
/// canvas.fill_canvas(colors);
/// canvas.save_binary("./works.ppm").expect("Works");
/// let other = external::ppmify("./works.ppm", false).expect("Life is wrong");
/// assert_eq!(canvas.pixels(), other.pixels());
/// ```
pub fn ppmify(file_name: &str, pos_glitch: bool) -> Result<Canvas, Box<dyn std::error::Error>> {
    let path = Path::new(file_name);
    if !path.exists() {
        return Err(format!("File does not exist: {file_name}").into());
    }

    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or("Invalid file extension")?;

    let canvas = if ext == "ppm" {
        parse_ppm(path)?
    } else {
        let converted = temp_ppm_path(path)?;
        let status = Command::new("magick").arg(path).arg(&converted).status()?;
        if !status.success() {
            let _ = fs::remove_file(&converted);
            return Err("ImageMagick `magick` failed to convert image to ppm".into());
        }

        let parsed = parse_ppm(&converted);
        let _ = fs::remove_file(&converted);
        parsed?
    };

    Ok(if pos_glitch {
        dimension_glitch(&canvas)
    } else {
        canvas
    })
}

fn temp_ppm_path(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or("Invalid file name")?;
    Ok(std::env::temp_dir().join(format!("gartus-ppmify-{stem}-{}.ppm", std::process::id())))
}

fn dimension_glitch(canvas: &Canvas) -> Canvas {
    let mut glitched = Canvas::new(canvas.height(), canvas.width(), canvas.line);
    glitched.fill_canvas(canvas.pixels().to_vec());
    glitched
}

fn next_token(buffer: &[u8], cursor: &mut usize) -> Option<String> {
    loop {
        while *cursor < buffer.len() && buffer[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        }

        if *cursor < buffer.len() && buffer[*cursor] == b'#' {
            while *cursor < buffer.len() && buffer[*cursor] != b'\n' {
                *cursor += 1;
            }
            continue;
        }

        break;
    }

    if *cursor >= buffer.len() {
        return None;
    }

    let start = *cursor;
    while *cursor < buffer.len()
        && !buffer[*cursor].is_ascii_whitespace()
        && buffer[*cursor] != b'#'
    {
        *cursor += 1;
    }

    Some(String::from_utf8_lossy(&buffer[start..*cursor]).into_owned())
}

fn scale_channel(value: u16, maxval: u16) -> Result<u8, Box<dyn Error>> {
    if value > maxval {
        return Err(format!("PPM channel value {value} exceeds maxval {maxval}").into());
    }

    Ok(u8::try_from((u32::from(value) * 255 + u32::from(maxval) / 2) / u32::from(maxval)).unwrap_or(255))
}

fn parse_ppm(path: &Path) -> Result<Canvas, Box<dyn Error>> {
    let buffer = fs::read(path)?;
    let mut cursor = 0;

    let magic = next_token(&buffer, &mut cursor).ok_or("Invalid PPM file: missing magic")?;
    let width = next_token(&buffer, &mut cursor)
        .ok_or("Invalid PPM file: missing width")?
        .parse::<u32>()?;
    let height = next_token(&buffer, &mut cursor)
        .ok_or("Invalid PPM file: missing height")?
        .parse::<u32>()?;
    let maxval = next_token(&buffer, &mut cursor)
        .ok_or("Invalid PPM file: missing maxval")?
        .parse::<u16>()?;

    if maxval == 0 || maxval > 255 {
        return Err(format!("unsupported PPM maxval {maxval}; only 1..=255 is supported").into());
    }

    let pixel_count = u64::from(width) * u64::from(height);
    let pixel_count = usize::try_from(pixel_count).map_err(|_| "PPM image too large")?;
    let mut pixels = Vec::with_capacity(pixel_count);

    match magic.as_str() {
        "P3" => {
            for _ in 0..pixel_count {
                let red = next_token(&buffer, &mut cursor)
                    .ok_or("Invalid PPM file: missing red channel")?
                    .parse::<u16>()?;
                let green = next_token(&buffer, &mut cursor)
                    .ok_or("Invalid PPM file: missing green channel")?
                    .parse::<u16>()?;
                let blue = next_token(&buffer, &mut cursor)
                    .ok_or("Invalid PPM file: missing blue channel")?
                    .parse::<u16>()?;

                pixels.push(Rgb::new(
                    scale_channel(red, maxval)?,
                    scale_channel(green, maxval)?,
                    scale_channel(blue, maxval)?,
                ));
            }
        }
        "P6" => {
            if cursor >= buffer.len() || !buffer[cursor].is_ascii_whitespace() {
                return Err("Invalid PPM file: missing binary data separator".into());
            }
            cursor += 1;

            let needed = pixel_count
                .checked_mul(3)
                .ok_or("PPM image data is too large")?;
            if buffer.len().saturating_sub(cursor) < needed {
                return Err(format!(
                    "Invalid PPM file: expected {needed} bytes of pixel data, found {}",
                    buffer.len().saturating_sub(cursor)
                )
                .into());
            }

            for chunk in buffer[cursor..cursor + needed].chunks_exact(3) {
                pixels.push(Rgb::new(
                    scale_channel(u16::from(chunk[0]), maxval)?,
                    scale_channel(u16::from(chunk[1]), maxval)?,
                    scale_channel(u16::from(chunk[2]), maxval)?,
                ));
            }
        }
        _ => return Err(format!("Invalid PPM file: unsupported magic {magic}").into()),
    }

    let mut canvas = Canvas::new(width, height, Rgb::default());
    canvas.fill_canvas(pixels);
    Ok(canvas)
}

#[cfg(test)]
mod tests {
    use super::ppmify;
    use crate::graphics::colors::Rgb;
    use std::fs;
    use std::path::PathBuf;

    fn temp_file(name: &str, extension: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "gartus-external-{name}-{}.{}",
            std::process::id(),
            extension
        ))
    }

    #[test]
    fn parses_p3_comments_whitespace_and_scaled_maxval() {
        let path = temp_file("comments", "ppm");
        fs::write(
            &path,
            b"P3
# exported in 2026
2   1
# max value
100
100 0 50   0 100 25
",
        )
        .expect("write temp ppm");

        let canvas = ppmify(path.to_str().expect("utf8 path"), false).expect("parse ppm");

        assert_eq!(canvas.width(), 2);
        assert_eq!(canvas.height(), 1);
        assert_eq!(canvas.pixels()[0], Rgb::new(255, 0, 128));
        assert_eq!(canvas.pixels()[1], Rgb::new(0, 255, 64));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn parses_uppercase_ppm_extension_without_conversion() {
        let path = temp_file("uppercase", "PPM");
        fs::write(&path, b"P6\n1 1\n255\n\x01\x02\x03").expect("write temp ppm");

        let canvas = ppmify(path.to_str().expect("utf8 path"), false).expect("parse ppm");

        assert_eq!(canvas.pixels(), &[Rgb::new(1, 2, 3)]);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn truncated_p6_returns_error() {
        let path = temp_file("truncated", "ppm");
        fs::write(&path, b"P6\n2 1\n255\n\x01\x02\x03").expect("write temp ppm");

        let error = ppmify(path.to_str().expect("utf8 path"), false).expect_err("should fail");

        assert!(error.to_string().contains("expected 6 bytes"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn pos_glitch_is_applied_after_parsing() {
        let path = temp_file("glitch", "ppm");
        fs::write(&path, b"P3\n2 1\n255\n1 2 3 4 5 6\n").expect("write temp ppm");

        let canvas = ppmify(path.to_str().expect("utf8 path"), true).expect("parse ppm");

        assert_eq!(canvas.width(), 1);
        assert_eq!(canvas.height(), 2);
        assert_eq!(canvas.pixels(), &[Rgb::new(1, 2, 3), Rgb::new(4, 5, 6)]);
        let _ = fs::remove_file(path);
    }
}

#[test]
#[ignore = "requires external files and a display"]
fn external_fun() {
    let pos_glitch = true;
    let canvas = ppmify("./corro.png", pos_glitch).expect("Implmentation is wrong");
    canvas.display().expect("Could not display image");
    let sobel = canvas.sobel();
    sobel.display().expect("Could not display image");
    sobel
        .save_extension("pics/corro.png")
        .expect("Could not save image");
}

#[test]
#[ignore = "requires external files and a display"]
fn command_block() {
    let pos_glitch = true;
    let canvas = ppmify("./CAR.png", pos_glitch).expect("Implmentation is wrong");
    canvas.display().expect("Could not display image");
    let sobel = canvas.sobel();
    sobel.display().expect("Could not display image");
    sobel
        .save_extension("pics/corro.png")
        .expect("Could not save image");
}

#[test]
#[ignore = "requires external files and a display"]
fn parse_and_display() {
    let canvas = ppmify("./stop_1.ppm", false).expect("Implmentation is wrong");
    // let blur = canvas.blur();
    // let sobel = canvas.sobel();
    let edge = canvas.laplacian_edge_detection();
    // blur.display().expect("Could not display image");
    // sobel.display().expect("Could not display image");
    edge.display().expect("Could not display image");
}
