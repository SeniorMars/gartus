use crate::gmath::matrix::Matrix;
use crate::graphics::{colors::Rgb, display::Canvas};
use std::fmt;
use std::{fs, str::FromStr};

#[derive(Debug, Default)]
/**
```text
Goes through the file named filename and performs all of the actions listed in that file.
The file follows the following format:
     Every command is a single character that takes up a line
     Any command that requires arguments must have those arguments
     in the second line. The commands are as follows:
        circle: add a circle to the edge matrix -
            takes 4 arguments (cx, cy, cz, r)
        hermite: add a hermite curve to the edge matrix -
            takes 8 arguments (x0, y0, x1, y1, rx0, ry0, rx1, ry1)
        bezier: add a bezier curve to the edge matrix -
            takes 8 arguments (x0, y0, x1, y1, x2, y2, x3, y3)
        line: add a line to the edge matrix -
            takes 6 arguemnts (x0, y0, z0, x1, y1, z1)
        ident: set the transform matrix to the identity matrix -
        scale: create a scale matrix,
            then multiply the transform matrix by the scale matrix -
            takes 3 arguments (sx, sy, sz)
        move: create a translation matrix,
            then multiply the transform matrix by the translation matrix -
            takes 3 arguments (tx, ty, tz)
        rotate: create a rotation matrix,
            then multiply the transform matrix by the rotation matrix -
            takes 2 arguments (axis, theta) axis should be x y or z
        reflect: create a reflection matrix,
            then multiply the transform matrix by the rotation matrix -
            takes a argument (axis) - should be x y or z
        shear: create a shearing matrix,
            then multiply the transform matrix by the shearing matrix -
            takes 3 arguments (axis, sh_factor, sh_factor)  axis should be x, y, or z
        color: changes the line's color -- should be ONLY RGB
            takes 3 argument representing the new color parameters
        apply: apply the current transformation matrix to the edge matrix
        display: clear the screen, then
            draw the lines of the edge matrix to the screen
            display the screen
        save: clear the screen, then
            draw the lines of the edge matrix to the screen
            save the screen to a file -
            takes 1 argument (file name)
        quit: end parsing

```
*/
pub struct Parser {
    /// The name of the file being parsed
    file_name: String,
    /// The [Matrix] where points will be appended to draw onto the [Canvas]
    edge_matrix: Matrix,
    /// The [Matrix] that transformations will be applied to
    trans_matrix: Matrix,
    /// The [Canvas] where the image willl be drawn in
    canvas: Canvas<Rgb>,
    /// The default color of the drawing line
    color: Rgb,
}

#[derive(Debug)]
/// Custom Errors for Parser
pub enum ParserError {
    /// An error that specifies errors with Matrices..
    MatrixError(usize, String, String),
    /// An error that specifies errors with given arguments
    ArugmentError(usize, String),
    /// An unknown command for the Parser
    CommandError(usize, String),
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::MatrixError(line_num, line, matrx_type) => write!(
                f,
                "There was a error creating the {} matrix with line: {}:{}",
                matrx_type, line, line_num
            ),
            ParserError::CommandError(line_num, line) => {
                write!(f, "There was an unknown command: {}:{}", line, line_num)
            }
            ParserError::ArugmentError(line_num, line) => {
                write!(
                    f,
                    "There was an error with parsing line: {}:{}",
                    line, line_num
                )
            }
        }
    }
}

impl std::error::Error for ParserError {}

#[allow(dead_code)]
impl Parser {
    /// Returns a parser that can parse through `file_name`
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `color_depth` - An unsigned int that will represent maximum depth of colors in the [Canvas]
    /// * `color` - A [Rgb] that represents the color of the drawing line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::curves_rs::prelude::{Rgb};
    /// use crate::curves_rs::parser::Parser;
    /// let purplish = Rgb::new(17, 46, 81);
    /// let porygon = Parser::new("tests/porygon_script", 512, 512, 255, &purplish);
    /// ```
    pub fn new(file_name: &str, width: u32, height: u32, color_depth: u16, color: &Rgb) -> Self {
        let line = Rgb::default();
        Self {
            file_name: file_name.to_string(),
            edge_matrix: Matrix::new(4, 0, Vec::new()),
            trans_matrix: Matrix::identity_matrix(4),
            canvas: Canvas::new(width, height, color_depth, line),
            color: *color,
        }
    }

    /// Returns a parser that can parse through `file_name` that starts with [Canvas] filled by `bg`.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `range` - An unsigned int that will represent maximum depth of colors in the [Canvas]
    /// * `color_depth` - A [Rgb] that represents the color of the drawing line
    /// * `bg` - A [Rgb] the default background color of self.canvas
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::curves_rs::graphics::colors::Rgb;
    /// use crate::curves_rs::parser::Parser;
    /// let purplish = Rgb::new(17, 46, 81);
    /// let outline = Rgb::new(235, 219, 178);
    /// let porygon = Parser::new_with_bg("./tests/porygon_script", 512, 512, 255, &purplish, &outline);
    /// ```
    pub fn new_with_bg(
        file_name: &str,
        width: u32,
        height: u32,
        color_depth: u16,
        color: &Rgb,
        bg: &Rgb,
    ) -> Self {
        Self {
            file_name: file_name.to_string(),
            edge_matrix: Matrix::new(4, 0, Vec::new()),
            trans_matrix: Matrix::identity_matrix(4),
            canvas: Canvas::new_with_bg(width, height, color_depth, *bg),
            color: *color,
        }
    }

    /// Parses and runs through the commands in self.file_name
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::curves_rs::graphics::colors::Rgb;
    /// use crate::curves_rs::parser::Parser;
    /// let purplish = Rgb::new(17, 46, 81);
    /// let outline = Rgb::new(235, 219, 178);
    /// let mut porygon = Parser::new_with_bg("./tests/porygon_script", 512, 512, 255, &purplish, &outline);
    /// porygon.parse_file();
    /// ```
    pub fn parse_file(&mut self) -> Result<(), ParserError> {
        let contents =
            fs::read_to_string(&self.file_name).expect("Something went wrong reading the file");
        let mut iter = contents.lines().enumerate();
        while let Some((line_num, line)) = iter.next() {
            match line.trim() {
                other if other.starts_with('#') => {}
                empty if empty.is_empty() => {}
                "quit" => {}
                "line" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    match Parser::parse_as::<f64>(line.to_string()) {
                        Ok(edge) => self.edge_matrix.add_edge_vec(edge),
                        Err(_) => {
                            return Err(ParserError::ArugmentError(line_num, line.to_string()))
                        }
                    }
                }
                "scale" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    let args = match Parser::parse_as::<f64>(line.to_string()) {
                        Ok(args) => args,
                        Err(_) => {
                            return Err(ParserError::ArugmentError(line_num, line.to_string()))
                        }
                    };
                    assert_eq!(3, args.len());
                    let dilate_matrix = Matrix::scale(args[0], args[1], args[2]);
                    self.trans_matrix = &dilate_matrix * &self.trans_matrix;
                }
                "move" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    let args = match Parser::parse_as::<f64>(line.to_string()) {
                        Ok(args) => args,
                        Err(_) => {
                            return Err(ParserError::ArugmentError(line_num, line.to_string()))
                        }
                    };
                    assert_eq!(3, args.len());
                    let translation_matrix = Matrix::translate(args[0], args[1], args[2]);
                    self.trans_matrix = &translation_matrix * &self.trans_matrix;
                }
                "rotate" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    let args: Vec<&str> = line.split(' ').collect();
                    let (axis, theta): (&str, f64) =
                        (args[0], args[1].parse().expect("Error parsing number"));
                    let rotate_matrix = match axis {
                        "x" => Matrix::rotate_x(theta),
                        "y" => Matrix::rotate_y(theta),
                        "z" => Matrix::rotate_z(theta),
                        _ => {
                            return Err(ParserError::MatrixError(
                                line_num,
                                line.to_string(),
                                "rotate".to_string(),
                            ))
                        }
                    };
                    self.trans_matrix = &rotate_matrix * &self.trans_matrix;
                }
                "reflect" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    let axis = line.trim();
                    let reflect_matrix = match axis {
                        "x" => Matrix::reflect_xz(),
                        "y" => Matrix::reflect_yz(),
                        "z" => Matrix::reflect_xy(),
                        _ => {
                            return Err(ParserError::MatrixError(
                                line_num,
                                line.to_string(),
                                "reflect".to_string(),
                            ))
                        }
                    };
                    self.trans_matrix = &reflect_matrix * &self.trans_matrix;
                }
                "shear" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    let args: Vec<&str> = line.split(' ').collect();
                    let (axis, sh_factor_one, sh_factor_two): (&str, f64, f64) = (
                        args[0],
                        args[1].parse().expect("Error parsing number"),
                        args[2].parse().expect("Error parsing number"),
                    );
                    let shear_matrix = match axis {
                        "x" => Matrix::shearing_x(sh_factor_one, sh_factor_two),
                        "y" => Matrix::shearing_y(sh_factor_one, sh_factor_two),
                        "z" => Matrix::shearing_z(sh_factor_one, sh_factor_two),
                        _ => {
                            return Err(ParserError::MatrixError(
                                line_num,
                                line.to_string(),
                                "shear".to_string(),
                            ))
                        }
                    };
                    self.trans_matrix = &shear_matrix * &self.trans_matrix;
                }
                "color" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    let args = match Parser::parse_as::<u8>(line.to_string()) {
                        Ok(args) => args,
                        Err(_) => {
                            return Err(ParserError::ArugmentError(line_num, line.to_string()))
                        }
                    };
                    assert_eq!(3, args.len());
                    let color = Rgb::new(args[0], args[1], args[0]);
                    self.set_color(&color);
                }
                "ident" => {
                    self.trans_matrix = Matrix::identity_matrix(4);
                }
                "apply" => {
                    self.edge_matrix = &self.trans_matrix * &self.edge_matrix;
                }
                "display" => {
                    self.canvas.clear_canvas();
                    self.canvas.set_line_pixel(&self.color);
                    self.canvas.draw_lines(&self.edge_matrix);
                    self.canvas.display().unwrap();
                }
                "circle" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    let args = match Parser::parse_as::<f64>(line.to_string()) {
                        Ok(args) => args,
                        Err(_) => {
                            return Err(ParserError::ArugmentError(line_num, line.to_string()))
                        }
                    };
                    assert_eq!(4, args.len());
                    self.edge_matrix
                        .add_circle(args[0], args[1], args[2], args[3], 0.001);
                }
                "hermite" => {
                    let (line_num, line) = iter.next().expect("Error reading line");
                    let args = match Parser::parse_as::<f64>(line.to_string()) {
                        Ok(args) => args,
                        Err(_) => {
                            return Err(ParserError::ArugmentError(line_num, line.to_string()))
                        }
                    };
                    assert_eq!(8, args.len());
                    let p0 = (args[0], args[1]);
                    let p1 = (args[2], args[3]);
                    let r0 = (args[4], args[5]);
                    let r1 = (args[6], args[7]);
                    self.edge_matrix.add_hermite(p0, p1, r0, r1)
                }
                "bezier" => {
                    let (_, line) = iter.next().expect("Error reading line");
                    let args = Parser::parse_as::<f64>(line.to_string()).unwrap();
                    assert_eq!(8, args.len());
                    // bezier: add a bezier curve to the edge matrix -
                    //     takes 8 arguments (x0, y0, x1, y1, x2, y2, x3, y3)
                    // todo!()
                }
                "save" => {
                    self.canvas.clear_canvas();
                    self.canvas.set_line_pixel(&self.color);
                    self.canvas.draw_lines(&self.edge_matrix);
                    let (_, file_name) = iter.next().expect("Error reading line");
                    self.canvas.save_extension(file_name).unwrap();
                }
                _ => return Err(ParserError::CommandError(line_num, line.to_string())),
            }
        }
        Ok(())
    }

    fn parse_as<T: FromStr>(line: String) -> Result<Vec<T>, T::Err> {
        line.split(' ').map(|n| n.parse::<T>()).collect()
    }

    /// Set the parser's color.
    pub fn set_color(&mut self, color: &Rgb) {
        self.color = *color;
    }

    /// Get a reference to the parser's trans matrix.
    pub fn trans_matrix(&self) -> &Matrix {
        &self.trans_matrix
    }

    /// Get a reference to the parser's edge matrix.
    pub fn edge_matrix(&self) -> &Matrix {
        &self.edge_matrix
    }
}
