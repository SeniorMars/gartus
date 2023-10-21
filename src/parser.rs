use crate::gmath::matrix::Matrix;
use crate::graphics::config::CanvasConfig;
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
        sphere: add a sphere to the edge matrix -
            takes 4 arguemnts (cx, cy, cz, r)
        torus: add a torus to the edge matrix -
            takes 5 arguemnts (cx, cy, cz, r1, r2)
        box: add a rectangular prism to the edge matrix -
            takes 6 arguemnts (x, y, z, width, height, depth)
        circle: add a circle to the edge matrix -
            takes 4 arguments (cx, cy, cz, r)
        hermite: add a hermite curve to the edge matrix -
            takes 8 arguments (x0, y0, x1, y1, rx0, ry0, rx1, ry1)
        bezier: add a third degree bezier curve to the edge matrix -
            takes 8 arguments (x0, y0, x1, y1, x2, y2, x3, y3)
        beziern: add a nth degree bezier curve to the edge matrix -
            takes the n-degree and (n + 2) * 2 arguments for points for x, y i.e., "n x0 y0 x1 y1 x2 y2 ... xn yn"
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
        color: changes the line's color -- should be ONLY RGB or a color constant
            takes 3 argument representing the new color parameters
            takes 1 argument representing the new color constant
        filter: apply a filter to the canvas
            takes 1 or 2 argument representing the filter to be applied and the threshold
            options: "grayscale", "invert", "emboss", "motionblur", "gaussian", "edge", "watercolor", "sepia"
            threshold: 0.0 - 1.0
        apply: apply the current transformation matrix to the edge matrix
        reset: reset transformation matrix and edge matrix
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
#[allow(clippy::module_name_repetitions)]
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
                "There was a error creating the {matrx_type} matrix with line: {line}:{line_num}"
            ),
            ParserError::CommandError(line_num, line) => {
                write!(f, "There was an unknown command: {line}:{line_num}")
            }
            ParserError::ArugmentError(line_num, line) => {
                write!(
                    f,
                    "Read spec. There was an error with parsing the arugments in line: {line}:{line_num}"
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
    /// use crate::gartus::prelude::{Rgb};
    /// use crate::gartus::parser::Parser;
    /// let purplish = Rgb::new(17, 46, 81);
    /// let porygon = Parser::new("tests/porygon_script", 512, 512, 255, &purplish);
    /// ```
    #[must_use]
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
    /// use crate::gartus::graphics::colors::Rgb;
    /// use crate::gartus::parser::Parser;
    /// let purplish = Rgb::new(17, 46, 81);
    /// let outline = Rgb::new(235, 219, 178);
    /// let porygon = Parser::new_with_bg("./tests/porygon_script", 512, 512, 255, &purplish, &outline);
    /// ```
    #[must_use]
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

    /// Get the config of the parser's canvas.
    pub fn config(&mut self) -> &mut CanvasConfig {
        self.canvas.config_mut()
    }

    #[allow(clippy::too_many_lines)]
    /// Parses and runs through the commands in `self.file_name`
    ///
    /// # Panics
    /// If there is an error reading the file or a line
    ///
    /// # Errors
    ///
    /// Returns a `ParserError`
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::graphics::colors::Rgb;
    /// use crate::gartus::parser::Parser;
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
                comment if comment.starts_with('#') => {}
                empty if empty.is_empty() => {}
                "quit" => { return Ok(())}
                command => {
                    self.handle_command(&mut iter, line_num, command)?;
                }
            }
        }
        Ok(())
    }

    fn handle_command(
        &mut self,
        iter: &mut std::iter::Enumerate<std::str::Lines<'_>>,
        cline_num: usize,
        command: &str,
    ) -> Result<(), ParserError> {
        match command {
            "line" => self.parse_line(iter.next().expect("Error reading line"))?,
            "scale" => self.parse_scale(iter.next().expect("Error reading line"))?,
            "move" => self.parse_move(iter.next().expect("Error reading line"))?,
            "rotate" => self.parse_rotate(iter.next().expect("Error reading line"))?,
            "reflect" => self.parse_reflect(iter.next().expect("Error reading line"))?,
            "shear" => self.parse_shear(iter.next().expect("Error reading line"))?,
            "color" => self.parse_color(iter.next().expect("Error reading line"))?,
            "ident" => self.trans_matrix = Matrix::identity_matrix(4),
            "apply" => self.edge_matrix = &self.trans_matrix * &self.edge_matrix,
            "display" => self.display(),
            "clear" => self.canvas.clear_canvas(),
            "reset" => self.parse_reset(),
            "circle" => self.parse_circle(iter.next().expect("Error reading line"))?,
            "hermite" => self.parse_hermite(iter.next().expect("Error reading line"))?,
            "bezier" => self.parse_bezier(iter.next().expect("Error reading line"))?,
            "beziern" => self.parse_beziern(iter.next().expect("Error reading line"))?,
            "save" => self.save(iter.next().expect("Error reading line"))?,
            "filter" => self.parse_filter(iter.next().expect("Error reading line"))?,
            _ => return Err(ParserError::CommandError(cline_num, command.to_string())),
        }
        Ok(())
    }

    fn parse_rotate(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split(' ').collect();
        let (axis, theta): (&str, f64) = (args[0], args[1].parse().expect("Error parsing number"));
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
        // dbg!(&theta, &axis);
        // dbg!(&rotate_matrix);
        // dbg!(&self.trans_matrix);
        self.trans_matrix = &rotate_matrix * &self.trans_matrix;
        Ok(())
    }

    fn parse_reflect(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
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
        Ok(())
    }

    fn parse_shear(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
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
        Ok(())
    }

    fn parse_color(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<u8>(line)
            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;

        if args.len() == 3 {
            let color = Rgb::new(args[0], args[1], args[2]);
            self.set_color(&color);
        } else {
            let color = Rgb::name_to_const(&line.trim().to_lowercase());
            if let Some(color) = color {
                self.set_color(&color);
            } else {
                return Err(ParserError::ArugmentError(line_num, line.to_string()));
            }
        }
        Ok(())
    }

    fn parse_circle(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;

        if args.len() == 4 {
            self.edge_matrix
                .add_circle(args[0], args[1], args[2], args[3], 0.001);
        } else {
            return Err(ParserError::ArugmentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_hermite(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;

        if args.len() == 8 {
            let p0 = (args[0], args[1]);
            let p1 = (args[2], args[3]);
            let r0 = (args[4], args[5]);
            let r1 = (args[6], args[7]);
            self.edge_matrix.add_hermite(p0, p1, r0, r1);
        } else {
            return Err(ParserError::ArugmentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_bezier(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;

        if args.len() == 8 {
            let p0 = (args[0], args[1]);
            let p1 = (args[2], args[3]);
            let p2 = (args[4], args[5]);
            let p3 = (args[6], args[7]);
            self.edge_matrix.add_bezier3(p0, p1, p2, p3);
        } else {
            return Err(ParserError::ArugmentError(line_num, line.to_string()));
        }
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn parse_beziern(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;

        let n_degree = args[0] as usize;
        if args.len() == (n_degree + 1) * 2 + 1 {
            let mut x_points = Vec::with_capacity(n_degree + 1);
            let mut y_points = Vec::with_capacity(n_degree + 1);

            for i in (1..args.len()).step_by(2) {
                x_points.push(args[i]);
                y_points.push(args[i + 1]);
            }
            self.edge_matrix.add_beziern(n_degree, &x_points, &y_points);
            Ok(())
        } else {
            Err(ParserError::ArugmentError(line_num, line.to_string()))
        }
    }

    fn parse_scale(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;

        if args.len() == 3 {
            let dilate_matrix = Matrix::scale(args[0], args[1], args[2]);
            self.trans_matrix = &dilate_matrix * &self.trans_matrix;
        } else {
            return Err(ParserError::ArugmentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_move(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;

        if args.len() == 3 {
            let translation_matrix = Matrix::translate(args[0], args[1], args[2]);
            self.trans_matrix = &translation_matrix * &self.trans_matrix;
        } else {
            return Err(ParserError::ArugmentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_line(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        match Parser::parse_as::<f64>(line) {
            Ok(edge) => self.edge_matrix.add_edge_vec(&edge),
            Err(_) => return Err(ParserError::ArugmentError(line_num, line.to_string())),
        }
        Ok(())
    }

    fn display(&mut self) {
        // self.canvas.clear_canvas();
        self.canvas.set_line_pixel(&self.color);
        // dbg!(&self.edge_matrix);
        self.canvas.draw_lines(&self.edge_matrix);
        self.canvas.display().unwrap();
    }

    fn save(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        // self.canvas.clear_canvas();
        self.canvas.set_line_pixel(&self.color);
        self.canvas.draw_lines(&self.edge_matrix);
        let (line_num, file_name) = line;
        if file_name.ends_with("png") || file_name.ends_with("ppm") || file_name.ends_with("jpg") {
            self.canvas.save_extension(file_name).unwrap();
            Ok(())
        } else {
            Err(ParserError::ArugmentError(line_num, file_name.to_string()))
        }
    }

    fn parse_as<T: FromStr>(line: &str) -> Result<Vec<T>, T::Err> {
        line.split(' ').map(str::parse).collect()
    }

    /// Set the parser's color.
    pub fn set_color(&mut self, color: &Rgb) {
        self.color = *color;
    }

    /// Get a reference to the parser's trans matrix.
    #[must_use]
    pub fn trans_matrix(&self) -> &Matrix {
        &self.trans_matrix
    }

    /// Get a reference to the parser's edge matrix.
    #[must_use]
    pub fn edge_matrix(&self) -> &Matrix {
        &self.edge_matrix
    }

    /// Allows you to modify the parser's edge matrix.
    ///
    /// * `func`: A function that takes a mutable reference to the parser's edge matrix.
    ///
    /// have fun
    pub fn edge_matrix_fun(&mut self, func: &dyn Fn(&mut Matrix)) {
        func(&mut self.edge_matrix);
    }

    #[allow(clippy::match_on_vec_items)]
    fn parse_filter(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split(' ').collect();

        match args.len() {
            2 => {
                self.canvas = match args[0] {
                    "solarize" => {
                        let threshold = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;
                        self.canvas.solarize(threshold)
                    }
                    "black_and_white" => {
                        let threshold = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;
                        self.canvas.black_and_white(threshold)
                    }
                    "brightness" => {
                        let brightness = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;
                        self.canvas.adjust_brightness(brightness)
                    }
                    "posterize" => {
                        let levels = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;
                        self.canvas.posterize(levels)
                    }
                    "gaussian" => {
                        let radius = args[1]
                            .parse::<f32>()
                            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;
                        self.canvas.gaussian_blur(radius)
                    }
                    "contrast" => {
                        let constrast = args[1]
                            .parse::<f32>()
                            .map_err(|_| ParserError::ArugmentError(line_num, line.to_string()))?;
                        self.canvas.adjust_contrast(constrast)
                    }
                    _ => return Err(ParserError::ArugmentError(line_num, line.to_string())),
                };
                Ok(())
            }
            1 => {
                self.canvas = match args[0] {
                    "grayscale" => self.canvas.grayscale(),
                    "sepia" => self.canvas.sepia(),
                    "reflect" => self.canvas.reflect(),
                    "blur" => self.canvas.blur(),
                    "sobel" => self.canvas.sobel(),
                    "invert" => self.canvas.invert(),
                    "edge" => self.canvas.laplacian_edge_detection(),
                    "emboss" => self.canvas.emboss(),
                    "oil" => self.canvas.oil_painting(),
                    "watercolor" => self.canvas.watercolor(),
                    _ => return Err(ParserError::ArugmentError(line_num, line.to_string())),
                };
                Ok(())
            }
            _ => Err(ParserError::ArugmentError(line_num, line.to_string())),
        }
    }

    fn parse_reset(&mut self) {
        self.canvas.clear_canvas();
        self.edge_matrix = Matrix::new(4, 0, Vec::new());
        self.trans_matrix = Matrix::identity_matrix(4);
    }
}

#[test]
fn parser_test() {
    let mut test = Parser::new(
        "./tests/script_testing",
        500,
        500,
        255,
        &Rgb::new(0, 255, 0),
    );
    test.parse_file().expect("Script is valid");
}
