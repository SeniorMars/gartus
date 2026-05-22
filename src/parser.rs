use crate::gmath::edge_matrix::EdgeMatrix;
use crate::gmath::matrix::Matrix;
use crate::graphics::{colors::Rgb, display::Canvas};
use std::fmt;
use std::{fs, io, path::Path, str::FromStr};

#[derive(Debug, Default)]
/**
```text
Goes through the file named filename and performs all of the actions listed in that file.
The file follows the following format:
     Every command is a single character that takes up a line
     Any command that requires arguments must have those arguments
     in the second line. The commands are as follows:
        sphere: add a sphere to the polygon matrix -
            takes 4 arguments (cx, cy, cz, r)
        torus: add a torus to the polygon matrix -
            takes 5 arguments (cx, cy, cz, r1, r2)
        box: add a rectangular prism to the polygon matrix -
            takes 6 arguments (x, y, z, width, height, depth)
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
            options: "grayscale", "sepia", "reflect", "blur", "sobel", "invert", "edge",
                "emboss", "oil", "watercolor", "solarize", "black_and_white",
                "brightness", "posterize", "gaussian", "contrast", "bilateral",
                "unsharp", "histogram", "clahe", "canny", "floyd_steinberg"
        apply: apply the current transformation matrix to the edge and polygon matrices
        clear: clear the edge and polygon matrices and the canvas
        reset: reset transformation matrix and clear matrices and canvas
        display: clear the screen, then
            draw the lines and polygons to the screen
            display the screen
        save: clear the screen, then
            draw the lines and polygons to the screen
            save the screen to a file -
            takes 1 argument (file name)
        quit: end parsing
```
*/
pub struct Parser {
    /// The name of the file being parsed
    file_name: String,
    /// The [`EdgeMatrix`] where points will be appended to draw onto the [Canvas]
    edge_matrix: EdgeMatrix,
    /// The [`EdgeMatrix`] where triangles will be appended to draw onto the [Canvas]
    polygon_matrix: EdgeMatrix,
    /// The [Matrix] that transformations will be applied to
    trans_matrix: Matrix,
    /// The [Canvas] where the image will be drawn in
    canvas: Canvas,
    /// Whether the canvas needs to be rerendered from the edge matrix.
    canvas_dirty: bool,
}

#[derive(Debug)]
/// Custom Errors for Parser
#[allow(clippy::module_name_repetitions)]
pub enum ParserError {
    /// An I/O error while reading, saving, or displaying.
    Io(io::Error),
    /// An error that specifies errors with Matrices..
    MatrixError(usize, String, String),
    /// An error that specifies errors with given arguments
    ArgumentError(usize, String),
    /// An unknown command for the Parser
    CommandError(usize, String),
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::Io(err) => write!(f, "I/O error: {err}"),
            ParserError::MatrixError(line_num, line, matrix_type) => write!(
                f,
                "There was an error creating the {matrix_type} matrix with line: {line}:{line_num}"
            ),
            ParserError::CommandError(line_num, line) => {
                write!(f, "There was an unknown command: {line}:{line_num}")
            }
            ParserError::ArgumentError(line_num, line) => {
                write!(
                    f,
                    "Read spec. There was an error parsing the arguments in line: {line}:{line_num}"
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
    /// * `color` - A [Rgb] that represents the color of the drawing line
    ///
    /// # Examples
    ///
    /// Basic usage:
    /// ```no_run
    /// use crate::gartus::prelude::{Rgb};
    /// use crate::gartus::parser::Parser;
    /// let purplish = Rgb::new(17, 46, 81);
    /// let porygon = Parser::new("tests/porygon_script", 512, 512, &purplish);
    /// ```
    #[must_use]
    pub fn new(file_name: &str, width: u32, height: u32, color: &Rgb) -> Self {
        Self {
            file_name: file_name.to_string(),
            edge_matrix: EdgeMatrix::new(),
            polygon_matrix: EdgeMatrix::new(),
            trans_matrix: Matrix::identity_matrix(4),
            canvas: Canvas::new(width, height, *color),
            canvas_dirty: true,
        }
    }

    /// Returns a parser that can parse through `file_name` that starts with [Canvas] filled by `bg`.
    ///
    /// # Arguments
    ///
    /// * `file_name` - The name of the file that will be created.
    /// * `height` - An unsigned int that will represent height of the [Canvas]
    /// * `width` - An unsigned int that will represent width of the [Canvas]
    /// * `color` - A [Rgb] that represents the color of the drawing line
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
    /// let porygon = Parser::new_with_bg("./tests/porygon_script", 512, 512, &purplish, &outline);
    /// ```
    #[must_use]
    pub fn new_with_bg(file_name: &str, width: u32, height: u32, color: &Rgb, bg: &Rgb) -> Self {
        let mut canvas = Canvas::new_with_bg(width, height, *bg);
        canvas.line = *color;
        Self {
            file_name: file_name.to_string(),
            edge_matrix: EdgeMatrix::new(),
            polygon_matrix: EdgeMatrix::new(),
            trans_matrix: Matrix::identity_matrix(4),
            canvas,
            canvas_dirty: true,
        }
    }

    /// Get a mutable reference to the parser's canvas.
    pub fn canvas_mut(&mut self) -> &mut Canvas {
        &mut self.canvas
    }

    #[allow(clippy::too_many_lines)]
    /// Parses and runs through the commands in `self.file_name`
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
    /// let mut porygon = Parser::new_with_bg("./tests/porygon_script", 512, 512, &purplish, &outline);
    /// porygon.parse_file();
    /// ```
    pub fn parse_file(&mut self) -> Result<(), ParserError> {
        let contents = fs::read_to_string(&self.file_name).map_err(ParserError::Io)?;
        let mut iter = contents.lines().enumerate();
        while let Some((line_num, line)) = iter.next() {
            match line.trim() {
                comment if comment.starts_with('#') => {}
                "" => {}
                "quit" => return Ok(()),
                command => {
                    self.handle_command(&mut iter, line_num, command)?;
                }
            }
        }
        Ok(())
    }

    fn next_arg_line<'a>(
        iter: &mut std::iter::Enumerate<std::str::Lines<'a>>,
        command_line: usize,
        command: &str,
    ) -> Result<(usize, &'a str), ParserError> {
        iter.next().ok_or_else(|| {
            ParserError::ArgumentError(command_line, format!("missing arguments for `{command}`"))
        })
    }

    fn handle_command(
        &mut self,
        iter: &mut std::iter::Enumerate<std::str::Lines<'_>>,
        cline_num: usize,
        command: &str,
    ) -> Result<(), ParserError> {
        match command {
            "line" => self.parse_line(Self::next_arg_line(iter, cline_num, command)?)?,
            "scale" => self.parse_scale(Self::next_arg_line(iter, cline_num, command)?)?,
            "move" => self.parse_move(Self::next_arg_line(iter, cline_num, command)?)?,
            "rotate" => self.parse_rotate(Self::next_arg_line(iter, cline_num, command)?)?,
            "reflect" => self.parse_reflect(Self::next_arg_line(iter, cline_num, command)?)?,
            "shear" => self.parse_shear(Self::next_arg_line(iter, cline_num, command)?)?,
            "color" => self.parse_color(Self::next_arg_line(iter, cline_num, command)?)?,
            "ident" => self.trans_matrix = Matrix::identity_matrix(4),
            "apply" => {
                self.edge_matrix = self.edge_matrix.apply(&self.trans_matrix);
                self.polygon_matrix = self.polygon_matrix.apply(&self.trans_matrix);
                self.canvas_dirty = true;
            }
            "display" => self.display()?,
            "clear" => {
                self.edge_matrix = EdgeMatrix::new();
                self.polygon_matrix = EdgeMatrix::new();
                self.canvas.clear_canvas();
                self.canvas_dirty = false;
            }
            "reset" => self.parse_reset(),
            "circle" => self.parse_circle(Self::next_arg_line(iter, cline_num, command)?)?,
            "hermite" => self.parse_hermite(Self::next_arg_line(iter, cline_num, command)?)?,
            "bezier" => self.parse_bezier(Self::next_arg_line(iter, cline_num, command)?)?,
            "beziern" => self.parse_beziern(Self::next_arg_line(iter, cline_num, command)?)?,
            "box" => self.parse_box(Self::next_arg_line(iter, cline_num, command)?)?,
            "sphere" => self.parse_sphere(Self::next_arg_line(iter, cline_num, command)?)?,
            "torus" => self.parse_torus(Self::next_arg_line(iter, cline_num, command)?)?,
            "save" => self.save(Self::next_arg_line(iter, cline_num, command)?)?,
            "filter" => self.parse_filter(Self::next_arg_line(iter, cline_num, command)?)?,
            _ => return Err(ParserError::CommandError(cline_num, command.to_string())),
        }
        Ok(())
    }

    fn parse_rotate(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        if args.len() != 2 {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        let (axis, theta): (&str, f64) = (
            args[0],
            args[1]
                .parse()
                .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?,
        );
        let rotate_matrix = match axis {
            "x" => Matrix::rotate_x(theta),
            "y" => Matrix::rotate_y(theta),
            "z" => Matrix::rotate_z(theta),
            _ => {
                return Err(ParserError::MatrixError(
                    line_num,
                    line.to_string(),
                    "rotate".to_string(),
                ));
            }
        };
        self.trans_matrix = &rotate_matrix * &self.trans_matrix;
        Ok(())
    }

    fn parse_reflect(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        if args.len() != 1 {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        let axis = args[0];
        let reflect_matrix = match axis {
            "x" => Matrix::reflect_xz(),
            "y" => Matrix::reflect_yz(),
            "z" => Matrix::reflect_xy(),
            _ => {
                return Err(ParserError::MatrixError(
                    line_num,
                    line.to_string(),
                    "reflect".to_string(),
                ));
            }
        };
        self.trans_matrix = &reflect_matrix * &self.trans_matrix;
        Ok(())
    }

    fn parse_shear(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        if args.len() != 3 {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        let (axis, sh_factor_one, sh_factor_two): (&str, f64, f64) = (
            args[0],
            args[1]
                .parse()
                .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?,
            args[2]
                .parse()
                .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?,
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
                ));
            }
        };
        self.trans_matrix = &shear_matrix * &self.trans_matrix;
        Ok(())
    }

    fn parse_color(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        match args.as_slice() {
            [name] => match Rgb::name_to_const(&name.to_lowercase()) {
                Some(color) => {
                    self.canvas.line = color;
                    self.canvas_dirty = true;
                    Ok(())
                }
                None => Err(ParserError::ArgumentError(line_num, line.to_string())),
            },
            [red, green, blue] => {
                let red = red
                    .parse()
                    .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                let green = green
                    .parse()
                    .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                let blue = blue
                    .parse()
                    .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                self.canvas.line = Rgb::new(red, green, blue);
                self.canvas_dirty = true;
                Ok(())
            }
            _ => Err(ParserError::ArgumentError(line_num, line.to_string())),
        }
    }

    fn parse_circle(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;

        if args.len() == 4 {
            self.edge_matrix
                .add_circle(args[0], args[1], args[2], args[3], 0.001);
            self.canvas_dirty = true;
        } else {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_hermite(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;

        if args.len() == 8 {
            let p0 = (args[0], args[1]);
            let p1 = (args[2], args[3]);
            let r0 = (args[4], args[5]);
            let r1 = (args[6], args[7]);
            self.edge_matrix.add_hermite(p0, p1, r0, r1);
            self.canvas_dirty = true;
        } else {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_bezier(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;

        if args.len() == 8 {
            let p0 = (args[0], args[1]);
            let p1 = (args[2], args[3]);
            let p2 = (args[4], args[5]);
            let p3 = (args[6], args[7]);
            self.edge_matrix.add_bezier3(p0, p1, p2, p3);
            self.canvas_dirty = true;
        } else {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_beziern(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let mut parts = line.split_whitespace();
        let n_degree = parts
            .next()
            .ok_or_else(|| ParserError::ArgumentError(line_num, line.to_string()))?
            .parse::<usize>()
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
        let coords = parts
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;

        if coords.len() == (n_degree + 1) * 2 {
            let mut x_points = Vec::with_capacity(n_degree + 1);
            let mut y_points = Vec::with_capacity(n_degree + 1);

            for i in (0..coords.len()).step_by(2) {
                x_points.push(coords[i]);
                y_points.push(coords[i + 1]);
            }
            self.edge_matrix.add_beziern(n_degree, &x_points, &y_points);
            self.canvas_dirty = true;
            Ok(())
        } else {
            Err(ParserError::ArgumentError(line_num, line.to_string()))
        }
    }

    fn parse_scale(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;

        if args.len() == 3 {
            let dilate_matrix = Matrix::scale(args[0], args[1], args[2]);
            self.trans_matrix = &dilate_matrix * &self.trans_matrix;
        } else {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_move(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;

        if args.len() == 3 {
            let translation_matrix = Matrix::translate(args[0], args[1], args[2]);
            self.trans_matrix = &translation_matrix * &self.trans_matrix;
        } else {
            return Err(ParserError::ArgumentError(line_num, line.to_string()));
        }
        Ok(())
    }

    fn parse_line(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        match Parser::parse_as::<f64>(line) {
            Ok(edge) if edge.len() == 6 => {
                self.edge_matrix
                    .push_edge(edge[0], edge[1], edge[2], edge[3], edge[4], edge[5]);
                self.canvas_dirty = true;
            }
            _ => return Err(ParserError::ArgumentError(line_num, line.to_string())),
        }
        Ok(())
    }

    fn render_scene(&mut self) {
        self.canvas.clear_canvas();
        self.canvas.try_draw_lines(&self.edge_matrix);
        self.canvas.draw_polygons(&self.polygon_matrix);
        self.canvas_dirty = false;
    }

    fn ensure_rendered(&mut self) {
        if self.canvas_dirty {
            self.render_scene();
        }
    }

    fn display(&mut self) -> Result<(), ParserError> {
        self.ensure_rendered();
        self.canvas.display().map_err(ParserError::Io)
    }

    fn save(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, file_name) = line;
        self.ensure_rendered();
        let extension = Path::new(file_name)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);

        match extension.as_deref() {
            Some("ppm") => self.canvas.save_binary(file_name).map_err(ParserError::Io),
            Some("png" | "jpg" | "jpeg") => self
                .canvas
                .save_extension(file_name)
                .map_err(ParserError::Io),
            _ => Err(ParserError::ArgumentError(line_num, file_name.to_string())),
        }
    }

    fn parse_as<T: FromStr>(line: &str) -> Result<Vec<T>, T::Err> {
        line.split_whitespace().map(str::parse).collect()
    }

    /// Set the parser's color.
    pub fn set_color(&mut self, color: &Rgb) {
        self.canvas.line = *color;
        self.canvas_dirty = true;
    }

    /// Get a reference to the parser's trans matrix.
    #[must_use]
    pub fn trans_matrix(&self) -> &Matrix {
        &self.trans_matrix
    }

    /// Get a reference to the parser's edge matrix.
    #[must_use]
    pub fn edge_matrix(&self) -> &EdgeMatrix {
        &self.edge_matrix
    }

    /// Allows you to modify the parser's edge matrix.
    ///
    /// * `func`: A function that takes a mutable reference to the parser's edge matrix.
    ///
    /// have fun
    pub fn with_edge_matrix<F>(&mut self, func: F)
    where
        F: FnOnce(&mut EdgeMatrix),
    {
        func(&mut self.edge_matrix);
        self.canvas_dirty = true;
    }

    /// Applies a function to the internal edge matrix and marks canvas dirty.
    pub fn edge_matrix_fun(&mut self, func: &dyn Fn(&mut EdgeMatrix)) {
        self.with_edge_matrix(|edges| func(edges));
    }

    fn parse_filter(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args: Vec<&str> = line.split_whitespace().collect();
        self.ensure_rendered();

        match args.len() {
            2 => {
                self.canvas = match args[0] {
                    "solarize" => {
                        let threshold = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.solarize(threshold)
                    }
                    "black_and_white" => {
                        let threshold = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.black_and_white(threshold)
                    }
                    "brightness" => {
                        let brightness = args[1]
                            .parse::<i16>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.adjust_brightness(brightness)
                    }
                    "posterize" => {
                        let levels = args[1]
                            .parse::<u8>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.posterize(levels)
                    }
                    "gaussian" => {
                        let radius = args[1]
                            .parse::<f32>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.gaussian_blur(radius)
                    }
                    "contrast" => {
                        let constrast = args[1]
                            .parse::<f32>()
                            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
                        self.canvas.adjust_contrast(constrast)
                    }
                    _ => return Err(ParserError::ArgumentError(line_num, line.to_string())),
                };
                self.canvas_dirty = false;
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
                    "bilateral" => self.canvas.bilateral_filter(2, 3.0, 32.0),
                    "unsharp" => self.canvas.unsharp_mask(1.0, 1.0),
                    "histogram" | "histogram_equalization" => self.canvas.histogram_equalization(),
                    "clahe" => self.canvas.clahe(32, 16),
                    "canny" => self.canvas.canny(40, 100),
                    "floyd_steinberg" | "floyd" => self.canvas.floyd_steinberg_dither(),
                    _ => return Err(ParserError::ArgumentError(line_num, line.to_string())),
                };
                self.canvas_dirty = false;
                Ok(())
            }
            _ => Err(ParserError::ArgumentError(line_num, line.to_string())),
        }
    }

    fn parse_reset(&mut self) {
        self.canvas.clear_canvas();
        self.edge_matrix = EdgeMatrix::new();
        self.polygon_matrix = EdgeMatrix::new();
        self.trans_matrix = Matrix::identity_matrix(4);
        self.canvas_dirty = false;
    }

    fn parse_box(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
        if args.len() == 6 {
            self.polygon_matrix
                .add_box((args[0], args[1], args[2]), args[3], args[4], args[5]);
            self.canvas_dirty = true;
            Ok(())
        } else {
            Err(ParserError::ArgumentError(line_num, line.to_string()))
        }
    }

    fn parse_sphere(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
        if args.len() == 4 {
            self.polygon_matrix
                .add_sphere((args[0], args[1], args[2]), args[3], 24);
            self.canvas_dirty = true;
            Ok(())
        } else {
            Err(ParserError::ArgumentError(line_num, line.to_string()))
        }
    }

    fn parse_torus(&mut self, line: (usize, &str)) -> Result<(), ParserError> {
        let (line_num, line) = line;
        let args = Parser::parse_as::<f64>(line)
            .map_err(|_| ParserError::ArgumentError(line_num, line.to_string()))?;
        if args.len() == 5 {
            self.polygon_matrix
                .add_torus((args[0], args[1], args[2]), args[3], args[4], 24);
            self.canvas_dirty = true;
            Ok(())
        } else {
            Err(ParserError::ArgumentError(line_num, line.to_string()))
        }
    }
}

#[ignore = "Display test, not meant for CI"]
#[test]
fn parser_test() {
    let mut test = Parser::new("./tests/script_testing", 500, 500, &Rgb::new(0, 255, 0));
    test.parse_file().expect("Script is valid");
}

#[cfg(test)]
mod tests {
    use super::{Parser, ParserError};
    use crate::graphics::colors::Rgb;
    use std::fs;
    use std::path::PathBuf;

    fn temp_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("gartus-parser-{name}-{}.cg", std::process::id()))
    }

    #[test]
    fn missing_file_returns_io_error() {
        let mut parser = Parser::new("/definitely/not/a/real/script.cg", 2, 2, &Rgb::GREEN);
        assert!(matches!(parser.parse_file(), Err(ParserError::Io(_))));
    }

    #[test]
    fn missing_argument_line_returns_parser_error() {
        let path = temp_file("missing-arg");
        fs::write(&path, "line\n").expect("write temp script");

        let mut parser = Parser::new(path.to_str().expect("utf8 path"), 2, 2, &Rgb::GREEN);
        let error = parser.parse_file().expect_err("script should fail");

        assert!(matches!(error, ParserError::ArgumentError(0, _)));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn extra_whitespace_in_arguments_is_accepted() {
        let path = temp_file("whitespace");
        fs::write(&path, "line\n0   0  0    1 1 0\n").expect("write temp script");

        let mut parser = Parser::new(path.to_str().expect("utf8 path"), 2, 2, &Rgb::GREEN);
        parser.parse_file().expect("script should parse");

        assert_eq!(parser.edge_matrix().cols(), 2);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn beziern_degree_must_be_integer() {
        let path = temp_file("beziern-degree");
        fs::write(&path, "beziern\n3.7 0 0 1 1 2 2 3 3\n").expect("write temp script");

        let mut parser = Parser::new(path.to_str().expect("utf8 path"), 2, 2, &Rgb::GREEN);
        let error = parser.parse_file().expect_err("script should fail");

        assert!(matches!(error, ParserError::ArgumentError(1, _)));
        let _ = fs::remove_file(path);
    }
}
