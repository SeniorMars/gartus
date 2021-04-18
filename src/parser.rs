use crate::graphics::{
    display::{Canvas, Pixel},
    matrix::Matrix,
};
use std::fs;

/// Goes through the file named filename and performs all of the actions listed in that file.
/// The file follows the following format:
///      Every command is a single character that takes up a line
///      Any command that requires arguments must have those arguments in the second line.
///      The commands are as follows:
///          line: add a line to the edge matrix -
///                takes 6 arguemnts (x0, y0, z0, x1, y1, z1)
///          ident: set the transform matrix to the identity matrix -
///          scale: create a scale matrix,
///                 then multiply the transform matrix by the scale matrix -
///                 takes 3 arguments (sx, sy, sz)
///          move: create a translation matrix,
///                then multiply the transform matrix by the translation matrix -
///                takes 3 arguments (tx, ty, tz)
///          rotate: create a rotation matrix,
///                  then multiply the transform matrix by the rotation matrix -
///                  takes 2 arguments (axis, theta) axis should be x y or z
///          apply: apply the current transformation matrix to the edge matrix
///          display: clear the screen, then
///                   draw the lines of the edge matrix to the screen
///                   display the screen
///          save: clear the screen, then
///                draw the lines of the edge matrix to the screen
///                save the screen to a file -
///                takes 1 argument (file name)
///          quit: end parsing

pub struct Parser {
    file_name: String,
    edge_matrix: Matrix,
    trans_matrix: Matrix,
    canvas: Canvas,
    color: Pixel,
}

impl Parser {
    pub fn new(file_name: &str, width: u32, height: u32, color: Pixel) -> Self {
        Self {
            file_name: file_name.to_string(),
            edge_matrix: Matrix::new(0, 4, Vec::new()),
            trans_matrix: Matrix::identity_matrix(4),
            canvas: Canvas::new(width, height, 255),
            color,
        }
    }

    pub fn parse_file(&mut self) {
        let contents =
            fs::read_to_string(&self.file_name).expect("Something went wrong reading the file");
        let lines = contents.lines();
        let mut iter = lines.clone().enumerate();
        while let Some((line_num, line)) = iter.next() {
            match line.trim() {
                other if other.starts_with("#") => {}
                "quit" => {}
                "line" => {
                    let next_line = lines.clone().nth(line_num + 1).unwrap().trim();
                    let edge = Parser::parse_as_float(next_line.to_string());
                    assert_eq!(6, edge.len());
                    self.edge_matrix.add_edge_vec(edge);
                    iter.next();
                }
                "scale" => {
                    let next_line = lines.clone().nth(line_num + 1).unwrap().trim();
                    let args = Parser::parse_as_float(next_line.to_string());
                    assert_eq!(3, args.len());
                    let dilate_matrix = Matrix::scale(args[0], args[1], args[2]);
                    self.trans_matrix = self.trans_matrix.mult_matrix(&dilate_matrix);
                    iter.next();
                }
                "move" => {
                    let next_line = lines.clone().nth(line_num + 1).unwrap().trim();
                    let args = Parser::parse_as_float(next_line.to_string());
                    assert_eq!(3, args.len());
                    let translation_matrix = Matrix::translate(args[0], args[1], args[2]);
                    self.trans_matrix = self.trans_matrix.mult_matrix(&translation_matrix);
                    iter.next();
                }
                "rotate" => {
                    let next_line = lines.clone().nth(line_num + 1).unwrap().trim();
                    let args: Vec<&str> = next_line.split(" ").collect();
                    let (axis, theta): (&str, f64) =
                        (args[0], args[1].parse().expect("Error parsing number"));
                    let rotate_matrix = match axis {
                        "x" => Matrix::rotate_x(theta),
                        "y" => Matrix::rotate_y(theta),
                        "z" => Matrix::rotate_z(theta),
                        _ => panic!("Unknown axis on line: {}", line_num + 1),
                    };
                    self.trans_matrix = self.trans_matrix.mult_matrix(&rotate_matrix);
                    iter.next();
                }
                "ident" => {
                    self.trans_matrix = Matrix::identity_matrix(4);
                }
                "apply" => {
                    self.edge_matrix = self.edge_matrix.mult_matrix(&self.trans_matrix);
                }
                "display" => {
                    self.canvas.clear_canvas();
                    self.canvas.set_line_pixel(self.color);
                    self.canvas.draw_lines(&self.edge_matrix);
                    self.canvas.display().unwrap();
                }
                "save" => {
                    self.canvas.clear_canvas();
                    self.canvas.set_line_pixel(self.color);
                    self.canvas.draw_lines(&self.edge_matrix);
                    let file_name = lines.clone().nth(line_num + 1).unwrap().trim();
                    self.canvas.save_extension(file_name).unwrap();
                    iter.next();
                }
                _ => panic!("Command not recognized on line {}: {}", line_num, line),
            }
        }
    }

    fn parse_as_float(line: String) -> Vec<f64> {
        line.split(" ").map(|n| n.parse::<f64>().unwrap()).collect()
    }
}
