use gartus::{
    graphics::{turtle::Turtle, config::CanvasConfig},
    prelude::{Canvas, Rgb},
};
use rand::seq::SliceRandom;

struct Maze {
    width: usize,
    height: usize,
    cells: Vec<bool>,
}

impl Maze {
    fn new(width: usize, height: usize) -> Self {
        let cells = vec![true; width * height];
        Self {
            width,
            height,
            cells,
        }
    }

    fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn set_cell(&mut self, x: usize, y: usize, value: bool) {
        let index = self.get_index(x, y);
        self.cells[index] = value;
    }

    fn get_neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut neighbors = vec![];

        if y > 1 {
            neighbors.push((x, y - 2));
        }
        if y < self.height - 2 {
            neighbors.push((x, y + 2));
        }
        if x > 1 {
            neighbors.push((x - 2, y));
        }
        if x < self.width - 2 {
            neighbors.push((x + 2, y));
        }

        neighbors
    }

    fn generate(
        &mut self,
        start_x: usize,
        start_y: usize,
        turtle: &mut Turtle<Rgb>,
        canvas: &mut Canvas<Rgb>,
    ) {
        let mut stack = Vec::new();
        let mut rng = rand::thread_rng();

        let mut x = start_x;
        let mut y = start_y;
        self.set_cell(x, y, false);

        stack.push((x, y));

        while !stack.is_empty() {
            let neighbors = self.get_neighbors(x, y);
            let neighbors = neighbors.choose_multiple(&mut rng, 4);

            let mut found = false;

            for &(nx, ny) in neighbors {
                if self.cells[self.get_index(nx, ny)] {
                    self.set_cell(nx, ny, false);
                    stack.push((nx, ny));

                    // Move the turtle between the current cell and the neighbor
                    let tx = (x + nx) / 2;
                    let ty = (y + ny) / 2;

                    // Draw a path between the cells
                    draw_path_with_turtle(x, y, tx, ty, canvas, turtle);

                    x = nx;
                    y = ny;
                    found = true;
                    break;
                }
            }

            if !found {
                if let Some((prev_x, prev_y)) = stack.pop() {
                    x = prev_x;
                    y = prev_y;
                }
            }
        }
    }
}

fn draw_path_with_turtle(
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    canvas: &mut Canvas<Rgb>,
    turtle: &mut Turtle<Rgb>,
) {
    let cell_size = 15; // Adjust the cell size based on your preference

    // Calculate the position for the top-left corner of the cell
    let cell_x = x1 * cell_size;
    let cell_y = y1 * cell_size;

    // Move the turtle to the top-left corner of the cell
    turtle.set_position(cell_x as u32, cell_y as u32);
    let cell_size = cell_size as f64;
    turtle.set_draw_mode(true);

    // Draw the top wall
    turtle.set_heading(0.0);
    turtle.move_turtle(canvas, cell_size);

    // Draw the right wall
    turtle.set_heading(90.0);
    turtle.move_turtle(canvas, cell_size);

    // Draw the bottom wall
    turtle.set_heading(180.0);
    turtle.move_turtle(canvas, cell_size);

    // Draw the left wall
    turtle.set_heading(270.0);
    turtle.move_turtle(canvas, cell_size);

    turtle.set_draw_mode(false);
}

fn main() {
    let width = 10; // Adjust the maze size as needed
    let height = 10;
    let mut canvas = Canvas::new_with_bg(width * 60, height * 60, 255, Rgb::new(255, 255, 255));
    canvas.set_config(CanvasConfig {
        upper_left_system: true,
        ..Default::default()
    });
    let mut turtle = Turtle::new(Rgb::new(0, 0, 0), 90.0, 0, 0);

    // Create a maze
    let mut maze = Maze::new(40, 40);
    maze.generate(0, 0, &mut turtle, &mut canvas);

    canvas.display().expect("Image is displayable");

    // Save or display the resulting maze
    // canvas.save_binary("./pics/maze.png").expect("Image is writeable");
}
