use gartus::graphics::colors::*;
use gartus::parser::Parser;

#[test]
#[ignore = "visual script uses display/ImageMagick"]
fn script_1() {
    let mut pig = Parser::new("./scripts/pig.cg", 400, 400, &Rgb::new(0, 255, 0));

    pig.canvas_mut().upper_left_origin = false;

    // pig.edge_matrix_fun(&|edges| {
    //     let mut i = 400;
    //     while i > -500 {
    //         let curr_i = i as f64;
    //         edges.push_edge(50.0 + curr_i, 450.0, 0.0, 100.0 + curr_i, 450.0, 0.0);
    //         edges.push_edge(50.0, 450.0, 0.0, 50.0, 400.0 + curr_i, 0.0);
    //         edges.push_edge(100.0, 450.0 + curr_i, 0.0, 100.0, 400.0 + curr_i, 0.0);
    //         edges.push_edge(100.0, 400.0, 0.0, 50.0, 400.0, 0.0);
    //
    //         edges.push_edge(200.0 + curr_i, 450.0 + curr_i, 0.0, 250.0, 450.0 + curr_i, 0.0);
    //         edges.push_edge(200.0, 450.0 + curr_i, 0.0, 200.0, 400.0 + curr_i, 0.0);
    //         edges.push_edge(250.0 + curr_i, 0.0450, 0.0, 250.0, 400.0, 0.0);
    //         edges.push_edge(250.0, 400.0, 0.0, 200.0, 400.0, 0.0);
    //
    //         edges.push_edge(150.0, 400.0 + curr_i, 0.0, 130.0 + curr_i, 360.0, 0.0);
    //         edges.push_edge(150.0, 400.0, 0.0, 170.0, 360.0 + curr_i, 0.0);
    //         edges.push_edge(130.0 + curr_i, 360.0, 0.0, 170.0 + curr_i, 360.0 + curr_i, 0.0);
    //
    //         edges.push_edge(100.0 + curr_i, 340.0, 0.0, 200.0, 340.0 + curr_i, 0.0);
    //         edges.push_edge(100.0, 320., 0.0, 200.0, 320.0, 0.0);
    //         edges.push_edge(100.0, 340. + curr_i, 0.0, 100.0 + curr_i, 320.0, 0.0);
    //         edges.push_edge(200.0, 340., 0.0, 200.0 + curr_i, 320.0, 0.0);
    //         i -= 14;
    //     }
    // });

    pig.parse_file().expect("Script is valid");
}

#[test]
#[ignore = "visual script uses display/ImageMagick"]
fn script2() {
    let file = "./scripts/transform.cg";
    let mut parser = Parser::new(file, 400, 400, &Rgb::new(0, 255, 0));
    parser.canvas_mut().upper_left_origin = false;

    parser.parse_file().expect("Script is valid");
}
