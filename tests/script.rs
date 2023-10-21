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

    pig.edge_matrix_fun(&|edges| {
        let mut curr_i = 120.0;
        while curr_i < 380.0 {
            edges.push_edge(170.0, curr_i, 0.0, 227.0, curr_i, 0.0);
            edges.push_edge(273.0, curr_i, 0.0, 330.0, curr_i, 0.0);
            edges.push_edge(227.0, curr_i, 0.0, 236.0, curr_i + 15.0, 0.0);
            edges.push_edge(236.0, curr_i + 15.0, 0.0, 248.0, curr_i + 7.0, 0.0);
            edges.push_edge(248.0, curr_i + 7.0, 0.0, 260.0, curr_i + 3.0, 0.0);
            edges.push_edge(260.0, curr_i + 3.0, 0.0, 273.0, curr_i, 0.0);

            curr_i += 4.0;

            edges.push_edge(170.0, curr_i, 0.0, 227.0, curr_i, 0.0);
            edges.push_edge(273.0, curr_i, 0.0, 330.0, curr_i, 0.0);
            edges.push_edge(227.0, curr_i, 0.0, 240.0, curr_i + 7.0, 0.0);
            edges.push_edge(240.0, curr_i + 7.0, 0.0, 252.0, curr_i + 1.0, 0.0);
            edges.push_edge(252.0, curr_i + 1.0, 0.0, 265.0, curr_i + 9.0, 0.0);
            edges.push_edge(265.0, curr_i + 9.0, 0.0, 273.0, curr_i, 0.0);

            curr_i += 4.0;

            edges.push_edge(170.0, curr_i, 0.0, 227.0, curr_i, 0.0);
            edges.push_edge(273.0, curr_i, 0.0, 330.0, curr_i, 0.0);
            edges.push_edge(227.0, curr_i, 0.0, 234.0, curr_i + 13.0, 0.0);
            edges.push_edge(234.0, curr_i + 13.0, 0.0, 246.0, curr_i + 5.0, 0.0);
            edges.push_edge(246.0, curr_i + 5.0, 0.0, 258.0, curr_i + 7.0, 0.0);
            edges.push_edge(258.0, curr_i + 7.0, 0.0, 273.0, curr_i, 0.0);

            curr_i += 4.0;

            edges.push_edge(170.0, curr_i, 0.0, 227.0, curr_i, 0.0);
            edges.push_edge(273.0, curr_i, 0.0, 330.0, curr_i, 0.0);
            edges.push_edge(227.0, curr_i, 0.0, 240.0, curr_i + 7.0, 0.0);
            edges.push_edge(240.0, curr_i + 7.0, 0.0, 252.0, curr_i + 3.0, 0.0);
            edges.push_edge(252.0, curr_i + 3.0, 0.0, 265.0, curr_i + 14.0, 0.0);
            edges.push_edge(265.0, curr_i + 14.0, 0.0, 273.0, curr_i, 0.0);

            curr_i += 4.0;
        }
    });

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
