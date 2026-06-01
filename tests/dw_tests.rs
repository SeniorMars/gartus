use gartus::gmath::edge_matrix::EdgeMatrix;
use gartus::gmath::matrix::*;
use gartus::gmath::polygon_matrix::PolygonMatrix;
use gartus::graphics::colors::*;
use gartus::parser::Parser;

#[test]
#[ignore]
fn script_polygons() {
    let color = Rgb::new(0, 255, 0);
    let mut dw = Parser::new("./tests/script_polygon", 500, 500, &color);

    // Initial state check
    assert!(dw.edge_matrix().is_empty());
    assert!(dw.polygon_matrix().is_empty());
    assert_eq!(dw.trans_matrix(), &Matrix::identity_matrix(4));
    assert!(dw.is_dirty()); // Starts dirty until first render

    // 1. box 0 0 0 200 100 400
    dw.parse_string("box\n0 0 0 200 100 400")
        .expect("box valid");
    let mut manual_poly = PolygonMatrix::new();
    manual_poly.add_box((0.0, 0.0, 0.0), 200.0, 100.0, 400.0);
    assert_eq!(dw.polygon_matrix(), &manual_poly, "Box geometry mismatch");
    assert!(dw.is_dirty(), "Box should mark canvas dirty");

    // 2. ident, rotate, move
    dw.parse_string("ident").expect("ident valid");
    let mut manual_trans = Matrix::identity_matrix(4);
    assert_eq!(dw.trans_matrix(), &manual_trans);

    dw.parse_string("rotate\nx 20").expect("rotate x valid");
    manual_trans = &Matrix::rotate_x(20.0) * &manual_trans;
    assert_eq!(dw.trans_matrix(), &manual_trans);

    dw.parse_string("rotate\ny 20").expect("rotate y valid");
    manual_trans = &Matrix::rotate_y(20.0) * &manual_trans;
    assert_eq!(dw.trans_matrix(), &manual_trans);

    dw.parse_string("move\n150 200 0").expect("move valid");
    manual_trans = &Matrix::translate(150.0, 200.0, 0.0) * &manual_trans;
    assert_eq!(dw.trans_matrix(), &manual_trans);

    // 3. apply (Transfers trans to poly)
    dw.parse_string("apply").expect("apply valid");
    manual_poly = manual_poly.apply(&manual_trans);
    assert_eq!(dw.polygon_matrix(), &manual_poly, "Apply mismatch");
    assert!(dw.is_dirty());

    // 4. display renders the canvas and sends it to the external viewer.
    dw.parse_string("display").expect("display valid");
    assert!(!dw.is_dirty(), "Display should clear dirty flag");

    let mut blank = true;
    for y in 0..500 {
        for x in 0..500 {
            if dw.canvas().get_pixel(x, y) != Some(&Rgb::default()) {
                blank = false;
                break;
            }
        }
    }
    assert!(!blank, "Canvas should not be blank after display");

    // 5. clear (Resets matrices and canvas)
    dw.parse_string("clear").expect("clear valid");
    assert!(dw.polygon_matrix().is_empty());
    assert!(dw.edge_matrix().is_empty());
    assert!(!dw.is_dirty());

    // 6. sphere
    dw.parse_string("sphere\n0 0 0 200").expect("sphere valid");
    manual_poly = PolygonMatrix::new();
    manual_poly.add_sphere((0.0, 0.0, 0.0), 200.0, 24);
    assert_eq!(dw.polygon_matrix(), &manual_poly);
    assert!(dw.is_dirty());

    // 7. Test transform stack reset logic in script
    dw.parse_string("ident\nrotate\ny 90\nmove\n250 250 0")
        .expect("sphere trans valid");
    manual_trans = Matrix::identity_matrix(4);
    manual_trans = &Matrix::rotate_y(90.0) * &manual_trans;
    manual_trans = &Matrix::translate(250.0, 250.0, 0.0) * &manual_trans;
    assert_eq!(dw.trans_matrix(), &manual_trans);

    // 8. Final Apply
    dw.parse_string("apply").expect("sphere apply valid");
    manual_poly = manual_poly.apply(&manual_trans);
    assert_eq!(dw.polygon_matrix(), &manual_poly);

    // 9. Full script regression check
    let mut dw_final = Parser::new("./tests/script_polygon", 500, 500, &color);
    dw_final
        .parse_file()
        .expect("Full script execution should succeed");

    let mut expected_poly = PolygonMatrix::new();
    expected_poly.add_torus((0.0, 0.0, 0.0), 25.0, 150.0, 24);

    let mut m1 = Matrix::identity_matrix(4);
    m1 = &Matrix::rotate_y(90.0) * &m1;
    m1 = &Matrix::translate(250.0, 250.0, 0.0) * &m1;
    expected_poly = expected_poly.apply(&m1);

    // Later torus `apply` commands have no newly-added geometry, so they should
    // not transform the already-finalized torus again.

    assert!(
        dw_final
            .polygon_matrix()
            .as_matrix()
            .approx_eq(expected_poly.as_matrix(), 1e-10),
        "Final state mismatch after pending-only transformations"
    );
    assert!(
        !dw_final.is_dirty(),
        "Final state should be rendered (due to display command)"
    );
}

#[test]
#[ignore]
fn script_3d() {
    let mut dw = Parser::new("./tests/script_3d", 500, 500, &Rgb::new(0, 255, 0));
    dw.parse_file().expect("Script is valid");
}

#[test]
#[ignore]
fn script_transform() {
    let mut dw = Parser::new("./tests/script_transform", 500, 500, &Rgb::new(0, 255, 0));

    let mut manual_edges = EdgeMatrix::new();
    let lines = [
        (0.0, 0.0, 0.0, 100.0, 0.0, 0.0),
        (100.0, 0.0, 0.0, 100.0, 100.0, 0.0),
        (100.0, 100.0, 0.0, 0.0, 100.0, 0.0),
        (0.0, 100.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 100.0, 100.0, 0.0, 100.0),
        (100.0, 0.0, 100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0, 0.0, 100.0, 100.0),
        (0.0, 100.0, 100.0, 0.0, 0.0, 100.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 100.0),
        (0.0, 100.0, 0.0, 0.0, 100.0, 100.0),
        (100.0, 100.0, 0.0, 100.0, 100.0, 100.0),
        (100.0, 0.0, 0.0, 100.0, 0.0, 100.0),
    ];

    for l in &lines {
        dw.parse_string(&format!(
            "line\n{} {} {} {} {} {}",
            l.0, l.1, l.2, l.3, l.4, l.5
        ))
        .expect("line valid");
        manual_edges.push_edge(l.0, l.1, l.2, l.3, l.4, l.5);
    }
    assert_eq!(
        dw.edge_matrix().as_matrix(),
        manual_edges.as_matrix(),
        "Initial line batch mismatch"
    );

    dw.parse_string("ident\nscale\n2 2 2\napply")
        .expect("scale valid");
    let mut m_scale = Matrix::identity_matrix(4);
    m_scale = &Matrix::scale(2.0, 2.0, 2.0) * &m_scale;
    manual_edges = manual_edges.apply(&m_scale);
    assert!(
        dw.edge_matrix()
            .as_matrix()
            .approx_eq(manual_edges.as_matrix(), 1e-10),
        "Scale apply mismatch"
    );

    dw.parse_string("ident\nmove\n100 100 0\napply")
        .expect("move valid");
    assert!(
        dw.edge_matrix()
            .as_matrix()
            .approx_eq(manual_edges.as_matrix(), 1e-10),
        "Move apply should not affect finalized edges"
    );

    dw.parse_string("ident\nrotate\nz 20\nrotate\nx 20\nrotate\ny 20\napply")
        .expect("multi-rotate valid");
    assert!(
        dw.edge_matrix()
            .as_matrix()
            .approx_eq(manual_edges.as_matrix(), 1e-10),
        "Multi-rotate apply should not affect finalized edges"
    );

    dw.parse_string("ident\nrotate\ny 20\napply")
        .expect("final rotate valid");
    assert!(
        dw.edge_matrix()
            .as_matrix()
            .approx_eq(manual_edges.as_matrix(), 1e-10),
        "Final rotate apply should not affect finalized edges"
    );

    let mut dw_full = Parser::new("./tests/script_transform", 500, 500, &Rgb::new(0, 255, 0));
    dw_full.parse_file().expect("Full script should run");
    assert!(
        dw_full
            .edge_matrix()
            .as_matrix()
            .approx_eq(manual_edges.as_matrix(), 1e-10),
        "Full script result mismatch"
    );
}

#[test]
#[ignore]
fn curve_script() {
    let mut dw = Parser::new("./tests/script_curves", 500, 500, &Rgb::new(0, 255, 0));

    let mut expected_edges = EdgeMatrix::new();

    dw.parse_string("circle\n250 250 0 200")
        .expect("circle 1 valid");
    expected_edges.add_circle(250.0, 250.0, 0.0, 200.0, 0.001);
    assert!(
        dw.edge_matrix()
            .as_matrix()
            .approx_eq(expected_edges.as_matrix(), 1e-10)
    );

    dw.parse_string("circle\n175 325 0 50")
        .expect("circle 2 valid");
    expected_edges.add_circle(175.0, 325.0, 0.0, 50.0, 0.001);
    assert!(
        dw.edge_matrix()
            .as_matrix()
            .approx_eq(expected_edges.as_matrix(), 1e-10)
    );

    dw.parse_string("hermite\n150 150 350 150 -100 -100 100 150")
        .expect("hermite valid");
    expected_edges.add_hermite(
        (150.0, 150.0),
        (350.0, 150.0),
        (-100.0, -100.0),
        (100.0, 150.0),
    );
    assert!(
        dw.edge_matrix()
            .as_matrix()
            .approx_eq(expected_edges.as_matrix(), 1e-10)
    );

    dw.parse_string("bezier\n200 250 150 50 300 250 300 250")
        .expect("bezier valid");
    expected_edges.add_bezier3(
        (200.0, 250.0),
        (150.0, 50.0),
        (300.0, 250.0),
        (300.0, 250.0),
    );
    assert!(
        dw.edge_matrix()
            .as_matrix()
            .approx_eq(expected_edges.as_matrix(), 1e-10)
    );

    let mut dw_full = Parser::new("./tests/script_curves", 500, 500, &Rgb::new(0, 255, 0));
    dw_full
        .parse_file()
        .expect("Full script execution should succeed");

    let mut final_expected = EdgeMatrix::new();
    final_expected.add_circle(250.0, 250.0, 0.0, 200.0, 0.001);
    final_expected.add_circle(175.0, 325.0, 0.0, 50.0, 0.001);
    final_expected.add_circle(325.0, 325.0, 0.0, 50.0, 0.001);
    final_expected.add_circle(175.0, 325.0, 0.0, 10.0, 0.001);
    final_expected.add_circle(325.0, 325.0, 0.0, 10.0, 0.001);
    final_expected.add_hermite(
        (150.0, 150.0),
        (350.0, 150.0),
        (-100.0, -100.0),
        (100.0, 150.0),
    );
    final_expected.add_bezier3(
        (200.0, 250.0),
        (150.0, 50.0),
        (300.0, 250.0),
        (300.0, 250.0),
    );

    final_expected.add_bezier3((46.0, 494.0), (7.0, 488.0), (47.0, 455.0), (10.0, 450.0));
    final_expected.add_hermite((77.0, 492.0), (73.0, 455.0), (3.0, 32.0), (-6.0, 25.0));
    final_expected.add_bezier3((82.0, 479.0), (80.0, 490.0), (69.0, 469.0), (68.0, 481.0));
    final_expected.add_bezier3((91.0, 486.0), (91.0, 444.0), (112.0, 448.0), (111.0, 487.0));
    final_expected.add_bezier3(
        (161.0, 489.0),
        (114.0, 455.0),
        (132.0, 469.0),
        (139.0, 490.0),
    );
    final_expected.add_bezier3(
        (111.0, 451.0),
        (114.0, 455.0),
        (132.0, 469.0),
        (139.0, 490.0),
    );
    final_expected.add_hermite(
        (185.0, 496.0),
        (179.0, 453.0),
        (-105.0, -23.0),
        (100.0, 23.0),
    );
    final_expected.add_hermite(
        (220.0, 493.0),
        (204.0, 453.0),
        (-112.0, -20.0),
        (-125.0, -29.0),
    );

    assert!(
        dw_full
            .edge_matrix()
            .as_matrix()
            .approx_eq(final_expected.as_matrix(), 1e-10),
        "Full curve script result mismatch"
    );
}

#[test]
fn matrix_test() {
    let mut edge_matrix = EdgeMatrix::new();
    edge_matrix.push_edge(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let ident = Matrix::identity_matrix(4);
    let res = ident.mult_matrix(edge_matrix.as_matrix());
    assert_eq!(res, *edge_matrix.as_matrix());
}

#[test]
fn test_transformation_order() {
    let mut dw = Parser::new("test", 10, 10, &Rgb::new(0, 0, 0));
    dw.parse_string("line\n0 0 0 0 1 0").expect("line valid");
    dw.parse_string("rotate\nx 90\nmove\n0 10 0\napply")
        .expect("ops valid");
    let edges = dw.edge_matrix();
    let p1 = edges.iter_points().nth(1).expect("second point of line");
    assert!((p1[0] - 0.0).abs() < 1e-10);
    assert!((p1[1] - 10.0).abs() < 1e-10);
    assert!((p1[2] - 1.0).abs() < 1e-10);
}
