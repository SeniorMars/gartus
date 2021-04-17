use matrix_rs::graphics::display::*;
use std::io;

#[test]
fn cat() -> io::Result<()> {
    let mut cat = Canvas::new(610, 610, 255);
    let mut head: Vec<(i32, i32)> = Vec::new();
    cat.upper_left_system = true;
    cat.set_line_color(255, 255, 255);

    head.push((350 - 150, 222));
    head.push((358 - 150, 294));
    head.push((347 - 150, 337));
    head.push((351 - 150, 362));
    head.push((451 - 150, 408));
    head.push((547 - 150, 364));
    head.push((553 - 150, 337));
    head.push((544 - 150, 299));
    head.push((552 - 150, 225));
    head.push((482 - 150, 276));
    head.push((450 - 150, 271));
    head.push((419 - 150, 277));
    head.push((349 - 150, 221));

    for i in 0..head.len() {
        if i + 1 != head.len() {
            cat.draw_line(
                cat.line,
                head[i].0 as f64,
                head[i].1 as f64,
                head[i + 1].0 as f64,
                head[i + 1].1 as f64,
            )
        }
    }
    let mut eye_left: Vec<(i32, i32)> = Vec::new();
    eye_left.push((382 - 150, 308));
    eye_left.push((436 - 150, 322));
    eye_left.push((424 - 150, 340));
    eye_left.push((405 - 150, 345));
    eye_left.push((390 - 150, 339));
    eye_left.push((382 - 150, 308));

    for i in 0..eye_left.len() {
        if i + 1 != eye_left.len() {
            cat.draw_line(
                cat.line,
                eye_left[i].0 as f64,
                eye_left[i].1 as f64,
                eye_left[i + 1].0 as f64,
                eye_left[i + 1].1 as f64,
            )
        }
    }
    let mut eye_right: Vec<(i32, i32)> = Vec::new();
    eye_right.push((466 - 150, 320));
    eye_right.push((518 - 150, 308));
    eye_right.push((512 - 150, 331));
    eye_right.push((502 - 150, 343));
    eye_right.push((487 - 150, 345));
    eye_right.push((465 - 150, 322));

    cat.set_line_color(255, 255, 255);
    for i in 0..eye_right.len() {
        if i + 1 != eye_right.len() {
            cat.draw_line(
                cat.line,
                eye_right[i].0 as f64,
                eye_right[i].1 as f64,
                eye_right[i + 1].0 as f64,
                eye_right[i + 1].1 as f64,
            )
        }
    }
    let mut mouth: Vec<(i32, i32)> = Vec::new();
    mouth.push((441 - 150, 372));
    mouth.push((461 - 150, 372));
    for i in 0..mouth.len() {
        if i + 1 != mouth.len() {
            // cat.set_line_color(
            //     rng.gen_range(0..=255),
            //     rng.gen_range(0..=255),
            //     rng.gen_range(0..=255),
            // );
            cat.draw_line(
                cat.line,
                mouth[i].0 as f64,
                mouth[i].1 as f64,
                mouth[i + 1].0 as f64,
                mouth[i + 1].1 as f64,
            )
        }
    }
    cat.draw_line(
        cat.line,
        (371 - 150) as f64,
        (341) as f64,
        (317 - 150) as f64,
        (337) as f64,
    );
    cat.draw_line(
        cat.line,
        (370 - 150) as f64,
        (325) as f64,
        (321 - 150) as f64,
        (307) as f64,
    );
    cat.draw_line(
        cat.line,
        (375 - 150) as f64,
        (360) as f64,
        (327 - 150) as f64,
        (369) as f64,
    );


    cat.draw_line(
        cat.line,
        (538 - 150) as f64,
        (341) as f64,
        (585 - 150) as f64,
        (337) as f64,
    );
    cat.draw_line(
        cat.line,
        (536- 150) as f64,
        (325) as f64,
        (578- 150) as f64,
        (307) as f64,
    );
    cat.draw_line(
        cat.line,
        (534- 150) as f64,
        (360) as f64,
        (579- 150) as f64,
        (369) as f64,
    );
    // cat.display()
    // cat.save_binary("binary.ppm")?;
    // cat.save_ascii("ascii.ppm")?;
    cat.save_extension("./pics/cat.png")
}
