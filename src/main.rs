// mod graphics;
// use graphics::display::Canvas;
fn main() {
    let string = "Hello world".to_string();
    string.split(' ').for_each(|word| {
        println!("{}", word);
    });
}
