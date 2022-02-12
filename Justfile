# just manual: https://github.com/casey/just/#readme

_default:
    @just --list

# Runs our project
run:
	cargo run --release

# Runs clippy on the sources 
check:
	cargo clippy --locked -- -D warnings

# Runs unit tests
test:
	cargo test --locked

# cleans our project
clean: 
	cargo clean

# cleans our images
rm:
	rm anim/*.ppm anim/*.png
	rm pics/*.ppm pics/*.png
