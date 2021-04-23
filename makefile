.POSIX:
.PHONY: run release clean test rmf

run:
	cargo run

release:
	cargo run --release

test:
	cargo test

clean: 
	cargo clean

rmf:
	rm anim/*.ppm anim/*.png
	rm pics/*.ppm pics/*.png
