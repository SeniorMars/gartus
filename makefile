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
	rm *.ppm *.png

rmf:
	rm pics/*.ppm pics/*.png
	rm anim/*.ppm
