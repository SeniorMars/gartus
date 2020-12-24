.POSIX:
.PHONY: all run

all:
	cargo run --release
	display image.ppm

clean: 
	cargo clean
