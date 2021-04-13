.POSIX:
.PHONY: all clean test

all:
	cargo run --release

test:
	cargo test

clean: 
	cargo clean
	rm *.ppm *.png
