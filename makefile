.POSIX:
.PHONY: all run

all:
	cargo run --release

clean: 
	cargo clean
