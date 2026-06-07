# just manual: https://github.com/casey/just/#readme

_default:
    @just --list

# Runs our project
run:
	cargo run --release

# Runs rustfmt in CI mode
fmt:
	cargo ci-fmt

# Verifies ImageMagick is available for image output tests
image-deps:
	@command -v magick >/dev/null || command -v convert >/dev/null || (echo "ImageMagick is required for PNG/GIF output tests" >&2; exit 1)

# Runs clippy on the sources
check:
	cargo ci-clippy

# Runs unit tests
test: image-deps
	cargo ci-test

# Runs tests with all features enabled
test-all: image-deps
	cargo ci-test-all

# Runs tests with default features disabled
test-no-default: image-deps
	cargo ci-test-no-default

# Runs tests with only the old parser feature enabled
test-old-parser: image-deps
	cargo ci-test-old-parser

# Runs the full local CI suite
ci: fmt check test test-all test-no-default test-old-parser

# cleans our project
clean:
	cargo clean

# cleans our images
rm:
	rm -f anim/*.ppm anim/*.png anim/*.gif
	rm -f pics/*.ppm pics/*.png pics/*.jpg pics/*.gif
	rm -rf gifs
