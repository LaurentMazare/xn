.PHONY: build clean

build:
	wasm-pack build --target web --release
	cp www/index.html pkg/

clean:
	rm -rf pkg
