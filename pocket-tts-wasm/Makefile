.PHONY: build profiling clean

build:
	wasm-pack build --target web --release
	cp www/index.html www/worker.js pkg/

profiling:
	wasm-pack build --target web --profiling
	cp www/index.html www/worker.js pkg/

clean:
	rm -rf pkg
