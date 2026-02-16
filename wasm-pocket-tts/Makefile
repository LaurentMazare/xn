.PHONY: build profiling clean

build:
	wasm-pack build --target web --release
	cp www/index.html www/gen_worker.js www/decode_worker.js pkg/

profiling:
	wasm-pack build --target web --profiling
	cp www/index.html www/gen_worker.js www/decode_worker.js pkg/

clean:
	rm -rf pkg
