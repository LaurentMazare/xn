fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/compatibility.cuh");

        let builder = bindgen_cuda::Builder::default().kernel_paths_glob("cuda-kernels/**/*.cu");
        println!("cargo:info={builder:?}");
        let bindings = builder.build_ptx().unwrap();
        bindings.write("src/cuda_kernels.rs").unwrap();
    }
}
