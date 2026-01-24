// Build script for Rust FFI to CUDA DLL

fn main() {
    // Tell Rust where to find the CUDA library
    println!("cargo:rustc-link-search=native=C:/Users/Hamza/Desktop/GAFIME");
    println!("cargo:rustc-link-lib=dylib=gafime_cuda");
    
    // Rerun if library changes
    println!("cargo:rerun-if-changed=C:/Users/Hamza/Desktop/GAFIME/gafime_cuda.dll");
}
