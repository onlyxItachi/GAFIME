//! CPU-only wrapper safety tests run in ordinary CI; CUDA execution is explicit.
#[allow(dead_code)]
#[path = "../../../tests/release_measure/issue73_gpu_reuse.rs"]
mod reuse;
