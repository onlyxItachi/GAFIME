//! GAFIME CPU Backend
//! 
//! Rust-based CPU components for GAFIME:
//! 1. CatBoost Ordered Target Statistics (OTS) Encoder
//! 2. GPU-Aware Batch Launcher
//! 3. Async Pipeline Producer (zero Python overhead)

use pyo3::prelude::*;

mod ots_encoder;
mod batch_launcher;
mod async_pipeline;

use ots_encoder::PyOTSEncoder;
use batch_launcher::PyBatchScheduler;
use async_pipeline::PyAsyncPipeline;

/// GAFIME CPU Backend Python Module
#[pymodule]
fn gafime_cpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOTSEncoder>()?;
    m.add_class::<PyBatchScheduler>()?;
    m.add_class::<PyAsyncPipeline>()?;
    
    // Add version info
    m.add("__version__", "0.2.0")?;
    
    Ok(())
}
