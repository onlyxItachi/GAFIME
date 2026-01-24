//! GAFIME CPU Backend
//! 
//! Rust-based CPU components for GAFIME:
//! 1. CatBoost Ordered Target Statistics (OTS) Encoder
//! 2. GPU-Aware Batch Launcher (TODO)

use pyo3::prelude::*;

mod ots_encoder;
mod batch_launcher;

use ots_encoder::PyOTSEncoder;
use batch_launcher::PyBatchScheduler;

/// GAFIME CPU Backend Python Module
#[pymodule]
fn gafime_cpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOTSEncoder>()?;
    m.add_class::<PyBatchScheduler>()?;
    
    // Add version info
    m.add("__version__", "0.1.0")?;
    
    Ok(())
}
