//! GAFIME CPU Backend
//! 
//! Rust-based CPU components for GAFIME:
//! 1. CatBoost Ordered Target Statistics (OTS) Encoder
//! 2. GPU-Aware Batch Launcher
//! 3. Async Pipeline Producer (zero Python overhead)
//! 4. L2 Cache-Aware Feature Scheduler

use pyo3::prelude::*;

mod ots_encoder;
mod batch_launcher;
mod async_pipeline;
mod cache_scheduler;

use ots_encoder::PyOTSEncoder;
use batch_launcher::PyBatchScheduler;
use async_pipeline::PyAsyncPipeline;
use cache_scheduler::PyCacheAwareScheduler;

/// GAFIME CPU Backend Python Module
#[pymodule]
fn gafime_cpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOTSEncoder>()?;
    m.add_class::<PyBatchScheduler>()?;
    m.add_class::<PyAsyncPipeline>()?;
    m.add_class::<PyCacheAwareScheduler>()?;
    
    // Add version info
    m.add("__version__", "0.3.0")?;
    
    Ok(())
}
