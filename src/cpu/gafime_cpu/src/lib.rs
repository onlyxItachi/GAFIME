//! GAFIME CPU Backend
//!
//! Rust-based CPU components for GAFIME:
//! 1. CatBoost Ordered Target Statistics (OTS) Encoder
//! 2. GPU-Aware Batch Launcher
//! 3. Async Pipeline Producer (zero Python overhead)
//! 4. Cache-local feature scheduler
//! 5. Data Quality Analyzer (NaN/Inf, missing values, entropy)
//! 6. Contiguous Memory Layout (optimal GPU memory coalescing)

use pyo3::prelude::*;

mod async_pipeline;
mod batch_launcher;
mod cache_scheduler;
mod contiguous_layout;
mod data_quality;
mod ots_encoder;
mod smart_scheduler;

use async_pipeline::PyAsyncPipeline;
use batch_launcher::PyBatchScheduler;
use cache_scheduler::PyCacheAwareScheduler;
use contiguous_layout::{PyContiguousBucket, PyContiguousLayout};
use data_quality::PyDataQualityAnalyzer;
use ots_encoder::PyOTSEncoder;
use smart_scheduler::PySmartScheduler;

/// GAFIME CPU Backend Python Module
#[pymodule]
fn gafime_cpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOTSEncoder>()?;
    m.add_class::<PyBatchScheduler>()?;
    m.add_class::<PyAsyncPipeline>()?;
    m.add_class::<PyCacheAwareScheduler>()?;
    m.add_class::<PyDataQualityAnalyzer>()?;
    m.add_class::<PyContiguousLayout>()?;
    m.add_class::<PyContiguousBucket>()?;
    m.add_class::<PySmartScheduler>()?;

    // Add version info
    m.add("__version__", "0.4.1")?;

    Ok(())
}
