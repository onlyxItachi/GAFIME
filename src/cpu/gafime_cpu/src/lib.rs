//! GAFIME CPU Backend
//!
//! Rust-based CPU components for GAFIME:
//! 1. CatBoost Ordered Target Statistics (OTS) Encoder
//! 2. GPU-Aware Batch Launcher
//! 3. Cache-local feature scheduler
//! 4. Data Quality Analyzer

use pyo3::prelude::*;

mod batch_launcher;
mod cache_scheduler;
mod data_quality;
mod ots_encoder;
mod smart_scheduler;

use batch_launcher::PyBatchScheduler;
use cache_scheduler::PyCacheAwareScheduler;
use data_quality::PyDataQualityAnalyzer;
use ots_encoder::PyOTSEncoder;
use smart_scheduler::PySmartScheduler;

/// GAFIME CPU Backend Python Module
#[pymodule]
fn gafime_cpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOTSEncoder>()?;
    m.add_class::<PyBatchScheduler>()?;
    m.add_class::<PyCacheAwareScheduler>()?;
    m.add_class::<PyDataQualityAnalyzer>()?;
    m.add_class::<PySmartScheduler>()?;

    // Add version info
    m.add("__version__", "0.4.7")?;

    Ok(())
}
