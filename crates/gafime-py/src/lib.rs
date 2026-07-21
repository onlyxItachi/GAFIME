mod artifact;
mod common;
mod continuous;
mod generated;
mod legacy_helpers;
mod py_api;
mod runtime;

use pyo3::prelude::*;

use artifact::PyCompiledContinuousArtifact;
use common::native_version;
use generated::{
    analyze_decision_path, analyze_time_series, compile_decision_path, compile_time_series,
};
use legacy_helpers::{
    PyBatchScheduler, PyCacheAwareScheduler, PyDataQualityAnalyzer, PyOTSEncoder, PySmartScheduler,
};
use py_api::{
    analyze_continuous, analyze_continuous_arrow, analyze_continuous_buffers,
    analyze_continuous_cpu, analyze_continuous_nested, compile_continuous,
    compile_continuous_buffers, compile_continuous_nested, PyContinuousRecord, PyContinuousReport,
};
use runtime::runtime_capabilities;

pub use common::{
    boundary_name, public_package_version, result_table_to_arrow, ContinuousRecord,
    ContinuousReport, PyBoundaryError, SignificanceEntry, BOUNDARY_NAME,
};
pub use continuous::analyze_continuous_cpu_rows;

#[pymodule]
fn gafime_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", public_package_version())?;
    m.add("BOUNDARY_NAME", BOUNDARY_NAME)?;
    m.add_class::<PyCompiledContinuousArtifact>()?;
    m.add_class::<PyContinuousRecord>()?;
    m.add_class::<PyContinuousReport>()?;
    m.add_class::<PyOTSEncoder>()?;
    m.add_class::<PyBatchScheduler>()?;
    m.add_class::<PyCacheAwareScheduler>()?;
    m.add_class::<PyDataQualityAnalyzer>()?;
    m.add_class::<PySmartScheduler>()?;
    m.add_function(wrap_pyfunction!(compile_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(compile_continuous_nested, m)?)?;
    m.add_function(wrap_pyfunction!(compile_continuous_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_nested, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_continuous_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(compile_time_series, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_time_series, m)?)?;
    m.add_function(wrap_pyfunction!(compile_decision_path, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_decision_path, m)?)?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(runtime_capabilities, m)?)?;
    Ok(())
}
