use gafime_gpu_sys::{GpuBackend, GpuDeviceProfile};
use pyo3::{prelude::*, types::PyDict};

#[derive(Clone, Debug)]
pub(super) struct RuntimeProbe {
    supports_decision_path_membership: bool,
    supports_decision_path_score: bool,
}

pub(super) fn probe(backend: &GpuBackend) -> RuntimeProbe {
    RuntimeProbe {
        supports_decision_path_membership: backend.supports_decision_path_membership(),
        supports_decision_path_score: backend.supports_decision_path_score(),
    }
}

pub(super) fn add_runtime_capabilities(
    py: Python<'_>,
    runtime: &Bound<'_, PyDict>,
    profile: &GpuDeviceProfile,
    probe: &RuntimeProbe,
) -> PyResult<()> {
    let rt = PyDict::new(py);
    rt.set_item("available", profile.local_cmake_experiment_available())?;
    rt.set_item(
        "decision_path_membership_abi",
        probe.supports_decision_path_membership,
    )?;
    rt.set_item(
        "decision_path_score_abi",
        probe.supports_decision_path_score,
    )?;
    runtime.set_item("rt", rt)
}
