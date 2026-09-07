//! Operation-aware backend choice at the public boundary. This is dispatch,
//! not a second owner of program, evidence, acceptance or numerical policy.
use super::error;
use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_gpu_sys::{GpuBackend, GpuNativeEvidenceExecutor, GpuSysError};
use gafime_orchestrator::semantic::{NativeEvidenceExecutor, SemanticError};
use gafime_types::{
    PrecisionProfile, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA,
    GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY,
    GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_PEARSON, GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER,
    GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE,
    GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT, GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN,
    GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE, GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

pub(super) enum TabularExecutor {
    Core(CoreEvidenceExecutor),
    Gpu(Box<GpuNativeEvidenceExecutor>),
}

fn gpu_error(value: GpuSysError) -> PyErr {
    match value {
        GpuSysError::SemanticAbiUnavailable | GpuSysError::MissingFunction(_) => {
            error(SemanticError::Unsupported(
                "selected payload does not expose the complete optional tabular primitive table",
            ))
        }
        other => PyRuntimeError::new_err(format!("tabular backend initialization failed: {other}")),
    }
}

impl TabularExecutor {
    pub(super) fn new(
        py: Python<'_>,
        configured: &str,
        profile: PrecisionProfile,
        device: i32,
    ) -> PyResult<Self> {
        if matches!(configured, "core" | "auto") {
            // Auto commits to the supported public vocabulary at construction,
            // not to an accelerator that will need hidden fallback later. The
            // current GPU lowering lacks rank/fixed-NMI channels.
            return Ok(Self::Core(CoreEvidenceExecutor::default()));
        }
        if configured == "metal" {
            return Err(error(SemanticError::Unsupported(
                "Metal does not implement the tabular semantic primitive lowering",
            )));
        }
        if !matches!(configured, "cuda" | "rocm") {
            return Err(error(SemanticError::Unsupported(
                "unsupported tabular backend",
            )));
        }
        // Reuse installed-package identity/path policy. This bounded metadata
        // discovery is not a Python data plane and never overrides explicit env.
        py.import("gafime._payloads")?
            .getattr("discover_payloads")?
            .call1((configured,))?;
        let backend = match configured {
            "cuda" => GpuBackend::cuda_from_env(device as u32),
            _ => GpuBackend::rocm_from_env(device as u32),
        }
        .map_err(gpu_error)?;
        let executor = backend.semantic_executor().map_err(gpu_error)?;
        if executor.capabilities().profile_mask & profile.capability_mask() == 0 {
            return Err(error(SemanticError::Unsupported(
                "selected semantic payload does not support the requested precision",
            )));
        }
        Ok(Self::Gpu(Box::new(executor)))
    }

    pub(super) fn native(&mut self) -> &mut dyn NativeEvidenceExecutor {
        match self {
            Self::Core(executor) => executor,
            Self::Gpu(executor) => executor.as_mut(),
        }
    }

    pub(super) fn backend_kind(&self) -> u32 {
        match self {
            Self::Core(_) => GAFIME_BACKEND_CPU,
            Self::Gpu(executor) => executor.backend_kind(),
        }
    }

    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Core(_) => "core",
            Self::Gpu(executor) if executor.backend_kind() == GAFIME_BACKEND_CUDA => "cuda",
            Self::Gpu(_) => "rocm",
        }
    }

    pub(super) fn capabilities<'py>(
        &self,
        py: Python<'py>,
        configured: &str,
        device: i32,
        precision: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("configured_backend", configured)?;
        out.set_item("selected_backend", self.name())?;
        out.set_item("configured_device_id", device)?;
        out.set_item("precision", precision)?;
        match self {
            Self::Core(_) => {
                out.set_item("selected_device_id", py.None())?;
                out.set_item(
                    "programs",
                    [
                        "source",
                        "absolute_difference",
                        "softsign",
                        "centered_product",
                    ],
                )?;
                out.set_item(
                    "statistics",
                    ["pearson", "spearman", "fixed_nmi", "graph_energy"],
                )?;
                out.set_item("contexts", ["reference", "paired_view", "labels", "graph"])?;
                out.set_item("selection_reason", "Core supports the complete tabular semantic vocabulary; supervised GPU route support alone is insufficient")?;
                out.set_item("source", "static")?;
            }
            Self::Gpu(executor) => {
                let caps = executor.capabilities();
                let programs: Vec<_> = [
                    (GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE, "source"),
                    (
                        GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE,
                        "absolute_difference",
                    ),
                    (GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN, "softsign"),
                    (
                        GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT,
                        "centered_product",
                    ),
                ]
                .into_iter()
                .filter_map(|(mask, name)| (caps.program_op_mask & mask != 0).then_some(name))
                .collect();
                let mut statistics = Vec::new();
                let mut contexts = Vec::new();
                // Capabilities are the intersection of payload primitives and
                // this Rust adapter's actual lowerings, never raw advertised bits.
                if caps.association_statistic_mask & GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON != 0
                    && caps.primitive_mask & GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_PEARSON != 0
                {
                    statistics.push("pearson");
                    contexts.extend(["reference", "paired_view"]);
                    if caps.primitive_mask & GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER != 0 {
                        contexts.push("labels");
                    }
                }
                if caps.primitive_mask & GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY != 0 {
                    statistics.push("graph_energy");
                    contexts.push("graph");
                }
                out.set_item("selected_device_id", executor.device_id())?;
                out.set_item("programs", programs)?;
                out.set_item("statistics", statistics)?;
                out.set_item("contexts", contexts)?;
                out.set_item("selection_reason", "Explicit backend; each request must fit the negotiated tabular lowering, with no Core substitution")?;
                out.set_item("source", "runtime")?;
                out.set_item(
                    "payload",
                    executor
                        .loaded_library_path()
                        .map(|p| p.to_string_lossy().into_owned()),
                )?;
                out.set_item("primitive_abi_version", caps.abi_version)?;
            }
        }
        Ok(out)
    }

    pub(super) fn diagnostics<'py>(
        &self,
        py: Python<'py>,
        retained: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        match self {
            Self::Core(executor) => {
                out.set_item("materialized_nodes", executor.materialized_nodes())?;
                out.set_item("retained_hits", executor.retained_hits())?;
                out.set_item("source_shares", executor.source_shares())?;
                out.set_item("output_allocations", executor.output_allocations())?;
                out.set_item("output_bytes", executor.output_bytes())?;
                out.set_item("evidence_kernel_calls", executor.evidence_kernel_calls())?;
            }
            Self::Gpu(_) => {
                // Do not substitute estimated counters for observed work. Device
                // execution/parity tests provide evidence independently of this
                // resource snapshot; unsupported counters are explicitly absent.
                out.set_item("backend", self.name())?;
                out.set_item("native_work_counters_available", false)?;
            }
        }
        out.set_item("retained_bytes", retained)?;
        Ok(out)
    }
}
