use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use gafime_orchestrator::semantic::{SemanticError, SemanticResult};
use gafime_orchestrator::MatrixHandle;
use gafime_types::{
    BackendKind, GafimeDecisionPathBatch, GafimeDecisionPathScoreBatch, GafimeDecisionPathTerm,
    GafimeGpuMatrix, GafimeGpuSemanticBank, GafimeResultTable, GafimeSemanticProgramBatch,
    GafimeStatus, PrecisionProfile, GAFIME_ABI_VERSION, GAFIME_BACKEND_CUDA,
    GAFIME_DECISION_PATH_FLAG_REQUIRE_RT, GAFIME_GPU_DEVICE_FLAG_OPTIX_RT,
    GAFIME_MAX_DECISION_PATH_COUNT, GAFIME_STATUS_UNSUPPORTED_BACKEND,
};
use libloading::Library;

use crate::{
    abi::{load_optional_symbol, status_to_gpu_result, GpuFunctionTable, GpuSysError},
    backend::GpuBackend,
    profile::GpuDeviceProfile,
    semantic::{GpuNativeEvidenceExecutor, OwnedSemanticBank, SemanticProgramNode},
};

pub type GafimeGpuDecisionPathMembershipFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    paths: *const GafimeDecisionPathBatch,
) -> GafimeStatus;
pub type GafimeGpuDecisionPathScoreFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    paths: *const GafimeDecisionPathScoreBatch,
    result_out: *mut GafimeResultTable,
) -> GafimeStatus;
pub type GafimeGpuDecisionPathReleaseDeviceStateFn =
    unsafe extern "C" fn(device_id: u32) -> GafimeStatus;

pub type GafimeGpuSemanticRegionMaterializeRtFn = unsafe extern "C" fn(
    bank: GafimeGpuSemanticBank,
    batch: *const GafimeSemanticProgramBatch,
    max_temporary_bytes: u64,
    peak_bytes_out: *mut u64,
) -> GafimeStatus;

pub use crate::semantic::local_compact::LocalCompactDiagnostics;

#[derive(Clone, Copy, Default)]
pub struct LocalCmakeExperimentFunctions {
    pub decision_path_membership: Option<GafimeGpuDecisionPathMembershipFn>,
    pub decision_path_score: Option<GafimeGpuDecisionPathScoreFn>,
    pub decision_path_release_device_state: Option<GafimeGpuDecisionPathReleaseDeviceStateFn>,
    pub semantic_region_materialize_rt: Option<GafimeGpuSemanticRegionMaterializeRtFn>,
    pub semantic_region_query_create_rt: Option<crate::semantic::local_compact::CreateFn>,
    pub semantic_region_query_execute_rt: Option<crate::semantic::local_compact::ExecuteFn>,
    pub semantic_region_query_free_rt: Option<crate::semantic::local_compact::FreeFn>,
    pub semantic_region_query_materialize_coverage_rt:
        Option<crate::semantic::local_compact::CoverageFn>,
}

impl LocalCmakeExperimentFunctions {
    pub(crate) fn has_region_coverage(&self) -> bool {
        self.semantic_region_query_create_rt.is_some()
            && self.semantic_region_query_execute_rt.is_some()
            && self.semantic_region_query_free_rt.is_some()
            && self.semantic_region_query_materialize_coverage_rt.is_some()
    }
}

/// # Safety
///
/// The library must be a trusted local CMake payload implementing the
/// experimental ABI declarations in src/cuda/rt_abi.hpp.
pub(crate) unsafe fn load_function_table(library: &Library) -> LocalCmakeExperimentFunctions {
    // SAFETY: the caller established the trusted payload boundary.
    unsafe {
        LocalCmakeExperimentFunctions {
            decision_path_membership: load_optional_symbol(
                library,
                "gafime_gpu_decision_path_membership",
            ),
            decision_path_score: load_optional_symbol(library, "gafime_gpu_decision_path_score"),
            decision_path_release_device_state: load_optional_symbol(
                library,
                "gafime_gpu_decision_path_release_device_state",
            ),
            semantic_region_materialize_rt: load_optional_symbol(
                library,
                "gafime_gpu_semantic_region_materialize_rt_v1",
            ),
            semantic_region_query_create_rt: load_optional_symbol(
                library,
                "gafime_gpu_semantic_region_query_create_rt_v1",
            ),
            semantic_region_query_execute_rt: load_optional_symbol(
                library,
                "gafime_gpu_semantic_region_query_execute_rt_v1",
            ),
            semantic_region_query_free_rt: load_optional_symbol(
                library,
                "gafime_gpu_semantic_region_query_free_rt_v1",
            ),
            semantic_region_query_materialize_coverage_rt: load_optional_symbol(
                library,
                "gafime_gpu_semantic_region_query_materialize_coverage_rt_v1",
            ),
        }
    }
}

/// Local source-build diagnostics, not a supported Python/backend selection API.
/// Every counted batch completed the explicit RequireRT entrypoint; no fallback
/// or ordinary arithmetic batch contributes to this record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalSemanticRtDiagnostics {
    pub completed_region_batches: u64,
    pub completed_regions: u64,
    pub completed_coverage_features: u64,
    pub peak_explicit_temporary_bytes: u64,
}

pub(crate) struct LocalSemanticRegionExecution {
    materialize: GafimeGpuSemanticRegionMaterializeRtFn,
    diagnostics: LocalSemanticRtDiagnostics,
}

#[cfg(test)]
mod semantic_region_group_tests {
    use super::*;
    use crate::semantic::{SemanticFrozenRegionTerm, SemanticRegionRelation};

    fn region(slots: &[u32]) -> SemanticProgramNode {
        SemanticProgramNode::FrozenRegionConjunction {
            output_slot: 99,
            terms: slots
                .iter()
                .map(|&input_slot| SemanticFrozenRegionTerm {
                    input_slot,
                    relation: SemanticRegionRelation::LessEqual,
                    threshold_bits: u64::from(1.0f32.to_bits()),
                })
                .collect(),
        }
    }

    #[test]
    fn groups_preserve_order_and_split_at_native_axis_limit() {
        let nodes = [region(&[0, 1]), region(&[1, 2]), region(&[2, 3])];
        assert_eq!(local_region_run_len(&nodes).unwrap(), 2);
        assert_eq!(local_region_run_len(&nodes[2..]).unwrap(), 1);
        assert!(local_region_run_len(&[region(&[0, 1, 2, 3])]).is_err());
    }

    #[test]
    fn groups_bound_region_count_and_stop_before_other_arithmetic() {
        assert_eq!(local_region_run_len(&vec![region(&[0]); 257]).unwrap(), 256);
        let nodes = [region(&[0]), SemanticProgramNode::Source { output_slot: 1 }];
        assert_eq!(local_region_run_len(&nodes).unwrap(), 1);
        assert!(local_region_run_len(&[region(&[])]).is_err());
        assert!(local_region_run_len(&[region(&[0; 65])]).is_err());
    }

    #[test]
    fn dependent_regions_begin_a_later_launch() {
        let nodes = [region(&[0]), region(&[99])];
        assert_eq!(local_region_run_len(&nodes).unwrap(), 1);
    }
}

fn local_semantic_error(error: GpuSysError) -> SemanticError {
    match error {
        GpuSysError::BackendStatus {
            status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            ..
        }
        | GpuSysError::MissingFunction(_) => SemanticError::Unsupported(
            "local semantic RT region execution is unavailable or ineligible; no fallback",
        ),
        _ => SemanticError::Invalid(
            "local semantic RT region execution rejected its descriptors or temporary budget",
        ),
    }
}

// Geometry admission is an execution lowering, not candidate semantics. Keep
// the canonical order while making the three-coordinate OptiX envelope explicit;
// never silently route a wider individual region to ordinary CUDA.
fn local_region_run_len(nodes: &[SemanticProgramNode]) -> SemanticResult<usize> {
    let mut axes = [0u32; 3];
    let mut axis_count = 0;
    let mut count = 0;
    for node in nodes.iter().take(256) {
        let SemanticProgramNode::FrozenRegionConjunction { terms, .. } = node else {
            break;
        };
        if terms.is_empty() || terms.len() > 64 {
            return Err(SemanticError::Unsupported(
                "local RT requires 1..=64 terms per region",
            ));
        }
        let mut next_axes = axes;
        let mut next_count = axis_count;
        for term in terms {
            if nodes[..count].iter().any(|previous| {
                matches!(previous, SemanticProgramNode::FrozenRegionConjunction {
                    output_slot, ..
                } if *output_slot == term.input_slot)
            }) {
                // An accepted region may itself be a later logical atom.
                // Its values must be initialized by a preceding launch.
                return Ok(count);
            }
            if !next_axes[..next_count].contains(&term.input_slot) {
                if next_count == next_axes.len() {
                    if count == 0 {
                        return Err(SemanticError::Unsupported(
                            "local RT requires at most three physical axes per region",
                        ));
                    }
                    return Ok(count);
                }
                next_axes[next_count] = term.input_slot;
                next_count += 1;
            }
        }
        axes = next_axes;
        axis_count = next_count;
        count += 1;
    }
    Ok(count)
}

impl LocalSemanticRegionExecution {
    pub(crate) fn materialize(
        &mut self,
        bank: &OwnedSemanticBank,
        nodes: &[SemanticProgramNode],
        available_bytes: usize,
    ) -> SemanticResult<()> {
        if bank.backend_kind() != GAFIME_BACKEND_CUDA || bank.profile() != PrecisionProfile::Fp32 {
            return Err(SemanticError::Unsupported(
                "local semantic RT requires CUDA fp32",
            ));
        }
        let budget = u64::try_from(available_bytes)
            .map_err(|_| SemanticError::Invalid("local RT temporary budget exceeds u64"))?;
        let is_region = |node: &SemanticProgramNode| {
            matches!(node, SemanticProgramNode::FrozenRegionConjunction { .. })
        };
        let is_special = |node: &SemanticProgramNode| {
            is_region(node) || matches!(node, SemanticProgramNode::RegionCount { .. })
        };
        let mut first = 0;
        while first < nodes.len() {
            if let SemanticProgramNode::RegionCount {
                output_slot,
                regions,
            } = &nodes[first]
            {
                let peak =
                    bank.materialize_local_region_count(*output_slot, regions, available_bytes)?;
                self.diagnostics.completed_coverage_features += 1;
                self.diagnostics.peak_explicit_temporary_bytes = self
                    .diagnostics
                    .peak_explicit_temporary_bytes
                    .max(peak as u64);
                first += 1;
                continue;
            }
            let region = is_region(&nodes[first]);
            let end = if region {
                first + local_region_run_len(&nodes[first..])?
            } else {
                nodes[first..]
                    .iter()
                    .position(is_special)
                    .map_or(nodes.len(), |offset| first + offset)
            };
            let run = &nodes[first..end];
            if region {
                // Canonical predicates only consume raw/previously accepted
                // atoms, but direct physical callers may supply dependent
                // regions. The native boundary must validate that a parallel
                // RT batch reads initialized inputs, never another output.
                let mut peak = 0u64;
                bank.with_program_batch(run, |raw, batch| {
                    // SAFETY: same marshaller/lock/lifetime as ordinary CUDA;
                    // this pointer came from this executor's retained payload.
                    // Native repeats route, slot, capability and budget checks.
                    let status = unsafe { (self.materialize)(raw, batch, budget, &mut peak) };
                    status_to_gpu_result("gafime_gpu_semantic_region_materialize_rt_v1", status)
                })
                .map_err(local_semantic_error)?;
                if peak > budget {
                    return Err(SemanticError::Invalid(
                        "local RT reported an inadmissible temporary peak",
                    ));
                }
                self.diagnostics.completed_region_batches += 1;
                self.diagnostics.completed_regions += run.len() as u64;
                self.diagnostics.peak_explicit_temporary_bytes =
                    self.diagnostics.peak_explicit_temporary_bytes.max(peak);
            } else {
                bank.materialize(run).map_err(local_semantic_error)?;
            }
            first = end;
        }
        Ok(())
    }
}

impl GpuBackend {
    /// Explicit local-only RT region lowering. Normal algebra still executes
    /// through the standard semantic table. Mixed/fp64 and ineligible regions
    /// fail closed; this never changes semantic auto or Python EngineConfig.
    pub fn local_rt_semantic_executor(&self) -> Result<GpuNativeEvidenceExecutor, GpuSysError> {
        if self.kind != GAFIME_BACKEND_CUDA
            || !self.device_profile()?.local_cmake_experiment_available()
        {
            return Err(GpuSysError::InvalidInput(
                "local semantic RT requires an actual RT-capable CUDA payload/device",
            ));
        }
        let materialize = self
            .functions
            .local_cmake_experiment
            .semantic_region_materialize_rt
            .ok_or(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_region_materialize_rt_v1",
            ))?;
        let mut executor = self.semantic_executor()?;
        executor.local_region_execution = Some(LocalSemanticRegionExecution {
            materialize,
            diagnostics: LocalSemanticRtDiagnostics::default(),
        });
        Ok(executor)
    }
}

impl GpuNativeEvidenceExecutor {
    /// None for the ordinary executor, even when its payload happens to support
    /// local RT. Driver/context overhead is outside the explicit-buffer peak.
    pub fn local_rt_diagnostics(&self) -> Option<LocalSemanticRtDiagnostics> {
        self.local_region_execution
            .as_ref()
            .map(|value| value.diagnostics)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecisionPathRtPolicy {
    #[default]
    AllowSmFallback,
    RequireRt,
}

impl DecisionPathRtPolicy {
    pub(crate) fn abi_flags(self) -> u32 {
        match self {
            Self::AllowSmFallback => 0,
            Self::RequireRt => GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        }
    }
}

impl GpuDeviceProfile {
    pub fn local_cmake_experiment_available(&self) -> bool {
        (self.flags & GAFIME_GPU_DEVICE_FLAG_OPTIX_RT) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LocalCmakeExperimentOwnerKey {
    payload_identity: usize,
    device_id: u32,
}

pub(crate) struct LocalCmakeExperimentDeviceStateOwner {
    key: LocalCmakeExperimentOwnerKey,
    device_id: u32,
    release: GafimeGpuDecisionPathReleaseDeviceStateFn,
    _library: Option<Arc<Library>>,
}

impl Drop for LocalCmakeExperimentDeviceStateOwner {
    fn drop(&mut self) {
        let mut owners = rt_device_state_owners()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let is_registered_owner = owners
            .get(&self.key)
            .is_some_and(|owner| std::ptr::eq(owner.as_ptr(), self));
        if !is_registered_owner {
            return;
        }
        owners.remove(&self.key);
        // SAFETY: the optional function came from the same trusted payload kept
        // alive by `_library`; the device id was validated when the backend was
        // constructed. Holding the registry lock prevents a replacement owner
        // from being installed until this final-owner cleanup completes. Drop
        // is best-effort because it cannot return an error.
        unsafe { (self.release)(self.device_id) };
    }
}

static RT_DEVICE_STATE_OWNERS: OnceLock<
    Mutex<HashMap<LocalCmakeExperimentOwnerKey, Weak<LocalCmakeExperimentDeviceStateOwner>>>,
> = OnceLock::new();

fn rt_device_state_owners(
) -> &'static Mutex<HashMap<LocalCmakeExperimentOwnerKey, Weak<LocalCmakeExperimentDeviceStateOwner>>>
{
    RT_DEVICE_STATE_OWNERS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn acquire_device_state_owner(
    kind: BackendKind,
    device_id: u32,
    functions: &GpuFunctionTable,
    library: &Option<Arc<Library>>,
) -> Option<Arc<LocalCmakeExperimentDeviceStateOwner>> {
    if kind != GAFIME_BACKEND_CUDA {
        return None;
    }
    let release = functions
        .local_cmake_experiment
        .decision_path_release_device_state?;
    // The symbol address identifies the loaded payload even when callers reach
    // the same DSO through different hard-linked paths or loader Arc values.
    let payload_identity = release as usize;
    let key = LocalCmakeExperimentOwnerKey {
        payload_identity,
        device_id,
    };
    let mut owners = rt_device_state_owners()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(owner) = owners.get(&key).and_then(Weak::upgrade) {
        return Some(owner);
    }
    let owner = Arc::new(LocalCmakeExperimentDeviceStateOwner {
        key,
        device_id,
        release,
        _library: library.clone(),
    });
    owners.insert(key, Arc::downgrade(&owner));
    Some(owner)
}

static LEGACY_CUDA_DECISION_PATH_LOCKS: OnceLock<
    Mutex<HashMap<LocalCmakeExperimentOwnerKey, Weak<Mutex<()>>>>,
> = OnceLock::new();

fn local_cmake_experiment_locks(
) -> &'static Mutex<HashMap<LocalCmakeExperimentOwnerKey, Weak<Mutex<()>>>> {
    LEGACY_CUDA_DECISION_PATH_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn acquire_local_cmake_experiment_lock(
    kind: BackendKind,
    device_id: u32,
    functions: &GpuFunctionTable,
) -> Option<Arc<Mutex<()>>> {
    if kind != GAFIME_BACKEND_CUDA
        || functions
            .local_cmake_experiment
            .decision_path_release_device_state
            .is_some()
    {
        return None;
    }
    let payload_identity = functions
        .local_cmake_experiment
        .decision_path_score
        .map(|function| function as usize)
        .or_else(|| {
            functions
                .local_cmake_experiment
                .decision_path_membership
                .map(|function| function as usize)
        })?;
    let key = LocalCmakeExperimentOwnerKey {
        payload_identity,
        device_id,
    };
    let mut locks = local_cmake_experiment_locks()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(execution_lock) = locks.get(&key).and_then(Weak::upgrade) {
        return Some(execution_lock);
    }
    let execution_lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&execution_lock));
    Some(execution_lock)
}

pub(crate) fn validate_decision_path_count(path_count: usize) -> Result<(), GpuSysError> {
    if path_count > GAFIME_MAX_DECISION_PATH_COUNT as usize {
        Err(GpuSysError::SizeOverflow)
    } else {
        Ok(())
    }
}

impl GpuBackend {
    pub fn supports_decision_path_membership(&self) -> bool {
        self.functions
            .local_cmake_experiment
            .decision_path_membership
            .is_some()
    }

    pub fn supports_decision_path_score(&self) -> bool {
        self.functions
            .local_cmake_experiment
            .decision_path_score
            .is_some()
    }

    pub fn decision_path_membership(
        &mut self,
        matrix: &MatrixHandle,
        terms: &[GafimeDecisionPathTerm],
        path_offsets: &[u32],
    ) -> Result<Option<Vec<f32>>, GpuSysError> {
        self.decision_path_membership_with_policy(
            matrix,
            terms,
            path_offsets,
            DecisionPathRtPolicy::AllowSmFallback,
        )
    }

    pub fn decision_path_membership_with_policy(
        &mut self,
        matrix: &MatrixHandle,
        terms: &[GafimeDecisionPathTerm],
        path_offsets: &[u32],
        policy: DecisionPathRtPolicy,
    ) -> Result<Option<Vec<f32>>, GpuSysError> {
        let Some(decision_path_membership) = self
            .functions
            .local_cmake_experiment
            .decision_path_membership
        else {
            return match policy {
                DecisionPathRtPolicy::AllowSmFallback => Ok(None),
                DecisionPathRtPolicy::RequireRt => Err(GpuSysError::BackendStatus {
                    operation: "gafime_gpu_decision_path_membership",
                    status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
                }),
            };
        };
        if policy == DecisionPathRtPolicy::RequireRt
            && (self.kind != GAFIME_BACKEND_CUDA
                || (self.device_flags & GAFIME_GPU_DEVICE_FLAG_OPTIX_RT) == 0)
        {
            return Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_membership",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            });
        }
        if matrix.backend_kind() != self.kind {
            return Err(GpuSysError::InvalidInput(
                "matrix backend does not match GPU backend",
            ));
        }
        if matrix.raw().is_null() {
            return Err(GpuSysError::InvalidInput(
                "GPU decision-path membership requires a native resident matrix",
            ));
        }
        if terms.is_empty() || path_offsets.len() < 2 {
            return Err(GpuSysError::InvalidInput(
                "decision-path terms and offsets must be nonempty",
            ));
        }
        let path_count = path_offsets.len() - 1;
        validate_decision_path_count(path_count)?;
        if terms.len() > u32::MAX as usize {
            return Err(GpuSysError::SizeOverflow);
        }
        if path_offsets[0] != 0
            || path_offsets[path_count] as usize != terms.len()
            || path_offsets
                .windows(2)
                .any(|offsets| offsets[0] > offsets[1] || offsets[1] as usize > terms.len())
        {
            return Err(GpuSysError::InvalidInput(
                "decision-path offsets must be monotonic and cover exactly the terms buffer",
            ));
        }
        let rows = usize::try_from(matrix.rows()).map_err(|_| GpuSysError::SizeOverflow)?;
        let output_len = rows
            .checked_mul(path_count)
            .ok_or(GpuSysError::SizeOverflow)?;
        let mut membership = vec![f32::NAN; output_len];
        let batch = GafimeDecisionPathBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: path_count as u32,
            term_count: terms.len() as u32,
            flags: policy.abi_flags(),
            terms: terms.as_ptr(),
            path_offsets: path_offsets.as_ptr(),
            membership_host: membership.as_mut_ptr(),
            reserved: [0; 8],
        };
        let _legacy_execution_guard = self.local_cmake_experiment_lock.as_ref().map(|lock| {
            lock.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        // SAFETY: matrix identity and non-nullness, all slice lengths, monotonic
        // offsets, output size, and path-count bounds were checked above. The
        // slices and output Vec remain live for this synchronous payload call.
        let status = unsafe { decision_path_membership(matrix.raw(), &batch) };
        status_to_gpu_result("gafime_gpu_decision_path_membership", status)?;
        Ok(Some(membership))
    }

    /// Score decision paths into a caller-provided raw ABI result table.
    ///
    /// # Safety
    ///
    /// Every non-null pointer in `result` must reference uniquely borrowed,
    /// writable storage covering the declared capacity and strides for this
    /// synchronous call. The matrix handle must identify a live allocation
    /// owned by this backend.
    pub unsafe fn decision_path_score(
        &mut self,
        matrix: &MatrixHandle,
        terms: &[GafimeDecisionPathTerm],
        path_offsets: &[u32],
        metric_ids: &[u32],
        result: &mut GafimeResultTable,
    ) -> Result<bool, GpuSysError> {
        // SAFETY: this convenience entry point forwards the caller's complete
        // raw-result contract unchanged to the policy-aware implementation.
        unsafe {
            self.decision_path_score_with_policy(
                matrix,
                terms,
                path_offsets,
                metric_ids,
                result,
                DecisionPathRtPolicy::AllowSmFallback,
            )
        }
    }

    /// Score decision paths into a caller-provided raw ABI result table under
    /// the selected RT policy.
    ///
    /// # Safety
    ///
    /// Every non-null pointer in `result` must reference uniquely borrowed,
    /// writable storage covering the declared capacity and strides for this
    /// synchronous call. The matrix handle must identify a live allocation
    /// owned by this backend.
    pub unsafe fn decision_path_score_with_policy(
        &mut self,
        matrix: &MatrixHandle,
        terms: &[GafimeDecisionPathTerm],
        path_offsets: &[u32],
        metric_ids: &[u32],
        result: &mut GafimeResultTable,
        policy: DecisionPathRtPolicy,
    ) -> Result<bool, GpuSysError> {
        let Some(decision_path_score) = self.functions.local_cmake_experiment.decision_path_score
        else {
            return match policy {
                DecisionPathRtPolicy::AllowSmFallback => Ok(false),
                DecisionPathRtPolicy::RequireRt => Err(GpuSysError::BackendStatus {
                    operation: "gafime_gpu_decision_path_score",
                    status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
                }),
            };
        };
        if policy == DecisionPathRtPolicy::RequireRt
            && (self.kind != GAFIME_BACKEND_CUDA
                || (self.device_flags & GAFIME_GPU_DEVICE_FLAG_OPTIX_RT) == 0)
        {
            return Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_score",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            });
        }
        if matrix.backend_kind() != self.kind {
            return Err(GpuSysError::InvalidInput(
                "matrix backend does not match GPU backend",
            ));
        }
        if matrix.raw().is_null() {
            return Err(GpuSysError::InvalidInput(
                "GPU decision-path score requires a native resident matrix",
            ));
        }
        if terms.is_empty() || path_offsets.len() < 2 || metric_ids.is_empty() {
            return Err(GpuSysError::InvalidInput(
                "decision-path score terms, offsets, and metrics must be nonempty",
            ));
        }
        let path_count = path_offsets.len() - 1;
        validate_decision_path_count(path_count)?;
        if terms.len() > u32::MAX as usize || metric_ids.len() > u32::MAX as usize {
            return Err(GpuSysError::SizeOverflow);
        }
        if path_offsets[0] != 0
            || path_offsets[path_count] as usize != terms.len()
            || path_offsets
                .windows(2)
                .any(|offsets| offsets[0] > offsets[1] || offsets[1] as usize > terms.len())
        {
            return Err(GpuSysError::InvalidInput(
                "decision-path offsets must be monotonic and cover exactly the terms buffer",
            ));
        }
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: path_count as u32,
            term_count: terms.len() as u32,
            flags: policy.abi_flags(),
            terms: terms.as_ptr(),
            path_offsets: path_offsets.as_ptr(),
            metric_ids: metric_ids.as_ptr(),
            metric_count: metric_ids.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };
        let _legacy_execution_guard = self.local_cmake_experiment_lock.as_ref().map(|lock| {
            lock.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        // SAFETY: matrix identity/non-nullness and every batch slice/offset were
        // validated above. The caller-owned result table obeys the v1 ABI
        // allocation contract, and all borrowed storage remains live for this
        // synchronous call into the retained trusted payload.
        let status = unsafe { decision_path_score(matrix.raw(), &batch, result) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status)?;
        Ok(true)
    }
}
