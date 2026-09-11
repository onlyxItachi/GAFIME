//! Local-only compact conditional query lowering. The native query owns physical
//! acceleration structures, never candidate or evidence identities. Its banks
//! and payload remain alive through synchronous destruction; labels are copied
//! for each execution, so rebinding labels cannot return stale evidence.

use super::*;
use std::ffi::c_void;

use gafime_orchestrator::semantic::{
    BinaryPairedStatistic, CompactEvidenceBatch, EvidenceChannel, LabelSet,
};

const QUERY_ABI: u32 = 0x0001_0000;
const OCCUPANCY: u32 = 1;
const PAIRED: u32 = 2;
const LABELED: u32 = 4;
const OCCUPANCY_SCORE: u32 = 1;
const AGREEMENT_SCORE: u32 = 2;
const IOU_SCORE: u32 = 4;
const GINI_SCORE: u32 = 8;

#[repr(C)]
pub struct QueryDesc {
    abi_version: u32,
    struct_size: u32,
    primary_bank: GafimeGpuSemanticBank,
    paired_bank: GafimeGpuSemanticBank,
    terms: *const GafimeSemanticFrozenRegionTerm,
    paired_term_slots: *const u32,
    region_offsets: *const u32,
    partition_offsets: *const u32,
    term_count: u64,
    region_count: u32,
    partition_count: u32,
    flags: u32,
    reserved32: u32,
    max_persistent_bytes: u64,
    reserved: [u64; 7],
}

#[repr(C)]
pub struct BinaryLabels {
    abi_version: u32,
    struct_size: u32,
    row_indices: *const u64,
    values: *const u8,
    count: u64,
    reserved: [u64; 6],
}

#[repr(C)]
pub struct ExecuteDesc {
    abi_version: u32,
    struct_size: u32,
    statistic_mask: u32,
    finalizer_mask: u32,
    labels: BinaryLabels,
    max_temporary_bytes: u64,
    reserved: [u64; 7],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactStats {
    row_count: u64,
    label_support: u64,
    occupancy_inside: u64,
    paired_n00: u64,
    paired_n01: u64,
    paired_n10: u64,
    paired_n11: u64,
    label_outside_0: u64,
    label_outside_1: u64,
    label_inside_0: u64,
    label_inside_1: u64,
    occupancy: f32,
    paired_agreement: f32,
    paired_iou: f32,
    labeled_gini_gain: f32,
    occupancy_state: u32,
    paired_agreement_state: u32,
    paired_iou_state: u32,
    labeled_gini_gain_state: u32,
    reserved: [u64; 4],
}

#[repr(C)]
pub struct StatsTable {
    abi_version: u32,
    struct_size: u32,
    requested_statistic_mask: u32,
    finalized_mask: u32,
    capacity: u64,
    count: u64,
    records: *mut ExactStats,
    reserved: [u64; 8],
}

pub type CreateFn = unsafe extern "C" fn(*const QueryDesc, *mut *mut c_void, *mut u64) -> i32;
pub type ExecuteFn =
    unsafe extern "C" fn(*mut c_void, *const ExecuteDesc, *mut StatsTable, *mut u64) -> i32;
pub type FreeFn = unsafe extern "C" fn(*mut c_void) -> i32;
pub type CoverageFn = unsafe extern "C" fn(*mut c_void, u32, u64, *mut u64) -> i32;

#[derive(Clone, Copy)]
struct Functions {
    create: CreateFn,
    execute: ExecuteFn,
    free: FreeFn,
    coverage: CoverageFn,
}

struct Query {
    raw: *mut c_void,
    functions: Functions,
    primary: OwnedSemanticBank,
    _paired: Option<OwnedSemanticBank>,
    persistent_bytes: usize,
    regions: usize,
}

// SAFETY: Query is uniquely owned by its executor, and calls need &mut Query.
// Both banks retain their DSO and synchronize accesses. Native calls serialize
// query state, synchronize before returning, and restore the caller device.
unsafe impl Send for Query {}

impl Drop for Query {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: this is the unique live query; every operation has
            // synchronized before returning. Bank/DSO fields outlive this
            // teardown attempt. As with semantic banks, Drop is best-effort
            // after CUDA runtime/device loss: it cannot promise reclamation
            // or retry if native code cannot select the owning device.
            let _status = unsafe { (self.functions.free)(self.raw) };
        }
    }
}

fn checked_add(a: usize, b: usize) -> SemanticResult<usize> {
    a.checked_add(b)
        .ok_or(SemanticError::Invalid("compact native byte count overflow"))
}

fn remaining(budget: usize, used: usize) -> SemanticResult<usize> {
    budget.checked_sub(used).ok_or(SemanticError::Invalid(
        "compact native query exceeds execution budget",
    ))
}

impl Query {
    fn create(
        functions: Functions,
        primary: OwnedSemanticBank,
        paired: Option<OwnedSemanticBank>,
        terms: &[GafimeSemanticFrozenRegionTerm],
        paired_slots: &[u32],
        offsets: &[u32],
        budget: usize,
    ) -> SemanticResult<Self> {
        let regions = offsets
            .len()
            .checked_sub(1)
            .ok_or(SemanticError::Invalid("empty compact region query"))?;
        let region_count = u32::try_from(regions)
            .map_err(|_| SemanticError::Invalid("compact region count overflow"))?;
        let partitions = [0, region_count];
        let desc = QueryDesc {
            abi_version: QUERY_ABI,
            struct_size: std::mem::size_of::<QueryDesc>() as u32,
            primary_bank: primary.inner.raw,
            paired_bank: paired
                .as_ref()
                .map_or(std::ptr::null_mut(), |bank| bank.inner.raw),
            terms: terms.as_ptr(),
            paired_term_slots: if paired.is_some() {
                paired_slots.as_ptr()
            } else {
                std::ptr::null()
            },
            region_offsets: offsets.as_ptr(),
            partition_offsets: partitions.as_ptr(),
            term_count: terms.len() as u64,
            region_count,
            partition_count: 1,
            flags: 1,
            reserved32: 0,
            max_persistent_bytes: budget as u64,
            reserved: [0; 7],
        };
        let mut raw = std::ptr::null_mut();
        let mut bytes = 0;
        let call = || {
            // SAFETY: descriptor arrays and both bank leases outlive this
            // synchronous call; the native owner copies descriptors/points.
            let status = unsafe { (functions.create)(&desc, &mut raw, &mut bytes) };
            status_to_gpu_result("gafime_gpu_semantic_region_query_create_rt_v1", status)
        };
        let status = if let Some(other) = &paired {
            primary.with_peer_lock(other, call)
        } else {
            let _guard = primary.lock();
            let mut call = call;
            call()
        };
        let query = Self {
            raw,
            functions,
            primary,
            _paired: paired,
            persistent_bytes: usize::try_from(bytes).unwrap_or(usize::MAX),
            regions,
        };
        // Some native failures can leave free-only ownership after allocation.
        // Adopt it before propagating status so those paths still clean up.
        status.map_err(GpuNativeEvidenceExecutor::semantic_error)?;
        if query.raw.is_null() || query.persistent_bytes > budget {
            return Err(SemanticError::Invalid(
                "compact native query returned invalid ownership or budget",
            ));
        }
        Ok(query)
    }

    fn execute(
        &mut self,
        mask: u32,
        finalizers: u32,
        labels: Option<&LabelSet>,
        budget: usize,
    ) -> SemanticResult<(Vec<ExactStats>, usize)> {
        let label_count = labels.map_or(0, |value| value.rows().len());
        let host_bytes = checked_add(
            label_count
                .checked_mul(9)
                .ok_or(SemanticError::Invalid("compact label bytes overflow"))?,
            self.regions
                .checked_mul(std::mem::size_of::<ExactStats>())
                .ok_or(SemanticError::Invalid("compact output bytes overflow"))?,
        )?;
        let native_budget = remaining(budget, host_bytes)?;
        let rows: Vec<u64> = labels.map_or_else(Vec::new, |value| {
            value.rows().iter().map(|row| *row as u64).collect()
        });
        let values: Vec<u8> = labels
            .map(|value| {
                value
                    .values()?
                    .iter()
                    .map(|value| match *value {
                        0.0 => Ok(0),
                        1.0 => Ok(1),
                        _ => Err(SemanticError::Invalid(
                            "compact Gini requires binary labels",
                        )),
                    })
                    .collect::<SemanticResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        let desc = ExecuteDesc {
            abi_version: QUERY_ABI,
            struct_size: std::mem::size_of::<ExecuteDesc>() as u32,
            statistic_mask: mask,
            finalizer_mask: finalizers,
            labels: BinaryLabels {
                abi_version: QUERY_ABI,
                struct_size: std::mem::size_of::<BinaryLabels>() as u32,
                row_indices: rows.as_ptr(),
                values: values.as_ptr(),
                count: rows.len() as u64,
                reserved: [0; 6],
            },
            max_temporary_bytes: native_budget as u64,
            reserved: [0; 7],
        };
        let mut records = vec![ExactStats::default(); self.regions];
        let mut table = StatsTable {
            abi_version: QUERY_ABI,
            struct_size: std::mem::size_of::<StatsTable>() as u32,
            requested_statistic_mask: mask,
            finalized_mask: finalizers,
            capacity: self.regions as u64,
            count: 0,
            records: records.as_mut_ptr(),
            reserved: [0; 8],
        };
        let mut peak = 0;
        // SAFETY: this exclusive owner retains its immutable bank snapshots
        // and DSO. All input/output buffers remain live through synchronization.
        let status = unsafe { (self.functions.execute)(self.raw, &desc, &mut table, &mut peak) };
        status_to_gpu_result("gafime_gpu_semantic_region_query_execute_rt_v1", status)
            .map_err(GpuNativeEvidenceExecutor::semantic_error)?;
        if table.count != self.regions as u64
            || table.requested_statistic_mask != mask
            || table.finalized_mask != finalizers
            || peak > native_budget as u64
        {
            return Err(SemanticError::Invalid(
                "compact native results violate shape, masks or budget",
            ));
        }
        for record in &records {
            validate_record(
                record,
                self.primary.rows(),
                label_count as u64,
                mask,
                finalizers,
            )?;
        }
        Ok((records, checked_add(host_bytes, peak as usize)?))
    }
}

// These are protocol consistency checks, not another statistics engine. The
// device owns floating finalization; Core/device bit parity qualifies that
// arithmetic separately. Cross-channel counts and definedness must nevertheless
// agree before a native record can participate in Rust selection.
fn validate_record(
    record: &ExactStats,
    rows: u64,
    labels: u64,
    mask: u32,
    finalizers: u32,
) -> SemanticResult<()> {
    let invalid = || SemanticError::Invalid("compact native count/state consistency failure");
    let paired = [
        record.paired_n00,
        record.paired_n01,
        record.paired_n10,
        record.paired_n11,
    ];
    let labeled = [
        record.label_outside_0,
        record.label_outside_1,
        record.label_inside_0,
        record.label_inside_1,
    ];
    let sum = |counts: &[u64]| counts.iter().copied().try_fold(0u64, u64::checked_add);
    if record.row_count != rows
        || record.reserved != [0; 4]
        || record.occupancy_inside > rows
        || labels > rows
        || (mask & OCCUPANCY == 0 && record.occupancy_inside != 0)
        || (mask & PAIRED != 0 && sum(&paired) != Some(rows))
        || (mask & PAIRED == 0 && paired != [0; 4])
        || (mask & LABELED != 0
            && (record.label_support != labels || sum(&labeled) != Some(labels)))
        || (mask & LABELED == 0 && (record.label_support != 0 || labeled != [0; 4]))
    {
        return Err(invalid());
    }
    let primary_inside = if mask & PAIRED != 0 {
        let paired_inside = record
            .paired_n10
            .checked_add(record.paired_n11)
            .ok_or_else(invalid)?;
        if mask & OCCUPANCY != 0 && record.occupancy_inside != paired_inside {
            return Err(invalid());
        }
        Some(paired_inside)
    } else if mask & OCCUPANCY != 0 {
        Some(record.occupancy_inside)
    } else {
        None
    };
    if let Some(inside) = primary_inside {
        if mask & LABELED != 0
            && (sum(&labeled[2..]).ok_or_else(invalid)? > inside
                || sum(&labeled[..2]).ok_or_else(invalid)? > rows - inside)
        {
            return Err(invalid());
        }
    }
    let scalar = |flag, value: f32, state, expected_state| -> SemanticResult<()> {
        if finalizers & flag == 0 {
            if value.to_bits() != 0 || state != 0 {
                return Err(invalid());
            }
        } else if state != expected_state
            || !value.is_finite()
            || (state != GAFIME_SEMANTIC_SCALAR_MEASURED && value.to_bits() != 0)
        {
            return Err(invalid());
        }
        Ok(())
    };
    let fraction_state = if rows == 0 {
        GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT
    } else {
        GAFIME_SEMANTIC_SCALAR_MEASURED
    };
    scalar(
        OCCUPANCY_SCORE,
        record.occupancy,
        record.occupancy_state,
        fraction_state,
    )?;
    scalar(
        AGREEMENT_SCORE,
        record.paired_agreement,
        record.paired_agreement_state,
        fraction_state,
    )?;
    let union = sum(&paired[1..]).ok_or_else(invalid)?;
    scalar(
        IOU_SCORE,
        record.paired_iou,
        record.paired_iou_state,
        if union == 0 {
            GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND
        } else {
            GAFIME_SEMANTIC_SCALAR_MEASURED
        },
    )?;
    // Each pair is one class or one branch marginal in the 2x2 table.
    let empty_margin = [
        (labeled[0], labeled[2]),
        (labeled[1], labeled[3]),
        (labeled[0], labeled[1]),
        (labeled[2], labeled[3]),
    ]
    .into_iter()
    .any(|(left, right)| left == 0 && right == 0);
    let gini_state = if labels < 2 {
        GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT
    } else if empty_margin {
        GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND
    } else {
        GAFIME_SEMANTIC_SCALAR_MEASURED
    };
    scalar(
        GINI_SCORE,
        record.labeled_gini_gain,
        record.labeled_gini_gain_state,
        gini_state,
    )?;
    for (flag, value) in [
        (OCCUPANCY_SCORE, record.occupancy),
        (AGREEMENT_SCORE, record.paired_agreement),
        (IOU_SCORE, record.paired_iou),
    ] {
        if finalizers & flag != 0 && !(0.0..=1.0).contains(&value) {
            return Err(invalid());
        }
    }
    Ok(())
}

struct CachedQuery {
    frame_id: u64,
    paired_id: Option<u64>,
    candidates: Vec<FeatureId>,
    dependencies: MaterializedColumns,
    paired_dependencies: Option<MaterializedColumns>,
    query: Query,
}

impl CachedQuery {
    fn identity_bytes(candidate_capacity: usize) -> SemanticResult<usize> {
        candidate_capacity
            .checked_mul(std::mem::size_of::<FeatureId>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()))
            .ok_or(SemanticError::Invalid(
                "compact cache identity bytes overflow",
            ))
    }

    fn bytes(&self) -> SemanticResult<usize> {
        checked_add(
            checked_add(
                self.query.persistent_bytes,
                Self::identity_bytes(self.candidates.capacity())?,
            )?,
            checked_add(
                self.dependencies.bytes(),
                self.paired_dependencies
                    .as_ref()
                    .map_or(0, MaterializedColumns::bytes),
            )?,
        )
    }
}

fn region_predicates<'a>(
    registry: &'a CandidateRegistry,
    candidate: &'a FeatureId,
) -> SemanticResult<&'a [FeatureId]> {
    match registry.program(*candidate)?.op() {
        FeatureOp::HardPredicate { .. } => Ok(std::slice::from_ref(candidate)),
        FeatureOp::DecisionRegion { terms } => Ok(terms),
        _ => Err(SemanticError::Unsupported(
            "compact binary evidence requires canonical predicates or regions",
        )),
    }
}

fn query_host_reservation(candidates: usize, terms: usize) -> SemanticResult<usize> {
    let descriptors_and_axes = terms
        .checked_mul(
            std::mem::size_of::<GafimeSemanticFrozenRegionTerm>()
                + std::mem::size_of::<u32>()
                + std::mem::size_of::<FeatureId>(),
        )
        .and_then(|bytes| {
            candidates
                .checked_add(1)
                .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
                .and_then(|offsets| bytes.checked_add(offsets))
        })
        .ok_or(SemanticError::Invalid(
            "compact query descriptor bytes overflow",
        ))?;
    checked_add(
        descriptors_and_axes,
        CachedQuery::identity_bytes(candidates)?,
    )
}

/// Observed operations only, not inferred kernel/RT saturation measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalCompactDiagnostics {
    pub queries_created: u64,
    pub queries_reused: u64,
    pub executions: u64,
    pub evaluated_regions: u64,
    pub last_live_bytes: usize,
    pub peak_explicit_bytes: usize,
}

pub(crate) struct LocalCompactExecution {
    functions: Functions,
    cache: Option<CachedQuery>,
    diagnostics: LocalCompactDiagnostics,
}

impl LocalCompactExecution {
    pub(super) fn clear(&mut self) {
        self.cache = None;
        self.diagnostics.last_live_bytes = 0;
    }

    pub(super) fn evaluate(
        &mut self,
        executor: &mut GpuNativeEvidenceExecutor,
        context: (&CandidateRegistry, &FeatureFrame),
        candidates: &[FeatureId],
        channels: &[EvidenceChannel],
        retained: Option<&MaterializedColumns>,
        max_bytes: usize,
    ) -> SemanticResult<Option<CompactEvidenceBatch>> {
        let (registry, frame) = context;
        if !channels
            .iter()
            .any(|channel| binary_definition(channel.definition()))
        {
            self.clear();
            return Ok(None);
        }
        executor.validate_context(registry, frame)?;
        if !channels
            .iter()
            .all(|channel| binary_definition(channel.definition()))
        {
            return Err(SemanticError::Unsupported("local compact evaluation requires an all-binary evidence set; use separate evaluations for ordinary statistics"));
        }
        let mut paired_view: Option<&Arc<FeatureFrame>> = None;
        let mut labels: Option<&Arc<LabelSet>> = None;
        let mut mask = 0;
        let mut finalizers = 0;
        for channel in channels {
            match channel.definition() {
                EvidenceDefinition::BinaryOccupancy => {
                    mask |= OCCUPANCY;
                    finalizers |= OCCUPANCY_SCORE;
                }
                EvidenceDefinition::BinaryPaired { statistic, view } => {
                    if !frame.aligned_with(view)
                        || paired_view.is_some_and(|old| old.id() != view.id())
                    {
                        return Err(SemanticError::Unsupported(
                            "one compact batch requires one exactly aligned paired view",
                        ));
                    }
                    paired_view = Some(view);
                    mask |= PAIRED;
                    finalizers |= match statistic {
                        BinaryPairedStatistic::Agreement => AGREEMENT_SCORE,
                        BinaryPairedStatistic::IntersectionOverUnion => IOU_SCORE,
                    };
                }
                EvidenceDefinition::BinaryLabeledGiniGain { labels: context } => {
                    if let Some(context) = context {
                        if context.frame_id() != frame.id()
                            || labels.is_some_and(|old| old.id() != context.id())
                        {
                            return Err(SemanticError::Unsupported(
                                "one compact batch requires one frame-bound binary label set",
                            ));
                        }
                        labels = Some(context);
                        mask |= LABELED;
                        finalizers |= GINI_SCORE;
                    }
                }
                _ => unreachable!("binary vocabulary checked above"),
            }
        }
        if mask == 0 {
            self.clear();
            return Ok(Some(CompactEvidenceBatch::new(
                MaterializedColumns::empty_resident(registry, frame, executor.backend.kind)?,
                vec![
                    vec![
                        EvidenceValue::Unavailable {
                            reason: UnavailableReason::MissingLabels,
                            support: 0
                        };
                        candidates.len()
                    ];
                    channels.len()
                ],
                0,
            )));
        }
        let paired_id = paired_view.map(|view| view.id());
        let reuse = self.cache.as_ref().is_some_and(|cache| {
            cache.frame_id == frame.id()
                && cache.paired_id == paired_id
                && cache.candidates == candidates
        });
        let mut build_peak = 0;
        if !reuse {
            self.clear();
            let mut term_count = 0usize;
            // Borrow the canonical terms for an allocation-free sizing pass.
            // Do not copy a vector per region or allocate a tree of axes before
            // the request's explicit host buffers have been admitted.
            for candidate in candidates {
                let predicates = region_predicates(registry, candidate)?;
                term_count = checked_add(term_count, predicates.len())?;
                for &predicate in predicates {
                    if !matches!(
                        registry.program(predicate)?.op(),
                        FeatureOp::HardPredicate { .. }
                    ) {
                        return Err(SemanticError::Invalid(
                            "compact region contains a non-predicate term",
                        ));
                    }
                }
            }
            // Metadata, host marshalling and both input banks are admitted
            // before native ownership is acquired. No candidate columns are
            // allocated; only predicate operands are materialized/reused.
            let host_bytes = query_host_reservation(candidates.len(), term_count)?;
            let bank_budget = remaining(max_bytes, host_bytes)? / 2;
            let mut axes = Vec::with_capacity(term_count);
            for candidate in candidates {
                for &predicate in region_predicates(registry, candidate)? {
                    let FeatureOp::HardPredicate { input, .. } = registry.program(predicate)?.op()
                    else {
                        unreachable!("validated predicates");
                    };
                    axes.push(*input);
                }
            }
            axes.sort_unstable();
            axes.dedup();
            let dependencies =
                executor.materialize(registry, frame, &axes, retained, bank_budget)?;
            let paired_dependencies = paired_view
                .map(|view| executor.materialize(registry, view, &axes, None, bank_budget))
                .transpose()?;
            let primary = executor.resident_bank(&dependencies, frame)?;
            let paired = paired_dependencies
                .as_ref()
                .zip(paired_view)
                .map(|(values, view)| executor.resident_bank(values, view))
                .transpose()?;
            let source_slots = dependencies.resident_slots()?;
            let paired_slot_map = paired_dependencies
                .as_ref()
                .map(MaterializedColumns::resident_slots)
                .transpose()?;
            let mut terms = Vec::with_capacity(term_count);
            let mut paired_slots = Vec::with_capacity(if paired_slot_map.is_some() {
                term_count
            } else {
                0
            });
            let mut offsets = Vec::with_capacity(candidates.len() + 1);
            offsets.push(0);
            for candidate in candidates {
                for &predicate in region_predicates(registry, candidate)? {
                    let FeatureOp::HardPredicate {
                        input,
                        comparison,
                        threshold_bits,
                    } = registry.program(predicate)?.op()
                    else {
                        unreachable!("validated predicates");
                    };
                    let physical = GpuNativeEvidenceExecutor::lower_region_term(
                        frame.profile(),
                        source_slots,
                        *input,
                        *comparison,
                        threshold_bits,
                    )?;
                    terms.push(GafimeSemanticFrozenRegionTerm {
                        input_slot: physical.input_slot,
                        relation: physical.relation.raw(),
                        threshold_bits: physical.threshold_bits,
                    });
                    if let Some(map) = paired_slot_map {
                        paired_slots.push(
                            *map.get(input)
                                .ok_or(SemanticError::Invalid("paired query axis is missing"))?,
                        );
                    }
                }
                offsets.push(
                    u32::try_from(terms.len())
                        .map_err(|_| SemanticError::Invalid("compact term offsets overflow"))?,
                );
            }
            let input_bytes = checked_add(
                dependencies.bytes(),
                paired_dependencies
                    .as_ref()
                    .map_or(0, MaterializedColumns::bytes),
            )?;
            let used = checked_add(input_bytes, host_bytes)?;
            let query = Query::create(
                self.functions,
                primary,
                paired,
                &terms,
                &paired_slots,
                &offsets,
                remaining(max_bytes, used)?,
            )?;
            build_peak = checked_add(used, query.persistent_bytes)?;
            self.cache = Some(CachedQuery {
                frame_id: frame.id(),
                paired_id,
                candidates: candidates.to_vec(),
                dependencies,
                paired_dependencies,
                query,
            });
            self.diagnostics.queries_created += 1;
        } else {
            self.diagnostics.queries_reused += 1;
        }
        let cache = self.cache.as_mut().expect("query constructed above");
        let live_bytes = cache.bytes()?;
        let (records, execution_peak) = cache.query.execute(
            mask,
            finalizers,
            labels.map(AsRef::as_ref),
            remaining(max_bytes, live_bytes)?,
        )?;
        let peak = build_peak.max(checked_add(live_bytes, execution_peak)?);
        let channel_values = channels
            .iter()
            .map(|channel| {
                records
                    .iter()
                    .map(|record| {
                        let (value, state, support) = match channel.definition() {
                            EvidenceDefinition::BinaryOccupancy => {
                                (record.occupancy, record.occupancy_state, record.row_count)
                            }
                            EvidenceDefinition::BinaryPaired {
                                statistic: BinaryPairedStatistic::Agreement,
                                ..
                            } => (
                                record.paired_agreement,
                                record.paired_agreement_state,
                                record.row_count,
                            ),
                            EvidenceDefinition::BinaryPaired {
                                statistic: BinaryPairedStatistic::IntersectionOverUnion,
                                ..
                            } => (record.paired_iou, record.paired_iou_state, record.row_count),
                            EvidenceDefinition::BinaryLabeledGiniGain { labels: Some(_) } => (
                                record.labeled_gini_gain,
                                record.labeled_gini_gain_state,
                                record.label_support,
                            ),
                            EvidenceDefinition::BinaryLabeledGiniGain { labels: None } => {
                                return Ok(EvidenceValue::Unavailable {
                                    reason: UnavailableReason::MissingLabels,
                                    support: 0,
                                })
                            }
                            _ => unreachable!("binary vocabulary checked above"),
                        };
                        decode(value, state, support)
                    })
                    .collect::<SemanticResult<Vec<_>>>()
            })
            .collect::<SemanticResult<Vec<_>>>()?;
        self.diagnostics.executions += 1;
        self.diagnostics.evaluated_regions += candidates.len() as u64;
        self.diagnostics.last_live_bytes = live_bytes;
        self.diagnostics.peak_explicit_bytes = self.diagnostics.peak_explicit_bytes.max(peak);
        Ok(Some(CompactEvidenceBatch::new(
            cache.dependencies.clone(),
            channel_values,
            peak,
        )))
    }
}

fn decode(value: f32, state: u32, support: u64) -> SemanticResult<EvidenceValue> {
    let support = usize::try_from(support)
        .map_err(|_| SemanticError::Invalid("compact support overflows host size"))?;
    let reason = match state {
        GAFIME_SEMANTIC_SCALAR_MEASURED if value.is_finite() => {
            return Ok(EvidenceValue::measured_f32(value, support))
        }
        GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT => UnavailableReason::InsufficientSupport,
        GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND => UnavailableReason::ConstantOperand,
        GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION => UnavailableReason::DegenerateReduction,
        GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION => UnavailableReason::NonFiniteReduction,
        _ => {
            return Err(SemanticError::Invalid(
                "compact finalizer returned an invalid scalar state",
            ))
        }
    };
    Ok(EvidenceValue::Unavailable { reason, support })
}

pub(super) fn binary_definition(definition: &EvidenceDefinition) -> bool {
    matches!(
        definition,
        EvidenceDefinition::BinaryOccupancy
            | EvidenceDefinition::BinaryPaired { .. }
            | EvidenceDefinition::BinaryLabeledGiniGain { .. }
    )
}

impl GpuBackend {
    /// Local experiment only: compact binary evidence may execute before dense
    /// region materialization. Unsupported channels/geometry fail closed.
    pub fn local_compact_rt_semantic_executor(
        &self,
    ) -> Result<GpuNativeEvidenceExecutor, GpuSysError> {
        let mut executor = self.local_rt_semantic_executor()?;
        let local = self.functions.local_cmake_experiment;
        executor.local_compact_execution = Some(LocalCompactExecution {
            functions: Functions {
                create: local.semantic_region_query_create_rt.ok_or(
                    GpuSysError::MissingFunction("gafime_gpu_semantic_region_query_create_rt_v1"),
                )?,
                execute: local.semantic_region_query_execute_rt.ok_or(
                    GpuSysError::MissingFunction("gafime_gpu_semantic_region_query_execute_rt_v1"),
                )?,
                free: local
                    .semantic_region_query_free_rt
                    .ok_or(GpuSysError::MissingFunction(
                        "gafime_gpu_semantic_region_query_free_rt_v1",
                    ))?,
                coverage: local.semantic_region_query_materialize_coverage_rt.ok_or(
                    GpuSysError::MissingFunction(
                        "gafime_gpu_semantic_region_query_materialize_coverage_rt_v1",
                    ),
                )?,
            },
            cache: None,
            diagnostics: LocalCompactDiagnostics::default(),
        });
        Ok(executor)
    }
}

impl OwnedSemanticBank {
    pub(crate) fn materialize_local_region_count(
        &self,
        output_slot: u32,
        regions: &[Vec<SemanticFrozenRegionTerm>],
        max_bytes: usize,
    ) -> SemanticResult<usize> {
        if self.profile() != PrecisionProfile::Fp32 || regions.len() < 2 || regions.len() > 64 {
            return Err(SemanticError::Unsupported(
                "local region-count requires fp32 and two to sixty-four regions",
            ));
        }
        let local = self.inner.functions.local_cmake_experiment;
        let missing =
            || SemanticError::Unsupported("local conditional-query function table is incomplete");
        let functions = Functions {
            create: local.semantic_region_query_create_rt.ok_or_else(missing)?,
            execute: local.semantic_region_query_execute_rt.ok_or_else(missing)?,
            free: local.semantic_region_query_free_rt.ok_or_else(missing)?,
            coverage: local
                .semantic_region_query_materialize_coverage_rt
                .ok_or_else(missing)?,
        };
        let term_count = regions
            .iter()
            .try_fold(0usize, |sum, terms| checked_add(sum, terms.len()))?;
        let host_bytes = checked_add(
            term_count
                .checked_mul(std::mem::size_of::<GafimeSemanticFrozenRegionTerm>())
                .ok_or(SemanticError::Invalid(
                    "region-count descriptor bytes overflow",
                ))?,
            (regions.len() + 1) * 4,
        )?;
        let budget = remaining(max_bytes, host_bytes)?;
        let mut terms = Vec::with_capacity(term_count);
        let mut offsets = Vec::with_capacity(regions.len() + 1);
        offsets.push(0);
        for region in regions {
            for term in region {
                if term.input_slot == output_slot {
                    return Err(SemanticError::Invalid(
                        "region-count output aliases an input",
                    ));
                }
                terms.push(GafimeSemanticFrozenRegionTerm {
                    input_slot: term.input_slot,
                    relation: term.relation.raw(),
                    threshold_bits: term.threshold_bits,
                });
            }
            offsets.push(
                u32::try_from(terms.len())
                    .map_err(|_| SemanticError::Invalid("region-count offset overflow"))?,
            );
        }
        let mut query =
            Query::create(functions, self.clone(), None, &terms, &[], &offsets, budget)?;
        let available = remaining(budget, query.persistent_bytes)?;
        let (counts, execute_peak) = query.execute(OCCUPANCY, 0, None, available)?;
        drop(counts);
        let mut peak = 0;
        let _guard = self.lock();
        // SAFETY: query and bank/DSO are retained, the native bank lock is held,
        // and output commits only after a complete synchronous successful call.
        let status = unsafe {
            (query.functions.coverage)(query.raw, output_slot, available as u64, &mut peak)
        };
        status_to_gpu_result(
            "gafime_gpu_semantic_region_query_materialize_coverage_rt_v1",
            status,
        )
        .map_err(GpuNativeEvidenceExecutor::semantic_error)?;
        if peak > available as u64 {
            return Err(SemanticError::Invalid(
                "region-count returned an invalid temporary peak",
            ));
        }
        checked_add(
            host_bytes,
            checked_add(query.persistent_bytes, execute_peak.max(peak as usize))?,
        )
    }
}

impl GpuNativeEvidenceExecutor {
    pub fn local_compact_diagnostics(&self) -> Option<LocalCompactDiagnostics> {
        self.local_compact_execution
            .as_ref()
            .map(|state| state.diagnostics)
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn local_compact_layout_matches_native_header() {
        assert_eq!(std::mem::size_of::<QueryDesc>(), 144);
        assert_eq!(std::mem::size_of::<BinaryLabels>(), 80);
        assert_eq!(std::mem::size_of::<ExecuteDesc>(), 160);
        assert_eq!(std::mem::size_of::<ExactStats>(), 152);
        assert_eq!(std::mem::size_of::<StatsTable>(), 104);
    }

    #[test]
    fn compact_cache_identity_reservation_tracks_capacity_and_overflow() {
        let base = CachedQuery::identity_bytes(0).unwrap();
        assert_eq!(base, std::mem::size_of::<CachedQuery>());
        assert_eq!(
            CachedQuery::identity_bytes(8192).unwrap() - base,
            8192 * std::mem::size_of::<FeatureId>()
        );
        assert!(CachedQuery::identity_bytes(usize::MAX).is_err());
    }

    #[test]
    fn compact_host_reservation_admits_axes_descriptors_offsets_and_identity() {
        let (candidates, terms) = (8192, 8192 * 64);
        assert_eq!(
            query_host_reservation(candidates, terms).unwrap(),
            CachedQuery::identity_bytes(candidates).unwrap()
                + (candidates + 1) * std::mem::size_of::<u32>()
                + terms
                    * (std::mem::size_of::<GafimeSemanticFrozenRegionTerm>()
                        + std::mem::size_of::<u32>()
                        + std::mem::size_of::<FeatureId>())
        );
        assert!(query_host_reservation(usize::MAX, 1).is_err());
        assert!(query_host_reservation(1, usize::MAX).is_err());
        let admitted = query_host_reservation(2, 8).unwrap();
        assert!(remaining(admitted - 1, admitted).is_err());
    }

    #[test]
    fn compact_records_reject_cross_channel_and_state_contradictions() {
        let valid = ExactStats {
            row_count: 8,
            occupancy_inside: 4,
            paired_n00: 3,
            paired_n01: 1,
            paired_n10: 1,
            paired_n11: 3,
            label_support: 4,
            label_outside_0: 1,
            label_outside_1: 1,
            label_inside_0: 1,
            label_inside_1: 1,
            occupancy: 0.5,
            paired_agreement: 0.75,
            paired_iou: 0.6,
            occupancy_state: GAFIME_SEMANTIC_SCALAR_MEASURED,
            paired_agreement_state: GAFIME_SEMANTIC_SCALAR_MEASURED,
            paired_iou_state: GAFIME_SEMANTIC_SCALAR_MEASURED,
            labeled_gini_gain_state: GAFIME_SEMANTIC_SCALAR_MEASURED,
            ..ExactStats::default()
        };
        let check = |record: &ExactStats| {
            validate_record(
                record,
                8,
                4,
                OCCUPANCY | PAIRED | LABELED,
                OCCUPANCY_SCORE | AGREEMENT_SCORE | IOU_SCORE | GINI_SCORE,
            )
        };
        check(&valid).unwrap();
        let mut bad = valid;
        bad.occupancy_inside = 0;
        assert!(
            check(&bad).is_err(),
            "primary counts disagree across channels"
        );
        bad = valid;
        bad.paired_n00 = u64::MAX;
        assert!(check(&bad).is_err(), "count addition must not wrap");
        bad = valid;
        bad.occupancy_state = GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT;
        assert!(
            check(&bad).is_err(),
            "nonempty occupancy cannot be undefined"
        );
        bad = valid;
        bad.label_inside_0 = 5;
        bad.label_inside_1 = 0;
        bad.label_outside_0 = 0;
        bad.label_outside_1 = 0;
        bad.label_support = 5;
        assert!(
            validate_record(&bad, 8, 5, OCCUPANCY | PAIRED | LABELED, 15).is_err(),
            "labeled inside exceeds all inside rows"
        );
        bad = valid;
        bad.paired_iou = f32::NAN;
        assert!(check(&bad).is_err());
    }

    #[test]
    fn compact_records_reject_unrequested_outputs_but_allow_degenerate_zero() {
        let mut record = ExactStats {
            row_count: 8,
            ..ExactStats::default()
        };
        validate_record(&record, 8, 0, OCCUPANCY, 0).unwrap();
        record.paired_n00 = 8;
        assert!(validate_record(&record, 8, 0, OCCUPANCY, 0).is_err());
        record.paired_n00 = 0;
        record.occupancy_state = GAFIME_SEMANTIC_SCALAR_MEASURED;
        assert!(validate_record(&record, 8, 0, OCCUPANCY, 0).is_err());
        record.occupancy_state = 0;
        record.paired_n00 = 8;
        record.paired_iou_state = GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND;
        validate_record(&record, 8, 0, PAIRED, IOU_SCORE).unwrap();
        record.paired_iou_state = GAFIME_SEMANTIC_SCALAR_MEASURED;
        assert!(validate_record(&record, 8, 0, PAIRED, IOU_SCORE).is_err());
    }
}
