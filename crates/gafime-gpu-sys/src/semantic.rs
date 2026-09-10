//! Safe ownership boundary for the optional resident semantic-arithmetic ABI.
//!
//! The native table owns only physical, typed column slots.  This module keeps
//! the payload library alive, validates profile/device/library identity before
//! dispatch, serializes mutable bank access, and exposes synchronous safe Rust
//! calls.  Candidate identities, evidence definitions, provenance and policy
//! deliberately stay in `gafime-orchestrator`.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use gafime_orchestrator::semantic::{
    AssociationContext, AssociationStatistic, CandidateRegistry, EvidenceDefinition, EvidenceValue,
    FeatureFrame, FeatureId, FeatureOp, FrozenMeans, FrozenThreshold, MaterializedColumns,
    NativeEvidenceExecutor, NumericColumn, PredicateComparator, SemanticError, SemanticResult,
    UnavailableReason,
};
use gafime_types::{
    BackendKind, GafimeConstBufferView, GafimeGpuSemanticBank, GafimeMutableBufferView,
    GafimeSemanticAssociationBatch, GafimeSemanticBankDesc, GafimeSemanticColumnMeanBatch,
    GafimeSemanticEdge, GafimeSemanticEdgeEnergyBatch, GafimeSemanticForecastRequest,
    GafimeSemanticFrozenRegionTerm, GafimeSemanticMemoryForecast, GafimeSemanticPearsonBatch,
    GafimeSemanticProgramBatch, GafimeSemanticProgramNode, GafimeSemanticRegionTermSlice,
    GafimeSemanticScalarResultTable, GafimeSemanticSparseGatherBatch, GafimeSliceU32,
    GafimeSliceU64, PrecisionProfile, SemanticScalarState, GAFIME_BACKEND_METAL,
    GAFIME_BUFFER_FLAG_CONTIGUOUS, GAFIME_BUFFER_FLAG_HOST, GAFIME_DTYPE_F32, GAFIME_DTYPE_F64,
    GAFIME_MATRIX_COLUMN_MAJOR, GAFIME_SEMANTIC_ASSOCIATION_ABSOLUTE,
    GAFIME_SEMANTIC_ASSOCIATION_FIXED_CORRECTED_NMI, GAFIME_SEMANTIC_ASSOCIATION_NONNEGATIVE,
    GAFIME_SEMANTIC_ASSOCIATION_PEARSON, GAFIME_SEMANTIC_ASSOCIATION_SIGNED,
    GAFIME_SEMANTIC_ASSOCIATION_SPEARMAN, GAFIME_SEMANTIC_PEARSON_ABSOLUTE,
    GAFIME_SEMANTIC_PEARSON_SIGNED, GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
    GAFIME_SEMANTIC_PRIMITIVE_MASK_COLUMN_MEANS,
    GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY,
    GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_ASSOCIATION,
    GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER, GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE,
    GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT, GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION,
    GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE,
    GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT,
    GAFIME_SEMANTIC_PROGRAM_OP_MASK_FROZEN_REGION_CONJUNCTION,
    GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN, GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE,
    GAFIME_SEMANTIC_PROGRAM_SOFTSIGN, GAFIME_SEMANTIC_PROGRAM_SOURCE,
    GAFIME_SEMANTIC_REGION_GREATER_THAN, GAFIME_SEMANTIC_REGION_LESS_EQUAL,
    GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND, GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION,
    GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT, GAFIME_SEMANTIC_SCALAR_MEASURED,
    GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION, GAFIME_SEMANTIC_STATISTIC_MASK_FIXED_CORRECTED_NMI,
    GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON, GAFIME_SEMANTIC_STATISTIC_MASK_SPEARMAN,
};
use libloading::Library;

use crate::{
    abi::{status_to_gpu_result, GpuFunctionTable, GpuSysError},
    backend::GpuBackend,
};

/// Profile-typed arithmetic operation in a resident physical bank.
#[derive(Clone, Debug)]
pub enum SemanticProgramNode {
    Source {
        output_slot: u32,
    },
    AbsoluteDifference {
        output_slot: u32,
        left_slot: u32,
        right_slot: u32,
    },
    Softsign {
        output_slot: u32,
        input_slot: u32,
    },
    CenteredProduct {
        output_slot: u32,
        operand_slots: Vec<u32>,
        mean_bits: Vec<u64>,
    },
    /// Closed hard-AND over physical input slots and exact frozen threshold
    /// bits. The semantic registry decides which atoms are eligible; native
    /// arithmetic receives no feature/evidence identity or fitting history.
    FrozenRegionConjunction {
        output_slot: u32,
        terms: Vec<SemanticFrozenRegionTerm>,
    },
}

/// Arithmetic association selector for the generic resident-slot ABI entry.
/// It deliberately carries no contextual provenance: Rust maps reference,
/// paired-view, and label operands to ordinary physical slots first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeAssociationStatistic {
    Pearson,
    Spearman,
    FixedCorrectedNmi { bins: u32 },
}

impl NativeAssociationStatistic {
    const fn raw(self) -> u32 {
        match self {
            Self::Pearson => GAFIME_SEMANTIC_ASSOCIATION_PEARSON,
            Self::Spearman => GAFIME_SEMANTIC_ASSOCIATION_SPEARMAN,
            Self::FixedCorrectedNmi { .. } => GAFIME_SEMANTIC_ASSOCIATION_FIXED_CORRECTED_NMI,
        }
    }

    const fn fixed_nmi_bins(self) -> u32 {
        match self {
            Self::FixedCorrectedNmi { bins } => bins,
            Self::Pearson | Self::Spearman => 0,
        }
    }
}

/// Arithmetic post-processing for association values. It is kept distinct
/// from semantic evidence names and policies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeAssociationPresentation {
    Signed,
    Absolute,
    Nonnegative,
}

impl NativeAssociationPresentation {
    const fn raw(self) -> u32 {
        match self {
            Self::Signed => GAFIME_SEMANTIC_ASSOCIATION_SIGNED,
            Self::Absolute => GAFIME_SEMANTIC_ASSOCIATION_ABSOLUTE,
            Self::Nonnegative => GAFIME_SEMANTIC_ASSOCIATION_NONNEGATIVE,
        }
    }
}

/// One physical hard-predicate term used by a frozen region conjunction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticFrozenRegionTerm {
    pub input_slot: u32,
    pub relation: SemanticRegionRelation,
    pub threshold_bits: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticRegionRelation {
    LessEqual,
    GreaterThan,
}

impl SemanticRegionRelation {
    const fn raw(self) -> u32 {
        match self {
            Self::LessEqual => GAFIME_SEMANTIC_REGION_LESS_EQUAL,
            Self::GreaterThan => GAFIME_SEMANTIC_REGION_GREATER_THAN,
        }
    }
}

/// Native Pearson presentation choice.  The operands remain generic columns;
/// this does not introduce a target field or an evidence identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticPearsonMode {
    Signed,
    Absolute,
}

impl SemanticPearsonMode {
    const fn raw(self) -> u32 {
        match self {
            Self::Signed => GAFIME_SEMANTIC_PEARSON_SIGNED,
            Self::Absolute => GAFIME_SEMANTIC_PEARSON_ABSOLUTE,
        }
    }
}

/// One declared graph edge.  The caller retains graph identity and declared
/// ordering; native arithmetic receives only these physical row positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticEdge {
    pub left_row: u64,
    pub right_row: u64,
}

/// A profile-preserving scalar returned from the native table.  Fp32 values
/// remain f32 until the orchestrator explicitly reports their exact widening.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SemanticScalarValue {
    F32(f32),
    F64(f64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SemanticScalarResult {
    pub value: SemanticScalarValue,
    pub state: SemanticScalarState,
    pub support: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticMemoryForecast {
    pub resident_bytes: u64,
    pub transient_bytes: u64,
    pub retained_bytes: u64,
}

// Keep validated shape/capability metadata named at allocation and retention
// sites; the several same-width counters must not be swapped positionally.
struct SemanticBankMetadata {
    profile: PrecisionProfile,
    rows: u64,
    source_slots: u32,
    slot_capacity: u32,
    max_program_nodes: u32,
    max_gather_rows: u64,
    max_region_terms: u32,
    bytes: u64,
}

struct SemanticBankInner {
    raw: GafimeGpuSemanticBank,
    functions: GpuFunctionTable,
    // Retains every function pointer's DSO until after the native free call.
    _library: Option<Arc<Library>>,
    owner: Arc<()>,
    backend_kind: BackendKind,
    device_id: u32,
    profile: PrecisionProfile,
    rows: u64,
    source_slots: u32,
    slot_capacity: u32,
    max_program_nodes: u32,
    max_gather_rows: u64,
    max_region_terms: u32,
    bytes: u64,
    // Native bank state is mutable (uploads/materialization), so every safe
    // operation takes this lock.  Pair operations acquire unique banks by
    // stable Arc address to avoid the reversed-bank deadlock case.
    lock: Mutex<()>,
}

// SAFETY: the opaque pointer is never exposed and every safe operation locks
// the bank before calling its synchronous native function.  The payload keeps
// device affinity inside its launcher and the retained library outlives every
// function pointer.  Cross-bank operations verify a shared backend owner,
// device, profile and row shape before native dispatch.
unsafe impl Send for SemanticBankInner {}
// SAFETY: see the `Send` invariant above; the mutex serializes all mutable
// native state reachable through this opaque handle.
unsafe impl Sync for SemanticBankInner {}

impl Drop for SemanticBankInner {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        if let Some(free) = self.functions.semantic_bank_free_v1 {
            // SAFETY: the bank is owned exclusively by this final Arc, the
            // function pointer belongs to the retained payload, and native
            // free is specified as synchronous and idempotent only for this
            // still-live handle.
            // Drop cannot report a release failure.  The ABI nevertheless
            // returns a status, so bind the exact C signature and explicitly
            // discard it here rather than invoking a mismatched void call.
            let _status = unsafe { free(self.raw) };
        }
    }
}

/// Cloneable safe ownership handle for a resident semantic bank.  Clones share
/// one native allocation and one lock; they do not duplicate device storage.
#[derive(Clone)]
pub struct OwnedSemanticBank {
    inner: Arc<SemanticBankInner>,
}

impl std::fmt::Debug for OwnedSemanticBank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedSemanticBank")
            .field("backend_kind", &self.backend_kind())
            .field("device_id", &self.device_id())
            .field("profile", &self.profile())
            .field("rows", &self.rows())
            .field("source_slots", &self.source_slots())
            .field("slot_capacity", &self.slot_capacity())
            .field("max_program_nodes", &self.inner.max_program_nodes)
            .field("max_gather_rows", &self.inner.max_gather_rows)
            .field("max_region_terms", &self.inner.max_region_terms)
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

impl OwnedSemanticBank {
    fn from_parts(
        raw: GafimeGpuSemanticBank,
        functions: GpuFunctionTable,
        library: Option<Arc<Library>>,
        owner: Arc<()>,
        backend_kind: BackendKind,
        device_id: u32,
        metadata: SemanticBankMetadata,
    ) -> Result<Self, GpuSysError> {
        if raw.is_null() {
            return Err(GpuSysError::InvalidInput(
                "semantic bank allocation returned a null handle",
            ));
        }
        Ok(Self {
            inner: Arc::new(SemanticBankInner {
                raw,
                functions,
                _library: library,
                owner,
                backend_kind,
                device_id,
                profile: metadata.profile,
                rows: metadata.rows,
                source_slots: metadata.source_slots,
                slot_capacity: metadata.slot_capacity,
                max_program_nodes: metadata.max_program_nodes,
                max_gather_rows: metadata.max_gather_rows,
                max_region_terms: metadata.max_region_terms,
                bytes: metadata.bytes,
                lock: Mutex::new(()),
            }),
        })
    }

    fn from_backend(
        backend: &GpuBackend,
        raw: GafimeGpuSemanticBank,
        metadata: SemanticBankMetadata,
    ) -> Result<Self, GpuSysError> {
        Self::from_parts(
            raw,
            backend.functions,
            backend.library.clone(),
            backend.instance_id.clone(),
            backend.kind,
            backend.device_id,
            metadata,
        )
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.inner.backend_kind
    }

    pub fn device_id(&self) -> u32 {
        self.inner.device_id
    }

    pub fn profile(&self) -> PrecisionProfile {
        self.inner.profile
    }

    pub fn rows(&self) -> u64 {
        self.inner.rows
    }

    pub fn source_slots(&self) -> u32 {
        self.inner.source_slots
    }

    pub fn slot_capacity(&self) -> u32 {
        self.inner.slot_capacity
    }

    pub fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    fn max_program_nodes(&self) -> u32 {
        self.inner.max_program_nodes
    }

    fn max_gather_rows(&self) -> u64 {
        self.inner.max_gather_rows
    }

    fn max_region_terms(&self) -> u32 {
        self.inner.max_region_terms
    }

    fn lock(&self) -> MutexGuard<'_, ()> {
        self.inner
            .lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn same_bank(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn same_backend_owner(&self, backend: &GpuBackend) -> bool {
        Arc::ptr_eq(&self.inner.owner, &backend.instance_id)
            && self.backend_kind() == backend.kind
            && self.device_id() == backend.device_id
    }

    fn require_same_backend_device_profile(&self, other: &Self) -> Result<(), GpuSysError> {
        if !Arc::ptr_eq(&self.inner.owner, &other.inner.owner)
            || self.backend_kind() != other.backend_kind()
            || self.device_id() != other.device_id()
            || self.profile() != other.profile()
        {
            return Err(GpuSysError::InvalidInput(
                "semantic banks require one payload, device and precision profile",
            ));
        }
        Ok(())
    }

    fn validate_slot_slice(&self, slots: &[u32]) -> Result<(), GpuSysError> {
        if slots.is_empty() || slots.iter().any(|slot| *slot >= self.slot_capacity()) {
            return Err(GpuSysError::InvalidInput(
                "semantic physical slot is empty or out of bounds",
            ));
        }
        Ok(())
    }

    fn with_peer_lock<T>(
        &self,
        other: &Self,
        call: impl FnOnce() -> Result<T, GpuSysError>,
    ) -> Result<T, GpuSysError> {
        self.require_same_backend_device_profile(other)?;
        if self.rows() != other.rows() {
            return Err(GpuSysError::InvalidInput(
                "semantic pairwise arithmetic requires equal bank row counts",
            ));
        }
        self.with_peer_device_lock(other, call)
    }

    fn with_peer_device_lock<T>(
        &self,
        other: &Self,
        call: impl FnOnce() -> Result<T, GpuSysError>,
    ) -> Result<T, GpuSysError> {
        self.require_same_backend_device_profile(other)?;
        if self.same_bank(other) {
            let _guard = self.lock();
            return call();
        }
        let self_key = Arc::as_ptr(&self.inner) as usize;
        let other_key = Arc::as_ptr(&other.inner) as usize;
        if self_key < other_key {
            let _first = self.lock();
            let _second = other.lock();
            call()
        } else {
            let _first = other.lock();
            let _second = self.lock();
            call()
        }
    }

    fn const_view<T>(values: &[T], dtype: u32) -> Result<GafimeConstBufferView, GpuSysError> {
        let element_count = u64::try_from(values.len()).map_err(|_| GpuSysError::SizeOverflow)?;
        let byte_stride =
            u64::try_from(std::mem::size_of::<T>()).map_err(|_| GpuSysError::SizeOverflow)?;
        let byte_length = element_count
            .checked_mul(byte_stride)
            .ok_or(GpuSysError::SizeOverflow)?;
        Ok(GafimeConstBufferView {
            dtype,
            flags: GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS,
            data: values.as_ptr().cast(),
            element_count,
            byte_length,
            byte_stride,
            ..Default::default()
        })
    }

    fn mutable_view<T>(
        values: &mut [T],
        dtype: u32,
    ) -> Result<GafimeMutableBufferView, GpuSysError> {
        let element_capacity =
            u64::try_from(values.len()).map_err(|_| GpuSysError::SizeOverflow)?;
        let byte_stride =
            u64::try_from(std::mem::size_of::<T>()).map_err(|_| GpuSysError::SizeOverflow)?;
        let byte_length = element_capacity
            .checked_mul(byte_stride)
            .ok_or(GpuSysError::SizeOverflow)?;
        Ok(GafimeMutableBufferView {
            dtype,
            flags: GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS,
            data: values.as_mut_ptr().cast(),
            element_capacity,
            byte_length,
            byte_stride,
            ..Default::default()
        })
    }

    fn expected_source_elements(&self) -> Result<usize, GpuSysError> {
        let elements = self
            .rows()
            .checked_mul(u64::from(self.source_slots()))
            .ok_or(GpuSysError::SizeOverflow)?;
        usize::try_from(elements).map_err(|_| GpuSysError::SizeOverflow)
    }

    pub fn upload_f32(&self, source_columns: &[f32]) -> Result<(), GpuSysError> {
        if !matches!(
            self.profile(),
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed
        ) || source_columns.len() != self.expected_source_elements()?
        {
            return Err(GpuSysError::InvalidInput(
                "f32 semantic source upload does not match bank profile or shape",
            ));
        }
        let upload =
            self.inner
                .functions
                .semantic_bank_upload_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_bank_upload_v1",
                ))?;
        let route = self.profile().numeric_route();
        let view = Self::const_view(source_columns, GAFIME_DTYPE_F32)?;
        let _guard = self.lock();
        // SAFETY: this live bank, route and host view are all validated above;
        // the payload call is synchronous and cannot retain the slice pointer.
        let status = unsafe { upload(self.inner.raw, &route, &view) };
        status_to_gpu_result("gafime_gpu_semantic_bank_upload_v1", status)
    }

    pub fn upload_f64(&self, source_columns: &[f64]) -> Result<(), GpuSysError> {
        if self.profile() != PrecisionProfile::Fp64
            || source_columns.len() != self.expected_source_elements()?
        {
            return Err(GpuSysError::InvalidInput(
                "f64 semantic source upload does not match bank profile or shape",
            ));
        }
        let upload =
            self.inner
                .functions
                .semantic_bank_upload_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_bank_upload_v1",
                ))?;
        let route = self.profile().numeric_route();
        let view = Self::const_view(source_columns, GAFIME_DTYPE_F64)?;
        let _guard = self.lock();
        // SAFETY: see `upload_f32`; only the profile-typed source slice differs.
        let status = unsafe { upload(self.inner.raw, &route, &view) };
        status_to_gpu_result("gafime_gpu_semantic_bank_upload_v1", status)
    }

    pub fn materialize(&self, nodes: &[SemanticProgramNode]) -> Result<(), GpuSysError> {
        if nodes.is_empty() {
            return Ok(());
        }
        let node_count = u32::try_from(nodes.len()).map_err(|_| GpuSysError::SizeOverflow)?;
        if node_count > self.max_program_nodes() {
            return Err(GpuSysError::InvalidInput(
                "semantic program node count exceeds payload capability",
            ));
        }
        let materialize =
            self.inner
                .functions
                .semantic_materialize_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_materialize_v1",
                ))?;
        let operand_capacity = nodes.iter().try_fold(0usize, |total, node| {
            let count = match node {
                SemanticProgramNode::Source { .. } | SemanticProgramNode::Softsign { .. } => 1,
                SemanticProgramNode::AbsoluteDifference { .. } => 2,
                SemanticProgramNode::CenteredProduct { operand_slots, .. } => operand_slots.len(),
                SemanticProgramNode::FrozenRegionConjunction { .. } => 0,
            };
            total.checked_add(count).ok_or(GpuSysError::SizeOverflow)
        })?;
        let mean_capacity = nodes.iter().try_fold(0usize, |total, node| {
            let count = match node {
                SemanticProgramNode::CenteredProduct { mean_bits, .. } => mean_bits.len(),
                _ => 0,
            };
            total.checked_add(count).ok_or(GpuSysError::SizeOverflow)
        })?;
        let region_capacity = nodes.iter().try_fold(0usize, |total, node| {
            let count = match node {
                SemanticProgramNode::FrozenRegionConjunction { terms, .. } => terms.len(),
                SemanticProgramNode::Source { .. }
                | SemanticProgramNode::AbsoluteDifference { .. }
                | SemanticProgramNode::Softsign { .. }
                | SemanticProgramNode::CenteredProduct { .. } => 0,
            };
            total.checked_add(count).ok_or(GpuSysError::SizeOverflow)
        })?;
        let mut raw_nodes = Vec::with_capacity(nodes.len());
        // Exact capacities make the executor's pre-dispatch host-temporary
        // accounting truthful instead of relying on allocator growth policy.
        let mut operand_slots = Vec::with_capacity(operand_capacity);
        let mut mean_bits = Vec::with_capacity(mean_capacity);
        let mut region_terms = Vec::with_capacity(region_capacity);
        for node in nodes {
            match node {
                SemanticProgramNode::Source { output_slot } => {
                    if *output_slot >= self.slot_capacity() {
                        return Err(GpuSysError::InvalidInput(
                            "semantic source output slot is out of bounds",
                        ));
                    }
                    let operand_offset = u32::try_from(operand_slots.len())
                        .map_err(|_| GpuSysError::SizeOverflow)?;
                    operand_slots.push(*output_slot);
                    raw_nodes.push(GafimeSemanticProgramNode {
                        opcode: GAFIME_SEMANTIC_PROGRAM_SOURCE,
                        output_slot: *output_slot,
                        operand_offset,
                        operand_count: 1,
                        mean_offset: u32::try_from(mean_bits.len())
                            .map_err(|_| GpuSysError::SizeOverflow)?,
                        mean_count: 0,
                        ..Default::default()
                    });
                }
                SemanticProgramNode::AbsoluteDifference {
                    output_slot,
                    left_slot,
                    right_slot,
                } => {
                    if *output_slot >= self.slot_capacity()
                        || *left_slot >= self.slot_capacity()
                        || *right_slot >= self.slot_capacity()
                    {
                        return Err(GpuSysError::InvalidInput(
                            "semantic absolute-difference slot is out of bounds",
                        ));
                    }
                    let operand_offset = u32::try_from(operand_slots.len())
                        .map_err(|_| GpuSysError::SizeOverflow)?;
                    operand_slots.extend_from_slice(&[*left_slot, *right_slot]);
                    raw_nodes.push(GafimeSemanticProgramNode {
                        opcode: GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE,
                        output_slot: *output_slot,
                        operand_offset,
                        operand_count: 2,
                        mean_offset: u32::try_from(mean_bits.len())
                            .map_err(|_| GpuSysError::SizeOverflow)?,
                        mean_count: 0,
                        ..Default::default()
                    });
                }
                SemanticProgramNode::Softsign {
                    output_slot,
                    input_slot,
                } => {
                    if *output_slot >= self.slot_capacity() || *input_slot >= self.slot_capacity() {
                        return Err(GpuSysError::InvalidInput(
                            "semantic softsign slot is out of bounds",
                        ));
                    }
                    let operand_offset = u32::try_from(operand_slots.len())
                        .map_err(|_| GpuSysError::SizeOverflow)?;
                    operand_slots.push(*input_slot);
                    raw_nodes.push(GafimeSemanticProgramNode {
                        opcode: GAFIME_SEMANTIC_PROGRAM_SOFTSIGN,
                        output_slot: *output_slot,
                        operand_offset,
                        operand_count: 1,
                        mean_offset: u32::try_from(mean_bits.len())
                            .map_err(|_| GpuSysError::SizeOverflow)?,
                        mean_count: 0,
                        ..Default::default()
                    });
                }
                SemanticProgramNode::CenteredProduct {
                    output_slot,
                    operand_slots: operands,
                    mean_bits: means,
                } => {
                    if *output_slot >= self.slot_capacity()
                        || operands.is_empty()
                        || operands.iter().any(|slot| *slot >= self.slot_capacity())
                        || operands.len() != means.len()
                    {
                        return Err(GpuSysError::InvalidInput(
                            "semantic centered-product slots or frozen means are invalid",
                        ));
                    }
                    let operand_offset = u32::try_from(operand_slots.len())
                        .map_err(|_| GpuSysError::SizeOverflow)?;
                    let mean_offset =
                        u32::try_from(mean_bits.len()).map_err(|_| GpuSysError::SizeOverflow)?;
                    let operand_count =
                        u32::try_from(operands.len()).map_err(|_| GpuSysError::SizeOverflow)?;
                    operand_slots.extend_from_slice(operands);
                    mean_bits.extend_from_slice(means);
                    raw_nodes.push(GafimeSemanticProgramNode {
                        opcode: GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT,
                        output_slot: *output_slot,
                        operand_offset,
                        operand_count,
                        mean_offset,
                        mean_count: operand_count,
                        ..Default::default()
                    });
                }
                SemanticProgramNode::FrozenRegionConjunction { output_slot, terms } => {
                    if *output_slot >= self.slot_capacity()
                        || terms.is_empty()
                        || terms.len()
                            > usize::try_from(self.max_region_terms())
                                .map_err(|_| GpuSysError::SizeOverflow)?
                    {
                        return Err(GpuSysError::InvalidInput(
                            "semantic frozen region has an invalid output or term count",
                        ));
                    }
                    let region_term_offset =
                        u32::try_from(region_terms.len()).map_err(|_| GpuSysError::SizeOverflow)?;
                    for term in terms {
                        let finite = match self.profile() {
                            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                                term.threshold_bits >> 32 == 0
                                    && f32::from_bits(term.threshold_bits as u32).is_finite()
                            }
                            PrecisionProfile::Fp64 => {
                                f64::from_bits(term.threshold_bits).is_finite()
                            }
                        };
                        if term.input_slot >= self.slot_capacity() || !finite {
                            return Err(GpuSysError::InvalidInput(
                                "semantic frozen region term is out of bounds or nonfinite",
                            ));
                        }
                        region_terms.push(GafimeSemanticFrozenRegionTerm {
                            input_slot: term.input_slot,
                            relation: term.relation.raw(),
                            threshold_bits: term.threshold_bits,
                        });
                    }
                    raw_nodes.push(GafimeSemanticProgramNode {
                        opcode: GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION,
                        output_slot: *output_slot,
                        operand_offset: u32::try_from(operand_slots.len())
                            .map_err(|_| GpuSysError::SizeOverflow)?,
                        operand_count: 0,
                        mean_offset: u32::try_from(mean_bits.len())
                            .map_err(|_| GpuSysError::SizeOverflow)?,
                        mean_count: 0,
                        region_term_offset,
                        region_term_count: u32::try_from(terms.len())
                            .map_err(|_| GpuSysError::SizeOverflow)?,
                        ..Default::default()
                    });
                }
            }
        }
        debug_assert_eq!(raw_nodes.len(), node_count as usize);
        let batch = GafimeSemanticProgramBatch {
            route: self.profile().numeric_route(),
            nodes: raw_nodes.as_ptr(),
            node_count,
            operand_slots: GafimeSliceU32 {
                ptr: operand_slots.as_ptr(),
                len: u64::try_from(operand_slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            mean_bits: GafimeSliceU64 {
                ptr: mean_bits.as_ptr(),
                len: u64::try_from(mean_bits.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            region_terms: GafimeSemanticRegionTermSlice {
                ptr: region_terms.as_ptr(),
                len: u64::try_from(region_terms.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            ..Default::default()
        };
        let _guard = self.lock();
        // SAFETY: all descriptor vectors remain live for this synchronous call,
        // slot bounds were checked locally and native validation repeats the
        // complete topological/route validation before dispatch.
        let status = unsafe { materialize(self.inner.raw, &batch) };
        status_to_gpu_result("gafime_gpu_semantic_materialize_v1", status)
    }

    fn scalar_results(
        &self,
        count: usize,
        call: impl FnOnce(&mut GafimeSemanticScalarResultTable) -> Result<(), GpuSysError>,
    ) -> Result<Vec<SemanticScalarResult>, GpuSysError> {
        let mut states = vec![0u32; count];
        let mut supports = vec![0u64; count];
        match self.profile() {
            PrecisionProfile::Fp32 => {
                let mut values = vec![0.0f32; count];
                let view = Self::mutable_view(&mut values, GAFIME_DTYPE_F32)?;
                let mut table = GafimeSemanticScalarResultTable {
                    route: self.profile().numeric_route(),
                    capacity: u64::try_from(count).map_err(|_| GpuSysError::SizeOverflow)?,
                    values: view,
                    states: states.as_mut_ptr(),
                    supports: supports.as_mut_ptr(),
                    ..Default::default()
                };
                call(&mut table)?;
                if table.count != count as u64 {
                    return Err(GpuSysError::InvalidInput(
                        "semantic scalar operation returned an unexpected result count",
                    ));
                }
                Ok(values
                    .into_iter()
                    .zip(states)
                    .zip(supports)
                    .map(|((value, state), support)| SemanticScalarResult {
                        value: SemanticScalarValue::F32(value),
                        state,
                        support,
                    })
                    .collect())
            }
            PrecisionProfile::Mixed | PrecisionProfile::Fp64 => {
                let mut values = vec![0.0f64; count];
                let view = Self::mutable_view(&mut values, GAFIME_DTYPE_F64)?;
                let mut table = GafimeSemanticScalarResultTable {
                    route: self.profile().numeric_route(),
                    capacity: u64::try_from(count).map_err(|_| GpuSysError::SizeOverflow)?,
                    values: view,
                    states: states.as_mut_ptr(),
                    supports: supports.as_mut_ptr(),
                    ..Default::default()
                };
                call(&mut table)?;
                if table.count != count as u64 {
                    return Err(GpuSysError::InvalidInput(
                        "semantic scalar operation returned an unexpected result count",
                    ));
                }
                Ok(values
                    .into_iter()
                    .zip(states)
                    .zip(supports)
                    .map(|((value, state), support)| SemanticScalarResult {
                        value: SemanticScalarValue::F64(value),
                        state,
                        support,
                    })
                    .collect())
            }
        }
    }

    pub fn pairwise_pearson(
        &self,
        right: &Self,
        left_slots: &[u32],
        right_slots: &[u32],
        mode: SemanticPearsonMode,
    ) -> Result<Vec<SemanticScalarResult>, GpuSysError> {
        if left_slots.len() != right_slots.len() {
            return Err(GpuSysError::InvalidInput(
                "semantic Pearson slot arrays have different lengths",
            ));
        }
        if left_slots.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_slot_slice(left_slots)?;
        right.validate_slot_slice(right_slots)?;
        let pairwise = self.inner.functions.semantic_pairwise_pearson_v1.ok_or(
            GpuSysError::MissingFunction("gafime_gpu_semantic_pairwise_pearson_v1"),
        )?;
        let batch = GafimeSemanticPearsonBatch {
            mode: mode.raw(),
            left_slots: GafimeSliceU32 {
                ptr: left_slots.as_ptr(),
                len: u64::try_from(left_slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            right_slots: GafimeSliceU32 {
                ptr: right_slots.as_ptr(),
                len: u64::try_from(right_slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            ..Default::default()
        };
        self.with_peer_lock(right, || {
            self.scalar_results(left_slots.len(), |results| {
                // SAFETY: both live banks were peer-validated and uniquely
                // locked, while `batch` and caller-owned result buffers remain
                // live for the synchronous payload invocation.
                let status = unsafe { pairwise(self.inner.raw, right.inner.raw, &batch, results) };
                status_to_gpu_result("gafime_gpu_semantic_pairwise_pearson_v1", status)
            })
        })
    }

    /// Evaluate one generic association per corresponding physical-slot pair.
    /// The caller has already selected the contextual operands; this ABI call
    /// neither infers a target nor receives evidence/policy identity.
    pub fn pairwise_association(
        &self,
        right: &Self,
        left_slots: &[u32],
        right_slots: &[u32],
        statistic: NativeAssociationStatistic,
        presentation: NativeAssociationPresentation,
    ) -> Result<Vec<SemanticScalarResult>, GpuSysError> {
        if left_slots.len() != right_slots.len() {
            return Err(GpuSysError::InvalidInput(
                "semantic association slot arrays have different lengths",
            ));
        }
        if left_slots.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_slot_slice(left_slots)?;
        right.validate_slot_slice(right_slots)?;
        let pairwise = self
            .inner
            .functions
            .semantic_pairwise_association_v1
            .ok_or(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_pairwise_association_v1",
            ))?;
        let batch = GafimeSemanticAssociationBatch {
            statistic: statistic.raw(),
            presentation: presentation.raw(),
            fixed_nmi_bins: statistic.fixed_nmi_bins(),
            left_slots: GafimeSliceU32 {
                ptr: left_slots.as_ptr(),
                len: u64::try_from(left_slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            right_slots: GafimeSliceU32 {
                ptr: right_slots.as_ptr(),
                len: u64::try_from(right_slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            ..Default::default()
        };
        self.with_peer_lock(right, || {
            self.scalar_results(left_slots.len(), |results| {
                // SAFETY: both live banks were peer-validated and uniquely
                // locked. The fully initialized generic descriptor and owned
                // scalar buffers remain live for this synchronous call.
                let status = unsafe { pairwise(self.inner.raw, right.inner.raw, &batch, results) };
                status_to_gpu_result("gafime_gpu_semantic_pairwise_association_v1", status)
            })
        })
    }

    /// Fit one profile-lane mean for each requested resident physical slot.
    /// Native code returns raw typed scalars; freezing identity bits remains a
    /// Rust semantic-owner step in `GpuNativeEvidenceExecutor::fit_means`.
    pub fn column_means(
        &self,
        candidate_slots: &[u32],
    ) -> Result<Vec<SemanticScalarResult>, GpuSysError> {
        if candidate_slots.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_slot_slice(candidate_slots)?;
        let means =
            self.inner
                .functions
                .semantic_column_means_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_column_means_v1",
                ))?;
        let batch = GafimeSemanticColumnMeanBatch {
            candidate_slots: GafimeSliceU32 {
                ptr: candidate_slots.as_ptr(),
                len: u64::try_from(candidate_slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            ..Default::default()
        };
        let _guard = self.lock();
        self.scalar_results(candidate_slots.len(), |results| {
            // SAFETY: this bank is exclusively locked and descriptor/result
            // buffers are caller-owned, typed and live for the synchronous
            // native fitting operation.
            let status = unsafe { means(self.inner.raw, &batch, results) };
            status_to_gpu_result("gafime_gpu_semantic_column_means_v1", status)
        })
    }

    pub fn ordered_edge_energy_f32(
        &self,
        candidate_slots: &[u32],
        edges: &[SemanticEdge],
        weights: &[f32],
    ) -> Result<Vec<SemanticScalarResult>, GpuSysError> {
        if !matches!(
            self.profile(),
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed
        ) {
            return Err(GpuSysError::InvalidInput(
                "f32 graph weights do not match fp64 semantic bank",
            ));
        }
        self.ordered_edge_energy(
            candidate_slots,
            edges,
            Self::const_view(weights, GAFIME_DTYPE_F32)?,
        )
    }

    pub fn ordered_edge_energy_f64(
        &self,
        candidate_slots: &[u32],
        edges: &[SemanticEdge],
        weights: &[f64],
    ) -> Result<Vec<SemanticScalarResult>, GpuSysError> {
        if self.profile() != PrecisionProfile::Fp64 {
            return Err(GpuSysError::InvalidInput(
                "f64 graph weights require an fp64 semantic bank",
            ));
        }
        self.ordered_edge_energy(
            candidate_slots,
            edges,
            Self::const_view(weights, GAFIME_DTYPE_F64)?,
        )
    }

    fn ordered_edge_energy(
        &self,
        candidate_slots: &[u32],
        edges: &[SemanticEdge],
        weights: GafimeConstBufferView,
    ) -> Result<Vec<SemanticScalarResult>, GpuSysError> {
        if candidate_slots.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_slot_slice(candidate_slots)?;
        if edges.is_empty()
            || edges.len()
                != usize::try_from(weights.element_count).map_err(|_| GpuSysError::SizeOverflow)?
            || edges
                .iter()
                .any(|edge| edge.left_row >= self.rows() || edge.right_row >= self.rows())
        {
            return Err(GpuSysError::InvalidInput(
                "semantic graph topology, weights or row endpoints are invalid",
            ));
        }
        // The executor accounts this separately from its `SemanticEdge`
        // staging.  Pre-size it exactly so the declared host peak matches the
        // allocation used for this synchronous ABI descriptor.
        let mut native_edges = Vec::with_capacity(edges.len());
        native_edges.extend(edges.iter().map(|edge| GafimeSemanticEdge {
            left_row: edge.left_row,
            right_row: edge.right_row,
        }));
        let ordered_edge_energy = self.inner.functions.semantic_ordered_edge_energy_v1.ok_or(
            GpuSysError::MissingFunction("gafime_gpu_semantic_ordered_edge_energy_v1"),
        )?;
        let batch = GafimeSemanticEdgeEnergyBatch {
            edges: native_edges.as_ptr(),
            edge_count: u64::try_from(native_edges.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            weights,
            candidate_slots: GafimeSliceU32 {
                ptr: candidate_slots.as_ptr(),
                len: u64::try_from(candidate_slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            ..Default::default()
        };
        let _guard = self.lock();
        self.scalar_results(candidate_slots.len(), |results| {
            // SAFETY: the bank is exclusively locked, graph descriptors are
            // locally bounded/live, and result buffers remain live during this
            // synchronous call.
            let status = unsafe { ordered_edge_energy(self.inner.raw, &batch, results) };
            status_to_gpu_result("gafime_gpu_semantic_ordered_edge_energy_v1", status)
        })
    }

    pub fn sparse_gather_from(
        &self,
        source: &Self,
        source_slots: &[u32],
        destination_slots: &[u32],
        rows: &[u64],
    ) -> Result<(), GpuSysError> {
        if self.same_bank(source) {
            return Err(GpuSysError::InvalidInput(
                "semantic sparse gather requires distinct source and destination banks",
            ));
        }
        if source_slots.is_empty() && destination_slots.is_empty() {
            return Ok(());
        }
        if source_slots.len() != destination_slots.len() || rows.is_empty() {
            return Err(GpuSysError::InvalidInput(
                "semantic sparse gather requires equal nonempty slot arrays and rows",
            ));
        }
        source.validate_slot_slice(source_slots)?;
        self.validate_slot_slice(destination_slots)?;
        let row_count = u64::try_from(rows.len()).map_err(|_| GpuSysError::SizeOverflow)?;
        if rows.iter().any(|row| *row >= source.rows())
            || row_count != self.rows()
            || row_count > self.max_gather_rows()
        {
            return Err(GpuSysError::InvalidInput(
                "semantic sparse gather rows do not match source or destination bank",
            ));
        }
        let gather =
            self.inner
                .functions
                .semantic_sparse_gather_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_sparse_gather_v1",
                ))?;
        let batch = GafimeSemanticSparseGatherBatch {
            source_slots: GafimeSliceU32 {
                ptr: source_slots.as_ptr(),
                len: u64::try_from(source_slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            destination_slots: GafimeSliceU32 {
                ptr: destination_slots.as_ptr(),
                len: u64::try_from(destination_slots.len())
                    .map_err(|_| GpuSysError::SizeOverflow)?,
            },
            row_indices: GafimeSliceU64 {
                ptr: rows.as_ptr(),
                len: u64::try_from(rows.len()).map_err(|_| GpuSysError::SizeOverflow)?,
            },
            ..Default::default()
        };
        self.with_peer_device_lock(source, || {
            // SAFETY: peer banks were identity-checked and uniquely locked;
            // all source/destination/row descriptor slices are live for this
            // synchronous native gather.
            let status = unsafe { gather(source.inner.raw, self.inner.raw, &batch) };
            status_to_gpu_result("gafime_gpu_semantic_sparse_gather_v1", status)
        })
    }

    pub fn forecast(
        &self,
        request: GafimeSemanticForecastRequest,
    ) -> Result<SemanticMemoryForecast, GpuSysError> {
        let forecast =
            self.inner
                .functions
                .semantic_forecast_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_forecast_v1",
                ))?;
        let mut output = GafimeSemanticMemoryForecast::default();
        let _guard = self.lock();
        // SAFETY: the live bank is exclusively locked, request is copied and
        // fully initialized by the safe caller, and output is writable local
        // storage for this synchronous query.
        let status = unsafe { forecast(self.inner.raw, &request, &mut output) };
        status_to_gpu_result("gafime_gpu_semantic_forecast_v1", status)?;
        if output.abi_version != GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION
            || output.struct_size != std::mem::size_of::<GafimeSemanticMemoryForecast>() as u32
            || output.reserved.iter().any(|value| *value != 0)
            || output.resident_bytes != self.bytes()
        {
            return Err(GpuSysError::InvalidInput(
                "semantic forecast returned malformed metadata or an inconsistent resident size",
            ));
        }
        Ok(SemanticMemoryForecast {
            resident_bytes: output.resident_bytes,
            transient_bytes: output.transient_bytes,
            retained_bytes: output.retained_bytes,
        })
    }

    pub fn retain(&self, slots: &[u32]) -> Result<Self, GpuSysError> {
        self.validate_slot_slice(slots)?;
        let slot_count = u32::try_from(slots.len()).map_err(|_| GpuSysError::SizeOverflow)?;
        let bytes = self
            .rows()
            .checked_mul(u64::from(slot_count))
            .and_then(|n| n.checked_mul(storage_width(self.profile())))
            .ok_or(GpuSysError::SizeOverflow)?;
        let retain =
            self.inner
                .functions
                .semantic_bank_retain_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_bank_retain_v1",
                ))?;
        let mut raw = std::ptr::null_mut();
        let slot_slice = GafimeSliceU32 {
            ptr: slots.as_ptr(),
            len: u64::try_from(slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
        };
        let _guard = self.lock();
        // SAFETY: this live source bank is exclusively locked, slots are
        // bounds-checked and live, and `raw` is writable local output storage.
        let status = unsafe { retain(self.inner.raw, slot_slice, &mut raw) };
        // A failed native retain normally leaves `raw` null.  A cleanup
        // failure after native allocation is the exception: its non-null
        // output is free-only ownership.  Adopt it before checking `status`
        // so the local RAII owner makes the documented best-effort release
        // attempt rather than leaking an unreachable allocation.
        let retained = if raw.is_null() {
            None
        } else {
            Some(Self::from_parts(
                raw,
                self.inner.functions,
                self.inner._library.clone(),
                self.inner.owner.clone(),
                self.inner.backend_kind,
                self.inner.device_id,
                SemanticBankMetadata {
                    profile: self.profile(),
                    rows: self.rows(),
                    source_slots: slot_count,
                    slot_capacity: slot_count,
                    max_program_nodes: self.max_program_nodes(),
                    max_gather_rows: self.max_gather_rows(),
                    max_region_terms: self.max_region_terms(),
                    bytes,
                },
            )?)
        };
        status_to_gpu_result("gafime_gpu_semantic_bank_retain_v1", status)?;
        retained.ok_or(GpuSysError::InvalidInput(
            "semantic retain succeeded with a null output bank",
        ))
    }

    pub fn download_f32(&self, slots: &[u32]) -> Result<Vec<f32>, GpuSysError> {
        if !matches!(
            self.profile(),
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed
        ) {
            return Err(GpuSysError::InvalidInput(
                "f32 semantic download does not match fp64 bank",
            ));
        }
        self.download_slots_f32(slots)
    }

    pub fn download_f64(&self, slots: &[u32]) -> Result<Vec<f64>, GpuSysError> {
        if self.profile() != PrecisionProfile::Fp64 {
            return Err(GpuSysError::InvalidInput(
                "f64 semantic download requires an fp64 bank",
            ));
        }
        self.download_slots_f64(slots)
    }

    fn download_slots_f32(&self, slots: &[u32]) -> Result<Vec<f32>, GpuSysError> {
        self.validate_slot_slice(slots)?;
        let elements = checked_download_elements(self.rows(), slots.len())?;
        let mut columns = vec![0.0f32; elements];
        let mut view = Self::mutable_view(&mut columns, GAFIME_DTYPE_F32)?;
        self.download_into(slots, &mut view)?;
        Ok(columns)
    }

    fn download_slots_f64(&self, slots: &[u32]) -> Result<Vec<f64>, GpuSysError> {
        self.validate_slot_slice(slots)?;
        let elements = checked_download_elements(self.rows(), slots.len())?;
        let mut columns = vec![0.0f64; elements];
        let mut view = Self::mutable_view(&mut columns, GAFIME_DTYPE_F64)?;
        self.download_into(slots, &mut view)?;
        Ok(columns)
    }

    fn download_into(
        &self,
        slots: &[u32],
        columns: &mut GafimeMutableBufferView,
    ) -> Result<(), GpuSysError> {
        let download =
            self.inner
                .functions
                .semantic_bank_download_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_bank_download_v1",
                ))?;
        let slot_slice = GafimeSliceU32 {
            ptr: slots.as_ptr(),
            len: u64::try_from(slots.len()).map_err(|_| GpuSysError::SizeOverflow)?,
        };
        let route = self.profile().numeric_route();
        let _guard = self.lock();
        // SAFETY: the bank is exclusively locked, slots/view are validated
        // caller-owned buffers and live for this synchronous native transfer.
        let status = unsafe { download(self.inner.raw, slot_slice, &route, columns) };
        status_to_gpu_result("gafime_gpu_semantic_bank_download_v1", status)
    }
}

impl GpuBackend {
    /// Allocate a typed, column-major semantic bank after validating that the
    /// loaded optional table supports this exact profile and physical shape.
    pub fn allocate_semantic_bank(
        &self,
        profile: PrecisionProfile,
        rows: usize,
        source_slots: u32,
        slot_capacity: u32,
    ) -> Result<OwnedSemanticBank, GpuSysError> {
        let capabilities = self.semantic_capabilities()?;
        if capabilities.profile_mask & profile.capability_mask() == 0
            || rows == 0
            || u64::try_from(rows).map_err(|_| GpuSysError::SizeOverflow)? > capabilities.max_rows
            || slot_capacity == 0
            || slot_capacity > capabilities.max_slot_count
            || source_slots > slot_capacity
        {
            return Err(GpuSysError::InvalidInput(
                "semantic bank profile, rows or slot capacity exceeds payload capability",
            ));
        }
        let rows = u64::try_from(rows).map_err(|_| GpuSysError::SizeOverflow)?;
        let bytes = rows
            .checked_mul(u64::from(slot_capacity))
            .and_then(|count| count.checked_mul(storage_width(profile)))
            .ok_or(GpuSysError::SizeOverflow)?;
        usize::try_from(bytes).map_err(|_| GpuSysError::SizeOverflow)?;
        let allocate =
            self.functions
                .semantic_bank_alloc_v1
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_bank_alloc_v1",
                ))?;
        let desc = GafimeSemanticBankDesc {
            route: profile.numeric_route(),
            layout: GAFIME_MATRIX_COLUMN_MAJOR,
            rows,
            source_slots,
            slot_capacity,
            bytes,
            ..Default::default()
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: the table was capability-validated, descriptor byte count is
        // checked, and `raw` is writable local storage for this synchronous
        // allocation callback.
        let status = unsafe { allocate(self.device_id, &desc, &mut raw) };
        status_to_gpu_result("gafime_gpu_semantic_bank_alloc_v1", status)?;
        OwnedSemanticBank::from_backend(
            self,
            raw,
            SemanticBankMetadata {
                profile,
                rows,
                source_slots,
                slot_capacity,
                max_program_nodes: capabilities.max_program_nodes,
                max_gather_rows: capabilities.max_gather_rows,
                max_region_terms: capabilities.max_region_terms,
                bytes,
            },
        )
    }

    /// Construct the safe semantic executor after capability negotiation.
    /// Explicit backend selection will still fail closed later when an actual
    /// requested operation (for example Spearman) is absent from this record.
    pub fn semantic_executor(&self) -> Result<GpuNativeEvidenceExecutor, GpuSysError> {
        GpuNativeEvidenceExecutor::new(self.clone())
    }
}

const fn storage_width(profile: PrecisionProfile) -> u64 {
    match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => std::mem::size_of::<f32>() as u64,
        PrecisionProfile::Fp64 => std::mem::size_of::<f64>() as u64,
    }
}

fn checked_download_elements(rows: u64, slots: usize) -> Result<usize, GpuSysError> {
    let count = rows
        .checked_mul(u64::try_from(slots).map_err(|_| GpuSysError::SizeOverflow)?)
        .ok_or(GpuSysError::SizeOverflow)?;
    usize::try_from(count).map_err(|_| GpuSysError::SizeOverflow)
}

/// The executor implementation follows below this bank boundary.
pub struct GpuNativeEvidenceExecutor {
    backend: GpuBackend,
    capabilities: gafime_types::GafimeSemanticCapabilities,
}

impl GpuNativeEvidenceExecutor {
    /// Conservative scalar-work admission for Metal's bounded bitonic-sort
    /// Spearman lowering. `M` is the padded power-of-two row count and `L`
    /// its binary logarithm. The sort term deliberately overcharges the
    /// two-vector compare-exchange work, while the rank phase includes four
    /// bounded binary searches per original row. CUDA and HIP do not use this
    /// model: their current average-tie rank primitive remains quadratic.
    fn metal_spearman_work(pair_count: u64, rows: u64) -> SemanticResult<u64> {
        let padded_rows = rows
            .checked_next_power_of_two()
            .ok_or(SemanticError::Invalid(
                "Metal Spearman padded row count overflows u64",
            ))?;
        let log_rows = u64::from(padded_rows.ilog2());
        let per_pair = padded_rows
            .checked_mul(2)
            .and_then(|value| {
                padded_rows
                    .checked_mul(log_rows)
                    .and_then(|sort| sort.checked_mul(log_rows.checked_add(1)?))
                    .and_then(|sort| value.checked_add(sort))
            })
            .and_then(|value| {
                rows.checked_mul(4)
                    .and_then(|rank| rank.checked_mul(log_rows))
                    .and_then(|rank| value.checked_add(rank))
            })
            .and_then(|value| {
                rows.checked_mul(12)
                    .and_then(|finalize| value.checked_add(finalize))
            })
            .ok_or(SemanticError::Invalid(
                "Metal Spearman work bound overflows u64",
            ))?;
        pair_count
            .checked_mul(per_pair)
            .ok_or(SemanticError::Invalid(
                "Metal Spearman work bound overflows u64",
            ))
    }

    pub fn new(backend: GpuBackend) -> Result<Self, GpuSysError> {
        let capabilities = backend.semantic_capabilities()?;
        Ok(Self {
            backend,
            capabilities,
        })
    }

    pub const fn backend_kind(&self) -> BackendKind {
        self.backend.kind
    }

    pub const fn device_id(&self) -> u32 {
        self.backend.device_id
    }

    pub fn loaded_library_path(&self) -> Option<&Path> {
        self.backend.loaded_library_path()
    }

    pub const fn capabilities(&self) -> gafime_types::GafimeSemanticCapabilities {
        self.capabilities
    }

    fn semantic_error(error: GpuSysError) -> SemanticError {
        match error {
            GpuSysError::SemanticAbiUnavailable | GpuSysError::MissingFunction(_) => {
                SemanticError::Unsupported(
                    "selected GPU payload lacks a required semantic primitive",
                )
            }
            GpuSysError::BackendStatus { .. } => SemanticError::Invalid(
                "selected GPU payload rejected a validated semantic arithmetic request",
            ),
            GpuSysError::EnvMissing(_)
            | GpuSysError::LoadLibrary { .. }
            | GpuSysError::LoadSymbol { .. }
            | GpuSysError::PrecisionAbiUnavailable
            | GpuSysError::InvalidInput(_)
            | GpuSysError::AbiVersionMismatch { .. }
            | GpuSysError::BackendKindMismatch { .. }
            | GpuSysError::DeviceIdMismatch { .. }
            | GpuSysError::SizeOverflow => SemanticError::Invalid(
                "GPU semantic adapter rejected an invalid context, capacity or ABI identity",
            ),
        }
    }

    fn require_profile(&self, profile: PrecisionProfile) -> SemanticResult<()> {
        if self.capabilities.profile_mask & profile.capability_mask() == 0 {
            return Err(SemanticError::Unsupported(
                "selected GPU semantic payload does not support this precision profile",
            ));
        }
        Ok(())
    }

    fn require_program_op(&self, mask: u32) -> SemanticResult<()> {
        if self.capabilities.program_op_mask & mask == 0 {
            return Err(SemanticError::Unsupported(
                "selected GPU semantic payload does not support this program operation",
            ));
        }
        Ok(())
    }

    fn require_association(&self, statistic: AssociationStatistic) -> SemanticResult<()> {
        if self.capabilities.primitive_mask & GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_ASSOCIATION
            == 0
        {
            return Err(SemanticError::Unsupported(
                "selected GPU semantic payload does not support generic association arithmetic",
            ));
        }
        let supported = match statistic {
            AssociationStatistic::Pearson => {
                self.capabilities.association_statistic_mask
                    & GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON
                    != 0
            }
            AssociationStatistic::Spearman => {
                self.capabilities.association_statistic_mask
                    & GAFIME_SEMANTIC_STATISTIC_MASK_SPEARMAN
                    != 0
            }
            AssociationStatistic::FixedCorrectedNmi { bins } => {
                self.capabilities.association_statistic_mask
                    & GAFIME_SEMANTIC_STATISTIC_MASK_FIXED_CORRECTED_NMI
                    != 0
                    && Self::fixed_nmi_bin_capability(bins).is_some_and(|mask| {
                        self.capabilities.fixed_corrected_nmi_bin_mask & mask != 0
                    })
            }
        };
        if !supported {
            return Err(SemanticError::Unsupported(
                "selected GPU semantic payload does not support this association statistic",
            ));
        }
        Ok(())
    }

    fn require_column_means(&self) -> SemanticResult<()> {
        if self.capabilities.primitive_mask & GAFIME_SEMANTIC_PRIMITIVE_MASK_COLUMN_MEANS == 0 {
            return Err(SemanticError::Unsupported(
                "selected GPU semantic payload does not support ordered column means",
            ));
        }
        Ok(())
    }

    fn require_region_terms(&self, count: usize) -> SemanticResult<()> {
        if u64::try_from(count)
            .map_err(|_| SemanticError::Invalid("frozen-region term count overflows u64"))?
            > u64::from(self.capabilities.max_region_terms)
        {
            return Err(SemanticError::Unsupported(
                "frozen-region term count exceeds selected GPU semantic capability",
            ));
        }
        Ok(())
    }

    const fn fixed_nmi_bin_capability(bins: u32) -> Option<u32> {
        match bins {
            2 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_2),
            4 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_4),
            8 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_8),
            12 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_12),
            16 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_16),
            24 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_24),
            32 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_32),
            48 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_48),
            64 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_64),
            96 => Some(gafime_types::GAFIME_SEMANTIC_FIXED_CORRECTED_NMI_BIN_96),
            _ => None,
        }
    }

    const fn native_association_statistic(
        statistic: AssociationStatistic,
    ) -> NativeAssociationStatistic {
        match statistic {
            AssociationStatistic::Pearson => NativeAssociationStatistic::Pearson,
            AssociationStatistic::Spearman => NativeAssociationStatistic::Spearman,
            AssociationStatistic::FixedCorrectedNmi { bins } => {
                NativeAssociationStatistic::FixedCorrectedNmi { bins }
            }
        }
    }

    const fn native_association_presentation(
        statistic: AssociationStatistic,
        absolute: bool,
    ) -> NativeAssociationPresentation {
        match statistic {
            AssociationStatistic::FixedCorrectedNmi { .. } => {
                NativeAssociationPresentation::Nonnegative
            }
            AssociationStatistic::Pearson | AssociationStatistic::Spearman if absolute => {
                NativeAssociationPresentation::Absolute
            }
            AssociationStatistic::Pearson | AssociationStatistic::Spearman => {
                NativeAssociationPresentation::Signed
            }
        }
    }

    fn require_gather(&self) -> SemanticResult<()> {
        if self.capabilities.primitive_mask & GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER == 0 {
            return Err(SemanticError::Unsupported(
                "selected GPU semantic payload does not support sparse row gathering",
            ));
        }
        Ok(())
    }

    fn require_graph_energy(&self) -> SemanticResult<()> {
        if self.capabilities.primitive_mask & GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY
            == 0
        {
            return Err(SemanticError::Unsupported(
                "selected GPU semantic payload does not support ordered graph energy",
            ));
        }
        Ok(())
    }

    fn validate_context(
        &self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
    ) -> SemanticResult<()> {
        if registry.schema() != frame.schema() || registry.precision() != frame.profile() {
            return Err(SemanticError::Unsupported(
                "selected backend cannot lower this semantic context",
            ));
        }
        self.require_profile(frame.profile())
    }

    fn resident_bank(
        &self,
        values: &MaterializedColumns,
        frame: &FeatureFrame,
    ) -> SemanticResult<OwnedSemanticBank> {
        if values.frame_id() != frame.id()
            || values.profile() != frame.profile()
            || values.backend_kind() != self.backend.kind
            || !values.is_resident()
        {
            return Err(SemanticError::Invalid(
                "resident materialization context, profile or backend mismatch",
            ));
        }
        let lease = Arc::clone(values.resident_lease()?);
        let bank = lease
            .downcast::<OwnedSemanticBank>()
            .map_err(|_| SemanticError::Invalid("resident materialization lease is foreign"))?;
        let bank = (*bank).clone();
        if !bank.same_backend_owner(&self.backend)
            || bank.profile() != frame.profile()
            || bank.rows() != frame.rows() as u64
        {
            return Err(SemanticError::Invalid(
                "resident semantic bank identity does not match its executor",
            ));
        }
        Ok(bank)
    }

    fn slots_for(values: &MaterializedColumns, features: &[FeatureId]) -> SemanticResult<Vec<u32>> {
        let slots = values.resident_slots()?;
        features
            .iter()
            .map(|feature| {
                slots.get(feature).copied().ok_or(SemanticError::Invalid(
                    "semantic feature is absent from resident bank slot map",
                ))
            })
            .collect()
    }

    fn lower_region_term(
        profile: PrecisionProfile,
        slots: &BTreeMap<FeatureId, u32>,
        input: FeatureId,
        comparison: PredicateComparator,
        threshold_bits: &FrozenThreshold,
    ) -> SemanticResult<SemanticFrozenRegionTerm> {
        let threshold_bits = match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                u64::from(threshold_bits.as_f32_bits()?)
            }
            PrecisionProfile::Fp64 => threshold_bits.as_f64_bits()?,
        };
        Ok(SemanticFrozenRegionTerm {
            input_slot: slots.get(&input).copied().ok_or(SemanticError::Invalid(
                "missing frozen-region semantic input slot",
            ))?,
            relation: match comparison {
                PredicateComparator::LessEqual => SemanticRegionRelation::LessEqual,
                PredicateComparator::GreaterThan => SemanticRegionRelation::GreaterThan,
            },
            threshold_bits,
        })
    }

    fn dependency_ids(
        registry: &CandidateRegistry,
        roots: &[FeatureId],
        retained_slots: Option<&BTreeMap<FeatureId, u32>>,
    ) -> SemanticResult<(BTreeSet<FeatureId>, BTreeSet<FeatureId>)> {
        let mut needed = BTreeSet::new();
        let mut reused = BTreeSet::new();
        let mut pending = roots.to_vec();
        while let Some(id) = pending.pop() {
            if !needed.insert(id) {
                continue;
            }
            // A retained value is an accepted physical result for this exact
            // frame/profile/backend.  Stop traversal at that root, matching
            // the Core executor: gathering it into a fresh bank must not
            // rebuild or upload its dependencies.
            if retained_slots.is_some_and(|slots| slots.contains_key(&id)) {
                reused.insert(id);
                continue;
            }
            match registry.program(id)?.op() {
                FeatureOp::Source(_) => {}
                FeatureOp::AbsoluteDifference(left, right) => pending.extend([*left, *right]),
                FeatureOp::Softsign(input) => pending.push(*input),
                FeatureOp::CenteredProduct { operands, .. } => pending.extend(operands),
                FeatureOp::HardPredicate { input, .. } => pending.push(*input),
                FeatureOp::DecisionRegion { terms } => {
                    for term in terms {
                        match registry.program(*term)?.op() {
                            FeatureOp::HardPredicate { input, .. } => pending.push(*input),
                            _ => {
                                return Err(SemanticError::Invalid(
                                    "decision-region term is not a hard predicate",
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok((needed, reused))
    }

    fn storage_bytes(
        profile: PrecisionProfile,
        rows: usize,
        slots: usize,
    ) -> SemanticResult<usize> {
        rows.checked_mul(slots)
            .and_then(|elements| elements.checked_mul(storage_width(profile) as usize))
            .ok_or(SemanticError::Invalid(
                "semantic resident-bank byte count overflows host address space",
            ))
    }

    fn reserve_before_allocation(
        resident_bytes: usize,
        transient_bytes: usize,
        max_bytes: usize,
    ) -> SemanticResult<()> {
        if resident_bytes
            .checked_add(transient_bytes)
            .is_none_or(|bytes| bytes > max_bytes)
        {
            return Err(SemanticError::Invalid(
                "GPU semantic resident bank and temporary peak exceed execution budget",
            ));
        }
        Ok(())
    }

    fn maximum_operand_count(nodes: &[SemanticProgramNode]) -> SemanticResult<u64> {
        nodes
            .iter()
            .map(|node| match node {
                SemanticProgramNode::Source { .. } | SemanticProgramNode::Softsign { .. } => 1,
                SemanticProgramNode::AbsoluteDifference { .. } => 2,
                SemanticProgramNode::CenteredProduct { operand_slots, .. } => operand_slots.len(),
                SemanticProgramNode::FrozenRegionConjunction { .. } => 0,
            })
            .max()
            .map(|count| {
                u64::try_from(count).map_err(|_| {
                    SemanticError::Invalid("semantic program operand count overflows u64")
                })
            })
            .transpose()
            .map(|count| count.unwrap_or(0))
    }

    /// Exact flattened native descriptor spans.  The optional semantic table
    /// keeps these arrays immutable for an entire asynchronous program batch,
    /// so a maximum per-node arity would understate the device allocation.
    fn program_descriptor_counts(nodes: &[SemanticProgramNode]) -> SemanticResult<(u64, u64, u64)> {
        let (operands, means, region_terms) = nodes.iter().try_fold(
            (0usize, 0usize, 0usize),
            |(operands, means, region_terms), node| {
                let operand_count = match node {
                    SemanticProgramNode::Source { .. } | SemanticProgramNode::Softsign { .. } => 1,
                    SemanticProgramNode::AbsoluteDifference { .. } => 2,
                    SemanticProgramNode::CenteredProduct { operand_slots, .. } => operand_slots.len(),
                    SemanticProgramNode::FrozenRegionConjunction { .. } => 0,
                };
                let mean_count = match node {
                    SemanticProgramNode::CenteredProduct { mean_bits, .. } => mean_bits.len(),
                    _ => 0,
                };
                let region_term_count = match node {
                    SemanticProgramNode::FrozenRegionConjunction { terms, .. } => terms.len(),
                    _ => 0,
                };
                Ok::<_, SemanticError>((
                    operands.checked_add(operand_count).ok_or(SemanticError::Invalid(
                        "semantic flattened operand descriptor count overflows host address space",
                    ))?,
                    means.checked_add(mean_count).ok_or(SemanticError::Invalid(
                        "semantic flattened mean descriptor count overflows host address space",
                    ))?,
                    region_terms
                        .checked_add(region_term_count)
                        .ok_or(SemanticError::Invalid(
                            "semantic flattened region-term descriptor count overflows host address space",
                        ))?,
                ))
            },
        )?;
        Ok((
            u64::try_from(operands).map_err(|_| {
                SemanticError::Invalid("semantic flattened operand descriptor count overflows u64")
            })?,
            u64::try_from(means).map_err(|_| {
                SemanticError::Invalid("semantic flattened mean descriptor count overflows u64")
            })?,
            u64::try_from(region_terms).map_err(|_| {
                SemanticError::Invalid(
                    "semantic flattened region-term descriptor count overflows u64",
                )
            })?,
        ))
    }

    fn program_host_temporary_bytes(
        nodes: &[SemanticProgramNode],
        outer_node_capacity: usize,
    ) -> SemanticResult<usize> {
        let mut outer_nodes = outer_node_capacity
            .checked_mul(std::mem::size_of::<SemanticProgramNode>())
            .ok_or(SemanticError::Invalid(
                "semantic host program nodes exceed address space",
            ))?;
        // `OwnedSemanticBank::materialize` copies this typed lowering into one
        // ABI node array plus exactly pre-sized flattened operand/mean/region
        // arrays.  The descriptor vectors remain live together during the
        // synchronous call, so each one is part of the host peak.
        outer_nodes = outer_nodes
            .checked_add(
                nodes
                    .len()
                    .checked_mul(std::mem::size_of::<GafimeSemanticProgramNode>())
                    .ok_or(SemanticError::Invalid(
                        "semantic ABI node descriptors exceed address space",
                    ))?,
            )
            .ok_or(SemanticError::Invalid(
                "semantic host program descriptors exceed address space",
            ))?;
        nodes.iter().try_fold(outer_nodes, |total, node| {
            let (operand_capacity, mean_capacity, region_capacity) = match node {
                SemanticProgramNode::Source { .. }
                | SemanticProgramNode::Softsign { .. }
                | SemanticProgramNode::AbsoluteDifference { .. } => (0, 0, 0),
                SemanticProgramNode::CenteredProduct {
                    operand_slots,
                    mean_bits,
                    ..
                } => (operand_slots.capacity(), mean_bits.capacity(), 0),
                SemanticProgramNode::FrozenRegionConjunction { terms, .. } => {
                    (0, 0, terms.capacity())
                }
            };
            let outer_operands = operand_capacity
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or(SemanticError::Invalid(
                    "semantic host operands exceed address space",
                ))?;
            let outer_means = mean_capacity
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or(SemanticError::Invalid(
                    "semantic host means exceed address space",
                ))?;
            let outer_region_terms = region_capacity
                .checked_mul(std::mem::size_of::<SemanticFrozenRegionTerm>())
                .ok_or(SemanticError::Invalid(
                    "semantic host region terms exceed address space",
                ))?;
            let flattened_operands = match node {
                SemanticProgramNode::Source { .. } | SemanticProgramNode::Softsign { .. } => 1,
                SemanticProgramNode::AbsoluteDifference { .. } => 2,
                SemanticProgramNode::CenteredProduct { operand_slots, .. } => operand_slots.len(),
                SemanticProgramNode::FrozenRegionConjunction { .. } => 0,
            }
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(SemanticError::Invalid(
                "semantic ABI operands exceed address space",
            ))?;
            let flattened_means = match node {
                SemanticProgramNode::CenteredProduct { mean_bits, .. } => mean_bits.len(),
                _ => 0,
            }
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(SemanticError::Invalid(
                "semantic ABI means exceed address space",
            ))?;
            let flattened_region_terms = match node {
                SemanticProgramNode::FrozenRegionConjunction { terms, .. } => terms.len(),
                _ => 0,
            }
            .checked_mul(std::mem::size_of::<GafimeSemanticFrozenRegionTerm>())
            .ok_or(SemanticError::Invalid(
                "semantic ABI region terms exceed address space",
            ))?;
            total
                .checked_add(outer_operands)
                .and_then(|total| total.checked_add(outer_means))
                .and_then(|total| total.checked_add(outer_region_terms))
                .and_then(|total| total.checked_add(flattened_operands))
                .and_then(|total| total.checked_add(flattened_means))
                .and_then(|total| total.checked_add(flattened_region_terms))
                .ok_or(SemanticError::Invalid(
                    "semantic host program temporary exceeds address space",
                ))
        })
    }

    fn scalar_result_host_bytes(profile: PrecisionProfile, count: usize) -> SemanticResult<usize> {
        let raw = result_width(profile)
            .checked_add(std::mem::size_of::<u32>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .ok_or(SemanticError::Invalid(
                "semantic scalar result width exceeds address space",
            ))?;
        let converted = std::mem::size_of::<SemanticScalarResult>()
            .checked_add(std::mem::size_of::<EvidenceValue>())
            .ok_or(SemanticError::Invalid(
                "semantic evidence result width exceeds address space",
            ))?;
        count
            .checked_mul(raw.checked_add(converted).ok_or(SemanticError::Invalid(
                "semantic scalar result width exceeds address space",
            ))?)
            .ok_or(SemanticError::Invalid(
                "semantic scalar result storage exceeds address space",
            ))
    }

    /// Peak host storage for a native column-mean result and its exact frozen
    /// identity representation.  This is deliberately separate from
    /// `scalar_result_host_bytes`: fitting does not create `EvidenceValue`s.
    fn mean_result_host_bytes(profile: PrecisionProfile, count: usize) -> SemanticResult<usize> {
        let native = result_width(profile)
            .checked_add(std::mem::size_of::<u32>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
            .ok_or(SemanticError::Invalid(
                "semantic mean result width exceeds address space",
            ))?;
        let frozen = match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => std::mem::size_of::<u32>(),
            PrecisionProfile::Fp64 => std::mem::size_of::<u64>(),
        };
        let converted = std::mem::size_of::<SemanticScalarResult>()
            .checked_add(frozen)
            .ok_or(SemanticError::Invalid(
                "semantic mean conversion width exceeds address space",
            ))?;
        count
            .checked_mul(native.checked_add(converted).ok_or(SemanticError::Invalid(
                "semantic mean result width exceeds address space",
            ))?)
            .ok_or(SemanticError::Invalid(
                "semantic mean result storage exceeds address space",
            ))
    }

    fn u32_slice_bytes(count: usize, message: &'static str) -> SemanticResult<usize> {
        count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(SemanticError::Invalid(message))
    }

    fn u64_slice_bytes(count: usize, message: &'static str) -> SemanticResult<usize> {
        count
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(SemanticError::Invalid(message))
    }

    fn native_forecast(
        &self,
        bank: &OwnedSemanticBank,
        request: GafimeSemanticForecastRequest,
    ) -> SemanticResult<SemanticMemoryForecast> {
        bank.forecast(request).map_err(Self::semantic_error)
    }

    fn forecast_peak_bytes(
        forecast: SemanticMemoryForecast,
        include_resident: bool,
        host_temporary_bytes: usize,
    ) -> SemanticResult<usize> {
        let mut peak = host_temporary_bytes;
        if include_resident {
            peak = peak
                .checked_add(usize::try_from(forecast.resident_bytes).map_err(|_| {
                    SemanticError::Invalid("GPU forecast resident bytes exceed host address space")
                })?)
                .ok_or(SemanticError::Invalid(
                    "GPU forecast peak exceeds host address space",
                ))?;
        }
        for (bytes, description) in [
            (forecast.transient_bytes, "transient"),
            (forecast.retained_bytes, "retained"),
        ] {
            peak = peak
                .checked_add(usize::try_from(bytes).map_err(|_| {
                    SemanticError::Invalid(match description {
                        "transient" => "GPU forecast transient bytes exceed host address space",
                        _ => "GPU forecast retained bytes exceed host address space",
                    })
                })?)
                .ok_or(SemanticError::Invalid(
                    "GPU forecast peak exceeds host address space",
                ))?;
        }
        Ok(peak)
    }

    fn reserve_forecast(
        forecast: SemanticMemoryForecast,
        include_resident: bool,
        host_temporary_bytes: usize,
        max_bytes: usize,
    ) -> SemanticResult<()> {
        if Self::forecast_peak_bytes(forecast, include_resident, host_temporary_bytes)? > max_bytes
        {
            return Err(SemanticError::Invalid(
                "GPU semantic forecast peak exceeds execution budget",
            ));
        }
        Ok(())
    }

    fn materialized_columns(
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        backend: BackendKind,
        bank: OwnedSemanticBank,
        slots: BTreeMap<FeatureId, u32>,
    ) -> SemanticResult<MaterializedColumns> {
        let bytes = usize::try_from(bank.bytes()).map_err(|_| {
            SemanticError::Invalid("GPU semantic resident bank exceeds host address space")
        })?;
        let lease: gafime_orchestrator::semantic::ResidentMaterializationLease = Arc::new(bank);
        MaterializedColumns::from_resident(registry, frame, backend, slots, bytes, lease)
    }

    fn results_to_evidence(
        profile: PrecisionProfile,
        results: Vec<SemanticScalarResult>,
    ) -> SemanticResult<Vec<EvidenceValue>> {
        results
            .into_iter()
            .map(|result| {
                let support = usize::try_from(result.support).map_err(|_| {
                    SemanticError::Invalid(
                        "GPU semantic evidence support exceeds host address space",
                    )
                })?;
                match result.state {
                    GAFIME_SEMANTIC_SCALAR_MEASURED => match result.value {
                        SemanticScalarValue::F32(value) if profile == PrecisionProfile::Fp32 => {
                            Ok(EvidenceValue::measured_f32(value, support))
                        }
                        SemanticScalarValue::F64(value)
                            if matches!(
                                profile,
                                PrecisionProfile::Mixed | PrecisionProfile::Fp64
                            ) =>
                        {
                            Ok(EvidenceValue::measured(value, support))
                        }
                        _ => Err(SemanticError::Invalid(
                            "GPU semantic scalar result dtype does not match its precision profile",
                        )),
                    },
                    GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT => Ok(EvidenceValue::Unavailable {
                        reason: UnavailableReason::InsufficientSupport,
                        support,
                    }),
                    GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND => Ok(EvidenceValue::Unavailable {
                        reason: UnavailableReason::ConstantOperand,
                        support,
                    }),
                    GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION => Ok(EvidenceValue::Unavailable {
                        reason: UnavailableReason::DegenerateReduction,
                        support,
                    }),
                    GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION => Ok(EvidenceValue::Unavailable {
                        reason: UnavailableReason::NonFiniteReduction,
                        support,
                    }),
                    _ => Err(SemanticError::Invalid(
                        "GPU semantic scalar result contains an unknown state",
                    )),
                }
            })
            .collect()
    }

    fn unavailable_for_all(
        candidates: usize,
        reason: UnavailableReason,
        support: usize,
    ) -> Vec<EvidenceValue> {
        vec![EvidenceValue::Unavailable { reason, support }; candidates]
    }
}

impl NativeEvidenceExecutor for GpuNativeEvidenceExecutor {
    fn backend_kind(&self) -> u32 {
        self.backend.kind
    }

    fn validate_evidence_admission(
        &self,
        definition: &EvidenceDefinition,
        pair_count: usize,
        support_rows: usize,
    ) -> SemanticResult<usize> {
        let EvidenceDefinition::Association { statistic, .. } = definition else {
            return Ok(0);
        };
        self.require_association(*statistic)?;
        let pair_count = u64::try_from(pair_count)
            .map_err(|_| SemanticError::Invalid("association pair count overflows u64"))?;
        let support_rows = u64::try_from(support_rows)
            .map_err(|_| SemanticError::Invalid("association support rows overflow u64"))?;
        if pair_count > self.capabilities.max_association_pairs {
            return Err(SemanticError::Unsupported(
                "association pair count exceeds selected GPU semantic capability",
            ));
        }
        match statistic {
            AssociationStatistic::Pearson => Ok(0),
            AssociationStatistic::Spearman => {
                if support_rows > self.capabilities.max_spearman_rows {
                    return Err(SemanticError::Unsupported(
                        "Spearman support rows exceed selected GPU semantic capability",
                    ));
                }
                let work = if self.backend.kind == GAFIME_BACKEND_METAL {
                    Self::metal_spearman_work(pair_count, support_rows)?
                } else {
                    pair_count
                        .checked_mul(support_rows)
                        .and_then(|value| value.checked_mul(support_rows))
                        .and_then(|value| value.checked_mul(2))
                        .ok_or(SemanticError::Invalid(
                            "GPU Spearman work bound overflows u64",
                        ))?
                };
                usize::try_from(work).map_err(|_| {
                    SemanticError::Invalid(
                        "GPU Spearman work exceeds host semantic-work representation",
                    )
                })
            }
            AssociationStatistic::FixedCorrectedNmi { .. } => {
                if support_rows > self.capabilities.max_fixed_corrected_nmi_rows {
                    return Err(SemanticError::Unsupported(
                        "fixed corrected NMI support rows exceed selected GPU semantic capability",
                    ));
                }
                Ok(0)
            }
        }
    }

    fn fit_means(
        &mut self,
        values: &MaterializedColumns,
        candidates: &[FeatureId],
        max_bytes: usize,
    ) -> SemanticResult<FrozenMeans> {
        self.require_profile(values.profile())?;
        self.require_column_means()?;
        if candidates.is_empty() {
            return Ok(match values.profile() {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => FrozenMeans::F32(Vec::new()),
                PrecisionProfile::Fp64 => FrozenMeans::F64(Vec::new()),
            });
        }
        if values.backend_kind() != self.backend.kind || !values.is_resident() {
            return Err(SemanticError::Invalid(
                "GPU mean fitting requires a resident selected-backend bank",
            ));
        }
        let lease = Arc::clone(values.resident_lease()?);
        let bank = lease
            .downcast::<OwnedSemanticBank>()
            .map_err(|_| SemanticError::Invalid("resident materialization lease is foreign"))?;
        let bank = (*bank).clone();
        if !bank.same_backend_owner(&self.backend) || bank.profile() != values.profile() {
            return Err(SemanticError::Invalid(
                "GPU mean-fitting bank does not match its executor",
            ));
        }
        let candidate_slots = Self::slots_for(values, candidates)?;
        let host_temporary = Self::u32_slice_bytes(
            candidate_slots.len(),
            "semantic mean slot staging exceeds address space",
        )?
        .checked_add(Self::mean_result_host_bytes(
            values.profile(),
            candidate_slots.len(),
        )?)
        .ok_or(SemanticError::Invalid(
            "semantic mean fitting staging exceeds address space",
        ))?;
        let forecast = self.native_forecast(
            &bank,
            GafimeSemanticForecastRequest {
                mean_slot_count: u64::try_from(candidate_slots.len()).map_err(|_| {
                    SemanticError::Invalid("semantic mean candidate count overflows u64")
                })?,
                ..Default::default()
            },
        )?;
        Self::reserve_forecast(forecast, false, host_temporary, max_bytes)?;
        let expected_support = bank.rows();
        let results = bank
            .column_means(&candidate_slots)
            .map_err(Self::semantic_error)?;
        if results.len() != candidates.len() {
            return Err(SemanticError::Invalid(
                "GPU mean fitting returned an unexpected result count",
            ));
        }
        match values.profile() {
            PrecisionProfile::Fp32 => results
                .into_iter()
                .map(|result| match result {
                    SemanticScalarResult {
                        value: SemanticScalarValue::F32(value),
                        state: GAFIME_SEMANTIC_SCALAR_MEASURED,
                        support,
                    } if support == expected_support && value.is_finite() => Ok(value.to_bits()),
                    _ => Err(SemanticError::Invalid(
                        "GPU fp32 mean fitting produced an undefined or nonfinite result",
                    )),
                })
                .collect::<SemanticResult<Vec<_>>>()
                .map(FrozenMeans::F32),
            PrecisionProfile::Mixed => results
                .into_iter()
                .map(|result| match result {
                    SemanticScalarResult {
                        value: SemanticScalarValue::F64(value),
                        state: GAFIME_SEMANTIC_SCALAR_MEASURED,
                        support,
                    } if support == expected_support && value.is_finite() => {
                        let frozen = value as f32;
                        if frozen.is_finite() {
                            Ok(frozen.to_bits())
                        } else {
                            Err(SemanticError::Invalid(
                                "GPU mixed mean fitting cannot freeze a finite f32 constant",
                            ))
                        }
                    }
                    _ => Err(SemanticError::Invalid(
                        "GPU mixed mean fitting produced an undefined or nonfinite result",
                    )),
                })
                .collect::<SemanticResult<Vec<_>>>()
                .map(FrozenMeans::F32),
            PrecisionProfile::Fp64 => results
                .into_iter()
                .map(|result| match result {
                    SemanticScalarResult {
                        value: SemanticScalarValue::F64(value),
                        state: GAFIME_SEMANTIC_SCALAR_MEASURED,
                        support,
                    } if support == expected_support && value.is_finite() => Ok(value.to_bits()),
                    _ => Err(SemanticError::Invalid(
                        "GPU fp64 mean fitting produced an undefined or nonfinite result",
                    )),
                })
                .collect::<SemanticResult<Vec<_>>>()
                .map(FrozenMeans::F64),
        }
    }

    fn materialize(
        &mut self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        candidates: &[FeatureId],
        retained: Option<&MaterializedColumns>,
        max_bytes: usize,
    ) -> SemanticResult<MaterializedColumns> {
        self.validate_context(registry, frame)?;
        if candidates.is_empty() {
            return MaterializedColumns::empty_resident(registry, frame, self.backend.kind);
        }
        let retained_bank = match retained {
            Some(values) => {
                if values.frame_id() != frame.id() || values.profile() != frame.profile() {
                    return Err(SemanticError::Invalid(
                        "retained semantic values belong to another input context",
                    ));
                }
                Some((self.resident_bank(values, frame)?, values.resident_slots()?))
            }
            None => None,
        };
        let (needed, reused) = Self::dependency_ids(
            registry,
            candidates,
            retained_bank.as_ref().map(|(_, slots)| *slots),
        )?;
        let source_ids = needed
            .iter()
            .filter(|id| !reused.contains(id))
            .filter_map(|&id| match registry.program(id).ok()?.op() {
                FeatureOp::Source(index) => Some((*index, id)),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        if needed.len() > usize::try_from(self.capabilities.max_slot_count).unwrap_or(0)
            || needed.len() > registry.limits().max_nodes
        {
            return Err(SemanticError::Invalid(
                "GPU semantic dependency bank exceeds declared slot capacity",
            ));
        }
        if !source_ids.is_empty() {
            self.require_program_op(GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE)?;
        }

        let mut slots = BTreeMap::new();
        for (_, id) in &source_ids {
            let slot = u32::try_from(slots.len())
                .map_err(|_| SemanticError::Invalid("semantic source slot count overflows u32"))?;
            slots.insert(*id, slot);
        }
        for &id in &needed {
            if slots.contains_key(&id) {
                continue;
            }
            let slot = u32::try_from(slots.len())
                .map_err(|_| SemanticError::Invalid("semantic bank slot count overflows u32"))?;
            slots.insert(id, slot);
        }

        let mut native_nodes = Vec::new();
        let mut ordered: Vec<_> = needed.iter().copied().collect();
        ordered.sort_by_key(|id| {
            (
                registry
                    .program(*id)
                    .map_or(usize::MAX, |program| program.depth()),
                *id,
            )
        });
        for id in ordered {
            let program = registry.program(id)?;
            if reused.contains(&id) {
                continue;
            }
            if matches!(program.op(), FeatureOp::Source(_)) {
                continue;
            }
            let output_slot = *slots
                .get(&id)
                .ok_or(SemanticError::Invalid("missing semantic output slot"))?;
            match program.op() {
                FeatureOp::Source(_) => unreachable!("source nodes were filtered above"),
                FeatureOp::AbsoluteDifference(left, right) => {
                    self.require_program_op(GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE)?;
                    native_nodes.push(SemanticProgramNode::AbsoluteDifference {
                        output_slot,
                        left_slot: *slots
                            .get(left)
                            .ok_or(SemanticError::Invalid("missing left semantic operand slot"))?,
                        right_slot: *slots.get(right).ok_or(SemanticError::Invalid(
                            "missing right semantic operand slot",
                        ))?,
                    });
                }
                FeatureOp::Softsign(input) => {
                    self.require_program_op(GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN)?;
                    native_nodes.push(SemanticProgramNode::Softsign {
                        output_slot,
                        input_slot: *slots.get(input).ok_or(SemanticError::Invalid(
                            "missing softsign semantic operand slot",
                        ))?,
                    });
                }
                FeatureOp::CenteredProduct {
                    operands,
                    mean_bits,
                } => {
                    self.require_program_op(GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT)?;
                    let mean_bits = match mean_bits {
                        FrozenMeans::F32(bits) => {
                            bits.iter().map(|bits| u64::from(*bits)).collect()
                        }
                        FrozenMeans::F64(bits) => bits.clone(),
                    };
                    native_nodes.push(SemanticProgramNode::CenteredProduct {
                        output_slot,
                        operand_slots: operands
                            .iter()
                            .map(|operand| {
                                slots.get(operand).copied().ok_or(SemanticError::Invalid(
                                    "missing centered-product semantic operand slot",
                                ))
                            })
                            .collect::<SemanticResult<Vec<_>>>()?,
                        mean_bits,
                    });
                }
                FeatureOp::HardPredicate {
                    input,
                    comparison,
                    threshold_bits,
                } => {
                    self.require_program_op(
                        GAFIME_SEMANTIC_PROGRAM_OP_MASK_FROZEN_REGION_CONJUNCTION,
                    )?;
                    self.require_region_terms(1)?;
                    native_nodes.push(SemanticProgramNode::FrozenRegionConjunction {
                        output_slot,
                        terms: vec![Self::lower_region_term(
                            frame.profile(),
                            &slots,
                            *input,
                            *comparison,
                            threshold_bits,
                        )?],
                    });
                }
                FeatureOp::DecisionRegion { terms } => {
                    self.require_program_op(
                        GAFIME_SEMANTIC_PROGRAM_OP_MASK_FROZEN_REGION_CONJUNCTION,
                    )?;
                    self.require_region_terms(terms.len())?;
                    let terms = terms
                        .iter()
                        .map(|term| match registry.program(*term)?.op() {
                            FeatureOp::HardPredicate {
                                input,
                                comparison,
                                threshold_bits,
                            } => Self::lower_region_term(
                                frame.profile(),
                                &slots,
                                *input,
                                *comparison,
                                threshold_bits,
                            ),
                            _ => Err(SemanticError::Invalid(
                                "decision-region term is not a hard predicate",
                            )),
                        })
                        .collect::<SemanticResult<Vec<_>>>()?;
                    native_nodes
                        .push(SemanticProgramNode::FrozenRegionConjunction { output_slot, terms });
                }
            }
        }

        if native_nodes.len() > usize::try_from(self.capabilities.max_program_nodes).unwrap_or(0) {
            return Err(SemanticError::Invalid(
                "GPU semantic program node count exceeds payload capability",
            ));
        }

        if !reused.is_empty()
            && u64::try_from(frame.rows())
                .map_err(|_| SemanticError::Invalid("semantic frame rows overflow u64"))?
                > self.capabilities.max_gather_rows
        {
            return Err(SemanticError::Invalid(
                "GPU retained gather rows exceed payload capability",
            ));
        }
        let resident_bytes = Self::storage_bytes(frame.profile(), frame.rows(), slots.len())?;
        let source_staging_bytes =
            Self::storage_bytes(frame.profile(), frame.rows(), source_ids.len())?;
        let program_host_bytes =
            Self::program_host_temporary_bytes(&native_nodes, native_nodes.capacity())?;
        let reused_host_bytes = if reused.is_empty() {
            0
        } else {
            let reused_row_bytes =
                Self::u64_slice_bytes(frame.rows(), "retained row staging exceeds address space")?;
            Self::u32_slice_bytes(
                reused.len(),
                "retained source slot staging exceeds address space",
            )?
            .checked_add(Self::u32_slice_bytes(
                reused.len(),
                "retained destination slot staging exceeds address space",
            )?)
            .and_then(|bytes| bytes.checked_add(reused_row_bytes))
            .ok_or(SemanticError::Invalid(
                "retained gather host staging exceeds address space",
            ))?
        };
        let host_temporary_bytes = source_staging_bytes
            .checked_add(program_host_bytes)
            .and_then(|bytes| bytes.checked_add(reused_host_bytes))
            .ok_or(SemanticError::Invalid(
                "GPU semantic host staging exceeds execution budget address space",
            ))?;
        // The initial bank has no live handle for a forecast query.  Its exact
        // descriptor byte count is therefore the pre-allocation admission;
        // operation-specific native transient bytes are forecast immediately
        // after allocation and before upload/gather/arithmetic dispatch.
        Self::reserve_before_allocation(resident_bytes, host_temporary_bytes, max_bytes)?;

        let bank = self
            .backend
            .allocate_semantic_bank(
                frame.profile(),
                frame.rows(),
                u32::try_from(source_ids.len()).map_err(|_| {
                    SemanticError::Invalid("semantic source slot count overflows u32")
                })?,
                u32::try_from(slots.len()).map_err(|_| {
                    SemanticError::Invalid("semantic bank slot count overflows u32")
                })?,
            )
            .map_err(Self::semantic_error)?;

        let (program_operand_count, program_mean_count, program_region_term_count) =
            Self::program_descriptor_counts(&native_nodes)?;
        let forecast = self.native_forecast(
            &bank,
            GafimeSemanticForecastRequest {
                program_max_operand_count: Self::maximum_operand_count(&native_nodes)?,
                program_operand_count,
                program_mean_count,
                program_region_term_count,
                gather_slot_count: u64::try_from(reused.len()).map_err(|_| {
                    SemanticError::Invalid("retained gather slot count overflows u64")
                })?,
                gather_row_count: if reused.is_empty() {
                    0
                } else {
                    u64::try_from(frame.rows())
                        .map_err(|_| SemanticError::Invalid("semantic frame rows overflow u64"))?
                },
                ..Default::default()
            },
        )?;
        Self::reserve_forecast(forecast, true, host_temporary_bytes, max_bytes)?;

        if !source_ids.is_empty() {
            match frame.profile() {
                PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                    let mut columns =
                        Vec::with_capacity(source_ids.len().checked_mul(frame.rows()).ok_or(
                            SemanticError::Invalid(
                                "semantic source staging exceeds host address space",
                            ),
                        )?);
                    for (source, _) in &source_ids {
                        columns.extend_from_slice(frame.column_typed(*source as usize)?.as_f32()?);
                    }
                    bank.upload_f32(&columns).map_err(Self::semantic_error)?;
                }
                PrecisionProfile::Fp64 => {
                    let mut columns =
                        Vec::with_capacity(source_ids.len().checked_mul(frame.rows()).ok_or(
                            SemanticError::Invalid(
                                "semantic source staging exceeds host address space",
                            ),
                        )?);
                    for (source, _) in &source_ids {
                        columns.extend_from_slice(frame.column_typed(*source as usize)?.as_f64()?);
                    }
                    bank.upload_f64(&columns).map_err(Self::semantic_error)?;
                }
            }
        }

        if !reused.is_empty() {
            let (retained_bank, retained_slots) =
                retained_bank.as_ref().expect("reused requires bank");
            let source_slots = reused
                .iter()
                .map(|id| {
                    retained_slots
                        .get(id)
                        .copied()
                        .ok_or(SemanticError::Invalid(
                            "retained semantic slot disappeared during lowering",
                        ))
                })
                .collect::<SemanticResult<Vec<_>>>()?;
            let destination_slots = reused
                .iter()
                .map(|id| {
                    slots.get(id).copied().ok_or(SemanticError::Invalid(
                        "destination semantic slot disappeared during lowering",
                    ))
                })
                .collect::<SemanticResult<Vec<_>>>()?;
            let rows = (0..frame.rows())
                .map(|row| {
                    u64::try_from(row)
                        .map_err(|_| SemanticError::Invalid("row index overflows u64"))
                })
                .collect::<SemanticResult<Vec<_>>>()?;
            bank.sparse_gather_from(retained_bank, &source_slots, &destination_slots, &rows)
                .map_err(Self::semantic_error)?;
        }
        if !native_nodes.is_empty() {
            bank.materialize(&native_nodes)
                .map_err(Self::semantic_error)?;
        }

        let output_slots = candidates
            .iter()
            .map(|id| {
                slots
                    .get(id)
                    .copied()
                    .map(|slot| (*id, slot))
                    .ok_or(SemanticError::Invalid(
                        "requested semantic output disappeared during lowering",
                    ))
            })
            .collect::<SemanticResult<BTreeMap<_, _>>>()?;
        Self::materialized_columns(registry, frame, self.backend.kind, bank, output_slots)
    }

    fn evaluate_channel(
        &mut self,
        definition: &EvidenceDefinition,
        candidates: &[FeatureId],
        values: &MaterializedColumns,
        paired: Option<&MaterializedColumns>,
        max_bytes: usize,
    ) -> SemanticResult<Vec<EvidenceValue>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        // `MaterializedColumns` carries its immutable frame ID and the session
        // already validates each definition.  This executor additionally
        // validates that the opaque lease is owned by this exact GPU backend.
        if values.backend_kind() != self.backend.kind || !values.is_resident() {
            return Err(SemanticError::Invalid(
                "GPU semantic evidence requires a resident selected-backend bank",
            ));
        }
        self.require_profile(values.profile())?;
        let values_bank = {
            let lease = Arc::clone(values.resident_lease()?);
            let bank = lease
                .downcast::<OwnedSemanticBank>()
                .map_err(|_| SemanticError::Invalid("resident materialization lease is foreign"))?;
            let bank = (*bank).clone();
            if !bank.same_backend_owner(&self.backend) || bank.profile() != values.profile() {
                return Err(SemanticError::Invalid(
                    "GPU semantic evidence bank does not match its executor",
                ));
            }
            bank
        };
        let candidate_slots = Self::slots_for(values, candidates)?;
        let candidate_slot_bytes = Self::u32_slice_bytes(
            candidate_slots.len(),
            "semantic candidate slot staging exceeds address space",
        )?;
        match definition {
            EvidenceDefinition::Association {
                statistic,
                context: AssociationContext::Reference { reference },
            } => {
                self.require_association(*statistic)?;
                let reference_slot = values.resident_slots()?.get(reference).copied().ok_or(
                    SemanticError::Invalid(
                        "reference feature is absent from resident evidence bank",
                    ),
                )?;
                let right_slots = vec![reference_slot; candidate_slots.len()];
                let scalar_host =
                    Self::scalar_result_host_bytes(values.profile(), candidate_slots.len())?;
                let host_temporary = candidate_slot_bytes
                    .checked_add(Self::u32_slice_bytes(
                        right_slots.len(),
                        "semantic reference slot staging exceeds address space",
                    )?)
                    .and_then(|bytes| bytes.checked_add(scalar_host))
                    .ok_or(SemanticError::Invalid(
                        "semantic reference staging exceeds address space",
                    ))?;
                let forecast = self.native_forecast(
                    &values_bank,
                    GafimeSemanticForecastRequest {
                        pair_count: u64::try_from(candidate_slots.len()).map_err(|_| {
                            SemanticError::Invalid("semantic Pearson pair count overflows u64")
                        })?,
                        ..Default::default()
                    },
                )?;
                // The materialized values bank is already budgeted by the
                // session; only this operation's transient plus Rust staging
                // is admitted here.
                Self::reserve_forecast(forecast, false, host_temporary, max_bytes)?;
                Self::results_to_evidence(
                    values.profile(),
                    values_bank
                        .pairwise_association(
                            &values_bank,
                            &candidate_slots,
                            &right_slots,
                            Self::native_association_statistic(*statistic),
                            Self::native_association_presentation(*statistic, true),
                        )
                        .map_err(Self::semantic_error)?,
                )
            }
            EvidenceDefinition::Association {
                statistic,
                context: AssociationContext::PairedView { view },
            } => {
                self.require_association(*statistic)?;
                let paired = paired.ok_or(SemanticError::Invalid(
                    "paired GPU evidence requires a resident paired bank",
                ))?;
                if paired.frame_id() != view.id() || paired.profile() != values.profile() {
                    return Err(SemanticError::Invalid(
                        "paired GPU evidence context or profile mismatch",
                    ));
                }
                let paired_bank = self.resident_bank(paired, view)?;
                let right_slots = Self::slots_for(paired, candidates)?;
                let scalar_host =
                    Self::scalar_result_host_bytes(values.profile(), candidate_slots.len())?;
                let host_temporary = candidate_slot_bytes
                    .checked_add(Self::u32_slice_bytes(
                        right_slots.len(),
                        "semantic paired slot staging exceeds address space",
                    )?)
                    .and_then(|bytes| bytes.checked_add(scalar_host))
                    .ok_or(SemanticError::Invalid(
                        "semantic paired staging exceeds address space",
                    ))?;
                let forecast = self.native_forecast(
                    &values_bank,
                    GafimeSemanticForecastRequest {
                        pair_count: u64::try_from(candidate_slots.len()).map_err(|_| {
                            SemanticError::Invalid("semantic Pearson pair count overflows u64")
                        })?,
                        ..Default::default()
                    },
                )?;
                Self::reserve_forecast(forecast, false, host_temporary, max_bytes)?;
                Self::results_to_evidence(
                    values.profile(),
                    values_bank
                        .pairwise_association(
                            &paired_bank,
                            &candidate_slots,
                            &right_slots,
                            Self::native_association_statistic(*statistic),
                            Self::native_association_presentation(*statistic, false),
                        )
                        .map_err(Self::semantic_error)?,
                )
            }
            EvidenceDefinition::Association {
                context: AssociationContext::Labels { labels: None },
                ..
            } => Ok(Self::unavailable_for_all(
                candidates.len(),
                UnavailableReason::MissingLabels,
                0,
            )),
            EvidenceDefinition::Association {
                statistic,
                context:
                    AssociationContext::Labels {
                        labels: Some(labels),
                    },
            } => {
                self.require_association(*statistic)?;
                self.require_gather()?;
                if labels.frame_id() != values.frame_id() || labels.profile() != values.profile() {
                    return Err(SemanticError::Invalid(
                        "label GPU evidence context or profile mismatch",
                    ));
                }
                if labels.rows().len() < 2 {
                    return Ok(Self::unavailable_for_all(
                        candidates.len(),
                        UnavailableReason::InsufficientSupport,
                        labels.rows().len(),
                    ));
                }
                let capacity = candidates
                    .len()
                    .checked_add(1)
                    .ok_or(SemanticError::Invalid("label bank slot count overflow"))?;
                let label_rows = u64::try_from(labels.rows().len())
                    .map_err(|_| SemanticError::Invalid("label row count overflows u64"))?;
                if label_rows > self.capabilities.max_gather_rows {
                    return Err(SemanticError::Invalid(
                        "label gather rows exceed payload capability",
                    ));
                }
                let resident =
                    Self::storage_bytes(values.profile(), labels.rows().len(), capacity)?;
                let destination_slot_bytes = Self::u32_slice_bytes(
                    candidates.len(),
                    "label destination slot staging exceeds address space",
                )?;
                let label_slot_bytes = Self::u32_slice_bytes(
                    candidates.len(),
                    "label comparison slot staging exceeds address space",
                )?;
                let row_staging_bytes = Self::u64_slice_bytes(
                    labels.rows().len(),
                    "label row staging exceeds address space",
                )?;
                let scalar_host =
                    Self::scalar_result_host_bytes(values.profile(), candidates.len())?;
                let host_temporary = candidate_slot_bytes
                    .checked_add(destination_slot_bytes)
                    .and_then(|bytes| bytes.checked_add(label_slot_bytes))
                    .and_then(|bytes| bytes.checked_add(row_staging_bytes))
                    .and_then(|bytes| bytes.checked_add(scalar_host))
                    .ok_or(SemanticError::Invalid(
                        "label evidence host staging exceeds address space",
                    ))?;
                Self::reserve_before_allocation(resident, host_temporary, max_bytes)?;
                let label_bank = self
                    .backend
                    .allocate_semantic_bank(
                        values.profile(),
                        labels.rows().len(),
                        1,
                        u32::try_from(capacity).map_err(|_| {
                            SemanticError::Invalid("label bank slot count overflows u32")
                        })?,
                    )
                    .map_err(Self::semantic_error)?;
                let forecast = self.native_forecast(
                    &label_bank,
                    GafimeSemanticForecastRequest {
                        pair_count: u64::try_from(candidates.len()).map_err(|_| {
                            SemanticError::Invalid("label Pearson pair count overflows u64")
                        })?,
                        gather_slot_count: u64::try_from(candidates.len()).map_err(|_| {
                            SemanticError::Invalid("label gather slot count overflows u64")
                        })?,
                        gather_row_count: label_rows,
                        ..Default::default()
                    },
                )?;
                Self::reserve_forecast(forecast, true, host_temporary, max_bytes)?;
                match values.profile() {
                    PrecisionProfile::Fp32 | PrecisionProfile::Mixed => label_bank
                        .upload_f32(labels.values_typed().as_f32()?)
                        .map_err(Self::semantic_error)?,
                    PrecisionProfile::Fp64 => label_bank
                        .upload_f64(labels.values_typed().as_f64()?)
                        .map_err(Self::semantic_error)?,
                }
                let destination_slots = (1..=candidates.len())
                    .map(|slot| {
                        u32::try_from(slot).map_err(|_| {
                            SemanticError::Invalid("label destination slot overflows u32")
                        })
                    })
                    .collect::<SemanticResult<Vec<_>>>()?;
                let rows = labels
                    .rows()
                    .iter()
                    .map(|&row| {
                        u64::try_from(row)
                            .map_err(|_| SemanticError::Invalid("label row index overflows u64"))
                    })
                    .collect::<SemanticResult<Vec<_>>>()?;
                label_bank
                    .sparse_gather_from(&values_bank, &candidate_slots, &destination_slots, &rows)
                    .map_err(Self::semantic_error)?;
                let label_slots = vec![0; candidates.len()];
                Self::results_to_evidence(
                    values.profile(),
                    label_bank
                        .pairwise_association(
                            &label_bank,
                            &destination_slots,
                            &label_slots,
                            Self::native_association_statistic(*statistic),
                            Self::native_association_presentation(*statistic, true),
                        )
                        .map_err(Self::semantic_error)?,
                )
            }
            EvidenceDefinition::GraphEnergy { graph } => {
                self.require_graph_energy()?;
                if graph.frame_id() != values.frame_id() || graph.profile() != values.profile() {
                    return Err(SemanticError::Invalid(
                        "graph GPU evidence context or profile mismatch",
                    ));
                }
                let edges = graph
                    .edges()
                    .iter()
                    .map(|edge| {
                        Ok(SemanticEdge {
                            left_row: u64::try_from(edge.left).map_err(|_| {
                                SemanticError::Invalid("graph left row exceeds u64")
                            })?,
                            right_row: u64::try_from(edge.right).map_err(|_| {
                                SemanticError::Invalid("graph right row exceeds u64")
                            })?,
                        })
                    })
                    .collect::<SemanticResult<Vec<_>>>()?;
                let edge_staging_bytes = edges
                    .capacity()
                    .checked_mul(std::mem::size_of::<SemanticEdge>())
                    .ok_or(SemanticError::Invalid(
                        "graph edge staging exceeds address space",
                    ))?;
                // `ordered_edge_energy` lowers the semantic edge records to
                // a second exact-capacity C ABI vector while the source
                // staging remains live through the synchronous call.
                let native_edge_staging_bytes = edges
                    .len()
                    .checked_mul(std::mem::size_of::<GafimeSemanticEdge>())
                    .ok_or(SemanticError::Invalid(
                        "graph ABI edge staging exceeds address space",
                    ))?;
                let scalar_host =
                    Self::scalar_result_host_bytes(values.profile(), candidate_slots.len())?;
                let host_temporary = candidate_slot_bytes
                    .checked_add(edge_staging_bytes)
                    .and_then(|bytes| bytes.checked_add(native_edge_staging_bytes))
                    .and_then(|bytes| bytes.checked_add(scalar_host))
                    .ok_or(SemanticError::Invalid(
                        "graph evidence host staging exceeds address space",
                    ))?;
                let forecast = self.native_forecast(
                    &values_bank,
                    GafimeSemanticForecastRequest {
                        graph_candidate_count: u64::try_from(candidate_slots.len()).map_err(
                            |_| SemanticError::Invalid("graph candidate count overflows u64"),
                        )?,
                        graph_edge_count: u64::try_from(edges.len()).map_err(|_| {
                            SemanticError::Invalid("graph edge count overflows u64")
                        })?,
                        ..Default::default()
                    },
                )?;
                Self::reserve_forecast(forecast, false, host_temporary, max_bytes)?;
                let results = match values.profile() {
                    PrecisionProfile::Fp32 | PrecisionProfile::Mixed => values_bank
                        .ordered_edge_energy_f32(
                            &candidate_slots,
                            &edges,
                            graph.weights_typed().as_f32()?,
                        )
                        .map_err(Self::semantic_error)?,
                    PrecisionProfile::Fp64 => values_bank
                        .ordered_edge_energy_f64(
                            &candidate_slots,
                            &edges,
                            graph.weights_typed().as_f64()?,
                        )
                        .map_err(Self::semantic_error)?,
                };
                Self::results_to_evidence(values.profile(), results)
            }
        }
    }

    fn retain(
        &mut self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        source: &MaterializedColumns,
        prior: Option<&MaterializedColumns>,
        selected: &[FeatureId],
        max_live_bytes: usize,
    ) -> SemanticResult<MaterializedColumns> {
        self.validate_context(registry, frame)?;
        if source.frame_id() != frame.id()
            || source.profile() != frame.profile()
            || source.backend_kind() != self.backend.kind
            || prior.is_some_and(|values| {
                values.frame_id() != frame.id()
                    || values.profile() != frame.profile()
                    || values.backend_kind() != self.backend.kind
            })
        {
            return Err(SemanticError::Invalid(
                "GPU retained materialization context or backend mismatch",
            ));
        }
        if selected.is_empty() {
            return match prior {
                Some(values) => Ok(values.clone()),
                None => MaterializedColumns::empty_resident(registry, frame, self.backend.kind),
            };
        }
        let source_bank = self.resident_bank(source, frame)?;
        let source_slots = source.resident_slots()?;
        let prior_bank = match prior {
            Some(values) if !values.resident_slots()?.is_empty() => {
                Some((self.resident_bank(values, frame)?, values.resident_slots()?))
            }
            Some(_) | None => None,
        };

        let mut entries: BTreeMap<FeatureId, (OwnedSemanticBank, u32)> = BTreeMap::new();
        if let Some((bank, slots)) = &prior_bank {
            for (&feature, &slot) in *slots {
                entries.insert(feature, (bank.clone(), slot));
            }
        }
        for &feature in selected {
            registry.program(feature)?;
            let slot = source_slots
                .get(&feature)
                .copied()
                .ok_or(SemanticError::Invalid(
                    "selected feature is absent from its GPU materialization",
                ))?;
            entries
                .entry(feature)
                .or_insert((source_bank.clone(), slot));
        }
        let output_bytes = Self::storage_bytes(frame.profile(), frame.rows(), entries.len())?;

        struct CopyGroup {
            bank: OwnedSemanticBank,
            source_slots: Vec<u32>,
            destination_slots: Vec<u32>,
        }
        let mut groups: Vec<CopyGroup> = Vec::new();
        let mut output_slots = BTreeMap::new();
        for (index, (&feature, (bank, source_slot))) in entries.iter().enumerate() {
            let destination_slot = u32::try_from(index)
                .map_err(|_| SemanticError::Invalid("retained GPU slot count overflows u32"))?;
            output_slots.insert(feature, destination_slot);
            if let Some(group) = groups.iter_mut().find(|group| group.bank.same_bank(bank)) {
                group.source_slots.push(*source_slot);
                group.destination_slots.push(destination_slot);
            } else {
                groups.push(CopyGroup {
                    bank: bank.clone(),
                    source_slots: vec![*source_slot],
                    destination_slots: vec![destination_slot],
                });
            }
        }
        if entries.len() > usize::try_from(self.capabilities.max_slot_count).unwrap_or(0) {
            return Err(SemanticError::Invalid(
                "retained GPU slot count exceeds payload capability",
            ));
        }
        let max_gather_slots = groups
            .iter()
            .map(|group| group.source_slots.len())
            .max()
            .unwrap_or(0);
        let frame_rows = u64::try_from(frame.rows())
            .map_err(|_| SemanticError::Invalid("retained row count overflows u64"))?;
        if groups.len() > 1 && frame_rows > self.capabilities.max_gather_rows {
            return Err(SemanticError::Invalid(
                "retained gather rows exceed payload capability",
            ));
        }
        let prior_live = prior_bank
            .as_ref()
            .filter(|(bank, _)| !bank.same_bank(&source_bank))
            .map_or(Ok(0usize), |(bank, _)| {
                usize::try_from(bank.bytes()).map_err(|_| {
                    SemanticError::Invalid("retained GPU bank exceeds host address space")
                })
            })?;
        let copy_slot_staging = groups.iter().try_fold(0usize, |total, group| {
            let source_bytes = Self::u32_slice_bytes(
                group.source_slots.capacity(),
                "retained source slot staging exceeds address space",
            )?;
            let destination_bytes = Self::u32_slice_bytes(
                group.destination_slots.capacity(),
                "retained destination slot staging exceeds address space",
            )?;
            total
                .checked_add(source_bytes)
                .and_then(|total| total.checked_add(destination_bytes))
                .ok_or(SemanticError::Invalid(
                    "retained slot staging exceeds address space",
                ))
        })?;
        let host_temporary = if groups.len() > 1 {
            copy_slot_staging
                .checked_add(Self::u64_slice_bytes(
                    frame.rows(),
                    "retained row staging exceeds address space",
                )?)
                .ok_or(SemanticError::Invalid(
                    "retained host staging exceeds address space",
                ))?
        } else {
            copy_slot_staging
        };
        let forecast = self.native_forecast(
            &source_bank,
            GafimeSemanticForecastRequest {
                gather_slot_count: if groups.len() > 1 {
                    u64::try_from(max_gather_slots).map_err(|_| {
                        SemanticError::Invalid("retained gather slot count overflows u64")
                    })?
                } else {
                    0
                },
                gather_row_count: if groups.len() > 1 { frame_rows } else { 0 },
                retained_slot_count: u64::try_from(entries.len()).map_err(|_| {
                    SemanticError::Invalid("retained output slot count overflows u64")
                })?,
                ..Default::default()
            },
        )?;
        let forecast_output = usize::try_from(forecast.retained_bytes).map_err(|_| {
            SemanticError::Invalid("retained GPU forecast exceeds host address space")
        })?;
        if forecast_output != output_bytes {
            return Err(SemanticError::Invalid(
                "GPU retained forecast disagrees with its output descriptor bytes",
            ));
        }
        let live = Self::forecast_peak_bytes(forecast, true, host_temporary)?
            .checked_add(prior_live)
            .ok_or(SemanticError::Invalid(
                "retained GPU live byte count overflows host address space",
            ))?;
        if live > max_live_bytes {
            return Err(SemanticError::Invalid(
                "GPU retained materialization exceeds actual live byte budget",
            ));
        }

        let output_bank = if groups.len() == 1 {
            groups[0]
                .bank
                .retain(&groups[0].source_slots)
                .map_err(Self::semantic_error)?
        } else {
            let output = self
                .backend
                .allocate_semantic_bank(
                    frame.profile(),
                    frame.rows(),
                    0,
                    u32::try_from(entries.len()).map_err(|_| {
                        SemanticError::Invalid("retained GPU slot count overflows u32")
                    })?,
                )
                .map_err(Self::semantic_error)?;
            let rows = (0..frame.rows())
                .map(|row| {
                    u64::try_from(row)
                        .map_err(|_| SemanticError::Invalid("row index overflows u64"))
                })
                .collect::<SemanticResult<Vec<_>>>()?;
            for group in &groups {
                output
                    .sparse_gather_from(
                        &group.bank,
                        &group.source_slots,
                        &group.destination_slots,
                        &rows,
                    )
                    .map_err(Self::semantic_error)?;
            }
            output
        };
        if output_bank.bytes() != forecast.retained_bytes {
            return Err(SemanticError::Invalid(
                "GPU retained bank bytes disagree with the pre-allocation forecast",
            ));
        }
        Self::materialized_columns(
            registry,
            frame,
            self.backend.kind,
            output_bank,
            output_slots,
        )
    }

    fn download(
        &mut self,
        registry: &CandidateRegistry,
        frame: &FeatureFrame,
        source: &MaterializedColumns,
        max_bytes: usize,
    ) -> SemanticResult<MaterializedColumns> {
        self.validate_context(registry, frame)?;
        if source.frame_id() != frame.id()
            || source.profile() != frame.profile()
            || source.backend_kind() != self.backend.kind
        {
            return Err(SemanticError::Invalid(
                "GPU materialization download context or backend mismatch",
            ));
        }
        let slots = source.resident_slots()?;
        if slots.is_empty() {
            return MaterializedColumns::from_downloaded(
                registry,
                frame,
                self.backend.kind,
                BTreeMap::new(),
            );
        }
        let output_bytes = Self::storage_bytes(frame.profile(), frame.rows(), slots.len())?;
        let ordered = slots
            .iter()
            .map(|(&feature, &slot)| (feature, slot))
            .collect::<Vec<_>>();
        let ordered_staging_bytes = ordered
            .capacity()
            .checked_mul(std::mem::size_of::<(FeatureId, u32)>())
            .ok_or(SemanticError::Invalid(
                "GPU download slot-order staging exceeds host address space",
            ))?;
        if output_bytes > max_bytes {
            return Err(SemanticError::Invalid(
                "GPU materialization download exceeds explicit host output budget",
            ));
        }
        let bank = self.resident_bank(source, frame)?;
        let forecast = self.native_forecast(&bank, GafimeSemanticForecastRequest::default())?;
        // The source resident bank is budgeted by the session before this
        // call.  Account the final host columns, explicit slot-order staging,
        // and every native transfer temporary; each downloaded one-slot Vec
        // below moves directly into its NumericColumn, so no second full host
        // numeric copy is created.
        let host_temporary =
            output_bytes
                .checked_add(ordered_staging_bytes)
                .ok_or(SemanticError::Invalid(
                    "GPU download host staging exceeds address space",
                ))?;
        Self::reserve_forecast(forecast, false, host_temporary, max_bytes)?;
        let mut columns = BTreeMap::new();
        match frame.profile() {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                for (feature, slot) in &ordered {
                    let values = bank
                        .download_f32(std::slice::from_ref(slot))
                        .map_err(Self::semantic_error)?;
                    columns.insert(*feature, NumericColumn::from(values));
                }
            }
            PrecisionProfile::Fp64 => {
                for (feature, slot) in &ordered {
                    let values = bank
                        .download_f64(std::slice::from_ref(slot))
                        .map_err(Self::semantic_error)?;
                    columns.insert(*feature, NumericColumn::from(values));
                }
            }
        }
        MaterializedColumns::from_downloaded(registry, frame, self.backend.kind, columns)
    }
}

const fn result_width(profile: PrecisionProfile) -> usize {
    match profile {
        PrecisionProfile::Fp32 => std::mem::size_of::<f32>(),
        PrecisionProfile::Mixed | PrecisionProfile::Fp64 => std::mem::size_of::<f64>(),
    }
}
