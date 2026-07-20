pub mod combos;
mod legacy_rng;
pub mod shapes;
pub mod spine;

use gafime_types::{
    BackendKind, CandidateFamily, GafimeArityChunk, GafimeLaunchProtocol,
    GafimePermutationSchedule, GafimeRankSpec, GafimeShapeHint, GAFIME_ABI_VERSION,
    GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM,
    GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    GAFIME_METRIC_SPEARMAN,
};
use std::sync::{Arc, OnceLock};

use crate::backend::{OrchestratorError, OrchestratorResult};

/// Maximum number of `u32` descriptor words retained for one generated launch.
/// Bounded ranked plans above this ceiling use generated storage and this same
/// limit for batches. Unranked eager admission has a separate host-memory budget.
pub const DEFAULT_DESCRIPTOR_BATCH_WORDS: usize = 1 << 20;

#[derive(Debug)]
enum DescriptorStorage {
    Materialized(Arc<[u32]>),
    Generated {
        source: combos::CombinationDescriptorSource,
        materialized: OnceLock<Arc<[u32]>>,
    },
}

#[derive(Clone, Debug)]
enum DescriptorBatchSourceStorage {
    Materialized(Arc<[u32]>),
    Generated(combos::CombinationDescriptorSource),
}

/// Cloneable, thread-safe descriptor traversal detached from raw ABI pointers.
/// Cloning shares descriptor/feature storage and copies only the small chunk table.
#[derive(Clone, Debug)]
pub struct DescriptorBatchSource {
    descriptors: DescriptorBatchSourceStorage,
    chunks: Arc<[GafimeArityChunk]>,
}

impl DescriptorBatchSource {
    pub fn descriptor_batches(
        &self,
        max_descriptor_words: usize,
    ) -> OrchestratorResult<DescriptorBatchIter> {
        if max_descriptor_words == 0 {
            return Err(OrchestratorError::InvalidPlan(
                "descriptor batch word limit is zero",
            ));
        }
        let required_words = self
            .chunks
            .iter()
            .map(|chunk| chunk.arity as usize)
            .max()
            .unwrap_or(0);
        if max_descriptor_words < required_words {
            return Err(OrchestratorError::InvalidPlan(
                "descriptor batch word limit is smaller than plan arity",
            ));
        }
        Ok(DescriptorBatchIter {
            source: self.clone(),
            max_descriptor_words,
            chunk_index: 0,
            chunk_rows_emitted: 0,
            positions: Vec::new(),
        })
    }
}

#[derive(Debug)]
pub struct DescriptorBatch {
    source_chunk: GafimeArityChunk,
    logical_row_offset: u64,
    combo_indices: Vec<u32>,
}

impl DescriptorBatch {
    pub fn logical_row_offset(&self) -> u64 {
        self.logical_row_offset
    }

    pub fn arity(&self) -> u32 {
        self.source_chunk.arity
    }

    pub fn combo_count(&self) -> u64 {
        self.combo_indices.len() as u64 / u64::from(self.source_chunk.arity)
    }

    pub fn combo_indices(&self) -> &[u32] {
        &self.combo_indices
    }

    pub(crate) fn launch_chunk(&self) -> GafimeArityChunk {
        GafimeArityChunk {
            combo_row_offset: 0,
            combo_count: self.combo_count(),
            local_chunk_id: 0,
            descriptor_offset: 0,
            descriptor_count: self.combo_count(),
            ..self.source_chunk
        }
    }
}

#[derive(Debug)]
pub struct DescriptorBatchIter {
    source: DescriptorBatchSource,
    max_descriptor_words: usize,
    chunk_index: usize,
    chunk_rows_emitted: u64,
    positions: Vec<usize>,
}

#[derive(Debug)]
pub struct CompiledPlan {
    protocol: GafimeLaunchProtocol,
    descriptors: DescriptorStorage,
    metric_ids: Vec<u32>,
    chunks: Vec<GafimeArityChunk>,
    shape_hints: Vec<GafimeShapeHint>,
}

impl CompiledPlan {
    pub fn single_chunk(
        backend_kind: BackendKind,
        n_samples: u64,
        n_features: u32,
        family: CandidateFamily,
        arity: u32,
        combo_indices: Vec<u32>,
        metric_ids: Vec<u32>,
    ) -> Self {
        let combo_count = if arity == 0 {
            0
        } else {
            combo_indices.len() as u64 / arity as u64
        };
        let chunks = vec![GafimeArityChunk {
            arity,
            family,
            metric_mask: 0,
            shape_hint_index: 0,
            combo_row_offset: 0,
            combo_count,
            local_chunk_id: 0,
            flags: 0,
            descriptor_offset: 0,
            descriptor_count: combo_count,
        }];
        let mut shape_hint = shapes::default_shape_hint(backend_kind, arity);
        shape_hint.vendor_hint =
            combos::select_adaptive_mi_bins_for_backend(backend_kind, n_samples, 96);
        let shape_hints = vec![shape_hint];
        Self::from_parts(
            backend_kind,
            n_samples,
            n_features,
            arity,
            combo_indices,
            metric_ids,
            chunks,
            shape_hints,
            GafimeRankSpec::default(),
            GafimePermutationSchedule::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        backend_kind: BackendKind,
        n_samples: u64,
        n_features: u32,
        max_arity: u32,
        combo_indices: Vec<u32>,
        metric_ids: Vec<u32>,
        chunks: Vec<GafimeArityChunk>,
        shape_hints: Vec<GafimeShapeHint>,
        rank: GafimeRankSpec,
        permutations: GafimePermutationSchedule,
    ) -> Self {
        Self::from_descriptor_storage(
            backend_kind,
            n_samples,
            n_features,
            max_arity,
            DescriptorStorage::Materialized(combo_indices.into()),
            metric_ids,
            chunks,
            shape_hints,
            rank,
            permutations,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_combination_parts(
        backend_kind: BackendKind,
        n_samples: u64,
        n_features: u32,
        max_arity: u32,
        source: combos::CombinationDescriptorSource,
        metric_ids: Vec<u32>,
        chunks: Vec<GafimeArityChunk>,
        shape_hints: Vec<GafimeShapeHint>,
        rank: GafimeRankSpec,
        permutations: GafimePermutationSchedule,
    ) -> Self {
        Self::from_descriptor_storage(
            backend_kind,
            n_samples,
            n_features,
            max_arity,
            DescriptorStorage::Generated {
                source,
                materialized: OnceLock::new(),
            },
            metric_ids,
            chunks,
            shape_hints,
            rank,
            permutations,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_descriptor_storage(
        backend_kind: BackendKind,
        n_samples: u64,
        n_features: u32,
        max_arity: u32,
        descriptors: DescriptorStorage,
        metric_ids: Vec<u32>,
        chunks: Vec<GafimeArityChunk>,
        shape_hints: Vec<GafimeShapeHint>,
        rank: GafimeRankSpec,
        permutations: GafimePermutationSchedule,
    ) -> Self {
        let family_count = chunks
            .iter()
            .map(|chunk| chunk.family)
            .collect::<std::collections::BTreeSet<_>>()
            .len() as u32;
        let protocol = GafimeLaunchProtocol {
            abi_version: GAFIME_ABI_VERSION,
            backend_kind,
            flags: 0,
            max_arity,
            n_samples,
            n_features,
            family_count,
            combo_indices: Default::default(),
            metric_ids: Default::default(),
            chunks: core::ptr::null(),
            chunk_count: 0,
            reserved32_a: 0,
            shape_hints: core::ptr::null(),
            shape_hint_count: 0,
            reserved32_b: 0,
            rank,
            permutations,
            reserved: [0; 8],
        };
        let mut plan = Self {
            protocol,
            descriptors,
            metric_ids,
            chunks,
            shape_hints,
        };
        plan.rebind_protocol_views();
        plan
    }

    /// Return the plan's protocol metadata and currently bound descriptor view.
    /// Generated plans intentionally expose an empty descriptor slice here; use
    /// `descriptor_batches`, `execute_plan`, or `materialized_protocol` to launch.
    pub fn protocol(&self) -> &GafimeLaunchProtocol {
        &self.protocol
    }

    /// Explicitly opt into a monolithic ABI protocol. Calling this on generated
    /// storage materializes the full descriptor family; prepared execution and
    /// significance paths deliberately avoid this method.
    pub fn materialized_protocol(&self) -> GafimeLaunchProtocol {
        let combo_indices = self.combo_indices();
        let mut protocol = self.protocol;
        protocol.combo_indices = gafime_types::GafimeSliceU32 {
            ptr: combo_indices.as_ptr(),
            len: combo_indices.len() as u64,
        };
        protocol
    }

    /// Materialize a generated descriptor family only when it fits the caller's
    /// explicit word bound. Already materialized plans preserve the legacy path.
    pub fn try_materialized_protocol(
        &self,
        max_generated_descriptor_words: u64,
    ) -> OrchestratorResult<GafimeLaunchProtocol> {
        if self.uses_generated_descriptors()
            && self.logical_descriptor_words() > max_generated_descriptor_words
        {
            return Err(OrchestratorError::Unsupported(
                "generated launch protocol materialization exceeds the compatibility host-memory budget",
            ));
        }
        Ok(self.materialized_protocol())
    }

    pub(crate) fn protocol_template(&self) -> GafimeLaunchProtocol {
        self.protocol
    }

    pub fn planned_row_count(&self) -> u64 {
        self.chunks
            .iter()
            .fold(0u64, |total, chunk| total.saturating_add(chunk.combo_count))
    }

    pub fn max_arity(&self) -> u32 {
        self.protocol.max_arity
    }

    pub fn metric_count(&self) -> u32 {
        self.metric_ids.len() as u32
    }

    pub fn metric_ids(&self) -> &[u32] {
        &self.metric_ids
    }

    pub fn combo_indices(&self) -> &[u32] {
        match &self.descriptors {
            DescriptorStorage::Materialized(combo_indices) => combo_indices,
            DescriptorStorage::Generated {
                source,
                materialized,
            } => materialized
                .get_or_init(|| Arc::from(source.materialize(&self.chunks)))
                .as_ref(),
        }
    }

    pub fn uses_generated_descriptors(&self) -> bool {
        matches!(&self.descriptors, DescriptorStorage::Generated { .. })
    }

    pub fn materialized_descriptor_words(&self) -> usize {
        match &self.descriptors {
            DescriptorStorage::Materialized(combo_indices) => combo_indices.len(),
            DescriptorStorage::Generated { materialized, .. } => {
                materialized.get().map_or(0, |words| words.len())
            }
        }
    }

    pub fn retained_descriptor_metadata_words(&self) -> usize {
        match &self.descriptors {
            DescriptorStorage::Materialized(_) => 0,
            DescriptorStorage::Generated { source, .. } => source.retained_word_count(),
        }
    }

    pub fn logical_descriptor_words(&self) -> u64 {
        self.chunks.iter().fold(0u64, |total, chunk| {
            total.saturating_add(chunk.combo_count.saturating_mul(u64::from(chunk.arity)))
        })
    }

    pub fn peak_descriptor_batch_words(&self, max_descriptor_words: usize) -> u64 {
        self.chunks
            .iter()
            .map(|chunk| {
                let arity = chunk.arity as usize;
                let row_limit = (max_descriptor_words / arity).max(1) as u64;
                chunk
                    .combo_count
                    .min(row_limit)
                    .saturating_mul(u64::from(chunk.arity))
            })
            .max()
            .unwrap_or(0)
    }

    pub fn descriptors_are_materialized(&self) -> bool {
        match &self.descriptors {
            DescriptorStorage::Materialized(_) => true,
            DescriptorStorage::Generated { materialized, .. } => materialized.get().is_some(),
        }
    }

    pub fn descriptor_batches(
        &self,
        max_descriptor_words: usize,
    ) -> OrchestratorResult<DescriptorBatchIter> {
        self.validate()?;
        self.descriptor_batches_validated(max_descriptor_words)
    }

    pub fn descriptor_batch_source(&self) -> OrchestratorResult<DescriptorBatchSource> {
        self.validate()?;
        Ok(self.descriptor_batch_source_validated())
    }

    pub(crate) fn descriptor_batches_validated(
        &self,
        max_descriptor_words: usize,
    ) -> OrchestratorResult<DescriptorBatchIter> {
        self.descriptor_batch_source_validated()
            .descriptor_batches(max_descriptor_words)
    }

    fn descriptor_batch_source_validated(&self) -> DescriptorBatchSource {
        let descriptors = match &self.descriptors {
            DescriptorStorage::Materialized(combo_indices) => {
                DescriptorBatchSourceStorage::Materialized(Arc::clone(combo_indices))
            }
            DescriptorStorage::Generated { source, .. } => {
                DescriptorBatchSourceStorage::Generated(source.clone())
            }
        };
        DescriptorBatchSource {
            descriptors,
            chunks: Arc::from(self.chunks.as_slice()),
        }
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.protocol.backend_kind
    }

    pub fn flags(&self) -> u32 {
        self.protocol.flags
    }

    pub fn rank(&self) -> GafimeRankSpec {
        self.protocol.rank
    }

    pub fn permutations(&self) -> GafimePermutationSchedule {
        self.protocol.permutations
    }

    pub fn with_rank(mut self, rank: GafimeRankSpec) -> Self {
        self.protocol.rank = rank;
        self
    }

    pub fn with_permutations(mut self, permutations: GafimePermutationSchedule) -> Self {
        self.protocol.permutations = permutations;
        self
    }

    pub fn with_flags(mut self, flags: u32) -> Self {
        self.protocol.flags = flags;
        self
    }

    pub fn chunks(&self) -> &[GafimeArityChunk] {
        &self.chunks
    }

    pub fn validate(&self) -> OrchestratorResult<()> {
        if self.protocol.abi_version != GAFIME_ABI_VERSION {
            return Err(OrchestratorError::InvalidPlan("ABI version mismatch"));
        }
        if !matches!(
            self.protocol.backend_kind,
            GAFIME_BACKEND_CPU | GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ) {
            return Err(OrchestratorError::InvalidPlan("unknown backend kind"));
        }
        if self.protocol.n_samples == 0 {
            return Err(OrchestratorError::InvalidPlan("plan has no samples"));
        }
        if self.protocol.n_features == 0 {
            return Err(OrchestratorError::InvalidPlan("plan has no features"));
        }
        if self.protocol.max_arity == 0 {
            return Err(OrchestratorError::InvalidPlan("plan max arity is zero"));
        }
        if self.protocol.max_arity > self.protocol.n_features {
            return Err(OrchestratorError::InvalidPlan(
                "plan max arity exceeds feature count",
            ));
        }
        if self.metric_ids.is_empty() {
            return Err(OrchestratorError::InvalidPlan("plan has no metrics"));
        }
        if self.metric_ids.len() > u32::MAX as usize {
            return Err(OrchestratorError::InvalidPlan(
                "plan has too many metrics for the ABI",
            ));
        }
        if self.metric_ids.iter().any(|metric| {
            !matches!(
                *metric,
                GAFIME_METRIC_PEARSON
                    | GAFIME_METRIC_SPEARMAN
                    | GAFIME_METRIC_MUTUAL_INFO
                    | GAFIME_METRIC_R2
            )
        }) {
            return Err(OrchestratorError::InvalidPlan(
                "plan contains an unknown metric",
            ));
        }
        if self.chunks.is_empty() {
            return Err(OrchestratorError::InvalidPlan("plan has no chunks"));
        }
        if self.shape_hints.is_empty() {
            return Err(OrchestratorError::InvalidPlan("plan has no shape hints"));
        }
        if self.chunks.len() > u32::MAX as usize {
            return Err(OrchestratorError::InvalidPlan(
                "plan has too many chunks for the ABI",
            ));
        }
        if self.shape_hints.len() > u32::MAX as usize {
            return Err(OrchestratorError::InvalidPlan(
                "plan has too many shape hints for the ABI",
            ));
        }
        if self.protocol.family_count != 1 {
            return Err(OrchestratorError::InvalidPlan(
                "continuous plan must declare exactly one family",
            ));
        }
        if self.protocol.rank.top_k > 0
            && !self.metric_ids.contains(&self.protocol.rank.primary_metric)
        {
            return Err(OrchestratorError::InvalidPlan(
                "rank primary metric is not in the plan metric set",
            ));
        }
        if self.protocol.rank.top_k > 0 && self.protocol.rank.include_ties != 0 {
            return Err(OrchestratorError::Unsupported(
                "rank.include_ties is unsupported",
            ));
        }

        let materialized_combo_indices = match &self.descriptors {
            DescriptorStorage::Materialized(combo_indices) => Some(combo_indices.as_ref()),
            DescriptorStorage::Generated {
                source,
                materialized,
            } => {
                source.validate(self.protocol.n_features)?;
                materialized.get().map(Arc::as_ref)
            }
        };
        let combo_index_count = materialized_combo_indices
            .map(|combo_indices| {
                u64::try_from(combo_indices.len()).map_err(|_| {
                    OrchestratorError::InvalidPlan(
                        "combo index buffer exceeds the ABI address space",
                    )
                })
            })
            .transpose()?;
        let mut expected_row_offset = 0u64;
        let mut expected_descriptor_offset = 0u64;
        let mut observed_max_arity = 0u32;
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            if chunk.arity == 0 {
                return Err(OrchestratorError::InvalidPlan("chunk arity is zero"));
            }
            if chunk.arity > self.protocol.max_arity {
                return Err(OrchestratorError::InvalidPlan(
                    "chunk arity exceeds plan max arity",
                ));
            }
            observed_max_arity = observed_max_arity.max(chunk.arity);
            if chunk.family != GAFIME_FAMILY_CONTINUOUS {
                return Err(OrchestratorError::InvalidPlan(
                    "compiled scoring chunks must be continuous",
                ));
            }
            if chunk.combo_count == 0 {
                return Err(OrchestratorError::InvalidPlan("chunk has no combinations"));
            }
            if chunk.descriptor_count != chunk.combo_count {
                return Err(OrchestratorError::InvalidPlan(
                    "chunk descriptor count does not match combination count",
                ));
            }
            if chunk.combo_row_offset != expected_row_offset {
                return Err(OrchestratorError::InvalidPlan(
                    "chunk output row range is not contiguous",
                ));
            }
            if chunk.descriptor_offset != expected_descriptor_offset {
                return Err(OrchestratorError::InvalidPlan(
                    "chunk descriptor range is not contiguous",
                ));
            }
            if chunk.local_chunk_id != chunk_index as u32 {
                return Err(OrchestratorError::InvalidPlan(
                    "chunk id does not match chunk order",
                ));
            }
            let Some(shape_hint) = self.shape_hints.get(chunk.shape_hint_index as usize) else {
                return Err(OrchestratorError::InvalidPlan(
                    "chunk shape hint index is out of range",
                ));
            };
            if !combos::MI_TEMPLATE_BIN_LEVELS.contains(&shape_hint.vendor_hint)
                || (self.protocol.backend_kind == GAFIME_BACKEND_METAL
                    && shape_hint.vendor_hint > 48)
            {
                return Err(OrchestratorError::InvalidPlan(
                    "chunk has an unsupported MI shape hint",
                ));
            }

            let descriptor_span = chunk.combo_count.checked_mul(chunk.arity as u64).ok_or(
                OrchestratorError::InvalidPlan("chunk descriptor range overflows"),
            )?;
            let descriptor_end = chunk.descriptor_offset.checked_add(descriptor_span).ok_or(
                OrchestratorError::InvalidPlan("chunk descriptor range overflows"),
            )?;
            if let Some(combo_index_count) = combo_index_count {
                if descriptor_end > combo_index_count {
                    return Err(OrchestratorError::InvalidPlan(
                        "chunk exceeds combo index buffer",
                    ));
                }
                let descriptor_start = usize::try_from(chunk.descriptor_offset).map_err(|_| {
                    OrchestratorError::InvalidPlan("chunk descriptor offset exceeds address space")
                })?;
                let descriptor_end_index = usize::try_from(descriptor_end).map_err(|_| {
                    OrchestratorError::InvalidPlan("chunk descriptor range exceeds address space")
                })?;
                if materialized_combo_indices.expect("materialized descriptor count is present")
                    [descriptor_start..descriptor_end_index]
                    .iter()
                    .any(|&feature| feature >= self.protocol.n_features)
                {
                    return Err(OrchestratorError::InvalidPlan(
                        "chunk references a feature outside the matrix",
                    ));
                }
            } else if let DescriptorStorage::Generated { source, .. } = &self.descriptors {
                let feature_count = source.features_for_arity(chunk.arity).len() as u64;
                let available = combos::binomial_saturating_u128(feature_count, chunk.arity as u64);
                if u128::from(chunk.combo_count) > available {
                    return Err(OrchestratorError::InvalidPlan(
                        "generated chunk exceeds its combination family",
                    ));
                }
            }
            expected_row_offset = chunk
                .combo_row_offset
                .checked_add(chunk.combo_count)
                .ok_or(OrchestratorError::InvalidPlan(
                    "chunk output row range overflows",
                ))?;
            expected_descriptor_offset = descriptor_end;
        }
        if observed_max_arity != self.protocol.max_arity {
            return Err(OrchestratorError::InvalidPlan(
                "plan max arity does not match its chunks",
            ));
        }
        if let Some(combo_index_count) = combo_index_count {
            if expected_descriptor_offset != combo_index_count {
                return Err(OrchestratorError::InvalidPlan(
                    "combo index buffer contains unplanned descriptors",
                ));
            }
        }
        Ok(())
    }

    fn rebind_protocol_views(&mut self) {
        self.protocol.combo_indices = match &self.descriptors {
            DescriptorStorage::Materialized(combo_indices) => gafime_types::GafimeSliceU32 {
                ptr: combo_indices.as_ptr(),
                len: combo_indices.len() as u64,
            },
            DescriptorStorage::Generated { .. } => Default::default(),
        };
        self.protocol.metric_ids = gafime_types::GafimeSliceU32 {
            ptr: self.metric_ids.as_ptr(),
            len: self.metric_ids.len() as u64,
        };
        self.protocol.chunks = self.chunks.as_ptr();
        self.protocol.chunk_count = self.chunks.len() as u32;
        self.protocol.shape_hints = self.shape_hints.as_ptr();
        self.protocol.shape_hint_count = self.shape_hints.len() as u32;
    }
}

impl Iterator for DescriptorBatchIter {
    type Item = DescriptorBatch;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let source_chunk = *self.source.chunks.get(self.chunk_index)?;
            if self.chunk_rows_emitted == source_chunk.combo_count {
                self.chunk_index += 1;
                self.chunk_rows_emitted = 0;
                self.positions.clear();
                continue;
            }

            let arity = source_chunk.arity as usize;
            let remaining = source_chunk.combo_count - self.chunk_rows_emitted;
            let batch_row_limit = (self.max_descriptor_words / arity).max(1) as u64;
            let batch_count = remaining.min(batch_row_limit);
            let batch_count_usize = usize::try_from(batch_count)
                .expect("descriptor batch row count exceeds the host address space");
            let batch_word_count = batch_count_usize
                .checked_mul(arity)
                .expect("descriptor batch word count overflows");
            let logical_row_offset = source_chunk
                .combo_row_offset
                .checked_add(self.chunk_rows_emitted)
                .expect("validated descriptor row offset overflows");
            let mut combo_indices = Vec::with_capacity(batch_word_count);

            match &self.source.descriptors {
                DescriptorBatchSourceStorage::Materialized(all_combo_indices) => {
                    let descriptor_start = source_chunk
                        .descriptor_offset
                        .checked_add(
                            self.chunk_rows_emitted
                                .checked_mul(source_chunk.arity as u64)
                                .expect("validated descriptor batch offset overflows"),
                        )
                        .expect("validated descriptor batch offset overflows");
                    let descriptor_start = usize::try_from(descriptor_start)
                        .expect("validated descriptor offset exceeds the host address space");
                    let descriptor_end = descriptor_start
                        .checked_add(batch_word_count)
                        .expect("validated descriptor batch range overflows");
                    combo_indices
                        .extend_from_slice(&all_combo_indices[descriptor_start..descriptor_end]);
                }
                DescriptorBatchSourceStorage::Generated(source) => {
                    let features = source.features_for_arity(source_chunk.arity);
                    if self.positions.is_empty() {
                        self.positions.extend(0..arity);
                    }
                    for row in 0..batch_count {
                        combo_indices
                            .extend(self.positions.iter().map(|&position| features[position]));
                        let emitted = self.chunk_rows_emitted + row + 1;
                        if emitted < source_chunk.combo_count {
                            let advanced =
                                advance_combination_positions(&mut self.positions, features.len());
                            debug_assert!(advanced);
                        }
                    }
                }
            }

            self.chunk_rows_emitted += batch_count;
            return Some(DescriptorBatch {
                source_chunk,
                logical_row_offset,
                combo_indices,
            });
        }
    }
}

fn advance_combination_positions(positions: &mut [usize], feature_count: usize) -> bool {
    let arity = positions.len();
    for pivot in (0..arity).rev() {
        if positions[pivot] != pivot + feature_count - arity {
            positions[pivot] += 1;
            for index in pivot + 1..arity {
                positions[index] = positions[index - 1] + 1;
            }
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{GAFIME_BACKEND_CUDA, GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_R2};

    #[test]
    fn single_chunk_rebinds_protocol_views() {
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            128,
            5,
            GAFIME_FAMILY_CONTINUOUS,
            2,
            vec![0, 1, 0, 2],
            vec![GAFIME_METRIC_R2],
        );

        assert_eq!(plan.protocol().combo_indices.len, 4);
        assert_eq!(plan.protocol().metric_ids.len, 1);
        assert_eq!(plan.protocol().chunk_count, 1);
        assert_eq!(plan.protocol().family_count, 1);
        assert_eq!(plan.combo_indices(), &[0, 1, 0, 2]);
        plan.validate().unwrap();
    }

    #[test]
    fn descriptor_batch_limit_must_hold_one_complete_descriptor() {
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            128,
            5,
            GAFIME_FAMILY_CONTINUOUS,
            2,
            vec![0, 1, 0, 2],
            vec![GAFIME_METRIC_R2],
        );

        assert_eq!(
            plan.descriptor_batches(1).unwrap_err(),
            OrchestratorError::InvalidPlan(
                "descriptor batch word limit is smaller than plan arity"
            )
        );
    }

    #[test]
    fn validation_rejects_feature_indices_outside_the_matrix() {
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            128,
            2,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 2],
            vec![GAFIME_METRIC_R2],
        );

        assert_eq!(
            plan.validate(),
            Err(OrchestratorError::InvalidPlan(
                "chunk references a feature outside the matrix"
            ))
        );
    }

    #[test]
    fn validation_rejects_malformed_descriptor_ranges() {
        let mut plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            128,
            3,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_R2],
        );
        plan.chunks[0].descriptor_count = 2;

        assert_eq!(
            plan.validate(),
            Err(OrchestratorError::InvalidPlan(
                "chunk descriptor count does not match combination count"
            ))
        );

        plan.chunks[0].descriptor_count = 3;
        plan.chunks[0].descriptor_offset = u64::MAX;
        assert_eq!(
            plan.validate(),
            Err(OrchestratorError::InvalidPlan(
                "chunk descriptor range is not contiguous"
            ))
        );

        let mut overflow = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            128,
            3,
            GAFIME_FAMILY_CONTINUOUS,
            2,
            vec![0, 1],
            vec![GAFIME_METRIC_R2],
        );
        overflow.chunks[0].combo_count = u64::MAX;
        overflow.chunks[0].descriptor_count = u64::MAX;
        assert_eq!(
            overflow.validate(),
            Err(OrchestratorError::InvalidPlan(
                "chunk descriptor range overflows"
            ))
        );
    }

    #[test]
    fn validation_rejects_invalid_arity_family_and_shape() {
        let make_plan = || {
            CompiledPlan::single_chunk(
                GAFIME_BACKEND_CUDA,
                128,
                3,
                GAFIME_FAMILY_CONTINUOUS,
                2,
                vec![0, 1],
                vec![GAFIME_METRIC_R2],
            )
        };

        let mut arity = make_plan();
        arity.chunks[0].arity = 3;
        assert_eq!(
            arity.validate(),
            Err(OrchestratorError::InvalidPlan(
                "chunk arity exceeds plan max arity"
            ))
        );

        let mut family = make_plan();
        family.chunks[0].family = 999;
        assert_eq!(
            family.validate(),
            Err(OrchestratorError::InvalidPlan(
                "compiled scoring chunks must be continuous"
            ))
        );

        let mut shape = make_plan();
        shape.chunks[0].shape_hint_index = 1;
        assert_eq!(
            shape.validate(),
            Err(OrchestratorError::InvalidPlan(
                "chunk shape hint index is out of range"
            ))
        );
    }

    #[test]
    fn validation_rejects_unplanned_descriptor_tail() {
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            128,
            3,
            GAFIME_FAMILY_CONTINUOUS,
            2,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_R2],
        );

        assert_eq!(
            plan.validate(),
            Err(OrchestratorError::InvalidPlan(
                "combo index buffer contains unplanned descriptors"
            ))
        );
    }

    #[test]
    fn validation_rejects_inconsistent_rows_metrics_and_max_arity() {
        let make_plan = || {
            CompiledPlan::single_chunk(
                GAFIME_BACKEND_CUDA,
                128,
                3,
                GAFIME_FAMILY_CONTINUOUS,
                1,
                vec![0, 1, 2],
                vec![GAFIME_METRIC_R2],
            )
        };

        let mut rows = make_plan();
        rows.chunks[0].combo_row_offset = 1;
        assert_eq!(
            rows.validate(),
            Err(OrchestratorError::InvalidPlan(
                "chunk output row range is not contiguous"
            ))
        );

        let mut metric = make_plan();
        metric.metric_ids[0] = u32::MAX;
        assert_eq!(
            metric.validate(),
            Err(OrchestratorError::InvalidPlan(
                "plan contains an unknown metric"
            ))
        );

        let mut max_arity = make_plan();
        max_arity.protocol.max_arity = 2;
        assert_eq!(
            max_arity.validate(),
            Err(OrchestratorError::InvalidPlan(
                "plan max arity does not match its chunks"
            ))
        );
    }
}
