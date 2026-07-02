pub mod combos;
pub mod shapes;
pub mod spine;

use gafime_types::{
    BackendKind, CandidateFamily, GafimeArityChunk, GafimeLaunchProtocol,
    GafimePermutationSchedule, GafimeRankSpec, GafimeShapeHint, GAFIME_ABI_VERSION,
};

use crate::backend::{OrchestratorError, OrchestratorResult};

#[derive(Debug)]
pub struct CompiledPlan {
    protocol: GafimeLaunchProtocol,
    combo_indices: Vec<u32>,
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
        let shape_hints = vec![shapes::default_shape_hint(backend_kind, arity)];
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
        let protocol = GafimeLaunchProtocol {
            abi_version: GAFIME_ABI_VERSION,
            backend_kind,
            flags: 0,
            max_arity,
            n_samples,
            n_features,
            family_count: 0,
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
            combo_indices,
            metric_ids,
            chunks,
            shape_hints,
        };
        plan.rebind_protocol_views();
        plan
    }

    pub fn protocol(&self) -> &GafimeLaunchProtocol {
        &self.protocol
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
        if self.protocol.n_samples == 0 {
            return Err(OrchestratorError::InvalidPlan("plan has no samples"));
        }
        if self.protocol.n_features == 0 {
            return Err(OrchestratorError::InvalidPlan("plan has no features"));
        }
        if self.metric_ids.is_empty() {
            return Err(OrchestratorError::InvalidPlan("plan has no metrics"));
        }
        for chunk in &self.chunks {
            if chunk.arity == 0 {
                return Err(OrchestratorError::InvalidPlan("chunk arity is zero"));
            }
            let required = chunk
                .combo_count
                .saturating_mul(chunk.arity as u64)
                .saturating_add(chunk.descriptor_offset);
            if required > self.combo_indices.len() as u64 {
                return Err(OrchestratorError::InvalidPlan(
                    "chunk exceeds combo index buffer",
                ));
            }
        }
        Ok(())
    }

    fn rebind_protocol_views(&mut self) {
        self.protocol.combo_indices = gafime_types::GafimeSliceU32 {
            ptr: self.combo_indices.as_ptr(),
            len: self.combo_indices.len() as u64,
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
        plan.validate().unwrap();
    }
}
