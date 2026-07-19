//! GPU-Aware Batch Launcher
//!
//! Optimally batches CUDA kernel calls respecting GPU hyperparameters.
//! Uses FFI to call the CUDA DLL directly.

#![allow(dead_code)]

use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

/// GPU configuration for optimal batching
#[derive(Clone, Debug)]
pub struct GpuConfig {
    pub max_blocks: usize,
    pub sm_count: usize,
    pub gpu_name: String,
}

/// Batch of interactions ready for GPU execution
#[derive(Clone, Debug)]
pub struct Batch {
    pub kinds: Vec<i32>,     // [N] candidate kinds
    pub indices: Vec<i32>,   // [N * arity] flattened feature indices
    pub ops: Vec<i32>,       // [N * arity] flattened operators
    pub interact: Vec<i32>,  // [N * (arity - 1)] interaction types
    pub ts_params: Vec<i32>, // [N * 4] time-series parameters
    pub arity: usize,
    pub size: usize,
}

#[derive(Clone, Debug)]
pub struct CandidateDescriptor {
    pub kind: u32,
    pub features: Vec<u32>,
    pub ops: Vec<u32>,
    pub interactions: Vec<u32>,
    pub ts_params: [i32; 4],
}

#[derive(Clone, Debug)]
struct EquationOrderKey {
    original_index: usize,
    features: Vec<u32>,
    template_id: u32,
    anchor: u32,
    anchor_frequency: usize,
    rest: Vec<u32>,
}

impl Batch {
    pub fn new(candidates: &[CandidateDescriptor], arity: usize) -> Self {
        let size = candidates.len();
        let mut kinds = Vec::with_capacity(size);
        let mut indices = Vec::with_capacity(size * arity);
        let mut ops = Vec::with_capacity(size * arity);
        let interact_width = arity.saturating_sub(1).max(1);
        let mut interact = Vec::with_capacity(size * interact_width);
        let mut ts_params = Vec::with_capacity(size * 4);

        for candidate in candidates {
            kinds.push(candidate.kind as i32);
            indices.extend(candidate.features.iter().map(|&value| value as i32));
            ops.extend(candidate.ops.iter().map(|&value| value as i32));
            interact.extend(candidate.interactions.iter().map(|&value| value as i32));
            ts_params.extend(candidate.ts_params);
        }

        Self {
            kinds,
            indices,
            ops,
            interact,
            ts_params,
            arity,
            size,
        }
    }
}

/// GPU-Aware Batch Scheduler
///
/// Schedules interactions into optimal batches based on GPU configuration.
pub struct BatchScheduler {
    /// Maximum blocks per kernel launch
    max_blocks: usize,
    /// Optimal batch size (multiple of max_blocks)
    optimal_batch: usize,
    /// Path to CUDA DLL
    cuda_dll_path: PathBuf,
}

impl BatchScheduler {
    pub fn new(max_blocks: usize, cuda_dll_path: PathBuf) -> Self {
        // Optimal batch is max_blocks or smaller for memory efficiency
        // We cap at 1024 (CUDA kernel limitation)
        let optimal_batch = max_blocks.min(1024);

        Self {
            max_blocks,
            optimal_batch,
            cuda_dll_path,
        }
    }

    /// Get optimal batch sizes based on GPU config
    pub fn get_optimal_batch_sizes(&self) -> Vec<usize> {
        // Return batch sizes that are multiples of max_blocks
        // Limited by CUDA kernel max of 1024
        vec![
            self.max_blocks,
            (self.max_blocks * 2).min(1024),
            (self.max_blocks * 4).min(1024),
            1024,
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
    }

    /// Schedule interactions into cache-local optimal batches.
    ///
    /// The input equations may arrive in generation order, which is often close
    /// to random from the GPU cache's point of view. Reorder first so adjacent
    /// descriptors reuse the same hot feature columns, then chunk into launch
    /// batches. Equation internals are not rewritten; only launch order changes.
    pub fn schedule(&self, candidates: &[CandidateDescriptor]) -> Vec<Batch> {
        let mut batches = Vec::new();
        let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, candidate) in candidates.iter().enumerate() {
            grouped
                .entry(candidate.features.len())
                .or_default()
                .push(index);
        }

        for (arity, indices) in grouped {
            let local_candidates: Vec<CandidateDescriptor> = indices
                .iter()
                .map(|&index| candidates[index].clone())
                .collect();
            let order = self.order_candidates_cache_aware(&local_candidates);
            let ordered: Vec<CandidateDescriptor> = order
                .into_iter()
                .map(|local_index| local_candidates[local_index].clone())
                .collect();

            let mut start = 0usize;
            while start < ordered.len() {
                let end = (start + self.optimal_batch).min(ordered.len());
                batches.push(Batch::new(&ordered[start..end], arity));
                start = end;
            }
        }
        batches
    }

    pub fn order_candidates_cache_aware(&self, candidates: &[CandidateDescriptor]) -> Vec<usize> {
        let feature_sets: Vec<Vec<u32>> = candidates
            .iter()
            .map(|item| item.features.clone())
            .collect();
        let template_ids: Vec<u32> = candidates
            .iter()
            .map(|item| {
                item.kind
                    .saturating_mul(1_000_000)
                    .saturating_add(item.features.len() as u32)
            })
            .collect();

        self.order_equations_cache_aware(&feature_sets, Some(&template_ids))
    }

    pub fn order_equations_cache_aware(
        &self,
        feature_sets: &[Vec<u32>],
        template_ids: Option<&[u32]>,
    ) -> Vec<usize> {
        let keys = build_equation_order_keys(feature_sets, template_ids);
        keys.into_iter().map(|key| key.original_index).collect()
    }

    pub fn schedule_equation_indices(
        &self,
        feature_sets: &[Vec<u32>],
        template_ids: Option<&[u32]>,
    ) -> Vec<Vec<usize>> {
        let order = self.order_equations_cache_aware(feature_sets, template_ids);
        order
            .chunks(self.optimal_batch)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    pub fn schedule_template_equation_indices(
        &self,
        feature_sets: &[Vec<u32>],
        template_ids: &[u32],
    ) -> Vec<(u32, Vec<usize>)> {
        let mut grouped: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (index, &template_id) in template_ids.iter().enumerate() {
            grouped.entry(template_id).or_default().push(index);
        }

        let mut batches = Vec::new();
        for (template_id, indices) in grouped {
            let local_feature_sets: Vec<Vec<u32>> = indices
                .iter()
                .map(|&index| feature_sets[index].clone())
                .collect();
            let local_order = self.order_equations_cache_aware(&local_feature_sets, None);
            let ordered_indices: Vec<usize> = local_order
                .into_iter()
                .map(|local_index| indices[local_index])
                .collect();
            for chunk in ordered_indices.chunks(self.optimal_batch) {
                batches.push((template_id, chunk.to_vec()));
            }
        }

        batches
    }

    /// Get the optimal batch size for a given number of interactions
    pub fn get_optimal_size(&self, n_interactions: usize) -> usize {
        if n_interactions <= self.optimal_batch {
            n_interactions
        } else {
            self.optimal_batch
        }
    }
}

fn build_equation_order_keys(
    feature_sets: &[Vec<u32>],
    template_ids: Option<&[u32]>,
) -> Vec<EquationOrderKey> {
    let canonical_features: Vec<Vec<u32>> = feature_sets
        .iter()
        .map(|features| unique_sorted_features(features))
        .collect();

    let mut frequencies: HashMap<u32, usize> = HashMap::new();
    for features in &canonical_features {
        for &feature in features {
            *frequencies.entry(feature).or_insert(0) += 1;
        }
    }

    let mut keys: Vec<EquationOrderKey> = canonical_features
        .into_iter()
        .enumerate()
        .map(|(original_index, features)| {
            let (anchor, anchor_frequency) = choose_anchor(&features, &frequencies);
            let mut rest: Vec<u32> = features
                .iter()
                .copied()
                .filter(|&feature| feature != anchor)
                .collect();
            rest.sort_by(|a, b| compare_feature_by_frequency(*a, *b, &frequencies));
            EquationOrderKey {
                original_index,
                features,
                template_id: template_ids
                    .and_then(|ids| ids.get(original_index))
                    .copied()
                    .unwrap_or(0),
                anchor,
                anchor_frequency,
                rest,
            }
        })
        .collect();

    keys.sort_by(compare_equation_order_key);
    keys
}

fn unique_sorted_features(features: &[u32]) -> Vec<u32> {
    let mut out = features.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn choose_anchor(features: &[u32], frequencies: &HashMap<u32, usize>) -> (u32, usize) {
    features
        .iter()
        .copied()
        .map(|feature| (feature, *frequencies.get(&feature).unwrap_or(&0)))
        .max_by(|(feature_a, freq_a), (feature_b, freq_b)| {
            freq_a.cmp(freq_b).then_with(|| feature_b.cmp(feature_a))
        })
        .unwrap_or((0, 0))
}

fn compare_feature_by_frequency(a: u32, b: u32, frequencies: &HashMap<u32, usize>) -> Ordering {
    let freq_a = *frequencies.get(&a).unwrap_or(&0);
    let freq_b = *frequencies.get(&b).unwrap_or(&0);
    freq_b.cmp(&freq_a).then_with(|| a.cmp(&b))
}

fn compare_equation_order_key(a: &EquationOrderKey, b: &EquationOrderKey) -> Ordering {
    b.anchor_frequency
        .cmp(&a.anchor_frequency)
        .then_with(|| a.anchor.cmp(&b.anchor))
        .then_with(|| a.features.len().cmp(&b.features.len()))
        .then_with(|| a.rest.cmp(&b.rest))
        .then_with(|| a.features.cmp(&b.features))
        .then_with(|| a.template_id.cmp(&b.template_id))
        .then_with(|| a.original_index.cmp(&b.original_index))
}

// ============================================================================
// Python Bindings
// ============================================================================

#[pyclass(name = "BatchScheduler")]
pub struct PyBatchScheduler {
    inner: BatchScheduler,
}

#[pymethods]
impl PyBatchScheduler {
    /// Create a new BatchScheduler
    ///
    /// Args:
    ///     max_blocks: Maximum blocks for GPU (from GPU config)
    ///     cuda_dll_path: Path to gafime_cuda.dll
    #[new]
    #[pyo3(signature = (max_blocks=96, cuda_dll_path=None))]
    fn new(max_blocks: usize, cuda_dll_path: Option<String>) -> Self {
        let path = cuda_dll_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("gafime_cuda.dll"));

        Self {
            inner: BatchScheduler::new(max_blocks, path),
        }
    }

    /// Get optimal batch sizes for this GPU
    fn get_optimal_batch_sizes(&self) -> Vec<usize> {
        self.inner.get_optimal_batch_sizes()
    }

    /// Get the optimal batch size
    fn optimal_batch_size(&self) -> usize {
        self.inner.optimal_batch
    }

    /// Get max blocks
    fn max_blocks(&self) -> usize {
        self.inner.max_blocks
    }

    /// Create homogeneous-arity descriptor batches for the CUDA arity-template ABI.
    #[pyo3(signature = (candidate_kinds, feature_sets, op_sets, interaction_sets, ts_params=None))]
    fn create_batches(
        &self,
        candidate_kinds: Vec<u32>,
        feature_sets: Vec<Vec<u32>>,
        op_sets: Vec<Vec<u32>>,
        interaction_sets: Vec<Vec<u32>>,
        ts_params: Option<Vec<Vec<i32>>>,
    ) -> PyResult<
        Vec<(
            Vec<i32>,
            Vec<i32>,
            Vec<i32>,
            Vec<i32>,
            Vec<i32>,
            usize,
            usize,
        )>,
    > {
        let n = feature_sets.len();
        if candidate_kinds.len() != n || op_sets.len() != n || interaction_sets.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All inputs must have same length",
            ));
        }
        if let Some(params) = ts_params.as_ref() {
            if params.len() != n {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "ts_params must have same length as feature_sets",
                ));
            }
        }
        let mut candidates = Vec::with_capacity(n);
        for idx in 0..n {
            let arity = feature_sets[idx].len();
            if !(1..=5).contains(&arity) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "feature arity must be in [1, 5]",
                ));
            }
            if op_sets[idx].len() != arity
                || interaction_sets[idx].len() != arity.saturating_sub(1).max(1)
            {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "op/interactions lengths must match arity",
                ));
            }
            let mut params = [0i32; 4];
            if let Some(all_params) = ts_params.as_ref() {
                for (slot, value) in all_params[idx].iter().take(4).enumerate() {
                    params[slot] = *value;
                }
            }
            candidates.push(CandidateDescriptor {
                kind: candidate_kinds[idx],
                features: feature_sets[idx].clone(),
                ops: op_sets[idx].clone(),
                interactions: interaction_sets[idx].clone(),
                ts_params: params,
            });
        }
        let batches = self.inner.schedule(&candidates);
        Ok(batches
            .into_iter()
            .map(|b| {
                (
                    b.kinds,
                    b.indices,
                    b.ops,
                    b.interact,
                    b.ts_params,
                    b.arity,
                    b.size,
                )
            })
            .collect())
    }

    /// Return cache-locality-aware launch order for arbitrary equation
    /// templates. Each feature set is the full feature footprint of one
    /// equation; template IDs can encode operator/family parameters.
    #[pyo3(signature = (feature_sets, template_ids=None))]
    fn order_equations(
        &self,
        feature_sets: Vec<Vec<u32>>,
        template_ids: Option<Vec<u32>>,
    ) -> PyResult<Vec<usize>> {
        if let Some(ids) = template_ids.as_ref() {
            if ids.len() != feature_sets.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "template_ids must have the same length as feature_sets",
                ));
            }
        }

        Ok(self
            .inner
            .order_equations_cache_aware(&feature_sets, template_ids.as_deref()))
    }

    /// Return cache-locality-aware batches of original equation indices.
    #[pyo3(signature = (feature_sets, template_ids=None))]
    fn create_equation_batches(
        &self,
        feature_sets: Vec<Vec<u32>>,
        template_ids: Option<Vec<u32>>,
    ) -> PyResult<Vec<Vec<usize>>> {
        if let Some(ids) = template_ids.as_ref() {
            if ids.len() != feature_sets.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "template_ids must have the same length as feature_sets",
                ));
            }
        }

        Ok(self
            .inner
            .schedule_equation_indices(&feature_sets, template_ids.as_deref()))
    }

    /// Return cache-locality-aware batches grouped by a single execution
    /// template per batch. Template IDs represent static kernel shapes such as
    /// MI histogram capacity; mixed-template batches are intentionally not
    /// produced.
    fn create_template_batches(
        &self,
        feature_sets: Vec<Vec<u32>>,
        template_ids: Vec<u32>,
    ) -> PyResult<Vec<(u32, Vec<usize>)>> {
        if template_ids.len() != feature_sets.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "template_ids must have the same length as feature_sets",
            ));
        }

        Ok(self
            .inner
            .schedule_template_equation_indices(&feature_sets, &template_ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_scheduler() {
        let scheduler = BatchScheduler::new(96, PathBuf::from("test.dll"));

        assert_eq!(scheduler.optimal_batch, 96);

        let candidates: Vec<CandidateDescriptor> = (0..200)
            .map(|i| CandidateDescriptor {
                kind: 0,
                features: vec![0, 1],
                ops: vec![i % 5, 0],
                interactions: vec![0],
                ts_params: [0, 0, 0, 0],
            })
            .collect();

        let batches = scheduler.schedule(&candidates);

        // Should create 3 batches: 96, 96, 8
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].size, 96);
        assert_eq!(batches[1].size, 96);
        assert_eq!(batches[2].size, 8);
    }

    #[test]
    fn test_optimal_sizes() {
        let scheduler = BatchScheduler::new(96, PathBuf::from("test.dll"));
        let sizes = scheduler.get_optimal_batch_sizes();

        assert!(sizes.contains(&96));
        assert!(sizes.iter().all(|&s| s <= 1024));
    }

    #[test]
    fn test_cache_aware_order_groups_hot_feature() {
        let scheduler = BatchScheduler::new(1024, PathBuf::from("test.dll"));
        let candidates = vec![
            CandidateDescriptor {
                kind: 0,
                features: vec![8, 9],
                ops: vec![0, 0],
                interactions: vec![0],
                ts_params: [0, 0, 0, 0],
            },
            CandidateDescriptor {
                kind: 0,
                features: vec![1, 2],
                ops: vec![0, 0],
                interactions: vec![0],
                ts_params: [0, 0, 0, 0],
            },
            CandidateDescriptor {
                kind: 0,
                features: vec![5, 6],
                ops: vec![0, 0],
                interactions: vec![0],
                ts_params: [0, 0, 0, 0],
            },
            CandidateDescriptor {
                kind: 0,
                features: vec![1, 3],
                ops: vec![0, 0],
                interactions: vec![0],
                ts_params: [0, 0, 0, 0],
            },
            CandidateDescriptor {
                kind: 0,
                features: vec![1, 4],
                ops: vec![0, 0],
                interactions: vec![0],
                ts_params: [0, 0, 0, 0],
            },
        ];

        let order = scheduler.order_candidates_cache_aware(&candidates);
        assert_eq!(order.len(), candidates.len());
        assert!(order[..3]
            .iter()
            .all(|&idx| { candidates[idx].features.contains(&1) }));
    }

    #[test]
    fn test_equation_batches_return_original_indices_once() {
        let scheduler = BatchScheduler::new(3, PathBuf::from("test.dll"));
        let feature_sets = vec![vec![10, 11], vec![2], vec![2, 3], vec![2, 4], vec![8, 9]];
        let batches = scheduler.schedule_equation_indices(&feature_sets, None);
        let mut flattened: Vec<usize> = batches.into_iter().flatten().collect();
        flattened.sort_unstable();
        assert_eq!(flattened, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_template_equation_batches_are_homogeneous() {
        let scheduler = BatchScheduler::new(2, PathBuf::from("test.dll"));
        let feature_sets = vec![vec![0, 1], vec![0, 2], vec![5, 6], vec![0, 3], vec![5, 7]];
        let template_ids = vec![32, 64, 32, 64, 32];
        let batches = scheduler.schedule_template_equation_indices(&feature_sets, &template_ids);

        let mut flattened = Vec::new();
        for (template_id, indices) in batches {
            assert!(indices.len() <= 2);
            assert!(indices
                .iter()
                .all(|&index| template_ids[index] == template_id));
            flattened.extend(indices);
        }
        flattened.sort_unstable();
        assert_eq!(flattened, vec![0, 1, 2, 3, 4]);
    }
}
