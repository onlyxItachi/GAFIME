//! GPU-Aware Batch Launcher
//!
//! Optimally batches CUDA kernel calls respecting GPU hyperparameters.
//! Uses FFI to call the CUDA DLL directly.

use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

/// Interaction specification for batching
#[derive(Clone, Debug)]
pub struct Interaction {
    pub feature_a: u32,
    pub feature_b: u32,
    pub op_a: u32,
    pub op_b: u32,
    pub interaction_type: u32,
}

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
    pub indices: Vec<i32>,  // [N * 2] flattened feature indices
    pub ops: Vec<i32>,      // [N * 2] flattened operators
    pub interact: Vec<i32>, // [N] interaction types
    pub size: usize,
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
    pub fn new(interactions: &[Interaction]) -> Self {
        let size = interactions.len();
        let mut indices = Vec::with_capacity(size * 2);
        let mut ops = Vec::with_capacity(size * 2);
        let mut interact = Vec::with_capacity(size);

        for i in interactions {
            indices.push(i.feature_a as i32);
            indices.push(i.feature_b as i32);
            ops.push(i.op_a as i32);
            ops.push(i.op_b as i32);
            interact.push(i.interaction_type as i32);
        }

        Self {
            indices,
            ops,
            interact,
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
    pub fn schedule(&self, interactions: &[Interaction]) -> Vec<Batch> {
        let order = self.order_interactions_cache_aware(interactions);
        let ordered: Vec<Interaction> = order
            .into_iter()
            .map(|idx| interactions[idx].clone())
            .collect();

        ordered.chunks(self.optimal_batch).map(Batch::new).collect()
    }

    pub fn order_interactions_cache_aware(&self, interactions: &[Interaction]) -> Vec<usize> {
        let feature_sets: Vec<Vec<u32>> = interactions
            .iter()
            .map(|item| vec![item.feature_a, item.feature_b])
            .collect();
        let template_ids: Vec<u32> = interactions
            .iter()
            .map(|item| {
                // Keep all equations for the same feature pair together, while
                // still separating operator/interact templates deterministically.
                item.interaction_type
                    .saturating_mul(1_000_000)
                    .saturating_add(item.op_a.saturating_mul(1_000))
                    .saturating_add(item.op_b)
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

#[pyclass(name = "Interaction")]
#[derive(Clone)]
pub struct PyInteraction {
    inner: Interaction,
}

#[pymethods]
impl PyInteraction {
    #[new]
    fn new(feature_a: u32, feature_b: u32, op_a: u32, op_b: u32, interaction_type: u32) -> Self {
        Self {
            inner: Interaction {
                feature_a,
                feature_b,
                op_a,
                op_b,
                interaction_type,
            },
        }
    }
}

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

    /// Create optimally-sized batches from feature pairs
    ///
    /// Args:
    ///     feature_pairs: List of (f0, f1) tuples
    ///     op_pairs: List of (op0, op1) tuples
    ///     interactions: List of interaction types
    ///
    /// Returns:
    ///     List of (indices, ops, interact, size) tuples ready for GPU
    fn create_batches(
        &self,
        feature_pairs: Vec<(u32, u32)>,
        op_pairs: Vec<(u32, u32)>,
        interactions: Vec<u32>,
    ) -> PyResult<Vec<(Vec<i32>, Vec<i32>, Vec<i32>, usize)>> {
        if feature_pairs.len() != op_pairs.len() || feature_pairs.len() != interactions.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All inputs must have same length",
            ));
        }

        // Convert to Interaction structs
        let all_interactions: Vec<Interaction> = feature_pairs
            .iter()
            .zip(op_pairs.iter())
            .zip(interactions.iter())
            .map(|((&(fa, fb), &(oa, ob)), &it)| Interaction {
                feature_a: fa,
                feature_b: fb,
                op_a: oa,
                op_b: ob,
                interaction_type: it,
            })
            .collect();

        // Schedule into optimal batches
        let batches = self.inner.schedule(&all_interactions);

        // Convert to Python-friendly format
        Ok(batches
            .into_iter()
            .map(|b| (b.indices, b.ops, b.interact, b.size))
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
    fn create_template_equation_batches(
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

    /// Generate all pairwise combinations for given features and ops
    ///
    /// Useful for exhaustive feature search
    fn generate_all_pairs(
        &self,
        n_features: usize,
        ops: Vec<u32>,
        interaction_type: u32,
    ) -> Vec<(Vec<i32>, Vec<i32>, Vec<i32>, usize)> {
        let mut all_interactions = Vec::new();

        // Generate all (feature_i, feature_j, op_a, op_b) combinations
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                for &op_a in &ops {
                    for &op_b in &ops {
                        all_interactions.push(Interaction {
                            feature_a: i as u32,
                            feature_b: j as u32,
                            op_a,
                            op_b,
                            interaction_type,
                        });
                    }
                }
            }
        }

        // Schedule into optimal batches
        let batches = self.inner.schedule(&all_interactions);

        batches
            .into_iter()
            .map(|b| (b.indices, b.ops, b.interact, b.size))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_scheduler() {
        let scheduler = BatchScheduler::new(96, PathBuf::from("test.dll"));

        assert_eq!(scheduler.optimal_batch, 96);

        // Create 200 interactions
        let interactions: Vec<Interaction> = (0..200)
            .map(|i| Interaction {
                feature_a: 0,
                feature_b: 1,
                op_a: i % 5,
                op_b: 0,
                interaction_type: 0,
            })
            .collect();

        let batches = scheduler.schedule(&interactions);

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
        let interactions = vec![
            Interaction {
                feature_a: 8,
                feature_b: 9,
                op_a: 0,
                op_b: 0,
                interaction_type: 0,
            },
            Interaction {
                feature_a: 1,
                feature_b: 2,
                op_a: 0,
                op_b: 0,
                interaction_type: 0,
            },
            Interaction {
                feature_a: 5,
                feature_b: 6,
                op_a: 0,
                op_b: 0,
                interaction_type: 0,
            },
            Interaction {
                feature_a: 1,
                feature_b: 3,
                op_a: 0,
                op_b: 0,
                interaction_type: 0,
            },
            Interaction {
                feature_a: 1,
                feature_b: 4,
                op_a: 0,
                op_b: 0,
                interaction_type: 0,
            },
        ];

        let order = scheduler.order_interactions_cache_aware(&interactions);
        assert_eq!(order.len(), interactions.len());
        assert!(order[..3]
            .iter()
            .all(|&idx| { interactions[idx].feature_a == 1 || interactions[idx].feature_b == 1 }));
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
