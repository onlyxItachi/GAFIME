//! GPU-Aware Batch Launcher
//! 
//! Optimally batches CUDA kernel calls respecting GPU hyperparameters.
//! Uses FFI to call the CUDA DLL directly.

use pyo3::prelude::*;
use rayon::prelude::*;
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
    pub indices: Vec<i32>,      // [N * 2] flattened feature indices
    pub ops: Vec<i32>,          // [N * 2] flattened operators
    pub interact: Vec<i32>,     // [N] interaction types
    pub size: usize,
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
        
        Self { indices, ops, interact, size }
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
        ].into_iter().collect::<std::collections::BTreeSet<_>>().into_iter().collect()
    }
    
    /// Schedule interactions into optimal batches
    pub fn schedule(&self, interactions: &[Interaction]) -> Vec<Batch> {
        interactions
            .chunks(self.optimal_batch)
            .map(Batch::new)
            .collect()
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
            inner: Interaction { feature_a, feature_b, op_a, op_b, interaction_type }
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
                "All inputs must have same length"
            ));
        }
        
        // Convert to Interaction structs
        let all_interactions: Vec<Interaction> = feature_pairs.iter()
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
        Ok(batches.into_iter()
            .map(|b| (b.indices, b.ops, b.interact, b.size))
            .collect())
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
            for j in (i+1)..n_features {
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
        
        batches.into_iter()
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
}
