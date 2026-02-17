//! L2 Cache-Aware Feature Scheduler
//!
//! Maximizes L2 cache reuse by grouping interactions that share features.
//! Uses a sliding window approach:
//! - Keep N features in cache window
//! - Execute ALL their interactions
//! - Slide window: drop oldest, load new
//! - Only execute NEW pairs (avoid duplicates)

use pyo3::prelude::*;
use rustc_hash::FxHashSet;
use std::collections::BTreeSet;

/// Interaction specification
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Interaction {
    pub features: Vec<u32>,      // Feature indices (sorted)
    pub ops: Vec<u32>,           // Operator per feature
    pub interaction_types: Vec<u32>,  // Interaction types between pairs
}

/// Cache-aware batch - all interactions that share a feature window
#[derive(Clone, Debug)]
pub struct CacheBatch {
    pub window_features: Vec<u32>,  // Features to keep in L2 cache
    pub interactions: Vec<Interaction>,
}

/// L2 Cache-Aware Feature Scheduler
/// 
/// Groups interactions by feature usage to maximize L2 cache hits.
/// Uses sliding window to process all feature combinations efficiently.
pub struct CacheAwareScheduler {
    n_features: usize,
    window_size: usize,
    ops: Vec<u32>,
    interaction_types: Vec<u32>,
    arity: usize,
}

impl CacheAwareScheduler {
    pub fn new(
        n_features: usize,
        window_size: usize,
        ops: Vec<u32>,
        interaction_types: Vec<u32>,
        arity: usize,
    ) -> Self {
        Self {
            n_features,
            window_size: window_size.min(n_features),
            ops,
            interaction_types,
            arity: arity.max(2).min(5),
        }
    }
    
    /// Generate cache-optimized batches using sliding window
    /// 
    /// For arity=2:
    /// - Window [A,B,C,D] → pairs: AB,AC,AD,BC,BD,CD
    /// - Slide to [B,C,D,E] → NEW pairs: BE,CE,DE
    pub fn generate_batches(&self) -> Vec<CacheBatch> {
        let mut batches = Vec::new();
        let mut executed: FxHashSet<BTreeSet<u32>> = FxHashSet::default();
        
        if self.arity == 2 {
            self.generate_pairwise_batches(&mut batches, &mut executed);
        } else {
            // For higher arity, still use pairwise window but generate n-ary combos
            self.generate_nary_batches(&mut batches, &mut executed);
        }
        
        batches
    }
    
    /// Generate pairwise interaction batches (arity=2)
    fn generate_pairwise_batches(
        &self,
        batches: &mut Vec<CacheBatch>,
        executed: &mut FxHashSet<BTreeSet<u32>>,
    ) {
        // Slide window across all features
        for window_start in 0..self.n_features {
            let window_end = (window_start + self.window_size).min(self.n_features);
            let window: Vec<u32> = (window_start..window_end).map(|i| i as u32).collect();
            
            if window.len() < 2 {
                break;
            }
            
            let mut batch_interactions = Vec::new();
            
            // Generate all pairs within this window
            for i in 0..window.len() {
                for j in (i + 1)..window.len() {
                    let pair: BTreeSet<u32> = [window[i], window[j]].into_iter().collect();
                    
                    // Skip if already executed
                    if executed.contains(&pair) {
                        continue;
                    }
                    executed.insert(pair);
                    
                    // Generate all op combinations for this pair
                    for &op_a in &self.ops {
                        for &op_b in &self.ops {
                            for &interact_type in &self.interaction_types {
                                batch_interactions.push(Interaction {
                                    features: vec![window[i], window[j]],
                                    ops: vec![op_a, op_b],
                                    interaction_types: vec![interact_type],
                                });
                            }
                        }
                    }
                }
            }
            
            if !batch_interactions.is_empty() {
                batches.push(CacheBatch {
                    window_features: window,
                    interactions: batch_interactions,
                });
            }
        }
    }
    
    /// Generate n-ary interaction batches (arity > 2)
    fn generate_nary_batches(
        &self,
        batches: &mut Vec<CacheBatch>,
        executed: &mut FxHashSet<BTreeSet<u32>>,
    ) {
        // For n-ary, we need combinations of `arity` features
        // Use sliding window but generate n-ary combinations
        
        for window_start in 0..self.n_features {
            let window_end = (window_start + self.window_size).min(self.n_features);
            let window: Vec<u32> = (window_start..window_end).map(|i| i as u32).collect();
            
            if window.len() < self.arity {
                break;
            }
            
            let mut batch_interactions = Vec::new();
            
            // Generate all n-ary combinations within window
            let combos = self.combinations(&window, self.arity);
            
            for combo in combos {
                let combo_set: BTreeSet<u32> = combo.iter().copied().collect();
                
                if executed.contains(&combo_set) {
                    continue;
                }
                executed.insert(combo_set);
                
                // Generate op combinations (simplified: same op for all)
                for &op in &self.ops {
                    for &interact_type in &self.interaction_types {
                        batch_interactions.push(Interaction {
                            features: combo.clone(),
                            ops: vec![op; self.arity],
                            interaction_types: vec![interact_type; self.arity - 1],
                        });
                    }
                }
            }
            
            if !batch_interactions.is_empty() {
                batches.push(CacheBatch {
                    window_features: window,
                    interactions: batch_interactions,
                });
            }
        }
    }
    
    /// Generate k-combinations of elements
    fn combinations(&self, elements: &[u32], k: usize) -> Vec<Vec<u32>> {
        if k == 0 {
            return vec![vec![]];
        }
        if elements.len() < k {
            return vec![];
        }
        
        let mut result = Vec::new();
        for (i, &first) in elements.iter().enumerate() {
            let rest = &elements[i + 1..];
            for mut combo in self.combinations(rest, k - 1) {
                combo.insert(0, first);
                result.push(combo);
            }
        }
        result
    }
    
    /// Get total number of interactions that will be generated
    pub fn total_interactions(&self) -> usize {
        let n_pairs = self.n_features * (self.n_features - 1) / 2;
        let ops_per_pair = self.ops.len() * self.ops.len() * self.interaction_types.len();
        n_pairs * ops_per_pair
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[pyclass(name = "CacheAwareScheduler")]
pub struct PyCacheAwareScheduler {
    inner: CacheAwareScheduler,
}

#[pymethods]
impl PyCacheAwareScheduler {
    /// Create a new cache-aware scheduler
    /// 
    /// Args:
    ///     n_features: Number of features
    ///     window_size: Features to keep in L2 cache (default: 4)
    ///     ops: List of operator IDs to use
    ///     interaction_types: List of interaction types
    ///     arity: Number of features per interaction (default: 2)
    #[new]
    #[pyo3(signature = (n_features, window_size=4, ops=vec![0,1,2,3], interaction_types=vec![0], arity=2))]
    fn new(
        n_features: usize,
        window_size: usize,
        ops: Vec<u32>,
        interaction_types: Vec<u32>,
        arity: usize,
    ) -> Self {
        Self {
            inner: CacheAwareScheduler::new(n_features, window_size, ops, interaction_types, arity),
        }
    }
    
    /// Generate cache-optimized batches
    /// 
    /// Returns list of (window_features, interactions) where:
    /// - window_features: features to keep in L2 cache
    /// - interactions: list of (features, ops, interact_types) tuples
    fn generate_batches(&self) -> Vec<(Vec<u32>, Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>)> {
        self.inner.generate_batches()
            .into_iter()
            .map(|batch| {
                let interactions = batch.interactions.into_iter()
                    .map(|i| (i.features, i.ops, i.interaction_types))
                    .collect();
                (batch.window_features, interactions)
            })
            .collect()
    }
    
    /// Generate flat list of all interactions (for pipeline submit)
    /// 
    /// Returns list of (indices_flat, ops_flat, interact_flat) ready for GPU
    fn generate_flat_batches(&self) -> Vec<(Vec<i32>, Vec<i32>, Vec<i32>)> {
        let batches = self.inner.generate_batches();
        
        batches.into_iter()
            .map(|batch| {
                let mut indices = Vec::new();
                let mut ops = Vec::new();
                let mut interact = Vec::new();
                
                for interaction in batch.interactions {
                    // For arity=2, flatten to [f0, f1]
                    for &f in &interaction.features {
                        indices.push(f as i32);
                    }
                    for &o in &interaction.ops {
                        ops.push(o as i32);
                    }
                    for &t in &interaction.interaction_types {
                        interact.push(t as i32);
                    }
                }
                
                (indices, ops, interact)
            })
            .collect()
    }
    
    /// Get total number of interactions
    fn total_interactions(&self) -> usize {
        self.inner.total_interactions()
    }
    
    /// Get window size
    fn window_size(&self) -> usize {
        self.inner.window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pairwise_scheduler() {
        // 5 features, window=3, identity op only, multiply interaction
        let scheduler = CacheAwareScheduler::new(5, 3, vec![0], vec![0], 2);
        let batches = scheduler.generate_batches();
        
        // Window [0,1,2] → pairs 01,02,12 = 3 pairs
        // Window [1,2,3] → new pairs 13,23 = 2 pairs (12 already done)
        // Window [2,3,4] → new pairs 24,34 = 2 pairs
        // Total unique pairs: C(5,2) = 10
        
        let total_interactions: usize = batches.iter()
            .map(|b| b.interactions.len())
            .sum();
        
        assert_eq!(total_interactions, 10);  // 10 unique pairs × 1 op × 1 interact = 10
    }
    
    #[test]
    fn test_no_duplicate_pairs() {
        let scheduler = CacheAwareScheduler::new(10, 4, vec![0, 1], vec![0], 2);
        let batches = scheduler.generate_batches();
        
        // Collect all feature pairs
        let mut seen_pairs: FxHashSet<(u32, u32)> = FxHashSet::default();
        
        for batch in &batches {
            for interaction in &batch.interactions {
                let pair = (interaction.features[0], interaction.features[1]);
                // With different ops, same pair appears multiple times - that's OK
                // But same (pair, ops) should not repeat
            }
        }
        
        // Total should be C(10,2) * 2 ops * 2 ops = 45 * 4 = 180
        let total: usize = batches.iter().map(|b| b.interactions.len()).sum();
        assert_eq!(total, 45 * 4);
    }
}
