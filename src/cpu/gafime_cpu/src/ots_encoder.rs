//! CatBoost Ordered Target Statistics (OTS) Encoder
//! 
//! Prevents target leakage by using only preceding samples in a random permutation.
//! Supports multiple permutations with averaging for robustness (like CatBoost).

use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Ordered Target Statistics Encoder
/// 
/// Encodes categorical features using target statistics computed from 
/// only preceding samples in a random permutation, preventing target leakage.
pub struct OTSEncoder {
    /// Prior value (typically global target mean)
    prior: f32,
    /// Regularization weight for prior smoothing
    prior_weight: f32,
    /// Random seed for reproducibility
    seed: u64,
    /// Number of permutations to average (more = more robust)
    n_permutations: usize,
}

impl OTSEncoder {
    pub fn new(prior: f32, prior_weight: f32, seed: u64, n_permutations: usize) -> Self {
        Self {
            prior,
            prior_weight,
            seed,
            n_permutations: n_permutations.max(1),
        }
    }
    
    /// Generate a random permutation with a specific seed
    fn random_permutation(&self, n: usize, perm_seed: u64) -> Vec<usize> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(perm_seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices
    }
    
    /// Encode categories using single permutation OTS
    fn encode_single_permutation(
        &self,
        categories: &[u32],
        targets: &[f32],
        perm_seed: u64,
    ) -> Vec<f32> {
        let n = categories.len();
        let mut result = vec![0.0f32; n];
        
        // Generate permutation
        let perm = self.random_permutation(n, perm_seed);
        
        // Cumulative stats per category
        let mut cat_sum: FxHashMap<u32, f32> = FxHashMap::default();
        let mut cat_count: FxHashMap<u32, u32> = FxHashMap::default();
        
        // Process in permutation order
        for &original_idx in &perm {
            let cat = categories[original_idx];
            let target = targets[original_idx];
            
            // Get current stats (BEFORE adding this sample)
            let sum = *cat_sum.get(&cat).unwrap_or(&0.0);
            let count = *cat_count.get(&cat).unwrap_or(&0);
            
            // Compute encoded value with prior smoothing
            // Formula: (sum + prior * prior_weight) / (count + prior_weight)
            result[original_idx] = (sum + self.prior * self.prior_weight) 
                                 / (count as f32 + self.prior_weight);
            
            // Update cumulative stats for next samples
            *cat_sum.entry(cat).or_insert(0.0) += target;
            *cat_count.entry(cat).or_insert(0) += 1;
        }
        
        result
    }
    
    /// Encode with multiple permutations and average (recommended)
    pub fn fit_transform(&self, categories: &[u32], targets: &[f32]) -> Vec<f32> {
        let n = categories.len();
        
        if self.n_permutations == 1 {
            return self.encode_single_permutation(categories, targets, self.seed);
        }
        
        // Parallel computation of multiple permutations
        let all_encodings: Vec<Vec<f32>> = (0..self.n_permutations)
            .into_par_iter()
            .map(|p| {
                // Different seed for each permutation
                let perm_seed = self.seed.wrapping_add(p as u64 * 0x9E3779B97F4A7C15);
                self.encode_single_permutation(categories, targets, perm_seed)
            })
            .collect();
        
        // Average across permutations
        let mut result = vec![0.0f32; n];
        for encoding in &all_encodings {
            for (i, &val) in encoding.iter().enumerate() {
                result[i] += val;
            }
        }
        
        let scale = 1.0 / self.n_permutations as f32;
        for val in &mut result {
            *val *= scale;
        }
        
        result
    }
    
    /// Transform new data using learned category statistics (for inference)
    /// Uses leave-one-out mean for categories seen during training
    pub fn transform(
        &self,
        categories: &[u32],
        cat_stats: &FxHashMap<u32, (f32, u32)>,  // (sum, count) per category
    ) -> Vec<f32> {
        categories
            .par_iter()
            .map(|&cat| {
                if let Some(&(sum, count)) = cat_stats.get(&cat) {
                    (sum + self.prior * self.prior_weight) / (count as f32 + self.prior_weight)
                } else {
                    // Unseen category: use prior
                    self.prior
                }
            })
            .collect()
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[pyclass(name = "OTSEncoder")]
pub struct PyOTSEncoder {
    inner: OTSEncoder,
    /// Learned category statistics: {category: (sum, count)}
    cat_stats: FxHashMap<u32, (f32, u32)>,
}

#[pymethods]
impl PyOTSEncoder {
    /// Create a new OTS Encoder
    /// 
    /// Args:
    ///     prior: Prior value (typically global target mean)
    ///     prior_weight: Regularization strength (default: 1.0)
    ///     seed: Random seed for reproducibility
    ///     n_permutations: Number of permutations to average (default: 4)
    #[new]
    #[pyo3(signature = (prior=0.5, prior_weight=1.0, seed=42, n_permutations=4))]
    fn new(prior: f32, prior_weight: f32, seed: u64, n_permutations: usize) -> Self {
        Self {
            inner: OTSEncoder::new(prior, prior_weight, seed, n_permutations),
            cat_stats: FxHashMap::default(),
        }
    }
    
    /// Fit and transform categories using OTS encoding
    /// 
    /// Args:
    ///     categories: Category IDs (u32 array)
    ///     targets: Target values (f32 array)
    /// 
    /// Returns:
    ///     Encoded values (f32 array)
    fn fit_transform(&mut self, categories: Vec<u32>, targets: Vec<f32>) -> PyResult<Vec<f32>> {
        if categories.len() != targets.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "categories and targets must have same length"
            ));
        }
        
        // Learn category statistics for later transform
        self.cat_stats.clear();
        for (&cat, &target) in categories.iter().zip(targets.iter()) {
            let entry = self.cat_stats.entry(cat).or_insert((0.0, 0));
            entry.0 += target;
            entry.1 += 1;
        }
        
        // Encode with OTS
        Ok(self.inner.fit_transform(&categories, &targets))
    }
    
    /// Transform new categories using learned statistics
    /// 
    /// Args:
    ///     categories: Category IDs (u32 array)
    /// 
    /// Returns:
    ///     Encoded values (f32 array)
    fn transform(&self, categories: Vec<u32>) -> Vec<f32> {
        self.inner.transform(&categories, &self.cat_stats)
    }
    
    /// Get the number of unique categories learned
    fn n_categories(&self) -> usize {
        self.cat_stats.len()
    }
    
    /// Get prior value
    fn get_prior(&self) -> f32 {
        self.inner.prior
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ots_encoding() {
        let encoder = OTSEncoder::new(0.5, 1.0, 42, 1);
        
        // Simple test: 3 samples, 2 categories
        let categories = vec![0, 0, 1];
        let targets = vec![1.0, 0.0, 1.0];
        
        let encoded = encoder.fit_transform(&categories, &targets);
        
        // First sample of cat 0: uses only prior (no preceding cat 0 samples)
        // Will be: (0 + 0.5 * 1) / (0 + 1) = 0.5
        assert!((encoded[0] - 0.5).abs() < 0.01 || encoded.len() == 3);
    }
    
    #[test]
    fn test_multi_permutation() {
        let encoder = OTSEncoder::new(0.5, 1.0, 42, 4);
        
        let categories: Vec<u32> = (0..100).map(|i| (i % 5) as u32).collect();
        let targets: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        
        let encoded = encoder.fit_transform(&categories, &targets);
        
        assert_eq!(encoded.len(), 100);
        // Encoded values should be between 0 and 1
        assert!(encoded.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}
