use pyo3::prelude::*;
use rustc_hash::FxHashSet;
use std::hash::{Hash, Hasher};

/// Represents a single operand in an interaction: op(feature)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Operand {
    feature_idx: usize,
    op_idx: usize,
}

/// Key for canonicalization.
/// For commutative operations (ADD, MUL, MAX, MIN), operands are sorted.
/// For non-commutative (SUB, DIV), order is preserved.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CanonicalKey {
    operands: Vec<Operand>,
    interact_type: usize,
}

/// Helper to determine if an interaction type is commutative
fn is_commutative(interact_type: usize) -> bool {
    // 0=MULT, 1=ADD, 4=MAX, 5=MIN are commutative
    // 2=SUB, 3=DIV are NOT
    matches!(interact_type, 0 | 1 | 4 | 5)
}

#[pyclass(name = "SmartScheduler")]
pub struct PySmartScheduler {
    n_features: usize,
    seen_interactions: FxHashSet<CanonicalKey>,
    
    // Iteration state for Arity 2
    // Outer loop: feature A (fixed)
    idx_a: usize,
    // Inner loop: feature B (varying)
    idx_b: usize,
    
    // Operator iteration limits
    n_ops: usize,
    n_interact_types: usize,
    
    // Current operator state
    op_a: usize,
    op_b: usize,
    interact_type: usize,
}

#[pymethods]
impl PySmartScheduler {
    #[new]
    pub fn new(n_features: usize, n_ops: usize, n_interact_types: usize) -> Self {
        PySmartScheduler {
            n_features,
            seen_interactions: FxHashSet::default(),
            idx_a: 0,
            idx_b: 1, // Start at a+1
            n_ops,
            n_interact_types,
            op_a: 0,
            op_b: 0,
            interact_type: 0,
        }
    }
    
    /// Reset the scheduler to initial state
    pub fn reset(&mut self) {
        self.seen_interactions.clear();
        self.idx_a = 0;
        self.idx_b = 1;
        self.op_a = 0;
        self.op_b = 0;
        self.interact_type = 0;
    }
    
    /// Generate a batch of unique interactions.
    /// Returns tuple of lists: (feature_a, feature_b, op_a, op_b, interact_type)
    pub fn generate_batch(&mut self, batch_size: usize) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)> {
        let mut f_a_out = Vec::with_capacity(batch_size);
        let mut f_b_out = Vec::with_capacity(batch_size);
        let mut op_a_out = Vec::with_capacity(batch_size);
        let mut op_b_out = Vec::with_capacity(batch_size);
        let mut int_out = Vec::with_capacity(batch_size);
        
        // Loop until batch is full or we exhaust all combinations
        while f_a_out.len() < batch_size {
            // Check bounds
            if self.idx_a >= self.n_features {
                break; // Done
            }
            
            // Construct current candidate
            let ops = vec![
                Operand { feature_idx: self.idx_a, op_idx: self.op_a },
                Operand { feature_idx: self.idx_b, op_idx: self.op_b },
            ];
            
            // Canonicalize
            let mut canonical_ops = ops.clone();
            if is_commutative(self.interact_type) {
                canonical_ops.sort(); // Sorts by feature_idx then op_idx
            }
            
            let key = CanonicalKey {
                operands: canonical_ops,
                interact_type: self.interact_type,
            };
            
            // Check deduplication
            if !self.seen_interactions.contains(&key) {
                // New interaction! Add to batch
                self.seen_interactions.insert(key);
                
                f_a_out.push(self.idx_a);
                f_b_out.push(self.idx_b);
                op_a_out.push(self.op_a);
                op_b_out.push(self.op_b);
                int_out.push(self.interact_type);
            }
            
            // Advance state
            self.advance_state();
        }
        
        Ok((f_a_out, f_b_out, op_a_out, op_b_out, int_out))
    }
    
    /// Total number of unique interactions found so far
    pub fn count_seen(&self) -> usize {
        self.seen_interactions.len()
    }
}

impl PySmartScheduler {
    // Helper to advance indices in correct nested order
    // Order: InteractType -> OpB -> OpA -> FeatureB -> FeatureA
    // This order maximizes L2 locality for Feature A (and Feature B for inner loops of ops)
    fn advance_state(&mut self) {
        // 1. Advance Interaction Type
        self.interact_type += 1;
        if self.interact_type < self.n_interact_types { return; }
        self.interact_type = 0;
        
        // 2. Advance Op B
        self.op_b += 1;
        if self.op_b < self.n_ops { return; }
        self.op_b = 0;
        
        // 3. Advance Op A
        self.op_a += 1;
        if self.op_a < self.n_ops { return; }
        self.op_a = 0;
        
        // 4. Advance Feature B (varying)
        self.idx_b += 1;
        if self.idx_b < self.n_features {
            // Valid feature B.
            // Loop condition usually idx_b > idx_a, 
            // but we initialized idx_b = idx_a + 1.
            return; 
        }
        
        // 5. Advance Feature A (fixed)
        self.idx_a += 1;
        // Reset B to A + 1
        self.idx_b = self.idx_a + 1;
        
        // Critical Fix: Check if new pair is valid
        if self.idx_b >= self.n_features {
            // No valid B for this A (means A was the last feature).
            // Fast forward A to end to terminate loop next time.
            self.idx_a = self.n_features;
        }
        
        // Check termination in generate_batch
    }
}
