//! Data Quality and Anomaly Detection Module
//!
//! Preprocessing utilities for GAFIME:
//! 1. NaN/Inf detection and handling
//! 2. Missing value analysis (by feature, by row)
//! 3. Entropy-based feature analysis (constant detection, ID detection)

use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

// ============================================================================
// NaN/Inf Handler
// ============================================================================

/// Result of NaN/Inf analysis
#[derive(Clone, Debug)]
pub struct NanInfReport {
    pub total_values: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub neg_inf_count: usize,
    pub nan_positions: Vec<(usize, usize)>,  // (row, col)
    pub inf_positions: Vec<(usize, usize)>,
}

/// Strategies for handling NaN/Inf values
#[derive(Clone, Copy, Debug)]
pub enum NanInfStrategy {
    Replace(f32),       // Replace with constant
    Mean,               // Replace with feature mean
    Median,             // Replace with feature median
    Remove,             // Remove affected rows
    Interpolate,        // Linear interpolation (for time series)
}

/// NaN and Inf handler
pub struct NanInfHandler;

impl NanInfHandler {
    /// Analyze data for NaN and Inf values
    pub fn analyze(data: &[Vec<f32>]) -> NanInfReport {
        let n_features = data.len();
        let n_samples = if n_features > 0 { data[0].len() } else { 0 };
        
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut neg_inf_count = 0;
        let mut nan_positions = Vec::new();
        let mut inf_positions = Vec::new();
        
        for (col, feature) in data.iter().enumerate() {
            for (row, &val) in feature.iter().enumerate() {
                if val.is_nan() {
                    nan_count += 1;
                    nan_positions.push((row, col));
                } else if val.is_infinite() {
                    if val > 0.0 {
                        inf_count += 1;
                    } else {
                        neg_inf_count += 1;
                    }
                    inf_positions.push((row, col));
                }
            }
        }
        
        NanInfReport {
            total_values: n_features * n_samples,
            nan_count,
            inf_count,
            neg_inf_count,
            nan_positions,
            inf_positions,
        }
    }
    
    /// Clean data by replacing NaN/Inf with specified strategy
    pub fn clean(data: &mut [Vec<f32>], nan_strategy: NanInfStrategy, inf_strategy: NanInfStrategy) {
        for feature in data.iter_mut() {
            // Calculate replacement values if needed
            let mean = Self::calculate_mean(feature);
            let median = Self::calculate_median(feature);
            
            for val in feature.iter_mut() {
                if val.is_nan() {
                    *val = match nan_strategy {
                        NanInfStrategy::Replace(v) => v,
                        NanInfStrategy::Mean => mean,
                        NanInfStrategy::Median => median,
                        _ => 0.0,
                    };
                } else if val.is_infinite() {
                    *val = match inf_strategy {
                        NanInfStrategy::Replace(v) => v,
                        NanInfStrategy::Mean => mean,
                        NanInfStrategy::Median => median,
                        _ => 0.0,
                    };
                }
            }
        }
    }
    
    fn calculate_mean(data: &[f32]) -> f32 {
        let valid: Vec<f32> = data.iter()
            .filter(|x| x.is_finite())
            .copied()
            .collect();
        if valid.is_empty() { return 0.0; }
        valid.iter().sum::<f32>() / valid.len() as f32
    }
    
    fn calculate_median(data: &[f32]) -> f32 {
        let mut valid: Vec<f32> = data.iter()
            .filter(|x| x.is_finite())
            .copied()
            .collect();
        if valid.is_empty() { return 0.0; }
        valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = valid.len() / 2;
        if valid.len() % 2 == 0 {
            (valid[mid - 1] + valid[mid]) / 2.0
        } else {
            valid[mid]
        }
    }
}

// ============================================================================
// Data Orienter (Missing Value Analysis)
// ============================================================================

/// Missing value analysis result
#[derive(Clone, Debug)]
pub struct MissingValueReport {
    pub has_any_missing: bool,
    pub total_missing: usize,
    pub missing_by_feature: Vec<(usize, usize, f32)>,  // (feature_idx, count, percentage)
    pub missing_by_row: Vec<(usize, usize)>,           // (row_idx, count)
    pub complete_rows: usize,
    pub complete_features: usize,
    pub features_with_missing: Vec<usize>,
    pub rows_with_missing: Vec<usize>,
}

/// Data orienter for alignment and missing value checks
pub struct DataOrienter;

impl DataOrienter {
    /// Analyze missing values in dataset
    /// NaN values are treated as missing
    pub fn analyze_missing(data: &[Vec<f32>]) -> MissingValueReport {
        let n_features = data.len();
        if n_features == 0 {
            return MissingValueReport {
                has_any_missing: false,
                total_missing: 0,
                missing_by_feature: vec![],
                missing_by_row: vec![],
                complete_rows: 0,
                complete_features: 0,
                features_with_missing: vec![],
                rows_with_missing: vec![],
            };
        }
        
        let n_samples = data[0].len();
        
        // Count missing by feature
        let missing_by_feature: Vec<(usize, usize, f32)> = data.iter()
            .enumerate()
            .map(|(idx, feature)| {
                let count = feature.iter().filter(|x| x.is_nan()).count();
                let pct = count as f32 / n_samples as f32 * 100.0;
                (idx, count, pct)
            })
            .collect();
        
        // Count missing by row
        let mut row_missing: Vec<usize> = vec![0; n_samples];
        for feature in data {
            for (row, &val) in feature.iter().enumerate() {
                if val.is_nan() {
                    row_missing[row] += 1;
                }
            }
        }
        
        let missing_by_row: Vec<(usize, usize)> = row_missing.iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(idx, &count)| (idx, count))
            .collect();
        
        let total_missing: usize = missing_by_feature.iter().map(|(_, c, _)| c).sum();
        let features_with_missing: Vec<usize> = missing_by_feature.iter()
            .filter(|(_, c, _)| *c > 0)
            .map(|(idx, _, _)| *idx)
            .collect();
        let rows_with_missing: Vec<usize> = missing_by_row.iter()
            .map(|(idx, _)| *idx)
            .collect();
        
        MissingValueReport {
            has_any_missing: total_missing > 0,
            total_missing,
            missing_by_feature,
            missing_by_row,
            complete_rows: n_samples - rows_with_missing.len(),
            complete_features: n_features - features_with_missing.len(),
            features_with_missing,
            rows_with_missing,
        }
    }
    
    /// Check if all features have the same length
    pub fn check_alignment(data: &[Vec<f32>]) -> Result<usize, Vec<(usize, usize)>> {
        if data.is_empty() {
            return Ok(0);
        }
        
        let expected_len = data[0].len();
        let misaligned: Vec<(usize, usize)> = data.iter()
            .enumerate()
            .filter(|(_, f)| f.len() != expected_len)
            .map(|(idx, f)| (idx, f.len()))
            .collect();
        
        if misaligned.is_empty() {
            Ok(expected_len)
        } else {
            Err(misaligned)
        }
    }
}

// ============================================================================
// Entropy Analyzer
// ============================================================================

/// Feature type detected by entropy analysis
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureType {
    Constant,           // All values same (entropy ≈ 0)
    LowVariance,        // Very few unique values
    Normal,             // Good variance
    HighCardinality,    // Many unique values (possible ID)
    UniqueId,           // Each value unique (definitely ID)
    DateTime,           // Detected as timestamp
    Categorical,        // Few unique values, likely categorical
}

/// Entropy analysis result for a feature
#[derive(Clone, Debug)]
pub struct FeatureEntropyReport {
    pub feature_idx: usize,
    pub entropy: f32,
    pub unique_count: usize,
    pub unique_ratio: f32,  // unique_count / total
    pub feature_type: FeatureType,
    pub recommendation: String,
}

/// Entropy-based feature analyzer
pub struct EntropyAnalyzer {
    low_entropy_threshold: f32,    // Below this = constant/low variance
    high_cardinality_ratio: f32,   // Above this = possible ID
    categorical_max_unique: usize, // Max unique values for categorical
}

impl EntropyAnalyzer {
    pub fn new() -> Self {
        Self {
            low_entropy_threshold: 0.1,
            high_cardinality_ratio: 0.95,
            categorical_max_unique: 50,
        }
    }
    
    pub fn with_thresholds(
        low_entropy: f32,
        high_cardinality: f32,
        categorical_max: usize,
    ) -> Self {
        Self {
            low_entropy_threshold: low_entropy,
            high_cardinality_ratio: high_cardinality,
            categorical_max_unique: categorical_max,
        }
    }
    
    /// Analyze single feature
    pub fn analyze_feature(&self, feature: &[f32], feature_idx: usize) -> FeatureEntropyReport {
        let n = feature.len();
        if n == 0 {
            return FeatureEntropyReport {
                feature_idx,
                entropy: 0.0,
                unique_count: 0,
                unique_ratio: 0.0,
                feature_type: FeatureType::Constant,
                recommendation: "Empty feature - remove".to_string(),
            };
        }
        
        // Count unique values (discretize floats to handle precision)
        let mut value_counts: FxHashMap<i64, usize> = FxHashMap::default();
        for &val in feature {
            if val.is_finite() {
                // Discretize to 6 decimal places
                let key = (val * 1_000_000.0).round() as i64;
                *value_counts.entry(key).or_insert(0) += 1;
            }
        }
        
        let unique_count = value_counts.len();
        let unique_ratio = unique_count as f32 / n as f32;
        
        // Calculate entropy: H = -Σ p(x) * log2(p(x))
        let entropy: f32 = value_counts.values()
            .map(|&count| {
                let p = count as f32 / n as f32;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum();
        
        // Normalize entropy (0-1 scale)
        let max_entropy = (n as f32).log2();
        let normalized_entropy = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };
        
        // Determine feature type and recommendation
        let (feature_type, recommendation) = self.classify_feature(
            normalized_entropy, unique_count, unique_ratio, n
        );
        
        FeatureEntropyReport {
            feature_idx,
            entropy: normalized_entropy,
            unique_count,
            unique_ratio,
            feature_type,
            recommendation,
        }
    }
    
    /// Analyze all features in parallel
    pub fn analyze_all(&self, data: &[Vec<f32>]) -> Vec<FeatureEntropyReport> {
        data.par_iter()
            .enumerate()
            .map(|(idx, feature)| self.analyze_feature(feature, idx))
            .collect()
    }
    
    fn classify_feature(
        &self,
        entropy: f32,
        unique_count: usize,
        unique_ratio: f32,
        n: usize,
    ) -> (FeatureType, String) {
        // Constant feature
        if unique_count <= 1 {
            return (
                FeatureType::Constant,
                "DROP: Constant feature provides no information".to_string(),
            );
        }
        
        // Unique ID (every value different)
        if unique_count == n {
            return (
                FeatureType::UniqueId,
                "ID COLUMN: Each value unique - use for indexing, not modeling".to_string(),
            );
        }
        
        // Very low entropy
        if entropy < self.low_entropy_threshold {
            return (
                FeatureType::LowVariance,
                format!("CONSIDER DROP: Very low variance (entropy={:.3})", entropy),
            );
        }
        
        // High cardinality (possible ID or datetime)
        if unique_ratio > self.high_cardinality_ratio {
            // Check if values look like timestamps (large integers)
            return (
                FeatureType::HighCardinality,
                "HIGH CARDINALITY: Possible ID/datetime - investigate before using".to_string(),
            );
        }
        
        // Categorical
        if unique_count <= self.categorical_max_unique {
            return (
                FeatureType::Categorical,
                format!("CATEGORICAL: {} unique values - consider encoding", unique_count),
            );
        }
        
        // Normal continuous feature
        (
            FeatureType::Normal,
            "KEEP: Good variance for modeling".to_string(),
        )
    }
    
    /// Get features to drop based on analysis
    pub fn get_features_to_drop(&self, reports: &[FeatureEntropyReport]) -> Vec<usize> {
        reports.iter()
            .filter(|r| matches!(r.feature_type, FeatureType::Constant | FeatureType::LowVariance))
            .map(|r| r.feature_idx)
            .collect()
    }
    
    /// Get features that are likely IDs
    pub fn get_id_features(&self, reports: &[FeatureEntropyReport]) -> Vec<usize> {
        reports.iter()
            .filter(|r| matches!(r.feature_type, FeatureType::UniqueId | FeatureType::HighCardinality))
            .map(|r| r.feature_idx)
            .collect()
    }
    
    /// Get categorical features
    pub fn get_categorical_features(&self, reports: &[FeatureEntropyReport]) -> Vec<usize> {
        reports.iter()
            .filter(|r| matches!(r.feature_type, FeatureType::Categorical))
            .map(|r| r.feature_idx)
            .collect()
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[pyclass(name = "DataQualityAnalyzer")]
pub struct PyDataQualityAnalyzer {
    entropy_analyzer: EntropyAnalyzer,
}

#[pymethods]
impl PyDataQualityAnalyzer {
    #[new]
    #[pyo3(signature = (low_entropy_threshold=0.1, high_cardinality_ratio=0.95, categorical_max=50))]
    fn new(low_entropy_threshold: f32, high_cardinality_ratio: f32, categorical_max: usize) -> Self {
        Self {
            entropy_analyzer: EntropyAnalyzer::with_thresholds(
                low_entropy_threshold,
                high_cardinality_ratio,
                categorical_max,
            ),
        }
    }
    
    /// Analyze NaN/Inf in data
    /// Returns: (total, nan_count, inf_count, neg_inf_count)
    fn analyze_nan_inf(&self, data: Vec<Vec<f32>>) -> (usize, usize, usize, usize) {
        let report = NanInfHandler::analyze(&data);
        (report.total_values, report.nan_count, report.inf_count, report.neg_inf_count)
    }
    
    /// Clean NaN/Inf by replacing with value
    fn clean_nan_inf(&self, mut data: Vec<Vec<f32>>, nan_replace: f32, inf_replace: f32) -> Vec<Vec<f32>> {
        NanInfHandler::clean(
            &mut data,
            NanInfStrategy::Replace(nan_replace),
            NanInfStrategy::Replace(inf_replace),
        );
        data
    }
    
    /// Clean NaN/Inf by replacing with mean
    fn clean_nan_inf_mean(&self, mut data: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        NanInfHandler::clean(&mut data, NanInfStrategy::Mean, NanInfStrategy::Mean);
        data
    }
    
    /// Analyze missing values
    /// Returns: (has_missing, total_missing, complete_rows, complete_features, features_with_missing, rows_with_missing)
    fn analyze_missing(&self, data: Vec<Vec<f32>>) -> (bool, usize, usize, usize, Vec<usize>, Vec<usize>) {
        let report = DataOrienter::analyze_missing(&data);
        (
            report.has_any_missing,
            report.total_missing,
            report.complete_rows,
            report.complete_features,
            report.features_with_missing,
            report.rows_with_missing,
        )
    }
    
    /// Check data alignment
    /// Returns: Ok(n_samples) or list of misaligned (feature_idx, actual_length)
    fn check_alignment(&self, data: Vec<Vec<f32>>) -> PyResult<usize> {
        match DataOrienter::check_alignment(&data) {
            Ok(n) => Ok(n),
            Err(misaligned) => {
                let msg = misaligned.iter()
                    .map(|(idx, len)| format!("feature {}: len {}", idx, len))
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Misaligned features: {}", msg)
                ))
            }
        }
    }
    
    /// Analyze feature entropy
    /// Returns list of (feature_idx, entropy, unique_count, unique_ratio, type_str, recommendation)
    fn analyze_entropy(&self, data: Vec<Vec<f32>>) -> Vec<(usize, f32, usize, f32, String, String)> {
        let reports = self.entropy_analyzer.analyze_all(&data);
        reports.into_iter()
            .map(|r| {
                let type_str = match r.feature_type {
                    FeatureType::Constant => "constant",
                    FeatureType::LowVariance => "low_variance",
                    FeatureType::Normal => "normal",
                    FeatureType::HighCardinality => "high_cardinality",
                    FeatureType::UniqueId => "unique_id",
                    FeatureType::DateTime => "datetime",
                    FeatureType::Categorical => "categorical",
                };
                (r.feature_idx, r.entropy, r.unique_count, r.unique_ratio, type_str.to_string(), r.recommendation)
            })
            .collect()
    }
    
    /// Get indices of features to drop (constant/low variance)
    fn get_features_to_drop(&self, data: Vec<Vec<f32>>) -> Vec<usize> {
        let reports = self.entropy_analyzer.analyze_all(&data);
        self.entropy_analyzer.get_features_to_drop(&reports)
    }
    
    /// Get indices of ID columns
    fn get_id_features(&self, data: Vec<Vec<f32>>) -> Vec<usize> {
        let reports = self.entropy_analyzer.analyze_all(&data);
        self.entropy_analyzer.get_id_features(&reports)
    }
    
    /// Get indices of categorical features
    fn get_categorical_features(&self, data: Vec<Vec<f32>>) -> Vec<usize> {
        let reports = self.entropy_analyzer.analyze_all(&data);
        self.entropy_analyzer.get_categorical_features(&reports)
    }
    
    /// Full data quality report
    fn full_report(&self, data: Vec<Vec<f32>>) -> PyResult<String> {
        let n_features = data.len();
        let n_samples = if n_features > 0 { data[0].len() } else { 0 };
        
        // Alignment check
        let alignment = DataOrienter::check_alignment(&data);
        
        // NaN/Inf analysis
        let nan_report = NanInfHandler::analyze(&data);
        
        // Missing value analysis
        let missing_report = DataOrienter::analyze_missing(&data);
        
        // Entropy analysis
        let entropy_reports = self.entropy_analyzer.analyze_all(&data);
        let to_drop = self.entropy_analyzer.get_features_to_drop(&entropy_reports);
        let id_features = self.entropy_analyzer.get_id_features(&entropy_reports);
        let categorical = self.entropy_analyzer.get_categorical_features(&entropy_reports);
        
        let mut report = String::new();
        report.push_str(&format!("=== Data Quality Report ===\n"));
        report.push_str(&format!("Shape: {} features × {} samples\n\n", n_features, n_samples));
        
        report.push_str(&format!("ALIGNMENT: {}\n", 
            if alignment.is_ok() { "OK" } else { "MISALIGNED!" }));
        
        report.push_str(&format!("\nNaN/Inf Analysis:\n"));
        report.push_str(&format!("  NaN count: {}\n", nan_report.nan_count));
        report.push_str(&format!("  Inf count: {} (+) {} (-)\n", nan_report.inf_count, nan_report.neg_inf_count));
        
        report.push_str(&format!("\nMissing Values:\n"));
        report.push_str(&format!("  Total missing: {}\n", missing_report.total_missing));
        report.push_str(&format!("  Complete rows: {}/{}\n", missing_report.complete_rows, n_samples));
        report.push_str(&format!("  Complete features: {}/{}\n", missing_report.complete_features, n_features));
        
        report.push_str(&format!("\nEntropy Analysis:\n"));
        report.push_str(&format!("  Features to DROP: {:?}\n", to_drop));
        report.push_str(&format!("  ID columns: {:?}\n", id_features));
        report.push_str(&format!("  Categorical: {:?}\n", categorical));
        
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nan_detection() {
        let data = vec![
            vec![1.0, 2.0, f32::NAN, 4.0],
            vec![1.0, f32::INFINITY, 3.0, 4.0],
        ];
        let report = NanInfHandler::analyze(&data);
        assert_eq!(report.nan_count, 1);
        assert_eq!(report.inf_count, 1);
    }
    
    #[test]
    fn test_entropy_constant() {
        let analyzer = EntropyAnalyzer::new();
        let feature = vec![1.0; 100];
        let report = analyzer.analyze_feature(&feature, 0);
        assert_eq!(report.feature_type, FeatureType::Constant);
    }
    
    #[test]
    fn test_entropy_unique_id() {
        let analyzer = EntropyAnalyzer::new();
        let feature: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let report = analyzer.analyze_feature(&feature, 0);
        assert_eq!(report.feature_type, FeatureType::UniqueId);
    }
}
