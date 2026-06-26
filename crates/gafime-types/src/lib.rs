#![no_std]
#![forbid(unsafe_code)]

use core::ffi::c_void;

pub const GAFIME_ABI_VERSION_MAJOR: u16 = 1;
pub const GAFIME_ABI_VERSION_MINOR: u16 = 0;
pub const GAFIME_ABI_VERSION: u32 =
    ((GAFIME_ABI_VERSION_MAJOR as u32) << 16) | GAFIME_ABI_VERSION_MINOR as u32;

pub type GafimeStatus = i32;
pub const GAFIME_STATUS_OK: GafimeStatus = 0;
pub const GAFIME_STATUS_INVALID_ARGUMENT: GafimeStatus = -1;
pub const GAFIME_STATUS_UNSUPPORTED_BACKEND: GafimeStatus = -2;
pub const GAFIME_STATUS_OUT_OF_MEMORY: GafimeStatus = -3;
pub const GAFIME_STATUS_DEVICE_ERROR: GafimeStatus = -4;
pub const GAFIME_STATUS_GRAPH_UNSUPPORTED: GafimeStatus = -5;
pub const GAFIME_STATUS_ABI_MISMATCH: GafimeStatus = -6;

pub type BackendKind = u32;
pub const GAFIME_BACKEND_CPU: BackendKind = 1;
pub const GAFIME_BACKEND_CUDA: BackendKind = 2;
pub const GAFIME_BACKEND_ROCM: BackendKind = 3;
pub const GAFIME_BACKEND_METAL: BackendKind = 4;

pub type MetricId = u32;
pub const GAFIME_METRIC_PEARSON: MetricId = 1;
pub const GAFIME_METRIC_SPEARMAN: MetricId = 2;
pub const GAFIME_METRIC_MUTUAL_INFO: MetricId = 3;
pub const GAFIME_METRIC_R2: MetricId = 4;

pub type CandidateFamily = u32;
pub const GAFIME_FAMILY_CONTINUOUS: CandidateFamily = 1;
pub const GAFIME_FAMILY_DECISION_PATH: CandidateFamily = 2;
pub const GAFIME_FAMILY_TIME_SERIES: CandidateFamily = 3;

pub type DataType = u32;
pub const GAFIME_DTYPE_F32: DataType = 1;

pub type MatrixLayout = u32;
pub const GAFIME_MATRIX_ROW_MAJOR: MatrixLayout = 1;
pub const GAFIME_MATRIX_COLUMN_MAJOR: MatrixLayout = 2;
pub const GAFIME_MATRIX_ARROW_COLUMNAR: MatrixLayout = 3;
pub const GAFIME_MATRIX_DEVICE_NATIVE: MatrixLayout = 4;

pub type GraphMode = u32;
pub const GAFIME_GRAPH_UNSUPPORTED: GraphMode = 0;
pub const GAFIME_GRAPH_STREAM_CAPTURE: GraphMode = 1;
pub const GAFIME_GRAPH_HOST_REPLAY: GraphMode = 2;

pub type InputSourceKind = u32;
pub const GAFIME_INPUT_HOST_F32: InputSourceKind = 1;
pub const GAFIME_INPUT_ARROW_C_DATA: InputSourceKind = 2;
pub const GAFIME_INPUT_PARQUET_PATH: InputSourceKind = 3;
pub const GAFIME_INPUT_DEVICE_NATIVE: InputSourceKind = 4;

pub type GafimeGpuMatrix = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSliceU32 {
    pub ptr: *const u32,
    pub len: u64,
}

impl Default for GafimeSliceU32 {
    fn default() -> Self {
        Self {
            ptr: core::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSliceU64 {
    pub ptr: *const u64,
    pub len: u64,
}

impl Default for GafimeSliceU64 {
    fn default() -> Self {
        Self {
            ptr: core::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeSliceF32 {
    pub ptr: *const f32,
    pub len: u64,
}

impl Default for GafimeSliceF32 {
    fn default() -> Self {
        Self {
            ptr: core::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeStringView {
    pub ptr: *const u8,
    pub len: u64,
}

impl Default for GafimeStringView {
    fn default() -> Self {
        Self {
            ptr: core::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeComputeBudget {
    pub max_comb_size: u32,
    pub max_combinations_per_k: u64,
    pub top_features_for_higher_k: u32,
    pub max_generated_features: u32,
    pub max_time_series_candidates: u64,
    pub top_k_features_for_time_series: u32,
    pub max_feature_candidate: i64,
    pub vram_budget_mb: u64,
    pub flags: u32,
    pub reserved32: u32,
    pub reserved: [u64; 8],
}

impl Default for GafimeComputeBudget {
    fn default() -> Self {
        Self {
            max_comb_size: 2,
            max_combinations_per_k: 5_000,
            top_features_for_higher_k: 50,
            max_generated_features: 0,
            max_time_series_candidates: 100_000,
            top_k_features_for_time_series: 50,
            max_feature_candidate: 0,
            vram_budget_mb: 6_144,
            flags: 0,
            reserved32: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeEngineConfig {
    pub abi_version: u32,
    pub backend_kind: BackendKind,
    pub device_id: u32,
    pub flags: u32,
    pub metric_ids: GafimeSliceU32,
    pub budget: GafimeComputeBudget,
    pub num_repeats: u32,
    pub permutation_tests: u32,
    pub random_seed: u64,
    pub mi_bins: u32,
    pub stability_std_threshold_ppm: u32,
    pub permutation_p_threshold_ppm: u32,
    pub time_series_lags: GafimeSliceU32,
    pub time_series_windows: GafimeSliceU32,
    pub decision_path_max_depth: u32,
    pub decision_path_rounds: u32,
    pub decision_path_max_paths: u32,
    pub decision_path_max_bins: u32,
    pub decision_path_min_leaf: u32,
    pub decision_path_learning_rate_ppm: u32,
    pub decision_path_top_k_features: u32,
    pub reserved32: u32,
    pub reserved: [u64; 8],
}

impl Default for GafimeEngineConfig {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_ABI_VERSION,
            backend_kind: 0,
            device_id: 0,
            flags: 0,
            metric_ids: GafimeSliceU32::default(),
            budget: GafimeComputeBudget::default(),
            num_repeats: 3,
            permutation_tests: 25,
            random_seed: 7,
            mi_bins: 96,
            stability_std_threshold_ppm: 100_000,
            permutation_p_threshold_ppm: 50_000,
            time_series_lags: GafimeSliceU32::default(),
            time_series_windows: GafimeSliceU32::default(),
            decision_path_max_depth: 2,
            decision_path_rounds: 1,
            decision_path_max_paths: 32,
            decision_path_max_bins: 0,
            decision_path_min_leaf: 8,
            decision_path_learning_rate_ppm: 1_000_000,
            decision_path_top_k_features: 50,
            reserved32: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeInputDescriptor {
    pub abi_version: u32,
    pub source_kind: InputSourceKind,
    pub dtype: DataType,
    pub layout: MatrixLayout,
    pub flags: u32,
    pub rows: u64,
    pub cols: u32,
    pub row_stride: u32,
    pub features_ptr: *const c_void,
    pub target_ptr: *const c_void,
    pub schema_ptr: *const c_void,
    pub path: GafimeStringView,
    pub reserved: [u64; 8],
}

impl Default for GafimeInputDescriptor {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_ABI_VERSION,
            source_kind: GAFIME_INPUT_HOST_F32,
            dtype: GAFIME_DTYPE_F32,
            layout: GAFIME_MATRIX_ROW_MAJOR,
            flags: 0,
            rows: 0,
            cols: 0,
            row_stride: 0,
            features_ptr: core::ptr::null(),
            target_ptr: core::ptr::null(),
            schema_ptr: core::ptr::null(),
            path: GafimeStringView::default(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeMatrixDesc {
    pub abi_version: u32,
    pub dtype: DataType,
    pub layout: MatrixLayout,
    pub flags: u32,
    pub rows: u64,
    pub cols: u32,
    pub row_stride: u32,
    pub bytes: u64,
}

impl Default for GafimeMatrixDesc {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_ABI_VERSION,
            dtype: GAFIME_DTYPE_F32,
            layout: GAFIME_MATRIX_ROW_MAJOR,
            flags: 0,
            rows: 0,
            cols: 0,
            row_stride: 0,
            bytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeGpuDeviceInfo {
    pub abi_version: u32,
    pub backend_kind: BackendKind,
    pub device_id: u32,
    pub flags: u32,
    pub name: [u8; 128],
    pub total_global_mem_bytes: u64,
    pub multiprocessor_count: u32,
    pub warp_size: u32,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub driver_version: u32,
    pub runtime_version: u32,
    pub reserved: [u64; 8],
}

impl Default for GafimeGpuDeviceInfo {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_ABI_VERSION,
            backend_kind: 0,
            device_id: 0,
            flags: 0,
            name: [0; 128],
            total_global_mem_bytes: 0,
            multiprocessor_count: 0,
            warp_size: 0,
            compute_major: 0,
            compute_minor: 0,
            driver_version: 0,
            runtime_version: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GafimeGpuGraphCapability {
    pub abi_version: u32,
    pub backend_kind: BackendKind,
    pub graph_mode: GraphMode,
    pub flags: u32,
    pub supports_memcpy_nodes: u32,
    pub supports_kernel_param_update: u32,
    pub supports_device_ranking: u32,
    pub max_captured_nodes: u32,
    pub stable_pointer_flags: u64,
    pub reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GafimeShapeHint {
    pub threads_per_block: u32,
    pub items_per_thread: u32,
    pub blocks_per_sm: u32,
    pub min_blocks: u32,
    pub shared_bytes: u32,
    pub register_budget: u32,
    pub occupancy_target_pct: u32,
    pub vendor_hint: u32,
    pub reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GafimeArityChunk {
    pub arity: u32,
    pub family: CandidateFamily,
    pub metric_mask: u32,
    pub shape_hint_index: u32,
    pub combo_row_offset: u64,
    pub combo_count: u64,
    pub local_chunk_id: u32,
    pub flags: u32,
    pub descriptor_offset: u64,
    pub descriptor_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GafimeRankSpec {
    pub top_k: u32,
    pub primary_metric: MetricId,
    pub descending: u32,
    pub include_ties: u32,
    pub reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GafimePermutationSchedule {
    pub permutation_count: u32,
    pub mode: u32,
    pub flags: u32,
    pub reserved32: u32,
    pub seed: u64,
    pub target_offsets: GafimeSliceU64,
    pub reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GafimeLaunchProtocol {
    pub abi_version: u32,
    pub backend_kind: BackendKind,
    pub flags: u32,
    pub max_arity: u32,
    pub n_samples: u64,
    pub n_features: u32,
    pub family_count: u32,
    pub combo_indices: GafimeSliceU32,
    pub metric_ids: GafimeSliceU32,
    pub chunks: *const GafimeArityChunk,
    pub chunk_count: u32,
    pub reserved32_a: u32,
    pub shape_hints: *const GafimeShapeHint,
    pub shape_hint_count: u32,
    pub reserved32_b: u32,
    pub rank: GafimeRankSpec,
    pub permutations: GafimePermutationSchedule,
    pub reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeResultTable {
    pub abi_version: u32,
    pub max_arity: u32,
    pub metric_count: u32,
    pub flags: u32,
    pub capacity: u64,
    pub row_count: u64,
    pub combo_indices: *mut u32,
    pub metric_values: *mut f32,
    pub ranks: *mut u32,
    pub families: *mut u32,
    pub candidate_ids: *mut u64,
    pub row_flags: *mut u32,
    pub backend_private: *mut c_void,
    pub reserved: [u64; 8],
}

impl Default for GafimeResultTable {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_ABI_VERSION,
            max_arity: 0,
            metric_count: 0,
            flags: 0,
            capacity: 0,
            row_count: 0,
            combo_indices: core::ptr::null_mut(),
            metric_values: core::ptr::null_mut(),
            ranks: core::ptr::null_mut(),
            families: core::ptr::null_mut(),
            candidate_ids: core::ptr::null_mut(),
            row_flags: core::ptr::null_mut(),
            backend_private: core::ptr::null_mut(),
            reserved: [0; 8],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn slices_are_c_pointer_len_pairs() {
        assert_eq!(size_of::<GafimeSliceU32>(), 16);
        assert_eq!(size_of::<GafimeSliceU64>(), 16);
        assert_eq!(size_of::<GafimeSliceF32>(), 16);
        assert_eq!(align_of::<GafimeSliceU32>(), align_of::<usize>());
    }

    #[test]
    fn p0_contract_uses_v1_abi_and_compact_result_rows() {
        let table = GafimeResultTable::default();
        assert_eq!(table.abi_version, GAFIME_ABI_VERSION);
        assert_eq!(GAFIME_FAMILY_CONTINUOUS, 1);
        assert_eq!(GAFIME_FAMILY_DECISION_PATH, 2);
        assert_eq!(GAFIME_FAMILY_TIME_SERIES, 3);
    }

    #[test]
    fn config_and_input_defaults_match_legacy_surface() {
        let config = GafimeEngineConfig::default();
        assert_eq!(config.abi_version, GAFIME_ABI_VERSION);
        assert_eq!(config.budget.max_comb_size, 2);
        assert_eq!(config.mi_bins, 96);
        assert_eq!(config.num_repeats, 3);
        assert_eq!(config.permutation_tests, 25);

        let input = GafimeInputDescriptor::default();
        assert_eq!(input.source_kind, GAFIME_INPUT_HOST_F32);
        assert_eq!(input.dtype, GAFIME_DTYPE_F32);
    }
}
