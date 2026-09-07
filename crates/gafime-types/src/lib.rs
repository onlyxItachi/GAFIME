#![no_std]
#![forbid(unsafe_code)]

use core::ffi::c_void;

#[cfg(feature = "local-cmake-experiment")]
mod local_cmake_experiment;
#[cfg(feature = "local-cmake-experiment")]
pub use local_cmake_experiment::*;

pub const GAFIME_ABI_VERSION_MAJOR: u16 = 1;
pub const GAFIME_ABI_VERSION_MINOR: u16 = 0;
pub const GAFIME_ABI_VERSION: u32 =
    ((GAFIME_ABI_VERSION_MAJOR as u32) << 16) | GAFIME_ABI_VERSION_MINOR as u32;

/// Additive precision-profile ABI. ABI 1.0 remains available for legacy fp32
/// payloads; typed precision structures and symbols carry ABI 1.1 so an old
/// `float *` surface can never be reinterpreted as `double *`.
pub const GAFIME_PRECISION_ABI_VERSION_MAJOR: u16 = 1;
pub const GAFIME_PRECISION_ABI_VERSION_MINOR: u16 = 1;
pub const GAFIME_PRECISION_ABI_VERSION: u32 =
    ((GAFIME_PRECISION_ABI_VERSION_MAJOR as u32) << 16) | GAFIME_PRECISION_ABI_VERSION_MINOR as u32;
pub const GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR: u16 = 1;
/// Independent version for the optional resident semantic-arithmetic table.
/// It is additive beside the frozen matrix ABI rather than a revision of it.
pub const GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR: u16 = 1;
/// Minor 2 requires immutable-batch descriptor totals for native forecasts.
pub const GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR: u16 = 2;
pub const GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION: u32 =
    ((GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR as u32) << 16)
        | GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR as u32;
pub const GAFIME_ABI_IGNORABLE_FLAG_MASK: u32 = 0xffff_0000;
pub const GAFIME_ABI_REQUIRED_FLAG_MASK: u32 = 0x0000_ffff;

pub const GAFIME_LAUNCH_FLAG_GRAPH: u32 = 0x1;
/// Opt-in: use the fixed equal-width-bin MI (approximation backend, matches the
/// GPU) instead of the default adaptive-quantile MI on the CPU.
pub const GAFIME_LAUNCH_FLAG_MI_APPROX: u32 = 0x2;
/// The caller guarantees that protocol descriptor buffers remain immutable until
/// the resident matrix is uploaded or its target is updated. Backends may reuse
/// uploaded descriptors within that matrix-content epoch.
pub const GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL: u32 = 0x4;
pub const GAFIME_RESULT_FLAG_GRAPH_REPLAYED: u32 = 0x1;
pub const GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE: u32 = 0x1;
/// `GafimeLaunchProtocol::reserved` slot containing the caller-owned immutable
/// descriptor generation. Zero disables descriptor caching.
pub const GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT: usize = 0;

pub const GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY: u32 = 0x1;
pub const GAFIME_GPU_DEVICE_FLAG_INTEGRATED: u32 = 0x2;
pub const GAFIME_GPU_DEVICE_FLAG_DISCRETE: u32 = 0x4;
pub const GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY: u32 = 0x8;
pub const GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH: u32 = 0x10;
pub const GAFIME_GPU_DEVICE_FLAG_AMD_RDNA: u32 = 0x20;
pub const GAFIME_GPU_DEVICE_FLAG_AMD_CDNA: u32 = 0x40;
pub const GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY: u32 = 0x80;
/// Legacy ABI 1.0 capability for `GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL`.
/// This bit alone does not imply descriptor-generation support.
pub const GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL: u32 = 0x200;
/// The loaded payload keys immutable launch descriptors by the nonzero
/// generation in `GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT`.
pub const GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION: u32 = 0x400;
/// Legacy ABI 1.0 payload-wide MI accumulation mode. ABI 1.1 precision
/// execution derives MI arithmetic from the requested profile instead.
pub const GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64: u32 = 0x800;
/// Legacy device flag. ABI 1.1 f64 support is authoritative in the typed
/// authoritative numeric-route enumeration.
pub const GAFIME_GPU_DEVICE_FLAG_F64_STORAGE: u32 = 0x1000;

pub const GAFIME_GPU_ARCH_UNKNOWN: u64 = 0;
pub const GAFIME_GPU_ARCH_NVIDIA_TURING: u64 = 75;
pub const GAFIME_GPU_ARCH_NVIDIA_AMPERE: u64 = 80;
pub const GAFIME_GPU_ARCH_NVIDIA_ADA: u64 = 89;
pub const GAFIME_GPU_ARCH_NVIDIA_HOPPER: u64 = 90;
pub const GAFIME_GPU_ARCH_NVIDIA_BLACKWELL: u64 = 100;
pub const GAFIME_GPU_ARCH_AMD_RDNA: u64 = 1000;
pub const GAFIME_GPU_ARCH_AMD_CDNA: u64 = 2000;
pub const GAFIME_GPU_ARCH_APPLE: u64 = 3000;

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
pub const GAFIME_DTYPE_F64: DataType = 2;
pub const GAFIME_DTYPE_MASK_F32: u32 = 0x1;
pub const GAFIME_DTYPE_MASK_F64: u32 = 0x2;
pub const GAFIME_OVERFLOW_IEEE: u32 = 1;
pub const GAFIME_NUMERIC_ROUTE_FP32: u32 = 1;
pub const GAFIME_NUMERIC_ROUTE_MIXED: u32 = 2;
pub const GAFIME_NUMERIC_ROUTE_FP64: u32 = 3;
pub const GAFIME_BUFFER_FLAG_HOST: u32 = 0x1;
pub const GAFIME_BUFFER_FLAG_CONTIGUOUS: u32 = 0x2;

/// Canonical public precision profile. Structural planner values remain their
/// existing integer types; this enum identifies only the four floating-point
/// execution domains covered by the precision contract.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PrecisionProfile {
    Fp32 = 1,
    #[default]
    Mixed = 2,
    Fp64 = 3,
}

pub const GAFIME_PRECISION_FP32: u32 = PrecisionProfile::Fp32 as u32;
pub const GAFIME_PRECISION_MIXED: u32 = PrecisionProfile::Mixed as u32;
pub const GAFIME_PRECISION_FP64: u32 = PrecisionProfile::Fp64 as u32;
pub const GAFIME_PRECISION_PROFILE_MASK_FP32: u32 = 0x1;
pub const GAFIME_PRECISION_PROFILE_MASK_MIXED: u32 = 0x2;
pub const GAFIME_PRECISION_PROFILE_MASK_FP64: u32 = 0x4;

impl PrecisionProfile {
    pub const fn storage_dtype(self) -> DataType {
        match self {
            Self::Fp32 | Self::Mixed => GAFIME_DTYPE_F32,
            Self::Fp64 => GAFIME_DTYPE_F64,
        }
    }

    pub const fn result_dtype(self) -> DataType {
        match self {
            Self::Fp32 => GAFIME_DTYPE_F32,
            Self::Mixed | Self::Fp64 => GAFIME_DTYPE_F64,
        }
    }

    pub const fn capability_mask(self) -> u32 {
        match self {
            Self::Fp32 => GAFIME_PRECISION_PROFILE_MASK_FP32,
            Self::Mixed => GAFIME_PRECISION_PROFILE_MASK_MIXED,
            Self::Fp64 => GAFIME_PRECISION_PROFILE_MASK_FP64,
        }
    }

    pub const fn numeric_route(self) -> GafimeNumericRoute {
        match self {
            Self::Fp32 => GafimeNumericRoute::fp32(),
            Self::Mixed => GafimeNumericRoute::mixed(),
            Self::Fp64 => GafimeNumericRoute::fp64(),
        }
    }
}

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
pub const GAFIME_INPUT_HOST_F64: InputSourceKind = 5;

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeSliceF64 {
    pub ptr: *const f64,
    pub len: u64,
}

impl Default for GafimeSliceF64 {
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
            // -2 is the internal encoding of Python None: use every feature.
            // -1 retains the legacy guarded power-user mode.
            max_feature_candidate: -2,
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

/// Canonical ABI 1.1 numeric route. The record describes one supported
/// four-domain combination; its dtype fields are not independent knobs.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeNumericRoute {
    pub abi_version: u32,
    pub struct_size: u32,
    pub route_id: u32,
    pub profile: u32,
    pub storage_dtype: DataType,
    pub pointwise_dtype: DataType,
    pub reduction_dtype: DataType,
    pub result_dtype: DataType,
    pub overflow_policy: u32,
    pub flags: u32,
    pub reserved: [u64; 8],
}

impl GafimeNumericRoute {
    const fn new(
        route_id: u32,
        profile: u32,
        storage_dtype: DataType,
        pointwise_dtype: DataType,
        reduction_dtype: DataType,
        result_dtype: DataType,
    ) -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            route_id,
            profile,
            storage_dtype,
            pointwise_dtype,
            reduction_dtype,
            result_dtype,
            overflow_policy: GAFIME_OVERFLOW_IEEE,
            flags: 0,
            reserved: [0; 8],
        }
    }

    pub const fn fp32() -> Self {
        Self::new(
            GAFIME_NUMERIC_ROUTE_FP32,
            GAFIME_PRECISION_FP32,
            GAFIME_DTYPE_F32,
            GAFIME_DTYPE_F32,
            GAFIME_DTYPE_F32,
            GAFIME_DTYPE_F32,
        )
    }

    pub const fn mixed() -> Self {
        Self::new(
            GAFIME_NUMERIC_ROUTE_MIXED,
            GAFIME_PRECISION_MIXED,
            GAFIME_DTYPE_F32,
            GAFIME_DTYPE_F32,
            GAFIME_DTYPE_F64,
            GAFIME_DTYPE_F64,
        )
    }

    pub const fn fp64() -> Self {
        Self::new(
            GAFIME_NUMERIC_ROUTE_FP64,
            GAFIME_PRECISION_FP64,
            GAFIME_DTYPE_F64,
            GAFIME_DTYPE_F64,
            GAFIME_DTYPE_F64,
            GAFIME_DTYPE_F64,
        )
    }

    pub const fn for_profile(profile: PrecisionProfile) -> Self {
        profile.numeric_route()
    }
}

impl Default for GafimeNumericRoute {
    fn default() -> Self {
        Self::mixed()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeConstBufferView {
    pub abi_version: u32,
    pub struct_size: u32,
    pub dtype: DataType,
    pub flags: u32,
    pub data: *const c_void,
    pub element_count: u64,
    pub byte_length: u64,
    pub byte_stride: u64,
    pub reserved: [u64; 4],
}

impl Default for GafimeConstBufferView {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            dtype: 0,
            flags: GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS,
            data: core::ptr::null(),
            element_count: 0,
            byte_length: 0,
            byte_stride: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeMutableBufferView {
    pub abi_version: u32,
    pub struct_size: u32,
    pub dtype: DataType,
    pub flags: u32,
    pub data: *mut c_void,
    pub element_capacity: u64,
    pub byte_length: u64,
    pub byte_stride: u64,
    pub reserved: [u64; 4],
}

impl Default for GafimeMutableBufferView {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            dtype: 0,
            flags: GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS,
            data: core::ptr::null_mut(),
            element_capacity: 0,
            byte_length: 0,
            byte_stride: 0,
            reserved: [0; 4],
        }
    }
}

/// Opaque owner returned by the optional semantic-arithmetic table.  It is a
/// physical resident-column bank, never an evidence or feature identity.
pub type GafimeGpuSemanticBank = *mut c_void;

pub type SemanticProgramOp = u32;
pub const GAFIME_SEMANTIC_PROGRAM_SOURCE: SemanticProgramOp = 1;
pub const GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE: SemanticProgramOp = 2;
pub const GAFIME_SEMANTIC_PROGRAM_SOFTSIGN: SemanticProgramOp = 3;
pub const GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT: SemanticProgramOp = 4;
pub const GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE: u32 = 0x1;
pub const GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE: u32 = 0x2;
pub const GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN: u32 = 0x4;
pub const GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT: u32 = 0x8;

pub type SemanticPrimitiveKind = u32;
pub const GAFIME_SEMANTIC_PRIMITIVE_PAIRWISE_PEARSON: SemanticPrimitiveKind = 1;
pub const GAFIME_SEMANTIC_PRIMITIVE_ORDERED_EDGE_ENERGY: SemanticPrimitiveKind = 2;
pub const GAFIME_SEMANTIC_PRIMITIVE_SPARSE_GATHER: SemanticPrimitiveKind = 3;
pub const GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_PEARSON: u32 = 0x1;
pub const GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY: u32 = 0x2;
pub const GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER: u32 = 0x4;
pub const GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON: u32 = 0x1;
pub const GAFIME_SEMANTIC_STATISTIC_MASK_SPEARMAN: u32 = 0x2;
pub const GAFIME_SEMANTIC_STATISTIC_MASK_FIXED_CORRECTED_NMI: u32 = 0x4;

pub type SemanticPearsonMode = u32;
pub const GAFIME_SEMANTIC_PEARSON_SIGNED: SemanticPearsonMode = 1;
pub const GAFIME_SEMANTIC_PEARSON_ABSOLUTE: SemanticPearsonMode = 2;

pub type SemanticScalarState = u32;
pub const GAFIME_SEMANTIC_SCALAR_MEASURED: SemanticScalarState = 1;
pub const GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT: SemanticScalarState = 2;
pub const GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND: SemanticScalarState = 3;
pub const GAFIME_SEMANTIC_SCALAR_DEGENERATE_REDUCTION: SemanticScalarState = 4;
pub const GAFIME_SEMANTIC_SCALAR_NONFINITE_REDUCTION: SemanticScalarState = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticCapabilities {
    pub abi_version: u32,
    pub struct_size: u32,
    pub backend_kind: BackendKind,
    pub device_id: u32,
    pub profile_mask: u32,
    pub program_op_mask: u32,
    pub primitive_mask: u32,
    pub association_statistic_mask: u32,
    pub flags: u32,
    pub max_program_nodes: u32,
    pub max_slot_count: u32,
    pub max_rows: u64,
    pub max_gather_rows: u64,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticCapabilities {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            backend_kind: 0,
            device_id: 0,
            profile_mask: 0,
            program_op_mask: 0,
            primitive_mask: 0,
            association_statistic_mask: 0,
            flags: 0,
            max_program_nodes: 0,
            max_slot_count: 0,
            max_rows: 0,
            max_gather_rows: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticBankDesc {
    pub abi_version: u32,
    pub struct_size: u32,
    pub route: GafimeNumericRoute,
    pub layout: MatrixLayout,
    pub flags: u32,
    pub rows: u64,
    pub source_slots: u32,
    pub slot_capacity: u32,
    pub bytes: u64,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticBankDesc {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            route: GafimeNumericRoute::mixed(),
            layout: GAFIME_MATRIX_COLUMN_MAJOR,
            flags: 0,
            rows: 0,
            source_slots: 0,
            slot_capacity: 0,
            bytes: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GafimeSemanticProgramNode {
    pub opcode: SemanticProgramOp,
    pub output_slot: u32,
    pub operand_offset: u32,
    pub operand_count: u32,
    pub mean_offset: u32,
    pub mean_count: u32,
    pub reserved: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticProgramBatch {
    pub abi_version: u32,
    pub struct_size: u32,
    pub route: GafimeNumericRoute,
    pub nodes: *const GafimeSemanticProgramNode,
    pub node_count: u32,
    pub reserved32: u32,
    pub operand_slots: GafimeSliceU32,
    pub mean_bits: GafimeSliceU64,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticProgramBatch {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            route: GafimeNumericRoute::mixed(),
            nodes: core::ptr::null(),
            node_count: 0,
            reserved32: 0,
            operand_slots: GafimeSliceU32::default(),
            mean_bits: GafimeSliceU64::default(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticPearsonBatch {
    pub abi_version: u32,
    pub struct_size: u32,
    pub mode: SemanticPearsonMode,
    pub flags: u32,
    pub left_slots: GafimeSliceU32,
    pub right_slots: GafimeSliceU32,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticPearsonBatch {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            mode: GAFIME_SEMANTIC_PEARSON_SIGNED,
            flags: 0,
            left_slots: GafimeSliceU32::default(),
            right_slots: GafimeSliceU32::default(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticEdge {
    pub left_row: u64,
    pub right_row: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticEdgeEnergyBatch {
    pub abi_version: u32,
    pub struct_size: u32,
    pub flags: u32,
    pub reserved32: u32,
    pub edges: *const GafimeSemanticEdge,
    pub edge_count: u64,
    pub weights: GafimeConstBufferView,
    pub candidate_slots: GafimeSliceU32,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticEdgeEnergyBatch {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            flags: 0,
            reserved32: 0,
            edges: core::ptr::null(),
            edge_count: 0,
            weights: GafimeConstBufferView::default(),
            candidate_slots: GafimeSliceU32::default(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticSparseGatherBatch {
    pub abi_version: u32,
    pub struct_size: u32,
    pub flags: u32,
    pub reserved32: u32,
    pub source_slots: GafimeSliceU32,
    pub destination_slots: GafimeSliceU32,
    pub row_indices: GafimeSliceU64,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticSparseGatherBatch {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            flags: 0,
            reserved32: 0,
            source_slots: GafimeSliceU32::default(),
            destination_slots: GafimeSliceU32::default(),
            row_indices: GafimeSliceU64::default(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticScalarResultTable {
    pub abi_version: u32,
    pub struct_size: u32,
    pub route: GafimeNumericRoute,
    pub flags: u32,
    pub reserved32: u32,
    pub capacity: u64,
    pub count: u64,
    pub values: GafimeMutableBufferView,
    pub states: *mut SemanticScalarState,
    pub supports: *mut u64,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticScalarResultTable {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            route: GafimeNumericRoute::mixed(),
            flags: 0,
            reserved32: 0,
            capacity: 0,
            count: 0,
            values: GafimeMutableBufferView::default(),
            states: core::ptr::null_mut(),
            supports: core::ptr::null_mut(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticForecastRequest {
    pub abi_version: u32,
    pub struct_size: u32,
    /// Preserved semantic-ABI 1.1 per-node maximum. It is descriptive only
    /// once the payload uses immutable flattened program descriptors.
    pub program_max_operand_count: u64,
    pub pair_count: u64,
    pub graph_candidate_count: u64,
    pub graph_edge_count: u64,
    pub gather_slot_count: u64,
    pub gather_row_count: u64,
    pub retained_slot_count: u64,
    /// Exact `u32` physical-slot descriptor length for one program batch.
    pub program_operand_count: u64,
    /// Exact `u64` frozen-mean descriptor length for one program batch.
    pub program_mean_count: u64,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticForecastRequest {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            program_max_operand_count: 0,
            pair_count: 0,
            graph_candidate_count: 0,
            graph_edge_count: 0,
            gather_slot_count: 0,
            gather_row_count: 0,
            retained_slot_count: 0,
            program_operand_count: 0,
            program_mean_count: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeSemanticMemoryForecast {
    pub abi_version: u32,
    pub struct_size: u32,
    pub resident_bytes: u64,
    pub transient_bytes: u64,
    pub retained_bytes: u64,
    pub reserved: [u64; 8],
}

impl Default for GafimeSemanticMemoryForecast {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            resident_bytes: 0,
            transient_bytes: 0,
            retained_bytes: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeNumericMatrixDesc {
    pub abi_version: u32,
    pub struct_size: u32,
    pub route: GafimeNumericRoute,
    pub layout: MatrixLayout,
    pub flags: u32,
    pub rows: u64,
    pub cols: u32,
    pub row_stride: u32,
    pub bytes: u64,
    pub reserved: [u64; 8],
}

impl Default for GafimeNumericMatrixDesc {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            route: GafimeNumericRoute::mixed(),
            layout: GAFIME_MATRIX_ROW_MAJOR,
            flags: 0,
            rows: 0,
            cols: 0,
            row_stride: 0,
            bytes: 0,
            reserved: [0; 8],
        }
    }
}

/// ABI 1.1 matrix descriptor. The profile is part of resident state identity
/// and must match every typed upload, execution, graph, and target update.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimePrecisionMatrixDesc {
    pub abi_version: u32,
    pub profile: u32,
    pub dtype: DataType,
    pub layout: MatrixLayout,
    pub flags: u32,
    pub reserved32: u32,
    pub rows: u64,
    pub cols: u32,
    pub row_stride: u32,
    pub bytes: u64,
    pub reserved: [u64; 8],
}

impl Default for GafimePrecisionMatrixDesc {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: GAFIME_PRECISION_MIXED,
            dtype: GAFIME_DTYPE_F32,
            layout: GAFIME_MATRIX_ROW_MAJOR,
            flags: 0,
            reserved32: 0,
            rows: 0,
            cols: 0,
            row_stride: 0,
            bytes: 0,
            reserved: [0; 8],
        }
    }
}

/// ABI 1.1 profile capabilities. Profile masks state executable combinations;
/// dtype masks state physically accepted storage and public-result widths.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimePrecisionCapabilities {
    pub abi_version: u32,
    pub backend_kind: BackendKind,
    pub profile_mask: u32,
    pub storage_dtype_mask: u32,
    pub result_dtype_mask: u32,
    pub flags: u32,
    pub reserved: [u64; 8],
}

impl Default for GafimePrecisionCapabilities {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            backend_kind: 0,
            profile_mask: 0,
            storage_dtype_mask: 0,
            result_dtype_mask: 0,
            flags: 0,
            reserved: [0; 8],
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

/// ABI 1.1 wrapper around the immutable ABI 1.0 structural protocol. The
/// profile becomes part of descriptor and graph identity without widening any
/// planner bookkeeping field.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimePrecisionLaunchProtocol {
    pub abi_version: u32,
    pub profile: u32,
    pub base: *const GafimeLaunchProtocol,
    pub reserved: [u64; 8],
}

impl Default for GafimePrecisionLaunchProtocol {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: GAFIME_PRECISION_MIXED,
            base: core::ptr::null(),
            reserved: [0; 8],
        }
    }
}

/// Canonical native ABI 1.1 launch wrapper. The orchestration-only
/// `GafimePrecisionLaunchProtocol` remains an internal Rust planning type;
/// payloads receive this complete numeric route instead.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeNumericLaunchProtocol {
    pub abi_version: u32,
    pub struct_size: u32,
    pub route: GafimeNumericRoute,
    pub base: *const GafimeLaunchProtocol,
    pub reserved: [u64; 8],
}

impl Default for GafimeNumericLaunchProtocol {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            route: GafimeNumericRoute::mixed(),
            base: core::ptr::null(),
            reserved: [0; 8],
        }
    }
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

/// ABI 1.1 fp64 public result table used by mixed and fp64 profiles.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeResultTableF64 {
    pub abi_version: u32,
    pub max_arity: u32,
    pub metric_count: u32,
    pub flags: u32,
    pub capacity: u64,
    pub row_count: u64,
    pub combo_indices: *mut u32,
    pub metric_values: *mut f64,
    pub ranks: *mut u32,
    pub families: *mut u32,
    pub candidate_ids: *mut u64,
    pub row_flags: *mut u32,
    pub backend_private: *mut c_void,
    pub reserved: [u64; 8],
}

impl Default for GafimeResultTableF64 {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
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

/// Canonical ABI 1.1 result table. Structural arrays retain explicit integer
/// types while metric/ranking values use one checked mutable numeric view.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeNumericResultTable {
    pub abi_version: u32,
    pub struct_size: u32,
    pub max_arity: u32,
    pub metric_count: u32,
    pub flags: u32,
    pub reserved32: u32,
    pub capacity: u64,
    pub row_count: u64,
    pub combo_indices: *mut u32,
    pub metric_values: GafimeMutableBufferView,
    pub ranks: *mut u32,
    pub families: *mut u32,
    pub candidate_ids: *mut u64,
    pub row_flags: *mut u32,
    pub reserved: [u64; 8],
}

impl Default for GafimeNumericResultTable {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            max_arity: 0,
            metric_count: 0,
            flags: 0,
            reserved32: 0,
            capacity: 0,
            row_count: 0,
            combo_indices: core::ptr::null_mut(),
            metric_values: GafimeMutableBufferView::default(),
            ranks: core::ptr::null_mut(),
            families: core::ptr::null_mut(),
            candidate_ids: core::ptr::null_mut(),
            row_flags: core::ptr::null_mut(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimePermutationSignificanceTable {
    pub abi_version: u32,
    pub metric_count: u32,
    pub row_count: u64,
    pub candidate_ids: *const u64,
    pub observed_metric_values: *const f32,
    pub p_values: *mut f32,
    pub reserved: [u64; 8],
}

impl Default for GafimePermutationSignificanceTable {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_ABI_VERSION,
            metric_count: 0,
            row_count: 0,
            candidate_ids: core::ptr::null(),
            observed_metric_values: core::ptr::null(),
            p_values: core::ptr::null_mut(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimePermutationSignificanceTableF64 {
    pub abi_version: u32,
    pub metric_count: u32,
    pub row_count: u64,
    pub candidate_ids: *const u64,
    pub observed_metric_values: *const f64,
    pub p_values: *mut f64,
    pub reserved: [u64; 8],
}

impl Default for GafimePermutationSignificanceTableF64 {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            metric_count: 0,
            row_count: 0,
            candidate_ids: core::ptr::null(),
            observed_metric_values: core::ptr::null(),
            p_values: core::ptr::null_mut(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeNumericSignificanceTable {
    pub abi_version: u32,
    pub struct_size: u32,
    pub metric_count: u32,
    pub flags: u32,
    pub row_count: u64,
    pub candidate_ids: *const u64,
    pub observed_metric_values: GafimeConstBufferView,
    pub p_values: GafimeMutableBufferView,
    pub reserved: [u64; 8],
}

impl Default for GafimeNumericSignificanceTable {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            metric_count: 0,
            flags: 0,
            row_count: 0,
            candidate_ids: core::ptr::null(),
            observed_metric_values: GafimeConstBufferView::default(),
            p_values: GafimeMutableBufferView::default(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeInteractionDiagnosticBatch {
    pub abi_version: u32,
    pub max_arity: u32,
    pub row_count: u64,
    pub combo_indices: *const u32,
    pub combo_index_count: u64,
    pub overflow_row_counts: *mut u64,
    pub flags: *mut u32,
    pub reserved32: u32,
    pub reserved: [u64; 7],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GafimeNumericInteractionDiagnosticBatch {
    pub abi_version: u32,
    pub struct_size: u32,
    pub route: GafimeNumericRoute,
    pub max_arity: u32,
    pub flags: u32,
    pub row_count: u64,
    pub combo_indices: *const u32,
    pub combo_index_count: u64,
    pub overflow_row_counts: *mut u64,
    pub row_flags: *mut u32,
    pub reserved32: u32,
    pub reserved: [u64; 7],
}

impl Default for GafimeNumericInteractionDiagnosticBatch {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            struct_size: core::mem::size_of::<Self>() as u32,
            route: GafimeNumericRoute::mixed(),
            max_arity: 0,
            flags: 0,
            row_count: 0,
            combo_indices: core::ptr::null(),
            combo_index_count: 0,
            overflow_row_counts: core::ptr::null_mut(),
            row_flags: core::ptr::null_mut(),
            reserved32: 0,
            reserved: [0; 7],
        }
    }
}

impl Default for GafimeInteractionDiagnosticBatch {
    fn default() -> Self {
        Self {
            abi_version: GAFIME_ABI_VERSION,
            max_arity: 0,
            row_count: 0,
            combo_indices: core::ptr::null(),
            combo_index_count: 0,
            overflow_row_counts: core::ptr::null_mut(),
            flags: core::ptr::null_mut(),
            reserved32: 0,
            reserved: [0; 7],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};
    use memoffset::offset_of;

    const GPU_ABI_HEADER: &str = include_str!("../../../src/common/gafime_gpu_abi.hpp");
    const SEMANTIC_ABI_HEADER: &str =
        include_str!("../../../src/common/gafime_semantic_primitives_abi.hpp");

    #[test]
    fn slices_are_c_pointer_len_pairs() {
        assert_eq!(size_of::<GafimeSliceU32>(), 16);
        assert_eq!(size_of::<GafimeSliceU64>(), 16);
        assert_eq!(size_of::<GafimeSliceF32>(), 16);
        assert_eq!(size_of::<GafimeSliceF64>(), 16);
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

    #[test]
    fn semantic_abi_header_and_rust_layouts_stay_in_lockstep() {
        for needle in [
            "#define GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR 1u",
            "#define GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR 2u",
            "typedef struct GafimeSemanticCapabilities",
            "typedef struct GafimeSemanticBankDesc",
            "typedef struct GafimeSemanticProgramNode",
            "typedef struct GafimeSemanticProgramBatch",
            "typedef struct GafimeSemanticPearsonBatch",
            "typedef struct GafimeSemanticEdgeEnergyBatch",
            "typedef struct GafimeSemanticSparseGatherBatch",
            "typedef struct GafimeSemanticScalarResultTable",
            "typedef struct GafimeSemanticForecastRequest",
            "program_max_operand_count",
            "program_operand_count",
            "program_mean_count",
            "gather_slot_count",
            "gather_row_count",
            "gafime_gpu_semantic_bank_download_v1",
        ] {
            assert!(
                SEMANTIC_ABI_HEADER.contains(needle),
                "missing semantic C ABI header marker: {needle}"
            );
        }

        assert_eq!(GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION, (1u32 << 16) | 2);
        assert_eq!(size_of::<GafimeSemanticCapabilities>(), 128);
        assert_eq!(offset_of!(GafimeSemanticCapabilities, max_rows), 48);
        assert_eq!(offset_of!(GafimeSemanticCapabilities, reserved), 64);
        assert_eq!(size_of::<GafimeSemanticBankDesc>(), 208);
        assert_eq!(offset_of!(GafimeSemanticBankDesc, route), 8);
        assert_eq!(offset_of!(GafimeSemanticBankDesc, reserved), 144);
        assert_eq!(size_of::<GafimeSemanticProgramNode>(), 40);
        assert_eq!(offset_of!(GafimeSemanticProgramNode, reserved), 24);
        assert_eq!(size_of::<GafimeSemanticProgramBatch>(), 224);
        assert_eq!(offset_of!(GafimeSemanticProgramBatch, operand_slots), 128);
        assert_eq!(size_of::<GafimeSemanticPearsonBatch>(), 112);
        assert_eq!(offset_of!(GafimeSemanticPearsonBatch, left_slots), 16);
        assert_eq!(size_of::<GafimeSemanticEdgeEnergyBatch>(), 192);
        assert_eq!(offset_of!(GafimeSemanticEdgeEnergyBatch, weights), 32);
        assert_eq!(size_of::<GafimeSemanticSparseGatherBatch>(), 128);
        assert_eq!(offset_of!(GafimeSemanticSparseGatherBatch, row_indices), 48);
        assert_eq!(size_of::<GafimeSemanticScalarResultTable>(), 296);
        assert_eq!(offset_of!(GafimeSemanticScalarResultTable, values), 136);
        assert_eq!(offset_of!(GafimeSemanticScalarResultTable, reserved), 232);
        assert_eq!(size_of::<GafimeSemanticForecastRequest>(), 144);
        assert_eq!(offset_of!(GafimeSemanticForecastRequest, reserved), 80);
        assert_eq!(size_of::<GafimeSemanticMemoryForecast>(), 96);
        assert_eq!(offset_of!(GafimeSemanticMemoryForecast, reserved), 32);
    }

    #[test]
    fn gpu_abi_header_and_rust_layouts_stay_in_lockstep() {
        for needle in [
            "#define GAFIME_ABI_VERSION_MAJOR 1u",
            "#define GAFIME_ABI_VERSION_MINOR 0u",
            "#define GAFIME_PRECISION_ABI_VERSION_MINOR 1u",
            "#define GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR 1u",
            "#define GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL 0x4u",
            "#define GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT 0u",
            "#define GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY 0x1u",
            "#define GAFIME_GPU_DEVICE_FLAG_AMD_RDNA 0x20u",
            "#define GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY 0x80u",
            "#define GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL 0x200u",
            "#define GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION 0x400u",
            "#define GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64 0x800u",
            "#define GAFIME_GPU_DEVICE_FLAG_F64_STORAGE 0x1000u",
            "#define GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE 0x1u",
            "#define GAFIME_GPU_ARCH_NVIDIA_ADA 89u",
            "#define GAFIME_GPU_ARCH_AMD_CDNA 2000u",
            "GAFIME_BACKEND_CUDA = 2",
            "GAFIME_METRIC_R2 = 4",
            "GAFIME_DTYPE_F64 = 2",
            "GAFIME_PRECISION_MIXED = 2",
            "#define GAFIME_NUMERIC_ROUTE_MIXED 2u",
            "typedef struct GafimeNumericRoute",
            "typedef struct GafimeConstBufferView",
            "typedef struct GafimeMutableBufferView",
            "typedef struct GafimeNumericMatrixDesc",
            "typedef struct GafimeNumericLaunchProtocol",
            "typedef struct GafimeNumericResultTable",
            "typedef struct GafimeNumericSignificanceTable",
            "typedef struct GafimeNumericInteractionDiagnosticBatch",
            "typedef struct GafimeMatrixDesc",
            "typedef struct GafimeGpuDeviceInfo",
            "typedef struct GafimeGpuGraphCapability",
            "typedef struct GafimeShapeHint",
            "typedef struct GafimeArityChunk",
            "typedef struct GafimeRankSpec",
            "typedef struct GafimePermutationSchedule",
            "typedef struct GafimeLaunchProtocol",
            "typedef struct GafimeResultTable",
            "typedef struct GafimePermutationSignificanceTable",
            "typedef struct GafimeInteractionDiagnosticBatch",
            "gafime_gpu_permutation_pvalues",
            "gafime_gpu_numeric_routes_v2",
            "gafime_gpu_matrix_upload_v2",
            "gafime_gpu_execute_v2",
            "gafime_gpu_permutation_pvalues_v2",
            "gafime_gpu_interaction_diagnostics",
            "uint64_t reserved[8];",
        ] {
            assert!(
                GPU_ABI_HEADER.contains(needle),
                "missing C ABI header marker"
            );
        }

        for obsolete in [
            "gafime_gpu_precision_capabilities",
            "gafime_gpu_matrix_upload_f32_v2",
            "gafime_gpu_matrix_upload_f64_v2",
            "gafime_gpu_matrix_update_target_f32_v2",
            "gafime_gpu_matrix_update_target_f64_v2",
            "gafime_gpu_execute_f32_v2",
            "gafime_gpu_execute_f64_v2",
            "gafime_gpu_permutation_pvalues_f32_v2",
            "gafime_gpu_permutation_pvalues_f64_v2",
        ] {
            assert!(
                !GPU_ABI_HEADER.contains(obsolete),
                "obsolete pre-freeze ABI 1.1 symbol remained in the public header"
            );
        }

        assert_eq!(GAFIME_ABI_VERSION, (1u32 << 16));
        assert_eq!(GAFIME_PRECISION_ABI_VERSION, (1u32 << 16) | 1);
        assert_eq!(GAFIME_BACKEND_CUDA, 2);
        assert_eq!(GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL, 0x4);
        assert_eq!(GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT, 0);
        assert_eq!(GAFIME_METRIC_R2, 4);
        assert_eq!(GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY, 0x1);
        assert_eq!(GAFIME_GPU_DEVICE_FLAG_AMD_RDNA, 0x20);
        assert_eq!(GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY, 0x80);
        assert_eq!(GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL, 0x200);
        assert_eq!(GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION, 0x400);
        assert_eq!(GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64, 0x800);
        assert_eq!(GAFIME_GPU_DEVICE_FLAG_F64_STORAGE, 0x1000);
        assert_eq!(GAFIME_DTYPE_F64, 2);
        assert_eq!(PrecisionProfile::Mixed.storage_dtype(), GAFIME_DTYPE_F32);
        assert_eq!(PrecisionProfile::Mixed.result_dtype(), GAFIME_DTYPE_F64);
        assert_eq!(PrecisionProfile::Fp64.storage_dtype(), GAFIME_DTYPE_F64);
        assert_eq!(GAFIME_GPU_ARCH_NVIDIA_ADA, 89);
        assert_eq!(GAFIME_GPU_ARCH_AMD_CDNA, 2000);

        assert_eq!(size_of::<GafimeMatrixDesc>(), 40);
        assert_eq!(offset_of!(GafimeMatrixDesc, rows), 16);
        assert_eq!(offset_of!(GafimeMatrixDesc, bytes), 32);

        assert_eq!(size_of::<GafimeNumericRoute>(), 104);
        assert_eq!(offset_of!(GafimeNumericRoute, reserved), 40);
        assert_eq!(size_of::<GafimeConstBufferView>(), 80);
        assert_eq!(offset_of!(GafimeConstBufferView, reserved), 48);
        assert_eq!(size_of::<GafimeMutableBufferView>(), 80);
        assert_eq!(offset_of!(GafimeMutableBufferView, reserved), 48);
        assert_eq!(size_of::<GafimeNumericMatrixDesc>(), 208);
        assert_eq!(offset_of!(GafimeNumericMatrixDesc, route), 8);
        assert_eq!(offset_of!(GafimeNumericMatrixDesc, rows), 120);
        assert_eq!(offset_of!(GafimeNumericMatrixDesc, reserved), 144);

        assert_eq!(size_of::<GafimePrecisionMatrixDesc>(), 112);
        assert_eq!(offset_of!(GafimePrecisionMatrixDesc, rows), 24);
        assert_eq!(offset_of!(GafimePrecisionMatrixDesc, bytes), 40);
        assert_eq!(offset_of!(GafimePrecisionMatrixDesc, reserved), 48);

        assert_eq!(size_of::<GafimePrecisionCapabilities>(), 88);
        assert_eq!(offset_of!(GafimePrecisionCapabilities, reserved), 24);

        assert_eq!(size_of::<GafimeGpuDeviceInfo>(), 240);
        assert_eq!(offset_of!(GafimeGpuDeviceInfo, name), 16);
        assert_eq!(offset_of!(GafimeGpuDeviceInfo, total_global_mem_bytes), 144);
        assert_eq!(offset_of!(GafimeGpuDeviceInfo, reserved), 176);

        assert_eq!(size_of::<GafimeGpuGraphCapability>(), 104);
        assert_eq!(
            offset_of!(GafimeGpuGraphCapability, stable_pointer_flags),
            32
        );
        assert_eq!(offset_of!(GafimeGpuGraphCapability, reserved), 40);

        assert_eq!(size_of::<GafimeShapeHint>(), 64);
        assert_eq!(offset_of!(GafimeShapeHint, reserved), 32);

        assert_eq!(size_of::<GafimeArityChunk>(), 56);
        assert_eq!(offset_of!(GafimeArityChunk, combo_row_offset), 16);
        assert_eq!(offset_of!(GafimeArityChunk, descriptor_offset), 40);

        assert_eq!(size_of::<GafimeRankSpec>(), 48);
        assert_eq!(offset_of!(GafimeRankSpec, reserved), 16);

        assert_eq!(size_of::<GafimePermutationSchedule>(), 72);
        assert_eq!(offset_of!(GafimePermutationSchedule, target_offsets), 24);
        assert_eq!(offset_of!(GafimePermutationSchedule, reserved), 40);

        assert_eq!(size_of::<GafimeLaunchProtocol>(), 280);
        assert_eq!(offset_of!(GafimeLaunchProtocol, combo_indices), 32);
        assert_eq!(offset_of!(GafimeLaunchProtocol, chunks), 64);
        assert_eq!(offset_of!(GafimeLaunchProtocol, shape_hints), 80);
        assert_eq!(offset_of!(GafimeLaunchProtocol, rank), 96);
        assert_eq!(offset_of!(GafimeLaunchProtocol, permutations), 144);
        assert_eq!(offset_of!(GafimeLaunchProtocol, reserved), 216);

        assert_eq!(size_of::<GafimePrecisionLaunchProtocol>(), 80);
        assert_eq!(offset_of!(GafimePrecisionLaunchProtocol, base), 8);
        assert_eq!(offset_of!(GafimePrecisionLaunchProtocol, reserved), 16);

        assert_eq!(size_of::<GafimeNumericLaunchProtocol>(), 184);
        assert_eq!(offset_of!(GafimeNumericLaunchProtocol, route), 8);
        assert_eq!(offset_of!(GafimeNumericLaunchProtocol, base), 112);
        assert_eq!(offset_of!(GafimeNumericLaunchProtocol, reserved), 120);

        assert_eq!(size_of::<GafimeResultTable>(), 152);
        assert_eq!(offset_of!(GafimeResultTable, capacity), 16);
        assert_eq!(offset_of!(GafimeResultTable, combo_indices), 32);
        assert_eq!(offset_of!(GafimeResultTable, backend_private), 80);
        assert_eq!(offset_of!(GafimeResultTable, reserved), 88);

        assert_eq!(size_of::<GafimeResultTableF64>(), 152);
        assert_eq!(offset_of!(GafimeResultTableF64, metric_values), 40);
        assert_eq!(offset_of!(GafimeResultTableF64, reserved), 88);

        assert_eq!(size_of::<GafimeNumericResultTable>(), 224);
        assert_eq!(offset_of!(GafimeNumericResultTable, capacity), 24);
        assert_eq!(offset_of!(GafimeNumericResultTable, metric_values), 48);
        assert_eq!(offset_of!(GafimeNumericResultTable, reserved), 160);

        assert_eq!(size_of::<GafimePermutationSignificanceTable>(), 104);
        assert_eq!(offset_of!(GafimePermutationSignificanceTable, row_count), 8);
        assert_eq!(
            offset_of!(GafimePermutationSignificanceTable, candidate_ids),
            16
        );
        assert_eq!(offset_of!(GafimePermutationSignificanceTable, p_values), 32);
        assert_eq!(offset_of!(GafimePermutationSignificanceTable, reserved), 40);

        assert_eq!(size_of::<GafimePermutationSignificanceTableF64>(), 104);
        assert_eq!(
            offset_of!(
                GafimePermutationSignificanceTableF64,
                observed_metric_values
            ),
            24
        );

        assert_eq!(size_of::<GafimeNumericSignificanceTable>(), 256);
        assert_eq!(offset_of!(GafimeNumericSignificanceTable, row_count), 16);
        assert_eq!(
            offset_of!(GafimeNumericSignificanceTable, observed_metric_values),
            32
        );
        assert_eq!(offset_of!(GafimeNumericSignificanceTable, p_values), 112);
        assert_eq!(offset_of!(GafimeNumericSignificanceTable, reserved), 192);

        assert_eq!(size_of::<GafimeInteractionDiagnosticBatch>(), 112);
        assert_eq!(offset_of!(GafimeInteractionDiagnosticBatch, row_count), 8);
        assert_eq!(
            offset_of!(GafimeInteractionDiagnosticBatch, combo_indices),
            16
        );
        assert_eq!(
            offset_of!(GafimeInteractionDiagnosticBatch, overflow_row_counts),
            32
        );
        assert_eq!(offset_of!(GafimeInteractionDiagnosticBatch, flags), 40);
        assert_eq!(offset_of!(GafimeInteractionDiagnosticBatch, reserved), 56);

        assert_eq!(size_of::<GafimeNumericInteractionDiagnosticBatch>(), 224);
        assert_eq!(
            offset_of!(GafimeNumericInteractionDiagnosticBatch, route),
            8
        );
        assert_eq!(
            offset_of!(GafimeNumericInteractionDiagnosticBatch, row_count),
            120
        );
        assert_eq!(
            offset_of!(GafimeNumericInteractionDiagnosticBatch, reserved),
            168
        );
    }
}
