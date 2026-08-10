//! Standalone Rust ABI 1.1 consumer.
//!
//! This source intentionally does not depend on gafime-gpu-sys or any private
//! Rust type.  Its repr(C) declarations and symbol types are derived only from
//! the published C ABI and are checked against explicit layout constants.

#![allow(non_camel_case_types)]

use std::env;
use std::ffi::{c_char, c_void, CString};
use std::mem::{align_of, offset_of, size_of};
use std::ptr;

const ABI_1_0: u32 = (1 << 16) | 0;
const ABI_1_1: u32 = (1 << 16) | 1;
const STATUS_OK: i32 = 0;
const STATUS_UNSUPPORTED: i32 = -2;
const STATUS_DEVICE_ERROR: i32 = -4;
const DTYPE_F32: u32 = 1;
const DTYPE_F64: u32 = 2;
const PROFILE_FP32: u32 = 1;
const PROFILE_MIXED: u32 = 2;
const PROFILE_FP64: u32 = 3;
const ROUTE_FP32: u32 = 1;
const ROUTE_MIXED: u32 = 2;
const ROUTE_FP64: u32 = 3;
const OVERFLOW_IEEE: u32 = 1;
const BUFFER_HOST_CONTIGUOUS: u32 = 0x1 | 0x2;
const ABI_IGNORABLE_FLAG_MASK: u32 = 0xffff_0000;
const ABI_REQUIRED_FLAG_MASK: u32 = 0x0000_ffff;
const MATRIX_ROW_MAJOR: u32 = 1;
const METRIC_PEARSON: u32 = 1;
const FAMILY_CONTINUOUS: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct NumericRoute {
    abi_version: u32,
    struct_size: u32,
    route_id: u32,
    profile: u32,
    storage_dtype: u32,
    pointwise_dtype: u32,
    reduction_dtype: u32,
    result_dtype: u32,
    overflow_policy: u32,
    flags: u32,
    reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FutureRouteRecord {
    known: NumericRoute,
    future_fields: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConstBufferView {
    abi_version: u32,
    struct_size: u32,
    dtype: u32,
    flags: u32,
    data: *const c_void,
    element_count: u64,
    byte_length: u64,
    byte_stride: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MutableBufferView {
    abi_version: u32,
    struct_size: u32,
    dtype: u32,
    flags: u32,
    data: *mut c_void,
    element_capacity: u64,
    byte_length: u64,
    byte_stride: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NumericMatrixDesc {
    abi_version: u32,
    struct_size: u32,
    route: NumericRoute,
    layout: u32,
    flags: u32,
    rows: u64,
    cols: u32,
    row_stride: u32,
    bytes: u64,
    reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SliceU32 {
    ptr: *const u32,
    len: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SliceU64 {
    ptr: *const u64,
    len: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ArityChunk {
    arity: u32,
    family: u32,
    metric_mask: u32,
    shape_hint_index: u32,
    combo_row_offset: u64,
    combo_count: u64,
    local_chunk_id: u32,
    flags: u32,
    descriptor_offset: u64,
    descriptor_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ShapeHint {
    threads_per_block: u32,
    items_per_thread: u32,
    blocks_per_sm: u32,
    min_blocks: u32,
    shared_bytes: u32,
    register_budget: u32,
    occupancy_target_pct: u32,
    vendor_hint: u32,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RankSpec {
    top_k: u32,
    primary_metric: u32,
    descending: u32,
    include_ties: u32,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PermutationSchedule {
    permutation_count: u32,
    mode: u32,
    flags: u32,
    reserved32: u32,
    seed: u64,
    target_offsets: SliceU64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LaunchProtocol {
    abi_version: u32,
    backend_kind: u32,
    flags: u32,
    max_arity: u32,
    n_samples: u64,
    n_features: u32,
    family_count: u32,
    combo_indices: SliceU32,
    metric_ids: SliceU32,
    chunks: *const ArityChunk,
    chunk_count: u32,
    reserved32_a: u32,
    shape_hints: *const ShapeHint,
    shape_hint_count: u32,
    reserved32_b: u32,
    rank: RankSpec,
    permutations: PermutationSchedule,
    reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NumericLaunchProtocol {
    abi_version: u32,
    struct_size: u32,
    route: NumericRoute,
    base: *const LaunchProtocol,
    reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NumericResultTable {
    abi_version: u32,
    struct_size: u32,
    max_arity: u32,
    metric_count: u32,
    flags: u32,
    reserved32: u32,
    capacity: u64,
    row_count: u64,
    combo_indices: *mut u32,
    metric_values: MutableBufferView,
    ranks: *mut u32,
    families: *mut u32,
    candidate_ids: *mut u64,
    row_flags: *mut u32,
    reserved: [u64; 8],
}

const _: () = assert!(size_of::<NumericRoute>() == 104);
const _: () = assert!(align_of::<NumericRoute>() == 8);
const _: () = assert!(offset_of!(NumericRoute, route_id) == 8);
const _: () = assert!(offset_of!(NumericRoute, result_dtype) == 28);
const _: () = assert!(offset_of!(NumericRoute, reserved) == 40);
const _: () = assert!(size_of::<FutureRouteRecord>() == 120);
const _: () = assert!(align_of::<FutureRouteRecord>() == 8);
const _: () = assert!(size_of::<ConstBufferView>() == 80);
const _: () = assert!(align_of::<ConstBufferView>() == 8);
const _: () = assert!(offset_of!(ConstBufferView, data) == 16);
const _: () = assert!(offset_of!(ConstBufferView, reserved) == 48);
const _: () = assert!(size_of::<MutableBufferView>() == 80);
const _: () = assert!(align_of::<MutableBufferView>() == 8);
const _: () = assert!(size_of::<NumericMatrixDesc>() == 208);
const _: () = assert!(align_of::<NumericMatrixDesc>() == 8);
const _: () = assert!(offset_of!(NumericMatrixDesc, reserved) == 144);
const _: () = assert!(size_of::<LaunchProtocol>() == 280);
const _: () = assert!(align_of::<LaunchProtocol>() == 8);
const _: () = assert!(offset_of!(LaunchProtocol, rank) == 96);
const _: () = assert!(offset_of!(LaunchProtocol, permutations) == 144);
const _: () = assert!(size_of::<NumericLaunchProtocol>() == 184);
const _: () = assert!(align_of::<NumericLaunchProtocol>() == 8);
const _: () = assert!(offset_of!(NumericLaunchProtocol, base) == 112);
const _: () = assert!(size_of::<NumericResultTable>() == 224);
const _: () = assert!(align_of::<NumericResultTable>() == 8);
const _: () = assert!(offset_of!(NumericResultTable, metric_values) == 48);

type NumericRoutesFn = unsafe extern "C" fn(u32, u32, u32, *mut NumericRoute, u32, *mut u32) -> i32;
type MatrixAllocFn = unsafe extern "C" fn(u32, *const NumericMatrixDesc, *mut *mut c_void) -> i32;
type MatrixUploadFn = unsafe extern "C" fn(
    *mut c_void,
    *const NumericRoute,
    *const ConstBufferView,
    *const ConstBufferView,
    u64,
    u32,
) -> i32;
type MatrixUpdateTargetFn =
    unsafe extern "C" fn(*mut c_void, *const NumericRoute, *const ConstBufferView, u64) -> i32;
type ExecuteFn =
    unsafe extern "C" fn(*mut c_void, *const NumericLaunchProtocol, *mut NumericResultTable) -> i32;
type ExecutionMemoryFn =
    unsafe extern "C" fn(*mut c_void, *const NumericLaunchProtocol, *mut u64) -> i32;
type PermutationMemoryFn =
    unsafe extern "C" fn(*mut c_void, *const NumericLaunchProtocol, u64, *mut u64) -> i32;
type PermutationFn =
    unsafe extern "C" fn(*mut c_void, *const NumericLaunchProtocol, *mut c_void) -> i32;
type DiagnosticsFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type MatrixFreeFn = unsafe extern "C" fn(*mut c_void) -> i32;

#[allow(dead_code)]
struct Api {
    routes: NumericRoutesFn,
    alloc: MatrixAllocFn,
    upload: MatrixUploadFn,
    update_target: MatrixUpdateTargetFn,
    execute: ExecuteFn,
    execution_memory: ExecutionMemoryFn,
    permutation_memory: PermutationMemoryFn,
    permutation: PermutationFn,
    diagnostics: DiagnosticsFn,
    free_matrix: MatrixFreeFn,
}

#[cfg(unix)]
mod dynamic {
    use super::*;

    #[cfg_attr(not(target_os = "macos"), link(name = "dl"))]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> i32;
        fn dlerror() -> *const c_char;
    }

    pub struct Library(*mut c_void);

    impl Library {
        pub fn open(path: &str) -> Result<Self, String> {
            let path = CString::new(path).map_err(|error| error.to_string())?;
            let handle = unsafe { dlopen(path.as_ptr(), 2) };
            if handle.is_null() {
                let error = unsafe { dlerror() };
                let message = if error.is_null() {
                    "dlopen failed without an error string".to_owned()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(error) }
                        .to_string_lossy()
                        .into_owned()
                };
                Err(message)
            } else {
                Ok(Self(handle))
            }
        }

        pub unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T, String> {
            assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
            let name = CString::new(name).map_err(|error| error.to_string())?;
            let raw = unsafe { dlsym(self.0, name.as_ptr()) };
            if raw.is_null() {
                return Err(format!("missing symbol {}", name.to_string_lossy()));
            }
            Ok(unsafe { std::mem::transmute_copy(&raw) })
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            unsafe {
                dlclose(self.0);
            }
        }
    }
}

#[cfg(windows)]
mod dynamic {
    use super::*;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(filename: *const c_char) -> *mut c_void;
        fn GetProcAddress(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn FreeLibrary(handle: *mut c_void) -> i32;
    }

    pub struct Library(*mut c_void);

    impl Library {
        pub fn open(path: &str) -> Result<Self, String> {
            let path = CString::new(path).map_err(|error| error.to_string())?;
            let handle = unsafe { LoadLibraryA(path.as_ptr()) };
            if handle.is_null() {
                Err("LoadLibraryA failed".to_owned())
            } else {
                Ok(Self(handle))
            }
        }

        pub unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T, String> {
            assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
            let name = CString::new(name).map_err(|error| error.to_string())?;
            let raw = unsafe { GetProcAddress(self.0, name.as_ptr()) };
            if raw.is_null() {
                return Err(format!("missing symbol {}", name.to_string_lossy()));
            }
            Ok(unsafe { std::mem::transmute_copy(&raw) })
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            unsafe {
                FreeLibrary(self.0);
            }
        }
    }
}

fn zeroed<T>() -> T {
    unsafe { std::mem::zeroed() }
}

fn dtype_size(dtype: u32) -> u64 {
    match dtype {
        DTYPE_F32 => size_of::<f32>() as u64,
        DTYPE_F64 => size_of::<f64>() as u64,
        _ => 0,
    }
}

fn const_view(dtype: u32, data: *const c_void, count: u64) -> ConstBufferView {
    ConstBufferView {
        abi_version: ABI_1_1,
        struct_size: size_of::<ConstBufferView>() as u32,
        dtype,
        flags: BUFFER_HOST_CONTIGUOUS,
        data,
        element_count: count,
        byte_length: count * dtype_size(dtype),
        byte_stride: dtype_size(dtype),
        reserved: [0; 4],
    }
}

fn mutable_view(dtype: u32, data: *mut c_void, count: u64) -> MutableBufferView {
    MutableBufferView {
        abi_version: ABI_1_1,
        struct_size: size_of::<MutableBufferView>() as u32,
        dtype,
        flags: BUFFER_HOST_CONTIGUOUS,
        data,
        element_capacity: count,
        byte_length: count * dtype_size(dtype),
        byte_stride: dtype_size(dtype),
        reserved: [0; 4],
    }
}

fn canonical(route: &NumericRoute, route_stride: u32) -> bool {
    if route.abi_version >> 16 != 1
        || route.struct_size < offset_of!(NumericRoute, reserved) as u32
        || route.overflow_policy != OVERFLOW_IEEE
        || route.flags & ABI_REQUIRED_FLAG_MASK != 0
        || (route_stride >= size_of::<NumericRoute>() as u32
            && route.struct_size >= size_of::<NumericRoute>() as u32
            && route.reserved != [0; 8])
    {
        return false;
    }
    match route.route_id {
        ROUTE_FP32 => {
            route.profile == PROFILE_FP32
                && route.storage_dtype == DTYPE_F32
                && route.pointwise_dtype == DTYPE_F32
                && route.reduction_dtype == DTYPE_F32
                && route.result_dtype == DTYPE_F32
        }
        ROUTE_MIXED => {
            route.profile == PROFILE_MIXED
                && route.storage_dtype == DTYPE_F32
                && route.pointwise_dtype == DTYPE_F32
                && route.reduction_dtype == DTYPE_F64
                && route.result_dtype == DTYPE_F64
        }
        ROUTE_FP64 => {
            route.profile == PROFILE_FP64
                && route.storage_dtype == DTYPE_F64
                && route.pointwise_dtype == DTYPE_F64
                && route.reduction_dtype == DTYPE_F64
                && route.result_dtype == DTYPE_F64
        }
        _ => false,
    }
}

fn known_route_id(route_id: u32) -> bool {
    matches!(route_id, ROUTE_FP32 | ROUTE_MIXED | ROUTE_FP64)
}

fn expected_route_mask(expected: u32) -> Result<u32, String> {
    match expected {
        1 => Ok(1 << ROUTE_FP32),
        3 => Ok((1 << ROUTE_FP32) | (1 << ROUTE_MIXED) | (1 << ROUTE_FP64)),
        _ => Err(format!("unsupported expected route count {expected}")),
    }
}

/* Returns Some(normalized known route), None for an unknown additive route. */
fn parse_route_record(
    record: &FutureRouteRecord,
    route_stride: u32,
    seen_ids: &mut Vec<u32>,
) -> Result<Option<NumericRoute>, String> {
    let mut route = record.known;
    let stable_prefix = offset_of!(NumericRoute, reserved) as u32;
    let route_size = route.struct_size;
    let major = route.abi_version >> 16;
    let minor = route.abi_version & 0xffff;
    if major != 1
        || minor < (ABI_1_1 & 0xffff)
        || route_size < stable_prefix
        || route.route_id == 0
        || route.flags & ABI_REQUIRED_FLAG_MASK != 0
    {
        return Err(format!("invalid route prefix/id {}", route.route_id));
    }
    if route_stride >= size_of::<NumericRoute>() as u32
        && route_size >= size_of::<NumericRoute>() as u32
        && route.reserved != [0; 8]
    {
        return Err(format!(
            "nonzero reserved fields in route {}",
            route.route_id
        ));
    }
    if seen_ids.contains(&route.route_id) {
        return Err(format!("duplicate route record {}", route.route_id));
    }
    seen_ids.push(route.route_id);
    if !known_route_id(route.route_id) {
        /* Unknown profile/dtype/overflow values are never dispatched. */
        return Ok(None);
    }
    if !canonical(&route, route_stride) {
        return Err(format!("contradictory known route {}", route.route_id));
    }
    /* Normalize before copying into a fixed ABI 1.1 embedded route field. */
    route.struct_size = size_of::<NumericRoute>() as u32;
    Ok(Some(route))
}

fn collect_route_records(
    records: &[FutureRouteRecord],
    expected_mask: u32,
) -> Result<Vec<NumericRoute>, String> {
    let route_stride = size_of::<FutureRouteRecord>() as u32;
    let mut seen_ids = Vec::with_capacity(records.len());
    let mut known_routes = Vec::new();
    let mut known_mask = 0_u32;
    for record in records {
        if let Some(route) = parse_route_record(record, route_stride, &mut seen_ids)? {
            known_mask |= 1 << route.route_id;
            known_routes.push(route);
        }
    }
    if known_mask != expected_mask {
        return Err(format!(
            "known route mask {known_mask:#x}, expected {expected_mask:#x}"
        ));
    }
    Ok(known_routes)
}

fn unknown_future_route() -> FutureRouteRecord {
    let mut record = zeroed::<FutureRouteRecord>();
    record.known.abi_version = (1 << 16) | 2;
    record.known.struct_size = size_of::<FutureRouteRecord>() as u32;
    record.known.route_id = 0x10001;
    record.known.profile = 0x10001;
    record.known.storage_dtype = 0x10001;
    record.known.pointwise_dtype = 0x10001;
    record.known.reduction_dtype = 0x10001;
    record.known.result_dtype = 0x10001;
    record.known.overflow_policy = 0x10001;
    record.known.flags = ABI_IGNORABLE_FLAG_MASK;
    record.future_fields[0] = 0x1234_5678_9abc_def0;
    record
}

fn adversarial_route_fixture_tests(
    api: &Api,
    backend: u32,
    expected_mask: u32,
    current_routes: &[NumericRoute],
) -> Result<(), String> {
    let ids: Vec<u32> = if expected_mask == (1 << ROUTE_FP32) {
        vec![ROUTE_FP32]
    } else {
        vec![ROUTE_FP32, ROUTE_MIXED, ROUTE_FP64]
    };
    if current_routes.len() != ids.len() {
        return Err(format!(
            "current route count {} does not match expected {}",
            current_routes.len(),
            ids.len()
        ));
    }
    let mut records: Vec<FutureRouteRecord> = current_routes
        .iter()
        .map(|route| {
            let mut record = zeroed::<FutureRouteRecord>();
            record.known = *route;
            record.known.abi_version = (1 << 16) | 2;
            record.known.struct_size = size_of::<FutureRouteRecord>() as u32;
            record
        })
        .collect();
    records.push(unknown_future_route());
    let known = collect_route_records(&records, expected_mask)?;
    if known.len() != ids.len() {
        return Err(format!(
            "future route fixture retained {} known routes",
            known.len()
        ));
    }
    for route in known {
        if let Err(status) = unsafe { run_route(api, route, backend) } {
            if status == STATUS_UNSUPPORTED || status == STATUS_DEVICE_ERROR {
                std::process::exit(77);
            }
            return Err(format!(
                "future route {} lifecycle failed: {status}",
                route.route_id
            ));
        }
    }

    let mut duplicate = records.clone();
    duplicate.push(unknown_future_route());
    if collect_route_records(&duplicate, expected_mask).is_ok() {
        return Err("duplicate unknown route ID was accepted".to_owned());
    }

    let mut contradictory = records[..ids.len()].to_vec();
    contradictory[0].known.result_dtype = if contradictory[0].known.result_dtype == DTYPE_F32 {
        DTYPE_F64
    } else {
        DTYPE_F32
    };
    if collect_route_records(&contradictory, expected_mask).is_ok() {
        return Err("contradictory known route was accepted".to_owned());
    }

    let mut required_flag = records.clone();
    required_flag[ids.len()].known.flags = 1;
    if collect_route_records(&required_flag, expected_mask).is_ok() {
        return Err("unknown required route flag was accepted".to_owned());
    }

    let mut major_mismatch = records.clone();
    major_mismatch[ids.len()].known.abi_version = 2 << 16;
    if collect_route_records(&major_mismatch, expected_mask).is_ok() {
        return Err("future route major mismatch was accepted".to_owned());
    }

    /* A producer may report a larger ABI 1.2 record than this consumer's
     * caller stride. The unknown tail is not read, so the stable prefix still
     * parses and the unknown route is skipped. */
    let mut oversized_claim = records;
    oversized_claim[ids.len()].known.struct_size = size_of::<FutureRouteRecord>() as u32 + 8;
    if collect_route_records(&oversized_claim, expected_mask).is_err() {
        return Err("larger producer route record was rejected".to_owned());
    }
    Ok(())
}

unsafe fn run_route(api: &Api, route: NumericRoute, backend: u32) -> Result<(), i32> {
    let features_f32 = [1.0_f32, 7.0, 2.0, 5.0, 3.0, 3.0, 4.0, 1.0];
    let target_f32 = [1.0_f32, 2.0, 3.0, 4.0];
    let epsilon = 2.0_f64.powi(-30);
    let features_f64 = [
        1.0 + epsilon,
        7.0,
        2.0 + epsilon,
        5.0,
        3.0 + epsilon,
        3.0,
        4.0 + epsilon,
        1.0,
    ];
    let target_f64 = [1.0 + epsilon, 2.0 + epsilon, 3.0 + epsilon, 4.0 + epsilon];
    let feature_ptr = if route.storage_dtype == DTYPE_F32 {
        features_f32.as_ptr().cast()
    } else {
        features_f64.as_ptr().cast()
    };
    let target_ptr = if route.storage_dtype == DTYPE_F32 {
        target_f32.as_ptr().cast()
    } else {
        target_f64.as_ptr().cast()
    };
    let desc = NumericMatrixDesc {
        abi_version: ABI_1_1,
        struct_size: size_of::<NumericMatrixDesc>() as u32,
        route,
        layout: MATRIX_ROW_MAJOR,
        flags: 0,
        rows: 4,
        cols: 2,
        row_stride: 2,
        bytes: 8 * dtype_size(route.storage_dtype),
        reserved: [0; 8],
    };
    let mut matrix = ptr::null_mut();
    let status = unsafe { (api.alloc)(0, &desc, &mut matrix) };
    if status != STATUS_OK || matrix.is_null() {
        return Err(status);
    }
    let feature_view = const_view(route.storage_dtype, feature_ptr, 8);
    let target_view = const_view(route.storage_dtype, target_ptr, 4);
    let status = unsafe { (api.upload)(matrix, &route, &feature_view, &target_view, 4, 2) };
    if status != STATUS_OK {
        unsafe { (api.free_matrix)(matrix) };
        return Err(status);
    }

    let combo = 0_u32;
    let metric = METRIC_PEARSON;
    let chunk = ArityChunk {
        arity: 1,
        family: FAMILY_CONTINUOUS,
        metric_mask: 0,
        shape_hint_index: 0,
        combo_row_offset: 0,
        combo_count: 1,
        local_chunk_id: 0,
        flags: 0,
        descriptor_offset: 0,
        descriptor_count: 1,
    };
    let mut base: LaunchProtocol = zeroed();
    base.abi_version = ABI_1_0;
    base.backend_kind = backend;
    base.max_arity = 1;
    base.n_samples = 4;
    base.n_features = 2;
    base.family_count = 1;
    base.combo_indices = SliceU32 {
        ptr: &combo,
        len: 1,
    };
    base.metric_ids = SliceU32 {
        ptr: &metric,
        len: 1,
    };
    base.chunks = &chunk;
    base.chunk_count = 1;
    let protocol = NumericLaunchProtocol {
        abi_version: ABI_1_1,
        struct_size: size_of::<NumericLaunchProtocol>() as u32,
        route,
        base: &base,
        reserved: [0; 8],
    };

    let mut combo_out = u32::MAX;
    let mut metric_f32 = 0.0_f32;
    let mut metric_f64 = 0.0_f64;
    let metric_ptr = if route.result_dtype == DTYPE_F32 {
        (&mut metric_f32 as *mut f32).cast()
    } else {
        (&mut metric_f64 as *mut f64).cast()
    };
    let mut rank = 0_u32;
    let mut family = 0_u32;
    let mut candidate_id = u64::MAX;
    let mut row_flags = u32::MAX;
    let mut result = NumericResultTable {
        abi_version: ABI_1_1,
        struct_size: size_of::<NumericResultTable>() as u32,
        max_arity: 1,
        metric_count: 1,
        flags: 0,
        reserved32: 0,
        capacity: 1,
        row_count: 0,
        combo_indices: &mut combo_out,
        metric_values: mutable_view(route.result_dtype, metric_ptr, 1),
        ranks: &mut rank,
        families: &mut family,
        candidate_ids: &mut candidate_id,
        row_flags: &mut row_flags,
        reserved: [0; 8],
    };
    let execute_status = unsafe { (api.execute)(matrix, &protocol, &mut result) };
    let free_status = unsafe { (api.free_matrix)(matrix) };
    if execute_status != STATUS_OK || free_status != STATUS_OK {
        return Err(if execute_status != STATUS_OK {
            execute_status
        } else {
            free_status
        });
    }
    let visible = if route.result_dtype == DTYPE_F32 {
        metric_f32 as f64
    } else {
        metric_f64
    };
    if result.row_count != 1
        || combo_out != 0
        || !visible.is_finite()
        || (visible - 1.0).abs() > 1.0e-5
    {
        eprintln!(
            "route {} result mismatch: rows={} combo={} value={visible:.17}",
            route.route_id, result.row_count, combo_out
        );
        return Err(-100);
    }
    Ok(())
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(format!(
            "usage: {} PAYLOAD BACKEND_KIND EXPECTED_ROUTE_COUNT",
            args.first().map_or("abi_1_1_rust_consumer", String::as_str)
        ));
    }
    let backend = args[2].parse::<u32>().map_err(|error| error.to_string())?;
    let expected = args[3].parse::<u32>().map_err(|error| error.to_string())?;
    let expected_mask = expected_route_mask(expected)?;
    let library = dynamic::Library::open(&args[1])?;
    let api = unsafe {
        Api {
            routes: library.symbol("gafime_gpu_numeric_routes_v2")?,
            alloc: library.symbol("gafime_gpu_matrix_alloc_v2")?,
            upload: library.symbol("gafime_gpu_matrix_upload_v2")?,
            update_target: library.symbol("gafime_gpu_matrix_update_target_v2")?,
            execute: library.symbol("gafime_gpu_execute_v2")?,
            execution_memory: library.symbol("gafime_gpu_execution_memory_peak_v2")?,
            permutation_memory: library.symbol("gafime_gpu_permutation_memory_peak_v2")?,
            permutation: library.symbol("gafime_gpu_permutation_pvalues_v2")?,
            diagnostics: library.symbol("gafime_gpu_interaction_diagnostics_v2")?,
            free_matrix: library.symbol("gafime_gpu_matrix_free_v2")?,
        }
    };
    let mut count = 0_u32;
    let route_stride = size_of::<FutureRouteRecord>() as u32;
    let status = unsafe { (api.routes)(0, ABI_1_1, route_stride, ptr::null_mut(), 0, &mut count) };
    if status == STATUS_UNSUPPORTED || status == STATUS_DEVICE_ERROR {
        std::process::exit(77);
    }
    if status != STATUS_OK || count < expected || count > 16 {
        return Err(format!(
            "route count mismatch: status={status} count={count} expected={expected}"
        ));
    }
    let mut routes = vec![zeroed::<FutureRouteRecord>(); count as usize];
    let mut capacity = count;
    let status = unsafe {
        (api.routes)(
            0,
            ABI_1_1,
            route_stride,
            routes.as_mut_ptr().cast::<NumericRoute>(),
            capacity,
            &mut capacity,
        )
    };
    if status != STATUS_OK {
        return Err(format!("route enumeration failed: {status}"));
    }
    if capacity as usize > routes.len() {
        return Err(format!(
            "payload returned route count {capacity} above capacity"
        ));
    }
    routes.truncate(capacity as usize);
    let known_routes = collect_route_records(&routes, expected_mask)?;
    if known_routes.len() != expected as usize {
        return Err(format!(
            "known route count {} does not match expected {expected}",
            known_routes.len()
        ));
    }
    adversarial_route_fixture_tests(&api, backend, expected_mask, &known_routes)?;
    Ok(())
}

fn main() {
    if let Err(error) = real_main() {
        eprintln!("ABI 1.1 Rust consumer: {error}");
        std::process::exit(1);
    }
}
