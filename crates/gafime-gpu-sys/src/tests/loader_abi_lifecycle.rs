use super::*;
use gafime_orchestrator::OrchestratorError;

static OWNED_MATRIX_FREES: AtomicUsize = AtomicUsize::new(0);
static PRECISION_MATRIX_ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_F32_UPLOAD_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_F64_UPLOAD_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_F32_UPDATE_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_F64_UPDATE_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_F32_EXECUTE_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_F64_EXECUTE_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_EXECUTION_PEAK_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_PERMUTATION_PEAK_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_F32_PERMUTATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static PRECISION_F64_PERMUTATION_CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn owned_matrix_free(matrix: GafimeGpuMatrix) {
    if !matrix.is_null() {
        // SAFETY: this callback is paired only with test_matrix_alloc.
        unsafe { drop(Box::from_raw(matrix.cast::<u8>())) };
        OWNED_MATRIX_FREES.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn owned_matrix_free_v2(matrix: GafimeGpuMatrix) -> GafimeStatus {
    // SAFETY: this is the status-returning ABI 1.1 adapter for the same test
    // allocation owned by `owned_matrix_free`.
    unsafe { owned_matrix_free(matrix) };
    GAFIME_STATUS_OK
}

unsafe extern "C" fn count_precision_matrix_alloc(
    device_id: u32,
    matrix_desc: *const GafimeNumericMatrixDesc,
    matrix_out: *mut GafimeGpuMatrix,
) -> GafimeStatus {
    PRECISION_MATRIX_ALLOC_CALLS.fetch_add(1, Ordering::SeqCst);
    // SAFETY: this wrapper forwards the exact ABI arguments to the paired test
    // allocator after recording whether preflight reached native allocation.
    unsafe { test_matrix_alloc_v2(device_id, matrix_desc, matrix_out) }
}

unsafe extern "C" fn test_matrix_upload_v2(
    _matrix: GafimeGpuMatrix,
    route: *const GafimeNumericRoute,
    _features_host: *const GafimeConstBufferView,
    _target_host: *const GafimeConstBufferView,
    _rows: u64,
    _cols: u32,
) -> GafimeStatus {
    if route.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the test caller supplies one live route record.
    match unsafe { (*route).storage_dtype } {
        GAFIME_DTYPE_F32 => PRECISION_F32_UPLOAD_CALLS.fetch_add(1, Ordering::SeqCst),
        GAFIME_DTYPE_F64 => PRECISION_F64_UPLOAD_CALLS.fetch_add(1, Ordering::SeqCst),
        _ => return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT,
    };
    GAFIME_STATUS_OK
}

unsafe extern "C" fn test_matrix_update_target_v2(
    _matrix: GafimeGpuMatrix,
    route: *const GafimeNumericRoute,
    _target_host: *const GafimeConstBufferView,
    _rows: u64,
) -> GafimeStatus {
    if route.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the test caller supplies one live route record.
    match unsafe { (*route).storage_dtype } {
        GAFIME_DTYPE_F32 => PRECISION_F32_UPDATE_CALLS.fetch_add(1, Ordering::SeqCst),
        GAFIME_DTYPE_F64 => PRECISION_F64_UPDATE_CALLS.fetch_add(1, Ordering::SeqCst),
        _ => return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT,
    };
    GAFIME_STATUS_OK
}

unsafe extern "C" fn test_execute_v2(
    _matrix: GafimeGpuMatrix,
    protocol: *const GafimeNumericLaunchProtocol,
    _result_out: *mut GafimeNumericResultTable,
) -> GafimeStatus {
    if protocol.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the test caller supplies one live numeric launch protocol.
    match unsafe { (*protocol).route.result_dtype } {
        GAFIME_DTYPE_F32 => PRECISION_F32_EXECUTE_CALLS.fetch_add(1, Ordering::SeqCst),
        GAFIME_DTYPE_F64 => PRECISION_F64_EXECUTE_CALLS.fetch_add(1, Ordering::SeqCst),
        _ => return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT,
    };
    GAFIME_STATUS_OK
}

unsafe extern "C" fn test_execution_memory_peak_v2(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeNumericLaunchProtocol,
    _peak_bytes_out: *mut u64,
) -> GafimeStatus {
    PRECISION_EXECUTION_PEAK_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

unsafe extern "C" fn test_permutation_memory_peak_v2(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeNumericLaunchProtocol,
    _selected_row_count: u64,
    _peak_bytes_out: *mut u64,
) -> GafimeStatus {
    PRECISION_PERMUTATION_PEAK_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

unsafe extern "C" fn test_permutation_pvalues_v2(
    _matrix: GafimeGpuMatrix,
    protocol: *const GafimeNumericLaunchProtocol,
    _significance_out: *mut GafimeNumericSignificanceTable,
) -> GafimeStatus {
    if protocol.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the test caller supplies one live numeric launch protocol.
    match unsafe { (*protocol).route.result_dtype } {
        GAFIME_DTYPE_F32 => PRECISION_F32_PERMUTATION_CALLS.fetch_add(1, Ordering::SeqCst),
        GAFIME_DTYPE_F64 => PRECISION_F64_PERMUTATION_CALLS.fetch_add(1, Ordering::SeqCst),
        _ => return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT,
    };
    GAFIME_STATUS_OK
}

unsafe extern "C" fn test_interaction_diagnostics_v2(
    _matrix: GafimeGpuMatrix,
    _diagnostics: *mut GafimeNumericInteractionDiagnosticBatch,
) -> GafimeStatus {
    GAFIME_STATUS_OK
}

unsafe extern "C" fn test_numeric_routes_fp32_only(
    _device_id: u32,
    _consumer_abi_version: u32,
    route_stride: u32,
    routes_out: *mut GafimeNumericRoute,
    route_capacity: u32,
    route_count_out: *mut u32,
) -> GafimeStatus {
    if route_count_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the count slot was checked above.
    unsafe { *route_count_out = 1 };
    if routes_out.is_null() {
        return if route_capacity == 0 {
            GAFIME_STATUS_OK
        } else {
            gafime_types::GAFIME_STATUS_INVALID_ARGUMENT
        };
    }
    if route_capacity < 1 || route_stride < core::mem::size_of::<GafimeNumericRoute>() as u32 {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: capacity and stride were checked.
    unsafe { routes_out.write(GafimeNumericRoute::fp32()) };
    GAFIME_STATUS_OK
}

unsafe fn write_numeric_route_fixture(
    route_stride: u32,
    routes_out: *mut GafimeNumericRoute,
    route_capacity: u32,
    route_count_out: *mut u32,
    routes: &[GafimeNumericRoute],
) -> GafimeStatus {
    if route_count_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the required count slot was checked above.
    unsafe { *route_count_out = routes.len() as u32 };
    if routes_out.is_null() {
        return if route_capacity == 0 {
            GAFIME_STATUS_OK
        } else {
            gafime_types::GAFIME_STATUS_INVALID_ARGUMENT
        };
    }
    if route_capacity < routes.len() as u32
        || route_stride < core::mem::size_of::<GafimeNumericRoute>() as u32
    {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (index, route) in routes.iter().enumerate() {
        // SAFETY: capacity and stride were checked and each destination is a
        // distinct aligned caller-owned route record.
        let destination = unsafe {
            routes_out
                .cast::<u8>()
                .add(index * route_stride as usize)
                .cast::<GafimeNumericRoute>()
        };
        // SAFETY: the destination is writable for one current record.
        unsafe { destination.write(*route) };
    }
    GAFIME_STATUS_OK
}

unsafe extern "C" fn test_numeric_routes_with_unknown_future_route(
    _device_id: u32,
    _consumer_abi_version: u32,
    route_stride: u32,
    routes_out: *mut GafimeNumericRoute,
    route_capacity: u32,
    route_count_out: *mut u32,
) -> GafimeStatus {
    let mut future = GafimeNumericRoute::fp64();
    future.abi_version = (1u32 << 16) | 2;
    // A newer producer may report a larger record than this consumer's
    // caller-owned stride. The stable ABI 1.1 prefix remains usable and the
    // unknown tail is deliberately ignored.
    future.struct_size = 128;
    future.route_id = 0x1_0000;
    future.profile = 0x1_0000;
    future.storage_dtype = 0x1_0000;
    future.pointwise_dtype = 0x1_0001;
    future.reduction_dtype = 0x1_0002;
    future.result_dtype = GAFIME_DTYPE_F64;
    let mut fp32 = GafimeNumericRoute::fp32();
    fp32.abi_version = (1u32 << 16) | 2;
    fp32.struct_size = 128;
    let mut mixed = GafimeNumericRoute::mixed();
    mixed.abi_version = (1u32 << 16) | 2;
    mixed.struct_size = 128;
    let mut fp64 = GafimeNumericRoute::fp64();
    fp64.abi_version = (1u32 << 16) | 2;
    fp64.struct_size = 128;
    let routes = [future, fp32, mixed, fp64];
    // SAFETY: this extern fixture forwards the caller-owned storage to the
    // checked writer above.
    unsafe {
        write_numeric_route_fixture(
            route_stride,
            routes_out,
            route_capacity,
            route_count_out,
            &routes,
        )
    }
}

unsafe extern "C" fn test_numeric_routes_with_duplicate_unknown_future_route(
    _device_id: u32,
    _consumer_abi_version: u32,
    route_stride: u32,
    routes_out: *mut GafimeNumericRoute,
    route_capacity: u32,
    route_count_out: *mut u32,
) -> GafimeStatus {
    let mut future = GafimeNumericRoute::fp64();
    future.abi_version = (1u32 << 16) | 2;
    future.route_id = 0x1_0000;
    future.profile = 0x1_0000;
    future.storage_dtype = 0x1_0000;
    future.pointwise_dtype = 0x1_0001;
    future.reduction_dtype = 0x1_0002;
    future.result_dtype = GAFIME_DTYPE_F64;
    let routes = [future, future, GafimeNumericRoute::fp32()];
    // SAFETY: this extern fixture forwards the caller-owned storage to the
    // checked writer above.
    unsafe {
        write_numeric_route_fixture(
            route_stride,
            routes_out,
            route_capacity,
            route_count_out,
            &routes,
        )
    }
}

unsafe extern "C" fn test_numeric_routes_with_contradictory_known_id(
    _device_id: u32,
    _consumer_abi_version: u32,
    route_stride: u32,
    routes_out: *mut GafimeNumericRoute,
    route_capacity: u32,
    route_count_out: *mut u32,
) -> GafimeStatus {
    let mut contradictory = GafimeNumericRoute::fp32();
    contradictory.profile = 0x1_0000;
    let routes = [contradictory, GafimeNumericRoute::mixed()];
    // SAFETY: this extern fixture forwards the caller-owned storage to the
    // checked writer above.
    unsafe {
        write_numeric_route_fixture(
            route_stride,
            routes_out,
            route_capacity,
            route_count_out,
            &routes,
        )
    }
}

fn complete_precision_test_function_table() -> GpuFunctionTable {
    let mut functions = complete_test_function_table();
    functions.numeric_routes_v2 = Some(test_numeric_routes_v2);
    functions.matrix_alloc_v2 = Some(count_precision_matrix_alloc);
    functions.matrix_upload_v2 = Some(test_matrix_upload_v2);
    functions.matrix_update_target_v2 = Some(test_matrix_update_target_v2);
    functions.execute_v2 = Some(test_execute_v2);
    functions.execution_memory_peak_v2 = Some(test_execution_memory_peak_v2);
    functions.permutation_memory_peak_v2 = Some(test_permutation_memory_peak_v2);
    functions.permutation_pvalues_v2 = Some(test_permutation_pvalues_v2);
    functions.interaction_diagnostics_v2 = Some(test_interaction_diagnostics_v2);
    functions.matrix_free_v2 = Some(owned_matrix_free_v2);
    functions
}

fn precision_alloc_error(backend: &GpuBackend, precision: PrecisionProfile) -> GpuSysError {
    match backend.alloc_matrix_for_profile(precision, 2, 2) {
        Ok(_) => panic!("precision allocation unexpectedly passed preflight"),
        Err(error) => error,
    }
}

fn cross_generation_test_function_table() -> GpuFunctionTable {
    let mut functions = complete_test_function_table();
    functions.numeric_routes_v2 = Some(test_numeric_routes_v2);
    functions.matrix_alloc_v2 = Some(test_matrix_alloc_v2);
    functions.matrix_upload = Some(count_legacy_matrix_upload);
    functions.matrix_update_target = Some(count_legacy_matrix_update_target);
    functions.execute = Some(count_legacy_execute);
    functions.execution_memory_peak = Some(count_legacy_execution_memory_peak);
    functions.permutation_pvalues = Some(count_legacy_permutation_pvalues);
    functions.matrix_upload_v2 = Some(count_precision_matrix_upload_f32);
    functions.matrix_update_target_v2 = Some(count_precision_matrix_update_target_f32);
    functions.execute_v2 = Some(count_precision_execute_f64);
    functions.execution_memory_peak_v2 = Some(count_precision_execution_memory_peak);
    functions.permutation_memory_peak_v2 = Some(test_permutation_memory_peak_v2);
    functions.permutation_pvalues_v2 = Some(count_precision_permutation_pvalues_f64);
    functions.interaction_diagnostics_v2 = Some(test_interaction_diagnostics_v2);
    functions.matrix_free_v2 = Some(owned_matrix_free_v2);
    functions
}

#[test]
fn configured_payload_libraries_are_process_cached() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    assert_configured_library_is_process_cached(CUDA_LIBRARY_ENV, GAFIME_BACKEND_CUDA);
    assert_configured_library_is_process_cached(ROCM_LIBRARY_ENV, GAFIME_BACKEND_ROCM);
    assert_configured_library_is_process_cached(METAL_LIBRARY_ENV, GAFIME_BACKEND_METAL);
}

#[test]
fn gpu_backend_declares_vendor_kind() {
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
    assert_eq!(backend.backend_kind(), GAFIME_BACKEND_CUDA);
    assert!(!backend.supports_permutation_pvalues());
    assert!(!backend.supports_interaction_diagnostics());
    assert!(!backend.supports_permutation_memory_peak());
    assert!(!backend.supports_immutable_protocol());
    assert!(!backend.supports_descriptor_generation());
    assert!(!backend.uses_fp64_mi_accumulation());
    assert!(!backend.supports_f64_storage());
    assert!(matches!(
        GpuBackend::new(GAFIME_BACKEND_CPU, complete_test_function_table()),
        Err(GpuSysError::InvalidInput(
            "GPU backend kind must be CUDA, ROCm, or Metal"
        ))
    ));
}

#[test]
fn precision_allocation_preflight_distinguishes_legacy_partial_and_unsupported_payloads() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    PRECISION_MATRIX_ALLOC_CALLS.store(0, Ordering::SeqCst);
    let legacy = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
    assert!(matches!(
        precision_alloc_error(&legacy, PrecisionProfile::Fp32),
        GpuSysError::PrecisionAbiUnavailable
    ));
    assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 0);

    let mut missing_routes = complete_precision_test_function_table();
    missing_routes.numeric_routes_v2 = None;
    let partial = GpuBackend::new(GAFIME_BACKEND_CUDA, missing_routes).unwrap();
    assert!(matches!(
        precision_alloc_error(&partial, PrecisionProfile::Fp32),
        GpuSysError::MissingFunction("gafime_gpu_numeric_routes_v2")
    ));
    assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 0);

    let mut missing_alloc = complete_precision_test_function_table();
    missing_alloc.matrix_alloc_v2 = None;
    let partial = GpuBackend::new(GAFIME_BACKEND_CUDA, missing_alloc).unwrap();
    assert!(matches!(
        precision_alloc_error(&partial, PrecisionProfile::Fp32),
        GpuSysError::MissingFunction("gafime_gpu_matrix_alloc_v2")
    ));
    assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 0);

    let mut missing_peak = complete_precision_test_function_table();
    missing_peak.execution_memory_peak_v2 = None;
    let partial = GpuBackend::new(GAFIME_BACKEND_CUDA, missing_peak).unwrap();
    assert!(matches!(
        precision_alloc_error(&partial, PrecisionProfile::Fp32),
        GpuSysError::MissingFunction("gafime_gpu_execution_memory_peak_v2")
    ));
    assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 0);

    let mut unsupported = complete_precision_test_function_table();
    unsupported.numeric_routes_v2 = Some(test_numeric_routes_fp32_only);
    let unsupported = GpuBackend::new(GAFIME_BACKEND_CUDA, unsupported).unwrap();
    assert!(matches!(
        precision_alloc_error(&unsupported, PrecisionProfile::Mixed),
        GpuSysError::InvalidInput("requested precision profile is unsupported by this GPU payload")
    ));
    assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn numeric_route_negotiation_skips_one_unknown_future_route() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut functions = complete_precision_test_function_table();
    functions.numeric_routes_v2 = Some(test_numeric_routes_with_unknown_future_route);
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let routes = backend.numeric_routes().unwrap();
    assert_eq!(routes.len(), 3);
    assert_eq!(routes[0].route_id, GafimeNumericRoute::fp32().route_id);
    assert_eq!(routes[1].route_id, GafimeNumericRoute::mixed().route_id);
    assert_eq!(routes[2].route_id, GafimeNumericRoute::fp64().route_id);
}

#[test]
fn numeric_route_negotiation_rejects_duplicate_unknown_future_route_ids() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut functions = complete_precision_test_function_table();
    functions.numeric_routes_v2 = Some(test_numeric_routes_with_duplicate_unknown_future_route);
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    assert!(matches!(
        backend.numeric_routes(),
        Err(GpuSysError::InvalidInput(
            "GPU payload advertised a duplicate numeric-route ID"
        ))
    ));
}

#[test]
fn numeric_route_negotiation_rejects_a_known_id_with_an_unknown_profile() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut functions = complete_precision_test_function_table();
    functions.numeric_routes_v2 = Some(test_numeric_routes_with_contradictory_known_id);
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    assert!(matches!(
        backend.numeric_routes(),
        Err(GpuSysError::InvalidInput(
            "GPU payload advertised a contradictory known numeric route ID"
        ))
    ));
}

#[test]
fn precision_allocation_preflight_requires_one_generic_operation_surface() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    macro_rules! assert_missing_before_alloc {
        ($precision:expr, $field:ident, $symbol:literal) => {{
            PRECISION_MATRIX_ALLOC_CALLS.store(0, Ordering::SeqCst);
            let mut functions = complete_precision_test_function_table();
            functions.$field = None;
            let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
            assert!(matches!(
                precision_alloc_error(&backend, $precision),
                GpuSysError::MissingFunction($symbol)
            ));
            assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 0);
        }};
    }

    for precision in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        assert_missing_before_alloc!(precision, matrix_upload_v2, "gafime_gpu_matrix_upload_v2");
        assert_missing_before_alloc!(
            precision,
            matrix_update_target_v2,
            "gafime_gpu_matrix_update_target_v2"
        );
        assert_missing_before_alloc!(precision, execute_v2, "gafime_gpu_execute_v2");
        assert_missing_before_alloc!(precision, matrix_free_v2, "gafime_gpu_matrix_free_v2");
    }

    for counter in [
        &PRECISION_MATRIX_ALLOC_CALLS,
        &PRECISION_F32_UPLOAD_CALLS,
        &PRECISION_F64_UPLOAD_CALLS,
        &PRECISION_F32_UPDATE_CALLS,
        &PRECISION_F64_UPDATE_CALLS,
        &PRECISION_F32_EXECUTE_CALLS,
        &PRECISION_F64_EXECUTE_CALLS,
        &PRECISION_EXECUTION_PEAK_CALLS,
        &PRECISION_PERMUTATION_PEAK_CALLS,
        &PRECISION_F32_PERMUTATION_CALLS,
        &PRECISION_F64_PERMUTATION_CALLS,
    ] {
        counter.store(0, Ordering::SeqCst);
    }
    let mut backend = GpuBackend::new(
        GAFIME_BACKEND_CUDA,
        complete_precision_test_function_table(),
    )
    .unwrap();
    for precision in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let matrix = backend.alloc_matrix_for_profile(precision, 2, 2).unwrap();
        match precision {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                matrix.upload_f32_v2(&[0.0; 4], &[0.0; 2]).unwrap();
                matrix.update_target_f32_v2(&[0.0; 2]).unwrap();
            }
            PrecisionProfile::Fp64 => {
                matrix.upload_f64_v2(&[0.0; 4], &[0.0; 2]).unwrap();
                matrix.update_target_f64_v2(&[0.0; 2]).unwrap();
            }
        }
        let base = GafimeLaunchProtocol {
            abi_version: GAFIME_ABI_VERSION,
            backend_kind: GAFIME_BACKEND_CUDA,
            n_samples: 2,
            n_features: 2,
            ..Default::default()
        };
        let protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: precision as u32,
            base: &base,
            reserved: [0; 8],
        };
        assert_eq!(
            gafime_orchestrator::PrecisionComputeBackend::execution_device_memory_peak_bytes_v2(
                &mut backend,
                matrix.handle(),
                &protocol,
            )
            .unwrap(),
            Some(0)
        );
        match precision {
            PrecisionProfile::Fp32 => {
                gafime_orchestrator::PrecisionComputeBackend::execute_fp32(
                    &mut backend,
                    matrix.handle(),
                    &protocol,
                    &mut GafimeResultTable::default(),
                )
                .unwrap();
                assert!(backend
                    .permutation_pvalues_fp32_v2_with_budget(
                        matrix.handle(),
                        &protocol,
                        &[0],
                        &[0.0],
                        1,
                        Some(u64::MAX),
                    )
                    .unwrap()
                    .is_some());
            }
            PrecisionProfile::Mixed | PrecisionProfile::Fp64 => {
                gafime_orchestrator::PrecisionComputeBackend::execute_f64(
                    &mut backend,
                    matrix.handle(),
                    &protocol,
                    &mut GafimeResultTableF64::default(),
                )
                .unwrap();
                assert!(backend
                    .permutation_pvalues_f64_v2_with_budget(
                        matrix.handle(),
                        &protocol,
                        &[0],
                        &[0.0],
                        1,
                        Some(u64::MAX),
                    )
                    .unwrap()
                    .is_some());
            }
        }
    }
    assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 3);
    assert_eq!(PRECISION_F32_UPLOAD_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(PRECISION_F64_UPLOAD_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(PRECISION_F32_UPDATE_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(PRECISION_F64_UPDATE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(PRECISION_F32_EXECUTE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(PRECISION_F64_EXECUTE_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(PRECISION_EXECUTION_PEAK_CALLS.load(Ordering::SeqCst), 3);
    assert_eq!(PRECISION_PERMUTATION_PEAK_CALLS.load(Ordering::SeqCst), 3);
    assert_eq!(PRECISION_F32_PERMUTATION_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(PRECISION_F64_PERMUTATION_CALLS.load(Ordering::SeqCst), 2);
}

#[test]
fn canonical_precision_operation_table_requires_all_ten_symbols() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    for (missing_symbol, remove) in [
        ("gafime_gpu_numeric_routes_v2", 0u8),
        ("gafime_gpu_matrix_alloc_v2", 1u8),
        ("gafime_gpu_matrix_upload_v2", 2u8),
        ("gafime_gpu_matrix_update_target_v2", 3u8),
        ("gafime_gpu_execute_v2", 4u8),
        ("gafime_gpu_execution_memory_peak_v2", 5u8),
        ("gafime_gpu_permutation_memory_peak_v2", 6u8),
        ("gafime_gpu_permutation_pvalues_v2", 7u8),
        ("gafime_gpu_interaction_diagnostics_v2", 8u8),
        ("gafime_gpu_matrix_free_v2", 9u8),
    ] {
        for precision in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            let mut partial = complete_precision_test_function_table();
            match remove {
                0 => partial.numeric_routes_v2 = None,
                1 => partial.matrix_alloc_v2 = None,
                2 => partial.matrix_upload_v2 = None,
                3 => partial.matrix_update_target_v2 = None,
                4 => partial.execute_v2 = None,
                5 => partial.execution_memory_peak_v2 = None,
                6 => partial.permutation_memory_peak_v2 = None,
                7 => partial.permutation_pvalues_v2 = None,
                8 => partial.interaction_diagnostics_v2 = None,
                9 => partial.matrix_free_v2 = None,
                _ => unreachable!(),
            }
            let partial = GpuBackend::new(GAFIME_BACKEND_CUDA, partial).unwrap();
            PRECISION_MATRIX_ALLOC_CALLS.store(0, Ordering::SeqCst);
            assert!(matches!(
                precision_alloc_error(&partial, precision),
                GpuSysError::MissingFunction(symbol) if symbol == missing_symbol
            ));
            assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 0);
        }
    }

    let complete = GpuBackend::new(
        GAFIME_BACKEND_CUDA,
        complete_precision_test_function_table(),
    )
    .unwrap();
    for precision in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        PRECISION_MATRIX_ALLOC_CALLS.store(0, Ordering::SeqCst);
        drop(complete.alloc_matrix_for_profile(precision, 2, 2).unwrap());
        assert!(complete.supports_precision_permutation_pvalues(precision));
        assert_eq!(PRECISION_MATRIX_ALLOC_CALLS.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn interaction_diagnostics_remain_optional_and_preserve_u64_counts() {
    let legacy_backend =
        GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
    let legacy_matrix = legacy_backend.alloc_matrix(4, 2).unwrap();
    let combos = [0, u32::MAX, 0, 1];
    assert_eq!(
        legacy_backend
            .interaction_diagnostics(legacy_matrix.handle(), &combos, 2, 2)
            .unwrap(),
        None
    );

    let mut functions = complete_test_function_table();
    functions.interaction_diagnostics = Some(test_interaction_diagnostics);
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let matrix = backend.alloc_matrix(4, 2).unwrap();
    assert!(backend.supports_interaction_diagnostics());
    assert_eq!(
        backend
            .interaction_diagnostics(matrix.handle(), &combos, 2, 2)
            .unwrap()
            .unwrap(),
        vec![
            GpuInteractionDiagnostic {
                overflow_row_count: 2,
                source_nonfinite: true,
            },
            GpuInteractionDiagnostic {
                overflow_row_count: 3,
                source_nonfinite: false,
            },
        ]
    );
}

#[test]
fn same_abi_payload_ranking_support_remains_probe_driven() {
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
    let capability = backend.graph_capability().unwrap();

    assert_eq!(capability.abi_version, GAFIME_ABI_VERSION);
    assert_eq!(capability.supports_device_ranking, 0);
}

#[test]
fn execution_memory_peak_remains_optional_and_calls_capable_payloads() {
    let mut legacy_backend =
        GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
    let legacy_matrix = legacy_backend.alloc_matrix(4, 2).unwrap();
    let mut config = EngineConfig {
        backend_kind: GAFIME_BACKEND_CUDA,
        metric_ids: vec![GAFIME_METRIC_PEARSON],
        ..Default::default()
    };
    config.budget.max_comb_size = 1;
    let prepared = prepare_continuous_execution(&config, 4, 2).unwrap();
    assert_eq!(
        legacy_backend
            .execution_device_memory_peak_bytes(legacy_matrix.handle(), prepared.plan().protocol(),)
            .unwrap(),
        None
    );

    let mut functions = complete_test_function_table();
    functions.execution_memory_peak = Some(test_execution_memory_peak);
    let mut capable_backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let capable_matrix = capable_backend.alloc_matrix(4, 2).unwrap();
    assert_eq!(
        capable_backend
            .execution_device_memory_peak_bytes(
                capable_matrix.handle(),
                prepared.plan().protocol(),
            )
            .unwrap(),
        Some(0x5A5A_A5A5)
    );
}

#[test]
fn permutation_pvalues_reject_peak_between_normal_and_significance_budget() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut config = EngineConfig {
        backend_kind: GAFIME_BACKEND_CUDA,
        metric_ids: vec![GAFIME_METRIC_PEARSON],
        ..Default::default()
    };
    config.budget.max_comb_size = 1;
    config.permutation_tests = 1;
    let prepared = prepare_continuous_execution(&config, 4, 2).unwrap();
    let protocol = prepared.plan().protocol();

    let mut functions = complete_test_function_table();
    functions.execution_memory_peak = Some(test_small_execution_memory_peak);
    functions.permutation_memory_peak = Some(test_permutation_memory_peak);
    functions.permutation_pvalues = Some(test_permutation_pvalues);
    let mut backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let matrix = backend.alloc_matrix(4, 2).unwrap();
    let budget = 12 * 1024 * 1024;
    assert_eq!(
        backend
            .execution_device_memory_peak_bytes(matrix.handle(), protocol)
            .unwrap(),
        Some(TEST_NORMAL_EXECUTION_PEAK)
    );
    assert!(TEST_NORMAL_EXECUTION_PEAK < budget);
    assert!(budget < TEST_PERMUTATION_EXECUTION_PEAK);

    TEST_PERMUTATION_PVALUE_CALLS.store(0, Ordering::SeqCst);
    TEST_PERMUTATION_PEAK_SELECTED_ROWS.store(0, Ordering::SeqCst);
    let error = backend
        .permutation_pvalues_with_budget(
            matrix.handle(),
            protocol,
            &[0, 1],
            &[0.1, 0.2],
            1,
            Some(budget),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        GpuSysError::InvalidInput(
            "permutation p-value device-memory peak exceeds budget.vram_budget_mb"
        )
    ));
    assert_eq!(
        TEST_PERMUTATION_PEAK_SELECTED_ROWS.load(Ordering::SeqCst),
        2
    );
    assert_eq!(TEST_PERMUTATION_PVALUE_CALLS.load(Ordering::SeqCst), 0);

    let pvalues = backend
        .permutation_pvalues_with_budget(
            matrix.handle(),
            protocol,
            &[0, 1],
            &[0.1, 0.2],
            1,
            Some(TEST_PERMUTATION_EXECUTION_PEAK),
        )
        .unwrap()
        .unwrap();
    assert_eq!(pvalues, vec![0.5, 0.5]);
    assert_eq!(TEST_PERMUTATION_PVALUE_CALLS.load(Ordering::SeqCst), 1);

    let mut legacy_functions = functions;
    legacy_functions.permutation_memory_peak = None;
    let mut legacy_backend = GpuBackend::new(GAFIME_BACKEND_CUDA, legacy_functions).unwrap();
    let legacy_matrix = legacy_backend.alloc_matrix(4, 2).unwrap();
    assert_eq!(
        legacy_backend
            .permutation_pvalues_with_budget(
                legacy_matrix.handle(),
                protocol,
                &[0, 1],
                &[0.1, 0.2],
                1,
                Some(budget),
            )
            .unwrap(),
        None
    );
    assert_eq!(TEST_PERMUTATION_PVALUE_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn descriptor_generation_is_sent_only_to_generation_capable_payloads() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut config = EngineConfig {
        backend_kind: GAFIME_BACKEND_CUDA,
        metric_ids: vec![GAFIME_METRIC_PEARSON],
        ..Default::default()
    };
    config.budget.max_comb_size = 1;
    let prepared = prepare_continuous_execution(&config, 4, 2).unwrap();
    assert_eq!(
        prepared.plan().protocol().flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
        0
    );

    let execute_with = |device_info: GafimeGpuDeviceInfoFn| {
        let mut functions = complete_test_function_table();
        functions.device_info = Some(device_info);
        functions.execute = Some(test_execute_captures_launch_flags);
        let mut backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
        let matrix = backend.alloc_matrix(4, 2).unwrap();
        let mut result = GafimeResultTable::default();
        TEST_EXECUTE_FLAGS.store(u32::MAX, Ordering::SeqCst);
        TEST_EXECUTE_DESCRIPTOR_GENERATION.store(u64::MAX, Ordering::SeqCst);
        prepared
            .execute(&mut backend, matrix.handle(), &mut result)
            .unwrap();
        (
            backend.supports_immutable_protocol(),
            backend.supports_descriptor_generation(),
            TEST_EXECUTE_FLAGS.load(Ordering::SeqCst),
            TEST_EXECUTE_DESCRIPTOR_GENERATION.load(Ordering::SeqCst),
        )
    };

    let (legacy_immutable, legacy_generation, legacy_flags, legacy_token) =
        execute_with(test_device_info);
    assert!(!legacy_immutable);
    assert!(!legacy_generation);
    assert_eq!(legacy_flags, prepared.plan().protocol().flags);
    assert_eq!(legacy_token, 0);

    let (old_immutable, old_generation, old_flags, old_token) =
        execute_with(test_device_info_with_old_immutable_protocol);
    assert!(old_immutable);
    assert!(!old_generation);
    assert_eq!(old_flags, prepared.plan().protocol().flags);
    assert_eq!(old_token, 0);

    let (current_immutable, current_generation, current_flags, current_token) =
        execute_with(test_device_info_with_descriptor_generation);
    assert!(current_immutable);
    assert!(current_generation);
    assert_eq!(
        current_flags,
        prepared.plan().protocol().flags | GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL
    );
    assert_ne!(current_token, 0);
}

#[test]
fn gpu_backend_requires_every_mandatory_function() {
    macro_rules! assert_missing {
        ($field:ident, $symbol:literal) => {{
            let mut functions = complete_test_function_table();
            functions.$field = None;
            assert!(matches!(
                GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
                Err(GpuSysError::MissingFunction($symbol))
            ));
        }};
    }

    assert_missing!(device_info, "gafime_gpu_device_info");
    assert_missing!(graph_capability, "gafime_gpu_graph_capability");
    assert_missing!(matrix_alloc, "gafime_gpu_matrix_alloc");
    assert_missing!(matrix_upload, "gafime_gpu_matrix_upload");
    assert_missing!(matrix_update_target, "gafime_gpu_matrix_update_target");
    assert_missing!(matrix_free, "gafime_gpu_matrix_free");
    assert_missing!(execute, "gafime_gpu_execute");
}

#[test]
fn gpu_backend_rejects_mismatched_payload_identity() {
    let mut functions = complete_test_function_table();
    functions.device_info = Some(test_device_info_wrong_abi);
    assert!(matches!(
        GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
        Err(GpuSysError::AbiVersionMismatch {
            expected: GAFIME_ABI_VERSION,
            actual,
        }) if actual == GAFIME_ABI_VERSION + 1
    ));

    let mut functions = complete_test_function_table();
    functions.device_info = Some(test_device_info_wrong_backend);
    assert!(matches!(
        GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
        Err(GpuSysError::BackendKindMismatch {
            expected: GAFIME_BACKEND_CUDA,
            actual: GAFIME_BACKEND_ROCM,
        })
    ));

    let mut functions = complete_test_function_table();
    functions.device_info = Some(test_device_info_wrong_device);
    assert!(matches!(
        GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
        Err(GpuSysError::DeviceIdMismatch {
            expected: 0,
            actual: 1,
        })
    ));

    let mut functions = complete_test_function_table();
    functions.graph_capability = Some(test_graph_capability_wrong_backend);
    assert!(matches!(
        GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
        Err(GpuSysError::BackendKindMismatch {
            expected: GAFIME_BACKEND_CUDA,
            actual: GAFIME_BACKEND_METAL,
        })
    ));
}

#[test]
fn owned_gpu_matrix_exposes_only_a_borrowed_handle_and_frees_once() {
    OWNED_MATRIX_FREES.store(0, Ordering::SeqCst);
    let mut functions = complete_test_function_table();
    functions.matrix_free = Some(owned_matrix_free);
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let matrix = backend.alloc_matrix(2, 3).unwrap();
    let handle_fn: for<'a> fn(&'a OwnedGpuMatrix) -> &'a MatrixHandle = OwnedGpuMatrix::handle;
    {
        let handle = handle_fn(&matrix);
        assert_eq!(handle.backend_kind(), GAFIME_BACKEND_CUDA);
        assert_eq!((handle.rows(), handle.cols()), (2, 3));
        assert_eq!(OWNED_MATRIX_FREES.load(Ordering::SeqCst), 0);
    }

    drop(matrix);
    assert_eq!(OWNED_MATRIX_FREES.load(Ordering::SeqCst), 1);
}

#[test]
fn legacy_operations_reject_precision_handles_before_ffi() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    TEST_LEGACY_ABI_SURFACE_CALLS.store(0, Ordering::SeqCst);

    let functions = cross_generation_test_function_table();
    let mut backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let matrix = backend
        .alloc_matrix_for_profile(PrecisionProfile::Mixed, 2, 2)
        .unwrap();
    assert_eq!(
        matrix.handle().native_abi_version(),
        Some(GAFIME_PRECISION_ABI_VERSION)
    );

    for error in [
        matrix.upload(&[0.0; 4], &[0.0; 2]).unwrap_err(),
        matrix.update_target(&[0.0; 2]).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            GpuSysError::InvalidInput("legacy matrix operation requires an ABI 1.0 matrix handle")
        ));
    }

    let protocol = GafimeLaunchProtocol {
        abi_version: GAFIME_ABI_VERSION,
        backend_kind: GAFIME_BACKEND_CUDA,
        n_samples: 2,
        n_features: 2,
        ..Default::default()
    };
    assert!(matches!(
        backend
            .execution_device_memory_peak_bytes(matrix.handle(), &protocol)
            .unwrap_err(),
        OrchestratorError::InvalidPlan("legacy GPU operation requires an ABI 1.0 matrix handle")
    ));
    assert!(matches!(
        backend
            .execute(
                matrix.handle(),
                &protocol,
                &mut GafimeResultTable::default(),
            )
            .unwrap_err(),
        OrchestratorError::InvalidPlan("legacy GPU operation requires an ABI 1.0 matrix handle")
    ));
    assert!(matches!(
        backend
            .permutation_pvalues(matrix.handle(), &protocol, &[0], &[0.0], 1)
            .unwrap_err(),
        GpuSysError::InvalidInput("legacy GPU operation requires an ABI 1.0 matrix handle")
    ));
    assert_eq!(TEST_LEGACY_ABI_SURFACE_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn precision_operations_reject_legacy_handles_before_ffi() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    TEST_PRECISION_ABI_SURFACE_CALLS.store(0, Ordering::SeqCst);

    let functions = cross_generation_test_function_table();
    let mut backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let matrix = backend.alloc_matrix(2, 2).unwrap();
    assert_eq!(
        matrix.handle().native_abi_version(),
        Some(GAFIME_ABI_VERSION)
    );

    for error in [
        matrix.upload_f32_v2(&[0.0; 4], &[0.0; 2]).unwrap_err(),
        matrix.update_target_f32_v2(&[0.0; 2]).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            GpuSysError::InvalidInput(
                "precision matrix operation requires an ABI 1.1 matrix handle"
            )
        ));
    }

    let base = GafimeLaunchProtocol {
        abi_version: GAFIME_ABI_VERSION,
        backend_kind: GAFIME_BACKEND_CUDA,
        n_samples: 2,
        n_features: 2,
        ..Default::default()
    };
    let protocol = GafimePrecisionLaunchProtocol {
        abi_version: GAFIME_PRECISION_ABI_VERSION,
        profile: PrecisionProfile::Mixed as u32,
        base: &base,
        reserved: [0; 8],
    };
    assert!(matches!(
        gafime_orchestrator::PrecisionComputeBackend::execution_device_memory_peak_bytes_v2(
            &mut backend,
            matrix.handle(),
            &protocol,
        )
        .unwrap_err(),
        OrchestratorError::InvalidPlan("precision GPU operation requires an ABI 1.1 matrix handle")
    ));
    assert!(matches!(
        gafime_orchestrator::PrecisionComputeBackend::execute_f64(
            &mut backend,
            matrix.handle(),
            &protocol,
            &mut GafimeResultTableF64::default(),
        )
        .unwrap_err(),
        OrchestratorError::InvalidPlan("precision GPU operation requires an ABI 1.1 matrix handle")
    ));
    assert!(matches!(
        backend
            .permutation_pvalues_f64_v2_with_budget(
                matrix.handle(),
                &protocol,
                &[0],
                &[0.0],
                1,
                None,
            )
            .unwrap_err(),
        GpuSysError::InvalidInput("precision GPU operation requires an ABI 1.1 matrix handle")
    ));
    assert_eq!(TEST_PRECISION_ABI_SURFACE_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn cuda_loader_requires_explicit_payload_path() {
    if env::var_os(CUDA_LIBRARY_ENV).is_some() {
        return;
    }
    assert!(matches!(
        GpuBackend::cuda_from_env(0),
        Err(GpuSysError::EnvMissing(CUDA_LIBRARY_ENV))
    ));
}

#[test]
fn device_profile_interprets_portable_architecture_flags() {
    let mut cuda = GafimeGpuDeviceInfo {
        backend_kind: GAFIME_BACKEND_CUDA,
        flags: GAFIME_GPU_DEVICE_FLAG_DISCRETE
            | GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH
            | GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL
            | GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION
            | GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64,
        reserved: [0; 8],
        ..Default::default()
    };
    cuda.reserved[0] = GAFIME_GPU_ARCH_NVIDIA_ADA;
    let profile = GpuDeviceProfile::from_info(&cuda);
    assert_eq!(profile.architecture, GpuArchitectureClass::NvidiaAda);
    assert!(profile.discrete);
    assert!(profile.high_bandwidth);
    assert!(profile.immutable_protocol);
    assert!(profile.descriptor_generation);
    assert!(profile.mi_accumulation_fp64);
    assert!(!profile.f64_storage);
    assert!(!profile.unified_memory);

    let mut rocm = GafimeGpuDeviceInfo {
        backend_kind: GAFIME_BACKEND_ROCM,
        flags: GAFIME_GPU_DEVICE_FLAG_INTEGRATED
            | GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY
            | GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY
            | GAFIME_GPU_DEVICE_FLAG_AMD_CDNA,
        reserved: [0; 8],
        ..Default::default()
    };
    rocm.reserved[0] = GAFIME_GPU_ARCH_AMD_CDNA;
    let profile = GpuDeviceProfile::from_info(&rocm);
    assert_eq!(profile.architecture, GpuArchitectureClass::AmdCdna);
    assert!(profile.integrated);
    assert!(profile.unified_memory);
    assert!(profile.managed_memory);
    assert!(profile.amd_cdna);

    let mut metal = GafimeGpuDeviceInfo {
        backend_kind: GAFIME_BACKEND_METAL,
        flags: GAFIME_GPU_DEVICE_FLAG_INTEGRATED
            | GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY
            | GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY,
        reserved: [0; 8],
        ..Default::default()
    };
    metal.reserved[0] = GAFIME_GPU_ARCH_APPLE;
    let profile = GpuDeviceProfile::from_info(&metal);
    assert_eq!(profile.architecture, GpuArchitectureClass::Apple);
    assert!(profile.apple_family);
    assert!(profile.unified_memory);
}
