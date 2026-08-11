use super::*;

pub(crate) static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());
pub(crate) static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());
pub(crate) static ABI_TEST_LOCK: Mutex<()> = Mutex::new(());
pub(crate) static TEST_MATRIX_FREES: AtomicUsize = AtomicUsize::new(0);
pub(crate) static TEST_EXECUTE_FLAGS: AtomicU32 = AtomicU32::new(0);
pub(crate) static TEST_EXECUTE_DESCRIPTOR_GENERATION: AtomicU64 = AtomicU64::new(0);
pub(crate) static TEST_PERMUTATION_PVALUE_CALLS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static TEST_PERMUTATION_PEAK_SELECTED_ROWS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TEST_LEGACY_ABI_SURFACE_CALLS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static TEST_PRECISION_ABI_SURFACE_CALLS: AtomicUsize = AtomicUsize::new(0);

pub(crate) const TEST_NORMAL_EXECUTION_PEAK: u64 = 8 * 1024 * 1024;
pub(crate) const TEST_PERMUTATION_EXECUTION_PEAK: u64 = 16 * 1024 * 1024;

pub(crate) unsafe extern "C" fn test_device_info(
    device_id: u32,
    info_out: *mut GafimeGpuDeviceInfo,
) -> GafimeStatus {
    if info_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the null check above establishes a writable ABI output slot.
    unsafe {
        *info_out = GafimeGpuDeviceInfo {
            abi_version: GAFIME_ABI_VERSION,
            backend_kind: GAFIME_BACKEND_CUDA,
            device_id,
            ..Default::default()
        };
    }
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_device_info_wrong_abi(
    device_id: u32,
    info_out: *mut GafimeGpuDeviceInfo,
) -> GafimeStatus {
    // SAFETY: this stub forwards the caller's ABI arguments unchanged.
    let status = unsafe { test_device_info(device_id, info_out) };
    if status == GAFIME_STATUS_OK {
        // SAFETY: the successful helper call initialized the output slot.
        unsafe { (*info_out).abi_version = GAFIME_ABI_VERSION + 1 };
    }
    status
}

pub(crate) unsafe extern "C" fn test_device_info_with_old_immutable_protocol(
    device_id: u32,
    info_out: *mut GafimeGpuDeviceInfo,
) -> GafimeStatus {
    // SAFETY: this stub forwards the caller's ABI arguments unchanged.
    let status = unsafe { test_device_info(device_id, info_out) };
    if status == GAFIME_STATUS_OK {
        // SAFETY: the successful helper call initialized the output slot.
        unsafe { (*info_out).flags |= GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL };
    }
    status
}

pub(crate) unsafe extern "C" fn test_device_info_with_descriptor_generation(
    device_id: u32,
    info_out: *mut GafimeGpuDeviceInfo,
) -> GafimeStatus {
    // SAFETY: this stub forwards the caller's ABI arguments unchanged.
    let status = unsafe { test_device_info_with_old_immutable_protocol(device_id, info_out) };
    if status == GAFIME_STATUS_OK {
        // SAFETY: the successful helper call initialized the output slot.
        unsafe { (*info_out).flags |= GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION };
    }
    status
}

pub(crate) unsafe extern "C" fn test_device_info_wrong_backend(
    device_id: u32,
    info_out: *mut GafimeGpuDeviceInfo,
) -> GafimeStatus {
    // SAFETY: this stub forwards the caller's ABI arguments unchanged.
    let status = unsafe { test_device_info(device_id, info_out) };
    if status == GAFIME_STATUS_OK {
        // SAFETY: the successful helper call initialized the output slot.
        unsafe { (*info_out).backend_kind = GAFIME_BACKEND_ROCM };
    }
    status
}

pub(crate) unsafe extern "C" fn test_device_info_wrong_device(
    device_id: u32,
    info_out: *mut GafimeGpuDeviceInfo,
) -> GafimeStatus {
    // SAFETY: this stub forwards the caller's ABI arguments unchanged.
    let status = unsafe { test_device_info(device_id, info_out) };
    if status == GAFIME_STATUS_OK {
        // SAFETY: the successful helper call initialized the output slot.
        unsafe { (*info_out).device_id = device_id.saturating_add(1) };
    }
    status
}

pub(crate) unsafe extern "C" fn test_graph_capability(
    _device_id: u32,
    capability_out: *mut GafimeGpuGraphCapability,
) -> GafimeStatus {
    if capability_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the null check above establishes a writable ABI output slot.
    unsafe {
        *capability_out = GafimeGpuGraphCapability {
            abi_version: GAFIME_ABI_VERSION,
            backend_kind: GAFIME_BACKEND_CUDA,
            ..Default::default()
        };
    }
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_graph_capability_wrong_backend(
    device_id: u32,
    capability_out: *mut GafimeGpuGraphCapability,
) -> GafimeStatus {
    // SAFETY: this stub forwards the caller's ABI arguments unchanged.
    let status = unsafe { test_graph_capability(device_id, capability_out) };
    if status == GAFIME_STATUS_OK {
        // SAFETY: the successful helper call initialized the output slot.
        unsafe { (*capability_out).backend_kind = GAFIME_BACKEND_METAL };
    }
    status
}

pub(crate) unsafe extern "C" fn test_matrix_alloc(
    _device_id: u32,
    _matrix_desc: *const GafimeMatrixDesc,
    matrix_out: *mut GafimeGpuMatrix,
) -> GafimeStatus {
    if matrix_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the null check establishes a writable output slot; the paired
    // test free function owns the allocation returned here.
    unsafe { *matrix_out = Box::into_raw(Box::new(0u8)).cast() };
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_numeric_routes_v2(
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
    // SAFETY: the caller supplied the required writable count slot.
    unsafe { *route_count_out = 3 };
    if routes_out.is_null() {
        return if route_capacity == 0 {
            GAFIME_STATUS_OK
        } else {
            gafime_types::GAFIME_STATUS_INVALID_ARGUMENT
        };
    }
    if route_capacity < 3 || route_stride < core::mem::size_of::<GafimeNumericRoute>() as u32 {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (index, route) in [
        GafimeNumericRoute::fp32(),
        GafimeNumericRoute::mixed(),
        GafimeNumericRoute::fp64(),
    ]
    .into_iter()
    .enumerate()
    {
        // SAFETY: capacity and stride were checked; the fixture writes only
        // the ABI 1.1 record prefix into caller-owned storage.
        let destination = unsafe {
            routes_out
                .cast::<u8>()
                .add(index * route_stride as usize)
                .cast::<GafimeNumericRoute>()
        };
        // SAFETY: the destination points at this record's checked writable
        // prefix within the caller-owned route array.
        unsafe { destination.write(route) };
    }
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_matrix_alloc_v2(
    _device_id: u32,
    _matrix_desc: *const GafimeNumericMatrixDesc,
    matrix_out: *mut GafimeGpuMatrix,
) -> GafimeStatus {
    if matrix_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the null check establishes a writable output slot; the paired
    // test free function owns the allocation returned here.
    unsafe { *matrix_out = Box::into_raw(Box::new(0u8)).cast() };
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_legacy_matrix_upload(
    _matrix: GafimeGpuMatrix,
    _features_host: *const f32,
    _target_host: *const f32,
    _rows: u64,
    _cols: u32,
) -> GafimeStatus {
    TEST_LEGACY_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_legacy_matrix_update_target(
    _matrix: GafimeGpuMatrix,
    _target_host: *const f32,
    _rows: u64,
) -> GafimeStatus {
    TEST_LEGACY_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_legacy_execute(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeLaunchProtocol,
    _result_out: *mut GafimeResultTable,
) -> GafimeStatus {
    TEST_LEGACY_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_legacy_execution_memory_peak(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeLaunchProtocol,
    _peak_bytes_out: *mut u64,
) -> GafimeStatus {
    TEST_LEGACY_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_legacy_permutation_pvalues(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeLaunchProtocol,
    _significance_out: *mut GafimePermutationSignificanceTable,
) -> GafimeStatus {
    TEST_LEGACY_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_precision_matrix_upload_f32(
    _matrix: GafimeGpuMatrix,
    _route: *const GafimeNumericRoute,
    _features_host: *const GafimeConstBufferView,
    _target_host: *const GafimeConstBufferView,
    _rows: u64,
    _cols: u32,
) -> GafimeStatus {
    TEST_PRECISION_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_precision_matrix_update_target_f32(
    _matrix: GafimeGpuMatrix,
    _route: *const GafimeNumericRoute,
    _target_host: *const GafimeConstBufferView,
    _rows: u64,
) -> GafimeStatus {
    TEST_PRECISION_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_precision_execute_f64(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeNumericLaunchProtocol,
    _result_out: *mut GafimeNumericResultTable,
) -> GafimeStatus {
    TEST_PRECISION_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_precision_execution_memory_peak(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeNumericLaunchProtocol,
    _peak_bytes_out: *mut u64,
) -> GafimeStatus {
    TEST_PRECISION_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn count_precision_permutation_pvalues_f64(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeNumericLaunchProtocol,
    _significance_out: *mut GafimeNumericSignificanceTable,
) -> GafimeStatus {
    TEST_PRECISION_ABI_SURFACE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_matrix_upload(
    _matrix: GafimeGpuMatrix,
    _features_host: *const f32,
    _target_host: *const f32,
    _rows: u64,
    _cols: u32,
) -> GafimeStatus {
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_matrix_update_target(
    _matrix: GafimeGpuMatrix,
    _target_host: *const f32,
    _rows: u64,
) -> GafimeStatus {
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_matrix_free(matrix: GafimeGpuMatrix) {
    if !matrix.is_null() {
        // SAFETY: this function is paired only with test_matrix_alloc.
        unsafe { drop(Box::from_raw(matrix.cast::<u8>())) };
        TEST_MATRIX_FREES.fetch_add(1, Ordering::SeqCst);
    }
}

pub(crate) unsafe extern "C" fn test_execute(
    _matrix: GafimeGpuMatrix,
    _protocol: *const GafimeLaunchProtocol,
    _result_out: *mut GafimeResultTable,
) -> GafimeStatus {
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_execution_memory_peak(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    peak_bytes_out: *mut u64,
) -> GafimeStatus {
    if matrix.is_null() || protocol.is_null() || peak_bytes_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the null checks establish a writable output slot.
    unsafe { *peak_bytes_out = 0x5A5A_A5A5 };
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_small_execution_memory_peak(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    peak_bytes_out: *mut u64,
) -> GafimeStatus {
    if matrix.is_null() || protocol.is_null() || peak_bytes_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the null checks establish a writable output slot.
    unsafe { *peak_bytes_out = TEST_NORMAL_EXECUTION_PEAK };
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_permutation_memory_peak(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    selected_row_count: u64,
    peak_bytes_out: *mut u64,
) -> GafimeStatus {
    if matrix.is_null() || protocol.is_null() || peak_bytes_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    TEST_PERMUTATION_PEAK_SELECTED_ROWS.store(selected_row_count, Ordering::SeqCst);
    // SAFETY: the null checks establish a writable output slot.
    unsafe { *peak_bytes_out = TEST_PERMUTATION_EXECUTION_PEAK };
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_permutation_pvalues(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    significance_out: *mut GafimePermutationSignificanceTable,
) -> GafimeStatus {
    if matrix.is_null() || protocol.is_null() || significance_out.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the null check establishes a writable table supplied by the
    // adapter test, including row_count*metric_count p-value slots.
    let significance = unsafe { &mut *significance_out };
    let value_count = significance
        .row_count
        .checked_mul(significance.metric_count as u64)
        .and_then(|count| usize::try_from(count).ok());
    let Some(value_count) = value_count else {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    };
    if value_count != 0 && significance.p_values.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for index in 0..value_count {
        // SAFETY: the test adapter allocated exactly value_count slots.
        unsafe { *significance.p_values.add(index) = 0.5 };
    }
    TEST_PERMUTATION_PVALUE_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_interaction_diagnostics(
    matrix: GafimeGpuMatrix,
    diagnostics: *mut GafimeInteractionDiagnosticBatch,
) -> GafimeStatus {
    if matrix.is_null() || diagnostics.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the adapter test supplies one initialized batch and output arrays
    // sized to row_count.
    let diagnostics = unsafe { &mut *diagnostics };
    let Some(row_count) = usize::try_from(diagnostics.row_count).ok() else {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    };
    if row_count != 0 && (diagnostics.overflow_row_counts.is_null() || diagnostics.flags.is_null())
    {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for row in 0..row_count {
        // SAFETY: the test adapter allocated row_count entries in both arrays.
        unsafe {
            *diagnostics.overflow_row_counts.add(row) = row as u64 + 2;
            *diagnostics.flags.add(row) = if row == 0 {
                GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE
            } else {
                0
            };
        }
    }
    GAFIME_STATUS_OK
}

pub(crate) unsafe extern "C" fn test_execute_captures_launch_flags(
    _matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    _result_out: *mut GafimeResultTable,
) -> GafimeStatus {
    if protocol.is_null() {
        return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the null check establishes a readable launch protocol.
    let protocol = unsafe { &*protocol };
    TEST_EXECUTE_FLAGS.store(protocol.flags, Ordering::SeqCst);
    TEST_EXECUTE_DESCRIPTOR_GENERATION.store(
        protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT],
        Ordering::SeqCst,
    );
    GAFIME_STATUS_OK
}

pub(crate) fn complete_test_function_table() -> GpuFunctionTable {
    GpuFunctionTable {
        device_info: Some(test_device_info),
        graph_capability: Some(test_graph_capability),
        matrix_alloc: Some(test_matrix_alloc),
        matrix_upload: Some(test_matrix_upload),
        matrix_update_target: Some(test_matrix_update_target),
        matrix_free: Some(test_matrix_free),
        execute: Some(test_execute),
        execution_memory_peak: None,
        permutation_memory_peak: None,
        permutation_pvalues: None,
        interaction_diagnostics: None,
        numeric_routes_v2: None,
        matrix_alloc_v2: None,
        matrix_upload_v2: None,
        matrix_update_target_v2: None,
        execute_v2: None,
        execution_memory_peak_v2: None,
        permutation_memory_peak_v2: None,
        permutation_pvalues_v2: None,
        interaction_diagnostics_v2: None,
        matrix_free_v2: None,
        #[cfg(feature = "local-cmake-experiment")]
        local_cmake_experiment: Default::default(),
    }
}

pub(crate) fn cuda_test_lock() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

pub(crate) fn metal_test_lock() -> MutexGuard<'static, ()> {
    METAL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

pub(crate) fn metal_backend_for_test() -> Option<GpuBackend> {
    env::var_os(METAL_LIBRARY_ENV)?;
    Some(
        GpuBackend::metal_from_env(0)
            .unwrap_or_else(|error| panic!("configured Metal payload failed to load: {error}")),
    )
}

pub(crate) fn cuda_backend_for_specialization_test() -> Option<GpuBackend> {
    env::var_os(CUDA_LIBRARY_ENV)?;
    Some(
        GpuBackend::cuda_from_env(0)
            .unwrap_or_else(|error| panic!("configured CUDA payload failed to load: {error}")),
    )
}

pub(crate) fn rocm_backend_for_specialization_test() -> Option<GpuBackend> {
    env::var_os(ROCM_LIBRARY_ENV)?;
    Some(
        GpuBackend::rocm_from_env(0)
            .unwrap_or_else(|error| panic!("configured ROCm payload failed to load: {error}")),
    )
}

pub(crate) fn assert_configured_library_is_process_cached(env_name: &str, kind: BackendKind) {
    let Some(path) = env::var_os(env_name) else {
        return;
    };
    // SAFETY: the test only reaches this path for an explicitly configured,
    // trusted GAFIME payload implementing the requested backend ABI.
    let first = unsafe { GpuBackend::load_abi_from_path(&path, 0, kind) }
        .unwrap_or_else(|error| panic!("configured payload failed to load: {error}"));
    // SAFETY: this repeats the same trusted payload load to verify process
    // caching; the path and requested ABI are unchanged.
    let second = unsafe { GpuBackend::load_abi_from_path(&path, 0, kind) }
        .unwrap_or_else(|error| panic!("configured payload failed to reload: {error}"));
    let first_library = first.library.as_ref().expect("loaded payload owns its DSO");
    let second_library = second
        .library
        .as_ref()
        .expect("loaded payload owns its DSO");
    assert!(Arc::ptr_eq(first_library, second_library));
}

pub(crate) struct TestResultTable {
    pub(crate) raw: GafimeResultTable,
    pub(crate) combo_indices: Vec<u32>,
    pub(crate) metric_values: Vec<f32>,
    pub(crate) ranks: Vec<u32>,
    pub(crate) families: Vec<u32>,
    pub(crate) candidate_ids: Vec<u64>,
    pub(crate) row_flags: Vec<u32>,
}

impl TestResultTable {
    pub(crate) fn new(capacity: u64, max_arity: u32, metric_count: u32) -> Self {
        let mut table = Self {
            raw: GafimeResultTable {
                abi_version: GAFIME_ABI_VERSION,
                max_arity,
                metric_count,
                flags: 0,
                capacity,
                row_count: 0,
                combo_indices: ptr::null_mut(),
                metric_values: ptr::null_mut(),
                ranks: ptr::null_mut(),
                families: ptr::null_mut(),
                candidate_ids: ptr::null_mut(),
                row_flags: ptr::null_mut(),
                backend_private: ptr::null_mut(),
                reserved: [0; 8],
            },
            combo_indices: vec![u32::MAX; capacity as usize * max_arity as usize],
            metric_values: vec![0.0; capacity as usize * metric_count as usize],
            ranks: vec![0; capacity as usize],
            families: vec![0; capacity as usize],
            candidate_ids: vec![0; capacity as usize],
            row_flags: vec![0; capacity as usize],
        };
        table.rebind();
        table
    }

    pub(crate) fn raw_mut(&mut self) -> &mut GafimeResultTable {
        self.rebind();
        &mut self.raw
    }

    pub(crate) fn metric_values(&self) -> &[f32] {
        &self.metric_values[..self.raw.row_count as usize * self.raw.metric_count as usize]
    }

    pub(crate) fn combo_indices(&self) -> &[u32] {
        &self.combo_indices[..self.raw.row_count as usize * self.raw.max_arity as usize]
    }

    pub(crate) fn ranks(&self) -> &[u32] {
        &self.ranks[..self.raw.row_count as usize]
    }

    pub(crate) fn candidate_ids(&self) -> &[u64] {
        &self.candidate_ids[..self.raw.row_count as usize]
    }

    fn rebind(&mut self) {
        self.raw.combo_indices = self.combo_indices.as_mut_ptr();
        self.raw.metric_values = self.metric_values.as_mut_ptr();
        self.raw.ranks = self.ranks.as_mut_ptr();
        self.raw.families = self.families.as_mut_ptr();
        self.raw.candidate_ids = self.candidate_ids.as_mut_ptr();
        self.raw.row_flags = self.row_flags.as_mut_ptr();
    }
}

pub(crate) fn assert_adaptive_mi_templates_match_cpu_for_arity_1_to_5(
    gpu_backend: &mut GpuBackend,
    backend_kind: u32,
    bins_to_test: &[u32],
) {
    assert!(!bins_to_test.is_empty());
    let rows = 73_728u64;
    let cols = 5u32;
    let (features, target) = parity_dataset(rows, cols);
    let prepare = |planned_backend_kind, bins| {
        let mut config = EngineConfig {
            backend_kind: planned_backend_kind,
            metric_ids: vec![GAFIME_METRIC_MUTUAL_INFO],
            mi_bins: bins,
            mi_approximate: true,
            permutation_tests: 0,
            ..Default::default()
        };
        config.budget.max_comb_size = 5;
        config.budget.max_combinations_per_k = 100;
        prepare_continuous_execution(&config, rows, cols).unwrap()
    };

    let cpu_matrix =
        CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
    let mut cpu_backend = CpuBackend;
    let gpu_matrix = gpu_backend.alloc_matrix(rows, cols).unwrap();
    gpu_matrix.upload(&features, &target).unwrap();

    for &bins in bins_to_test {
        let cpu_prepared = prepare(GAFIME_BACKEND_CPU, bins);
        let gpu_prepared = prepare(backend_kind, bins);

        let mut cpu_result = TestResultTable::new(
            cpu_prepared.result_capacity(),
            cpu_prepared.result_max_arity(),
            cpu_prepared.result_metric_count(),
        );
        execute_plan!(
            &mut cpu_backend,
            &cpu_matrix.handle(),
            cpu_prepared.plan(),
            cpu_result.raw_mut(),
        )
        .unwrap();

        let mut gpu_result = TestResultTable::new(
            gpu_prepared.result_capacity(),
            gpu_prepared.result_max_arity(),
            gpu_prepared.result_metric_count(),
        );
        execute_plan!(
            gpu_backend,
            gpu_matrix.handle(),
            gpu_prepared.plan(),
            gpu_result.raw_mut(),
        )
        .unwrap();

        assert_eq!(cpu_result.raw.row_count, 31);
        assert_eq!(cpu_result.raw.row_count, gpu_result.raw.row_count);
        assert_eq!(cpu_result.combo_indices(), gpu_result.combo_indices());
        assert_eq!(cpu_result.candidate_ids(), gpu_result.candidate_ids());
        for (index, (&cpu_value, &gpu_value)) in cpu_result
            .metric_values()
            .iter()
            .zip(gpu_result.metric_values())
            .enumerate()
        {
            let delta = (cpu_value - gpu_value).abs();
            let ulps = cpu_value.to_bits().abs_diff(gpu_value.to_bits());
            assert!(
                cpu_value.is_finite() && gpu_value.is_finite() && ulps <= 3,
                "MI mismatch at {index}: backend={backend_kind} bins={bins} \
                 cpu={cpu_value} gpu={gpu_value} delta={delta} ulps={ulps}"
            );
        }
    }
}

fn deterministic_unit_interval(index: u64) -> f32 {
    let mut value = index.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value >> 40) as u32 as f32 + 0.5) * (1.0 / 16_777_216.0)
}

pub(crate) fn assert_low_signal_mi_matches_cpu(gpu_backend: &mut GpuBackend, backend_kind: u32) {
    let rows = 4_608u64;
    let cols = 16u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for row in 0..rows {
        for col in 0..cols {
            let index = row * u64::from(cols) + u64::from(col);
            features.push(2.0 * deterministic_unit_interval(index ^ 0x0240_4608) - 1.0);
        }
        target.push(2.0 * deterministic_unit_interval(row ^ 0xd1b5_4a32_d192_ed03) - 1.0);
    }

    let prepare = |planned_backend_kind| {
        let mut config = EngineConfig {
            backend_kind: planned_backend_kind,
            metric_ids: vec![GAFIME_METRIC_MUTUAL_INFO],
            mi_bins: 24,
            mi_approximate: true,
            permutation_tests: 0,
            ..Default::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 1_000;
        prepare_continuous_execution(&config, rows, cols).unwrap()
    };

    let cpu_prepared = prepare(GAFIME_BACKEND_CPU);
    let gpu_prepared = prepare(backend_kind);
    assert_eq!(cpu_prepared.result_capacity(), 136);
    assert_eq!(gpu_prepared.result_capacity(), 136);

    let cpu_matrix =
        CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
    let mut cpu_backend = CpuBackend;
    let mut cpu_result = TestResultTable::new(136, 2, 1);
    execute_plan!(
        &mut cpu_backend,
        &cpu_matrix.handle(),
        cpu_prepared.plan(),
        cpu_result.raw_mut(),
    )
    .unwrap();

    let gpu_matrix = gpu_backend.alloc_matrix(rows, cols).unwrap();
    gpu_matrix.upload(&features, &target).unwrap();
    let mut gpu_result = TestResultTable::new(136, 2, 1);
    execute_plan!(
        gpu_backend,
        gpu_matrix.handle(),
        gpu_prepared.plan(),
        gpu_result.raw_mut(),
    )
    .unwrap();

    assert_eq!(cpu_result.combo_indices(), gpu_result.combo_indices());
    let mut positive_low_signal_count = 0;
    for (index, (&cpu_value, &gpu_value)) in cpu_result
        .metric_values()
        .iter()
        .zip(gpu_result.metric_values())
        .enumerate()
    {
        if cpu_value > 0.0 && cpu_value < 0.01 {
            positive_low_signal_count += 1;
        }
        let delta = (cpu_value - gpu_value).abs();
        assert!(
            cpu_value.is_finite() && gpu_value.is_finite() && delta <= 2.0e-8,
            "low-signal MI mismatch at {index}: backend={backend_kind} \
             cpu={cpu_value} gpu={gpu_value} delta={delta}"
        );
    }
    assert!(
        positive_low_signal_count > 0,
        "the precision fixture must exercise positive MI below 0.01"
    );
}

pub(crate) fn continuous_config(backend_kind: u32) -> EngineConfig {
    let mut config = EngineConfig {
        backend_kind,
        metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        permutation_tests: 0,
        ..Default::default()
    };
    config.budget.max_comb_size = 5;
    config.budget.max_combinations_per_k = 10_000;
    config
}

pub(crate) fn assert_nonfinite_correlation_is_not_laundered(
    backend: &mut GpuBackend,
    backend_kind: u32,
) {
    let rows = 32u64;
    let cols = 3u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for row in 0..rows as usize {
        let first_sign = if row & 1 == 0 { -1.0 } else { 1.0 };
        let second_sign = if row & 2 == 0 { -1.0 } else { 1.0 };
        features.extend([
            first_sign * 1.0e30,
            first_sign * 1.0e10,
            second_sign * 1.0e10,
        ]);
        target.push(first_sign * second_sign * 1.0e30);
    }

    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();
    for (arity, combos) in [(1, vec![0]), (2, vec![1, 2])] {
        let plan = CompiledPlan::single_chunk(
            backend_kind,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            arity,
            combos,
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        );
        let mut result = TestResultTable::new(1, arity, 2);
        execute_plan!(backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(result.raw.row_count, 1);
        let expected = if arity == 1 { [0.0, 0.0] } else { [1.0, 1.0] };
        for (metric, (&actual, expected)) in result.metric_values().iter().zip(expected).enumerate()
        {
            let delta = (actual - expected).abs();
            assert!(
                actual.is_finite() && delta <= 1.0e-5,
                "backend {backend_kind} arity {arity} metric {metric} failed the \
                 high-dynamic correlation contract: actual={actual} expected={expected} \
                 delta={delta}"
            );
        }
    }
}

pub(crate) fn assert_scaled_covariance_matches_cpu_across_dynamic_range(
    backend: &mut GpuBackend,
    backend_kind: u32,
    tolerance: f32,
) {
    let rows = 512u64;
    let cols = 5u32;
    let sample = |row: usize, lane: usize| {
        let mixed = (row * (37 + lane * 12) + lane * 101 + row * row * 3) % 997;
        (mixed as f32 + 0.5) / 498.5 - 1.0
    };
    let build_unary = |scale: f32, offset: f32| {
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            let signal = sample(row, 0);
            let noise = sample(row, 7);
            features.extend((0..cols as usize).map(|col| offset + sample(row, col) * scale));
            target.push((0.7 * signal + 0.3 * noise) * scale);
        }
        (features, target, 1u32, vec![0u32])
    };

    let mut cases = vec![
        ("large-unary", build_unary(1.0e12, 0.0)),
        ("tiny-unary", build_unary(1.0e-20, 0.0)),
        ("timestamp-offset", build_unary(1.0e13, 1.77e18)),
    ];
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    for row in 0..rows as usize {
        features.extend((0..cols as usize).map(|col| sample(row, col) * 1.0e4));
    }
    let means = (0..cols as usize)
        .map(|col| {
            (0..rows as usize)
                .map(|row| features[row * cols as usize + col] as f64)
                .sum::<f64>() as f32
                / rows as f32
        })
        .collect::<Vec<_>>();
    let target = (0..rows as usize)
        .map(|row| {
            let interaction = (0..cols as usize).fold(1.0f32, |product, col| {
                product * (features[row * cols as usize + col] - means[col])
            });
            interaction * 0.75 + sample(row, 9) * 2.5e19
        })
        .collect::<Vec<_>>();
    cases.push(("arity-five", (features, target, 5, vec![0, 1, 2, 3, 4])));

    for (label, (features, target, arity, combo)) in cases {
        let cpu_matrix =
            CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
        let gpu_matrix = backend.alloc_matrix(rows, cols).unwrap();
        gpu_matrix.upload(&features, &target).unwrap();
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let cpu_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            arity,
            combo.clone(),
            metrics.clone(),
        );
        let gpu_plan = CompiledPlan::single_chunk(
            backend_kind,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            arity,
            combo,
            metrics,
        );
        let mut cpu_result = TestResultTable::new(1, arity, 2);
        let mut gpu_result = TestResultTable::new(1, arity, 2);
        execute_plan!(
            &mut CpuBackend,
            &cpu_matrix.handle(),
            &cpu_plan,
            cpu_result.raw_mut(),
        )
        .unwrap();
        execute_plan!(
            backend,
            gpu_matrix.handle(),
            &gpu_plan,
            gpu_result.raw_mut(),
        )
        .unwrap();

        for (metric, (&cpu_value, &gpu_value)) in cpu_result
            .metric_values()
            .iter()
            .zip(gpu_result.metric_values())
            .enumerate()
        {
            let delta = (cpu_value - gpu_value).abs();
            assert!(
                cpu_value.is_finite() && gpu_value.is_finite() && delta <= tolerance,
                "backend {backend_kind} {label} metric {metric}: cpu={cpu_value} \
                 gpu={gpu_value} delta={delta} tolerance={tolerance}"
            );
        }
    }
}

pub(crate) fn continuous_cached_target_stats_refresh_after_target_update(
    backend: &mut GpuBackend,
    backend_kind: u32,
) {
    let rows = 8u64;
    let cols = 2u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    for row in 0..rows as usize {
        features.push(row as f32);
        features.push((rows as usize - 1 - row) as f32);
    }
    let target_a = (0..rows as usize).map(|row| row as f32).collect::<Vec<_>>();
    let target_b = vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target_a).unwrap();

    let graph_plan = CompiledPlan::single_chunk(
        backend_kind,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
    )
    .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);
    let normal_plan = CompiledPlan::single_chunk(
        backend_kind,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
    );

    let mut first_graph_result = TestResultTable::new(2, 1, 2);
    execute_plan!(
        backend,
        matrix.handle(),
        &graph_plan,
        first_graph_result.raw_mut(),
    )
    .unwrap();
    assert_ne!(
        first_graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
        0
    );

    matrix.update_target(&target_b).unwrap();

    let mut updated_graph_result = TestResultTable::new(2, 1, 2);
    execute_plan!(
        backend,
        matrix.handle(),
        &graph_plan,
        updated_graph_result.raw_mut(),
    )
    .unwrap();
    assert_ne!(
        updated_graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
        0
    );

    let mut updated_normal_result = TestResultTable::new(2, 1, 2);
    execute_plan!(
        backend,
        matrix.handle(),
        &normal_plan,
        updated_normal_result.raw_mut(),
    )
    .unwrap();

    assert_eq!(
        updated_graph_result.combo_indices(),
        updated_normal_result.combo_indices()
    );
    for (idx, (&graph, &normal)) in updated_graph_result
        .metric_values()
        .iter()
        .zip(updated_normal_result.metric_values())
        .enumerate()
    {
        assert!(
            (graph - normal).abs() <= 1.0e-5,
            "metric {idx}: graph={graph} normal={normal}"
        );
    }
    assert!(
        (first_graph_result.metric_values()[0] - updated_graph_result.metric_values()[0]).abs()
            > 1.0e-3,
        "target update must materially change the cached-target fast path"
    );
}

pub(crate) fn parity_dataset(rows: u64, cols: u32) -> (Vec<f32>, Vec<f32>) {
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    for row in 0..rows as usize {
        let r = row as f32 + 1.0;
        for col in 0..cols as usize {
            let c = col as f32 + 1.0;
            let wave = ((row * (col + 3)) % 11) as f32 * 0.017;
            features.push((r * 0.031 * c) + wave + (c * c * 0.003));
        }
    }
    let target = (0..rows as usize)
        .map(|row| {
            let r = row as f32 + 1.0;
            (r * 0.071) + ((row % 5) as f32 * 0.043) - ((row % 3) as f32 * 0.019)
        })
        .collect();
    (features, target)
}

pub(crate) fn metal_parity_dataset(
    rows: u64,
    cols: u32,
    inject_nonfinite: bool,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(cols, 5);
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for row in 0..rows as usize {
        let x0 = 1_000_000.0 + ((row * 17) % 127) as f32 * 0.25;
        let x1 = -500_000.0 + ((row * 29) % 113) as f32 * 0.125;
        let x2 = ((row * 7) % 41) as f32 - 20.0;
        let x3 = ((row * 13) % 67) as f32 * 0.5 + (row % 3) as f32 * 0.125;
        let x4 = ((row * 31) % 59) as f32 * 0.75 - 20.0;
        features.extend([x0, x1, x2, x3, x4]);
        target.push(
            250_000.0 + (x0 - 1_000_000.0) * 0.375 - (x1 + 500_000.0) * 0.25
                + x2 * 1.125
                + x4 * 0.5
                + ((row * 19) % 23) as f32 * 0.0625,
        );
    }
    if inject_nonfinite {
        features[17 * cols as usize] = f32::NAN;
        features[53 * cols as usize + 2] = f32::INFINITY;
        target[91] = f32::NEG_INFINITY;
    }
    (features, target)
}
