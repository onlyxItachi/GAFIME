use super::*;
use gafime_orchestrator::OrchestratorError;

static OWNED_MATRIX_FREES: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn owned_matrix_free(matrix: GafimeGpuMatrix) {
    if !matrix.is_null() {
        // SAFETY: this callback is paired only with test_matrix_alloc.
        unsafe { drop(Box::from_raw(matrix.cast::<u8>())) };
        OWNED_MATRIX_FREES.fetch_add(1, Ordering::SeqCst);
    }
}

fn cross_generation_test_function_table() -> GpuFunctionTable {
    let mut functions = complete_test_function_table();
    functions.precision_capabilities = Some(test_precision_capabilities);
    functions.matrix_alloc_v2 = Some(test_matrix_alloc_v2);
    functions.matrix_upload = Some(count_legacy_matrix_upload);
    functions.matrix_update_target = Some(count_legacy_matrix_update_target);
    functions.execute = Some(count_legacy_execute);
    functions.execution_memory_peak = Some(count_legacy_execution_memory_peak);
    functions.permutation_pvalues = Some(count_legacy_permutation_pvalues);
    functions.matrix_upload_f32_v2 = Some(count_precision_matrix_upload_f32);
    functions.matrix_update_target_f32_v2 = Some(count_precision_matrix_update_target_f32);
    functions.execute_f64_v2 = Some(count_precision_execute_f64);
    functions.execution_memory_peak_v2 = Some(count_precision_execution_memory_peak);
    functions.permutation_pvalues_f64_v2 = Some(count_precision_permutation_pvalues_f64);
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
