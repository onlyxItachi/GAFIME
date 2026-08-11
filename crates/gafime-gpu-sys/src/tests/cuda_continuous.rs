use super::*;

#[test]
fn cuda_device_profile_reports_runtime_architecture_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    let info = backend.device_info().unwrap();
    let profile = GpuDeviceProfile::from_info(&info);
    assert_eq!(profile.backend_kind, GAFIME_BACKEND_CUDA);
    assert!(profile.discrete || profile.integrated);
    assert_ne!(profile.architecture, GpuArchitectureClass::Unknown);
    assert!(info.compute_major > 0);
    assert!(info.warp_size > 0);
    assert!(info.reserved[0] > 0);
}

#[test]
fn cuda_adapter_executes_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    let info = backend.device_info().unwrap();
    assert_eq!(info.backend_kind, GAFIME_BACKEND_CUDA);

    let rows = 4;
    let cols = 2;
    let features = vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0];
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
        return;
    };
    matrix.upload(&features, &target).unwrap();
    matrix.update_target(&target).unwrap();

    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
    );
    let mut result = TestResultTable::new(2, 1, 2);
    let stats = execute_plan!(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

    assert_eq!(stats.launched_chunks, 1);
    assert_eq!(stats.rows_written, 2);
    assert_eq!(result.raw.row_count, 2);
    let values = result.metric_values();
    assert!((values[0] - 1.0).abs() < 1.0e-5);
    assert!((values[1] - 1.0).abs() < 1.0e-5);
    assert!((values[2] + 1.0).abs() < 1.0e-5);
    assert!((values[3] - 1.0).abs() < 1.0e-5);
}

#[test]
fn cuda_nonfinite_correlation_is_not_laundered_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    assert_nonfinite_correlation_is_not_laundered(&mut backend, GAFIME_BACKEND_CUDA);
}

#[test]
fn cuda_scaled_covariance_matches_cpu_across_dynamic_range_when_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    assert_scaled_covariance_matches_cpu_across_dynamic_range(
        &mut backend,
        GAFIME_BACKEND_CUDA,
        5.0e-4,
    );
}

#[test]
fn cuda_cabi_rejects_stale_abi_overflow_and_malformed_inputs_when_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };

    let matrix_alloc = backend.functions.matrix_alloc.unwrap();
    let mut raw = ptr::null_mut();
    // SAFETY: the function pointer comes from the live trusted payload and
    // `raw` is writable; the null descriptor intentionally tests rejection
    // before the payload dereferences it.
    let status = unsafe { matrix_alloc(0, ptr::null(), &mut raw) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);

    let stale_desc = GafimeMatrixDesc {
        abi_version: GAFIME_ABI_VERSION + 1,
        rows: 1,
        cols: 1,
        row_stride: 1,
        bytes: std::mem::size_of::<f32>() as u64,
        ..Default::default()
    };
    // SAFETY: the descriptor and output slot are live and correctly aligned;
    // only the semantic ABI version is intentionally stale.
    let status = unsafe { matrix_alloc(0, &stale_desc, &mut raw) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
    assert!(raw.is_null());

    let mismatched_bytes_desc = GafimeMatrixDesc {
        rows: 2,
        cols: 2,
        row_stride: 2,
        bytes: std::mem::size_of::<f32>() as u64,
        ..Default::default()
    };
    // SAFETY: both pointers are valid; only the descriptor's declared byte
    // count is intentionally inconsistent with its shape.
    let status = unsafe { matrix_alloc(0, &mismatched_bytes_desc, &mut raw) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);
    assert!(raw.is_null());

    let huge_desc = GafimeMatrixDesc {
        rows: u64::MAX,
        cols: 2,
        row_stride: 2,
        bytes: u64::MAX,
        ..Default::default()
    };
    // SAFETY: both pointers are valid; the extreme shape intentionally drives
    // checked allocation overflow without providing any data buffer.
    let status = unsafe { matrix_alloc(0, &huge_desc, &mut raw) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_OUT_OF_MEMORY);
    assert!(raw.is_null());

    let rows = 4u64;
    let cols = 2u32;
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_PEARSON],
    );
    let mut result = TestResultTable::new(2, 1, 1);
    let execute = backend.functions.execute.unwrap();

    // SAFETY: matrix, plan, and owned result buffers are live. The matrix is
    // intentionally not uploaded, which is a semantic error the ABI rejects.
    let status = unsafe { execute(matrix.handle().raw(), plan.protocol(), result.raw_mut()) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);
    let mut stale_protocol = *plan.protocol();
    stale_protocol.abi_version = GAFIME_ABI_VERSION + 1;
    // SAFETY: all protocol pointers still reference the live plan; only its
    // copied ABI version is intentionally stale.
    let status = unsafe { execute(matrix.handle().raw(), &stale_protocol, result.raw_mut()) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
    result.raw_mut().abi_version = GAFIME_ABI_VERSION + 1;
    // SAFETY: OwnedResultTable still backs every output pointer; only the table
    // ABI version is intentionally stale.
    let status = unsafe { execute(matrix.handle().raw(), plan.protocol(), &mut result.raw) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
    result.raw.abi_version = GAFIME_ABI_VERSION;
    matrix
        .upload(
            &[0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0],
            &[0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
    let mut malformed = *plan.protocol();
    let mut malformed_chunk = plan.chunks()[0];
    malformed_chunk.descriptor_count = 0;
    malformed.chunks = &malformed_chunk;
    // SAFETY: every pointer references live storage; the copied chunk is
    // intentionally semantically malformed and must be rejected before launch.
    let status = unsafe { execute(matrix.handle().raw(), &malformed, result.raw_mut()) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);

    // SAFETY: all pointers remain live and correctly sized; only the copied
    // protocol ABI version is intentionally stale.
    let status = unsafe { execute(matrix.handle().raw(), &stale_protocol, result.raw_mut()) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);

    result.raw_mut().abi_version = GAFIME_ABI_VERSION + 1;
    // SAFETY: the owned output buffers remain live; only the result descriptor
    // version is intentionally stale.
    let status = unsafe { execute(matrix.handle().raw(), plan.protocol(), &mut result.raw) };
    assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
    result.raw.abi_version = GAFIME_ABI_VERSION;
}

#[test]
fn cuda_device_topk_returns_only_selected_rows_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };

    let rows = 4;
    let cols = 3;
    let features = vec![1.0, 5.0, 1.0, 2.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 2.0, 1.0];
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
        return;
    };
    matrix.upload(&features, &target).unwrap();

    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1, 2],
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
    )
    .with_rank(GafimeRankSpec {
        top_k: 2,
        primary_metric: GAFIME_METRIC_R2,
        descending: 1,
        include_ties: 0,
        reserved: [0; 4],
    });
    let mut result = TestResultTable::new(2, 1, 2);
    let stats = execute_plan!(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

    assert_eq!(stats.launched_chunks, 1);
    assert_eq!(stats.rows_written, 2);
    assert_eq!(result.raw.row_count, 2);
    assert_eq!(result.combo_indices(), &[0, 1]);
    assert_eq!(result.ranks(), &[0, 1]);
    assert_eq!(result.candidate_ids(), &[0, 1]);
    let values = result.metric_values();
    assert!((values[0] - 1.0).abs() < 1.0e-5);
    assert!((values[1] - 1.0).abs() < 1.0e-5);
    assert!((values[2] + 1.0).abs() < 1.0e-5);
    assert!((values[3] - 1.0).abs() < 1.0e-5);
}

#[test]
fn cuda_device_topk_keeps_large_rank_scratch_bounded_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };

    let rows = 4u64;
    let cols = 600u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    for row in 0..rows {
        features.extend(std::iter::repeat_n(row as f32, cols as usize));
    }
    let target = vec![0.0, 1.0, 2.0, 3.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        (0..cols).collect(),
        vec![GAFIME_METRIC_R2],
    )
    .with_rank(GafimeRankSpec {
        top_k: 400,
        primary_metric: GAFIME_METRIC_R2,
        descending: 1,
        include_ties: 0,
        reserved: [0; 4],
    });
    let mut result = TestResultTable::new(400, 1, 1);
    execute_plan!(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

    assert_eq!(result.raw.row_count, 400);
    assert_eq!(result.combo_indices(), (0..400).collect::<Vec<_>>());
    assert!(result.metric_values().iter().all(|value| *value > 0.999));
}

#[test]
fn cuda_continuous_cached_target_stats_refresh_after_target_update() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    continuous_cached_target_stats_refresh_after_target_update(&mut backend, GAFIME_BACKEND_CUDA);
}

#[test]
fn cuda_permutation_protocol_preserves_observed_metrics_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };

    let rows = 32u64;
    let cols = 4u32;
    let (features, target) = parity_dataset(rows, cols);
    let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
        return;
    };
    matrix.upload(&features, &target).unwrap();

    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1, 2, 3],
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
    )
    .with_permutations(GafimePermutationSchedule {
        permutation_count: 4,
        seed: 123,
        ..Default::default()
    })
    .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);

    let mut result = TestResultTable::new(4, 1, 2);
    let stats = execute_plan!(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

    assert_eq!(stats.launched_chunks, 1);
    assert_eq!(stats.graph_replays, 1);
    assert_eq!(stats.rows_written, 4);
    assert_ne!(result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED, 0);
    let observed_metrics = result.metric_values().to_vec();

    let no_permutation_plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1, 2, 3],
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
    );
    let mut restored_result = TestResultTable::new(4, 1, 2);
    execute_plan!(
        &mut backend,
        matrix.handle(),
        &no_permutation_plan,
        restored_result.raw_mut(),
    )
    .unwrap();

    assert_eq!(result.combo_indices(), restored_result.combo_indices());
    for (left, right) in observed_metrics
        .iter()
        .zip(restored_result.metric_values().iter())
    {
        assert!((*left - *right).abs() <= 5.0e-4);
    }
}

#[test]
fn cuda_reports_permutation_pvalues_when_library_exposes_optional_abi() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    if !backend.supports_permutation_pvalues() {
        return;
    }

    let rows = 64u64;
    let cols = 2u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for row in 0..rows as usize {
        let signal = row as f32;
        let noise = ((row * 17) % 29) as f32;
        features.extend([signal, noise]);
        target.push(signal);
    }

    let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
        return;
    };
    matrix.upload(&features, &target).unwrap();
    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
    )
    .with_permutations(GafimePermutationSchedule {
        permutation_count: 16,
        seed: 99,
        ..Default::default()
    })
    .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);

    let mut result = TestResultTable::new(2, 1, 2);
    execute_plan!(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();
    // SAFETY: `plan` owns the protocol buffers, and `result` owns the selected
    // candidate/metric slices for the duration of the synchronous call.
    let pvalues = unsafe {
        backend.permutation_pvalues(
            matrix.handle(),
            plan.protocol(),
            result.candidate_ids(),
            result.metric_values(),
            2,
        )
    }
    .unwrap()
    .expect("CUDA payload should expose permutation p-value ABI");

    assert_eq!(pvalues.len(), 4);
    assert!(pvalues.iter().all(|value| value.is_finite()));
    assert!(pvalues.iter().all(|&value| value > 0.0 && value <= 1.0));
    assert!(pvalues[0] <= 0.25, "signal pearson p-value={}", pvalues[0]);
    assert!(pvalues[1] <= 0.25, "signal r2 p-value={}", pvalues[1]);
}

#[test]
fn cuda_permutation_maxt_includes_hidden_family_candidates_when_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    if !backend.supports_permutation_pvalues() {
        return;
    }

    let rows = 64u64;
    let cols = 2u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for row in 0..rows as usize {
        features.extend([1.0, row as f32]);
        target.push(((row * 17 + 3) % 61) as f32);
    }
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();
    let permutations = GafimePermutationSchedule {
        permutation_count: 32,
        seed: 0x5A17,
        ..Default::default()
    };
    let selected_only = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0],
        vec![GAFIME_METRIC_PEARSON],
    )
    .with_permutations(permutations);
    let full_family = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_PEARSON],
    )
    .with_permutations(permutations);

    // SAFETY: `selected_only` owns its protocol buffers and the selected-row
    // arrays are initialized and live for this synchronous call.
    let selected_p = unsafe {
        backend.permutation_pvalues(matrix.handle(), selected_only.protocol(), &[0], &[0.1], 1)
    }
    .unwrap()
    .unwrap()[0];
    // SAFETY: `full_family` owns its protocol buffers and the selected-row
    // arrays are initialized and live for this synchronous call.
    let family_p = unsafe {
        backend.permutation_pvalues(matrix.handle(), full_family.protocol(), &[0], &[0.1], 1)
    }
    .unwrap()
    .unwrap()[0];

    let floor = 1.0 / (permutations.permutation_count as f32 + 1.0);
    assert!((selected_p - floor).abs() <= f32::EPSILON);
    assert!(
        family_p > selected_p,
        "hidden family candidate must raise maxT p-value: selected={selected_p}, family={family_p}"
    );
}

#[test]
fn cuda_matches_cpu_for_configured_continuous_plan_arity_1_to_5() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut cuda_backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };

    let rows = 32u64;
    let cols = 6u32;
    let (features, target) = parity_dataset(rows, cols);

    let mut cpu_config = continuous_config(GAFIME_BACKEND_CPU);
    let cpu_prepared = prepare_continuous_execution(&cpu_config, rows, cols).unwrap();
    let cpu_matrix =
        CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
    let mut cpu_backend = CpuBackend;
    let mut cpu_result = TestResultTable::new(
        cpu_prepared.result_capacity(),
        cpu_prepared.result_max_arity(),
        cpu_prepared.result_metric_count(),
    );
    let cpu_stats = execute_plan!(
        &mut cpu_backend,
        &cpu_matrix.handle(),
        cpu_prepared.plan(),
        cpu_result.raw_mut(),
    )
    .unwrap();

    let cuda_matrix = cuda_backend.alloc_matrix(rows, cols).unwrap();
    cuda_matrix.upload(&features, &target).unwrap();
    let cuda_config = continuous_config(GAFIME_BACKEND_CUDA);
    let cuda_prepared = prepare_continuous_execution(&cuda_config, rows, cols).unwrap();
    let mut cuda_result = TestResultTable::new(
        cuda_prepared.result_capacity(),
        cuda_prepared.result_max_arity(),
        cuda_prepared.result_metric_count(),
    );
    let cuda_stats = execute_plan!(
        &mut cuda_backend,
        cuda_matrix.handle(),
        cuda_prepared.plan(),
        cuda_result.raw_mut(),
    )
    .unwrap();

    assert_eq!(cpu_prepared.result_capacity(), 62);
    assert_eq!(cuda_prepared.result_capacity(), 62);
    assert_eq!(cpu_stats.launched_chunks, 5);
    assert_eq!(cuda_stats.launched_chunks, 5);
    assert_eq!(cpu_result.raw.row_count, cuda_result.raw.row_count);
    assert_eq!(cpu_result.combo_indices(), cuda_result.combo_indices());

    for (index, (&cpu_value, &cuda_value)) in cpu_result
        .metric_values()
        .iter()
        .zip(cuda_result.metric_values())
        .enumerate()
    {
        let delta = (cpu_value - cuda_value).abs();
        assert!(
            delta <= 5.0e-4,
            "metric mismatch at {index}: cpu={cpu_value} cuda={cuda_value} delta={delta}"
        );
    }

    cpu_config.backend_kind = GAFIME_BACKEND_CUDA;
    let explicit_cuda_prepared = prepare_continuous_execution(&cpu_config, rows, cols).unwrap();
    assert_eq!(
        explicit_cuda_prepared.plan().protocol().backend_kind,
        GAFIME_BACKEND_CUDA
    );
}
