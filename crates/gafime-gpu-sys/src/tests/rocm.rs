use super::*;

#[test]
fn rocm_matches_cpu_for_continuous_pearson_r2_when_library_is_available() {
    // ROCm supports the continuous pearson/r2 subset on gfx1150; validate it
    // against the CPU reference over the same plan (skips without the payload).
    let Ok(mut rocm_backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };

    let rows = 32u64;
    let cols = 6u32;
    let (features, target) = parity_dataset(rows, cols);

    let cpu_config = continuous_config(GAFIME_BACKEND_CPU);
    let cpu_prepared = prepare_continuous_execution(&cpu_config, rows, cols).unwrap();
    let cpu_matrix =
        CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
    let mut cpu_backend = CpuBackend;
    let mut cpu_result = TestResultTable::new(
        cpu_prepared.result_capacity(),
        cpu_prepared.result_max_arity(),
        cpu_prepared.result_metric_count(),
    );
    execute_plan(
        &mut cpu_backend,
        &cpu_matrix.handle(),
        cpu_prepared.plan(),
        cpu_result.raw_mut(),
    )
    .unwrap();

    let rocm_matrix = rocm_backend.alloc_matrix(rows, cols).unwrap();
    rocm_matrix.upload(&features, &target).unwrap();
    let rocm_config = continuous_config(GAFIME_BACKEND_ROCM);
    let rocm_prepared = prepare_continuous_execution(&rocm_config, rows, cols).unwrap();
    let mut rocm_result = TestResultTable::new(
        rocm_prepared.result_capacity(),
        rocm_prepared.result_max_arity(),
        rocm_prepared.result_metric_count(),
    );
    let rocm_stats = execute_plan(
        &mut rocm_backend,
        &rocm_matrix.handle(),
        rocm_prepared.plan(),
        rocm_result.raw_mut(),
    )
    .unwrap();

    assert_eq!(
        rocm_prepared.plan().protocol().backend_kind,
        GAFIME_BACKEND_ROCM
    );
    assert_eq!(rocm_backend.backend_kind(), GAFIME_BACKEND_ROCM);
    assert_eq!(rocm_stats.rows_written, cpu_result.raw.row_count);
    assert_eq!(cpu_result.raw.row_count, rocm_result.raw.row_count);
    assert_eq!(cpu_result.combo_indices(), rocm_result.combo_indices());
    for (index, (&cpu_value, &rocm_value)) in cpu_result
        .metric_values()
        .iter()
        .zip(rocm_result.metric_values())
        .enumerate()
    {
        let delta = (cpu_value - rocm_value).abs();
        assert!(
            delta <= 5.0e-4,
            "metric mismatch at {index}: cpu={cpu_value} rocm={rocm_value} delta={delta}"
        );
    }
}

#[test]
fn rocm_nonfinite_correlation_is_not_laundered_when_library_is_available() {
    let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };
    assert_nonfinite_correlation_is_not_laundered(&mut backend, GAFIME_BACKEND_ROCM);
}

#[test]
fn rocm_scaled_covariance_matches_cpu_across_dynamic_range_when_available() {
    let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };
    assert_scaled_covariance_matches_cpu_across_dynamic_range(
        &mut backend,
        GAFIME_BACKEND_ROCM,
        5.0e-4,
    );
}

#[test]
fn rocm_mutual_info_detects_signal_and_matches_cuda_when_available() {
    let Ok(mut rocm_backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };

    let rows = 128u64;
    let cols = 2u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for row in 0..rows as usize {
        let x0 = (row % 16) as f32 / 15.0;
        let x1 = ((row * 7) % 23) as f32 / 22.0;
        features.extend([x0, x1]);
        target.push(if x0 > 0.5 { 1.0 } else { 0.0 });
    }

    let rocm_matrix = rocm_backend.alloc_matrix(rows, cols).unwrap();
    rocm_matrix.upload(&features, &target).unwrap();
    let rocm_plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_ROCM,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_MUTUAL_INFO],
    );
    let mut rocm_result = TestResultTable::new(2, 1, 1);
    execute_plan(
        &mut rocm_backend,
        &rocm_matrix.handle(),
        &rocm_plan,
        rocm_result.raw_mut(),
    )
    .unwrap();
    let rocm_mi = rocm_result.metric_values().to_vec();
    assert!(rocm_mi[0].is_finite() && rocm_mi[1].is_finite());
    assert!(rocm_mi[0] >= 0.0);
    assert!(
        rocm_mi[0] > rocm_mi[1],
        "MI must detect the x0->target signal: {rocm_mi:?}"
    );

    // The ROCm MI kernel is a verbatim port of the CUDA fixed-binning kernel,
    // so their outputs match within fp tolerance on the same input.
    let _cuda_guard = cuda_test_lock();
    if let Ok(mut cuda_backend) = GpuBackend::cuda_from_env(0) {
        let cuda_matrix = cuda_backend.alloc_matrix(rows, cols).unwrap();
        cuda_matrix.upload(&features, &target).unwrap();
        let cuda_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_MUTUAL_INFO],
        );
        let mut cuda_result = TestResultTable::new(2, 1, 1);
        execute_plan(
            &mut cuda_backend,
            &cuda_matrix.handle(),
            &cuda_plan,
            cuda_result.raw_mut(),
        )
        .unwrap();
        for (i, (&r, &c)) in rocm_mi.iter().zip(cuda_result.metric_values()).enumerate() {
            assert!(
                (r - c).abs() <= 1.0e-3,
                "ROCm/CUDA MI mismatch at {i}: rocm={r} cuda={c}"
            );
        }
    }
}

#[test]
fn rocm_all_adaptive_mi_templates_match_cpu_for_arity_1_to_5_when_library_is_available() {
    let Some(mut rocm_backend) = rocm_backend_for_specialization_test() else {
        return;
    };
    assert_adaptive_mi_templates_match_cpu_for_arity_1_to_5(
        &mut rocm_backend,
        GAFIME_BACKEND_ROCM,
        MI_TEMPLATE_BIN_LEVELS,
    );
}

#[test]
fn rocm_low_signal_mi_matches_cpu_when_library_is_available() {
    let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };
    assert_low_signal_mi_matches_cpu(&mut backend, GAFIME_BACKEND_ROCM);
}

#[test]
fn rocm_adaptive_mi_96_matches_cpu_for_arity_1_to_5_when_library_is_available() {
    let require_wave64 = env::var_os("GAFIME_REQUIRE_ROCM_WAVE64_MI").is_some();
    if !require_wave64 {
        return;
    }
    let mut rocm_backend = GpuBackend::rocm_from_env(0)
        .unwrap_or_else(|error| panic!("required wave64 ROCm payload failed to load: {error}"));
    let device_info = rocm_backend.device_info().unwrap();
    assert_eq!(device_info.warp_size, 64, "wave64 MI validation required");
    assert_ne!(
        device_info.flags & GAFIME_GPU_DEVICE_FLAG_AMD_CDNA,
        0,
        "wave64 MI validation requires a CDNA device"
    );
    assert_adaptive_mi_templates_match_cpu_for_arity_1_to_5(
        &mut rocm_backend,
        GAFIME_BACKEND_ROCM,
        &[96],
    );
}

#[test]
fn rocm_device_topk_selects_by_primary_metric_when_available() {
    // ROCm device top-k should keep the same deterministic primary-metric
    // ordering as the CPU/CUDA paths.
    let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };
    if std::env::var_os("GAFIME_REQUIRE_CURRENT_DEVICE_RANKING").is_some() {
        assert_eq!(
            backend.graph_capability().unwrap().supports_device_ranking,
            1
        );
    }

    let rows = 4;
    let cols = 3;
    let features = vec![1.0, 5.0, 1.0, 2.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 2.0, 1.0];
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_ROCM,
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
    let stats = execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

    assert_eq!(stats.rows_written, 2);
    assert_eq!(result.raw.row_count, 2);
    assert_eq!(result.combo_indices(), &[0, 1]);
    assert_eq!(result.ranks(), &[0, 1]);
    assert_eq!(result.candidate_ids(), &[0, 1]);
    let values = result.metric_values();
    assert!((values[1] - 1.0).abs() < 1.0e-5); // r2 of feature 0
    assert!((values[3] - 1.0).abs() < 1.0e-5); // r2 of feature 1
}

#[test]
fn rocm_device_topk_keeps_large_rank_scratch_bounded_when_library_is_available() {
    let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };

    let rows = 4u64;
    let cols = 600u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    for row in 0..rows {
        features.extend(std::iter::repeat(row as f32).take(cols as usize));
    }
    let target = vec![0.0, 1.0, 2.0, 3.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_ROCM,
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
    execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

    assert_eq!(result.raw.row_count, 400);
    assert_eq!(result.combo_indices(), (0..400).collect::<Vec<_>>());
    assert!(result.metric_values().iter().all(|value| *value > 0.999));
}

#[test]
fn rocm_spearman_matches_cpu_when_library_is_available() {
    let Ok(mut rocm_backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };

    let rows = 48u64;
    let cols = 3u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for r in 0..rows as usize {
        let a = r as f32 * 0.13;
        let b = ((r * 7) % 17) as f32;
        let c = (rows as usize - r) as f32;
        features.extend([a, b, c]);
        target.push(a * a * a);
    }

    let cpu_plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CPU,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1, 2],
        vec![GAFIME_METRIC_SPEARMAN],
    );
    let cpu_matrix =
        CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
    let mut cpu_backend = CpuBackend;
    let mut cpu_result = TestResultTable::new(3, 1, 1);
    execute_plan(
        &mut cpu_backend,
        &cpu_matrix.handle(),
        &cpu_plan,
        cpu_result.raw_mut(),
    )
    .unwrap();

    let rocm_matrix = rocm_backend.alloc_matrix(rows, cols).unwrap();
    rocm_matrix.upload(&features, &target).unwrap();
    let rocm_plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_ROCM,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1, 2],
        vec![GAFIME_METRIC_SPEARMAN],
    );
    let mut rocm_result = TestResultTable::new(3, 1, 1);
    execute_plan(
        &mut rocm_backend,
        &rocm_matrix.handle(),
        &rocm_plan,
        rocm_result.raw_mut(),
    )
    .unwrap();

    let cpu_vals = cpu_result.metric_values();
    let rocm_vals = rocm_result.metric_values();
    assert!(cpu_vals[0] > 0.999);
    assert!(cpu_vals[2] < -0.999);
    for (i, (&c, &g)) in cpu_vals.iter().zip(rocm_vals).enumerate() {
        assert!(
            (c - g).abs() <= 1.0e-4,
            "spearman mismatch at {i}: cpu={c} rocm={g}"
        );
    }
}

#[test]
fn rocm_graph_captures_and_replays_the_sweep_when_available() {
    // ROCm device-copy stream-capture: the multi-arity sweep is captured once
    // and replayed; results must match a normal launch, and a second run must
    // reuse the cached graph.
    let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };

    let rows = 32u64;
    let cols = 4u32;
    let (features, target) = parity_dataset(rows, cols);
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let request = |flags: u32| {
        let mut plan = build_continuous_plan(ContinuousPlanRequest {
            backend_kind: GAFIME_BACKEND_ROCM,
            n_samples: rows,
            n_features: cols,
            max_arity: 2,
            max_combinations_per_arity: 1_000,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            mi_bins: 96,
            rank: Default::default(),
        })
        .unwrap();
        if flags != 0 {
            plan = plan.with_flags(flags);
        }
        plan
    };

    let graph_plan = request(GAFIME_LAUNCH_FLAG_GRAPH);
    let planned: u64 = graph_plan.chunks().iter().map(|c| c.combo_count).sum();

    let mut graph_result = TestResultTable::new(planned, 2, 2);
    execute_plan(
        &mut backend,
        &matrix.handle(),
        &graph_plan,
        graph_result.raw_mut(),
    )
    .unwrap();
    assert_ne!(
        graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
        0,
        "ROCm should capture + replay the sweep as a graph"
    );

    // Second run reuses the cached graph (same shape/signature).
    let mut graph_result2 = TestResultTable::new(planned, 2, 2);
    execute_plan(
        &mut backend,
        &matrix.handle(),
        &request(GAFIME_LAUNCH_FLAG_GRAPH),
        graph_result2.raw_mut(),
    )
    .unwrap();
    assert_ne!(
        graph_result2.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
        0
    );

    let normal_plan = request(0);
    let mut normal_result = TestResultTable::new(planned, 2, 2);
    execute_plan(
        &mut backend,
        &matrix.handle(),
        &normal_plan,
        normal_result.raw_mut(),
    )
    .unwrap();

    assert_eq!(graph_result.combo_indices(), normal_result.combo_indices());
    for (g, n) in graph_result
        .metric_values()
        .iter()
        .zip(normal_result.metric_values())
    {
        assert!((g - n).abs() <= 5.0e-4, "graph vs normal: {g} vs {n}");
    }
    for (g, n) in graph_result2
        .metric_values()
        .iter()
        .zip(normal_result.metric_values())
    {
        assert!((g - n).abs() <= 5.0e-4);
    }
}

#[test]
fn rocm_continuous_cached_target_stats_refresh_after_target_update() {
    let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
        return;
    };
    continuous_cached_target_stats_refresh_after_target_update(&mut backend, GAFIME_BACKEND_ROCM);
}
