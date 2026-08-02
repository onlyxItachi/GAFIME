use super::*;

#[test]
fn cuda_graph_flag_replays_same_continuous_result_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    let capability = backend.graph_capability().unwrap();
    assert_eq!(capability.graph_mode, GAFIME_GRAPH_STREAM_CAPTURE);
    assert_eq!(capability.supports_device_ranking, 1);

    let rows = 32u64;
    let cols = 6u32;
    let (features, target) = parity_dataset(rows, cols);
    let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
        return;
    };
    matrix.upload(&features, &target).unwrap();

    let config = continuous_config(GAFIME_BACKEND_CUDA);
    let normal_prepared = prepare_continuous_execution(&config, rows, cols).unwrap();
    let graph_plan = prepare_continuous_execution(&config, rows, cols)
        .unwrap()
        .into_plan()
        .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);

    let mut normal_result = TestResultTable::new(
        normal_prepared.result_capacity(),
        normal_prepared.result_max_arity(),
        normal_prepared.result_metric_count(),
    );
    execute_plan(
        &mut backend,
        matrix.handle(),
        normal_prepared.plan(),
        normal_result.raw_mut(),
    )
    .unwrap();

    let mut first_graph_result = TestResultTable::new(
        normal_prepared.result_capacity(),
        normal_prepared.result_max_arity(),
        normal_prepared.result_metric_count(),
    );
    let first_stats = execute_plan(
        &mut backend,
        matrix.handle(),
        &graph_plan,
        first_graph_result.raw_mut(),
    )
    .unwrap();
    assert_eq!(first_stats.graph_replays, 1);
    assert_ne!(
        first_graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
        0
    );

    let mut second_graph_result = TestResultTable::new(
        normal_prepared.result_capacity(),
        normal_prepared.result_max_arity(),
        normal_prepared.result_metric_count(),
    );
    let second_stats = execute_plan(
        &mut backend,
        matrix.handle(),
        &graph_plan,
        second_graph_result.raw_mut(),
    )
    .unwrap();
    assert_eq!(second_stats.graph_replays, 1);

    assert_eq!(
        normal_result.raw.row_count,
        first_graph_result.raw.row_count
    );
    assert_eq!(
        normal_result.combo_indices(),
        first_graph_result.combo_indices()
    );
    assert_eq!(
        first_graph_result.combo_indices(),
        second_graph_result.combo_indices()
    );
    for ((normal, first), second) in normal_result
        .metric_values()
        .iter()
        .zip(first_graph_result.metric_values())
        .zip(second_graph_result.metric_values())
    {
        assert!((*normal - *first).abs() <= 5.0e-4);
        assert!((*first - *second).abs() <= 1.0e-6);
    }
}

#[test]
fn cuda_graph_captures_whole_multi_arity_sweep_when_available() {
    // The CUDA host captures the entire multi-arity sweep (every chunk +
    // metric) into ONE graph, not one graph per shape. Validate that a
    // multi-chunk plan replays as a single graph with results identical to a
    // normal launch.
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };

    let rows = 32u64;
    let cols = 5u32;
    let (features, target) = parity_dataset(rows, cols);
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let request = |flags: u32| {
        let mut plan = build_continuous_plan(ContinuousPlanRequest {
            precision: PrecisionProfile::Fp32,
            backend_kind: GAFIME_BACKEND_CUDA,
            n_samples: rows,
            n_features: cols,
            max_arity: 3,
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
    assert!(
        graph_plan.chunks().len() >= 3,
        "arity 1..3 should produce several chunks (a real sweep)"
    );
    let planned: u64 = graph_plan.chunks().iter().map(|c| c.combo_count).sum();

    let mut graph_result = TestResultTable::new(planned, 3, 2);
    execute_plan(
        &mut backend,
        matrix.handle(),
        &graph_plan,
        graph_result.raw_mut(),
    )
    .unwrap();
    assert_ne!(
        graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
        0,
        "the whole multi-arity sweep must replay as one graph"
    );

    let normal_plan = request(0);
    let mut normal_result = TestResultTable::new(planned, 3, 2);
    execute_plan(
        &mut backend,
        matrix.handle(),
        &normal_plan,
        normal_result.raw_mut(),
    )
    .unwrap();

    assert_eq!(graph_result.raw.row_count, normal_result.raw.row_count);
    assert_eq!(graph_result.combo_indices(), normal_result.combo_indices());
    for (g, n) in graph_result
        .metric_values()
        .iter()
        .zip(normal_result.metric_values())
    {
        assert!((g - n).abs() <= 5.0e-4, "graph vs normal: {g} vs {n}");
    }
}
