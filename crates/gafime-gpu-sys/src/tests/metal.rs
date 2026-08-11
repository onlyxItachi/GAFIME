use super::*;

#[test]
fn metal_execution_memory_peak_tracks_descriptor_and_ranking_state_when_available() {
    let _metal_guard = metal_test_lock();
    let Some(mut backend) = metal_backend_for_test() else {
        return;
    };

    let rows = 5u64;
    let cols = 4u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for row in 0..rows as usize {
        let value = row as f32;
        features.extend([value, value * value, (row % 2) as f32, 1.0]);
        target.push(value);
    }
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();
    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_METAL,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        (0..cols).collect(),
        vec![GAFIME_METRIC_R2],
    );
    let mut protocol = *plan.protocol();
    protocol.flags |= GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 7_001;

    // SAFETY: `protocol` copies a descriptor whose pointed-to spans remain
    // owned by `plan` for this synchronous query.
    let cold_peak =
        unsafe { backend.execution_device_memory_peak_bytes(matrix.handle(), &protocol) }
            .unwrap()
            .expect("current Metal payload must export execution-memory preflight");
    // SAFETY: `plan` still owns every span referenced by `protocol`.
    let repeated_peak =
        unsafe { backend.execution_device_memory_peak_bytes(matrix.handle(), &protocol) }
            .unwrap()
            .expect("current Metal payload must keep execution-memory preflight available");
    assert_eq!(
        cold_peak, repeated_peak,
        "preflight must not mutate cache state"
    );

    let mut result = TestResultTable::new(cols as u64, 1, 1);
    // SAFETY: `plan` owns every input span and `result` owns correctly sized,
    // uniquely borrowed output buffers for the synchronous execution.
    unsafe { backend.execute(matrix.handle(), &protocol, result.raw_mut()) }.unwrap();
    // SAFETY: `plan` still owns every span referenced by `protocol`.
    let warm_peak =
        unsafe { backend.execution_device_memory_peak_bytes(matrix.handle(), &protocol) }
            .unwrap()
            .unwrap();
    assert!(warm_peak <= cold_peak);

    let mut ranked_protocol = protocol;
    ranked_protocol.rank = GafimeRankSpec {
        top_k: 2,
        primary_metric: GAFIME_METRIC_R2,
        descending: 1,
        include_ties: 0,
        reserved: [0; 4],
    };
    // SAFETY: ranking changes only inline fields; all pointed-to spans remain
    // owned by `plan` and live for this query.
    let ranked_peak =
        unsafe { backend.execution_device_memory_peak_bytes(matrix.handle(), &ranked_protocol) }
            .unwrap()
            .unwrap();
    assert!(ranked_peak > warm_peak, "ranking buffers must be admitted");
    let mut ranked_result = TestResultTable::new(2, 1, 1);
    // SAFETY: `plan` owns every input span and `ranked_result` owns correctly
    // sized, uniquely borrowed output buffers.
    unsafe { backend.execute(matrix.handle(), &ranked_protocol, ranked_result.raw_mut()) }.unwrap();
    assert_eq!(ranked_result.raw.row_count, 2);
    assert_eq!(
        // SAFETY: `plan` still owns every span referenced by `ranked_protocol`.
        unsafe { backend.execution_device_memory_peak_bytes(matrix.handle(), &ranked_protocol) }
            .unwrap(),
        Some(ranked_peak)
    );
}

#[test]
fn metal_descriptor_cache_generation_refreshes_reused_addresses_when_available() {
    let _metal_guard = metal_test_lock();
    let Some(mut backend) = metal_backend_for_test() else {
        return;
    };
    assert!(
        backend.supports_descriptor_generation(),
        "configured Metal payload must advertise descriptor-generation support"
    );

    let rows = 4u64;
    let cols = 2u32;
    let features = vec![1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0];
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_METAL,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0],
        vec![GAFIME_METRIC_PEARSON],
    );
    let mut descriptors = [0u32];
    let descriptor_address = descriptors.as_ptr();
    let mut first_protocol = *plan.protocol();
    first_protocol.combo_indices.ptr = descriptor_address;
    first_protocol.flags |= GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    first_protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 101;

    let mut first_result = TestResultTable::new(1, 1, 1);
    // SAFETY: `first_protocol` points into live `plan` storage plus the live
    // `descriptors` array, and `first_result` owns its output buffers.
    unsafe { backend.execute(matrix.handle(), &first_protocol, first_result.raw_mut()) }.unwrap();
    assert_eq!(first_result.combo_indices(), &[0]);
    assert!(first_result.metric_values()[0] > 0.999);

    descriptors[0] = 1;
    assert_eq!(descriptors.as_ptr(), descriptor_address);
    let mut second_protocol = first_protocol;
    second_protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 102;
    let mut second_result = TestResultTable::new(1, 1, 1);
    // SAFETY: the updated descriptor array and all plan-owned spans are live;
    // `second_result` uniquely owns correctly sized output buffers.
    unsafe { backend.execute(matrix.handle(), &second_protocol, second_result.raw_mut()) }.unwrap();
    assert_eq!(second_result.combo_indices(), &[1]);
    assert!(second_result.metric_values()[0] < -0.999);

    descriptors[0] = 0;
    assert_eq!(descriptors.as_ptr(), descriptor_address);
    let mut replay_result = TestResultTable::new(1, 1, 1);
    // SAFETY: `second_protocol` still references live descriptor/plan storage,
    // and `replay_result` uniquely owns correctly sized output buffers.
    unsafe { backend.execute(matrix.handle(), &second_protocol, replay_result.raw_mut()) }.unwrap();
    assert!(replay_result.metric_values()[0] < -0.999);
    assert!((replay_result.metric_values()[0] - second_result.metric_values()[0]).abs() < 1.0e-5);
}

#[test]
fn metal_device_topk_covers_split_directions_ties_and_large_k_when_available() {
    let _metal_guard = metal_test_lock();
    let Some(mut backend) = metal_backend_for_test() else {
        return;
    };
    if std::env::var_os("GAFIME_REQUIRE_CURRENT_DEVICE_RANKING").is_some() {
        assert_eq!(
            backend.graph_capability().unwrap().supports_device_ranking,
            1
        );
    }

    {
        let rows = 5u64;
        let cols = 4u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            let value = row as f32;
            features.extend([value, value * value, (row % 2) as f32, 1.0]);
            target.push(value);
        }
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        for (descending, expected) in [(1, [0u32, 1]), (0, [2u32, 3])] {
            let plan = CompiledPlan::single_chunk(
                GAFIME_BACKEND_METAL,
                rows,
                cols,
                GAFIME_FAMILY_CONTINUOUS,
                1,
                (0..cols).collect(),
                vec![GAFIME_METRIC_R2],
            )
            .with_rank(GafimeRankSpec {
                top_k: 2,
                primary_metric: GAFIME_METRIC_R2,
                descending,
                include_ties: 0,
                reserved: [0; 4],
            });
            let mut result = TestResultTable::new(2, 1, 1);
            execute_plan!(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

            assert_eq!(result.combo_indices(), &expected);
            assert_eq!(result.candidate_ids(), &expected.map(u64::from));
            if descending != 0 {
                assert!(result.metric_values()[0] > 0.999);
                assert!(result.metric_values()[1] > 0.8);
            } else {
                assert!(result
                    .metric_values()
                    .iter()
                    .all(|value| value.abs() < 1.0e-5));
            }
        }
    }

    let rows = 4u64;
    let cols = 600u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    for row in 0..rows {
        features.extend(std::iter::repeat_n(row as f32, cols as usize));
    }
    let target = vec![0.0, 1.0, 2.0, 3.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    for descending in [0, 1] {
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_METAL,
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
            descending,
            include_ties: 0,
            reserved: [0; 4],
        });
        let mut result = TestResultTable::new(400, 1, 1);
        execute_plan!(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(result.raw.row_count, 400);
        assert_eq!(result.combo_indices(), (0..400).collect::<Vec<_>>());
        assert_eq!(result.candidate_ids(), (0u64..400).collect::<Vec<_>>());
        assert!(result.metric_values().iter().all(|value| *value > 0.999));
    }

    let oversized_plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_METAL,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        (0..cols).collect(),
        vec![GAFIME_METRIC_R2],
    )
    .with_rank(GafimeRankSpec {
        top_k: 700,
        primary_metric: GAFIME_METRIC_R2,
        descending: 1,
        include_ties: 0,
        reserved: [0; 4],
    });
    let mut oversized_result = TestResultTable::new(700, 1, 1);
    execute_plan!(
        &mut backend,
        matrix.handle(),
        &oversized_plan,
        oversized_result.raw_mut(),
    )
    .unwrap();
    assert_eq!(oversized_result.raw.row_count, u64::from(cols));
    assert_eq!(
        oversized_result.combo_indices(),
        (0..cols).collect::<Vec<_>>()
    );
}

#[test]
fn metal_continuous_metrics_match_cpu_on_high_dynamic_and_nonfinite_inputs_when_available() {
    const DEFAULT_METAL_PARITY_TOLERANCE: f32 = 5.0e-5;
    const METRIC_NAMES: [&str; 4] = ["pearson", "r2", "mutual_info", "spearman"];

    let _metal_guard = metal_test_lock();
    let Some(mut metal_backend) = metal_backend_for_test() else {
        return;
    };
    let tolerance = match env::var("GAFIME_METAL_PARITY_TOLERANCE") {
        Ok(value) => value
            .parse::<f32>()
            .expect("GAFIME_METAL_PARITY_TOLERANCE must be a finite positive float"),
        Err(env::VarError::NotPresent) => DEFAULT_METAL_PARITY_TOLERANCE,
        Err(env::VarError::NotUnicode(_)) => {
            panic!("GAFIME_METAL_PARITY_TOLERANCE must be valid UTF-8")
        }
    };
    assert!(
        tolerance.is_finite() && tolerance > 0.0,
        "GAFIME_METAL_PARITY_TOLERANCE must be a finite positive float"
    );

    let rows = 160u64;
    let cols = 5u32;
    for inject_nonfinite in [false, true] {
        let (features, target) = metal_parity_dataset(rows, cols, inject_nonfinite);
        let prepare = |backend_kind| {
            let mut config = EngineConfig {
                backend_kind,
                metric_ids: vec![
                    GAFIME_METRIC_PEARSON,
                    GAFIME_METRIC_R2,
                    GAFIME_METRIC_MUTUAL_INFO,
                    GAFIME_METRIC_SPEARMAN,
                ],
                mi_bins: 96,
                mi_approximate: true,
                permutation_tests: 0,
                ..Default::default()
            };
            config.budget.max_comb_size = 5;
            config.budget.max_combinations_per_k = 100;
            prepare_continuous_execution(&config, rows, cols).unwrap()
        };

        let cpu_prepared = prepare(GAFIME_BACKEND_CPU);
        let cpu_matrix =
            CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
        let mut cpu_backend = CpuBackend;
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

        let metal_prepared = prepare(GAFIME_BACKEND_METAL);
        let metal_matrix = metal_backend.alloc_matrix(rows, cols).unwrap();
        metal_matrix.upload(&features, &target).unwrap();
        let mut metal_result = TestResultTable::new(
            metal_prepared.result_capacity(),
            metal_prepared.result_max_arity(),
            metal_prepared.result_metric_count(),
        );
        execute_plan!(
            &mut metal_backend,
            metal_matrix.handle(),
            metal_prepared.plan(),
            metal_result.raw_mut(),
        )
        .unwrap();

        assert_eq!(cpu_result.raw.row_count, 31);
        assert_eq!(cpu_result.raw.row_count, metal_result.raw.row_count);
        assert_eq!(cpu_result.combo_indices(), metal_result.combo_indices());
        assert_eq!(cpu_result.candidate_ids(), metal_result.candidate_ids());
        assert_eq!(
            cpu_result.raw.metric_count as usize,
            METRIC_NAMES.len(),
            "Metal parity evidence metric names must cover every result metric"
        );
        let mut max_abs_by_metric = [0.0_f32; METRIC_NAMES.len()];
        for (index, (&cpu_value, &metal_value)) in cpu_result
            .metric_values()
            .iter()
            .zip(metal_result.metric_values())
            .enumerate()
        {
            let delta = (cpu_value - metal_value).abs();
            let metric_index = index % METRIC_NAMES.len();
            max_abs_by_metric[metric_index] = max_abs_by_metric[metric_index].max(delta);
            assert!(
                cpu_value.is_finite() && metal_value.is_finite() && delta <= tolerance,
                "Metal parity mismatch at metric value {index} (nonfinite={inject_nonfinite}): \
                 cpu={cpu_value} metal={metal_value} delta={delta} tolerance={tolerance}"
            );
        }
        eprintln!(
            "METAL_PARITY_EVIDENCE nonfinite={inject_nonfinite} rows={rows} cols={cols} \
             candidates={} tolerance={tolerance:.9e} pearson_max_abs={:.9e} \
             r2_max_abs={:.9e} mutual_info_max_abs={:.9e} spearman_max_abs={:.9e}",
            cpu_result.raw.row_count,
            max_abs_by_metric[0],
            max_abs_by_metric[1],
            max_abs_by_metric[2],
            max_abs_by_metric[3],
        );
    }
}

#[test]
fn metal_fp32_precision_metrics_match_core_fp32_on_high_dynamic_and_nonfinite_inputs_when_available(
) {
    const DEFAULT_METAL_PARITY_TOLERANCE: f32 = 5.0e-5;
    const DEFAULT_METAL_FP32_CROSS_BACKEND_TOLERANCE: f32 = 2.0e-4;
    const METAL_CORE_ORDERED_FP32_MAX_ROWS: u64 = 4 * 64;
    const METRIC_NAMES: [&str; 4] = ["pearson", "r2", "mutual_info", "spearman"];

    let _metal_guard = metal_test_lock();
    let Some(mut metal_backend) = metal_backend_for_test() else {
        return;
    };
    let tolerance = match env::var("GAFIME_METAL_PARITY_TOLERANCE") {
        Ok(value) => value
            .parse::<f32>()
            .expect("GAFIME_METAL_PARITY_TOLERANCE must be a finite positive float"),
        Err(env::VarError::NotPresent) => DEFAULT_METAL_PARITY_TOLERANCE,
        Err(env::VarError::NotUnicode(_)) => {
            panic!("GAFIME_METAL_PARITY_TOLERANCE must be valid UTF-8")
        }
    };
    assert!(
        tolerance.is_finite() && tolerance > 0.0,
        "GAFIME_METAL_PARITY_TOLERANCE must be a finite positive float"
    );

    let cols = 5u32;
    for rows in [160, 255, 256, 257] {
        let case_tolerance = if rows <= METAL_CORE_ORDERED_FP32_MAX_ROWS {
            tolerance
        } else {
            DEFAULT_METAL_FP32_CROSS_BACKEND_TOLERANCE
        };
        for inject_nonfinite in [false, true] {
            let (features, target) = metal_parity_dataset(rows, cols, inject_nonfinite);
            let prepare = |backend_kind| {
                let mut config = EngineConfig {
                    precision: PrecisionProfile::Fp32,
                    backend_kind,
                    metric_ids: vec![
                        GAFIME_METRIC_PEARSON,
                        GAFIME_METRIC_R2,
                        GAFIME_METRIC_MUTUAL_INFO,
                        GAFIME_METRIC_SPEARMAN,
                    ],
                    mi_bins: 96,
                    mi_approximate: true,
                    permutation_tests: 0,
                    ..Default::default()
                };
                config.budget.max_comb_size = 5;
                config.budget.max_combinations_per_k = 100;
                prepare_continuous_execution(&config, rows, cols).unwrap()
            };

            let cpu_prepared = prepare(GAFIME_BACKEND_CPU);
            let cpu_matrix = CpuPrecisionMatrix::from_row_major_f32(
                PrecisionProfile::Fp32,
                rows,
                cols,
                features.clone(),
                target.clone(),
            )
            .unwrap();
            let mut cpu_backend = CpuBackend;
            let mut cpu_result = TestResultTable::new(
                cpu_prepared.result_capacity(),
                cpu_prepared.result_max_arity(),
                cpu_prepared.result_metric_count(),
            );
            // SAFETY: `cpu_prepared` owns the protocol graph and `cpu_result`
            // owns result buffers sized from that prepared execution.
            unsafe {
                cpu_prepared.execute_precision_fp32(
                    &mut cpu_backend,
                    &cpu_matrix.handle(),
                    cpu_result.raw_mut(),
                )
            }
            .unwrap();

            let metal_prepared = prepare(GAFIME_BACKEND_METAL);
            let metal_matrix = metal_backend
                .alloc_matrix_for_profile(PrecisionProfile::Fp32, rows, cols)
                .unwrap();
            metal_matrix.upload_f32_v2(&features, &target).unwrap();
            let mut metal_result = TestResultTable::new(
                metal_prepared.result_capacity(),
                metal_prepared.result_max_arity(),
                metal_prepared.result_metric_count(),
            );
            // SAFETY: `metal_prepared` owns the protocol graph and
            // `metal_result` owns result buffers sized from it.
            unsafe {
                metal_prepared.execute_precision_fp32(
                    &mut metal_backend,
                    metal_matrix.handle(),
                    metal_result.raw_mut(),
                )
            }
            .unwrap();

            assert_eq!(cpu_result.raw.row_count, 31);
            assert_eq!(cpu_result.raw.row_count, metal_result.raw.row_count);
            assert_eq!(cpu_result.combo_indices(), metal_result.combo_indices());
            assert_eq!(cpu_result.candidate_ids(), metal_result.candidate_ids());
            assert_eq!(
                cpu_result.raw.metric_count as usize,
                METRIC_NAMES.len(),
                "typed Metal fp32 parity evidence metric names must cover every result metric"
            );
            let mut max_abs_by_metric = [0.0_f32; METRIC_NAMES.len()];
            for (index, (&cpu_value, &metal_value)) in cpu_result
                .metric_values()
                .iter()
                .zip(metal_result.metric_values())
                .enumerate()
            {
                let delta = (cpu_value - metal_value).abs();
                let metric_index = index % METRIC_NAMES.len();
                max_abs_by_metric[metric_index] = max_abs_by_metric[metric_index].max(delta);
                assert!(
                    cpu_value.is_finite() && metal_value.is_finite(),
                    "typed Metal fp32 parity produced a nonfinite metric value at {index} \
                     (rows={rows} nonfinite={inject_nonfinite}): \
                     core={cpu_value} metal={metal_value}"
                );
            }
            eprintln!(
                "METAL_FP32_PARITY_EVIDENCE nonfinite={inject_nonfinite} rows={rows} cols={cols} \
                 candidates={} tolerance={case_tolerance:.9e} pearson_max_abs={:.9e} \
                 r2_max_abs={:.9e} mutual_info_max_abs={:.9e} spearman_max_abs={:.9e}",
                cpu_result.raw.row_count,
                max_abs_by_metric[0],
                max_abs_by_metric[1],
                max_abs_by_metric[2],
                max_abs_by_metric[3],
            );
            assert!(
                max_abs_by_metric
                    .iter()
                    .all(|&delta| delta <= case_tolerance),
                "typed Metal fp32 parity exceeded the documented cross-backend bound \
                 (rows={rows} nonfinite={inject_nonfinite}): maxima={max_abs_by_metric:?} \
                 tolerance={case_tolerance}"
            );
        }
    }
}

#[test]
fn metal_nonfinite_correlation_is_not_laundered_when_library_is_available() {
    let _metal_guard = metal_test_lock();
    let Some(mut backend) = metal_backend_for_test() else {
        return;
    };
    assert_nonfinite_correlation_is_not_laundered(&mut backend, GAFIME_BACKEND_METAL);
}

#[test]
fn metal_scaled_covariance_matches_cpu_across_dynamic_range_when_available() {
    let _metal_guard = metal_test_lock();
    let Some(mut backend) = metal_backend_for_test() else {
        return;
    };
    assert_scaled_covariance_matches_cpu_across_dynamic_range(
        &mut backend,
        GAFIME_BACKEND_METAL,
        2.0e-3,
    );
}
