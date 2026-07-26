use super::*;

#[test]
fn cuda_mutual_info_metric_returns_finite_signal_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
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
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();
    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_MUTUAL_INFO],
    );
    let mut result = TestResultTable::new(2, 1, 1);
    execute_plan(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

    assert_eq!(result.raw.row_count, 2);
    let values = result.metric_values();
    assert!(values[0].is_finite());
    assert!(values[1].is_finite());
    assert!(values[0] >= 0.0);
    assert!(values[0] > values[1]);
}

#[test]
fn cuda_fixed_mi_extreme_bin_mapping_matches_cpu_when_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };

    let rows = 512u64;
    let cols = 2u32;
    let mut wide = Vec::with_capacity(rows as usize);
    let mut subnormal = Vec::with_capacity(rows as usize);
    let mut target = Vec::with_capacity(rows as usize);
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    for row in 0..rows as usize {
        let wide_value = if row % 2 == 0 { -f32::MAX } else { f32::MAX };
        let subnormal_value = f32::from_bits((row % 9) as u32);
        let target_value = (row % 9) as f32;
        wide.push(wide_value);
        subnormal.push(subnormal_value);
        target.push(target_value);
        features.extend([wide_value, subnormal_value]);
    }

    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();
    let plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1],
        vec![GAFIME_METRIC_MUTUAL_INFO],
    );
    let mut result = TestResultTable::new(2, 1, 1);
    execute_plan(&mut backend, matrix.handle(), &plan, result.raw_mut()).unwrap();

    let expected_wide = gafime_cpu::kernels::mutual_info_fixed(&wide, &target, 8);
    let expected_subnormal = gafime_cpu::kernels::mutual_info_fixed(&subnormal, &target, 8);
    let actual = result.metric_values();
    assert_eq!(expected_wide, 0.0);
    assert_eq!(actual[0].to_bits(), expected_wide.to_bits());
    assert!(
        (actual[1] - expected_subnormal).abs() <= 1.0e-5,
        "subnormal MI mismatch: CUDA={}, CPU={expected_subnormal}",
        actual[1]
    );
}

#[test]
fn cuda_all_adaptive_mi_templates_match_cpu_for_arity_1_to_5_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Some(mut cuda_backend) = cuda_backend_for_specialization_test() else {
        return;
    };
    assert_adaptive_mi_templates_match_cpu_for_arity_1_to_5(
        &mut cuda_backend,
        GAFIME_BACKEND_CUDA,
        MI_TEMPLATE_BIN_LEVELS,
    );
}

#[test]
fn cuda_low_signal_mi_matches_cpu_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    assert_low_signal_mi_matches_cpu(&mut backend, GAFIME_BACKEND_CUDA);
}

#[test]
fn cuda_spearman_matches_cpu_when_library_is_available() {
    // Spearman = pearson on ranks; the CUDA count-based ranks must match the
    // CPU rankdata (including average-tie ranks) within fp tolerance.
    let _cuda_guard = cuda_test_lock();
    let Ok(mut cuda_backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };

    let rows = 48u64;
    let cols = 3u32;
    let mut features = Vec::with_capacity(rows as usize * cols as usize);
    let mut target = Vec::with_capacity(rows as usize);
    for r in 0..rows as usize {
        let a = r as f32 * 0.13; // strictly increasing
        let b = ((r * 7) % 17) as f32; // repeated values -> ties
        let c = (rows as usize - r) as f32; // strictly decreasing
        features.extend([a, b, c]);
        target.push(a * a * a); // strictly monotone in feature 0
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

    let cuda_matrix = cuda_backend.alloc_matrix(rows, cols).unwrap();
    cuda_matrix.upload(&features, &target).unwrap();
    let cuda_plan = CompiledPlan::single_chunk(
        GAFIME_BACKEND_CUDA,
        rows,
        cols,
        GAFIME_FAMILY_CONTINUOUS,
        1,
        vec![0, 1, 2],
        vec![GAFIME_METRIC_SPEARMAN],
    );
    let mut cuda_result = TestResultTable::new(3, 1, 1);
    execute_plan(
        &mut cuda_backend,
        cuda_matrix.handle(),
        &cuda_plan,
        cuda_result.raw_mut(),
    )
    .unwrap();

    let cpu_vals = cpu_result.metric_values();
    let cuda_vals = cuda_result.metric_values();
    // feature 0 is strictly monotone in target -> spearman == 1; feature 2 is
    // strictly anti-monotone -> spearman == -1.
    assert!(cpu_vals[0] > 0.999, "cpu spearman(f0)={}", cpu_vals[0]);
    assert!(cpu_vals[2] < -0.999, "cpu spearman(f2)={}", cpu_vals[2]);
    for (i, (&c, &g)) in cpu_vals.iter().zip(cuda_vals).enumerate() {
        assert!(
            (c - g).abs() <= 1.0e-4,
            "spearman mismatch at {i}: cpu={c} cuda={g}"
        );
    }
}
