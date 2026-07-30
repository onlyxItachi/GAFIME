use std::{env, ffi::OsString, ptr, sync::Mutex};

use gafime_cpu::decision_path::{path_membership, PathNode, SplitSign};
use gafime_gpu_sys::{
    DecisionPathRtPolicy, GpuBackend, GpuSysError, OwnedGpuMatrix, CUDA_LIBRARY_ENV,
};
use gafime_types::{
    GafimeDecisionPathTerm, GafimeResultTable, GAFIME_ABI_VERSION, GAFIME_DECISION_PATH_SIGN_GT,
    GAFIME_DECISION_PATH_SIGN_LE, GAFIME_FAMILY_DECISION_PATH, GAFIME_METRIC_PEARSON,
    GAFIME_METRIC_R2, GAFIME_STATUS_UNSUPPORTED_BACKEND,
};

const FIRSTHIT_TOLERANCE: f32 = 1.0e-4;
const SCORE_SENTINEL: f32 = 12_345.5;

static RT_TEST_LOCK: Mutex<()> = Mutex::new(());

struct EnvOverride {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvOverride {
    fn set(key: &'static str, value: &'static str) -> Self {
        let previous = env::var_os(key);
        env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvOverride {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => env::set_var(self.key, value),
            None => env::remove_var(self.key),
        }
    }
}

struct ResultBuffer {
    raw: GafimeResultTable,
    combo_indices: Vec<u32>,
    metric_values: Vec<f32>,
    ranks: Vec<u32>,
    families: Vec<u32>,
    candidate_ids: Vec<u64>,
    row_flags: Vec<u32>,
}

impl ResultBuffer {
    fn new(capacity: usize, metric_count: usize) -> Self {
        let mut buffer = Self {
            raw: GafimeResultTable {
                abi_version: GAFIME_ABI_VERSION,
                max_arity: 1,
                metric_count: metric_count as u32,
                flags: 0,
                capacity: capacity as u64,
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
            combo_indices: vec![u32::MAX; capacity],
            metric_values: vec![SCORE_SENTINEL; capacity * metric_count],
            ranks: vec![u32::MAX; capacity],
            families: vec![u32::MAX; capacity],
            candidate_ids: vec![u64::MAX; capacity],
            row_flags: vec![u32::MAX; capacity],
        };
        buffer.rebind();
        buffer
    }

    fn raw_mut(&mut self) -> &mut GafimeResultTable {
        self.rebind();
        &mut self.raw
    }

    fn reset(&mut self) {
        self.raw.flags = 0;
        self.raw.row_count = 0;
        self.combo_indices.fill(u32::MAX);
        self.metric_values.fill(SCORE_SENTINEL);
        self.ranks.fill(u32::MAX);
        self.families.fill(u32::MAX);
        self.candidate_ids.fill(u64::MAX);
        self.row_flags.fill(u32::MAX);
        self.rebind();
    }

    fn written_metrics(&self) -> &[f32] {
        let len = self.raw.row_count as usize * self.raw.metric_count as usize;
        &self.metric_values[..len]
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

fn configured_cuda_backend(test_name: &str) -> Option<GpuBackend> {
    let Some(path) = env::var_os(CUDA_LIBRARY_ENV) else {
        eprintln!("skipping {test_name}: {CUDA_LIBRARY_ENV} is not configured");
        return None;
    };
    let backend = GpuBackend::cuda_from_env(0).unwrap_or_else(|error| {
        panic!(
            "configured CUDA payload {} failed to load: {error}",
            std::path::Path::new(&path).display()
        )
    });
    Some(backend)
}

fn configured_optix_backend(test_name: &str) -> Option<GpuBackend> {
    let backend = configured_cuda_backend(test_name)?;
    let profile = backend
        .device_profile()
        .unwrap_or_else(|error| panic!("configured CUDA payload device query failed: {error}"));
    if !profile.optix_rt {
        eprintln!("skipping {test_name}: configured CUDA payload has no OptiX RT capability");
        return None;
    }
    assert!(
        backend.supports_decision_path_membership(),
        "OptiX payload must expose decision-path membership"
    );
    assert!(
        backend.supports_decision_path_score(),
        "OptiX payload must expose compact decision-path scoring"
    );
    Some(backend)
}

fn term(feature: u32, sign: u32, threshold: f32) -> GafimeDecisionPathTerm {
    GafimeDecisionPathTerm {
        feature,
        sign,
        threshold,
        ..Default::default()
    }
}

fn bounded_box(x_lo: f32, x_hi: f32, y_lo: f32, y_hi: f32) -> Vec<GafimeDecisionPathTerm> {
    bounded_pair(0, x_lo, x_hi, 1, y_lo, y_hi)
}

fn bounded_pair(
    first_feature: u32,
    first_lo: f32,
    first_hi: f32,
    second_feature: u32,
    second_lo: f32,
    second_hi: f32,
) -> Vec<GafimeDecisionPathTerm> {
    vec![
        term(first_feature, GAFIME_DECISION_PATH_SIGN_GT, first_lo),
        term(first_feature, GAFIME_DECISION_PATH_SIGN_LE, first_hi),
        term(second_feature, GAFIME_DECISION_PATH_SIGN_GT, second_lo),
        term(second_feature, GAFIME_DECISION_PATH_SIGN_LE, second_hi),
    ]
}

fn bounded_box_3d(
    x_lo: f32,
    x_hi: f32,
    y_lo: f32,
    y_hi: f32,
    z_lo: f32,
    z_hi: f32,
) -> Vec<GafimeDecisionPathTerm> {
    let mut terms = bounded_box(x_lo, x_hi, y_lo, y_hi);
    terms.extend([
        term(2, GAFIME_DECISION_PATH_SIGN_GT, z_lo),
        term(2, GAFIME_DECISION_PATH_SIGN_LE, z_hi),
    ]);
    terms
}

fn flatten_paths(paths: &[Vec<GafimeDecisionPathTerm>]) -> (Vec<GafimeDecisionPathTerm>, Vec<u32>) {
    let mut terms = Vec::new();
    let mut offsets = Vec::with_capacity(paths.len() + 1);
    offsets.push(0);
    for path in paths {
        assert!(
            !path.is_empty(),
            "decision paths must contain at least one term"
        );
        terms.extend_from_slice(path);
        offsets.push(u32::try_from(terms.len()).expect("test term count fits the ABI"));
    }
    (terms, offsets)
}

fn cpu_memberships(
    features: &[f32],
    rows: usize,
    cols: usize,
    paths: &[Vec<GafimeDecisionPathTerm>],
) -> Vec<f32> {
    assert_eq!(features.len(), rows * cols);
    let mut columns = vec![0.0; features.len()];
    for row in 0..rows {
        for col in 0..cols {
            columns[col * rows + row] = features[row * cols + col];
        }
    }

    let mut expected = Vec::with_capacity(rows * paths.len());
    for path in paths {
        let nodes = path
            .iter()
            .map(|term| PathNode {
                feature: term.feature,
                threshold: term.threshold,
                sign: match term.sign {
                    GAFIME_DECISION_PATH_SIGN_LE => SplitSign::Le,
                    GAFIME_DECISION_PATH_SIGN_GT => SplitSign::Gt,
                    other => panic!("unsupported test split sign {other}"),
                },
            })
            .collect::<Vec<_>>();
        expected.extend(path_membership(&columns, rows, &nodes));
    }
    expected
}

fn explicit_mask(rows: usize, member_rows: &[usize]) -> Vec<f32> {
    let mut mask = vec![0.0; rows];
    for &row in member_rows {
        mask[row] = 1.0;
    }
    mask
}

fn assert_memberships(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} output length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "{label}[{index}] expected NaN, got {actual}"
            );
        } else {
            assert_eq!(actual, expected, "{label}[{index}]");
        }
    }
}

fn gpu_memberships(
    backend: &mut GpuBackend,
    matrix: &OwnedGpuMatrix,
    paths: &[Vec<GafimeDecisionPathTerm>],
    policy: DecisionPathRtPolicy,
) -> Result<Vec<f32>, GpuSysError> {
    let (terms, offsets) = flatten_paths(paths);
    backend
        .decision_path_membership_with_policy(matrix.handle(), &terms, &offsets, policy)
        .map(|membership| membership.expect("configured OptiX payload exposes membership"))
}

fn assert_unsupported(error: GpuSysError, operation: &'static str) {
    match error {
        GpuSysError::BackendStatus {
            operation: actual_operation,
            status,
        } => {
            assert_eq!(actual_operation, operation);
            assert_eq!(status, GAFIME_STATUS_UNSUPPORTED_BACKEND);
        }
        other => panic!("expected unsupported {operation}, got {other}"),
    }
}

fn execute_firsthit_score(
    backend: &mut GpuBackend,
    matrix: &OwnedGpuMatrix,
    paths: &[Vec<GafimeDecisionPathTerm>],
    result: &mut ResultBuffer,
) -> Result<(), GpuSysError> {
    execute_score_with_policy(
        backend,
        matrix,
        paths,
        result,
        DecisionPathRtPolicy::RequireRt,
    )
}

fn execute_score_with_policy(
    backend: &mut GpuBackend,
    matrix: &OwnedGpuMatrix,
    paths: &[Vec<GafimeDecisionPathTerm>],
    result: &mut ResultBuffer,
    policy: DecisionPathRtPolicy,
) -> Result<(), GpuSysError> {
    let (terms, offsets) = flatten_paths(paths);
    let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let executed = backend.decision_path_score_with_policy(
        matrix.handle(),
        &terms,
        &offsets,
        &metrics,
        result.raw_mut(),
        policy,
    )?;
    assert!(
        executed,
        "configured OptiX payload must execute compact scoring"
    );
    Ok(())
}

fn assert_firsthit_scores(
    label: &str,
    result: &ResultBuffer,
    expected_memberships: &[f32],
    rows: usize,
    target: &[f32],
    path_count: usize,
) {
    assert_eq!(result.raw.row_count, path_count as u64, "{label} row count");
    assert_eq!(result.raw.metric_count, 2, "{label} metric count");
    assert_eq!(
        &result.combo_indices[..path_count],
        &(0..path_count as u32).collect::<Vec<_>>(),
        "{label} compact combo indices"
    );
    assert_eq!(
        &result.candidate_ids[..path_count],
        &(0..path_count as u64).collect::<Vec<_>>(),
        "{label} compact candidate ids"
    );
    assert_eq!(
        &result.families[..path_count],
        &vec![GAFIME_FAMILY_DECISION_PATH; path_count],
        "{label} result families"
    );

    let mut expected_scores = Vec::with_capacity(path_count * 2);
    for membership in expected_memberships.chunks_exact(rows) {
        let pearson = gafime_cpu::kernels::pearson(membership, target);
        assert!(
            pearson.is_finite(),
            "{label} CPU Pearson oracle must be finite"
        );
        expected_scores.extend([pearson, pearson * pearson]);
    }
    assert_eq!(expected_scores.len(), path_count * 2);
    for (index, (&actual, &expected)) in result
        .written_metrics()
        .iter()
        .zip(&expected_scores)
        .enumerate()
    {
        let difference = (actual - expected).abs();
        assert!(
            difference <= FIRSTHIT_TOLERANCE,
            "{label} metric[{index}] mismatch: actual={actual}, expected={expected}, difference={difference}"
        );
    }
}

fn next_up_positive(value: f32) -> f32 {
    assert!(value.is_finite() && value > 0.0);
    f32::from_bits(value.to_bits() + 1)
}

fn next_down_positive(value: f32) -> f32 {
    assert!(value.is_finite() && value > 0.0);
    f32::from_bits(value.to_bits() - 1)
}

#[test]
fn rt_bucket_lattice_narrow_spans_and_aspect_ratios_match_cpu() {
    let _lock = RT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) =
        configured_optix_backend("rt_bucket_lattice_narrow_spans_and_aspect_ratios")
    else {
        return;
    };

    let cutoff = 2.0_f32.powi(-60);
    let below = next_down_positive(cutoff);
    let above = next_up_positive(cutoff);
    let rows = 9usize;
    let features = vec![
        0.0, 0.0, below, below, cutoff, cutoff, above, above, cutoff, 1.0, 1.0, cutoff, 1.0, 1.0,
        below, 1.0, 1.0, below,
    ];
    let target = (0..rows).map(|row| row as f32).collect::<Vec<_>>();
    let matrix = backend.alloc_matrix(rows as u64, 2).unwrap();
    matrix.upload(&features, &target).unwrap();

    for (label, path, member_rows) in [
        (
            "below-former-cutoff square",
            bounded_box(0.0, below, 0.0, below),
            vec![1],
        ),
        (
            "cutoff square",
            bounded_box(0.0, cutoff, 0.0, cutoff),
            vec![1, 2],
        ),
        (
            "above-cutoff square",
            bounded_box(0.0, above, 0.0, above),
            vec![1, 2, 3],
        ),
        (
            "2^-60 by 1 aspect ratio",
            bounded_box(0.0, cutoff, 0.0, 1.0),
            vec![1, 2, 4, 7],
        ),
        (
            "1 by 2^-60 aspect ratio",
            bounded_box(0.0, 1.0, 0.0, cutoff),
            vec![1, 2, 5, 8],
        ),
    ] {
        let paths = vec![path];
        let expected = cpu_memberships(&features, rows, 2, &paths);
        assert_eq!(
            expected,
            explicit_mask(rows, &member_rows),
            "{label} oracle"
        );
        let actual = gpu_memberships(
            &mut backend,
            &matrix,
            &paths,
            DecisionPathRtPolicy::RequireRt,
        )
        .unwrap();
        assert_memberships(label, &actual, &expected);
    }
}

#[test]
fn rt_three_dimensional_custom_aabbs_match_cpu_membership_and_direct_score() {
    let _lock = RT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = configured_optix_backend("rt_three_dimensional_custom_aabbs") else {
        return;
    };
    let _score_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");

    let rows = 8usize;
    let features = vec![
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 0.5, 0.5, 0.5, -0.0, 0.5, 0.5, 2.0, 2.0, 2.0,
        1.0, 2.0, 2.0, 1.5, 1.5, 1.5,
    ];
    let target = vec![0.0, 1.0, -1.0, 3.0, 0.5, 2.0, -2.0, 4.0];
    let paths = vec![
        bounded_box_3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        bounded_box_3d(1.0, 2.0, 1.0, 2.0, 1.0, 2.0),
    ];
    let expected = cpu_memberships(&features, rows, 3, &paths);
    assert_eq!(
        expected,
        [explicit_mask(rows, &[1, 3]), explicit_mask(rows, &[5, 7])].concat()
    );

    let matrix = backend.alloc_matrix(rows as u64, 3).unwrap();
    matrix.upload(&features, &target).unwrap();
    let actual = gpu_memberships(
        &mut backend,
        &matrix,
        &paths,
        DecisionPathRtPolicy::RequireRt,
    )
    .unwrap();
    assert_memberships("3D required-RT membership", &actual, &expected);

    let mut result = ResultBuffer::new(paths.len(), 2);
    execute_firsthit_score(&mut backend, &matrix, &paths, &mut result).unwrap();
    assert_firsthit_scores(
        "3D direct RT score",
        &result,
        &expected,
        rows,
        &target,
        paths.len(),
    );
}

#[test]
fn rt_compact_scores_are_stable_for_large_target_offsets() {
    let _lock = RT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = configured_optix_backend("rt_compact_score_large_target_offset") else {
        return;
    };

    let rows = 4usize;
    let features = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let target = vec![100_000_000.0, 100_000_008.0, 100_000_016.0, 100_000_024.0];
    let paths = vec![bounded_box(0.5, 1.0, 0.5, 1.0)];
    let expected = cpu_memberships(&features, rows, 2, &paths);
    assert_eq!(expected, explicit_mask(rows, &[2, 3]));

    let matrix = backend.alloc_matrix(rows as u64, 2).unwrap();
    matrix.upload(&features, &target).unwrap();
    let mut result = ResultBuffer::new(1, 2);

    for mode in ["bitset", "direct", "firsthit"] {
        let _score_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", mode);
        result.reset();
        execute_firsthit_score(&mut backend, &matrix, &paths, &mut result).unwrap();
        assert_firsthit_scores(
            &format!("large-offset {mode} score"),
            &result,
            &expected,
            rows,
            &target,
            1,
        );
    }
}

#[test]
fn cuda_sm_compact_scores_are_stable_for_large_target_offsets() {
    let _lock = RT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = configured_cuda_backend("cuda_sm_compact_score_large_target_offset")
    else {
        return;
    };
    if !backend.supports_decision_path_score() {
        eprintln!(
            "skipping cuda_sm_compact_score_large_target_offset: configured CUDA \
             payload has no local RT/decision-path surface"
        );
        return;
    }
    let _rt_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT", "off");
    let _score_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "bitset");

    let rows = 4usize;
    let features = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let target = vec![100_000_000.0, 100_000_008.0, 100_000_016.0, 100_000_024.0];
    let paths = vec![bounded_box(0.5, 1.0, 0.5, 1.0)];
    let expected = cpu_memberships(&features, rows, 2, &paths);
    assert_eq!(expected, explicit_mask(rows, &[2, 3]));

    let matrix = backend.alloc_matrix(rows as u64, 2).unwrap();
    matrix.upload(&features, &target).unwrap();
    let mut result = ResultBuffer::new(1, 2);
    execute_score_with_policy(
        &mut backend,
        &matrix,
        &paths,
        &mut result,
        DecisionPathRtPolicy::AllowSmFallback,
    )
    .unwrap();
    assert_firsthit_scores(
        "large-offset forced-SM score",
        &result,
        &expected,
        rows,
        &target,
        1,
    );
}

#[test]
fn rt_three_axis_pair_batch_uses_exact_pair_groups() {
    let _lock = RT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = configured_optix_backend("rt_three_axis_pair_groups") else {
        return;
    };

    let rows = 6usize;
    let features = vec![
        0.0, 0.0, 0.0, 0.5, 0.5, 2.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 1.5, 1.5,
    ];
    let target = vec![-2.0, 0.0, 1.0, 4.0, 7.0, 3.0];
    let paths = vec![
        bounded_pair(0, 0.0, 1.0, 1, 0.0, 1.0),
        bounded_pair(1, 1.0, 2.0, 2, 1.0, 2.0),
    ];
    let expected = cpu_memberships(&features, rows, 3, &paths);
    assert_eq!(
        expected,
        [explicit_mask(rows, &[1, 2]), explicit_mask(rows, &[3, 5])].concat()
    );

    let matrix = backend.alloc_matrix(rows as u64, 3).unwrap();
    matrix.upload(&features, &target).unwrap();
    let mut result = ResultBuffer::new(paths.len(), 2);

    for mode in ["direct", "firsthit"] {
        let _score_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", mode);
        result.reset();
        execute_firsthit_score(&mut backend, &matrix, &paths, &mut result).unwrap();
        assert_firsthit_scores(
            &format!("three-axis exact-pair {mode} score"),
            &result,
            &expected,
            rows,
            &target,
            paths.len(),
        );
    }
}

#[test]
fn rt_normal_matrix_executes_and_subnormal_matrix_fails_closed() {
    let _lock = RT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = configured_optix_backend("rt_normal_and_subnormal_matrix_domain")
    else {
        return;
    };
    let _score_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");

    let rows = 5usize;
    let min_normal = f32::MIN_POSITIVE;
    let max_subnormal = f32::from_bits(min_normal.to_bits() - 1);
    let normal_features = vec![
        0.0, 0.0, min_normal, min_normal, 1.0, 1.0, 2.0, 0.5, 0.5, 2.0,
    ];
    let subnormal_features = vec![
        0.0,
        0.0,
        max_subnormal,
        max_subnormal,
        1.0,
        1.0,
        2.0,
        0.5,
        0.5,
        2.0,
    ];
    let target = vec![0.0, 1.0, 3.0, 2.0, 4.0];
    let paths = vec![bounded_box(0.0, 1.0, 0.0, 1.0)];

    let normal_matrix = backend.alloc_matrix(rows as u64, 2).unwrap();
    normal_matrix.upload(&normal_features, &target).unwrap();
    let normal_expected = cpu_memberships(&normal_features, rows, 2, &paths);
    assert_eq!(normal_expected, explicit_mask(rows, &[1, 2]));
    let normal_actual = gpu_memberships(
        &mut backend,
        &normal_matrix,
        &paths,
        DecisionPathRtPolicy::RequireRt,
    )
    .unwrap();
    assert_memberships(
        "smallest-normal RT membership",
        &normal_actual,
        &normal_expected,
    );

    let mut normal_score = ResultBuffer::new(1, 2);
    execute_firsthit_score(&mut backend, &normal_matrix, &paths, &mut normal_score).unwrap();
    assert_firsthit_scores(
        "smallest-normal firsthit score",
        &normal_score,
        &normal_expected,
        rows,
        &target,
        1,
    );

    let subnormal_matrix = backend.alloc_matrix(rows as u64, 2).unwrap();
    subnormal_matrix
        .upload(&subnormal_features, &target)
        .unwrap();
    let subnormal_expected = cpu_memberships(&subnormal_features, rows, 2, &paths);
    assert_eq!(subnormal_expected, explicit_mask(rows, &[1, 2]));
    let membership_error = gpu_memberships(
        &mut backend,
        &subnormal_matrix,
        &paths,
        DecisionPathRtPolicy::RequireRt,
    )
    .expect_err("subnormal matrix data must be rejected by required RT traversal");
    assert_unsupported(membership_error, "gafime_gpu_decision_path_membership");

    let fallback = gpu_memberships(
        &mut backend,
        &subnormal_matrix,
        &paths,
        DecisionPathRtPolicy::AllowSmFallback,
    )
    .unwrap();
    assert_memberships("subnormal SM fallback", &fallback, &subnormal_expected);

    let mut rejected_score = ResultBuffer::new(1, 2);
    let score_error =
        execute_firsthit_score(&mut backend, &subnormal_matrix, &paths, &mut rejected_score)
            .expect_err("subnormal matrix data must be rejected by required firsthit RT scoring");
    assert_unsupported(score_error, "gafime_gpu_decision_path_score");
    assert_eq!(rejected_score.raw.row_count, 0);
    assert!(
        rejected_score
            .metric_values
            .iter()
            .all(|&value| value == SCORE_SENTINEL),
        "rejected subnormal score must not write compact output"
    );

    let _bitset_score_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "bitset");
    rejected_score.reset();
    execute_score_with_policy(
        &mut backend,
        &subnormal_matrix,
        &paths,
        &mut rejected_score,
        DecisionPathRtPolicy::AllowSmFallback,
    )
    .unwrap();
    assert_firsthit_scores(
        "subnormal compact-score SM fallback",
        &rejected_score,
        &subnormal_expected,
        rows,
        &target,
        1,
    );

    let mut nonfinite_features = subnormal_features.clone();
    nonfinite_features[0] = f32::NAN;
    let nonfinite_matrix = backend.alloc_matrix(rows as u64, 2).unwrap();
    nonfinite_matrix
        .upload(&nonfinite_features, &target)
        .unwrap();
    rejected_score.reset();
    let optional_score_error = execute_score_with_policy(
        &mut backend,
        &nonfinite_matrix,
        &paths,
        &mut rejected_score,
        DecisionPathRtPolicy::AllowSmFallback,
    )
    .expect_err("compact score must reject missing-feature matrices before fallback");
    assert_unsupported(optional_score_error, "gafime_gpu_decision_path_score");
    assert_eq!(rejected_score.raw.row_count, 0);
    assert!(
        rejected_score
            .metric_values
            .iter()
            .all(|&value| value == SCORE_SENTINEL),
        "rejected optional score must not write compact output"
    );
}

#[test]
fn rt_translated_ulp_zero_crossing_and_adjacent_boxes_match_cpu_and_firsthit() {
    let _lock = RT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = configured_optix_backend("rt_translated_and_adjacent_domain") else {
        return;
    };

    let cutoff = 2.0_f32.powi(-60);
    let cutoff_above = next_up_positive(cutoff);
    let one_hi = next_up_positive(1.0);
    let one_after = next_up_positive(one_hi);
    let large = 2.0_f32.powi(100);
    let large_hi = next_up_positive(large);
    let large_after = next_up_positive(large_hi);
    let three_after = next_up_positive(3.0);
    let rows = 18usize;
    let features = vec![
        1.0,
        one_hi,
        one_hi,
        one_hi,
        one_after,
        one_hi,
        large,
        large_hi,
        large_hi,
        large_hi,
        large_after,
        large_hi,
        -cutoff,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        cutoff,
        cutoff,
        cutoff_above,
        0.0,
        0.0,
        2.5,
        1.0,
        2.5,
        one_hi,
        2.5,
        2.0,
        2.5,
        1.0,
        2.0,
        1.0,
        3.0,
        1.0,
        three_after,
    ];
    let target = vec![
        0.2, 3.0, 1.1, 0.7, 4.5, 2.2, -0.4, 1.5, 2.7, 3.3, -1.0, 0.9, 5.0, -2.0, 4.0, 1.2, 6.0, 0.1,
    ];
    let paths = vec![
        bounded_box(1.0, one_hi, 1.0, one_hi),
        bounded_box(large, large_hi, large, large_hi),
        bounded_box(-cutoff, cutoff, -cutoff, cutoff),
        bounded_box(0.0, 1.0, 2.0, 3.0),
        bounded_box(1.0, 2.0, 2.0, 3.0),
    ];
    let matrix = backend.alloc_matrix(rows as u64, 2).unwrap();
    matrix.upload(&features, &target).unwrap();

    let expected = cpu_memberships(&features, rows, 2, &paths);
    let explicit = [
        explicit_mask(rows, &[1]),
        explicit_mask(rows, &[4]),
        explicit_mask(rows, &[7, 8, 9]),
        explicit_mask(rows, &[12, 16]),
        explicit_mask(rows, &[13, 14]),
    ]
    .concat();
    assert_eq!(
        expected, explicit,
        "CPU oracle exercises every numeric domain"
    );

    let actual = gpu_memberships(
        &mut backend,
        &matrix,
        &paths,
        DecisionPathRtPolicy::RequireRt,
    )
    .unwrap();
    assert_memberships("translated/open-closed RT membership", &actual, &expected);

    let _score_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
    let mut result = ResultBuffer::new(paths.len(), 2);
    execute_firsthit_score(&mut backend, &matrix, &paths, &mut result).unwrap();
    assert_firsthit_scores(
        "translated/open-closed firsthit score",
        &result,
        &expected,
        rows,
        &target,
        paths.len(),
    );
}

#[test]
fn rt_narrow_plan_rebuilds_cached_gas_and_compact_output() {
    let _lock = RT_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut backend) = configured_optix_backend("rt_narrow_plan_rebuilds_cached_gas") else {
        return;
    };
    let _score_mode = EnvOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");

    let cutoff = 2.0_f32.powi(-60);
    let below = next_down_positive(cutoff);
    let above = next_up_positive(cutoff);
    let rows = 6usize;
    let features = vec![
        0.0, 0.5, below, 0.5, cutoff, 0.5, above, 0.5, cutoff, 1.0, 1.0, 0.5,
    ];
    let target = vec![0.0, 1.0, 3.0, 10.0, 5.0, -2.0];
    let matrix = backend.alloc_matrix(rows as u64, 2).unwrap();
    matrix.upload(&features, &target).unwrap();

    let safe_paths = vec![bounded_box(0.0, cutoff, 0.0, 1.0)];
    let narrow_paths = vec![bounded_box(0.0, below, 0.0, 1.0)];
    let alternate_safe_paths = vec![bounded_box(0.0, above, 0.0, 1.0)];
    let safe_expected = cpu_memberships(&features, rows, 2, &safe_paths);
    let narrow_expected = cpu_memberships(&features, rows, 2, &narrow_paths);
    let alternate_expected = cpu_memberships(&features, rows, 2, &alternate_safe_paths);
    assert_eq!(safe_expected, explicit_mask(rows, &[1, 2, 4]));
    assert_eq!(narrow_expected, explicit_mask(rows, &[1]));
    assert_eq!(alternate_expected, explicit_mask(rows, &[1, 2, 3, 4]));

    let safe_membership = gpu_memberships(
        &mut backend,
        &matrix,
        &safe_paths,
        DecisionPathRtPolicy::RequireRt,
    )
    .unwrap();
    assert_memberships("safe membership", &safe_membership, &safe_expected);
    let narrow_membership = gpu_memberships(
        &mut backend,
        &matrix,
        &narrow_paths,
        DecisionPathRtPolicy::RequireRt,
    )
    .unwrap();
    assert_memberships(
        "narrow membership after safe GAS",
        &narrow_membership,
        &narrow_expected,
    );

    let mut result = ResultBuffer::new(1, 2);
    execute_firsthit_score(&mut backend, &matrix, &safe_paths, &mut result).unwrap();
    assert_firsthit_scores(
        "baseline score before narrow GAS",
        &result,
        &safe_expected,
        rows,
        &target,
        1,
    );
    let safe_score = result.written_metrics().to_vec();

    result.reset();
    execute_firsthit_score(&mut backend, &matrix, &narrow_paths, &mut result).unwrap();
    assert_firsthit_scores(
        "narrow score after safe GAS",
        &result,
        &narrow_expected,
        rows,
        &target,
        1,
    );
    let narrow_score = result.written_metrics().to_vec();
    assert!(
        (narrow_score[0] - safe_score[0]).abs() > 1.0e-3,
        "narrow geometry must not reuse the first safe score"
    );

    result.reset();
    execute_firsthit_score(&mut backend, &matrix, &alternate_safe_paths, &mut result).unwrap();
    assert_firsthit_scores(
        "alternate score after narrow GAS",
        &result,
        &alternate_expected,
        rows,
        &target,
        1,
    );
    assert!(
        (result.written_metrics()[0] - narrow_score[0]).abs() > 1.0e-3,
        "alternate geometry must not reuse the narrow score"
    );
}
