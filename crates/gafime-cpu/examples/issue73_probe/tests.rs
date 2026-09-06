use super::probe::{
    run, Candidate, EvidenceValue, GraphEdge, LabeledRows, MatrixSide, ProbeBackend, ProbeError,
    ProbeInput, ProbePrecision, Selector, SparseGraph, UnavailableReason,
};

fn fixture(labels: Option<LabeledRows>, graph: Option<SparseGraph>) -> ProbeInput {
    ProbeInput::new(
        vec![
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 2.0],
        ],
        vec![
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 2.0],
        ],
        1,
        graph,
        labels,
    )
    .expect("fixture is valid")
}

fn value(evidence: &EvidenceValue) -> f64 {
    match evidence {
        EvidenceValue::Value(value) => *value,
        EvidenceValue::Unavailable(reason) => panic!("expected value, got {reason:?}"),
    }
}

fn scalar_mixed_oracle(left: &[f32], right: &[f32]) -> f64 {
    let n = left.len();
    let mean_left = left.iter().map(|&value| f64::from(value)).sum::<f64>() / n as f64;
    let mean_right = right.iter().map(|&value| f64::from(value)).sum::<f64>() / n as f64;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    let mut covariance = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let centered_left = f64::from(left) - mean_left;
        let centered_right = f64::from(right) - mean_right;
        left_variance += centered_left * centered_left;
        right_variance += centered_right * centered_right;
        covariance += centered_left * centered_right;
    }
    covariance / (left_variance * right_variance).sqrt()
}

#[test]
fn unlabeled_input_runs_without_a_target_adapter() {
    let report = run(&fixture(None, None), Selector::core_mixed()).unwrap();
    assert_eq!(report.rows.len(), 3);
    let EvidenceValue::Unavailable(reason) = &report.rows[0].hybrid_labeled_pearson else {
        panic!("unlabeled input must not create a hybrid score");
    };
    assert_eq!(*reason, UnavailableReason::NotRequested);
    assert_eq!(reason.as_str(), "not_requested");
    assert!(value(&report.rows[0].redundancy_abs_pearson) > 0.0);
}

#[test]
fn identity_view_is_one_and_a_perturbed_view_changes_it() {
    let identity_report = run(&fixture(None, None), Selector::core_mixed()).unwrap();
    for row in &identity_report.rows {
        assert!((value(&row.paired_view_consistency) - 1.0).abs() <= 1.0e-12);
    }
    let perturbed = ProbeInput::new(
        vec![
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 2.0],
        ],
        vec![
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 8.0],
            vec![1.0, 0.0, 2.0, 1.0, 3.0, 2.0],
        ],
        1,
        None,
        None,
    )
    .unwrap();
    let perturbed_report = run(&perturbed, Selector::core_mixed()).unwrap();
    assert!(value(&perturbed_report.rows[0].paired_view_consistency) < 0.999);
}

#[test]
fn graph_energy_matches_the_declared_normalized_formula() {
    let graph = SparseGraph::new(vec![GraphEdge::new(0, 1, 1.0), GraphEdge::new(1, 2, 1.0)]);
    let input = ProbeInput::new(
        vec![vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 0.0]],
        vec![vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 0.0]],
        1,
        Some(graph),
        None,
    )
    .unwrap();
    let report = run(&input, Selector::core_mixed()).unwrap();
    assert!((value(&report.rows[0].graph_normalized_dirichlet) - 1.0 / 9.0).abs() < 1.0e-15);
}

#[test]
fn graph_restricted_zero_norm_is_not_mislabeled_as_a_constant_candidate() {
    let input = ProbeInput::new(
        vec![vec![0.0, 0.0, 5.0], vec![1.0, 2.0, 3.0]],
        vec![vec![0.0, 0.0, 5.0], vec![1.0, 2.0, 3.0]],
        1,
        Some(SparseGraph::new(vec![GraphEdge::new(0, 1, 1.0)])),
        None,
    )
    .unwrap();
    let report = run(&input, Selector::core_mixed()).unwrap();
    assert!(matches!(
        report.rows[0].graph_normalized_dirichlet,
        EvidenceValue::Unavailable(UnavailableReason::ZeroGraphEnergy)
    ));
}

#[test]
fn constants_are_unavailable_instead_of_best_scores() {
    let graph = SparseGraph::new(vec![GraphEdge::new(0, 1, 1.0)]);
    let input = ProbeInput::new(
        vec![vec![2.0, 2.0, 2.0], vec![0.0, 1.0, 2.0]],
        vec![vec![2.0, 2.0, 2.0], vec![0.0, 1.0, 2.0]],
        1,
        Some(graph),
        None,
    )
    .unwrap();
    let report = run(&input, Selector::core_mixed()).unwrap();
    let identity = &report.rows[0];
    assert!(matches!(
        identity.redundancy_abs_pearson,
        EvidenceValue::Unavailable(UnavailableReason::ConstantCandidate)
    ));
    assert!(matches!(
        identity.graph_normalized_dirichlet,
        EvidenceValue::Unavailable(UnavailableReason::ConstantCandidate)
    ));
}

#[test]
fn labels_use_only_validated_unique_membership() {
    let labels = LabeledRows::new(vec![0, 2, 4], vec![0.0, 2.0, 4.0]);
    let report = run(&fixture(Some(labels), None), Selector::core_mixed()).unwrap();
    assert!((value(&report.rows[0].hybrid_labeled_pearson) - 1.0).abs() <= 1.0e-12);

    let duplicate = ProbeInput::new(
        vec![vec![0.0, 1.0, 2.0], vec![2.0, 1.0, 0.0]],
        vec![vec![0.0, 1.0, 2.0], vec![2.0, 1.0, 0.0]],
        0,
        None,
        Some(LabeledRows::new(vec![0, 0], vec![1.0, 2.0])),
    );
    assert!(matches!(
        duplicate,
        Err(ProbeError::DuplicateLabelRow { label: 1, row: 0 })
    ));
    let out_of_bounds = ProbeInput::new(
        vec![vec![0.0, 1.0, 2.0], vec![2.0, 1.0, 0.0]],
        vec![vec![0.0, 1.0, 2.0], vec![2.0, 1.0, 0.0]],
        0,
        None,
        Some(LabeledRows::new(vec![0, 3], vec![1.0, 2.0])),
    );
    assert!(matches!(
        out_of_bounds,
        Err(ProbeError::LabelIndexOutOfBounds {
            label: 1,
            row: 3,
            rows: 3
        })
    ));
    let mismatched = ProbeInput::new(
        vec![vec![0.0, 1.0, 2.0], vec![2.0, 1.0, 0.0]],
        vec![vec![0.0, 1.0, 2.0], vec![2.0, 1.0, 0.0]],
        0,
        None,
        Some(LabeledRows::new(vec![0, 1], vec![1.0])),
    );
    assert!(matches!(
        mismatched,
        Err(ProbeError::LabelShape { rows: 2, values: 1 })
    ));
}

#[test]
fn malformed_shapes_and_selectors_fail_closed() {
    let malformed = ProbeInput::new(
        vec![vec![0.0, 1.0], vec![1.0]],
        vec![vec![0.0, 1.0], vec![1.0, 2.0]],
        0,
        None,
        None,
    );
    assert!(matches!(
        malformed,
        Err(ProbeError::ColumnShape {
            side: MatrixSide::Original,
            column: 1,
            expected: 2,
            found: 1
        })
    ));
    let invalid_weight = ProbeInput::new(
        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        0,
        Some(SparseGraph::new(vec![GraphEdge::new(0, 1, 0.0)])),
        None,
    );
    assert!(matches!(
        invalid_weight,
        Err(ProbeError::GraphWeightInvalid { edge: 0 })
    ));
    let input = fixture(None, None);
    for backend in [
        ProbeBackend::Cuda,
        ProbeBackend::Rocm,
        ProbeBackend::Metal,
        ProbeBackend::Auto,
    ] {
        assert_eq!(
            run(&input, Selector::new(backend, ProbePrecision::Mixed)).unwrap_err(),
            ProbeError::UnsupportedBackend(backend)
        );
    }
    for precision in [ProbePrecision::Fp32, ProbePrecision::Fp64] {
        assert_eq!(
            run(&input, Selector::new(ProbeBackend::Core, precision)).unwrap_err(),
            ProbeError::UnsupportedPrecision(precision)
        );
    }
}

#[test]
fn empty_nonfinite_and_graph_bounds_are_rejected() {
    assert!(matches!(
        ProbeInput::new(vec![], vec![], 0, None, None),
        Err(ProbeError::EmptyColumns)
    ));
    let nonfinite = ProbeInput::new(
        vec![vec![0.0, f32::NAN], vec![1.0, 2.0]],
        vec![vec![0.0, 1.0], vec![1.0, 2.0]],
        0,
        None,
        None,
    );
    assert!(matches!(
        nonfinite,
        Err(ProbeError::NonFiniteInput {
            side: MatrixSide::Original,
            column: 0,
            row: 1
        })
    ));
    let endpoint = ProbeInput::new(
        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        0,
        Some(SparseGraph::new(vec![GraphEdge::new(0, 2, 1.0)])),
        None,
    );
    assert!(matches!(
        endpoint,
        Err(ProbeError::GraphEndpointOutOfBounds { edge: 0, rows: 2 })
    ));
    let empty_graph = ProbeInput::new(
        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        0,
        Some(SparseGraph::new(vec![])),
        None,
    );
    assert!(matches!(empty_graph, Err(ProbeError::EmptyGraph)));
}

#[test]
fn centered_native_mixed_result_matches_a_row_ordered_f64_oracle() {
    let original = (0..73)
        .map(|row| (row as f32 * 0.071).sin() + row as f32 * 0.003)
        .collect::<Vec<_>>();
    let aligned = (0..73)
        .map(|row| (row as f32 * 0.053).cos() - row as f32 * 0.002)
        .collect::<Vec<_>>();
    let expected = scalar_mixed_oracle(&original, &aligned);
    let input = ProbeInput::new(
        vec![original.clone(), (0..73).map(|row| row as f32).collect()],
        vec![aligned.clone(), (0..73).map(|row| row as f32).collect()],
        1,
        None,
        None,
    )
    .unwrap();
    let report = run(&input, Selector::core_mixed()).unwrap();
    assert!((value(&report.rows[0].paired_view_consistency) - expected).abs() <= 1.0e-12);
}

#[test]
fn rayon_worker_counts_preserve_exact_rows_and_order() {
    let input = fixture(
        Some(LabeledRows::new(vec![0, 2, 4, 5], vec![0.0, 2.0, 4.0, 5.0])),
        Some(SparseGraph::new(vec![
            GraphEdge::new(0, 1, 1.0),
            GraphEdge::new(1, 2, 2.0),
            GraphEdge::new(2, 3, 1.0),
        ])),
    );
    let one_worker = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run(&input, Selector::core_mixed()).unwrap());
    let four_workers = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| run(&input, Selector::core_mixed()).unwrap());
    assert_eq!(one_worker, four_workers);
    assert_eq!(
        one_worker
            .rows
            .iter()
            .map(|row| row.candidate)
            .collect::<Vec<_>>(),
        vec!["identity_col0", "abs_difference_col0_col1", "softsign_col0"]
    );
}

#[test]
fn fixed_catalog_materializes_f32_forms_before_statistics() {
    let columns = vec![vec![-2.0, 0.0, 3.0], vec![1.0, -4.0, 1.0]];
    let mut buffer = Vec::new();
    let catalog = Candidate::catalog();
    catalog[0].materialize_into(&columns, &mut buffer);
    assert_eq!(buffer, vec![-2.0, 0.0, 3.0]);
    catalog[1].materialize_into(&columns, &mut buffer);
    assert_eq!(buffer, vec![3.0, 4.0, 2.0]);
    catalog[2].materialize_into(&columns, &mut buffer);
    assert_eq!(buffer, vec![-2.0 / 3.0, 0.0, 3.0 / 4.0]);
}

#[test]
fn late_abs_difference_overflow_is_unavailable_not_filtered_evidence() {
    let mut left = (0..20).map(|row| row as f32).collect::<Vec<_>>();
    let mut right = (0..20).map(|row| row as f32 * 0.5).collect::<Vec<_>>();
    left[19] = f32::MAX;
    right[19] = -f32::MAX;
    let input = ProbeInput::new(
        vec![left.clone(), right.clone()],
        vec![left, right],
        0,
        None,
        None,
    )
    .unwrap();
    let report = run(&input, Selector::core_mixed()).unwrap();
    let difference = &report.rows[1];
    assert!(matches!(
        difference.redundancy_abs_pearson,
        EvidenceValue::Unavailable(UnavailableReason::CandidateNonFinite)
    ));
    assert!(matches!(
        difference.paired_view_consistency,
        EvidenceValue::Unavailable(UnavailableReason::CandidateNonFinite)
    ));
}
