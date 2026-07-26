use super::*;

fn call_decision_path_score(
    score: GafimeGpuDecisionPathScoreFn,
    matrix: &OwnedGpuMatrix,
    batch: &GafimeDecisionPathScoreBatch,
    result: &mut TestResultTable,
) -> GafimeStatus {
    // SAFETY: every caller keeps the owned matrix, batch backing slices, and
    // TestResultTable buffers live for this synchronous call. Semantic edge
    // cases vary ABI flags and path geometry without invalidating pointers.
    unsafe { score(matrix.handle().raw(), batch, result.raw_mut()) }
}

#[test]
fn decision_path_count_reserves_the_terminal_offset_slot() {
    assert!(validate_decision_path_count(GAFIME_MAX_DECISION_PATH_COUNT as usize).is_ok());
    assert!(matches!(
        validate_decision_path_count(GAFIME_MAX_DECISION_PATH_COUNT as usize + 1),
        Err(GpuSysError::SizeOverflow)
    ));
}

#[test]
fn require_rt_policy_rejects_an_unsupported_payload_in_rust() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
    let matrix = backend.alloc_matrix(2, 1).unwrap();
    let terms = [GafimeDecisionPathTerm {
        feature: 0,
        sign: GAFIME_DECISION_PATH_SIGN_LE,
        threshold: 0.5,
        ..Default::default()
    }];
    let offsets = [0, 1];

    assert!(backend
        .decision_path_membership(matrix.handle(), &terms, &offsets)
        .unwrap()
        .is_none());
    assert!(matches!(
        backend.decision_path_membership_with_policy(
            matrix.handle(),
            &terms,
            &offsets,
            DecisionPathRtPolicy::RequireRt,
        ),
        Err(GpuSysError::BackendStatus {
            operation: "gafime_gpu_decision_path_membership",
            status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
        })
    ));

    let mut functions = complete_test_function_table();
    functions.decision_path_membership = Some(test_decision_path_membership_captures_flags);
    functions.decision_path_score = Some(test_decision_path_score_captures_flags);
    let mut sm_only_backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let sm_only_matrix = sm_only_backend.alloc_matrix(2, 1).unwrap();
    TEST_DECISION_PATH_FLAGS.store(u32::MAX, Ordering::SeqCst);
    assert!(sm_only_backend
        .decision_path_membership(sm_only_matrix.handle(), &terms, &offsets)
        .unwrap()
        .is_some());
    assert_eq!(TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst), 0);
    TEST_DECISION_PATH_FLAGS.store(u32::MAX, Ordering::SeqCst);
    assert!(matches!(
        sm_only_backend.decision_path_membership_with_policy(
            sm_only_matrix.handle(),
            &terms,
            &offsets,
            DecisionPathRtPolicy::RequireRt,
        ),
        Err(GpuSysError::BackendStatus {
            operation: "gafime_gpu_decision_path_membership",
            status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
        })
    ));
    assert_eq!(TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst), u32::MAX);

    let mut result = GafimeResultTable::default();
    assert!(matches!(
        sm_only_backend.decision_path_score_with_policy(
            sm_only_matrix.handle(),
            &terms,
            &offsets,
            &[GAFIME_METRIC_PEARSON],
            &mut result,
            DecisionPathRtPolicy::RequireRt,
        ),
        Err(GpuSysError::BackendStatus {
            operation: "gafime_gpu_decision_path_score",
            status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
        })
    ));
    assert_eq!(TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst), u32::MAX);
}

#[test]
fn rust_decision_path_policy_sets_the_approved_abi_flag() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut functions = complete_test_function_table();
    functions.device_info = Some(test_device_info_with_optix_rt);
    functions.decision_path_membership = Some(test_decision_path_membership_captures_flags);
    functions.decision_path_score = Some(test_decision_path_score_captures_flags);
    let mut backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
    let matrix = backend.alloc_matrix(2, 1).unwrap();
    let terms = [GafimeDecisionPathTerm {
        feature: 0,
        sign: GAFIME_DECISION_PATH_SIGN_GT,
        threshold: 0.5,
        ..Default::default()
    }];
    let offsets = [0, 1];

    TEST_DECISION_PATH_FLAGS.store(u32::MAX, Ordering::SeqCst);
    backend
        .decision_path_membership(matrix.handle(), &terms, &offsets)
        .unwrap();
    assert_eq!(TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst), 0);

    backend
        .decision_path_membership_with_policy(
            matrix.handle(),
            &terms,
            &offsets,
            DecisionPathRtPolicy::RequireRt,
        )
        .unwrap();
    assert_eq!(
        TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst),
        GAFIME_DECISION_PATH_FLAG_REQUIRE_RT
    );

    let mut result = GafimeResultTable::default();
    backend
        .decision_path_score_with_policy(
            matrix.handle(),
            &terms,
            &offsets,
            &[GAFIME_METRIC_PEARSON],
            &mut result,
            DecisionPathRtPolicy::RequireRt,
        )
        .unwrap();
    assert_eq!(
        TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst),
        GAFIME_DECISION_PATH_FLAG_REQUIRE_RT
    );
}

#[test]
fn legacy_cuda_decision_path_payloads_share_a_host_execution_lock() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut legacy_functions = complete_test_function_table();
    legacy_functions.decision_path_membership = Some(test_decision_path_membership_captures_flags);
    legacy_functions.decision_path_score = Some(test_decision_path_score_captures_flags);

    let first = GpuBackend::new(GAFIME_BACKEND_CUDA, legacy_functions).unwrap();
    let second = GpuBackend::new(GAFIME_BACKEND_CUDA, legacy_functions).unwrap();
    let first_lock = first
        .legacy_cuda_decision_path_lock
        .as_ref()
        .expect("pre-lifecycle CUDA payload needs host serialization");
    let second_lock = second
        .legacy_cuda_decision_path_lock
        .as_ref()
        .expect("same payload/device needs the shared host lock");
    assert!(Arc::ptr_eq(first_lock, second_lock));

    let mut current_functions = legacy_functions;
    current_functions.decision_path_release_device_state =
        Some(test_release_decision_path_device_state);
    let current = GpuBackend::new(GAFIME_BACKEND_CUDA, current_functions).unwrap();
    assert!(current.legacy_cuda_decision_path_lock.is_none());

    assert!(
        acquire_legacy_cuda_decision_path_lock(GAFIME_BACKEND_ROCM, 0, &legacy_functions,)
            .is_none()
    );
}

#[test]
fn final_matrix_owner_releases_each_payload_device_state_once() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    TEST_RT_RELEASE_COUNT.store(0, Ordering::SeqCst);
    TEST_RT_RELEASE_DEVICE_MASK.store(0, Ordering::SeqCst);
    let mut functions = complete_test_function_table();
    functions.decision_path_release_device_state = Some(test_release_decision_path_device_state);
    let backend0 =
        GpuBackend::from_function_table(GAFIME_BACKEND_CUDA, 0, functions, None, None).unwrap();
    let backend1 =
        GpuBackend::from_function_table(GAFIME_BACKEND_CUDA, 1, functions, None, None).unwrap();

    let matrix0_first = backend0.alloc_matrix(2, 1).unwrap();
    let matrix0_second = backend0.alloc_matrix(2, 1).unwrap();
    let matrix1 = backend1.alloc_matrix(2, 1).unwrap();
    drop(matrix0_first);
    assert_eq!(TEST_RT_RELEASE_COUNT.load(Ordering::SeqCst), 0);
    drop(matrix1);
    assert_eq!(TEST_RT_RELEASE_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(TEST_RT_RELEASE_DEVICE_MASK.load(Ordering::SeqCst), 0b10);
    drop(matrix0_second);
    assert_eq!(TEST_RT_RELEASE_COUNT.load(Ordering::SeqCst), 2);
    assert_eq!(TEST_RT_RELEASE_DEVICE_MASK.load(Ordering::SeqCst), 0b11);
}

#[test]
fn cuda_decision_path_membership_matches_cpu_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    if !backend.supports_decision_path_membership() {
        return;
    }

    let rows = 5u64;
    let cols = 2u32;
    let features = vec![0.0, 0.0, 0.5, 0.6, 1.0, 1.0, f32::NAN, 1.0, 2.0, f32::NAN];
    let target = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let terms = vec![
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
    ];
    let offsets = vec![0u32, 1, 3];
    let actual = backend
        .decision_path_membership(matrix.handle(), &terms, &offsets)
        .unwrap()
        .expect("CUDA payload should expose decision-path membership");

    let columns = vec![0.0, 0.5, 1.0, f32::NAN, 2.0, 0.0, 0.6, 1.0, 1.0, f32::NAN];
    let expected0 = path_membership(
        &columns,
        rows as usize,
        &[PathNode {
            feature: 0,
            threshold: 0.5,
            sign: SplitSign::Le,
        }],
    );
    let expected1 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected = [expected0, expected1].concat();
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "membership[{idx}] expected NaN, got {a}");
        } else {
            assert_eq!(*a, *e, "membership[{idx}]");
        }
    }
}

#[test]
fn cuda_require_rt_policy_matches_loaded_payload_capability_when_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    if !backend.supports_decision_path_membership() {
        return;
    }
    let has_optix_rt = backend.device_profile().unwrap().optix_rt;
    let matrix = backend.alloc_matrix(4, 1).unwrap();
    matrix
        .upload(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0])
        .unwrap();
    let terms = [GafimeDecisionPathTerm {
        feature: 0,
        sign: GAFIME_DECISION_PATH_SIGN_GT,
        threshold: 1.0,
        ..Default::default()
    }];
    let result = backend.decision_path_membership_with_policy(
        matrix.handle(),
        &terms,
        &[0, 1],
        DecisionPathRtPolicy::RequireRt,
    );
    if has_optix_rt {
        assert_eq!(result.unwrap().unwrap(), [0.0, 0.0, 1.0, 1.0]);
    } else {
        assert!(matches!(
            result,
            Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_membership",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            })
        ));
    }
}

#[test]
fn cuda_rt_same_device_concurrency_is_deterministic_when_available() {
    let _cuda_guard = cuda_test_lock();
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    if !backend.supports_decision_path_membership() {
        return;
    }
    drop(backend);

    const WORKERS: usize = 8;
    const REPEATS: usize = 20;
    let rows = 64u64;
    let cols = 2u32;
    let features = Arc::new(
        (0..rows)
            .flat_map(|row| {
                let x = row as f32 / rows as f32;
                [x, 1.0 - x]
            })
            .collect::<Vec<_>>(),
    );
    let target = Arc::new((0..rows).map(|row| row as f32).collect::<Vec<_>>());
    let expected = Arc::new(
        (0..rows)
            .map(|row| {
                let x = row as f32 / rows as f32;
                if x > 0.25 && 1.0 - x <= 0.75 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>(),
    );
    let barrier = Arc::new(Barrier::new(WORKERS));
    let mut workers = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        let features = Arc::clone(&features);
        let target = Arc::clone(&target);
        let expected = Arc::clone(&expected);
        let barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            let mut backend = GpuBackend::cuda_from_env(0).unwrap();
            let matrix = backend.alloc_matrix(rows, cols).unwrap();
            matrix.upload(&features, &target).unwrap();
            let terms = [
                GafimeDecisionPathTerm {
                    feature: 0,
                    sign: GAFIME_DECISION_PATH_SIGN_GT,
                    threshold: 0.25,
                    ..Default::default()
                },
                GafimeDecisionPathTerm {
                    feature: 1,
                    sign: GAFIME_DECISION_PATH_SIGN_LE,
                    threshold: 0.75,
                    ..Default::default()
                },
            ];
            let offsets = [0u32, 2u32];
            barrier.wait();
            for _ in 0..REPEATS {
                let actual = backend
                    .decision_path_membership_with_policy(
                        matrix.handle(),
                        &terms,
                        &offsets,
                        DecisionPathRtPolicy::RequireRt,
                    )
                    .unwrap()
                    .expect("RT-capable CUDA payload exposes membership");
                assert_eq!(actual, *expected);
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn cuda_rt_explicit_cleanup_rebuilds_state_when_available() {
    let _cuda_guard = cuda_test_lock();
    let Some(mut backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let Some(release_device_state) = backend.functions.decision_path_release_device_state else {
        return;
    };
    let matrix = backend.alloc_matrix(4, 1).unwrap();
    matrix
        .upload(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0])
        .unwrap();
    let terms = [GafimeDecisionPathTerm {
        feature: 0,
        sign: GAFIME_DECISION_PATH_SIGN_GT,
        threshold: 1.0,
        ..Default::default()
    }];
    let offsets = [0u32, 1u32];
    let first = backend
        .decision_path_membership_with_policy(
            matrix.handle(),
            &terms,
            &offsets,
            DecisionPathRtPolicy::RequireRt,
        )
        .unwrap()
        .unwrap();
    // SAFETY: the optional function belongs to the loaded payload, and the
    // backend's validated device id identifies the state being released.
    let status = unsafe { release_device_state(backend.device_id()) };
    status_to_gpu_result("gafime_gpu_decision_path_release_device_state", status).unwrap();
    let rebuilt = backend
        .decision_path_membership_with_policy(
            matrix.handle(),
            &terms,
            &offsets,
            DecisionPathRtPolicy::RequireRt,
        )
        .unwrap()
        .unwrap();
    assert_eq!(rebuilt, first);
}

#[test]
fn cuda_decision_path_score_matches_cpu_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    if !backend.supports_decision_path_score() {
        return;
    }

    let rows = 6u64;
    let cols = 2u32;
    let features = vec![0.0, 0.0, 0.4, 0.2, 0.6, 0.7, 1.0, 0.8, 1.4, 0.1, 2.0, 0.9];
    let target = vec![0.0, 0.2, 1.0, 1.4, 0.3, 2.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let terms = vec![
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.8,
            ..Default::default()
        },
    ];
    let offsets = vec![0u32, 2, 5];
    let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let mut result = TestResultTable::new(2, 1, 2);
    let executed = backend
        .decision_path_score(
            matrix.handle(),
            &terms,
            &offsets,
            &metrics,
            result.raw_mut(),
        )
        .unwrap();
    assert!(executed);
    assert_eq!(result.raw.row_count, 2);
    assert_eq!(result.combo_indices(), &[0, 1]);
    assert_eq!(result.candidate_ids(), &[0, 1]);
    assert_eq!(
        &result.families[..result.raw.row_count as usize],
        &[GAFIME_FAMILY_DECISION_PATH, GAFIME_FAMILY_DECISION_PATH]
    );

    let columns = vec![0.0, 0.4, 0.6, 1.0, 1.4, 2.0, 0.0, 0.2, 0.7, 0.8, 0.1, 0.9];
    let expected0 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected1 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 0,
                threshold: 1.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 1,
                threshold: 0.8,
                sign: SplitSign::Le,
            },
        ],
    );
    let expected0_p = gafime_cpu::kernels::pearson(&expected0, &target);
    let expected1_p = gafime_cpu::kernels::pearson(&expected1, &target);
    let values = result.metric_values();
    assert!((values[0] - expected0_p).abs() < 1.0e-5);
    assert!((values[1] - expected0_p * expected0_p).abs() < 1.0e-5);
    assert!((values[2] - expected1_p).abs() < 1.0e-5);
    assert!((values[3] - expected1_p * expected1_p).abs() < 1.0e-5);
}

#[test]
fn cuda_decision_path_direct_score_matches_cpu_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    if !backend.supports_decision_path_score() {
        return;
    }

    let rows = 6u64;
    let cols = 2u32;
    let features = vec![0.0, 0.0, 0.4, 0.2, 0.6, 0.7, 1.0, 0.8, 1.4, 0.1, 2.0, 0.9];
    let target = vec![0.0, 0.2, 1.0, 1.4, 0.3, 2.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let terms = vec![
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.8,
            ..Default::default()
        },
    ];
    let offsets = vec![0u32, 2, 5];
    let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let mut result = TestResultTable::new(2, 1, 2);
    let executed = backend
        .decision_path_score(
            matrix.handle(),
            &terms,
            &offsets,
            &metrics,
            result.raw_mut(),
        )
        .unwrap();
    assert!(executed);
    assert_eq!(result.raw.row_count, 2);

    let columns = vec![0.0, 0.4, 0.6, 1.0, 1.4, 2.0, 0.0, 0.2, 0.7, 0.8, 0.1, 0.9];
    let expected0 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected1 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 0,
                threshold: 1.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 1,
                threshold: 0.8,
                sign: SplitSign::Le,
            },
        ],
    );
    let expected0_p = gafime_cpu::kernels::pearson(&expected0, &target);
    let expected1_p = gafime_cpu::kernels::pearson(&expected1, &target);
    let values = result.metric_values();
    assert!((values[0] - expected0_p).abs() < 1.0e-4);
    assert!((values[1] - expected0_p * expected0_p).abs() < 1.0e-4);
    assert!((values[2] - expected1_p).abs() < 1.0e-4);
    assert!((values[3] - expected1_p * expected1_p).abs() < 1.0e-4);
}

#[test]
fn cuda_decision_path_direct_score_groups_mixed_axes_when_rt_is_required() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let decision_path_score = backend
        .functions
        .decision_path_score
        .expect("OptiX CUDA payload must expose decision-path scoring");

    let rows = 8u64;
    let cols = 4u32;
    let features = vec![
        0.1, 0.1, 0.9, 0.2, 0.6, 0.7, 0.1, 0.8, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4, 0.4, 1.0, 0.5,
        0.8, 0.9, 0.2, 0.4, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7,
    ];
    let target = vec![0.1, 1.3, 1.1, 0.6, 1.7, 0.2, 1.2, 0.9];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let terms = [
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 2,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 3,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.4,
            ..Default::default()
        },
    ];
    let offsets = [0u32, 2, 4, 6];
    let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let mut result = TestResultTable::new(3, 1, 2);
    let batch = GafimeDecisionPathScoreBatch {
        abi_version: GAFIME_ABI_VERSION,
        path_count: 3,
        term_count: terms.len() as u32,
        flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        terms: terms.as_ptr(),
        path_offsets: offsets.as_ptr(),
        metric_ids: metrics.as_ptr(),
        metric_count: metrics.len() as u32,
        reserved32: 0,
        reserved: [0; 7],
    };
    let status = call_decision_path_score(decision_path_score, &matrix, &batch, &mut result);
    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
    assert_eq!(result.raw.row_count, 3);
    assert_eq!(result.combo_indices(), &[0, 1, 2]);
    assert_eq!(result.candidate_ids(), &[0, 1, 2]);

    let columns = vec![
        0.1, 0.6, 0.8, 0.3, 1.0, 0.2, 0.7, 0.4, 0.1, 0.7, 0.2, 0.9, 0.5, 0.4, 0.8, 0.6, 0.9, 0.1,
        0.7, 0.4, 0.8, 0.2, 0.6, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.3, 0.7,
    ];
    let expected0 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected1 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 2,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 3,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected2 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 1,
                threshold: 0.4,
                sign: SplitSign::Le,
            },
        ],
    );
    let values = result.metric_values();
    let expected = [
        gafime_cpu::kernels::pearson(&expected0, &target),
        gafime_cpu::kernels::pearson(&expected1, &target),
        gafime_cpu::kernels::pearson(&expected2, &target),
    ];
    for (path, pearson) in expected.iter().enumerate() {
        let base = path * 2;
        assert!(
            (values[base] - pearson).abs() < 1.0e-4,
            "path {path} pearson"
        );
        assert!(
            (values[base + 1] - pearson * pearson).abs() < 1.0e-4,
            "path {path} r2"
        );
    }
}

#[test]
fn cuda_decision_path_direct_score_groups_overlapping_pairs_when_rt_is_required() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let decision_path_score = backend
        .functions
        .decision_path_score
        .expect("OptiX CUDA payload must expose decision-path scoring");

    let rows = 8u64;
    let cols = 4u32;
    let features = vec![
        0.1, 0.1, 0.9, 0.2, 0.6, 0.7, 0.1, 0.8, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4, 0.4, 1.0, 0.5,
        0.8, 0.9, 0.2, 0.4, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7,
    ];
    let target = vec![0.1, 1.3, 1.1, 0.6, 1.7, 0.2, 1.2, 0.9];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let terms = [
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 2,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 2,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 3,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
    ];
    let offsets = [0u32, 2, 4, 6];
    let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let batch = GafimeDecisionPathScoreBatch {
        abi_version: GAFIME_ABI_VERSION,
        path_count: 3,
        term_count: terms.len() as u32,
        flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        terms: terms.as_ptr(),
        path_offsets: offsets.as_ptr(),
        metric_ids: metrics.as_ptr(),
        metric_count: metrics.len() as u32,
        reserved32: 0,
        reserved: [0; 7],
    };

    let columns = vec![
        0.1, 0.6, 0.8, 0.3, 1.0, 0.2, 0.7, 0.4, 0.1, 0.7, 0.2, 0.9, 0.5, 0.4, 0.8, 0.6, 0.9, 0.1,
        0.7, 0.4, 0.8, 0.2, 0.6, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.3, 0.7,
    ];
    let expected01 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected12 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 2,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected23 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 2,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 3,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );

    let mut result = TestResultTable::new(3, 1, 2);
    let status = call_decision_path_score(decision_path_score, &matrix, &batch, &mut result);
    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
    assert_eq!(result.raw.row_count, 3);
    let expected = [
        gafime_cpu::kernels::pearson(&expected01, &target),
        gafime_cpu::kernels::pearson(&expected12, &target),
        gafime_cpu::kernels::pearson(&expected23, &target),
    ];
    let values = result.metric_values();
    for (path, pearson) in expected.iter().enumerate() {
        let base = path * 2;
        assert!(
            (values[base] - pearson).abs() < 1.0e-4,
            "path {path} pearson"
        );
        assert!(
            (values[base + 1] - pearson * pearson).abs() < 1.0e-4,
            "path {path} r2"
        );
    }
}

#[test]
fn cuda_decision_path_direct_score_instanced_custom_aabbs_count_once() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let decision_path_score = backend
        .functions
        .decision_path_score
        .expect("OptiX CUDA payload must expose decision-path scoring");

    let rows = 6u64;
    let cols = 4u32;
    let features = vec![
        0.25, 0.25, 1.50, 0.50, 0.50, 0.50, 0.25, 0.25, 0.75, 0.75, 0.50, 0.50, 1.50, 0.50, 0.75,
        0.75, 0.50, 1.50, 1.50, 0.50, -0.10, 0.50, 0.50, 1.50,
    ];
    let target = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let terms = vec![
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 2,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 2,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 3,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 3,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.0,
            ..Default::default()
        },
    ];
    let offsets = [0u32, 4, 8];
    let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let batch = GafimeDecisionPathScoreBatch {
        abi_version: GAFIME_ABI_VERSION,
        path_count: 2,
        term_count: terms.len() as u32,
        flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        terms: terms.as_ptr(),
        path_offsets: offsets.as_ptr(),
        metric_ids: metrics.as_ptr(),
        metric_count: metrics.len() as u32,
        reserved32: 0,
        reserved: [0; 7],
    };

    let columns = vec![
        0.25, 0.50, 0.75, 1.50, 0.50, -0.10, 0.25, 0.50, 0.75, 0.50, 1.50, 0.50, 1.50, 0.25, 0.50,
        0.75, 1.50, 0.50, 0.50, 0.25, 0.50, 0.75, 0.50, 1.50,
    ];
    let expected01 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.0,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 0,
                threshold: 1.0,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 1,
                threshold: 0.0,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 1.0,
                sign: SplitSign::Le,
            },
        ],
    );
    let expected23 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 2,
                threshold: 0.0,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 2,
                threshold: 1.0,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 3,
                threshold: 0.0,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 3,
                threshold: 1.0,
                sign: SplitSign::Le,
            },
        ],
    );
    let expected = [
        gafime_cpu::kernels::pearson(&expected01, &target),
        gafime_cpu::kernels::pearson(&expected23, &target),
    ];

    let mut result = TestResultTable::new(2, 1, 2);
    let status = call_decision_path_score(decision_path_score, &matrix, &batch, &mut result);
    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
    assert_eq!(result.raw.row_count, 2);
    let values = result.metric_values();
    for (path, pearson) in expected.iter().enumerate() {
        let base = path * 2;
        assert!(
            (values[base] - pearson).abs() < 1.0e-4,
            "path {path} pearson: got {}, expected {}",
            values[base],
            pearson
        );
        assert!(
            (values[base + 1] - pearson * pearson).abs() < 1.0e-4,
            "path {path} r2: got {}, expected {}",
            values[base + 1],
            pearson * pearson
        );
    }
}

#[test]
fn cuda_decision_path_firsthit_score_partitioned_groups_match_cpu_when_rt_is_required() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let decision_path_score = backend
        .functions
        .decision_path_score
        .expect("OptiX CUDA payload must expose decision-path scoring");

    let rows = 8u64;
    let cols = 4u32;
    let features = vec![
        0.2, 0.2, 0.2, 0.2, 0.7, 0.2, 0.7, 0.2, 0.2, 0.7, 0.2, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5,
        0.5, 0.5, 0.5001, 0.5, 0.5001, 0.5, 0.5, 0.5001, 0.5, 0.5001, 0.9, 0.1, 0.1, 0.9,
    ];
    let target = vec![0.1, 1.0, 0.4, 1.4, 0.8, 1.2, 0.6, 0.3];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let mut terms = Vec::new();
    let mut offsets = vec![0u32];
    for &(feature0, feature1, sign0, sign1) in &[
        (
            0u32,
            1u32,
            GAFIME_DECISION_PATH_SIGN_LE,
            GAFIME_DECISION_PATH_SIGN_LE,
        ),
        (
            0u32,
            1u32,
            GAFIME_DECISION_PATH_SIGN_GT,
            GAFIME_DECISION_PATH_SIGN_LE,
        ),
        (
            2u32,
            3u32,
            GAFIME_DECISION_PATH_SIGN_LE,
            GAFIME_DECISION_PATH_SIGN_LE,
        ),
        (
            2u32,
            3u32,
            GAFIME_DECISION_PATH_SIGN_GT,
            GAFIME_DECISION_PATH_SIGN_LE,
        ),
    ] {
        terms.push(GafimeDecisionPathTerm {
            feature: feature0,
            sign: sign0,
            threshold: 0.5,
            ..Default::default()
        });
        if sign0 == GAFIME_DECISION_PATH_SIGN_GT {
            terms.push(GafimeDecisionPathTerm {
                feature: feature0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.0,
                ..Default::default()
            });
        } else {
            terms.push(GafimeDecisionPathTerm {
                feature: feature0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: -0.1,
                ..Default::default()
            });
        }
        terms.push(GafimeDecisionPathTerm {
            feature: feature1,
            sign: sign1,
            threshold: 0.5,
            ..Default::default()
        });
        terms.push(GafimeDecisionPathTerm {
            feature: feature1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: -0.1,
            ..Default::default()
        });
        offsets.push(terms.len() as u32);
    }
    let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let batch = GafimeDecisionPathScoreBatch {
        abi_version: GAFIME_ABI_VERSION,
        path_count: 4,
        term_count: terms.len() as u32,
        flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        terms: terms.as_ptr(),
        path_offsets: offsets.as_ptr(),
        metric_ids: metrics.as_ptr(),
        metric_count: metrics.len() as u32,
        reserved32: 0,
        reserved: [0; 7],
    };

    let columns = vec![
        0.2, 0.7, 0.2, 0.7, 0.5, 0.5001, 0.5, 0.9, 0.2, 0.2, 0.7, 0.7, 0.5, 0.5, 0.5001, 0.1, 0.2,
        0.7, 0.2, 0.7, 0.5, 0.5001, 0.5, 0.1, 0.2, 0.2, 0.7, 0.7, 0.5, 0.5, 0.5001, 0.9,
    ];
    let expected_paths = [
        vec![
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 0,
                threshold: -0.1,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 1,
                threshold: -0.1,
                sign: SplitSign::Gt,
            },
        ],
        vec![
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 0,
                threshold: 1.0,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 1,
                threshold: -0.1,
                sign: SplitSign::Gt,
            },
        ],
        vec![
            PathNode {
                feature: 2,
                threshold: 0.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 2,
                threshold: -0.1,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 3,
                threshold: 0.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 3,
                threshold: -0.1,
                sign: SplitSign::Gt,
            },
        ],
        vec![
            PathNode {
                feature: 2,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 2,
                threshold: 1.0,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 3,
                threshold: 0.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 3,
                threshold: -0.1,
                sign: SplitSign::Gt,
            },
        ],
    ];
    let mut result = TestResultTable::new(4, 1, 2);
    let status = call_decision_path_score(decision_path_score, &matrix, &batch, &mut result);
    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
    assert_eq!(result.raw.row_count, 4);
    assert_eq!(result.combo_indices(), &[0, 1, 2, 3]);

    let values = result.metric_values();
    for (path, nodes) in expected_paths.iter().enumerate() {
        let membership = path_membership(&columns, rows as usize, nodes);
        let pearson = gafime_cpu::kernels::pearson(&membership, &target);
        let base = path * 2;
        assert!(
            (values[base] - pearson).abs() < 1.0e-5,
            "path {path} pearson: got {}, expected {}",
            values[base],
            pearson
        );
        assert!(
            (values[base + 1] - pearson * pearson).abs() < 1.0e-5,
            "path {path} r2: got {}, expected {}",
            values[base + 1],
            pearson * pearson
        );
    }
}

#[test]
fn cuda_decision_path_tiny_bounded_regions_respect_rt_numeric_domain() {
    let _cuda_guard = cuda_test_lock();
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let decision_path_score = backend
        .functions
        .decision_path_score
        .expect("OptiX CUDA payload must expose decision-path scoring");
    let rows = 4u64;
    let cols = 2u32;
    let target = vec![1.0, 0.0, 0.0, 0.0];
    let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let offsets = [0u32, 4];

    for (threshold, rt_representable) in [(f32::from_bits(1), false), (f32::MIN_POSITIVE, true)] {
        let features = vec![
            threshold, threshold, 0.0, threshold, threshold, 0.0, 0.0, 0.0,
        ];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();
        let terms = [
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold,
                ..Default::default()
            },
        ];

        {
            let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
            let required_batch = GafimeDecisionPathScoreBatch {
                abi_version: GAFIME_ABI_VERSION,
                path_count: 1,
                term_count: terms.len() as u32,
                flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
                terms: terms.as_ptr(),
                path_offsets: offsets.as_ptr(),
                metric_ids: metrics.as_ptr(),
                metric_count: metrics.len() as u32,
                reserved32: 0,
                reserved: [0; 7],
            };
            let mut result = TestResultTable::new(1, 1, 2);
            let status = call_decision_path_score(
                decision_path_score,
                &matrix,
                &required_batch,
                &mut result,
            );
            if rt_representable {
                status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
                assert_eq!(result.metric_values(), &[1.0, 1.0]);
            } else {
                assert_eq!(status, GAFIME_STATUS_UNSUPPORTED_BACKEND);
            }
        }

        {
            let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
            let fallback_batch = GafimeDecisionPathScoreBatch {
                abi_version: GAFIME_ABI_VERSION,
                path_count: 1,
                term_count: terms.len() as u32,
                flags: 0,
                terms: terms.as_ptr(),
                path_offsets: offsets.as_ptr(),
                metric_ids: metrics.as_ptr(),
                metric_count: metrics.len() as u32,
                reserved32: 0,
                reserved: [0; 7],
            };
            let mut result = TestResultTable::new(1, 1, 2);
            let status = call_decision_path_score(
                decision_path_score,
                &matrix,
                &fallback_batch,
                &mut result,
            );
            status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
            assert_eq!(result.metric_values(), &[1.0, 1.0]);
        }
    }
}

#[test]
fn cuda_decision_path_firsthit_bucket_lattice_covers_narrow_float_boundaries() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let decision_path_score = backend
        .functions
        .decision_path_score
        .expect("OptiX CUDA payload must expose decision-path scoring");
    let cutoff = 2.0_f32.powi(-60);
    for threshold in [
        f32::from_bits(cutoff.to_bits() - 1),
        cutoff,
        f32::from_bits(cutoff.to_bits() + 1),
    ] {
        let above_threshold = f32::from_bits(threshold.to_bits() + 1);
        let min_normal = f32::MIN_POSITIVE;
        let rows = 5u64;
        let cols = 2u32;
        let features = vec![
            min_normal,
            min_normal,
            threshold,
            threshold,
            0.0,
            min_normal,
            min_normal,
            0.0,
            above_threshold,
            above_threshold,
        ];
        let target = vec![1.0, 1.0, 0.0, 0.0, 0.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();
        let terms = [
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold,
                ..Default::default()
            },
        ];
        let offsets = [0u32, terms.len() as u32];
        let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: 1,
            term_count: terms.len() as u32,
            flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            terms: terms.as_ptr(),
            path_offsets: offsets.as_ptr(),
            metric_ids: metrics.as_ptr(),
            metric_count: metrics.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };
        let mut result = TestResultTable::new(1, 1, 2);

        let status = call_decision_path_score(decision_path_score, &matrix, &batch, &mut result);

        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
        assert_eq!(result.metric_values(), &[1.0, 1.0]);
    }
}

#[test]
fn cuda_decision_path_firsthit_score_rejects_overlap_without_sm_fallback() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
    let Ok(backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    let Some(decision_path_score) = backend.functions.decision_path_score else {
        return;
    };

    let rows = 4u64;
    let cols = 2u32;
    let features = vec![0.2, 0.2, 0.5, 0.5, 0.8, 0.8, 0.4, 0.7];
    let target = vec![0.0, 1.0, 2.0, 3.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let terms = vec![
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.75,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.75,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.25,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.0,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.25,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.0,
            ..Default::default()
        },
    ];
    let offsets = [0u32, 4, 8];
    let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let batch = GafimeDecisionPathScoreBatch {
        abi_version: GAFIME_ABI_VERSION,
        path_count: 2,
        term_count: terms.len() as u32,
        flags: 0,
        terms: terms.as_ptr(),
        path_offsets: offsets.as_ptr(),
        metric_ids: metrics.as_ptr(),
        metric_count: metrics.len() as u32,
        reserved32: 0,
        reserved: [0; 7],
    };

    let mut result = TestResultTable::new(2, 1, 2);
    let status = call_decision_path_score(decision_path_score, &matrix, &batch, &mut result);
    assert_eq!(status, gafime_types::GAFIME_STATUS_UNSUPPORTED_BACKEND);
}

#[test]
fn cuda_decision_path_direct_score_recomputes_target_stats_with_cached_points() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let decision_path_score = backend
        .functions
        .decision_path_score
        .expect("OptiX CUDA payload must expose decision-path scoring");

    let rows = 8u64;
    let cols = 4u32;
    let features = vec![
        0.1, 0.1, 0.9, 0.2, 0.6, 0.7, 0.1, 0.8, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4, 0.4, 1.0, 0.5,
        0.8, 0.9, 0.2, 0.4, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7,
    ];
    let target0 = vec![0.1, 1.3, 1.1, 0.6, 1.7, 0.2, 1.2, 0.9];
    let target1 = vec![1.6, 0.1, 0.4, 1.9, 0.3, 1.4, 0.2, 1.1];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target0).unwrap();

    let terms = [
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 2,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 3,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
    ];
    let offsets = [0u32, 2, 4];
    let metrics = [GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
    let batch = GafimeDecisionPathScoreBatch {
        abi_version: GAFIME_ABI_VERSION,
        path_count: 2,
        term_count: terms.len() as u32,
        flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        terms: terms.as_ptr(),
        path_offsets: offsets.as_ptr(),
        metric_ids: metrics.as_ptr(),
        metric_count: metrics.len() as u32,
        reserved32: 0,
        reserved: [0; 7],
    };

    let columns = vec![
        0.1, 0.6, 0.8, 0.3, 1.0, 0.2, 0.7, 0.4, 0.1, 0.7, 0.2, 0.9, 0.5, 0.4, 0.8, 0.6, 0.9, 0.1,
        0.7, 0.4, 0.8, 0.2, 0.6, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.3, 0.7,
    ];
    let expected0 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected1 = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 2,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 3,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );

    let mut result0 = TestResultTable::new(2, 1, 2);
    let status = call_decision_path_score(decision_path_score, &matrix, &batch, &mut result0);
    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
    matrix.update_target(&target1).unwrap();
    let mut result1 = TestResultTable::new(2, 1, 2);
    let status = call_decision_path_score(decision_path_score, &matrix, &batch, &mut result1);
    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();

    let expected_first = [
        gafime_cpu::kernels::pearson(&expected0, &target0),
        gafime_cpu::kernels::pearson(&expected1, &target0),
    ];
    let expected_second = [
        gafime_cpu::kernels::pearson(&expected0, &target1),
        gafime_cpu::kernels::pearson(&expected1, &target1),
    ];
    let values0 = result0.metric_values();
    let values1 = result1.metric_values();
    assert!((values0[0] - expected_first[0]).abs() < 1.0e-4);
    assert!((values0[2] - expected_first[1]).abs() < 1.0e-4);
    assert!((values1[0] - expected_second[0]).abs() < 1.0e-4);
    assert!((values1[2] - expected_second[1]).abs() < 1.0e-4);
    assert!(
        (values0[0] - values1[0]).abs() > 1.0e-3 || (values0[2] - values1[2]).abs() > 1.0e-3,
        "target-only update must change direct RT scores while reusing packed points"
    );
}

#[test]
fn cuda_decision_path_direct_score_refreshes_cached_scatter_map() {
    let _cuda_guard = cuda_test_lock();
    let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
    let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
        return;
    };
    let decision_path_score = backend
        .functions
        .decision_path_score
        .expect("OptiX CUDA payload must expose decision-path scoring");

    let rows = 8u64;
    let cols = 4u32;
    let features = vec![
        0.1, 0.1, 0.9, 0.2, 0.6, 0.7, 0.1, 0.8, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4, 0.4, 1.0, 0.5,
        0.8, 0.9, 0.2, 0.4, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7,
    ];
    let target = vec![0.1, 1.3, 1.1, 0.6, 1.7, 0.2, 1.2, 0.9];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();

    let path01_gt = [
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
    ];
    let path23_gt = [
        GafimeDecisionPathTerm {
            feature: 2,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 3,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        },
    ];
    let path01_le = [
        GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.5,
            ..Default::default()
        },
        GafimeDecisionPathTerm {
            feature: 1,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.4,
            ..Default::default()
        },
    ];
    let terms_first = [path01_gt, path23_gt, path01_le].concat();
    let terms_second = [path01_gt, path01_le, path23_gt].concat();
    let offsets = [0u32, 2, 4, 6];
    let metrics = [GAFIME_METRIC_PEARSON];
    let make_batch = |terms: &[GafimeDecisionPathTerm]| GafimeDecisionPathScoreBatch {
        abi_version: GAFIME_ABI_VERSION,
        path_count: 3,
        term_count: terms.len() as u32,
        flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        terms: terms.as_ptr(),
        path_offsets: offsets.as_ptr(),
        metric_ids: metrics.as_ptr(),
        metric_count: metrics.len() as u32,
        reserved32: 0,
        reserved: [0; 7],
    };

    let columns = vec![
        0.1, 0.6, 0.8, 0.3, 1.0, 0.2, 0.7, 0.4, 0.1, 0.7, 0.2, 0.9, 0.5, 0.4, 0.8, 0.6, 0.9, 0.1,
        0.7, 0.4, 0.8, 0.2, 0.6, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.3, 0.7,
    ];
    let expected01_gt = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected23_gt = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 2,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 3,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ],
    );
    let expected01_le = path_membership(
        &columns,
        rows as usize,
        &[
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Le,
            },
            PathNode {
                feature: 1,
                threshold: 0.4,
                sign: SplitSign::Le,
            },
        ],
    );
    let expected = [
        gafime_cpu::kernels::pearson(&expected01_gt, &target),
        gafime_cpu::kernels::pearson(&expected23_gt, &target),
        gafime_cpu::kernels::pearson(&expected01_le, &target),
    ];

    let mut result_first = TestResultTable::new(3, 1, 1);
    let batch_first = make_batch(&terms_first);
    let status = call_decision_path_score(
        decision_path_score,
        &matrix,
        &batch_first,
        &mut result_first,
    );
    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();

    let mut result_second = TestResultTable::new(3, 1, 1);
    let batch_second = make_batch(&terms_second);
    let status = call_decision_path_score(
        decision_path_score,
        &matrix,
        &batch_second,
        &mut result_second,
    );
    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();

    let first = result_first.metric_values();
    let second = result_second.metric_values();
    assert!((first[0] - expected[0]).abs() < 1.0e-4);
    assert!((first[1] - expected[1]).abs() < 1.0e-4);
    assert!((first[2] - expected[2]).abs() < 1.0e-4);
    assert!((second[0] - expected[0]).abs() < 1.0e-4);
    assert!((second[1] - expected[2]).abs() < 1.0e-4);
    assert!((second[2] - expected[1]).abs() < 1.0e-4);
}

#[test]
fn cuda_decision_path_score_rejects_unsupported_metrics_when_library_is_available() {
    let _cuda_guard = cuda_test_lock();
    let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
        return;
    };
    if !backend.supports_decision_path_score() {
        return;
    }

    let rows = 4u64;
    let cols = 1u32;
    let features = vec![0.0, 0.5, 1.0, 1.5];
    let target = vec![0.0, 1.0, 2.0, 3.0];
    let matrix = backend.alloc_matrix(rows, cols).unwrap();
    matrix.upload(&features, &target).unwrap();
    let terms = vec![GafimeDecisionPathTerm {
        feature: 0,
        sign: GAFIME_DECISION_PATH_SIGN_GT,
        threshold: 0.75,
        ..Default::default()
    }];
    let offsets = vec![0u32, 1];
    let metrics = vec![GAFIME_METRIC_MUTUAL_INFO];
    let mut result = TestResultTable::new(1, 1, 1);
    let err = backend
        .decision_path_score(
            matrix.handle(),
            &terms,
            &offsets,
            &metrics,
            result.raw_mut(),
        )
        .expect_err("MI must be unsupported for compact decision-path score");
    assert!(matches!(
        err,
        GpuSysError::BackendStatus {
            operation: "gafime_gpu_decision_path_score",
            status: gafime_types::GAFIME_STATUS_UNSUPPORTED_BACKEND,
        }
    ));
}
