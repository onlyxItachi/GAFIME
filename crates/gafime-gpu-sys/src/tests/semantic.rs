//! Host-only contract checks for the optional semantic-arithmetic table.
//!
//! These fixtures deliberately never allocate a device bank.  They establish
//! that legacy payloads remain loadable, while a payload that starts exposing
//! this table must expose every required callback before capability negotiation
//! can reach native code.

use super::*;

use std::{
    sync::atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    sync::Arc,
};

use gafime_orchestrator::semantic::{
    CandidateRegistry, EvaluationRole, FeatureFrame, FeatureId, NativeEvidenceExecutor,
    ProgramLimits, SemanticError,
};
use gafime_types::PrecisionProfile;

static CAPABILITY_CALLS: AtomicUsize = AtomicUsize::new(0);
static PAIRWISE_CALLS: AtomicUsize = AtomicUsize::new(0);
static RETAIN_CALLS: AtomicUsize = AtomicUsize::new(0);
static DOWNLOAD_CALLS: AtomicUsize = AtomicUsize::new(0);
static FREE_CALLS: AtomicUsize = AtomicUsize::new(0);
static UPLOAD_CALLS: AtomicUsize = AtomicUsize::new(0);
static RETAIN_RESULT: AtomicI32 = AtomicI32::new(GAFIME_STATUS_OK);
static MATERIALIZE_CALLS: AtomicUsize = AtomicUsize::new(0);
static LAST_MATERIALIZE_NODES: AtomicU32 = AtomicU32::new(0);
static LAST_MATERIALIZE_OPERANDS: AtomicU64 = AtomicU64::new(0);
static LAST_MATERIALIZE_MEANS: AtomicU64 = AtomicU64::new(0);
static GATHER_CALLS: AtomicUsize = AtomicUsize::new(0);
static FORECAST_CALLS: AtomicUsize = AtomicUsize::new(0);
static LAST_FORECAST_MAX_OPERANDS: AtomicU64 = AtomicU64::new(0);
static LAST_FORECAST_OPERANDS: AtomicU64 = AtomicU64::new(0);
static LAST_FORECAST_MEANS: AtomicU64 = AtomicU64::new(0);
static FORECAST_EXTRA_TRANSIENT: AtomicU64 = AtomicU64::new(0);
static FORECAST_MALFORMED: AtomicU32 = AtomicU32::new(0);
static CAPABILITY_FLAGS: AtomicU32 = AtomicU32::new(0);
static MAX_PROGRAM_NODES: AtomicU32 = AtomicU32::new(64);
static MAX_GATHER_ROWS: AtomicU64 = AtomicU64::new(1_024);

struct MockBank {
    rows: u64,
    storage_bytes: u64,
    bytes: u64,
    uploads: AtomicUsize,
}

unsafe extern "C" fn semantic_capabilities(
    device_id: u32,
    consumer_abi_version: u32,
    output: *mut GafimeSemanticCapabilities,
) -> GafimeStatus {
    if output.is_null() || consumer_abi_version != GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    CAPABILITY_CALLS.fetch_add(1, Ordering::SeqCst);
    // SAFETY: null was rejected above and the test fixture writes one fully
    // initialized ABI record into the caller-owned output slot.
    unsafe {
        *output = GafimeSemanticCapabilities {
            backend_kind: GAFIME_BACKEND_CUDA,
            device_id,
            profile_mask: GAFIME_PRECISION_PROFILE_MASK_FP32
                | GAFIME_PRECISION_PROFILE_MASK_MIXED
                | GAFIME_PRECISION_PROFILE_MASK_FP64,
            program_op_mask: GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE
                | GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE
                | GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN
                | GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT,
            primitive_mask: GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_PEARSON
                | GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY
                | GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER,
            association_statistic_mask: GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON,
            flags: CAPABILITY_FLAGS.load(Ordering::SeqCst),
            max_program_nodes: MAX_PROGRAM_NODES.load(Ordering::SeqCst),
            max_slot_count: 64,
            max_rows: 1_024,
            max_gather_rows: MAX_GATHER_ROWS.load(Ordering::SeqCst),
            ..Default::default()
        };
    }
    GAFIME_STATUS_OK
}

unsafe extern "C" fn semantic_bank_alloc(
    _: u32,
    desc: *const GafimeSemanticBankDesc,
    output: *mut GafimeGpuSemanticBank,
) -> GafimeStatus {
    if desc.is_null() || output.is_null() {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: null was rejected above; the fixture reads the descriptor only
    // to preserve its shape metadata in an owned mock bank.
    let desc = unsafe { &*desc };
    let storage_bytes = match desc.rows.checked_mul(u64::from(desc.slot_capacity)) {
        Some(elements) if elements != 0 && desc.bytes % elements == 0 => desc.bytes / elements,
        _ => return GAFIME_STATUS_INVALID_ARGUMENT,
    };
    let bank = Box::new(MockBank {
        rows: desc.rows,
        storage_bytes,
        bytes: desc.bytes,
        uploads: AtomicUsize::new(0),
    });
    // SAFETY: null was rejected above and ownership of this allocation moves
    // to the paired `semantic_bank_free` fixture.
    unsafe { *output = Box::into_raw(bank).cast() };
    GAFIME_STATUS_OK
}

unsafe extern "C" fn semantic_bank_upload(
    bank: GafimeGpuSemanticBank,
    _: *const GafimeNumericRoute,
    _: *const GafimeConstBufferView,
) -> GafimeStatus {
    if bank.is_null() {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: null was rejected above and the safe wrapper serializes every
    // operation on this mock bank.
    let bank = unsafe { &*bank.cast::<MockBank>() };
    if bank.uploads.fetch_add(1, Ordering::SeqCst) != 0 {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    UPLOAD_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

unsafe extern "C" fn semantic_materialize(
    _: GafimeGpuSemanticBank,
    batch: *const GafimeSemanticProgramBatch,
) -> GafimeStatus {
    if batch.is_null() {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: null was rejected above; the fixture snapshots only scalar
    // descriptor lengths while the safe wrapper keeps all arrays live.
    let batch = unsafe { &*batch };
    MATERIALIZE_CALLS.fetch_add(1, Ordering::SeqCst);
    LAST_MATERIALIZE_NODES.store(batch.node_count, Ordering::SeqCst);
    LAST_MATERIALIZE_OPERANDS.store(batch.operand_slots.len, Ordering::SeqCst);
    LAST_MATERIALIZE_MEANS.store(batch.mean_bits.len, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

unsafe fn write_scalar_results(
    output: *mut GafimeSemanticScalarResultTable,
    count: u64,
) -> GafimeStatus {
    if output.is_null() {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: null was rejected above; the safe adapter supplies one writable
    // result record with caller-owned vectors sized to `capacity`.
    let output = unsafe { &mut *output };
    if output.capacity < count
        || (count != 0
            && (output.values.data.is_null()
                || output.states.is_null()
                || output.supports.is_null()))
    {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    let count = match usize::try_from(count) {
        Ok(count) => count,
        Err(_) => return GAFIME_STATUS_INVALID_ARGUMENT,
    };
    for index in 0..count {
        // SAFETY: `capacity >= count` above and the adapter supplied typed
        // result storage with this exact element count.
        unsafe {
            match output.values.dtype {
                GAFIME_DTYPE_F32 => output.values.data.cast::<f32>().add(index).write(0.5),
                GAFIME_DTYPE_F64 => output.values.data.cast::<f64>().add(index).write(0.5),
                _ => return GAFIME_STATUS_INVALID_ARGUMENT,
            }
            output
                .states
                .add(index)
                .write(GAFIME_SEMANTIC_SCALAR_MEASURED);
            output.supports.add(index).write(4);
        }
    }
    output.count = count as u64;
    GAFIME_STATUS_OK
}

unsafe extern "C" fn semantic_pairwise_pearson(
    _: GafimeGpuSemanticBank,
    _: GafimeGpuSemanticBank,
    batch: *const GafimeSemanticPearsonBatch,
    output: *mut GafimeSemanticScalarResultTable,
) -> GafimeStatus {
    if batch.is_null() {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    PAIRWISE_CALLS.fetch_add(1, Ordering::SeqCst);
    // SAFETY: null was rejected above; only the count field is read from the
    // adapter-owned descriptor for this host-only mock.
    let count = unsafe { (*batch).left_slots.len };
    // SAFETY: `write_scalar_results` validates the caller-owned output.
    unsafe { write_scalar_results(output, count) }
}

unsafe extern "C" fn semantic_ordered_edge_energy(
    _: GafimeGpuSemanticBank,
    batch: *const GafimeSemanticEdgeEnergyBatch,
    output: *mut GafimeSemanticScalarResultTable,
) -> GafimeStatus {
    if batch.is_null() {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: null was rejected above; only the candidate count is read.
    let count = unsafe { (*batch).candidate_slots.len };
    // SAFETY: `write_scalar_results` validates the caller-owned output.
    unsafe { write_scalar_results(output, count) }
}

unsafe extern "C" fn semantic_sparse_gather(
    _: GafimeGpuSemanticBank,
    _: GafimeGpuSemanticBank,
    _: *const GafimeSemanticSparseGatherBatch,
) -> GafimeStatus {
    GATHER_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

unsafe extern "C" fn semantic_forecast(
    bank: GafimeGpuSemanticBank,
    request: *const GafimeSemanticForecastRequest,
    output: *mut GafimeSemanticMemoryForecast,
) -> GafimeStatus {
    if bank.is_null() || request.is_null() || output.is_null() {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: null was rejected above; the fixture reads only its owned mock
    // metadata and writes one fully initialized ABI record.
    let (bank, request) = unsafe { (&*bank.cast::<MockBank>(), &*request) };
    FORECAST_CALLS.fetch_add(1, Ordering::SeqCst);
    LAST_FORECAST_MAX_OPERANDS.store(request.program_max_operand_count, Ordering::SeqCst);
    LAST_FORECAST_OPERANDS.store(request.program_operand_count, Ordering::SeqCst);
    LAST_FORECAST_MEANS.store(request.program_mean_count, Ordering::SeqCst);
    let retained_bytes = match bank
        .rows
        .checked_mul(request.retained_slot_count)
        .and_then(|elements| elements.checked_mul(bank.storage_bytes))
    {
        Some(bytes) => bytes,
        None => return GAFIME_STATUS_INVALID_ARGUMENT,
    };
    let descriptor_bytes = match request
        .program_operand_count
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .and_then(|bytes| {
            request
                .program_mean_count
                .checked_mul(std::mem::size_of::<u64>() as u64)
                .and_then(|means| bytes.checked_add(means))
        })
        .and_then(|bytes| {
            bytes.checked_add(if request.program_operand_count == 0 {
                0
            } else {
                std::mem::size_of::<u32>() as u64
            })
        })
        .and_then(|bytes| bytes.checked_add(FORECAST_EXTRA_TRANSIENT.load(Ordering::SeqCst)))
    {
        Some(bytes) => bytes,
        None => return GAFIME_STATUS_INVALID_ARGUMENT,
    };
    let resident_bytes = if FORECAST_MALFORMED.load(Ordering::SeqCst) == 0 {
        bank.bytes
    } else {
        bank.bytes.saturating_add(1)
    };
    // SAFETY: null was rejected above and the fixture writes only its owned
    // output record.
    unsafe {
        *output = GafimeSemanticMemoryForecast {
            resident_bytes,
            transient_bytes: descriptor_bytes,
            retained_bytes,
            ..Default::default()
        }
    };
    GAFIME_STATUS_OK
}

unsafe extern "C" fn semantic_bank_retain(
    source: GafimeGpuSemanticBank,
    slots: GafimeSliceU32,
    output: *mut GafimeGpuSemanticBank,
) -> GafimeStatus {
    if source.is_null() || output.is_null() || slots.len == 0 {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the safe bank wrapper retains this live mock allocation for the
    // whole synchronous call and `output` is a checked writable out-pointer.
    let source = unsafe { &*source.cast::<MockBank>() };
    let bytes = match source
        .rows
        .checked_mul(slots.len)
        .and_then(|elements| elements.checked_mul(source.storage_bytes))
    {
        Some(bytes) => bytes,
        None => return GAFIME_STATUS_INVALID_ARGUMENT,
    };
    let bank = Box::new(MockBank {
        rows: source.rows,
        storage_bytes: source.storage_bytes,
        bytes,
        // Retained slots are already initialized physical values and must not
        // accept a new source-content epoch.
        uploads: AtomicUsize::new(1),
    });
    RETAIN_CALLS.fetch_add(1, Ordering::SeqCst);
    // SAFETY: `output` was checked non-null above and receives the owned mock
    // allocation released by the paired free fixture.
    unsafe { *output = Box::into_raw(bank).cast() };
    RETAIN_RESULT.load(Ordering::SeqCst)
}

unsafe extern "C" fn semantic_bank_download(
    bank: GafimeGpuSemanticBank,
    slots: GafimeSliceU32,
    _: *const GafimeNumericRoute,
    output: *mut GafimeMutableBufferView,
) -> GafimeStatus {
    if bank.is_null() || output.is_null() {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the live mock bank remains owned by the safe wrapper; the
    // output pointer was checked before reading its caller-owned capacity.
    let bank = unsafe { &*bank.cast::<MockBank>() };
    // SAFETY: output is the non-null caller-owned descriptor borrowed
    // exclusively by this synchronous fixture invocation.
    let output = unsafe { &mut *output };
    let elements = match bank.rows.checked_mul(slots.len) {
        Some(value) => value,
        None => return GAFIME_STATUS_INVALID_ARGUMENT,
    };
    if output.element_capacity < elements || (elements != 0 && output.data.is_null()) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    DOWNLOAD_CALLS.fetch_add(1, Ordering::SeqCst);
    GAFIME_STATUS_OK
}

unsafe extern "C" fn semantic_bank_free(bank: GafimeGpuSemanticBank) -> GafimeStatus {
    if !bank.is_null() {
        FREE_CALLS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: every mock handle originates from `Box::into_raw` above and
        // the safe owner calls free once when the final handle is dropped.
        unsafe { drop(Box::from_raw(bank.cast::<MockBank>())) };
    }
    GAFIME_STATUS_OK
}

fn complete_semantic_table() -> GpuFunctionTable {
    let mut functions = complete_test_function_table();
    functions.semantic_capabilities_v1 = Some(semantic_capabilities);
    functions.semantic_bank_alloc_v1 = Some(semantic_bank_alloc);
    functions.semantic_bank_upload_v1 = Some(semantic_bank_upload);
    functions.semantic_materialize_v1 = Some(semantic_materialize);
    functions.semantic_pairwise_pearson_v1 = Some(semantic_pairwise_pearson);
    functions.semantic_ordered_edge_energy_v1 = Some(semantic_ordered_edge_energy);
    functions.semantic_sparse_gather_v1 = Some(semantic_sparse_gather);
    functions.semantic_forecast_v1 = Some(semantic_forecast);
    functions.semantic_bank_retain_v1 = Some(semantic_bank_retain);
    functions.semantic_bank_download_v1 = Some(semantic_bank_download);
    functions.semantic_bank_free_v1 = Some(semantic_bank_free);
    functions
}

#[test]
fn semantic_table_is_optional_for_legacy_and_all_or_nothing_when_present() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    CAPABILITY_CALLS.store(0, Ordering::SeqCst);

    let legacy = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
    assert!(matches!(
        legacy.semantic_capabilities(),
        Err(GpuSysError::SemanticAbiUnavailable)
    ));
    assert_eq!(CAPABILITY_CALLS.load(Ordering::SeqCst), 0);

    let mut partial = complete_semantic_table();
    partial.semantic_sparse_gather_v1 = None;
    let partial = GpuBackend::new(GAFIME_BACKEND_CUDA, partial).unwrap();
    assert!(matches!(
        partial.semantic_capabilities(),
        Err(GpuSysError::MissingFunction(
            "gafime_gpu_semantic_sparse_gather_v1"
        ))
    ));
    assert_eq!(CAPABILITY_CALLS.load(Ordering::SeqCst), 0);

    let complete = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let capabilities = complete.semantic_capabilities().unwrap();
    assert_eq!(capabilities.backend_kind, GAFIME_BACKEND_CUDA);
    assert!(complete
        .supports_semantic_profile(PrecisionProfile::Fp32)
        .unwrap());
    assert!(complete
        .supports_semantic_profile(PrecisionProfile::Mixed)
        .unwrap());
    assert!(complete
        .supports_semantic_profile(PrecisionProfile::Fp64)
        .unwrap());
    let executor = complete.semantic_executor().unwrap();
    assert_eq!(executor.backend_kind(), GAFIME_BACKEND_CUDA);
    assert_eq!(CAPABILITY_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn semantic_capabilities_accept_ignorable_hints_and_reject_required_or_zero_limits() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    CAPABILITY_FLAGS.store(0x8000_0000, Ordering::SeqCst);
    let hinted = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    assert!(hinted.semantic_capabilities().is_ok());

    CAPABILITY_FLAGS.store(0x1, Ordering::SeqCst);
    let required = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    assert!(matches!(
        required.semantic_capabilities(),
        Err(GpuSysError::InvalidInput(
            "GPU payload advertised invalid semantic capabilities"
        ))
    ));

    CAPABILITY_FLAGS.store(0, Ordering::SeqCst);
    MAX_PROGRAM_NODES.store(0, Ordering::SeqCst);
    let missing_node_cap = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    assert!(matches!(
        missing_node_cap.semantic_capabilities(),
        Err(GpuSysError::InvalidInput(
            "GPU payload advertised invalid semantic capabilities"
        ))
    ));

    MAX_PROGRAM_NODES.store(64, Ordering::SeqCst);
    MAX_GATHER_ROWS.store(0, Ordering::SeqCst);
    let missing_gather_cap =
        GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    assert!(matches!(
        missing_gather_cap.semantic_capabilities(),
        Err(GpuSysError::InvalidInput(
            "GPU payload advertised invalid semantic capabilities"
        ))
    ));
    MAX_GATHER_ROWS.store(1_024, Ordering::SeqCst);
}

fn test_semantic_bank(backend: &GpuBackend) -> OwnedSemanticBank {
    let bank = backend
        .allocate_semantic_bank(PrecisionProfile::Fp32, 4, 2, 4)
        .expect("complete fixture table allocates a semantic bank");
    bank.upload_f32(&[
        0.0, 1.0, 2.0, 3.0, // source slot 0
        3.0, 2.0, 1.0, 0.0, // source slot 1
    ])
    .expect("fixture accepts the typed source upload");
    bank
}

fn semantic_fixture_frame() -> Arc<FeatureFrame> {
    Arc::new(
        FeatureFrame::new(
            vec!["left".into(), "right".into(), "anchor".into()],
            "semantic-native-mock-rows".into(),
            vec![11, 12, 13, 14],
            EvaluationRole::Discovery,
            "semantic-native-mock-frame".into(),
            vec![
                vec![0.0, 1.0, 2.0, 3.0],
                vec![3.0, 2.0, 1.0, 0.0],
                vec![1.0, 4.0, 2.0, 5.0],
            ],
        )
        .expect("finite aligned mock frame"),
    )
}

fn descriptor_fixture_registry(
    frame: &FeatureFrame,
) -> (CandidateRegistry, FeatureId, FeatureId, FeatureId) {
    let mut registry = CandidateRegistry::new(
        frame.schema().to_vec(),
        PrecisionProfile::Mixed,
        ProgramLimits::default(),
    )
    .expect("mock registry");
    let left = registry.source(0).expect("left source");
    let right = registry.source(1).expect("right source");
    let anchor = registry.source(2).expect("anchor source");
    let difference = registry
        .abs_difference(left, right)
        .expect("difference program");
    let softened = registry.softsign(difference).expect("softsign program");
    let product = registry
        .centered_product(vec![left, anchor], vec![1.5, 2.5])
        .expect("centered product program");
    (registry, left, softened, product)
}

#[test]
fn semantic_forecast_uses_immutable_flattened_descriptor_totals() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    FORECAST_CALLS.store(0, Ordering::SeqCst);
    MATERIALIZE_CALLS.store(0, Ordering::SeqCst);
    UPLOAD_CALLS.store(0, Ordering::SeqCst);
    LAST_MATERIALIZE_NODES.store(0, Ordering::SeqCst);
    LAST_MATERIALIZE_OPERANDS.store(0, Ordering::SeqCst);
    LAST_MATERIALIZE_MEANS.store(0, Ordering::SeqCst);
    LAST_FORECAST_MAX_OPERANDS.store(0, Ordering::SeqCst);
    LAST_FORECAST_OPERANDS.store(0, Ordering::SeqCst);
    LAST_FORECAST_MEANS.store(0, Ordering::SeqCst);
    FORECAST_EXTRA_TRANSIENT.store(0, Ordering::SeqCst);
    FORECAST_MALFORMED.store(0, Ordering::SeqCst);

    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let mut executor = backend.semantic_executor().unwrap();
    let frame = semantic_fixture_frame();
    let (registry, _, softened, product) = descriptor_fixture_registry(&frame);
    let materialized = executor
        .materialize(&registry, &frame, &[softened, product], None, 1 << 20)
        .expect("forecast admits the bounded mock materialization");

    assert!(materialized.is_resident());
    // Difference (2) + softsign (1) + product (2) produce five immutable
    // operand entries and only the product contributes two frozen means.
    assert_eq!(LAST_MATERIALIZE_NODES.load(Ordering::SeqCst), 3);
    assert_eq!(LAST_MATERIALIZE_OPERANDS.load(Ordering::SeqCst), 5);
    assert_eq!(LAST_MATERIALIZE_MEANS.load(Ordering::SeqCst), 2);
    assert_eq!(LAST_FORECAST_MAX_OPERANDS.load(Ordering::SeqCst), 2);
    assert_eq!(LAST_FORECAST_OPERANDS.load(Ordering::SeqCst), 5);
    assert_eq!(LAST_FORECAST_MEANS.load(Ordering::SeqCst), 2);
    assert_eq!(FORECAST_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(UPLOAD_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(MATERIALIZE_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn semantic_forecast_is_authoritative_for_malformed_and_peak_rejection() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    FORECAST_EXTRA_TRANSIENT.store(0, Ordering::SeqCst);
    FORECAST_MALFORMED.store(1, Ordering::SeqCst);

    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let bank = test_semantic_bank(&backend);
    assert!(matches!(
        bank.forecast(GafimeSemanticForecastRequest::default()),
        Err(GpuSysError::InvalidInput(
            "semantic forecast returned malformed metadata or an inconsistent resident size"
        ))
    ));
    FORECAST_MALFORMED.store(0, Ordering::SeqCst);

    MATERIALIZE_CALLS.store(0, Ordering::SeqCst);
    UPLOAD_CALLS.store(0, Ordering::SeqCst);
    FORECAST_EXTRA_TRANSIENT.store(1 << 20, Ordering::SeqCst);
    let mut executor = backend.semantic_executor().unwrap();
    let frame = semantic_fixture_frame();
    let (registry, _, softened, product) = descriptor_fixture_registry(&frame);
    assert!(matches!(
        executor.materialize(&registry, &frame, &[softened, product], None, 4096),
        Err(SemanticError::Invalid(
            "GPU semantic forecast peak exceeds execution budget"
        ))
    ));
    // Admission happens before source upload or native program dispatch.
    assert_eq!(UPLOAD_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(MATERIALIZE_CALLS.load(Ordering::SeqCst), 0);
    FORECAST_EXTRA_TRANSIENT.store(0, Ordering::SeqCst);
}

#[test]
fn retained_roots_prune_dependencies_and_keep_only_selected_bank_bytes() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    FORECAST_EXTRA_TRANSIENT.store(0, Ordering::SeqCst);
    FORECAST_MALFORMED.store(0, Ordering::SeqCst);
    RETAIN_RESULT.store(GAFIME_STATUS_OK, Ordering::SeqCst);
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let mut executor = backend.semantic_executor().unwrap();
    let frame = semantic_fixture_frame();
    let (registry, source, softened, _) = descriptor_fixture_registry(&frame);

    // A deep accepted root initially needs two sources, AbsDiff, and Softsign:
    // four f32 columns of four rows. Retention carries only the selected atom.
    let deep = executor
        .materialize(&registry, &frame, &[softened], None, 1 << 20)
        .expect("deep source materialization");
    assert_eq!(deep.bytes(), 4 * 4 * std::mem::size_of::<f32>());
    RETAIN_CALLS.store(0, Ordering::SeqCst);
    // 64 old-bank bytes + 16 retained bytes + 8 bytes of two slot vectors.
    assert!(matches!(
        executor.retain(&registry, &frame, &deep, None, &[softened], 87),
        Err(SemanticError::Invalid(
            "GPU retained materialization exceeds actual live byte budget"
        ))
    ));
    assert_eq!(RETAIN_CALLS.load(Ordering::SeqCst), 0);
    let retained_deep = executor
        .retain(&registry, &frame, &deep, None, &[softened], 88)
        .expect("exact all-bank retained admission");
    assert_eq!(retained_deep.bytes(), 4 * std::mem::size_of::<f32>());
    assert_eq!(RETAIN_CALLS.load(Ordering::SeqCst), 1);

    MATERIALIZE_CALLS.store(0, Ordering::SeqCst);
    UPLOAD_CALLS.store(0, Ordering::SeqCst);
    GATHER_CALLS.store(0, Ordering::SeqCst);
    let reused_deep = executor
        .materialize(
            &registry,
            &frame,
            &[softened],
            Some(&retained_deep),
            1 << 20,
        )
        .expect("retained deep root gathers without recomputing dependencies");
    assert_eq!(reused_deep.bytes(), retained_deep.bytes());
    assert_eq!(UPLOAD_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(MATERIALIZE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(GATHER_CALLS.load(Ordering::SeqCst), 1);

    // The same pruning applies to an accepted source: no fresh upload is
    // permitted merely because a source normally starts a dependency walk.
    let source_values = executor
        .materialize(&registry, &frame, &[source], None, 1 << 20)
        .expect("source materialization");
    let retained_source = executor
        .retain(&registry, &frame, &source_values, None, &[source], 40)
        .expect("single source retention");
    assert_eq!(retained_source.bytes(), 4 * std::mem::size_of::<f32>());
    MATERIALIZE_CALLS.store(0, Ordering::SeqCst);
    UPLOAD_CALLS.store(0, Ordering::SeqCst);
    GATHER_CALLS.store(0, Ordering::SeqCst);
    let reused_source = executor
        .materialize(
            &registry,
            &frame,
            &[source],
            Some(&retained_source),
            1 << 20,
        )
        .expect("retained source gathers without upload");
    assert_eq!(reused_source.bytes(), retained_source.bytes());
    assert_eq!(UPLOAD_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(MATERIALIZE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(GATHER_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn semantic_bank_peer_validation_locks_once_and_rejects_foreign_payloads() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    PAIRWISE_CALLS.store(0, Ordering::SeqCst);

    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let left = test_semantic_bank(&backend);
    let right = test_semantic_bank(&backend);

    // Reference evidence commonly compares two slots of the same bank.  This
    // must acquire one mutex rather than attempting to lock the same bank
    // twice.
    let same_bank = left
        .pairwise_pearson(&left, &[0], &[1], crate::SemanticPearsonMode::Absolute)
        .expect("same-bank Pearson is safe");
    assert_eq!(same_bank.len(), 1);

    // Both serial orderings exercise the stable peer-lock ordering used by
    // concurrent reversed calls, without relying on a timing-sensitive test.
    left.pairwise_pearson(&right, &[0], &[1], crate::SemanticPearsonMode::Signed)
        .expect("left-to-right peer Pearson is safe");
    right
        .pairwise_pearson(&left, &[1], &[0], crate::SemanticPearsonMode::Signed)
        .expect("right-to-left peer Pearson is safe");
    assert_eq!(PAIRWISE_CALLS.load(Ordering::SeqCst), 3);

    let foreign = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let foreign_bank = test_semantic_bank(&foreign);
    assert!(matches!(
        left.pairwise_pearson(
            &foreign_bank,
            &[0],
            &[0],
            crate::SemanticPearsonMode::Signed,
        ),
        Err(GpuSysError::InvalidInput(_))
    ));
    // The payload callback is unreachable for foreign bank identity.
    assert_eq!(PAIRWISE_CALLS.load(Ordering::SeqCst), 3);
}

#[test]
fn semantic_safe_wrappers_reject_caps_and_same_bank_gather_before_native_dispatch() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    MATERIALIZE_CALLS.store(0, Ordering::SeqCst);
    GATHER_CALLS.store(0, Ordering::SeqCst);
    MAX_PROGRAM_NODES.store(1, Ordering::SeqCst);
    MAX_GATHER_ROWS.store(1_024, Ordering::SeqCst);

    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let bank = test_semantic_bank(&backend);
    assert!(matches!(
        bank.materialize(&[
            crate::SemanticProgramNode::AbsoluteDifference {
                output_slot: 2,
                left_slot: 0,
                right_slot: 1,
            },
            crate::SemanticProgramNode::Softsign {
                output_slot: 3,
                input_slot: 2,
            },
        ]),
        Err(GpuSysError::InvalidInput(_))
    ));
    assert_eq!(MATERIALIZE_CALLS.load(Ordering::SeqCst), 0);

    assert!(matches!(
        bank.sparse_gather_from(&bank, &[0], &[2], &[0, 1, 2, 3]),
        Err(GpuSysError::InvalidInput(_))
    ));
    assert_eq!(GATHER_CALLS.load(Ordering::SeqCst), 0);

    MAX_PROGRAM_NODES.store(64, Ordering::SeqCst);
    MAX_GATHER_ROWS.store(3, Ordering::SeqCst);
    let limited = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let source = test_semantic_bank(&limited);
    let destination = limited
        .allocate_semantic_bank(PrecisionProfile::Fp32, 4, 0, 4)
        .expect("fixture allocates a distinct destination bank");
    assert!(matches!(
        destination.sparse_gather_from(&source, &[0], &[2], &[0, 1, 2, 3]),
        Err(GpuSysError::InvalidInput(_))
    ));
    assert_eq!(GATHER_CALLS.load(Ordering::SeqCst), 0);
    MAX_GATHER_ROWS.store(1_024, Ordering::SeqCst);
}

#[test]
fn semantic_bank_source_upload_has_one_content_epoch() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let bank = test_semantic_bank(&backend);

    assert!(matches!(
        bank.upload_f32(&[
            3.0, 2.0, 1.0, 0.0, // attempted replacement source slot 0
            0.0, 1.0, 2.0, 3.0, // attempted replacement source slot 1
        ]),
        Err(GpuSysError::BackendStatus {
            operation: "gafime_gpu_semantic_bank_upload_v1",
            status: GAFIME_STATUS_INVALID_ARGUMENT,
        })
    ));
}

#[test]
fn semantic_retain_uses_native_copy_without_host_download() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    RETAIN_CALLS.store(0, Ordering::SeqCst);
    DOWNLOAD_CALLS.store(0, Ordering::SeqCst);
    RETAIN_RESULT.store(GAFIME_STATUS_OK, Ordering::SeqCst);

    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let bank = test_semantic_bank(&backend);
    let retained = bank.retain(&[1]).expect("fixture retains one slot");

    assert_eq!(retained.rows(), bank.rows());
    assert_eq!(retained.source_slots(), 1);
    assert_eq!(retained.slot_capacity(), 1);
    assert_eq!(RETAIN_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(DOWNLOAD_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn semantic_retain_adopts_nonnull_failure_output_for_raii_cleanup() {
    let _guard = ABI_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    RETAIN_CALLS.store(0, Ordering::SeqCst);
    FREE_CALLS.store(0, Ordering::SeqCst);
    RETAIN_RESULT.store(GAFIME_STATUS_DEVICE_ERROR, Ordering::SeqCst);

    let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_semantic_table()).unwrap();
    let bank = test_semantic_bank(&backend);
    let result = bank.retain(&[1]);

    assert!(matches!(
        result,
        Err(GpuSysError::BackendStatus {
            operation: "gafime_gpu_semantic_bank_retain_v1",
            status: GAFIME_STATUS_DEVICE_ERROR,
        })
    ));
    assert_eq!(RETAIN_CALLS.load(Ordering::SeqCst), 1);
    // The callback supplied a non-null error output.  The safe wrapper must
    // take temporary RAII ownership before returning the error, so it remains
    // reachable for its best-effort free attempt rather than leaking.
    assert_eq!(FREE_CALLS.load(Ordering::SeqCst), 1);
    RETAIN_RESULT.store(GAFIME_STATUS_OK, Ordering::SeqCst);
}
