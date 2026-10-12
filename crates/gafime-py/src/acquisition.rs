//! File acquisition is transport, not a second execution policy. Foreign Arrow
//! owners are released while attached to Python; only validated, owned numeric
//! storage crosses into the existing planner and resident artifact lifecycle.

use std::{ffi::CStr, sync::Arc};

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
        Int64Array, Int8Array, StructArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    },
    datatypes::{DataType, Schema, SchemaRef},
    error::ArrowError,
    ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema},
    ffi_stream::FFI_ArrowArrayStream,
    record_batch::{RecordBatch, RecordBatchReader},
};
use blake2::{
    digest::{Update, VariableOutput},
    Blake2bVar,
};
use gafime_types::PrecisionProfile;
use pyo3::{
    exceptions::{PyNotImplementedError, PyOverflowError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBytes, PyCapsule, PyDict, PyString},
};

use crate::{
    artifact::{compile_continuous_input, PyCompiledContinuousArtifact},
    common::{validate_shape, OwnedNumericInput},
    generated::{compile_decision_path_input, compile_time_series_input},
    runtime::{get_u32, parse_engine_config},
};

/// Private, one-use transport owner. It exposes metadata and fingerprints, never
/// borrowed writable buffers. A cache key and the compiled matrix therefore
/// cannot accidentally refer to two different acquisitions of mutable input.
#[pyclass(name = "_AcquiredNumericInput", unsendable)]
pub(crate) struct PyAcquiredNumericInput {
    input: Option<OwnedNumericInput>,
    #[pyo3(get)]
    pub(crate) rows: u64,
    #[pyo3(get)]
    pub(crate) cols: u32,
    profile: PrecisionProfile,
    feature_digest: [u8; 16],
    target_digest: [u8; 16],
}

impl PyAcquiredNumericInput {
    fn new(
        profile: PrecisionProfile,
        rows: u64,
        cols: u32,
        input: OwnedNumericInput,
    ) -> PyResult<Self> {
        validate_shape(rows, cols, input.feature_len(), input.target_len())?;
        let (feature_digest, target_digest) = match &input {
            OwnedNumericInput::F32 { features, target } => {
                (digest_f32(profile, features), digest_f32(profile, target))
            }
            OwnedNumericInput::F64 { features, target } => {
                (digest_f64(profile, features), digest_f64(profile, target))
            }
        };
        Ok(Self {
            input: Some(input),
            rows,
            cols,
            profile,
            feature_digest,
            target_digest,
        })
    }

    pub(crate) fn take_input(&mut self, profile: PrecisionProfile) -> PyResult<OwnedNumericInput> {
        if profile != self.profile {
            return Err(PyValueError::new_err(
                "acquired input precision does not match the execution configuration",
            ));
        }
        self.input
            .take()
            .ok_or_else(|| PyValueError::new_err("acquired input has already been consumed"))
    }
}

#[pymethods]
impl PyAcquiredNumericInput {
    #[getter]
    fn precision(&self) -> &'static str {
        profile_name(self.profile)
    }

    #[getter]
    fn feature_digest<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.feature_digest)
    }

    #[getter]
    fn target_digest<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.target_digest)
    }
}

fn profile_name(profile: PrecisionProfile) -> &'static str {
    match profile {
        PrecisionProfile::Fp32 => "fp32",
        PrecisionProfile::Mixed => "mixed",
        PrecisionProfile::Fp64 => "fp64",
    }
}

fn digest_start(profile: PrecisionProfile, count: usize) -> Blake2bVar {
    let mut digest = Blake2bVar::new(16).expect("fixed valid BLAKE2b output size");
    digest.update(profile_name(profile).as_bytes());
    digest.update(&[0]);
    digest.update(&(count as u64).to_le_bytes());
    digest
}

fn digest_finish(digest: Blake2bVar) -> [u8; 16] {
    let mut output = [0; 16];
    digest
        .finalize_variable(&mut output)
        .expect("fixed valid BLAKE2b output size");
    output
}

fn digest_f32(profile: PrecisionProfile, values: &[f32]) -> [u8; 16] {
    let mut digest = digest_start(profile, values.len());
    // Canonical little-endian transport identity, matching the existing Python
    // cache exactly, without constructing a second full-size byte snapshot.
    let mut block = [0_u8; 4096];
    for chunk in values.chunks(block.len() / 4) {
        for (value, bytes) in chunk.iter().zip(block.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        digest.update(&block[..chunk.len() * 4]);
    }
    digest_finish(digest)
}

fn digest_f64(profile: PrecisionProfile, values: &[f64]) -> [u8; 16] {
    let mut digest = digest_start(profile, values.len());
    let mut block = [0_u8; 4096];
    for chunk in values.chunks(block.len() / 8) {
        for (value, bytes) in chunk.iter().zip(block.chunks_exact_mut(8)) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        digest.update(&block[..chunk.len() * 8]);
    }
    digest_finish(digest)
}

/// The pinned arrow-rs stream iterator unwraps the optional error description.
/// Normalize that one C Stream edge here, while keeping schema/array importing
/// and their independent release ownership in arrow-rs. The producer must honor
/// the C Stream valid-pointer/layout/lifetime contract; callbacks stay attached
/// and serialized, and no foreign storage is exposed as a Rust-owned borrow.
pub(crate) struct AcquisitionStreamReader {
    stream: FFI_ArrowArrayStream,
    schema: SchemaRef,
}

impl AcquisitionStreamReader {
    fn try_new(mut stream: FFI_ArrowArrayStream) -> PyResult<Self> {
        if stream.release.is_none()
            || stream.get_schema.is_none()
            || stream.get_next.is_none()
            || stream.get_last_error.is_none()
        {
            return Err(PyValueError::new_err(
                "Arrow stream is released or missing required callbacks",
            ));
        }
        let mut schema = FFI_ArrowSchema::empty();
        // SAFETY: the consumed capsule supplies a live, initialized C Stream;
        // callbacks were checked above. The initialized output schema is owned
        // by arrow-rs and drops on every outcome, independently of the stream.
        let code = unsafe { stream.get_schema.unwrap()(&mut stream, &mut schema) };
        if code != 0 {
            return Err(PyValueError::new_err(format!(
                "arrow stream import failed: {}",
                stream_error(&mut stream, code, "get_schema")
            )));
        }
        let schema = Schema::try_from(&schema)
            .map_err(|error| PyValueError::new_err(format!("arrow stream schema: {error}")))?;
        Ok(Self {
            stream,
            schema: Arc::new(schema),
        })
    }
}

fn stream_error(stream: &mut FFI_ArrowArrayStream, code: i32, operation: &str) -> ArrowError {
    let detail = stream.get_last_error.and_then(|callback| {
        // SAFETY: only called after a nonzero stream status and before release.
        // A non-NULL producer string is NUL-terminated and valid until the next
        // callback, so copy it now; NULL is a valid absence of detailed text.
        let ptr = unsafe { callback(stream) };
        if ptr.is_null() {
            None
        } else {
            // SAFETY: guaranteed by the same C Stream error-string contract.
            Some(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    });
    ArrowError::CDataInterface(detail.unwrap_or_else(|| {
        format!("{operation} returned error code {code} without a detailed description")
    }))
}

impl Iterator for AcquisitionStreamReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut array = FFI_ArrowArray::empty();
        // SAFETY: required callbacks were checked at construction. The stream
        // remains owned/live and is accessed on the attached importing thread.
        let code = unsafe { self.stream.get_next.unwrap()(&mut self.stream, &mut array) };
        if code != 0 {
            return Some(Err(stream_error(&mut self.stream, code, "get_next")));
        }
        if array.is_released() {
            return None;
        }
        // SAFETY: a successful C Stream batch is a struct array matching the
        // imported schema. arrow-rs retains the array's release owner on import
        // and drops it on errors; buffers are never borrowed past that owner.
        let data = unsafe {
            from_ffi_and_data_type(array, DataType::Struct(self.schema.fields().clone()))
        };
        Some(data.and_then(|data| {
            let array = StructArray::from(data);
            if array.null_count() != 0 {
                return Err(ArrowError::CDataInterface(
                    "Arrow record batch has null parent rows".into(),
                ));
            }
            Ok(RecordBatch::from(array))
        }))
    }
}

impl RecordBatchReader for AcquisitionStreamReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

pub(crate) fn import_stream(obj: &Bound<'_, PyAny>) -> PyResult<AcquisitionStreamReader> {
    // Older supported Polars exporters require this argument positionally;
    // None requests the producer's native schema without dtype negotiation.
    let capsule = obj.call_method1("__arrow_c_stream__", (obj.py().None(),))?;
    let cap: Bound<'_, PyCapsule> = capsule.extract()?;
    let ptr = cap
        .pointer_checked(Some(c"arrow_array_stream"))?
        .cast::<FFI_ArrowArrayStream>()
        .as_ptr();
    // SAFETY: a correctly named Arrow capsule owns this initialized C stream.
    // Replacing it with an empty stream transfers ownership exactly once and
    // leaves its destructor with nothing to release. arrow-rs owns subsequent
    // batch/stream callbacks; this whole operation stays attached to Python.
    let stream = unsafe { std::ptr::replace(ptr, FFI_ArrowArrayStream::empty()) };
    AcquisitionStreamReader::try_new(stream)
}

fn supported_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean
    )
}

enum NumericStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl NumericStorage {
    fn empty(profile: PrecisionProfile) -> Self {
        match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => Self::F32(Vec::new()),
            PrecisionProfile::Fp64 => Self::F64(Vec::new()),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    fn reserve_exact_count(&mut self, count: usize) -> PyResult<()> {
        let additional = count.checked_sub(self.len()).ok_or_else(|| {
            PyValueError::new_err("Arrow acquisition size does not advance monotonically")
        })?;
        match self {
            Self::F32(values) => values.try_reserve_exact(additional),
            Self::F64(values) => values.try_reserve_exact(additional),
        }
        .map_err(|error| PyValueError::new_err(format!("numeric input allocation failed: {error}")))
    }

    fn resize(&mut self, count: usize) -> PyResult<()> {
        let additional = count.checked_sub(self.len()).ok_or_else(|| {
            PyValueError::new_err("Arrow acquisition size does not advance monotonically")
        })?;
        match self {
            Self::F32(values) => {
                values.try_reserve(additional).map_err(|error| {
                    PyValueError::new_err(format!("numeric input allocation failed: {error}"))
                })?;
                values.resize(count, 0.0);
            }
            Self::F64(values) => {
                values.try_reserve(additional).map_err(|error| {
                    PyValueError::new_err(format!("numeric input allocation failed: {error}"))
                })?;
                values.resize(count, 0.0);
            }
        }
        Ok(())
    }

    fn push(&mut self, value: f64, label: &str) -> PyResult<()> {
        let count = self
            .len()
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("numeric input exceeds host address space"))?;
        self.resize(count)?;
        match self {
            Self::F32(values) => values[count - 1] = checked_f32(value, label)?,
            Self::F64(values) => values[count - 1] = value,
        }
        Ok(())
    }
}

fn checked_f32(value: f64, label: &str) -> PyResult<f32> {
    // Test the source value before narrowing: values just above f32::MAX can
    // still round to finite f32, whereas a larger finite value becomes Inf.
    // Source NaN/Inf are intentionally preserved, not reclassified as overflow.
    if value.is_finite() && value.abs() > f64::from(f32::MAX) {
        return Err(PyValueError::new_err(format!(
            "{label} contains a value outside fp32 range."
        )));
    }
    Ok(value as f32)
}

fn copy_numeric_column(
    output: &mut NumericStorage,
    column: &ArrayRef,
    base: usize,
    width: usize,
    column_index: usize,
    label: &str,
) -> PyResult<()> {
    // Dispatch once per column. Safe Arrow accessors retain sliced-array
    // offsets; integer conversion matches the prior Python float(value) step.
    macro_rules! copy_column {
        ($array_type:ty, $convert:expr) => {{
            let values = column
                .as_any()
                .downcast_ref::<$array_type>()
                .ok_or_else(|| {
                    PyValueError::new_err("Arrow column does not match its declared type")
                })?;
            match output {
                NumericStorage::F32(out) => {
                    for row in 0..values.len() {
                        let value = ($convert)(values.value(row));
                        out[base + row * width + column_index] = checked_f32(value, label)?;
                    }
                }
                NumericStorage::F64(out) => {
                    for row in 0..values.len() {
                        out[base + row * width + column_index] = ($convert)(values.value(row));
                    }
                }
            }
        }};
    }
    match column.data_type() {
        DataType::Float32 => {
            let values = column
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    PyValueError::new_err("Arrow Float32 column has an incompatible representation")
                })?;
            match output {
                NumericStorage::F32(out) => {
                    for row in 0..values.len() {
                        out[base + row * width + column_index] = values.value(row);
                    }
                }
                NumericStorage::F64(out) => {
                    for row in 0..values.len() {
                        out[base + row * width + column_index] = f64::from(values.value(row));
                    }
                }
            }
        }
        DataType::Float64 => copy_column!(Float64Array, |value: f64| value),
        DataType::Int8 => copy_column!(Int8Array, |value: i8| f64::from(value)),
        DataType::Int16 => copy_column!(Int16Array, |value: i16| f64::from(value)),
        DataType::Int32 => copy_column!(Int32Array, |value: i32| f64::from(value)),
        DataType::Int64 => copy_column!(Int64Array, |value: i64| value as f64),
        DataType::UInt8 => copy_column!(UInt8Array, |value: u8| f64::from(value)),
        DataType::UInt16 => copy_column!(UInt16Array, |value: u16| f64::from(value)),
        DataType::UInt32 => copy_column!(UInt32Array, |value: u32| f64::from(value)),
        DataType::UInt64 => copy_column!(UInt64Array, |value: u64| value as f64),
        DataType::Boolean => {
            copy_column!(BooleanArray, |value: bool| if value { 1.0 } else { 0.0 })
        }
        dtype => {
            return Err(PyNotImplementedError::new_err(format!(
                "native Arrow acquisition does not support {dtype:?}"
            )))
        }
    }
    Ok(())
}

fn append_batch(
    output: &mut NumericStorage,
    batch: &RecordBatch,
    rows: &mut usize,
    cols: usize,
    label: &str,
) -> PyResult<()> {
    if batch.num_columns() != cols {
        return Err(PyValueError::new_err("Arrow batch column count changed"));
    }
    for column in batch.columns() {
        if column.null_count() != 0 {
            return Err(PyValueError::new_err(if label == "X" {
                "null feature values are not supported"
            } else {
                "null target values are not supported"
            }));
        }
    }
    let new_rows = rows
        .checked_add(batch.num_rows())
        .ok_or_else(|| PyValueError::new_err("Arrow row count exceeds host address space"))?;
    let count = new_rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols exceed host address space"))?;
    let base = output.len();
    output.resize(count)?;
    for (index, column) in batch.columns().iter().enumerate() {
        copy_numeric_column(output, column, base, cols, index, label)?;
    }
    *rows = new_rows;
    Ok(())
}

fn read_stream(
    mut reader: AcquisitionStreamReader,
    profile: PrecisionProfile,
    label: &str,
    expected_rows: Option<usize>,
) -> PyResult<(usize, usize, NumericStorage)> {
    let schema = reader.schema();
    let cols = schema.fields().len();
    if cols == 0 {
        return Err(PyValueError::new_err("empty Arrow input"));
    }
    if label == "y" && cols != 1 {
        return Err(PyValueError::new_err(
            "target must contain exactly one column",
        ));
    }
    u32::try_from(cols)
        .map_err(|_| PyValueError::new_err("X feature count exceeds supported range"))?;
    for field in schema.fields() {
        if !supported_dtype(field.data_type()) {
            return Err(PyNotImplementedError::new_err(format!(
                "native Arrow acquisition does not support {:?}",
                field.data_type()
            )));
        }
    }
    let mut rows = 0;
    let mut storage = NumericStorage::empty(profile);
    if let Some(expected_rows) = expected_rows {
        let count = expected_rows
            .checked_mul(cols)
            .ok_or_else(|| PyValueError::new_err("rows*cols exceed host address space"))?;
        // A materialized Polars frame knows its height. Reserve its logical
        // storage once, rather than retaining geometric surplus that depends on
        // Arrow chunking. This is allocation metadata, not a trusted row count:
        // each batch and EOS must still agree. Unknown-length streams retain
        // amortized growth; per-batch exact growth would cause quadratic copying.
        storage.reserve_exact_count(count)?;
    }
    for batch in &mut reader {
        let batch =
            batch.map_err(|error| PyValueError::new_err(format!("arrow batch: {error}")))?;
        if expected_rows.is_some_and(|expected| batch.num_rows() > expected - rows) {
            return Err(PyValueError::new_err(format!(
                "Arrow {label} row count exceeds expected_rows"
            )));
        }
        append_batch(&mut storage, &batch, &mut rows, cols, label)?;
        // The previous batch's foreign owner is released here, while attached;
        // do not retain every chunk or concatenate/rechunk the Arrow source.
    }
    if expected_rows.is_some_and(|expected| rows != expected) {
        return Err(PyValueError::new_err(format!(
            "Arrow {label} row count does not match expected_rows"
        )));
    }
    Ok((rows, cols, storage))
}

fn owned_input(
    profile: PrecisionProfile,
    features: NumericStorage,
    target: NumericStorage,
) -> PyResult<OwnedNumericInput> {
    match (features, target) {
        (NumericStorage::F32(features), NumericStorage::F32(target)) => {
            OwnedNumericInput::from_f32(profile, features, target).map_err(PyErr::from)
        }
        (NumericStorage::F64(features), NumericStorage::F64(target)) => {
            OwnedNumericInput::from_f64(profile, features, target).map_err(PyErr::from)
        }
        _ => Err(PyValueError::new_err(
            "acquisition changed the selected storage dtype",
        )),
    }
}

#[pyfunction(name = "_acquire_arrow_input")]
#[pyo3(signature = (config, features, target, expected_rows=None))]
pub(crate) fn acquire_arrow_input(
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    expected_rows: Option<u64>,
) -> PyResult<PyAcquiredNumericInput> {
    let config = parse_engine_config(config)?;
    let expected_rows = expected_rows
        .map(usize::try_from)
        .transpose()
        .map_err(|_| PyValueError::new_err("expected_rows exceeds host address space"))?;
    let (rows, cols, features) = read_stream(
        import_stream(features)?,
        config.precision,
        "X",
        expected_rows,
    )?;
    let (target_rows, _, target) =
        read_stream(import_stream(target)?, config.precision, "y", expected_rows)?;
    if target_rows != rows {
        return Err(PyValueError::new_err(
            "target length must match feature rows",
        ));
    }
    let rows = u64::try_from(rows)
        .map_err(|_| PyValueError::new_err("X row count exceeds supported range"))?;
    let cols = u32::try_from(cols)
        .map_err(|_| PyValueError::new_err("X feature count exceeds supported range"))?;
    let input = owned_input(config.precision, features, target)?;
    PyAcquiredNumericInput::new(config.precision, rows, cols, input)
}

fn compatibility_float(
    float: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    label: &str,
    profile: PrecisionProfile,
) -> PyResult<f64> {
    float
        .call1((value,))
        .and_then(|value| value.extract::<f64>())
        .map_err(|error| {
            if profile == PrecisionProfile::Fp64
                || error.is_instance_of::<PyOverflowError>(value.py())
            {
                PyValueError::new_err(format!(
                    "{label} contains a value outside {} range.",
                    if profile == PrecisionProfile::Fp64 {
                        "fp64"
                    } else {
                        "fp32"
                    }
                ))
            } else if error.is_instance_of::<PyTypeError>(value.py())
                || error.is_instance_of::<PyValueError>(value.py())
            {
                PyValueError::new_err(format!("{label} contains a non-numeric value."))
            } else {
                error
            }
        })
}

#[pyfunction(name = "_acquire_rows_input")]
pub(crate) fn acquire_rows_input(
    py: Python<'_>,
    config: &Bound<'_, PyDict>,
    features: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyAcquiredNumericInput> {
    let config = parse_engine_config(config)?;
    let float = py.import("builtins")?.getattr("float")?;
    let mut feature_storage = NumericStorage::empty(config.precision);
    let mut target_storage = NumericStorage::empty(config.precision);
    let mut rows = 0_usize;
    let mut cols = None;
    if features.is_instance_of::<PyString>() || features.is_instance_of::<PyBytes>() {
        return Err(PyValueError::new_err("X must be numeric, not a string."));
    }
    for row in features
        .try_iter()
        .map_err(|_| PyValueError::new_err("X must be an iterable numeric sequence."))?
    {
        let row = row?;
        if row.is_instance_of::<PyString>() || row.is_instance_of::<PyBytes>() {
            return Err(PyValueError::new_err(format!(
                "X[{rows}] must be numeric, not a string."
            )));
        }
        let mut count = 0_usize;
        for value in row.try_iter().map_err(|_| {
            PyValueError::new_err(format!("X[{rows}] must be an iterable numeric sequence."))
        })? {
            let value = value?;
            feature_storage.push(
                compatibility_float(&float, &value, "X", config.precision)?,
                "X",
            )?;
            count = count
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("X feature count exceeds supported range"))?;
        }
        match cols {
            None if count == 0 => {
                return Err(PyValueError::new_err(
                    "X must contain at least one feature.",
                ))
            }
            None => cols = Some(count),
            Some(cols) if count != cols => {
                return Err(PyValueError::new_err(format!(
                    "X row {rows} has length {count}; expected {cols}."
                )))
            }
            _ => {}
        }
        rows = rows
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("X row count exceeds supported range"))?;
    }
    if rows == 0 {
        return Err(PyValueError::new_err("X must contain at least one sample."));
    }
    if target.is_instance_of::<PyString>() || target.is_instance_of::<PyBytes>() {
        return Err(PyValueError::new_err("y must be numeric, not a string."));
    }
    for value in target
        .try_iter()
        .map_err(|_| PyValueError::new_err("y must be an iterable numeric sequence."))?
    {
        let value = value?;
        target_storage.push(
            compatibility_float(&float, &value, "y", config.precision)?,
            "y",
        )?;
    }
    let cols = u32::try_from(cols.unwrap_or(0))
        .map_err(|_| PyValueError::new_err("X feature count exceeds supported range"))?;
    let rows = u64::try_from(rows)
        .map_err(|_| PyValueError::new_err("X row count exceeds supported range"))?;
    let input = owned_input(config.precision, feature_storage, target_storage)?;
    PyAcquiredNumericInput::new(config.precision, rows, cols, input)
}

#[pyfunction(name = "_compile_acquired_continuous")]
pub(crate) fn compile_acquired_continuous(
    config: &Bound<'_, PyDict>,
    mut input: PyRefMut<'_, PyAcquiredNumericInput>,
) -> PyResult<PyCompiledContinuousArtifact> {
    let config = parse_engine_config(config)?;
    let numeric = input.take_input(config.precision)?;
    compile_continuous_input(config, input.rows, input.cols, numeric).map_err(PyErr::from)
}

#[pyfunction(name = "_compile_acquired_time_series")]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors existing temporal compile inputs"
)]
pub(crate) fn compile_acquired_time_series(
    config: &Bound<'_, PyDict>,
    mut input: PyRefMut<'_, PyAcquiredNumericInput>,
    base_names: Vec<String>,
    lags: Vec<u32>,
    windows: Vec<u32>,
    velocity: bool,
) -> PyResult<(PyCompiledContinuousArtifact, Vec<String>)> {
    let config = parse_engine_config(config)?;
    let numeric = input.take_input(config.precision)?;
    let artifact = compile_time_series_input(
        config, input.rows, input.cols, numeric, base_names, lags, windows, velocity,
    )?;
    let names = artifact.feature_names.clone();
    Ok((artifact, names))
}

#[pyfunction(name = "_compile_acquired_decision_path")]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors existing decision-path compile inputs"
)]
pub(crate) fn compile_acquired_decision_path(
    config: &Bound<'_, PyDict>,
    mut input: PyRefMut<'_, PyAcquiredNumericInput>,
    base_names: Vec<String>,
    max_depth: u32,
    rounds: u32,
    max_paths: u32,
    max_bins: u32,
    min_leaf: u32,
    learning_rate: f64,
) -> PyResult<(PyCompiledContinuousArtifact, Vec<String>)> {
    let parsed = parse_engine_config(config)?;
    let top_k = get_u32(config, "decision_path_top_k_features", 50)?;
    let numeric = input.take_input(parsed.precision)?;
    let params = gafime_cpu::decision_path::DecisionPathParams {
        max_depth,
        rounds,
        max_paths,
        max_bins,
        min_leaf,
        learning_rate,
    };
    compile_decision_path_input(
        parsed, input.rows, input.cols, numeric, base_names, params, top_k,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use arrow::{
        array::StringArray, datatypes::SchemaRef, error::ArrowError,
        record_batch::RecordBatchIterator,
    };

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(columns).unwrap()
    }

    fn reader(batches: Vec<RecordBatch>) -> AcquisitionStreamReader {
        let schema = batches[0].schema();
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        AcquisitionStreamReader::try_new(FFI_ArrowArrayStream::new(Box::new(batches))).unwrap()
    }

    fn hex(bytes: [u8; 16]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn fingerprints_match_existing_python_le_byte_format() {
        let values = [1.25_f32, -0.0, f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(
            hex(digest_f32(PrecisionProfile::Fp32, &values)),
            "b69200c59f08638c27c8de66f4d2f05f"
        );
        assert_eq!(
            hex(digest_f32(PrecisionProfile::Mixed, &values)),
            "b669bf4930e22ec1ec1d9f56b4fde417"
        );
        let values = values.map(f64::from);
        assert_eq!(
            hex(digest_f64(PrecisionProfile::Fp64, &values)),
            "f1b861dc028636fd063ef2722aa1563b"
        );
        // Cross the scratch-block boundary and compare to a canonical streamed
        // byte reference, preventing a last-block or count-format regression.
        let values = (0..1025).map(|value| value as f32).collect::<Vec<_>>();
        let mut reference = digest_start(PrecisionProfile::Mixed, values.len());
        for value in &values {
            reference.update(&value.to_le_bytes());
        }
        assert_eq!(
            digest_f32(PrecisionProfile::Mixed, &values),
            digest_finish(reference)
        );
    }

    #[test]
    fn finite_range_is_checked_before_narrowing_but_source_nonfinite_is_preserved() {
        let maximum = f64::from(f32::MAX);
        let above = f64::from_bits(maximum.to_bits() + 1);
        assert_eq!(above as f32, f32::MAX);
        assert_eq!(checked_f32(maximum, "X").unwrap(), f32::MAX);
        for value in [above, -above, maximum + 2.0_f64.powi(103), 1.0e39, 1.0e100] {
            assert!(checked_f32(value, "X").is_err());
            assert!(checked_f32(value, "y").is_err());
        }
        assert!(checked_f32(f64::NAN, "X").unwrap().is_nan());
        assert_eq!(checked_f32(f64::INFINITY, "X").unwrap(), f32::INFINITY);
        assert_eq!(
            checked_f32(f64::NEG_INFINITY, "X").unwrap(),
            f32::NEG_INFINITY
        );
        assert_eq!(
            checked_f32(-0.0, "X").unwrap().to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            checked_f32(1.0e-310, "X").unwrap().to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            checked_f32(1.0e-40, "X").unwrap().to_bits(),
            (1.0e-40_f64 as f32).to_bits()
        );
        assert_eq!(
            checked_f32(1.0e-46, "X").unwrap().to_bits(),
            0.0_f32.to_bits()
        );
    }

    #[test]
    fn chunks_and_sliced_offsets_preserve_row_major_order() {
        let x = Arc::new(Float64Array::from(vec![99.0, 1.0, 2.0, 3.0, 4.0])) as ArrayRef;
        let z = Arc::new(Int64Array::from(vec![99, 10, 20, 30, 40])) as ArrayRef;
        let batches = vec![
            batch(vec![("x", x.slice(1, 1)), ("z", z.slice(1, 1))]),
            batch(vec![("x", x.slice(2, 0)), ("z", z.slice(2, 0))]),
            batch(vec![("x", x.slice(2, 3)), ("z", z.slice(2, 3))]),
        ];
        let (rows, cols, storage) =
            read_stream(reader(batches), PrecisionProfile::Mixed, "X", None).unwrap();
        assert_eq!((rows, cols), (4, 2));
        let NumericStorage::F32(values) = storage else {
            panic!("wrong profile")
        };
        assert_eq!(values, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]);
        let y = Arc::new(Float64Array::from(vec![99.0, 4.0, 3.0, 2.0, 1.0])) as ArrayRef;
        let batches = vec![
            batch(vec![("y", y.slice(1, 2))]),
            batch(vec![("y", y.slice(3, 2))]),
        ];
        let (target_rows, target_cols, target) =
            read_stream(reader(batches), PrecisionProfile::Mixed, "y", None).unwrap();
        assert_eq!((target_rows, target_cols), (4, 1));
        let NumericStorage::F32(target) = target else {
            panic!("wrong profile")
        };
        let input = OwnedNumericInput::from_f32(PrecisionProfile::Mixed, values, target).unwrap();
        PyAcquiredNumericInput::new(PrecisionProfile::Mixed, rows as u64, cols as u32, input)
            .unwrap();
    }

    #[test]
    fn known_rows_avoid_chunk_dependent_geometric_surplus_without_changing_values() {
        let capacity = |storage: &NumericStorage| match storage {
            NumericStorage::F32(values) => values.capacity(),
            NumericStorage::F64(values) => values.capacity(),
        };
        for profile in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            let values =
                Arc::new(Float64Array::from_iter_values((0..10).map(f64::from))) as ArrayRef;
            let batches = vec![
                batch(vec![("x", values.slice(0, 6)), ("z", values.slice(0, 6))]),
                batch(vec![("x", values.slice(6, 4)), ("z", values.slice(6, 4))]),
            ];
            let (_, _, grown) = read_stream(reader(batches.clone()), profile, "X", None).unwrap();
            let (_, _, reserved) = read_stream(reader(batches), profile, "X", Some(10)).unwrap();
            assert_eq!(grown.len(), 20);
            assert_eq!(reserved.len(), 20);
            assert_eq!(capacity(&reserved), 20);
            assert!(capacity(&grown) > capacity(&reserved));
            match (grown, reserved) {
                (NumericStorage::F32(grown), NumericStorage::F32(reserved)) => {
                    assert_eq!(grown, reserved);
                }
                (NumericStorage::F64(grown), NumericStorage::F64(reserved)) => {
                    assert_eq!(grown, reserved);
                }
                _ => panic!("selected profile changed"),
            }
        }
    }

    #[test]
    fn fp64_and_f32_native_values_do_not_gain_an_intermediate_round_trip() {
        let base = 1.0_f64;
        let adjacent = f64::from_bits(base.to_bits() + 1);
        let values = Arc::new(Float64Array::from(vec![base, adjacent, 1.0e100])) as ArrayRef;
        let (_, _, storage) = read_stream(
            reader(vec![batch(vec![("x", values)])]),
            PrecisionProfile::Fp64,
            "X",
            None,
        )
        .unwrap();
        let NumericStorage::F64(values) = storage else {
            panic!("wrong profile")
        };
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![base.to_bits(), adjacent.to_bits(), 1.0e100_f64.to_bits()]
        );
        let source = [f32::from_bits(0x7fc0_1234), -0.0_f32];
        let values = Arc::new(Float32Array::from(source.to_vec())) as ArrayRef;
        let (_, _, storage) = read_stream(
            reader(vec![batch(vec![("x", values)])]),
            PrecisionProfile::Fp32,
            "X",
            None,
        )
        .unwrap();
        let NumericStorage::F32(values) = storage else {
            panic!("wrong profile")
        };
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            source.map(f32::to_bits)
        );
    }

    #[test]
    fn numeric_schema_and_null_rejections_are_explicit() {
        for label in ["X", "y"] {
            let column = Arc::new(Float32Array::from(vec![Some(1.0), None])) as ArrayRef;
            assert!(read_stream(
                reader(vec![batch(vec![("x", column)])]),
                PrecisionProfile::Mixed,
                label,
                None,
            )
            .is_err());
        }
        let column = Arc::new(StringArray::from(vec!["1.5", "2.5"])) as ArrayRef;
        assert!(read_stream(
            reader(vec![batch(vec![("x", column)])]),
            PrecisionProfile::Mixed,
            "X",
            None,
        )
        .is_err());
        assert!(usize::MAX.checked_mul(2).is_none());
        let column = Arc::new(Float32Array::from(vec![1.0])) as ArrayRef;
        let batch = batch(vec![("x", column)]);
        let mut storage = NumericStorage::empty(PrecisionProfile::Mixed);
        let mut rows = usize::MAX;
        assert!(append_batch(&mut storage, &batch, &mut rows, 1, "X").is_err());
        let mut rows = usize::MAX / 2;
        assert!(append_batch(&mut storage, &batch, &mut rows, 3, "X").is_err());
    }

    #[test]
    fn integer_and_boolean_conversion_matches_existing_scalar_boundary() {
        // Integer -> f64 -> f32 differs from a direct integer -> f32 cast at
        // this midpoint. The old Python float(value) path defines compatibility.
        let midpoint_plus_one = (1_u64 << 60) + (1_u64 << 36) + 1;
        assert_ne!((midpoint_plus_one as f64) as f32, midpoint_plus_one as f32);
        let integers = Arc::new(UInt64Array::from(vec![midpoint_plus_one, u64::MAX])) as ArrayRef;
        let booleans = Arc::new(BooleanArray::from(vec![true, false])) as ArrayRef;
        let (_, _, storage) = read_stream(
            reader(vec![batch(vec![("x", integers), ("b", booleans)])]),
            PrecisionProfile::Mixed,
            "X",
            None,
        )
        .unwrap();
        let NumericStorage::F32(values) = storage else {
            panic!("wrong profile")
        };
        assert_eq!(
            values,
            vec![
                (midpoint_plus_one as f64) as f32,
                1.0,
                (u64::MAX as f64) as f32,
                0.0
            ]
        );
    }

    struct CountedReader {
        inner: RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, ArrowError>>>,
        reads: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }

    impl Iterator for CountedReader {
        type Item = Result<RecordBatch, ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.inner.next()
        }
    }

    impl RecordBatchReader for CountedReader {
        fn schema(&self) -> SchemaRef {
            self.inner.schema()
        }
    }

    impl Drop for CountedReader {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn no_stream_error_detail(
        _stream: *mut FFI_ArrowArrayStream,
    ) -> *const std::ffi::c_char {
        std::ptr::null()
    }

    // Test-only C Data layout views permit counted release callbacks without
    // relying on arrow-rs private data. Their layout is asserted before use;
    // each wrapper owns the original arrow-rs object and forwards its lifetime.
    #[repr(C)]
    struct SchemaReleaseView {
        format: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        metadata: *const std::ffi::c_char,
        flags: i64,
        n_children: i64,
        children: *mut *mut FFI_ArrowSchema,
        dictionary: *mut FFI_ArrowSchema,
        release: Option<unsafe extern "C" fn(*mut FFI_ArrowSchema)>,
        private_data: *mut std::ffi::c_void,
    }

    #[repr(C)]
    struct ArrayReleaseView {
        length: i64,
        null_count: i64,
        offset: i64,
        n_buffers: i64,
        n_children: i64,
        buffers: *mut *const std::ffi::c_void,
        children: *mut *mut FFI_ArrowArray,
        dictionary: *mut FFI_ArrowArray,
        release: Option<unsafe extern "C" fn(*mut FFI_ArrowArray)>,
        private_data: *mut std::ffi::c_void,
    }

    struct ReleaseOwner<T> {
        original: T,
        releases: Arc<AtomicUsize>,
    }

    unsafe extern "C" fn release_counted_schema(schema: *mut FFI_ArrowSchema) {
        // SAFETY: only installed on a counted schema with the asserted C layout.
        let view = unsafe { &mut *schema.cast::<SchemaReleaseView>() };
        let owner = view.private_data.cast::<ReleaseOwner<FFI_ArrowSchema>>();
        view.release = None;
        view.private_data = std::ptr::null_mut();
        // SAFETY: this callback consumes the unique box installed below once.
        let owner = unsafe { Box::from_raw(owner) };
        owner.releases.fetch_add(1, Ordering::SeqCst);
        drop(owner);
    }

    fn count_schema_release(
        original: FFI_ArrowSchema,
        releases: Arc<AtomicUsize>,
    ) -> FFI_ArrowSchema {
        assert_eq!(size_of::<SchemaReleaseView>(), size_of::<FFI_ArrowSchema>());
        assert_eq!(
            align_of::<SchemaReleaseView>(),
            align_of::<FFI_ArrowSchema>()
        );
        let owner = Box::into_raw(Box::new(ReleaseOwner { original, releases }));
        // SAFETY: the copied C fields borrow the original owner's allocations;
        // replacing release/private_data makes the copy own that box, not a
        // duplicate arrow-rs release obligation. No fallible work follows read.
        let mut output = unsafe { std::ptr::read(&(*owner).original) };
        // SAFETY: the asserted repr(C) layout matches this initialized schema.
        let view =
            unsafe { &mut *(&mut output as *mut FFI_ArrowSchema).cast::<SchemaReleaseView>() };
        view.release = Some(release_counted_schema);
        view.private_data = owner.cast();
        output
    }

    unsafe extern "C" fn release_counted_array(array: *mut FFI_ArrowArray) {
        // SAFETY: only installed on a counted array with the asserted C layout.
        let view = unsafe { &mut *array.cast::<ArrayReleaseView>() };
        let owner = view.private_data.cast::<ReleaseOwner<FFI_ArrowArray>>();
        view.release = None;
        view.private_data = std::ptr::null_mut();
        // SAFETY: this callback consumes the unique box installed below once.
        let owner = unsafe { Box::from_raw(owner) };
        owner.releases.fetch_add(1, Ordering::SeqCst);
        drop(owner);
    }

    fn count_array_release(original: FFI_ArrowArray, releases: Arc<AtomicUsize>) -> FFI_ArrowArray {
        assert_eq!(size_of::<ArrayReleaseView>(), size_of::<FFI_ArrowArray>());
        assert_eq!(align_of::<ArrayReleaseView>(), align_of::<FFI_ArrowArray>());
        let owner = Box::into_raw(Box::new(ReleaseOwner { original, releases }));
        // SAFETY: the copy borrows original allocations until this box is
        // released; the original arrow-rs release callback stays on the owner.
        let mut output = unsafe { std::ptr::read(&(*owner).original) };
        // SAFETY: the asserted repr(C) layout matches this initialized array.
        let view = unsafe { &mut *(&mut output as *mut FFI_ArrowArray).cast::<ArrayReleaseView>() };
        view.release = Some(release_counted_array);
        view.private_data = owner.cast();
        output
    }

    struct TrackedStream {
        original: FFI_ArrowArrayStream,
        schema_releases: Arc<AtomicUsize>,
        array_releases: Arc<AtomicUsize>,
        schema_error: bool,
        null_error_detail: bool,
    }

    unsafe extern "C" fn tracked_get_schema(
        stream: *mut FFI_ArrowArrayStream,
        output: *mut FFI_ArrowSchema,
    ) -> i32 {
        // SAFETY: this fixture owns the TrackedStream and initialized output;
        // its original is created by arrow-rs with all required callbacks. The
        // produced schema moves into exactly one counted owner before return.
        unsafe {
            let owner = &mut *(*stream).private_data.cast::<TrackedStream>();
            let status = owner.original.get_schema.unwrap()(&mut owner.original, output);
            if status == 0 {
                let original = std::ptr::replace(output, FFI_ArrowSchema::empty());
                *output = count_schema_release(original, owner.schema_releases.clone());
            }
            if owner.schema_error {
                5
            } else {
                status
            }
        }
    }

    unsafe extern "C" fn tracked_get_next(
        stream: *mut FFI_ArrowArrayStream,
        output: *mut FFI_ArrowArray,
    ) -> i32 {
        // SAFETY: this fixture owns the TrackedStream and initialized output;
        // its original is created by arrow-rs with all required callbacks. Only
        // a produced batch moves into the counted owner; EOS stays released.
        unsafe {
            let owner = &mut *(*stream).private_data.cast::<TrackedStream>();
            let status = owner.original.get_next.unwrap()(&mut owner.original, output);
            if status == 0 && !(&*output).is_released() {
                let original = std::ptr::replace(output, FFI_ArrowArray::empty());
                *output = count_array_release(original, owner.array_releases.clone());
            }
            status
        }
    }

    unsafe extern "C" fn tracked_get_last_error(
        stream: *mut FFI_ArrowArrayStream,
    ) -> *const std::ffi::c_char {
        // SAFETY: only called on a live TrackedStream after a failed operation.
        let owner = unsafe { &mut *(*stream).private_data.cast::<TrackedStream>() };
        if owner.null_error_detail || owner.schema_error {
            std::ptr::null()
        } else {
            // SAFETY: the original arrow-rs callback is live and has just
            // returned a nonzero get_next status on this importing thread.
            unsafe { owner.original.get_last_error.unwrap()(&mut owner.original) }
        }
    }

    unsafe extern "C" fn tracked_release_stream(stream: *mut FFI_ArrowArrayStream) {
        // SAFETY: the unique fixture box is consumed once; dropping its original
        // stream reaches CountedReader::drop and releases its remaining batches.
        let stream = unsafe { &mut *stream };
        let owner = stream.private_data.cast::<TrackedStream>();
        stream.release = None;
        stream.private_data = std::ptr::null_mut();
        // SAFETY: private_data came from one Box::into_raw in tracked_stream.
        drop(unsafe { Box::from_raw(owner) });
    }

    fn tracked_stream(
        source: CountedReader,
        schema_releases: Arc<AtomicUsize>,
        array_releases: Arc<AtomicUsize>,
        schema_error: bool,
        null_error_detail: bool,
    ) -> FFI_ArrowArrayStream {
        let owner = Box::new(TrackedStream {
            original: FFI_ArrowArrayStream::new(Box::new(source)),
            schema_releases,
            array_releases,
            schema_error,
            null_error_detail,
        });
        FFI_ArrowArrayStream {
            get_schema: Some(tracked_get_schema),
            get_next: Some(tracked_get_next),
            get_last_error: Some(tracked_get_last_error),
            release: Some(tracked_release_stream),
            private_data: Box::into_raw(owner).cast(),
        }
    }

    #[test]
    fn stream_schema_array_release_and_errors_obey_c_stream_contract() {
        Python::initialize();
        // Modes: EOS, detailed batch error, NULL-detail batch error, schema
        // error with an independently initialized output, and rejected nulls.
        for mode in 0..5 {
            let column = if mode == 4 {
                Float32Array::from(vec![None])
            } else {
                Float32Array::from(vec![Some(1.0)])
            };
            let batch = batch(vec![("x", Arc::new(column) as ArrayRef)]);
            let schema = batch.schema();
            let batches = if mode == 1 || mode == 2 {
                vec![
                    Ok(batch),
                    Err(ArrowError::ComputeError("specific detail".into())),
                ]
            } else {
                vec![Ok(batch)]
            };
            let reads = Arc::new(AtomicUsize::new(0));
            let drops = Arc::new(AtomicUsize::new(0));
            let schema_releases = Arc::new(AtomicUsize::new(0));
            let array_releases = Arc::new(AtomicUsize::new(0));
            let source = CountedReader {
                inner: RecordBatchIterator::new(batches.into_iter(), schema),
                reads: reads.clone(),
                drops: drops.clone(),
            };
            let stream = tracked_stream(
                source,
                schema_releases.clone(),
                array_releases.clone(),
                mode == 3,
                mode == 2,
            );
            let result = AcquisitionStreamReader::try_new(stream)
                .and_then(|reader| read_stream(reader, PrecisionProfile::Mixed, "X", None));
            assert_eq!(result.is_ok(), mode == 0);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert_eq!(schema_releases.load(Ordering::SeqCst), 1);
            assert_eq!(
                array_releases.load(Ordering::SeqCst),
                usize::from(mode != 3)
            );
            if let Err(error) = result {
                Python::attach(|py| {
                    assert!(error.is_instance_of::<PyValueError>(py));
                    let text = error.to_string();
                    if mode == 1 {
                        assert!(text.contains("specific detail"));
                    } else if mode == 2 || mode == 3 {
                        assert!(text.contains("without a detailed description"));
                    }
                });
            }
        }
    }

    #[test]
    fn expected_row_count_errors_release_foreign_owners_once() {
        Python::initialize();
        for expected in [0, 2, 4, usize::MAX] {
            let batch = batch(vec![
                (
                    "x",
                    Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef,
                ),
                (
                    "z",
                    Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])) as ArrayRef,
                ),
            ]);
            let schema = batch.schema();
            let reads = Arc::new(AtomicUsize::new(0));
            let drops = Arc::new(AtomicUsize::new(0));
            let schema_releases = Arc::new(AtomicUsize::new(0));
            let array_releases = Arc::new(AtomicUsize::new(0));
            let source = CountedReader {
                inner: RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
                reads: reads.clone(),
                drops: drops.clone(),
            };
            let reader = AcquisitionStreamReader::try_new(tracked_stream(
                source,
                schema_releases.clone(),
                array_releases.clone(),
                false,
                false,
            ))
            .unwrap();
            let error = read_stream(reader, PrecisionProfile::Fp64, "X", Some(expected))
                .err()
                .expect("inexact or overflowing count must fail");
            Python::attach(|py| assert!(error.is_instance_of::<PyValueError>(py)));
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert_eq!(schema_releases.load(Ordering::SeqCst), 1);
            assert_eq!(
                array_releases.load(Ordering::SeqCst),
                usize::from(expected != usize::MAX)
            );
            if expected == usize::MAX {
                assert_eq!(reads.load(Ordering::SeqCst), 0);
                assert!(error.to_string().contains("rows*cols"));
            } else {
                assert!(error.to_string().contains("expected_rows"));
            }
        }
    }

    #[test]
    fn stream_error_without_optional_detail_returns_value_error_and_releases_once() {
        let schema = batch(vec![(
            "x",
            Arc::new(Float32Array::from(vec![1.0])) as ArrayRef,
        )])
        .schema();
        let reads = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let source = CountedReader {
            inner: RecordBatchIterator::new(
                vec![Err(ArrowError::ComputeError("injected error".into()))].into_iter(),
                schema,
            ),
            reads: reads.clone(),
            drops: drops.clone(),
        };
        let mut stream = FFI_ArrowArrayStream::new(Box::new(source));
        // The callback is mandatory, but its NULL return is explicitly allowed
        // by the C Stream Interface when no detailed description is available.
        stream.get_last_error = Some(no_stream_error_detail);
        let reader = AcquisitionStreamReader::try_new(stream).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_stream(reader, PrecisionProfile::Mixed, "X", None)
        }));
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(result.is_ok(), "a conforming stream error must not unwind");
        let error = result
            .unwrap()
            .err()
            .expect("stream error must be reported");
        Python::initialize();
        Python::attach(|py| assert!(error.is_instance_of::<PyValueError>(py)));
    }

    #[test]
    fn stream_is_released_once_on_success_schema_and_batch_errors() {
        for mode in [0, 1, 2, 3] {
            let column = if mode == 1 {
                Arc::new(StringArray::from(vec!["1.5"])) as ArrayRef
            } else if mode == 3 {
                Arc::new(Float32Array::from(vec![None])) as ArrayRef
            } else {
                Arc::new(Float32Array::from(vec![1.0])) as ArrayRef
            };
            let batch = batch(vec![("x", column)]);
            let schema = batch.schema();
            let reads = Arc::new(AtomicUsize::new(0));
            let drops = Arc::new(AtomicUsize::new(0));
            let batches = if mode == 2 {
                vec![
                    Ok(batch),
                    Err(ArrowError::ComputeError(
                        "injected stream error".to_string(),
                    )),
                ]
            } else {
                vec![Ok(batch)]
            };
            let source = CountedReader {
                inner: RecordBatchIterator::new(batches.into_iter(), schema),
                reads: reads.clone(),
                drops: drops.clone(),
            };
            let reader =
                AcquisitionStreamReader::try_new(FFI_ArrowArrayStream::new(Box::new(source)))
                    .unwrap();
            let result = read_stream(reader, PrecisionProfile::Mixed, "X", None);
            assert_eq!(result.is_ok(), mode == 0);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            if mode == 1 {
                assert_eq!(reads.load(Ordering::SeqCst), 0);
            }
        }
    }

    #[test]
    fn acquired_storage_can_only_be_consumed_once_in_its_selected_profile() {
        let input =
            OwnedNumericInput::from_f32(PrecisionProfile::Mixed, vec![1.0, 2.0], vec![3.0, 4.0])
                .unwrap();
        let mut acquired =
            PyAcquiredNumericInput::new(PrecisionProfile::Mixed, 2, 1, input).unwrap();
        assert!(acquired.take_input(PrecisionProfile::Fp64).is_err());
        assert!(acquired.take_input(PrecisionProfile::Mixed).is_ok());
        assert!(acquired.take_input(PrecisionProfile::Mixed).is_err());
    }
}
