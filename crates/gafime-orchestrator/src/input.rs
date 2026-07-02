use core::ffi::c_void;

use gafime_types::{
    DataType, GafimeInputDescriptor, GafimeStringView, InputSourceKind, MatrixLayout,
    GAFIME_ABI_VERSION, GAFIME_DTYPE_F32, GAFIME_INPUT_ARROW_C_DATA, GAFIME_INPUT_HOST_F32,
    GAFIME_INPUT_PARQUET_PATH, GAFIME_MATRIX_ARROW_COLUMNAR, GAFIME_MATRIX_ROW_MAJOR,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputSource {
    HostF32 {
        features: *const c_void,
        target: *const c_void,
    },
    ArrowCData {
        features: *const c_void,
        target: *const c_void,
        schema: *const c_void,
    },
    ParquetPath(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeInputDescriptor {
    pub source: InputSource,
    pub rows: u64,
    pub cols: u32,
    pub dtype: DataType,
    pub layout: MatrixLayout,
}

impl NativeInputDescriptor {
    pub fn host_f32(rows: u64, cols: u32, features: *const c_void, target: *const c_void) -> Self {
        Self {
            source: InputSource::HostF32 { features, target },
            rows,
            cols,
            dtype: GAFIME_DTYPE_F32,
            layout: GAFIME_MATRIX_ROW_MAJOR,
        }
    }

    pub fn arrow_c_data(
        rows: u64,
        cols: u32,
        features: *const c_void,
        target: *const c_void,
        schema: *const c_void,
    ) -> Self {
        Self {
            source: InputSource::ArrowCData {
                features,
                target,
                schema,
            },
            rows,
            cols,
            dtype: GAFIME_DTYPE_F32,
            layout: GAFIME_MATRIX_ARROW_COLUMNAR,
        }
    }

    pub fn parquet_path(rows: u64, cols: u32, path: impl Into<String>) -> Self {
        Self {
            source: InputSource::ParquetPath(path.into()),
            rows,
            cols,
            dtype: GAFIME_DTYPE_F32,
            layout: GAFIME_MATRIX_ARROW_COLUMNAR,
        }
    }

    pub fn source_kind(&self) -> InputSourceKind {
        match self.source {
            InputSource::HostF32 { .. } => GAFIME_INPUT_HOST_F32,
            InputSource::ArrowCData { .. } => GAFIME_INPUT_ARROW_C_DATA,
            InputSource::ParquetPath(_) => GAFIME_INPUT_PARQUET_PATH,
        }
    }

    pub fn to_raw(&self) -> GafimeInputDescriptor {
        let mut raw = GafimeInputDescriptor {
            abi_version: GAFIME_ABI_VERSION,
            source_kind: self.source_kind(),
            dtype: self.dtype,
            layout: self.layout,
            rows: self.rows,
            cols: self.cols,
            row_stride: self.cols,
            ..Default::default()
        };
        match &self.source {
            InputSource::HostF32 { features, target } => {
                raw.features_ptr = *features;
                raw.target_ptr = *target;
            }
            InputSource::ArrowCData {
                features,
                target,
                schema,
            } => {
                raw.features_ptr = *features;
                raw.target_ptr = *target;
                raw.schema_ptr = *schema;
            }
            InputSource::ParquetPath(path) => {
                raw.path = GafimeStringView {
                    ptr: path.as_ptr(),
                    len: path.len() as u64,
                };
            }
        }
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parquet_path_binds_without_copying_string_bytes() {
        let input = NativeInputDescriptor::parquet_path(100, 8, "data/train.parquet");
        let raw = input.to_raw();

        assert_eq!(raw.source_kind, GAFIME_INPUT_PARQUET_PATH);
        assert_eq!(raw.rows, 100);
        assert_eq!(raw.cols, 8);
        assert_eq!(raw.path.len, "data/train.parquet".len() as u64);
        assert!(!raw.path.ptr.is_null());
    }
}
