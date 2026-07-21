use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use gafime_types::{BackendKind, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM};
use libloading::Library;

use crate::{abi::load_function_table, abi::GpuSysError, backend::GpuBackend};

pub const CUDA_LIBRARY_ENV: &str = "GAFIME_CUDA_V1_LIB";
pub const ROCM_LIBRARY_ENV: &str = "GAFIME_ROCM_V1_LIB";
pub const METAL_LIBRARY_ENV: &str = "GAFIME_METAL_V1_LIB";

type LibraryCacheKey = (BackendKind, PathBuf);

static GPU_LIBRARY_CACHE: OnceLock<Mutex<HashMap<LibraryCacheKey, Arc<Library>>>> = OnceLock::new();

fn library_cache() -> &'static Mutex<HashMap<LibraryCacheKey, Arc<Library>>> {
    GPU_LIBRARY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Keep immutable payload DSOs loaded for the process lifetime. This cache owns
/// no matrices, plans, results, or device buffers; cache-disabled eager calls
/// therefore remain one-shot executions without repeatedly paying loader and
/// native-runtime initialization costs.
unsafe fn load_process_library(
    path: &Path,
    kind: BackendKind,
) -> Result<Arc<Library>, GpuSysError> {
    let key_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let key = (kind, key_path);
    let mut cache = library_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(library) = cache.get(&key) {
        return Ok(library.clone());
    }
    let library = Arc::new(unsafe {
        Library::new(path).map_err(|err| GpuSysError::LoadLibrary {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?
    });
    cache.insert(key, library.clone());
    Ok(library)
}

impl GpuBackend {
    pub fn cuda_from_env(device_id: u32) -> Result<Self, GpuSysError> {
        let path =
            env::var_os(CUDA_LIBRARY_ENV).ok_or(GpuSysError::EnvMissing(CUDA_LIBRARY_ENV))?;
        unsafe { Self::load_cuda_from_path(path, device_id) }
    }

    /// Load a ROCm/HIP payload implementing the same vendor-agnostic
    /// `gafime_gpu_*` C ABI as CUDA (the loader is vendor-generic; only the env
    /// var and `BackendKind` differ).
    pub fn rocm_from_env(device_id: u32) -> Result<Self, GpuSysError> {
        let path =
            env::var_os(ROCM_LIBRARY_ENV).ok_or(GpuSysError::EnvMissing(ROCM_LIBRARY_ENV))?;
        unsafe { Self::load_rocm_from_path(path, device_id) }
    }

    /// Load an Apple Metal payload implementing the same vendor-agnostic
    /// `gafime_gpu_*` C ABI as CUDA/ROCm.
    pub fn metal_from_env(device_id: u32) -> Result<Self, GpuSysError> {
        let path =
            env::var_os(METAL_LIBRARY_ENV).ok_or(GpuSysError::EnvMissing(METAL_LIBRARY_ENV))?;
        unsafe { Self::load_metal_from_path(path, device_id) }
    }

    /// # Safety
    ///
    /// `path` must name a trusted native library implementing the GAFIME GPU
    /// C ABI. Loading an untrusted or layout-incompatible dynamic library can
    /// execute arbitrary initialization code or expose invalid function tables.
    pub unsafe fn load_cuda_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
    ) -> Result<Self, GpuSysError> {
        unsafe { Self::load_abi_from_path(path, device_id, GAFIME_BACKEND_CUDA) }
    }

    /// # Safety
    ///
    /// `path` must name a trusted native library implementing the GAFIME GPU
    /// C ABI. Loading an untrusted or layout-incompatible dynamic library can
    /// execute arbitrary initialization code or expose invalid function tables.
    pub unsafe fn load_rocm_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
    ) -> Result<Self, GpuSysError> {
        unsafe { Self::load_abi_from_path(path, device_id, GAFIME_BACKEND_ROCM) }
    }

    /// # Safety
    ///
    /// `path` must name a trusted native library implementing the GAFIME GPU
    /// C ABI. Loading an untrusted or layout-incompatible dynamic library can
    /// execute arbitrary initialization code or expose invalid function tables.
    pub unsafe fn load_metal_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
    ) -> Result<Self, GpuSysError> {
        unsafe { Self::load_abi_from_path(path, device_id, GAFIME_BACKEND_METAL) }
    }

    pub(crate) unsafe fn load_abi_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
        kind: BackendKind,
    ) -> Result<Self, GpuSysError> {
        let path = path.as_ref().to_path_buf();
        let library = unsafe { load_process_library(&path, kind)? };
        let functions = unsafe { load_function_table(&library)? };
        Self::from_function_table(kind, device_id, functions, Some(library), Some(path))
    }
}
