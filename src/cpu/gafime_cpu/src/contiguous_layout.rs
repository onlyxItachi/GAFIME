//! Contiguous Data Layout Module
//!
//! Prepares data in column-major contiguous layout for optimal GPU memory coalescing.
//! Layout: [Feature0][Feature1]...[FeatureN][Target]
//!         ^         ^             ^         ^
//!         0         N             N*k       N*(k+1)
//!
//! This module also provides FFI bindings to the CUDA ContiguousBucket API.

use pyo3::prelude::*;
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;
use std::path::PathBuf;

// FFI type aliases
type AllocFn = unsafe extern "C" fn(i32, i32, *mut *mut c_void) -> i32;
type UploadFn = unsafe extern "C" fn(*mut c_void, *const f32, *const u8) -> i32;
type ComputeFn = unsafe extern "C" fn(*mut c_void, i32, i32, i32, i32, i32, i32, *mut f32) -> i32;
type FreeFn = unsafe extern "C" fn(*mut c_void) -> i32;
type InfoFn = unsafe extern "C" fn(*mut c_void, *mut i32, *mut i32) -> i32;

const GAFIME_SUCCESS: i32 = 0;

/// Global library instance
static CUDA_LIB: Mutex<Option<Library>> = Mutex::new(None);

fn get_cuda_lib() -> Result<&'static Library, String> {
    let guard = CUDA_LIB.lock().unwrap();
    if guard.is_some() {
        drop(guard);
        let guard2 = CUDA_LIB.lock().unwrap();
        let lib_ref: &Library = guard2.as_ref().unwrap();
        return Ok(unsafe { &*(lib_ref as *const Library) });
    }
    drop(guard);
    
    let paths = vec![
        PathBuf::from("gafime_cuda.dll"),
        PathBuf::from("C:/Users/Hamza/Desktop/GAFIME/gafime_cuda.dll"),
    ];
    
    for path in &paths {
        if path.exists() {
            match unsafe { Library::new(path) } {
                Ok(lib) => {
                    let mut guard = CUDA_LIB.lock().unwrap();
                    *guard = Some(lib);
                    let lib_ref: &Library = guard.as_ref().unwrap();
                    return Ok(unsafe { &*(lib_ref as *const Library) });
                }
                Err(e) => eprintln!("Failed to load {:?}: {}", path, e),
            }
        }
    }
    
    Err("Could not load gafime_cuda.dll".to_string())
}

/// Contiguous data layout builder
/// 
/// Prepares feature data in column-major layout for optimal GPU access.
pub struct ContiguousLayoutBuilder {
    n_samples: usize,
    n_features: usize,
    data: Vec<f32>,      // [n_features * n_samples + n_samples] floats
    mask: Vec<u8>,       // [n_samples] bytes
    feature_count: usize, // Number of features added so far
    has_target: bool,
}

impl ContiguousLayoutBuilder {
    /// Create new builder
    pub fn new(n_samples: usize, n_features: usize) -> Self {
        let total_floats = (n_features + 1) * n_samples;  // features + target
        Self {
            n_samples,
            n_features,
            data: vec![0.0f32; total_floats],
            mask: vec![0u8; n_samples],
            feature_count: 0,
            has_target: false,
        }
    }
    
    /// Add a feature column
    pub fn add_feature(&mut self, feature: &[f32]) -> Result<(), String> {
        if feature.len() != self.n_samples {
            return Err(format!(
                "Feature length {} doesn't match n_samples {}",
                feature.len(), self.n_samples
            ));
        }
        if self.feature_count >= self.n_features {
            return Err("All features already added".to_string());
        }
        
        // Copy to correct offset: feature_idx * n_samples
        let offset = self.feature_count * self.n_samples;
        self.data[offset..offset + self.n_samples].copy_from_slice(feature);
        self.feature_count += 1;
        
        Ok(())
    }
    
    /// Set target column
    pub fn set_target(&mut self, target: &[f32]) -> Result<(), String> {
        if target.len() != self.n_samples {
            return Err(format!(
                "Target length {} doesn't match n_samples {}",
                target.len(), self.n_samples
            ));
        }
        
        // Target goes after all features: n_features * n_samples
        let offset = self.n_features * self.n_samples;
        self.data[offset..offset + self.n_samples].copy_from_slice(target);
        self.has_target = true;
        
        Ok(())
    }
    
    /// Set fold mask
    pub fn set_mask(&mut self, mask: &[u8]) -> Result<(), String> {
        if mask.len() != self.n_samples {
            return Err(format!(
                "Mask length {} doesn't match n_samples {}",
                mask.len(), self.n_samples
            ));
        }
        
        self.mask.copy_from_slice(mask);
        Ok(())
    }
    
    /// Check if ready for upload
    pub fn is_complete(&self) -> bool {
        self.feature_count == self.n_features && self.has_target
    }
    
    /// Get raw data pointer (for FFI)
    pub fn data_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }
    
    /// Get raw mask pointer (for FFI)
    pub fn mask_ptr(&self) -> *const u8 {
        self.mask.as_ptr()
    }
    
    /// Get total data size in bytes
    pub fn data_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Contiguous bucket handle for GPU execution
pub struct ContiguousBucket {
    handle: *mut c_void,
    n_samples: i32,
    n_features: i32,
    fn_compute: Symbol<'static, ComputeFn>,
    fn_free: Symbol<'static, FreeFn>,
}

unsafe impl Send for ContiguousBucket {}

impl ContiguousBucket {
    /// Create and upload data to GPU
    pub fn new(layout: &ContiguousLayoutBuilder) -> Result<Self, String> {
        if !layout.is_complete() {
            return Err("Layout not complete: missing features or target".to_string());
        }
        
        let lib = get_cuda_lib()?;
        
        // Load functions
        let fn_alloc: Symbol<AllocFn> = unsafe {
            lib.get(b"gafime_contiguous_bucket_alloc\0")
                .map_err(|e| format!("Failed to load alloc: {}", e))?
        };
        let fn_upload: Symbol<UploadFn> = unsafe {
            lib.get(b"gafime_contiguous_bucket_upload\0")
                .map_err(|e| format!("Failed to load upload: {}", e))?
        };
        let fn_compute: Symbol<ComputeFn> = unsafe {
            lib.get(b"gafime_contiguous_bucket_compute\0")
                .map_err(|e| format!("Failed to load compute: {}", e))?
        };
        let fn_free: Symbol<FreeFn> = unsafe {
            lib.get(b"gafime_contiguous_bucket_free\0")
                .map_err(|e| format!("Failed to load free: {}", e))?
        };
        
        // Allocate bucket
        let mut handle: *mut c_void = ptr::null_mut();
        let result = unsafe {
            fn_alloc(
                layout.n_samples as i32,
                layout.n_features as i32,
                &mut handle,
            )
        };
        
        if result != GAFIME_SUCCESS {
            return Err(format!("Bucket alloc failed: {}", result));
        }
        
        // Upload data
        let result = unsafe {
            fn_upload(handle, layout.data_ptr(), layout.mask_ptr())
        };
        
        if result != GAFIME_SUCCESS {
            unsafe { fn_free(handle); }
            return Err(format!("Upload failed: {}", result));
        }
        
        Ok(Self {
            handle,
            n_samples: layout.n_samples as i32,
            n_features: layout.n_features as i32,
            fn_compute: unsafe { std::mem::transmute(fn_compute) },
            fn_free: unsafe { std::mem::transmute(fn_free) },
        })
    }
    
    /// Compute single interaction
    pub fn compute(
        &self,
        feature_a: i32,
        feature_b: i32,
        op_a: i32,
        op_b: i32,
        interact_type: i32,
        val_fold_id: i32,
    ) -> Result<[f32; 12], i32> {
        let mut stats = [0.0f32; 12];
        
        let result = unsafe {
            (self.fn_compute)(
                self.handle,
                feature_a,
                feature_b,
                op_a,
                op_b,
                interact_type,
                val_fold_id,
                stats.as_mut_ptr(),
            )
        };
        
        if result == GAFIME_SUCCESS {
            Ok(stats)
        } else {
            Err(result)
        }
    }
}

impl Drop for ContiguousBucket {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.fn_free)(self.handle); }
        }
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[pyclass(name = "ContiguousLayout")]
pub struct PyContiguousLayout {
    inner: ContiguousLayoutBuilder,
}

#[pymethods]
impl PyContiguousLayout {
    /// Create new contiguous layout builder
    #[new]
    fn new(n_samples: usize, n_features: usize) -> Self {
        Self {
            inner: ContiguousLayoutBuilder::new(n_samples, n_features),
        }
    }
    
    /// Add a feature column
    fn add_feature(&mut self, feature: Vec<f32>) -> PyResult<()> {
        self.inner.add_feature(&feature)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }
    
    /// Set target column
    fn set_target(&mut self, target: Vec<f32>) -> PyResult<()> {
        self.inner.set_target(&target)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }
    
    /// Set fold mask
    fn set_mask(&mut self, mask: Vec<u8>) -> PyResult<()> {
        self.inner.set_mask(&mask)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }
    
    /// Check if layout is complete
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    
    /// Get info
    fn info(&self) -> (usize, usize, usize, bool) {
        (
            self.inner.n_samples,
            self.inner.n_features,
            self.inner.feature_count,
            self.inner.has_target,
        )
    }
}

#[pyclass(name = "ContiguousBucket")]
pub struct PyContiguousBucket {
    inner: Option<ContiguousBucket>,
}

#[pymethods]
impl PyContiguousBucket {
    /// Create bucket from layout and upload to GPU
    #[new]
    fn new(layout: &PyContiguousLayout) -> PyResult<Self> {
        let bucket = ContiguousBucket::new(&layout.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { inner: Some(bucket) })
    }
    
    /// Compute single interaction
    fn compute(
        &self,
        feature_a: i32,
        feature_b: i32,
        op_a: i32,
        op_b: i32,
        interact_type: i32,
        val_fold_id: i32,
    ) -> PyResult<Vec<f32>> {
        let bucket = self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Bucket closed"))?;
        
        let stats = bucket.compute(feature_a, feature_b, op_a, op_b, interact_type, val_fold_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Compute error: {}", e)))?;
        
        Ok(stats.to_vec())
    }
    
    /// Compute a batch of interactions
    fn compute_batch(
        &self,
        feature_a: Vec<i32>,
        feature_b: Vec<i32>,
        op_a: Vec<i32>,
        op_b: Vec<i32>,
        interact_type: Vec<i32>,
        val_fold_id: i32,
    ) -> PyResult<Vec<Vec<f32>>> {
        let bucket = self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Bucket closed"))?;
            
        let n = feature_a.len();
        if feature_b.len() != n || op_a.len() != n || op_b.len() != n || interact_type.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("All input/op lists must have equal length"));
        }
        
        let mut results = Vec::with_capacity(n);
        
        for i in 0..n {
            // Unsafe call wrapper
            let stats = bucket.compute(
                feature_a[i], 
                feature_b[i], 
                op_a[i], 
                op_b[i], 
                interact_type[i], 
                val_fold_id
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Compute error at index {}: {}", i, e)))?;
            
            results.push(stats.to_vec());
        }
        
        Ok(results)
    }

    /// Close bucket and free GPU memory
    fn close(&mut self) {
        self.inner = None;
    }
}
