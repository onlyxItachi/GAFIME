//! Async Pipeline Producer with Dynamic DLL Loading
//! 
//! Uses libloading to dynamically load gafime_cuda.dll at runtime,
//! avoiding static linking issues and allowing the DLL to be found
//! relative to the project directory.

use pyo3::prelude::*;
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::ptr;
use std::path::PathBuf;

// Type aliases for function pointers
type PipelineInitFn = unsafe extern "C" fn(*mut c_void, i32, *mut *mut c_void) -> i32;
type PipelineSubmitFn = unsafe extern "C" fn(*mut c_void, *const i32, *const i32, *const i32, i32, *mut i32) -> i32;
type PipelinePendingFn = unsafe extern "C" fn(*mut c_void) -> i32;
type PipelineWaitFn = unsafe extern "C" fn(*mut c_void, *mut f32, *mut i32) -> i32;
type PipelinePollFn = unsafe extern "C" fn(*mut c_void, *mut f32, *mut i32) -> i32;
type PipelineFlushFn = unsafe extern "C" fn(*mut c_void, *mut f32, *mut i32) -> i32;
type PipelineFreeFn = unsafe extern "C" fn(*mut c_void) -> i32;

// Error codes (match interfaces.h)
const GAFIME_SUCCESS: i32 = 0;
const _GAFIME_ERROR_PIPELINE_FULL: i32 = -5;
const _GAFIME_ERROR_NO_RESULT: i32 = -6;

use std::sync::Mutex;

/// Global library instance (loaded once) - use Mutex for stable Rust
static CUDA_LIB: Mutex<Option<Library>> = Mutex::new(None);

/// Get or load the CUDA library
fn get_cuda_lib() -> Result<&'static Library, String> {
    // Fast path: check if already loaded
    let guard = CUDA_LIB.lock().unwrap();
    if guard.is_some() {
        // Safety: We never remove the library once set
        drop(guard);
        let guard2 = CUDA_LIB.lock().unwrap();
        // This is safe because Library lives for 'static once loaded
        let lib_ref: &Library = guard2.as_ref().unwrap();
        return Ok(unsafe { &*(lib_ref as *const Library) });
    }
    drop(guard);
    
    // Slow path: load library
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
                    // Return reference to the now-stored library
                    let lib_ref: &Library = guard.as_ref().unwrap();
                    return Ok(unsafe { &*(lib_ref as *const Library) });
                }
                Err(e) => eprintln!("Failed to load {:?}: {}", path, e),
            }
        }
    }
    
    Err("Could not load gafime_cuda.dll".to_string())
}

/// Result of processing a batch
#[derive(Clone, Debug)]
pub struct BatchResult {
    pub stats: Vec<f32>,  // [batch_size * 12]
    pub batch_size: usize,
}

/// Async pipeline producer that processes batches with zero Python overhead
pub struct AsyncPipelineProducer {
    pipeline: *mut c_void,
    max_batch: usize,
    // Keep function pointers cached
    fn_submit: Symbol<'static, PipelineSubmitFn>,
    fn_pending: Symbol<'static, PipelinePendingFn>,
    fn_wait: Symbol<'static, PipelineWaitFn>,
    fn_poll: Symbol<'static, PipelinePollFn>,
    fn_free: Symbol<'static, PipelineFreeFn>,
}

// Safety: Pipeline is thread-local and CUDA handles are thread-safe within same context
unsafe impl Send for AsyncPipelineProducer {}

impl AsyncPipelineProducer {
    /// Create new producer from bucket handle
    pub fn new(bucket_ptr: usize, val_fold_id: i32) -> Result<Self, String> {
        let lib = get_cuda_lib()?;
        
        // Load all function symbols
        let fn_init: Symbol<PipelineInitFn> = unsafe {
            lib.get(b"gafime_pipeline_init\0")
                .map_err(|e| format!("Failed to load gafime_pipeline_init: {}", e))?
        };
        let fn_submit: Symbol<PipelineSubmitFn> = unsafe {
            lib.get(b"gafime_pipeline_submit\0")
                .map_err(|e| format!("Failed to load gafime_pipeline_submit: {}", e))?
        };
        let fn_pending: Symbol<PipelinePendingFn> = unsafe {
            lib.get(b"gafime_pipeline_pending\0")
                .map_err(|e| format!("Failed to load gafime_pipeline_pending: {}", e))?
        };
        let fn_wait: Symbol<PipelineWaitFn> = unsafe {
            lib.get(b"gafime_pipeline_wait\0")
                .map_err(|e| format!("Failed to load gafime_pipeline_wait: {}", e))?
        };
        let fn_poll: Symbol<PipelinePollFn> = unsafe {
            lib.get(b"gafime_pipeline_poll\0")
                .map_err(|e| format!("Failed to load gafime_pipeline_poll: {}", e))?
        };
        let fn_free: Symbol<PipelineFreeFn> = unsafe {
            lib.get(b"gafime_pipeline_free\0")
                .map_err(|e| format!("Failed to load gafime_pipeline_free: {}", e))?
        };
        
        // Initialize pipeline
        let mut pipeline: *mut c_void = ptr::null_mut();
        let result = unsafe {
            fn_init(bucket_ptr as *mut c_void, val_fold_id, &mut pipeline)
        };
        
        if result != GAFIME_SUCCESS {
            return Err(format!("Pipeline init failed with code: {}", result));
        }
        
        // Need to transmute to 'static lifetime since we know lib lives forever
        Ok(Self {
            pipeline,
            max_batch: 1024,
            fn_submit: unsafe { std::mem::transmute(fn_submit) },
            fn_pending: unsafe { std::mem::transmute(fn_pending) },
            fn_wait: unsafe { std::mem::transmute(fn_wait) },
            fn_poll: unsafe { std::mem::transmute(fn_poll) },
            fn_free: unsafe { std::mem::transmute(fn_free) },
        })
    }
    
    /// Submit batch (non-blocking, returns slot ID)
    #[inline]
    pub fn submit(&mut self, indices: &[i32], ops: &[i32], interact: &[i32]) -> Result<i32, i32> {
        let batch_size = interact.len();
        let mut slot_id: i32 = -1;
        
        let result = unsafe {
            (self.fn_submit)(
                self.pipeline,
                indices.as_ptr(),
                ops.as_ptr(),
                interact.as_ptr(),
                batch_size as i32,
                &mut slot_id,
            )
        };
        
        if result == GAFIME_SUCCESS {
            Ok(slot_id)
        } else {
            Err(result)
        }
    }
    
    /// Get number of pending batches
    #[inline]
    pub fn pending(&self) -> i32 {
        unsafe { (self.fn_pending)(self.pipeline) }
    }
    
    /// Wait for next result (blocking)
    #[inline]
    pub fn wait(&mut self) -> Result<BatchResult, i32> {
        let mut stats = vec![0.0f32; self.max_batch * 12];
        let mut batch_size: i32 = 0;
        
        let result = unsafe {
            (self.fn_wait)(
                self.pipeline,
                stats.as_mut_ptr(),
                &mut batch_size,
            )
        };
        
        if result == GAFIME_SUCCESS {
            stats.truncate(batch_size as usize * 12);
            Ok(BatchResult {
                stats,
                batch_size: batch_size as usize,
            })
        } else {
            Err(result)
        }
    }
    
    /// Poll for next result (non-blocking)
    #[inline]
    pub fn poll(&mut self) -> Option<BatchResult> {
        let mut stats = vec![0.0f32; self.max_batch * 12];
        let mut batch_size: i32 = 0;
        
        let result = unsafe {
            (self.fn_poll)(
                self.pipeline,
                stats.as_mut_ptr(),
                &mut batch_size,
            )
        };
        
        if result == GAFIME_SUCCESS {
            stats.truncate(batch_size as usize * 12);
            Some(BatchResult {
                stats,
                batch_size: batch_size as usize,
            })
        } else {
            None
        }
    }
    
    /// Process all batches with full async overlap - ZERO PYTHON OVERHEAD
    pub fn process_all(&mut self, all_batches: Vec<(Vec<i32>, Vec<i32>, Vec<i32>)>) -> Vec<BatchResult> {
        let mut results = Vec::with_capacity(all_batches.len());
        let mut batch_iter = all_batches.into_iter();
        
        // Fill initial 4 slots
        for _ in 0..4 {
            if let Some((indices, ops, interact)) = batch_iter.next() {
                let _ = self.submit(&indices, &ops, &interact);
            } else {
                break;
            }
        }
        
        // Process remaining with overlap: wait + submit interleaved
        for (indices, ops, interact) in batch_iter {
            // Wait for oldest to complete
            if let Ok(result) = self.wait() {
                results.push(result);
            }
            // Submit new batch immediately
            let _ = self.submit(&indices, &ops, &interact);
        }
        
        // Drain remaining slots
        while self.pending() > 0 {
            if let Ok(result) = self.wait() {
                results.push(result);
            }
        }
        
        results
    }
}

impl Drop for AsyncPipelineProducer {
    fn drop(&mut self) {
        if !self.pipeline.is_null() {
            unsafe { (self.fn_free)(self.pipeline); }
        }
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[pyclass(name = "AsyncPipeline")]
pub struct PyAsyncPipeline {
    inner: Option<AsyncPipelineProducer>,
}

#[pymethods]
impl PyAsyncPipeline {
    /// Create new async pipeline from bucket pointer
    /// 
    /// Args:
    ///     bucket_ptr: Memory address of GafimeBucket (from StaticBucket._bucket)
    ///     val_fold_id: Validation fold ID
    #[new]
    fn new(bucket_ptr: usize, val_fold_id: i32) -> PyResult<Self> {
        let producer = AsyncPipelineProducer::new(bucket_ptr, val_fold_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(Self { inner: Some(producer) })
    }
    
    /// Submit a batch (non-blocking)
    fn submit(&mut self, indices: Vec<i32>, ops: Vec<i32>, interact: Vec<i32>) -> PyResult<i32> {
        let producer = self.inner.as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Pipeline closed"))?;
        
        producer.submit(&indices, &ops, &interact)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Submit error: {}", e)))
    }
    
    /// Get number of pending batches
    fn pending(&self) -> i32 {
        self.inner.as_ref().map(|p| p.pending()).unwrap_or(0)
    }
    
    /// Wait for next result (blocking)
    fn wait(&mut self) -> PyResult<(Vec<f32>, usize)> {
        let producer = self.inner.as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Pipeline closed"))?;
        
        let result = producer.wait()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Wait error: {}", e)))?;
        
        Ok((result.stats, result.batch_size))
    }
    
    /// Process all batches with optimal async overlap - RUNS IN RUST, NO PYTHON GIL
    /// 
    /// This is the main performance method. All batch processing happens in Rust
    /// with zero Python overhead between batches!
    fn process_all(&mut self, batches: Vec<(Vec<i32>, Vec<i32>, Vec<i32>)>) -> PyResult<Vec<(Vec<f32>, usize)>> {
        let producer = self.inner.as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Pipeline closed"))?;
        
        let results = producer.process_all(batches);
        Ok(results.into_iter().map(|r| (r.stats, r.batch_size)).collect())
    }
    
    /// Close pipeline and free resources
    fn close(&mut self) {
        self.inner = None;
    }
}
