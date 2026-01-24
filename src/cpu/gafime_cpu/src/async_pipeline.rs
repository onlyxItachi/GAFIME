//! Async Pipeline Producer
//! 
//! Rust-native async producer that calls CUDA pipeline API directly,
//! eliminating Python GIL and ctypes overhead.

use pyo3::prelude::*;
use std::ffi::c_void;
use std::ptr;

// FFI declarations for CUDA pipeline
#[link(name = "gafime_cuda")]
extern "C" {
    fn gafime_pipeline_init(
        bucket: *mut c_void,
        val_fold_id: i32,
        pipeline_out: *mut *mut c_void,
    ) -> i32;
    
    fn gafime_pipeline_submit(
        pipeline: *mut c_void,
        h_indices: *const i32,
        h_ops: *const i32,
        h_interact: *const i32,
        batch_size: i32,
        slot_id_out: *mut i32,
    ) -> i32;
    
    fn gafime_pipeline_pending(pipeline: *mut c_void) -> i32;
    
    fn gafime_pipeline_wait(
        pipeline: *mut c_void,
        h_stats_out: *mut f32,
        batch_size_out: *mut i32,
    ) -> i32;
    
    fn gafime_pipeline_poll(
        pipeline: *mut c_void,
        h_stats_out: *mut f32,
        batch_size_out: *mut i32,
    ) -> i32;
    
    fn gafime_pipeline_flush(
        pipeline: *mut c_void,
        h_all_stats_out: *mut f32,
        total_batch_size_out: *mut i32,
    ) -> i32;
    
    fn gafime_pipeline_free(pipeline: *mut c_void) -> i32;
}

// Error codes (match interfaces.h)
const GAFIME_SUCCESS: i32 = 0;
const GAFIME_ERROR_PIPELINE_FULL: i32 = -5;
const GAFIME_ERROR_NO_RESULT: i32 = -6;

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
}

// Safety: Pipeline is thread-local and CUDA handles are thread-safe
unsafe impl Send for AsyncPipelineProducer {}

impl AsyncPipelineProducer {
    /// Create new producer from bucket handle
    pub unsafe fn new(bucket_ptr: usize, val_fold_id: i32) -> Result<Self, String> {
        let mut pipeline: *mut c_void = ptr::null_mut();
        let result = gafime_pipeline_init(
            bucket_ptr as *mut c_void,
            val_fold_id,
            &mut pipeline,
        );
        
        if result != GAFIME_SUCCESS {
            return Err(format!("Pipeline init failed: {}", result));
        }
        
        Ok(Self {
            pipeline,
            max_batch: 1024,
        })
    }
    
    /// Submit batch (non-blocking, returns slot ID)
    pub fn submit(&mut self, indices: &[i32], ops: &[i32], interact: &[i32]) -> Result<i32, i32> {
        let batch_size = interact.len();
        let mut slot_id: i32 = -1;
        
        let result = unsafe {
            gafime_pipeline_submit(
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
    pub fn pending(&self) -> i32 {
        unsafe { gafime_pipeline_pending(self.pipeline) }
    }
    
    /// Wait for next result (blocking)
    pub fn wait(&mut self) -> Result<BatchResult, i32> {
        let mut stats = vec![0.0f32; self.max_batch * 12];
        let mut batch_size: i32 = 0;
        
        let result = unsafe {
            gafime_pipeline_wait(
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
    pub fn poll(&mut self) -> Option<BatchResult> {
        let mut stats = vec![0.0f32; self.max_batch * 12];
        let mut batch_size: i32 = 0;
        
        let result = unsafe {
            gafime_pipeline_poll(
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
    
    /// Process all batches with full async overlap
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
        
        // Process remaining with overlap
        for (indices, ops, interact) in batch_iter {
            // Wait for oldest to complete
            if let Ok(result) = self.wait() {
                results.push(result);
            }
            // Submit new batch
            let _ = self.submit(&indices, &ops, &interact);
        }
        
        // Drain remaining
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
            unsafe { gafime_pipeline_free(self.pipeline); }
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
        let producer = unsafe { 
            AsyncPipelineProducer::new(bucket_ptr, val_fold_id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?
        };
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
    
    /// Process all batches with optimal async overlap
    /// 
    /// This runs entirely in Rust with no Python GIL overhead!
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
