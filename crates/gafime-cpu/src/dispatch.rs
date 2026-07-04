//! Backward-compatible re-export of the CPU SIMD dispatch surface.
//!
//! New code should prefer `crate::simd`; this module remains so existing
//! callers using `gafime_cpu::dispatch::*` keep compiling.

pub use crate::simd::*;
