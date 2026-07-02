use gafime_types::{
    BackendKind, GafimeShapeHint, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL,
    GAFIME_BACKEND_ROCM,
};

pub fn default_shape_hint(backend_kind: BackendKind, arity: u32) -> GafimeShapeHint {
    match backend_kind {
        GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM => gpu_shape_hint(arity),
        GAFIME_BACKEND_METAL => metal_shape_hint(arity),
        GAFIME_BACKEND_CPU | _ => cpu_shape_hint(),
    }
}

fn gpu_shape_hint(arity: u32) -> GafimeShapeHint {
    GafimeShapeHint {
        threads_per_block: 256,
        items_per_thread: arity.max(1),
        blocks_per_sm: 2,
        min_blocks: 1,
        occupancy_target_pct: 90,
        ..Default::default()
    }
}

fn metal_shape_hint(arity: u32) -> GafimeShapeHint {
    GafimeShapeHint {
        threads_per_block: 128,
        items_per_thread: arity.max(1),
        blocks_per_sm: 1,
        min_blocks: 1,
        occupancy_target_pct: 80,
        ..Default::default()
    }
}

fn cpu_shape_hint() -> GafimeShapeHint {
    GafimeShapeHint {
        threads_per_block: 1,
        items_per_thread: 1,
        blocks_per_sm: 1,
        min_blocks: 1,
        occupancy_target_pct: 0,
        ..Default::default()
    }
}
