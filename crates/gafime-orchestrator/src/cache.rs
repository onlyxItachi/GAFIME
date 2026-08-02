use gafime_types::{BackendKind, GafimeShapeHint, PrecisionProfile};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutotuneCacheEntry {
    pub backend_kind: BackendKind,
    pub precision: PrecisionProfile,
    pub device_key: String,
    pub max_arity: u32,
    pub shape_hint: GafimeShapeHint,
}
