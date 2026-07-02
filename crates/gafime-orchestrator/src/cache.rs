use gafime_types::{BackendKind, GafimeShapeHint};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutotuneCacheEntry {
    pub backend_kind: BackendKind,
    pub device_key: String,
    pub max_arity: u32,
    pub shape_hint: GafimeShapeHint,
}
