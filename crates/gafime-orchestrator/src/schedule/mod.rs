use gafime_types::BackendKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleDecision {
    pub backend_kind: BackendKind,
    pub matrix_resident: bool,
    pub graph_requested: bool,
}

impl ScheduleDecision {
    pub fn resident_graph(backend_kind: BackendKind) -> Self {
        Self {
            backend_kind,
            matrix_resident: true,
            graph_requested: true,
        }
    }
}
