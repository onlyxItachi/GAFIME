mod probe;
#[cfg(test)]
mod tests;

use probe::{run, EvidenceValue, GraphEdge, LabeledRows, ProbeInput, Selector, SparseGraph};

fn main() {
    let original = vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        vec![1.0, 0.0, 2.0, 1.0, 3.0, 2.0],
    ];
    let aligned_view = vec![
        vec![0.1, 0.9, 2.1, 2.9, 4.1, 4.9],
        vec![1.1, 0.1, 2.1, 0.9, 3.1, 1.9],
    ];
    let graph = SparseGraph::new(vec![
        GraphEdge::new(0, 1, 1.0),
        GraphEdge::new(1, 2, 1.0),
        GraphEdge::new(2, 3, 1.0),
        GraphEdge::new(3, 4, 1.0),
        GraphEdge::new(4, 5, 1.0),
    ]);
    let labels = LabeledRows::new(vec![0, 2, 4, 5], vec![0.0, 1.8, 4.2, 4.9]);
    let input = ProbeInput::new(original, aligned_view, 1, Some(graph), Some(labels))
        .expect("demo fixture is validated");
    let report = run(&input, Selector::core_mixed()).expect("Core mixed is the only probe route");
    print_report(&report.rows);
}

fn print_report(rows: &[probe::EvidenceRow]) {
    println!(
        "{{\"scope\":\"Core mixed feasibility probe; experimental candidates and evidence only\",\"backend\":\"core\",\"precision\":\"mixed\",\"pointwise_storage\":\"f32\",\"statistical_reductions\":\"f64\",\"public_runtime_api\":false,\"results\":["
    );
    for (index, row) in rows.iter().enumerate() {
        let comma = if index + 1 == rows.len() { "" } else { "," };
        println!(
            "{{\"candidate\":\"{}\",\"redundancy_abs_pearson\":{},\"paired_view_consistency\":{},\"graph_normalized_dirichlet\":{},\"hybrid_labeled_pearson\":{}}}{comma}",
            row.candidate,
            json_value(&row.redundancy_abs_pearson),
            json_value(&row.paired_view_consistency),
            json_value(&row.graph_normalized_dirichlet),
            json_value(&row.hybrid_labeled_pearson),
        );
    }
    println!("]}}");
}

fn json_value(value: &EvidenceValue) -> String {
    match value {
        EvidenceValue::Value(value) => value.to_string(),
        EvidenceValue::Unavailable(reason) => format!("\"unavailable:{}\"", reason.as_str()),
    }
}
