pub fn top_k_indices(values: &[f32], k: usize, descending: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| {
        let left_value = values[left];
        let right_value = values[right];
        let ordering = left_value
            .partial_cmp(&right_value)
            .unwrap_or(core::cmp::Ordering::Equal);
        if descending {
            ordering.reverse().then(left.cmp(&right))
        } else {
            ordering.then(left.cmp(&right))
        }
    });
    indices.truncate(k.min(indices.len()));
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_is_stable_on_ties() {
        assert_eq!(top_k_indices(&[0.5, 0.7, 0.7, 0.1], 2, true), vec![1, 2]);
    }
}
