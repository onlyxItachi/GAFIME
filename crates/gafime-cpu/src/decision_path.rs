//! Decision-path (GBDT-method) split finding for the `decision_path` family.
//!
//! Core primitive: the CART/GBDT variance-reduction best-split of a feature vs
//! the target (or residual). A decision_path candidate is a conjunction of such
//! splits (a root→leaf path); its materialized feature is the membership
//! indicator of that region, scored by the continuous engine. depth-k recursion
//! and residual boosting build on `best_variance_split`.

/// A single threshold split and its variance-reduction gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Split {
    pub threshold: f32,
    pub gain: f32,
}

/// Find the threshold on `feature` that maximizes variance reduction of `y`.
/// O(n log n): sort by feature, sweep boundaries maintaining running
/// left/right (sum, sum²) for incremental child variances. Returns `None` for a
/// constant feature, fewer than 2 finite pairs, or zero parent variance.
pub fn best_variance_split(feature: &[f32], y: &[f32]) -> Option<Split> {
    let n = feature.len().min(y.len());
    let mut pairs: Vec<(f32, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let (x, t) = (feature[i], y[i]);
        if x.is_finite() && t.is_finite() {
            pairs.push((x, t as f64));
        }
    }
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let total = pairs.len() as f64;
    let sum_all: f64 = pairs.iter().map(|p| p.1).sum();
    let sum2_all: f64 = pairs.iter().map(|p| p.1 * p.1).sum();
    let parent_var = sum2_all / total - (sum_all / total).powi(2);
    if parent_var <= 0.0 {
        return None;
    }

    let mut left_sum = 0.0f64;
    let mut left_sum2 = 0.0f64;
    let mut best: Option<Split> = None;
    for i in 0..pairs.len() - 1 {
        left_sum += pairs[i].1;
        left_sum2 += pairs[i].1 * pairs[i].1;
        if pairs[i].0 == pairs[i + 1].0 {
            continue; // can't split between equal feature values
        }
        let n_left = (i + 1) as f64;
        let n_right = total - n_left;
        let var_left = (left_sum2 / n_left - (left_sum / n_left).powi(2)).max(0.0);
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        let var_right = (right_sum2 / n_right - (right_sum / n_right).powi(2)).max(0.0);
        let weighted = (n_left * var_left + n_right * var_right) / total;
        let gain = (parent_var - weighted) as f32;
        let threshold = 0.5 * (pairs[i].0 + pairs[i + 1].0);
        if best.map_or(true, |b| gain > b.gain) {
            best = Some(Split { threshold, gain });
        }
    }
    best
}

/// Materialize a split's membership indicator into `out`: 1.0 where
/// `feature >= threshold`, 0.0 where `< threshold`, NaN where the feature is NaN
/// (so the finite-pair scoring skips it).
pub fn split_indicator(feature: &[f32], threshold: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(feature.len());
    for &x in feature {
        out.push(if x.is_nan() {
            f32::NAN
        } else if x >= threshold {
            1.0
        } else {
            0.0
        });
    }
}

/// Side of a threshold split taken by a path node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitSign {
    /// `feature <= threshold`
    Le,
    /// `feature > threshold`
    Gt,
}

/// One node of a root->leaf conjunction path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathNode {
    pub feature: u32,
    pub threshold: f32,
    pub sign: SplitSign,
}

/// A decision-path candidate: a conjunction of threshold conditions (an
/// axis-aligned region), with the variance-reduction proxy `gain`, the number of
/// training rows it covers (`support`), and the boosting `round` that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionPath {
    pub nodes: Vec<PathNode>,
    pub gain: f32,
    pub support: u32,
    pub round: u32,
}

/// Depth-k recursion + residual-boosting controls for `find_decision_paths`.
#[derive(Clone, Copy, Debug)]
pub struct DecisionPathParams {
    pub max_depth: u32,
    pub rounds: u32,
    pub max_paths: u32,
    pub max_bins: u32,
    pub min_leaf: u32,
    pub learning_rate: f32,
}

/// Best variance-reduction split of a pre-gathered `(value, target)` subset,
/// enforcing at least `min_leaf` rows on each side. `pairs` is sorted in place.
fn best_split_subset(pairs: &mut [(f32, f64)], min_leaf: usize, max_bins: u32) -> Option<Split> {
    let n = pairs.len();
    let min_leaf = min_leaf.max(1);
    if n < 2 * min_leaf {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    let total = n as f64;
    let sum_all: f64 = pairs.iter().map(|p| p.1).sum();
    let sum2_all: f64 = pairs.iter().map(|p| p.1 * p.1).sum();
    let parent_var = sum2_all / total - (sum_all / total).powi(2);
    if parent_var <= 0.0 {
        return None;
    }
    let mut best: Option<Split> = None;
    let mut consider = |i: usize, left_sum: f64, left_sum2: f64| {
        let n_left = i + 1usize;
        let n_right = n - n_left;
        if n_left < min_leaf || n_right < min_leaf {
            return;
        }
        let nl = n_left as f64;
        let nr = n_right as f64;
        let var_left = (left_sum2 / nl - (left_sum / nl).powi(2)).max(0.0);
        let right_sum = sum_all - left_sum;
        let right_sum2 = sum2_all - left_sum2;
        let var_right = (right_sum2 / nr - (right_sum / nr).powi(2)).max(0.0);
        let weighted = (nl * var_left + nr * var_right) / total;
        let gain = (parent_var - weighted) as f32;
        let threshold = 0.5 * (pairs[i].0 + pairs[i + 1].0);
        if best.map_or(true, |b| gain > b.gain) {
            best = Some(Split { threshold, gain });
        }
    };

    if max_bins == 0 {
        let mut left_sum = 0.0f64;
        let mut left_sum2 = 0.0f64;
        for i in 0..n - 1 {
            left_sum += pairs[i].1;
            left_sum2 += pairs[i].1 * pairs[i].1;
            if pairs[i].0 != pairs[i + 1].0 {
                consider(i, left_sum, left_sum2);
            }
        }
        return best;
    }

    let mut prefix_sum = Vec::with_capacity(n + 1);
    let mut prefix_sum2 = Vec::with_capacity(n + 1);
    prefix_sum.push(0.0f64);
    prefix_sum2.push(0.0f64);
    for pair in pairs.iter() {
        prefix_sum.push(prefix_sum.last().copied().unwrap_or_default() + pair.1);
        prefix_sum2.push(prefix_sum2.last().copied().unwrap_or_default() + pair.1 * pair.1);
    }
    let valid = (0..n - 1)
        .filter(|&i| pairs[i].0 != pairs[i + 1].0 && i + 1 >= min_leaf && n - (i + 1) >= min_leaf)
        .collect::<Vec<_>>();
    let cap = max_bins as usize;
    if valid.len() <= cap {
        for i in valid {
            consider(i, prefix_sum[i + 1], prefix_sum2[i + 1]);
        }
    } else if cap == 1 {
        let i = valid[valid.len() / 2];
        consider(i, prefix_sum[i + 1], prefix_sum2[i + 1]);
    } else {
        for position in 0..cap {
            let i = valid[position * (valid.len() - 1) / (cap - 1)];
            consider(i, prefix_sum[i + 1], prefix_sum2[i + 1]);
        }
    }
    best
}

struct LeafAcc {
    path: Vec<PathNode>,
    indices: Vec<usize>,
    mean: f32,
}

fn column<'a>(columns: &'a [f32], rows: usize, feature: usize) -> &'a [f32] {
    &columns[feature * rows..(feature + 1) * rows]
}

fn leaf_mean(residual: &[f32], indices: &[usize]) -> f32 {
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for &i in indices {
        let r = residual[i];
        if r.is_finite() {
            sum += r as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Greedy CART growth against a fixed `residual`. Emits one `LeafAcc` per
/// root->leaf region (leaves partition the rows). `prefix` carries the current
/// conjunction; `scratch` is reused to gather per-feature subsets.
#[allow(clippy::too_many_arguments)]
fn grow(
    columns: &[f32],
    rows: usize,
    cols: usize,
    residual: &[f32],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_leaf: usize,
    max_bins: u32,
    prefix: &mut Vec<PathNode>,
    leaves: &mut Vec<LeafAcc>,
) {
    if depth >= max_depth || indices.len() < 2 * min_leaf {
        leaves.push(LeafAcc {
            path: prefix.clone(),
            mean: leaf_mean(residual, indices),
            indices: indices.to_vec(),
        });
        return;
    }

    let mut best: Option<(u32, Split)> = None;
    let mut pairs: Vec<(f32, f64)> = Vec::with_capacity(indices.len());
    for feature in 0..cols {
        let col = column(columns, rows, feature);
        pairs.clear();
        for &i in indices {
            let x = col[i];
            let r = residual[i];
            if x.is_finite() && r.is_finite() {
                pairs.push((x, r as f64));
            }
        }
        if let Some(split) = best_split_subset(&mut pairs, min_leaf, max_bins) {
            if best.map_or(true, |(_, current)| split.gain > current.gain) {
                best = Some((feature as u32, split));
            }
        }
    }

    let (feature, split) = match best {
        Some(found) if found.1.gain > 0.0 => found,
        _ => {
            leaves.push(LeafAcc {
                path: prefix.clone(),
                mean: leaf_mean(residual, indices),
                indices: indices.to_vec(),
            });
            return;
        }
    };

    let col = column(columns, rows, feature as usize);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in indices {
        // NaN feature values follow the ">" branch deterministically; they carry
        // no split information but must land somewhere consistent.
        if col[i] <= split.threshold {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    if left.len() < min_leaf || right.len() < min_leaf {
        leaves.push(LeafAcc {
            path: prefix.clone(),
            mean: leaf_mean(residual, indices),
            indices: indices.to_vec(),
        });
        return;
    }

    prefix.push(PathNode {
        feature,
        threshold: split.threshold,
        sign: SplitSign::Le,
    });
    grow(
        columns,
        rows,
        cols,
        residual,
        &left,
        depth + 1,
        max_depth,
        min_leaf,
        max_bins,
        prefix,
        leaves,
    );
    prefix.pop();

    prefix.push(PathNode {
        feature,
        threshold: split.threshold,
        sign: SplitSign::Gt,
    });
    grow(
        columns,
        rows,
        cols,
        residual,
        &right,
        depth + 1,
        max_depth,
        min_leaf,
        max_bins,
        prefix,
        leaves,
    );
    prefix.pop();
}

fn path_key(nodes: &[PathNode]) -> String {
    let mut parts: Vec<String> = nodes
        .iter()
        .map(|n| {
            let sign = match n.sign {
                SplitSign::Le => 'L',
                SplitSign::Gt => 'G',
            };
            // Quantize the threshold so numerically-identical splits dedup.
            format!("{}:{}:{:.5}", n.feature, sign, n.threshold)
        })
        .collect();
    parts.sort();
    parts.join("&")
}

/// Discover decision-path conjunctions via depth-k greedy CART with residual
/// boosting. Each boosting round fits a depth-`max_depth` tree to the current
/// residual, records its leaf regions as candidate paths, then subtracts
/// `learning_rate * leaf_mean` from the residual (standard gradient boosting on
/// squared error). Paths are deduplicated and the top `max_paths` by `gain` are
/// returned. `columns` is column-major (`columns[f*rows + i]`).
pub fn find_decision_paths(
    columns: &[f32],
    rows: usize,
    cols: usize,
    target: &[f32],
    params: &DecisionPathParams,
) -> Vec<DecisionPath> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let max_depth = params.max_depth.max(1) as usize;
    let min_leaf = params.min_leaf.max(1) as usize;
    let rounds = params.rounds.max(1);
    let learning_rate = if params.learning_rate > 0.0 {
        params.learning_rate
    } else {
        1.0
    };

    let mut residual: Vec<f32> = target.to_vec();
    let mut collected: Vec<DecisionPath> = Vec::new();
    let all: Vec<usize> = (0..rows).collect();

    for round in 0..rounds {
        let mut leaves: Vec<LeafAcc> = Vec::new();
        let mut prefix: Vec<PathNode> = Vec::new();
        grow(
            columns,
            rows,
            cols,
            &residual,
            &all,
            0,
            max_depth,
            min_leaf,
            params.max_bins,
            &mut prefix,
            &mut leaves,
        );

        let mut produced_path = false;
        for leaf in &leaves {
            if leaf.path.is_empty() {
                continue; // degenerate (no split found) -> not a usable feature
            }
            produced_path = true;
            let support = leaf.indices.len() as u32;
            let gain = support as f32 * leaf.mean * leaf.mean;
            collected.push(DecisionPath {
                nodes: leaf.path.clone(),
                gain,
                support,
                round,
            });
        }
        if !produced_path {
            break; // nothing splits any more -> boosting has converged
        }
        // Residual update: leaves partition the rows, so each row is adjusted by
        // exactly one leaf mean.
        for leaf in &leaves {
            let shrink = learning_rate * leaf.mean;
            for &i in &leaf.indices {
                residual[i] -= shrink;
            }
        }
    }

    collected.sort_by(|a, b| {
        b.gain
            .partial_cmp(&a.gain)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for path in collected {
        if seen.insert(path_key(&path.nodes)) {
            unique.push(path);
        }
    }
    unique.truncate(params.max_paths.max(1) as usize);
    unique
}

/// Materialize a path's hard-AND membership indicator (column-major input): 1.0
/// where every condition holds, 0.0 where any concrete condition fails, NaN only
/// when membership is undetermined because a still-satisfied path hits a NaN
/// feature (so finite-pair scoring skips it).
pub fn path_membership(columns: &[f32], rows: usize, nodes: &[PathNode]) -> Vec<f32> {
    let mut out = vec![1.0f32; rows];
    for i in 0..rows {
        let mut member = 1.0f32;
        let mut undetermined = false;
        for node in nodes {
            let x = columns[node.feature as usize * rows + i];
            if x.is_nan() {
                undetermined = true;
                continue;
            }
            let holds = match node.sign {
                SplitSign::Le => x <= node.threshold,
                SplitSign::Gt => x > node.threshold,
            };
            if !holds {
                member = 0.0;
            }
        }
        out[i] = if member == 0.0 {
            0.0
        } else if undetermined {
            f32::NAN
        } else {
            1.0
        };
    }
    out
}

/// Expand a row-major feature matrix with decision-path membership columns
/// appended after the `cols` base features, mirroring `time_series::expand_row_major`
/// so the continuous engine mines base + path features on any backend. Returns
/// (expanded row-major, expanded column count, discovered paths in appended order).
pub fn expand_row_major(
    features: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
    params: &DecisionPathParams,
) -> (Vec<f32>, usize, Vec<DecisionPath>) {
    if rows == 0 || cols == 0 {
        return (features.to_vec(), cols, Vec::new());
    }
    let mut colmajor = vec![0.0f32; rows * cols];
    for t in 0..rows {
        let base = t * cols;
        for c in 0..cols {
            colmajor[c * rows + t] = features[base + c];
        }
    }
    let paths = find_decision_paths(&colmajor, rows, cols, target, params);
    let n_new = paths.len();
    let membership: Vec<Vec<f32>> = paths
        .iter()
        .map(|path| path_membership(&colmajor, rows, &path.nodes))
        .collect();

    let ecols = cols + n_new;
    let mut expanded = vec![0.0f32; rows * ecols];
    for t in 0..rows {
        let src = t * cols;
        let dst = t * ecols;
        for c in 0..cols {
            expanded[dst + c] = features[src + c];
        }
        for (j, column) in membership.iter().enumerate() {
            expanded[dst + cols + j] = column[t];
        }
    }
    (expanded, ecols, paths)
}

/// Human-readable label for a path, e.g. `path[f0>3.5000 & f2<=1.2000]`.
pub fn path_label(feature_names: &[String], nodes: &[PathNode]) -> String {
    let parts: Vec<String> = nodes
        .iter()
        .map(|node| {
            let name = feature_names
                .get(node.feature as usize)
                .map(String::as_str)
                .unwrap_or("f");
            let op = match node.sign {
                SplitSign::Le => "<=",
                SplitSign::Gt => ">",
            };
            format!("{name}{op}{:.4}", node.threshold)
        })
        .collect();
    format!("path[{}]", parts.join(" & "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_obvious_threshold() {
        // clean step: y jumps from 0 to 10 at x=3.5
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![0.0f32, 0.0, 0.0, 10.0, 10.0, 10.0];
        let s = best_variance_split(&x, &y).unwrap();
        assert!(
            (s.threshold - 3.5).abs() < 1e-6,
            "threshold={}",
            s.threshold
        );
        assert!(s.gain > 0.0);
    }

    #[test]
    fn constant_feature_has_no_split() {
        let x = vec![2.0f32, 2.0, 2.0, 2.0];
        let y = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!(best_variance_split(&x, &y).is_none());
    }

    #[test]
    fn constant_target_has_no_split() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 5.0, 5.0, 5.0];
        assert!(best_variance_split(&x, &y).is_none());
    }

    #[test]
    fn split_boundary_cap_samples_candidates_without_subsampling_rows() {
        let pairs = (0..10)
            .map(|value| (value as f32, if value >= 2 { 1.0 } else { 0.0 }))
            .collect::<Vec<_>>();
        let mut exhaustive_pairs = pairs.clone();
        let mut capped_pairs = pairs;

        let exhaustive = best_split_subset(&mut exhaustive_pairs, 1, 0).unwrap();
        let capped = best_split_subset(&mut capped_pairs, 1, 1).unwrap();

        assert_eq!(exhaustive.threshold, 1.5);
        assert_eq!(capped.threshold, 4.5);
        assert!(exhaustive.gain > capped.gain);
    }

    #[test]
    fn indicator_materializes_membership_and_skips_nan() {
        let x = vec![1.0f32, 4.0, f32::NAN, 5.0];
        let mut out = Vec::new();
        split_indicator(&x, 3.5, &mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert!(out[2].is_nan());
        assert_eq!(out[3], 1.0);
    }

    fn params(max_depth: u32) -> DecisionPathParams {
        DecisionPathParams {
            max_depth,
            rounds: 1,
            max_paths: 8,
            max_bins: 0,
            min_leaf: 2,
            learning_rate: 1.0,
        }
    }

    #[test]
    fn finds_a_depth2_and_conjunction() {
        // y is high only in the (f0 high AND f1 high) quadrant -> a depth-2 tree
        // must recover that 2-condition conjunction as its top-gain path.
        let mut f0 = Vec::new();
        let mut f1 = Vec::new();
        let mut y = Vec::new();
        for q0 in 0..2 {
            for q1 in 0..2 {
                for k in 0..10 {
                    f0.push((if q0 == 0 { 0.2 } else { 0.8 }) + 0.001 * k as f32);
                    f1.push((if q1 == 0 { 0.2 } else { 0.8 }) + 0.001 * k as f32);
                    y.push(if q0 == 1 && q1 == 1 { 5.0 } else { 0.0 });
                }
            }
        }
        let rows = y.len();
        let mut columns = Vec::with_capacity(rows * 2);
        columns.extend_from_slice(&f0);
        columns.extend_from_slice(&f1);

        let paths = find_decision_paths(&columns, rows, 2, &y, &params(2));
        assert!(!paths.is_empty());
        let top = &paths[0];
        assert_eq!(top.nodes.len(), 2, "top path should be a 2-way conjunction");
        // The top region must select exactly the high-y rows.
        let member = path_membership(&columns, rows, &top.nodes);
        for i in 0..rows {
            if member[i] == 1.0 {
                assert_eq!(y[i], 5.0, "row {i} in top region should be high-y");
            }
        }
        let selected = member.iter().filter(|&&m| m == 1.0).count();
        assert_eq!(
            selected, 10,
            "top region should be the 10-row high quadrant"
        );
    }

    #[test]
    fn path_membership_is_hard_and_with_nan_skip() {
        // 4 rows, 2 features (column-major).
        let columns = vec![
            0.0, 1.0, 0.0, 1.0, // f0
            0.0, 0.0, 1.0, 1.0, // f1
        ];
        let nodes = vec![
            PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
            PathNode {
                feature: 1,
                threshold: 0.5,
                sign: SplitSign::Gt,
            },
        ];
        let member = path_membership(&columns, 4, &nodes);
        assert_eq!(member, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn expand_appends_membership_columns_and_labels() {
        // 8-row AND structure -> at least one path column appended.
        let features = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        ];
        let target = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let p = DecisionPathParams {
            max_depth: 2,
            rounds: 1,
            max_paths: 4,
            max_bins: 0,
            min_leaf: 1,
            learning_rate: 1.0,
        };
        let (expanded, ecols, paths) = expand_row_major(&features, &target, 8, 2, &p);
        assert!(!paths.is_empty());
        assert_eq!(ecols, 2 + paths.len());
        assert_eq!(expanded.len(), 8 * ecols);
        let label = path_label(&["f0".to_string(), "f1".to_string()], &paths[0].nodes);
        assert!(label.starts_with("path["), "label={label}");
    }

    #[test]
    fn boosting_dedups_and_caps_paths() {
        // Repeated rounds on the same structure must not emit duplicate paths, and
        // max_paths caps the output.
        let mut f0 = Vec::new();
        let mut y = Vec::new();
        for k in 0..40 {
            let v = k as f32 / 40.0;
            f0.push(v);
            y.push(if v > 0.5 { 3.0 } else { 0.0 });
        }
        let rows = y.len();
        let p = DecisionPathParams {
            max_depth: 1,
            rounds: 5,
            max_paths: 2,
            max_bins: 0,
            min_leaf: 2,
            learning_rate: 0.5,
        };
        let paths = find_decision_paths(&f0, rows, 1, &y, &p);
        assert!(paths.len() <= 2, "max_paths must cap output");
        let mut keys: Vec<String> = paths.iter().map(|p| path_key(&p.nodes)).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), paths.len(), "paths must be deduplicated");
    }
}
