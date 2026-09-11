use core::cmp::Ordering;

use gafime_types::PrecisionProfile;

use super::{
    ordering::compare_f64, Direction, EvidenceId, EvidenceTable, EvidenceValue, FeatureId,
    NativeEvidenceExecutor, ParetoFrontierRequest, SemanticError, SemanticResult,
};

// Mirrors the explicit compact-query region envelope. This is an admission
// check only; the local executor repeats it before any native allocation.
const MAX_LOCAL_RT_PARETO_CANDIDATES: usize = 8_192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingEvidence {
    RejectCandidate,
    Error,
    /// Only valid on an individual optional constraint, never on the primary.
    IgnoreConstraint,
}

/// Inclusive bounds in the named channel's own units. No implicit conversion
/// from graph energy, consistency or association into one universal strength.
#[derive(Clone, Debug)]
pub struct EvidenceConstraint {
    pub channel: EvidenceId,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    /// None inherits the policy. Optional channels must opt in explicitly.
    pub missing: Option<MissingEvidence>,
}

#[derive(Clone, Debug)]
pub struct EvidenceObjective {
    pub channel: EvidenceId,
    pub direction: Direction,
}

#[derive(Clone, Debug)]
pub struct SelectionPolicy {
    pub primary: EvidenceId,
    pub direction: Direction,
    pub constraints: Vec<EvidenceConstraint>,
    pub missing: MissingEvidence,
    pub limit: usize,
    /// Empty preserves primary/constraint selection. Otherwise retain only the
    /// nondominated frontier before primary ordering and truncation. Each axis
    /// retains its own units; no weighted scalar utility is manufactured.
    pub pareto_objectives: Vec<EvidenceObjective>,
}

#[derive(Debug)]
struct RankedCandidate {
    feature: FeatureId,
    primary: f64,
    objectives: Vec<f64>,
}

/// The one policy-owned handoff between eligibility and arithmetic. It keeps
/// original profile-native report values for final primary ordering, while an
/// optional local executor receives only derived physical coordinates.
#[derive(Debug)]
struct PreparedSelection {
    profile: PrecisionProfile,
    direction: Direction,
    limit: usize,
    objectives: Vec<EvidenceObjective>,
    ranked: Vec<RankedCandidate>,
}

struct LocalCoordinates {
    values: Vec<f32>,
    coordinate_bytes: usize,
    equality_index_bytes: usize,
}

impl SelectionPolicy {
    /// Ordinary selection deliberately remains hookless. This preserves the
    /// existing Core and non-local executor behavior, including the established
    /// bounded host frontier pass.
    pub(crate) fn select(
        &self,
        table: &EvidenceTable,
        max_work: usize,
    ) -> SemanticResult<Vec<FeatureId>> {
        self.prepare(table, max_work)?.finish_core()
    }

    /// Use an explicitly negotiated arithmetic seam only after Rust has
    /// completed policy validation, missingness handling, constraints, and the
    /// same quadratic work admission used by the Core reference path.
    pub(crate) fn select_with_executor(
        &self,
        executor: &mut dyn NativeEvidenceExecutor,
        table: &EvidenceTable,
        max_work: usize,
        max_bytes: usize,
    ) -> SemanticResult<Vec<FeatureId>> {
        let prepared = self.prepare(table, max_work)?;
        if !prepared.has_pareto() || prepared.ranked.is_empty() || !executor.wants_pareto_frontier()
        {
            return prepared.finish_core();
        }

        // A local executor must see the unsupported profile/dimension request
        // and fail closed itself. There is intentionally no f64/mixed score
        // conversion, rank-position surrogate, or allocation on this branch.
        if prepared.profile != PrecisionProfile::Fp32
            || !(2..=3).contains(&prepared.objectives.len())
            || prepared.ranked.len() > MAX_LOCAL_RT_PARETO_CANDIDATES
        {
            let request = ParetoFrontierRequest::new(
                prepared.profile,
                prepared.objectives.len(),
                prepared.ranked.len(),
                None,
            );
            let answer = executor.pareto_weak_dominator_counts(request, max_bytes)?;
            if answer.is_none() {
                return Err(SemanticError::Invalid(
                    "local Pareto executor declined after explicit negotiation",
                ));
            }
            return Err(SemanticError::Unsupported(
                "local RT Pareto requires fp32 evidence with two or three objectives",
            ));
        }

        let coordinates = prepared.local_coordinates(max_bytes)?;
        let reserved_host = coordinates
            .coordinate_bytes
            .checked_add(coordinates.equality_index_bytes)
            .ok_or(SemanticError::Invalid(
                "local RT Pareto host reservation overflow",
            ))?;
        let native_budget = max_bytes
            .checked_sub(reserved_host)
            .ok_or(SemanticError::Invalid(
                "local RT Pareto coordinates exceed selection budget",
            ))?;
        let request = ParetoFrontierRequest::new(
            prepared.profile,
            prepared.objectives.len(),
            prepared.ranked.len(),
            Some(&coordinates.values),
        );
        let weak_counts = executor
            .pareto_weak_dominator_counts(request, native_budget)?
            .ok_or(SemanticError::Invalid(
                "local Pareto executor declined after explicit negotiation",
            ))?;
        prepared.finish_from_weak_counts(weak_counts, coordinates, max_bytes)
    }

    /// Validate policy shape and construct only Rust-owned eligible rows. The
    /// numeric frontier implementation consumes this result; it never repeats
    /// policy or evidence lookup work.
    fn prepare(&self, table: &EvidenceTable, max_work: usize) -> SemanticResult<PreparedSelection> {
        let has = |id| table.channels().iter().any(|c| c.id() == id);
        if !has(self.primary)
            || self.constraints.len() > 32
            || self.missing == MissingEvidence::IgnoreConstraint
        {
            return Err(SemanticError::Invalid(
                "selection requires existing bounded evidence channels",
            ));
        }
        if self.pareto_objectives.len() == 1
            || self.pareto_objectives.len() > 8
            || self
                .pareto_objectives
                .iter()
                .enumerate()
                .any(|(i, objective)| {
                    !has(objective.channel)
                        || self.pareto_objectives[..i]
                            .iter()
                            .any(|previous| previous.channel == objective.channel)
                })
        {
            return Err(SemanticError::Invalid(
                "Pareto selection requires two to eight distinct existing channels",
            ));
        }
        let threshold = |value: f64| -> SemanticResult<f64> {
            let value = if table.frame().profile() == PrecisionProfile::Fp32 {
                f64::from(value as f32)
            } else {
                value
            };
            if !value.is_finite() {
                return Err(SemanticError::Invalid(
                    "selection threshold is not finite in the numeric profile",
                ));
            }
            Ok(value)
        };
        let mut constraints = Vec::with_capacity(self.constraints.len());
        for c in &self.constraints {
            if !has(c.channel)
                || c.minimum.is_some_and(|v| !v.is_finite())
                || c.maximum.is_some_and(|v| !v.is_finite())
                || matches!((c.minimum,c.maximum), (Some(a),Some(b)) if a > b)
            {
                return Err(SemanticError::Invalid(
                    "invalid evidence selection constraint",
                ));
            }
            constraints.push((
                c,
                c.minimum.map(threshold).transpose()?,
                c.maximum.map(threshold).transpose()?,
            ));
        }
        let measured = |candidate, channel, missing| -> SemanticResult<Option<f64>> {
            match table.value(candidate, channel)? {
                EvidenceValue::Measured { value, .. } if value.is_finite() => Ok(Some(value)),
                _ if missing != MissingEvidence::Error => Ok(None),
                _ => Err(SemanticError::Invalid(
                    "required selection evidence is unavailable",
                )),
            }
        };
        let mut ranked = Vec::with_capacity(table.candidates().len());
        for &candidate in table.candidates() {
            let primary = measured(candidate, self.primary, self.missing)?;
            let mut eligible = primary.is_some();
            let mut objectives = Vec::with_capacity(self.pareto_objectives.len());
            for objective in &self.pareto_objectives {
                let value = measured(candidate, objective.channel, self.missing)?;
                eligible &= value.is_some();
                objectives.push(value.unwrap_or(0.0));
            }
            // Inspect every required channel even when another constraint
            // rejected this row, so Error is not dependent on filter order.
            for (c, minimum, maximum) in &constraints {
                let missing = c.missing.unwrap_or(self.missing);
                eligible &= match measured(candidate, c.channel, missing)? {
                    Some(value) => {
                        minimum.is_none_or(|min| value >= min)
                            && maximum.is_none_or(|max| value <= max)
                    }
                    None => missing == MissingEvidence::IgnoreConstraint,
                };
            }
            if eligible {
                ranked.push(RankedCandidate {
                    feature: candidate,
                    primary: primary.expect("eligible primary"),
                    objectives,
                });
            }
        }
        if !self.pareto_objectives.is_empty() {
            // Admission precedes both the existing Core pair scan and the
            // local RT mask query. The latter remains a bounded execution
            // route, not a claim of subquadratic frontier work.
            let work = ranked
                .len()
                .checked_mul(ranked.len().saturating_sub(1))
                .and_then(|pairs| pairs.checked_mul(self.pareto_objectives.len()))
                .ok_or(SemanticError::Invalid("Pareto comparison work overflow"))?;
            if work > max_work {
                return Err(SemanticError::Invalid(
                    "Pareto comparison work limit exceeded",
                ));
            }
        }
        Ok(PreparedSelection {
            profile: table.frame().profile(),
            direction: self.direction,
            limit: self.limit,
            objectives: self.pareto_objectives.clone(),
            ranked,
        })
    }
}

impl PreparedSelection {
    fn has_pareto(&self) -> bool {
        !self.objectives.is_empty()
    }

    fn finish_core(mut self) -> SemanticResult<Vec<FeatureId>> {
        if self.has_pareto() {
            let frontier: Vec<bool> = self
                .ranked
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    !self.ranked.iter().enumerate().any(|(j, other)| {
                        i != j && dominates(&other.objectives, &row.objectives, &self.objectives)
                    })
                })
                .collect();
            let mut index = 0;
            self.ranked.retain(|_| {
                let keep = frontier[index];
                index += 1;
                keep
            });
        }
        Ok(self.finish_ranking())
    }

    fn finish_ranking(mut self) -> Vec<FeatureId> {
        self.ranked.sort_by(|a, b| {
            let order = compare_f64(a.primary, b.primary, self.direction);
            order.then_with(|| a.feature.cmp(&b.feature))
        });
        self.ranked.truncate(self.limit);
        self.ranked.into_iter().map(|row| row.feature).collect()
    }

    /// Validate eligible f32 values before allocating any coordinate buffer.
    /// The f64 report representation is permitted here only when it is the
    /// exact widening of a binary32 result; mixed/fp64 have already been
    /// rejected without a downcast.
    fn local_coordinates(&self, max_bytes: usize) -> SemanticResult<LocalCoordinates> {
        debug_assert_eq!(self.profile, PrecisionProfile::Fp32);
        debug_assert!((2..=3).contains(&self.objectives.len()));
        let candidate_count = self.ranked.len();
        let coordinate_count =
            candidate_count
                .checked_mul(self.objectives.len())
                .ok_or(SemanticError::Invalid(
                    "local RT Pareto coordinate count overflow",
                ))?;
        let coordinate_bytes = coordinate_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(SemanticError::Invalid(
                "local RT Pareto coordinate bytes overflow",
            ))?;
        let equality_index_bytes = candidate_count
            .checked_mul(std::mem::size_of::<usize>())
            .ok_or(SemanticError::Invalid(
                "local RT Pareto equality index bytes overflow",
            ))?;
        if coordinate_bytes
            .checked_add(equality_index_bytes)
            .is_none_or(|required| required > max_bytes)
        {
            return Err(SemanticError::Invalid(
                "local RT Pareto coordinates exceed selection budget",
            ));
        }
        for row in &self.ranked {
            for value in &row.objectives {
                let narrowed = *value as f32;
                if !value.is_finite()
                    || !narrowed.is_finite()
                    || f64::from(narrowed) != *value
                    || narrowed.is_subnormal()
                {
                    return Err(SemanticError::Invalid(
                        "local RT Pareto requires finite non-subnormal exact fp32 objectives",
                    ));
                }
            }
        }
        let mut values = Vec::with_capacity(coordinate_count);
        if values.capacity().checked_mul(std::mem::size_of::<f32>()) != Some(coordinate_bytes) {
            return Err(SemanticError::Invalid(
                "local RT Pareto coordinate allocation exceeded its reservation",
            ));
        }
        values.resize(coordinate_count, 0.0);
        for (row_index, row) in self.ranked.iter().enumerate() {
            for (axis, (value, objective)) in
                row.objectives.iter().zip(&self.objectives).enumerate()
            {
                let value = *value as f32;
                // Negation transforms maximize to the same lower-is-better
                // coordinate relation without inventing a rank surrogate.
                values[axis * candidate_count + row_index] = match objective.direction {
                    Direction::Minimize => value,
                    Direction::Maximize => -value,
                };
            }
        }
        Ok(LocalCoordinates {
            values,
            coordinate_bytes,
            equality_index_bytes,
        })
    }

    fn finish_from_weak_counts(
        mut self,
        mut weak_counts: Vec<u64>,
        coordinates: LocalCoordinates,
        max_bytes: usize,
    ) -> SemanticResult<Vec<FeatureId>> {
        let candidate_count = self.ranked.len();
        let count_bytes = weak_counts
            .capacity()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(SemanticError::Invalid(
                "local RT Pareto count bytes overflow",
            ))?;
        let total_host = coordinates
            .coordinate_bytes
            .checked_add(coordinates.equality_index_bytes)
            .and_then(|bytes| bytes.checked_add(count_bytes))
            .ok_or(SemanticError::Invalid(
                "local RT Pareto host reservation overflow",
            ))?;
        let candidate_bound = u64::try_from(candidate_count)
            .map_err(|_| SemanticError::Invalid("local RT Pareto candidate count overflows u64"))?;
        if weak_counts.len() != candidate_count
            || total_host > max_bytes
            || weak_counts.iter().any(|count| *count > candidate_bound)
        {
            return Err(SemanticError::Invalid(
                "local RT Pareto weak-dominator counts violate the bounded request",
            ));
        }

        // Sort only row ordinals. The coordinate buffer remains column-major
        // for native arithmetic; this host grouping has no pairwise dominance
        // recomputation. `partial_cmp == Equal` intentionally treats -0 and
        // +0 as one exact vector group.
        let mut order = Vec::with_capacity(candidate_count);
        if order.capacity().checked_mul(std::mem::size_of::<usize>())
            != Some(coordinates.equality_index_bytes)
        {
            return Err(SemanticError::Invalid(
                "local RT Pareto equality index allocation exceeded its reservation",
            ));
        }
        order.extend(0..candidate_count);
        order.sort_unstable_by(|left, right| {
            compare_coordinate_rows(
                &coordinates.values,
                candidate_count,
                self.objectives.len(),
                *left,
                *right,
            )
        });
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len()
                && equal_coordinate_rows(
                    &coordinates.values,
                    candidate_count,
                    self.objectives.len(),
                    order[start],
                    order[end],
                )
            {
                end += 1;
            }
            let equal_group = u64::try_from(end - start).map_err(|_| {
                SemanticError::Invalid("local RT Pareto equal-vector group overflows u64")
            })?;
            for &row in &order[start..end] {
                weak_counts[row] =
                    weak_counts[row]
                        .checked_sub(equal_group)
                        .ok_or(SemanticError::Invalid(
                            "local RT Pareto weak-dominator count omits an equal coordinate group",
                        ))?;
            }
            start = end;
        }
        let mut row = 0;
        self.ranked.retain(|_| {
            let keep = weak_counts[row] == 0;
            row += 1;
            keep
        });
        Ok(self.finish_ranking())
    }
}

fn compare_coordinate_rows(
    coordinates: &[f32],
    candidate_count: usize,
    dimensions: usize,
    left: usize,
    right: usize,
) -> Ordering {
    for axis in 0..dimensions {
        // `local_coordinates` rejected NaN, so this is a total lexicographic
        // ordering that preserves partial-comparison equality for signed zero.
        let order = coordinates[axis * candidate_count + left]
            .partial_cmp(&coordinates[axis * candidate_count + right])
            .expect("validated local RT Pareto coordinates are ordered");
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

fn equal_coordinate_rows(
    coordinates: &[f32],
    candidate_count: usize,
    dimensions: usize,
    left: usize,
    right: usize,
) -> bool {
    (0..dimensions).all(|axis| {
        coordinates[axis * candidate_count + left]
            .partial_cmp(&coordinates[axis * candidate_count + right])
            == Some(Ordering::Equal)
    })
}

fn dominates(left: &[f64], right: &[f64], objectives: &[EvidenceObjective]) -> bool {
    let mut strictly_better = false;
    for ((a, b), objective) in left.iter().zip(right).zip(objectives) {
        let order = compare_f64(*a, *b, objective.direction);
        if order.is_gt() {
            return false;
        }
        strictly_better |= order.is_lt();
    }
    strictly_better
}

#[cfg(test)]
#[path = "selection_tests.rs"]
mod selection_tests;
