use super::{
    ordering::compare_f64, Direction, EvidenceId, EvidenceTable, EvidenceValue, FeatureId,
    SemanticError, SemanticResult,
};

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

impl SelectionPolicy {
    pub(crate) fn select(
        &self,
        table: &EvidenceTable,
        max_work: usize,
    ) -> SemanticResult<Vec<FeatureId>> {
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
            let value = if table.frame().profile() == gafime_types::PrecisionProfile::Fp32 {
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
        let mut ranked = Vec::new();
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
                ranked.push((candidate, primary.expect("eligible primary"), objectives));
            }
        }
        if !self.pareto_objectives.is_empty() {
            // Admission precedes the quadratic frontier pass. The explicit
            // session work ceiling makes this modest-channel policy usable
            // without disguising an unbounded candidate comparison campaign.
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
            let frontier: Vec<bool> = ranked
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    !ranked.iter().enumerate().any(|(j, other)| {
                        i != j && dominates(&other.2, &row.2, &self.pareto_objectives)
                    })
                })
                .collect();
            let mut index = 0;
            ranked.retain(|_| {
                let keep = frontier[index];
                index += 1;
                keep
            });
        }
        ranked.sort_by(|a, b| {
            let order = compare_f64(a.1, b.1, self.direction);
            order.then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(self.limit);
        Ok(ranked.into_iter().map(|(id, _, _)| id).collect())
    }
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
