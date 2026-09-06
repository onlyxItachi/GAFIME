use super::{EvidenceId, EvidenceTable, EvidenceValue, FeatureId, SemanticError, SemanticResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingEvidence {
    RejectCandidate,
    Error,
}

/// Inclusive bounds in the named channel's own units. No implicit conversion
/// from graph energy, consistency or association into one universal strength.
#[derive(Clone, Debug)]
pub struct EvidenceConstraint {
    pub channel: EvidenceId,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SelectionPolicy {
    pub primary: EvidenceId,
    pub direction: Direction,
    pub constraints: Vec<EvidenceConstraint>,
    pub missing: MissingEvidence,
    pub limit: usize,
}

impl SelectionPolicy {
    pub(crate) fn select(&self, table: &EvidenceTable) -> SemanticResult<Vec<FeatureId>> {
        let has = |id| table.channels().iter().any(|c| c.id() == id);
        if !has(self.primary) || self.constraints.len() > 32 {
            return Err(SemanticError::Invalid(
                "selection requires existing bounded evidence channels",
            ));
        }
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
        }
        let measured = |candidate, channel| -> SemanticResult<Option<f64>> {
            match table.value(candidate, channel)? {
                EvidenceValue::Measured { value, .. } if value.is_finite() => Ok(Some(value)),
                _ if self.missing == MissingEvidence::RejectCandidate => Ok(None),
                _ => Err(SemanticError::Invalid(
                    "required selection evidence is unavailable",
                )),
            }
        };
        let mut ranked = Vec::new();
        for &candidate in table.candidates() {
            let primary = measured(candidate, self.primary)?;
            let mut eligible = primary.is_some();
            // Inspect every required channel even when another constraint
            // rejected this row, so Error is not dependent on filter order.
            for c in &self.constraints {
                eligible &= measured(candidate, c.channel)?.is_some_and(|value| {
                    c.minimum.is_none_or(|min| value >= min)
                        && c.maximum.is_none_or(|max| value <= max)
                });
            }
            if eligible {
                ranked.push((candidate, primary.expect("eligible primary")));
            }
        }
        ranked.sort_by(|a, b| {
            let score = a.1.partial_cmp(&b.1).expect("finite measured evidence");
            let order = if self.direction == Direction::Maximize {
                score.reverse()
            } else {
                score
            };
            order.then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(self.limit);
        Ok(ranked.into_iter().map(|(id, _)| id).collect())
    }
}
