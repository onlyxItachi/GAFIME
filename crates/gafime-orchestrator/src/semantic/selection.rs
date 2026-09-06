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
        if !has(self.primary)
            || self.constraints.len() > 32
            || self.missing == MissingEvidence::IgnoreConstraint
        {
            return Err(SemanticError::Invalid(
                "selection requires existing bounded evidence channels",
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
                ranked.push((candidate, primary.expect("eligible primary")));
            }
        }
        ranked.sort_by(|a, b| {
            let order = compare_f64(a.1, b.1, self.direction);
            order.then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(self.limit);
        Ok(ranked.into_iter().map(|(id, _)| id).collect())
    }
}
