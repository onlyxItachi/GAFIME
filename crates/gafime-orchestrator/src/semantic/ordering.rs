//! Small, typed score-ordering primitives shared by semantic policy owners.
//!
//! Callers retain their own admission and tie policy.  In particular,
//! `SelectionPolicy` admits only finite evidence and adds a `FeatureId`
//! tie-break, while supervised compatibility intentionally preserves producer
//! order for equal or unordered values before its seeded shortlist shuffle.

use core::cmp::Ordering;

/// Direction of a policy-owned score ordering.
///
/// This remains an internal semantic type: it names how an already-admitted
/// score is ordered and does not assign a cross-channel utility meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}

/// Compare binary32 scores in a policy-selected direction.
///
/// An unordered comparison is deliberately equal.  Stable supervised sorting
/// then retains its source order; callers that require finite scores must
/// reject them before calling this function.
pub(crate) fn compare_f32(left: f32, right: f32, direction: Direction) -> Ordering {
    directional_order(
        left.partial_cmp(&right).unwrap_or(Ordering::Equal),
        direction,
    )
}

/// Compare binary64 scores in a policy-selected direction without narrowing.
///
/// See [`compare_f32`] for the explicit unordered-value rule.
pub(crate) fn compare_f64(left: f64, right: f64, direction: Direction) -> Ordering {
    directional_order(
        left.partial_cmp(&right).unwrap_or(Ordering::Equal),
        direction,
    )
}

fn directional_order(order: Ordering, direction: Direction) -> Ordering {
    if direction == Direction::Maximize {
        order.reverse()
    } else {
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_each_precision_without_widening_or_forcing_nan_order() {
        let lower = 1.0f64;
        let higher = f64::from_bits(lower.to_bits() + 1);
        assert_eq!(lower as f32, higher as f32);
        assert_eq!(
            compare_f64(lower, higher, Direction::Maximize),
            Ordering::Greater
        );
        assert_eq!(compare_f32(-2.0, 1.0, Direction::Minimize), Ordering::Less);
        assert_eq!(
            compare_f32(f32::NAN, 1.0, Direction::Maximize),
            Ordering::Equal
        );
    }
}
