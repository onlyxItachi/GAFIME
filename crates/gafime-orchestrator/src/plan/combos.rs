pub fn binomial_saturating_u128(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        let numerator = (n - k + i) as u128;
        result = result.saturating_mul(numerator) / i as u128;
    }
    result
}

pub fn saturating_u64_offset(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_without_materializing() {
        assert_eq!(binomial_saturating_u128(5, 2), 10);
        assert_eq!(binomial_saturating_u128(1_000_000, 0), 1);
        assert_eq!(saturating_u64_offset(u128::MAX), u64::MAX);
    }
}
