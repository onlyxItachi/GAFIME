pub const UINT32_MAX: u128 = u32::MAX as u128;
pub const UINT64_MAX: u128 = u64::MAX as u128;
pub const UINT128_MAX: u128 = u128::MAX;

pub fn saturating_u128_mul(left: u128, right: u128) -> u128 {
    left.saturating_mul(right)
}

pub fn saturating_u64_add(left: u64, right: u128) -> u64 {
    let total = (left as u128).saturating_add(right);
    total.min(UINT64_MAX) as u64
}

pub fn chunk_count(count: u128, chunk_size: u64) -> u128 {
    if count == 0 {
        return 0;
    }
    let size = chunk_size.max(1) as u128;
    count.saturating_add(size - 1) / size
}

pub fn saturating_comb(n: u64, k: u64) -> (u128, bool) {
    if k > n {
        return (0, false);
    }
    let k = k.min(n - k);
    if k == 0 {
        return (1, false);
    }

    let mut result: u128 = 1;
    for i in 1..=k {
        let mut numerator = (n - k + i) as u128;
        let mut denominator = i as u128;

        let g_num = gcd(numerator, denominator);
        numerator /= g_num;
        denominator /= g_num;

        let g_result = gcd(result, denominator);
        result /= g_result;
        denominator /= g_result;

        match result.checked_mul(numerator) {
            Some(value) => result = value,
            None => return (UINT128_MAX, true),
        }

        if denominator > 1 {
            result /= denominator;
        }
    }

    (result, false)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}
