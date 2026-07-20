//! Minimal CPython-compatible MT19937 stream for legacy candidate planning.
//!
//! GAFIME v0.4.7 and v0.5.0-legacy used `random.Random(seed).shuffle` to cap
//! unary candidates and order higher-arity descriptors. Keeping this small,
//! private implementation avoids making candidate identity depend on a new RNG
//! crate or on Python-side planning.

const STATE_LEN: usize = 624;
const STATE_PERIOD: usize = 397;

#[derive(Clone, Debug)]
pub(super) struct PythonRandom {
    state: [u32; STATE_LEN],
    index: usize,
}

impl PythonRandom {
    #[cfg(test)]
    pub(super) fn from_u64(seed: u64) -> Self {
        let key = if seed <= u32::MAX as u64 {
            vec![seed as u32]
        } else {
            vec![seed as u32, (seed >> 32) as u32]
        };
        Self::from_seed_words(&key)
    }

    pub(super) fn from_seed_words(seed_words: &[u32]) -> Self {
        let zero = [0u32];
        let key = if seed_words.is_empty() {
            &zero[..]
        } else {
            seed_words
        };
        let mut random = Self {
            state: [0; STATE_LEN],
            index: STATE_LEN,
        };
        random.init_by_array(key);
        random
    }

    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let swap_index = self.rand_below(index + 1);
            values.swap(index, swap_index);
        }
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..STATE_LEN {
            self.state[index] = 1_812_433_253u32
                .wrapping_mul(self.state[index - 1] ^ (self.state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = STATE_LEN;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut state_index = 1usize;
        let mut key_index = 0usize;
        for _ in 0..STATE_LEN.max(key.len()) {
            let previous = self.state[state_index - 1];
            self.state[state_index] = (self.state[state_index]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[key_index])
            .wrapping_add(key_index as u32);
            state_index += 1;
            key_index += 1;
            if state_index >= STATE_LEN {
                self.state[0] = self.state[STATE_LEN - 1];
                state_index = 1;
            }
            if key_index >= key.len() {
                key_index = 0;
            }
        }
        for _ in 0..STATE_LEN - 1 {
            let previous = self.state[state_index - 1];
            self.state[state_index] = (self.state[state_index]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(state_index as u32);
            state_index += 1;
            if state_index >= STATE_LEN {
                self.state[0] = self.state[STATE_LEN - 1];
                state_index = 1;
            }
        }
        self.state[0] = 0x8000_0000;
        self.index = STATE_LEN;
    }

    fn rand_below(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bits = usize::BITS - upper.leading_zeros();
        loop {
            let value = self.get_rand_bits(bits) as usize;
            if value < upper {
                return value;
            }
        }
    }

    fn get_rand_bits(&mut self, bits: u32) -> u32 {
        debug_assert!((1..=32).contains(&bits));
        self.gen_u32() >> (32 - bits)
    }

    fn gen_u32(&mut self) -> u32 {
        if self.index >= STATE_LEN {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn twist(&mut self) {
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        for index in 0..STATE_LEN {
            let joined = (self.state[index] & UPPER_MASK)
                | (self.state[(index + 1) % STATE_LEN] & LOWER_MASK);
            let mut next = self.state[(index + STATE_PERIOD) % STATE_LEN] ^ (joined >> 1);
            if joined & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[index] = next;
        }
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_matches_cpython_random_for_u64_seeds() {
        let cases = [
            (0, [4, 1, 5, 2, 0, 3, 7, 6]),
            (1, [3, 6, 1, 5, 7, 0, 4, 2]),
            (7, [6, 7, 2, 4, 0, 3, 1, 5]),
            (u32::MAX as u64 + 2, [2, 5, 1, 0, 7, 6, 4, 3]),
            (u64::MAX, [5, 7, 3, 6, 4, 2, 1, 0]),
        ];
        for (seed, expected) in cases {
            let mut values = [0, 1, 2, 3, 4, 5, 6, 7];
            PythonRandom::from_u64(seed).shuffle(&mut values);
            assert_eq!(values, expected, "seed {seed}");
        }
    }
}
