//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Compact bitset utilities for enumerating particle configurations.

/// Provides quick evaluation of binomial coefficients via a precomputed Pascals triangle.
#[derive(Debug)]
struct BinomialCoefficients {
    /// Largest allowed value for `n`.
    n_max: usize,

    /// Largest allowed value for `k`.
    k_max: usize,

    /// Table of binomial coefficients.
    table: Vec<usize>,
}

impl BinomialCoefficients {
    /// Creates a new triangle with binomial coefficients.
    ///
    /// This will return `None` if:
    ///     - `n_max < k_max`
    ///     - `n_max * k_max > usize::max`
    ///     - The binomial cofficients become to large and cause an integer overflow somewher.
    pub fn new(n_max: usize, k_max: usize) -> Option<Self> {
        if n_max < k_max {
            return None;
        }
        let num_rows = n_max.checked_add(1)?; // num_rows = n_max + 1
        let row_length = k_max.checked_add(1)?; // row_length = k_max + 1
        let table_length = num_rows.checked_mul(row_length)?;
        let mut table = vec![0; table_length];
        // Fill the first row:
        if table_length > 0 {
            table[0] = 1;
        }
        // Fill the second row:
        if table_length > row_length + 1 {
            table[row_length] = 1;
            table[row_length + 1] = 1;
        }
        // Fill all other rows.
        let mut prev_row_base = row_length;
        let mut row_base = 2 * row_length;
        for _ in 2..=n_max {
            table[row_base] = 1;
            for k in 1..row_length {
                let prev_1: usize = table[prev_row_base + k - 1];
                let prev_2: usize = table[prev_row_base + k];
                table[row_base + k] = prev_1.checked_add(prev_2)?;
            }
            prev_row_base = row_base;
            row_base += row_length;
        }
        Some(Self {
            n_max,
            k_max,
            table,
        })
    }

    /// Returns the binomial coefficient `n` over `k`.
    ///
    /// This method panics if:
    ///     - `n > n_max`
    ///     - `k_max < k < n - k_max`
    ///
    /// For parameters outside of the valid range, this method panics.
    #[inline]
    pub fn binom(&self, n: usize, k: usize) -> usize {
        let k_max = self.k_max;
        if n > self.n_max {
            panic!("Binomial coefficient n={n}, k={k} is not in the table.");
        }
        if k > n {
            return 0;
        }
        let mut k = k;
        if k + k_max >= n {
            // read this as: if k >= n - k_max
            k = n - k; // n - k >= n - k_max  -->  k <= k_max
        }
        if k > k_max {
            panic!("Binomial coefficient n={n}, k={k} is not in the table.");
        }
        // The coefficient is at n * (k_max + 1) + k
        self.table[n * (k_max + 1) + k]
    }
}

#[inline(always)]
fn index_and_bit_position(bit_position: u8) -> (usize, u8) {
    if bit_position < 128 {
        (0, bit_position)
    } else {
        (1, bit_position - 128)
    }
}

// pub struct BitCombinationIterator {
//     bits: [u128; 2],
//     bit_index: u128,
//     index: usize,
// }

/// Represents a 256-bit combination.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BitCombination {
    /// Two 128-bit words storing up to 256 site occupation bits.
    pub bits: [u128; 2],
}

impl BitCombination {
    /// Creates a combination with no bits set.
    #[inline]
    pub fn zeros() -> Self {
        BitCombination { bits: [0; 2] }
    }

    /// Creates a combination with ones at the given bit positions.
    #[inline]
    pub fn with_ones_at(bit_positions: &[u8]) -> Self {
        let mut bits = [0; 2];
        for bit_position in bit_positions {
            let (index, bit_position) = index_and_bit_position(*bit_position);
            bits[index] |= 1 << bit_position;
        }
        BitCombination { bits }
    }

    /// Tests whether a bit is set.
    #[inline]
    pub fn bit_at(&self, bit_position: u8) -> bool {
        let (index, bit_position) = index_and_bit_position(bit_position);
        (self.bits[index] & (1 << bit_position)) != 0
    }

    /// Flips one bit in place.
    #[inline]
    pub fn flip(&mut self, bit_position: u8) {
        let (index, bit_position) = index_and_bit_position(bit_position);
        self.bits[index] ^= 1 << bit_position;
    }

    /// Sets one bit to one.
    #[inline]
    pub fn set_bit_to_one(&mut self, bit_position: u8) {
        let (index, bit_position) = index_and_bit_position(bit_position);
        self.bits[index] |= 1 << bit_position;
    }

    /// Writes the positions of the "one" bits of this [`BitCombination`] into `dest`.
    ///
    /// Returns the number of "one" bits that were found.
    #[inline]
    pub fn locations_of_ones_into(&self, dest: &mut [u8; 256]) -> usize {
        let mut num_bits = 0;
        let mut offset: u8 = 0;
        for mut bits_part in self.bits {
            loop {
                let bit_index = bits_part.trailing_zeros() as u8;
                if bit_index > 127 {
                    // There are no trailing zeros in this bit pattern.
                    offset = 128;
                    break;
                }
                dest[num_bits] = bit_index + offset;
                num_bits += 1;
                // Trick for resetting the lowest bit:
                bits_part &= bits_part.wrapping_sub(1);
            }
        }
        num_bits
    }

    /// Counts occupied sites strictly between two site indices.
    pub fn count_particles_between(&self, site_index_0: u8, site_index_1: u8) -> usize {
        let (start, end) = if site_index_0 <= site_index_1 {
            (site_index_0, site_index_1)
        } else {
            (site_index_1, site_index_0)
        };
        let mut count = 0;
        for site_index in (start + 1)..end {
            if self.bit_at(site_index) {
                count += 1;
            }
        }
        count
    }

    // pub fn iterate(&self) -> BitCombinationIterator {
    //     BitCombinationIterator {
    //         bit_index: 0,
    //         bits: self.bits,
    //         index: 0,
    //     }
    //    }
}

// impl BitCombinationIterator {
//     /// Gets the position of the next "one" bit.
//     #[inline]
//     pub fn next(&mut self) -> Option<u8> {
//         let index = self.index;
//         let bits = &mut self.bits;
//         let bit_index = bits[index].trailing_zeros() as u8;
//         if bit_index > 127 {
//             // No more trailing zeros
//             if index == 0 {
//                 self.index = 1;
//                 return self.next();
//             } else {
//                 return None;
//             }
//         }
//         // Trick for resetting the lowest bit:
//         bits[index] &= bits[index].wrapping_sub(1);
//         Some(bit_index)
//     }
// }

/// Provides quick evaluation of binomial coefficients via a precomputed "Pascals triangle".
#[derive(Debug)]
pub struct BitCombinations {
    num_combinations: usize,
    num_ones: usize,
    binom_coefficients: BinomialCoefficients,
}

impl BitCombinations {
    /// Bit combinations of `n` bits and `k` bits set to 1.
    pub fn new(num_bits: usize, num_ones: usize) -> Option<Self> {
        let n = num_bits;
        let k = num_ones;
        if n == 0 || k > n {
            return None;
        }
        let binom_coefficients = BinomialCoefficients::new(n, k)?;
        let num_combinations = binom_coefficients.binom(n, k);
        Some(Self {
            num_combinations,
            num_ones,
            binom_coefficients,
        })
    }

    /// Calculates the index of the given combination.
    ///
    /// The numbers in the combination must be strictly increasing.
    /// Otherwise, the result of this method is undefined.
    ///
    /// The method panics if `n` is larger than the `n_max` used to initialize this struct.
    ///
    /// For better understanding this code, see the
    /// [page](https://en.wikipedia.org/wiki/Combinatorial_number_system)
    /// on Wikipedia regarding combinatorial number systems.
    #[inline(never)]
    pub fn index_of_combination(&self, combination: BitCombination) -> usize {
        let mut acc = 0;
        let mut k = 1;
        let mut dest = [0; 256];
        let len = combination.locations_of_ones_into(&mut dest);
        for i in 0..len {
            acc += self.binom_coefficients.binom(dest[i] as usize, k);
            k += 1;
        }
        acc
        // let mut iter = combination.iterate();
        // loop {
        //     match iter.next() {
        //         None => {
        //             return acc;
        //         }
        //         Some(pos) => {
        //             acc += self.pascals_triangle.binom(pos as usize, k);
        //             // acc += 0;
        //             k += 1;
        //         }
        //     }
        // }
    }

    /// Calculates the combination for a given index.
    ///
    /// The smallest index is `0`, the largest allowed index can be determined
    /// with the `get_max_index(k)`` method.
    #[inline(never)]
    pub fn combination_at_index(&self, index: usize) -> BitCombination {
        let mut combination = BitCombination::zeros();
        let n_max = self.binom_coefficients.n_max;
        let mut k = self.num_ones;
        if k == 0 {
            return BitCombination::zeros();
        }
        let mut index = index;
        loop {
            // Iterate over n to find the largest (n over k) not exceeding `index`.
            let mut n = k - 1;
            let lower_bound_n = loop {
                let n_over_k = self.binom_coefficients.binom(n, k);
                if n_over_k > index {
                    break n - 1;
                } else if n_over_k == index {
                    break n;
                } else {
                    if n < n_max {
                        n += 1;
                    } else {
                        break n;
                    }
                }
            };
            let n = lower_bound_n;
            combination.set_bit_to_one(n as u8);
            let base = self.binom_coefficients.binom(n, k);
            index -= base;
            if index == 0 {
                for j in 0..(k - 1) {
                    combination.set_bit_to_one(j as u8);
                }
                break;
            }
            k -= 1;
        }
        combination
    }

    /// Gets the total number of bit combinations.
    pub fn number_of_combinations(&self) -> usize {
        self.num_combinations
    }

    /// The number of bits set to one in each combination.
    pub fn number_of_ones(&self) -> usize {
        self.num_ones
    }
}

mod tests {

    #[test]
    fn test_binom() {
        use crate::bits::BinomialCoefficients;

        let binom = BinomialCoefficients::new(4, 4).unwrap();

        assert_eq!(binom.binom(0, 0), 1);

        assert_eq!(binom.binom(1, 0), 1);
        assert_eq!(binom.binom(1, 1), 1);

        assert_eq!(binom.binom(2, 0), 1);
        assert_eq!(binom.binom(2, 1), 2);
        assert_eq!(binom.binom(2, 2), 1);

        assert_eq!(binom.binom(3, 0), 1);
        assert_eq!(binom.binom(3, 1), 3);
        assert_eq!(binom.binom(3, 2), 3);
        assert_eq!(binom.binom(3, 3), 1);

        assert_eq!(binom.binom(4, 0), 1);
        assert_eq!(binom.binom(4, 1), 4);
        assert_eq!(binom.binom(4, 2), 6);
        assert_eq!(binom.binom(4, 3), 4);
        assert_eq!(binom.binom(4, 4), 1);
    }

    #[test]
    fn test_binom_large() {
        use crate::bits::BinomialCoefficients;

        let binom = BinomialCoefficients::new(256, 11).unwrap();

        assert_eq!(binom.binom(0, 0), 1);

        assert_eq!(binom.binom(1, 0), 1);
        assert_eq!(binom.binom(1, 1), 1);

        assert_eq!(binom.binom(4, 0), 1);
        assert_eq!(binom.binom(4, 1), 4);
        assert_eq!(binom.binom(4, 2), 6);
        assert_eq!(binom.binom(4, 3), 4);
        assert_eq!(binom.binom(4, 4), 1);

        assert_eq!(binom.binom(256, 0), 1);
        assert_eq!(binom.binom(256, 1), 256);
        assert_eq!(binom.binom(256, 2), 32640);
        assert_eq!(binom.binom(256, 3), 2763520);

        assert_eq!(binom.binom(256, 11), 6235568072914502400);

        assert_eq!(binom.binom(256, 253), 2763520);
        assert_eq!(binom.binom(256, 254), 32640);
        assert_eq!(binom.binom(256, 255), 256);
        assert_eq!(binom.binom(256, 256), 1);
    }

    #[test]
    fn test_locations_of_ones_into() {
        use super::BitCombination;

        let mut dest = [0_u8; 256];
        let mut s = BitCombination::zeros();
        s.flip(255);

        for i in 0..255 {
            assert_eq!(s.bit_at(i as u8), false);
        }
        assert_eq!(s.bit_at(255), true);

        let len = s.locations_of_ones_into(&mut dest);
        assert_eq!(len, 1);
        assert_eq!(dest[0], 255);

        s.flip(56);
        let len = s.locations_of_ones_into(&mut dest);
        assert_eq!(len, 2);
        assert_eq!(dest[0], 56);
        assert_eq!(dest[1], 255);

        s.flip(129);
        let len = s.locations_of_ones_into(&mut dest);
        assert_eq!(len, 3);
        assert_eq!(dest[0], 56);
        assert_eq!(dest[1], 129);
        assert_eq!(dest[2], 255);
    }

    #[test]
    fn test_combinations() {
        use crate::bits::{BitCombination, BitCombinations};

        let combinations = BitCombinations::new(2, 0).unwrap();
        assert_eq!(combinations.number_of_combinations(), 1);
        assert_eq!(combinations.number_of_ones(), 0);
        assert_eq!(
            combinations.index_of_combination(BitCombination::zeros()),
            0
        );

        let combinations = BitCombinations::new(5, 2).unwrap();

        let index_of = |bit_positions: &[u8]| -> usize {
            combinations.index_of_combination(BitCombination::with_ones_at(bit_positions))
        };

        assert_eq!(index_of(&[0, 1]), 0);
        assert_eq!(index_of(&[0, 2]), 1);
        assert_eq!(index_of(&[1, 2]), 2);
        assert_eq!(index_of(&[0, 3]), 3);
        assert_eq!(index_of(&[1, 3]), 4);
        assert_eq!(index_of(&[2, 3]), 5);
        assert_eq!(index_of(&[0, 4]), 6);
        assert_eq!(index_of(&[1, 4]), 7);
        assert_eq!(index_of(&[2, 4]), 8);
        assert_eq!(index_of(&[3, 4]), 9);
        assert_eq!(combinations.number_of_combinations(), 10);

        let combinations = BitCombinations::new(5, 3).unwrap();
        assert_eq!(index_of(&[0, 1, 2]), 0);
        assert_eq!(index_of(&[0, 1, 3]), 1);
        assert_eq!(index_of(&[0, 2, 3]), 2);
        assert_eq!(index_of(&[1, 2, 3]), 3);
        assert_eq!(index_of(&[0, 1, 4]), 4);
        assert_eq!(index_of(&[0, 2, 4]), 5);

        pub fn ones(bit_combination: BitCombination) -> Vec<usize> {
            let mut dest = [0; 256];
            let len = bit_combination.locations_of_ones_into(&mut dest);
            let mut vec = vec![0; len];
            for i in 0..len {
                vec[i] = dest[i] as usize;
            }
            vec
        }

        assert_eq!(ones(combinations.combination_at_index(0)), &[0, 1, 2]);
        assert_eq!(ones(combinations.combination_at_index(1)), &[0, 1, 3]);
        assert_eq!(ones(combinations.combination_at_index(2)), &[0, 2, 3]);
        assert_eq!(ones(combinations.combination_at_index(3)), &[1, 2, 3]);
        assert_eq!(ones(combinations.combination_at_index(4)), &[0, 1, 4]);
        assert_eq!(ones(combinations.combination_at_index(5)), &[0, 2, 4]);

        let combinations = BitCombinations::new(9, 5).unwrap();
        let combination = BitCombination::with_ones_at(&[0, 1, 3, 6, 8]);
        assert_eq!(combinations.index_of_combination(combination), 72);
        assert_eq!(combinations.combination_at_index(72), combination);
    }

    #[test]
    fn test_count_particles_between() {
        use crate::bits::BitCombination;

        let s = BitCombination::with_ones_at(&[3, 4]);
        assert_eq!(s.count_particles_between(0, 0), 0);
        assert_eq!(s.count_particles_between(0, 1), 0);
        assert_eq!(s.count_particles_between(0, 4), 1);
        assert_eq!(s.count_particles_between(4, 0), 1);

        let s = BitCombination::with_ones_at(&[1, 3]);
        assert_eq!(s.count_particles_between(0, 4), 2);
        assert_eq!(s.count_particles_between(1, 4), 1);
        assert_eq!(s.count_particles_between(3, 4), 0);
    }

    #[test]
    fn test_flip() {
        use crate::bits::BitCombination;

        let mut s = BitCombination::zeros();
        s.flip(0); // Flip bit 0
        assert_eq!(s.bits[0], 1_u128);
        assert_eq!(s.bits[1], 0_u128);

        s.flip(3); // Flip bit 3
        assert_eq!(s.bits[0], 9_u128);
        assert_eq!(s.bits[1], 0_u128);

        s.flip(127); // Flip bit 127
        assert_eq!(s.bits[0], 170141183460469231731687303715884105737_u128);
        assert_eq!(s.bits[1], 0_u128);

        s.flip(128); // Flip bit 128
        assert_eq!(s.bits[0], 170141183460469231731687303715884105737_u128);
        assert_eq!(s.bits[1], 1_u128);

        s.flip(127); // Flip bit 127 again
        s.flip(130); // Flip bit 130
        assert_eq!(s.bits[0], 9_u128);
        assert_eq!(s.bits[1], 5_u128);
    }
}
