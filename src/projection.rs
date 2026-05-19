//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Projection helpers for restricting a state to a lattice area.

use crate::{
    basis::Basis,
    bits::BitCombination,
    complex::{Complex, normalize_complex_vector},
    lattice::{Lattice, Location},
};

/// Tests whether any occupied site lies inside the inclusive area.
pub fn is_occupied_within(
    lattice: &Lattice,
    state_bits: BitCombination,
    start: Location,
    end: Location,
) -> bool {
    for x in start.x..=end.x {
        for y in start.y..=end.y {
            let location = Location::new(x, y);
            let bit_position = lattice.location_to_site_index(&location);
            let occupied = state_bits.bit_at(bit_position);
            if occupied {
                return true;
            }
        }
    }
    return false;
}

/// Runs a simple projection scheme where any particle must exist within the projection area.
pub fn perform_projection(
    basis: &Basis,
    state: &mut [Complex],
    start: Location,
    end: Location,
) -> f64 {
    let lattice = &basis.lattice;
    for (i, c_i) in state.iter_mut().enumerate() {
        let state_bits = basis.at(i);
        if !is_occupied_within(lattice, state_bits, start, end) {
            *c_i = Complex::zero();
        }
    }
    let norm = normalize_complex_vector(state);
    norm
}
