//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Basis construction for particle configurations on a lattice.

use crate::{
    bits::{BitCombination, BitCombinations},
    lattice::{InitializationError, Lattice, Location},
};

/// A binary basis in Fock space.
#[derive(Debug)]
pub struct Basis {
    /// The underlying lattice.
    pub lattice: Lattice,

    /// Enumerates the fixed-particle bit combinations in this basis.
    pub combinations: BitCombinations,
}

impl Basis {
    /// Creates a new basis.
    pub fn new(lattice: Lattice, num_particles: usize) -> Option<Self> {
        let combinations = BitCombinations::new(lattice.number_of_sites, num_particles)?;
        Some(Self {
            lattice,
            combinations,
        })
    }

    /// Gets the basis dimension.
    ///
    /// This is the number of basis vectors.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.combinations.number_of_combinations()
    }

    /// Gets the number of particles for this basis.
    #[inline]
    pub fn number_of_particles(&self) -> usize {
        self.combinations.number_of_ones()
    }

    /// Returns the basis state at the given index.
    #[inline]
    pub fn at(&self, index: usize) -> BitCombination {
        self.combinations.combination_at_index(index)
    }

    /// Finds the index of the given basis state.
    ///
    /// Careful: the behavior is undefined if the state is not in the basis set.
    #[inline]
    pub fn index_of(&self, state: BitCombination) -> usize {
        self.combinations.index_of_combination(state)
    }

    /// Finds the index of the basis state index with particles at exactly the given locations.
    pub fn state_index_from_particle_locations(
        &self,
        particle_locations: &[Location],
    ) -> Result<usize, InitializationError> {
        let lattice = &self.lattice;
        if particle_locations.len() != self.number_of_particles() {
            return Err(InitializationError::InconsistentNumberOfParticles);
        }
        if let Some(location) = lattice.find_overlapping_locations(&particle_locations) {
            return Err(InitializationError::OverlappingParticles(location));
        };
        for location in particle_locations.iter() {
            if !lattice.contains(location) {
                return Err(InitializationError::LocationOutOfBounds(location.clone()));
            }
        }
        let site_indices: Vec<u8> = particle_locations
            .into_iter()
            .map(|location| lattice.location_to_site_index(location))
            .collect();
        let combination = BitCombination::with_ones_at(&site_indices);
        Ok(self.combinations.index_of_combination(combination))
    }
}

mod tests {
    use super::Basis;

    #[allow(dead_code)]
    fn assert_matches(basis: &Basis, state_index: usize, site_indices: &[u8]) {
        let state = basis.at(state_index);
        for i in 0..basis.number_of_particles() {
            let bit_position: u8 = i.try_into().unwrap();
            let occupied_at_i: bool = site_indices.contains(&bit_position);
            assert_eq!(state.bit_at(bit_position), occupied_at_i);
        }
        assert_eq!(basis.index_of(state), state_index);
    }

    #[test]
    fn test_index_of() {
        use super::Basis;
        use crate::lattice::{Lattice, Location, Periodicity};
        let lattice_size = Location::new(256, 1);
        let periodicity = Periodicity {
            periodic_in_x: true,
            periodic_in_y: false,
        };
        let lattice = Lattice::new(lattice_size, periodicity).unwrap();
        let last_bond = lattice.bonds.last().unwrap();
        assert_eq!(last_bond.site_index_0, 255);
        assert_eq!(last_bond.site_index_1, 0);

        let basis = Basis::new(lattice.clone(), 1).unwrap();
        assert_matches(&basis, 0, &[0]);
        assert_matches(&basis, 127, &[127]);
        assert_matches(&basis, 128, &[128]);
        assert_matches(&basis, 255, &[255]);

        let basis = Basis::new(lattice.clone(), 2).unwrap();
        assert_matches(&basis, 0, &[0, 1]);
        assert_matches(&basis, 1, &[0, 2]);
        assert_matches(&basis, 2, &[1, 2]);
        assert_matches(&basis, 3, &[0, 3]);
        assert_matches(&basis, 4, &[1, 3]);
        assert_matches(&basis, 5, &[2, 3]);
        assert_matches(&basis, 6, &[0, 4]);
        assert_matches(&basis, 7, &[1, 4]);

        assert_matches(&basis, 32637, &[252, 255]);
        assert_matches(&basis, 32638, &[253, 255]);
        assert_matches(&basis, 32639, &[254, 255]);
        // assert_matches(&basis, 0, &[1, 0])), 255);
    }
}
