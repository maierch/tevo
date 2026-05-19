//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Cluster search for occupied lattice sites.

use crate::{
    bits::BitCombination,
    lattice::{Lattice, SiteNeighbors},
};

/// Reusable depth-first search state for finding occupied clusters.
pub struct ClusterSearch {
    cluster_sizes: Vec<u8>,
    stack: Vec<u8>,
    current_cluster: Vec<u8>,
    visited: BitCombination,
}

#[inline]
fn is_occupied(site_index: u8, state_bits: &BitCombination) -> bool {
    state_bits.bit_at(site_index)
}

#[inline]
fn neighbors_of(site_neighbors: &[SiteNeighbors], site_index: u8) -> &[u8] {
    let site_index: usize = site_index.into();
    let neighbor_sites = &site_neighbors[site_index];
    let num_neighbors: usize = neighbor_sites.number_of_neighbors.into();
    &neighbor_sites.neighbor_site_indices[0..num_neighbors]
}

impl ClusterSearch {
    /// Creates a reusable cluster search workspace.
    pub fn new() -> Self {
        let cluster_sizes = Vec::with_capacity(256);
        let sites_to_visit = Vec::with_capacity(256);
        let current_cluster = Vec::with_capacity(256);
        let visited = BitCombination::zeros();
        Self {
            cluster_sizes,
            stack: sites_to_visit,
            current_cluster,
            visited,
        }
    }

    #[inline]
    fn is_or_was_on_stack(&self, site_index: u8) -> bool {
        self.visited.bit_at(site_index)
    }

    #[inline]
    fn push_to_stack(&mut self, site_index: u8) {
        self.stack.push(site_index);
        self.visited.set_bit_to_one(site_index);
    }

    /// Finds occupied connected components in the given lattice state.
    pub fn find_clusters(&mut self, lattice: &Lattice, state_bits: &BitCombination) {
        let site_neighbors = lattice.get_site_neighbor_map();
        let num_sites = lattice.number_of_sites;
        self.stack.clear();
        self.visited = BitCombination::zeros();
        if num_sites != self.cluster_sizes.len() {
            self.cluster_sizes.resize(num_sites, 0);
        }
        self.cluster_sizes.fill(0);

        // println!("Cluster search starts");
        let num_sites_u8 = num_sites.try_into().unwrap();
        for start in 0..num_sites_u8 {
            // println!("Start site: {start}");
            let start_site_index = start;
            if self.is_or_was_on_stack(start_site_index) {
                // This site has already been visited. Skip it.
                // println!("Site {start} has already been visited.");
                continue;
            }
            // Begin a new cluster.
            self.current_cluster.clear();
            self.push_to_stack(start_site_index);
            // println!("Beginning a cluster search at site {start}.");
            while let Some(site) = self.stack.pop() {
                // println!("  Popped site {site} from the stack");
                if is_occupied(site, state_bits) {
                    self.current_cluster.push(site);
                    for neighbor in neighbors_of(site_neighbors, site) {
                        let neighbor = *neighbor;
                        if !self.is_or_was_on_stack(neighbor) {
                            // println!("    Pushed neighbor {neighbor} to the stack.");
                            self.push_to_stack(neighbor);
                        }
                    }
                }
            }
            // Register the cluster in the main array.
            if !self.current_cluster.is_empty() {
                let current_cluster_size: u8 = self.current_cluster.len().try_into().unwrap();
                for site_index in &self.current_cluster {
                    let index: usize = (*site_index).into();
                    // println!("{index} is part of this cluster");
                    self.cluster_sizes[index] = current_cluster_size;
                }
            }
        }
        // println!("Cluster search has ended");
    }

    /// Returns a mapping from site indices to the size of the cluster that this site is a part of.
    pub fn get_cluster_sizes(&self) -> &[u8] {
        &self.cluster_sizes
    }
}

mod tests {

    #[test]
    fn test_cluster_search_1() {
        use super::ClusterSearch;
        use crate::{
            basis::Basis,
            lattice::{Lattice, Location, Periodicity},
        };

        let lattice_size = Location::new(4, 3);
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: true,
        };
        // 0   3   6   9
        // 1   4   7  10
        // 2   5   8  11
        let lattice = Lattice::new(lattice_size, periodicity).unwrap();
        let num_particles = 5;
        let basis = Basis::new(lattice, num_particles).unwrap();
        let lattice = &basis.lattice;

        let mut search = ClusterSearch::new();
        let state_bits = basis.at(basis
            .state_index_from_particle_locations(&[
                Location::new(0, 0), // Site 2
                Location::new(0, 1), // Site 1
                Location::new(1, 0), // Site 3
                Location::new(1, 2), // Site 5
                Location::new(3, 2), // Site 11
            ])
            .unwrap());
        search.find_clusters(&lattice, &state_bits);
        let clusters = search.get_cluster_sizes();
        assert_eq!(clusters[0], 4); // (0, 0)
        assert_eq!(clusters[1], 4); // (0, 1)
        assert_eq!(clusters[2], 0); // (0, 2)
        assert_eq!(clusters[3], 4); // (1, 0)
        assert_eq!(clusters[5], 4); // (1, 2)
        assert_eq!(clusters[8], 0); // (2, 2)
        assert_eq!(clusters[10], 0); // (3, 1)
        assert_eq!(clusters[11], 1); // (3, 2)
    }

    #[test]
    fn test_cluster_search_2() {
        use super::ClusterSearch;
        use crate::{
            basis::Basis,
            lattice::{Lattice, Location, Periodicity},
        };

        let lattice_size = Location::new(4, 2);
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: true,
        };
        // 0   2   4   6
        // 1   3   5   7
        let lattice = Lattice::new(lattice_size, periodicity).unwrap();
        let num_particles = 4;
        let basis = Basis::new(lattice, num_particles).unwrap();
        let lattice = &basis.lattice;

        let mut search = ClusterSearch::new();
        let state_bits = basis.at(basis
            .state_index_from_particle_locations(&[
                Location::new(1, 0), // Site 2
                Location::new(2, 0), // Site 4
                Location::new(1, 1), // Site 3
                Location::new(2, 1), // Site 5
            ])
            .unwrap());
        search.find_clusters(&lattice, &state_bits);
        let clusters = search.get_cluster_sizes();
        assert_eq!(clusters[0], 0);
        assert_eq!(clusters[1], 0);
        assert_eq!(clusters[2], 4);
        assert_eq!(clusters[3], 4);
        assert_eq!(clusters[4], 4);
        assert_eq!(clusters[5], 4);
        assert_eq!(clusters[6], 0);
        assert_eq!(clusters[7], 0);
    }
}
