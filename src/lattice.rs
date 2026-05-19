//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Lattice geometry, bonds, and neighbor maps.

use std::error::Error;

use crate::bits::BitCombination;

/// Errors that can occur while building a lattice or basis.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum InitializationError {
    /// The lattice has no sites.
    LatticeTooSmall,

    /// The lattice exceeds the supported 256-site limit.
    LatticeTooLarge,

    /// A location lies outside the lattice.
    LocationOutOfBounds(Location),

    /// The number of particle locations does not match the basis.
    InconsistentNumberOfParticles,

    /// Multiple particles were placed at the same location.
    OverlappingParticles(Location),

    /// The lattice size is invalid.
    BadLatticeSize(Location),
}

impl InitializationError {
    /// Converts the error to a simple string.
    pub fn to_string(&self) -> String {
        format!("{:?}", self)
    }

    /// Converts the error into a boxed dynamic error.
    pub fn boxed(&self) -> Box<dyn Error> {
        self.to_string().into()
    }
}

#[inline]
fn within(x: i32, bounds: i32) -> bool {
    x >= 0 && x < bounds
}

/// Location of a particle.
///
/// Both x and y components must be in the range `0 <= x <= 15`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Location {
    /// Component in x-direction.
    pub x: i32,

    /// Component in y-direction.
    pub y: i32,
}

impl Location {
    /// Creates a lattice location.
    #[inline]
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Linear index of a location.
#[inline]
fn location_to_site_index(location: Location, lattice_size: Location) -> Option<u8> {
    let x = location.x;
    let y = location.y;
    if within(x, lattice_size.x) && within(y, lattice_size.y) {
        let index = y + x * lattice_size.y;
        index.try_into().ok()
    } else {
        None
    }
}

/// Location from a linear index
#[inline]
fn location_from_site_index(site_index: u8, lattice_size: Location) -> Option<Location> {
    let site_index_i32: i32 = site_index.into();
    let x = site_index_i32 / lattice_size.y;
    let y = site_index_i32 - x * lattice_size.y;
    if within(x, lattice_size.x) && within(y, lattice_size.y) {
        Some(Location::new(x, y))
    } else {
        None
    }
}

fn count_sites_from_lattice_size(lattice_size: Location) -> Result<usize, InitializationError> {
    let sx: Option<usize> = lattice_size.x.try_into().ok();
    let sy: Option<usize> = lattice_size.y.try_into().ok();
    if sx.is_none() || sy.is_none() {
        return Err(InitializationError::BadLatticeSize(lattice_size));
    }
    let sx = sx.unwrap();
    let sy = sy.unwrap();
    if sx > 256 || sy > 256 {
        return Err(InitializationError::LatticeTooLarge);
    }
    let number_of_sites = sx * sy;
    if number_of_sites < 1 {
        return Err(InitializationError::LatticeTooSmall);
    }
    if number_of_sites > 256 {
        return Err(InitializationError::LatticeTooLarge);
    }
    Ok(number_of_sites)
}

impl std::ops::Add<Location> for Location {
    type Output = Location;

    fn add(self, rhs: Location) -> Self::Output {
        Location {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

/// Alignment of a bond.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BondAlignment {
    /// Bond in x-direction.
    Horizontal,

    /// Bond in y-direction.
    Vertical,
}

/// A bond between two sites.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Bond {
    /// Hopping strength.
    pub t: f64,

    /// Interaction strength.
    pub v: f64,

    /// Direction of the bond.
    pub alignment: BondAlignment,

    /// First site index.
    pub site_index_0: u8,

    /// Second site index.
    pub site_index_1: u8,
}

/// Periodicity of a lattice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Periodicity {
    /// Whether the lattice is periodic in x-direction.
    pub periodic_in_x: bool,

    /// Whether the lattice is periodic in y-direction.
    pub periodic_in_y: bool,
}

/// A struct that says how many neighbors a site has and where they are.
#[derive(Clone, PartialEq, Debug)]
pub struct SiteNeighbors {
    /// Number of neighbors for this site.
    pub number_of_neighbors: u8,

    /// Array of site indices for all neighbor sites.
    ///
    /// Note that only the first `number_of_neighbors` in this array are valid.
    pub neighbor_site_indices: [u8; 4],
}

/// Rectangular lattice with nearest-neighbor bonds.
#[derive(Clone, PartialEq, Debug)]
pub struct Lattice {
    /// The size of the lattice.
    pub lattice_size: Location,

    /// The number of sites in the lattice.
    pub number_of_sites: usize,

    /// The bonds of the lattice.
    pub bonds: Vec<Bond>,

    // /// Additional potential term for each site index.
    // pub potential: Vec<f64>,
    /// Periodicity of the lattice.
    pub periodicity: Periodicity,

    /// Vector of site neighbors for each site.
    site_neighbor_map: Vec<SiteNeighbors>,
}

/// Wraps around a lattice coordinate on a periodic lattice.
fn wrap_around(size: i32, x: i32) -> i32 {
    x.rem_euclid(size)
}

fn make_bond(
    bonds: &mut Vec<Bond>,
    lattice_size: Location,
    periodicity: Periodicity,
    alignment: BondAlignment,
    x0: i32,
    y0: i32,
    dx: i32,
    dy: i32,
) {
    let nx = lattice_size.x;
    let ny = lattice_size.y;
    let site_index_0 = location_to_site_index(Location::new(x0, y0), lattice_size).unwrap();
    let x1 = if periodicity.periodic_in_x {
        wrap_around(nx, x0 + dx)
    } else {
        x0 + dx
    };
    let y1 = if periodicity.periodic_in_y {
        wrap_around(ny, y0 + dy)
    } else {
        y0 + dy
    };
    let t = 0.0;
    let v = 0.0;
    if let Some(site_index_1) = location_to_site_index(Location::new(x1, y1), lattice_size) {
        bonds.push(Bond {
            t,
            v,
            alignment,
            site_index_0,
            site_index_1,
        });
    }
}

/// Creates nearest-neighbor bonds for a rectangular lattice.
pub fn make_lattice_bonds(lattice_size: Location, periodicity: Periodicity) -> Vec<Bond> {
    let number_of_sites = lattice_size.x * lattice_size.y;
    let mut bonds = Vec::with_capacity(2 * number_of_sites as usize);
    let h = BondAlignment::Horizontal;
    let v = BondAlignment::Vertical;
    for x in 0..lattice_size.x {
        for y in 0..lattice_size.y {
            make_bond(&mut bonds, lattice_size, periodicity, h, x, y, 1, 0);
            make_bond(&mut bonds, lattice_size, periodicity, v, x, y, 0, 1);
        }
    }
    bonds
}

#[inline]
fn add_neighbor_site(site_neighbors: &mut SiteNeighbors, neighbor_site_index: u8) {
    let num_neighbors = site_neighbors.number_of_neighbors;
    let i: usize = num_neighbors.into();
    if !site_neighbors.neighbor_site_indices[..i].contains(&neighbor_site_index) {
        site_neighbors.neighbor_site_indices[i] = neighbor_site_index;
        site_neighbors.number_of_neighbors = num_neighbors + 1;
    }
}

fn make_site_neighbors(num_sites: usize, bonds: &Vec<Bond>) -> Vec<SiteNeighbors> {
    let empty_site_neighbor = SiteNeighbors {
        number_of_neighbors: 0,
        neighbor_site_indices: [0, 0, 0, 0],
    };
    let mut site_neighbors = vec![empty_site_neighbor; num_sites];
    for bond in bonds {
        let site_index_0: usize = bond.site_index_0.into();
        let site_index_1: usize = bond.site_index_1.into();
        // println!("{site_index_0} has neighbor {site_index_1}");
        add_neighbor_site(&mut site_neighbors[site_index_0], bond.site_index_1);
        // println!(
        //     "{0} has now {1} neighbors",
        //     site_index_0, site_neighbors[site_index_0].number_of_neighbors
        // );
        add_neighbor_site(&mut site_neighbors[site_index_1], bond.site_index_0);
        // println!(
        //     "{0} has now {1} neighbors",
        //     site_index_1, site_neighbors[site_index_1].number_of_neighbors
        // );
    }
    for site_neighbor in site_neighbors.iter_mut() {
        let len = usize::from(site_neighbor.number_of_neighbors);
        site_neighbor.neighbor_site_indices[0..len].sort();
    }
    site_neighbors
}

impl Lattice {
    /// Creates a new square lattice of the given size.
    pub fn new(
        lattice_size: Location,
        periodicity: Periodicity,
    ) -> Result<Self, InitializationError> {
        if lattice_size.x <= 0 || lattice_size.y <= 0 {
            return Err(InitializationError::LatticeTooSmall);
        }
        let number_of_sites = count_sites_from_lattice_size(lattice_size)?;
        let bonds = make_lattice_bonds(lattice_size, periodicity);
        let neighbors = make_site_neighbors(number_of_sites, &bonds);
        // let potential = vec![0.0; number_of_sites as usize];
        Ok(Self {
            lattice_size,
            number_of_sites,
            bonds,
            periodicity,
            site_neighbor_map: neighbors,
            // potential,
        })
    }

    /// Sets equal bond parameters for all horizontal and vertical bonds.
    pub fn set_translation_invariant_bonds(&mut self, tx: f64, vx: f64, ty: f64, vy: f64) {
        for bond in &mut self.bonds {
            match bond.alignment {
                BondAlignment::Horizontal => {
                    bond.t = tx;
                    bond.v = vx;
                }
                BondAlignment::Vertical => {
                    bond.t = ty;
                    bond.v = vy;
                }
            }
        }
    }

    // pub fn find_bond(&mut self, site: Location, alignment: BondAlignment) -> Option<&mut Bond> {
    //     let site_index = self.location_to_site_index(&site);
    //     for bond in &mut self.bonds {
    //         if bond.site_index_0 == site_index && bond.alignment == alignment {
    //             return Some(bond);
    //         }
    //     }
    //     None
    // }

    // pub fn set_potential_at(&mut self, site: Location, potential: f64) {
    //     let site_index = self.location_to_site_index(&site);
    //     self.potential[site_index as usize] = potential;
    // }

    /// Tests whether this location is within the given bounds.
    #[inline]
    pub fn contains(&self, location: &Location) -> bool {
        within(location.x, self.lattice_size.x) && within(location.y, self.lattice_size.y)
    }

    /// Returns an error if the location is outside the lattice.
    pub fn expect_contains(&self, location: &Location) -> Result<(), InitializationError> {
        if !self.contains(&location) {
            return Err(InitializationError::LocationOutOfBounds(location.clone()));
        }
        Ok(())
    }

    /// Location from a linear site index.
    #[inline]
    pub fn location_to_site_index(&self, location: &Location) -> u8 {
        location_to_site_index(*location, self.lattice_size).unwrap()
    }

    /// Linear index of a location.
    #[inline]
    pub fn site_index_to_location(&self, site_index: u8) -> Location {
        location_from_site_index(site_index, self.lattice_size).unwrap()
    }

    /// Finds and returns overlaps in a list of locations, if there are any.
    pub fn find_overlapping_locations(&self, locations: &[Location]) -> Option<Location> {
        let mut occupied = BitCombination::zeros();
        for location in locations.iter() {
            let site_index: u8 = self.location_to_site_index(location);
            if occupied.bit_at(site_index) {
                return Some(location.clone());
            }
            occupied.set_bit_to_one(site_index);
        }
        None
    }

    /// Gets a mapping from site indices to site neighbors.
    pub fn get_site_neighbor_map(&self) -> &[SiteNeighbors] {
        &self.site_neighbor_map
    }
}

mod tests {

    #[test]
    fn test_wrap_around() {
        use super::wrap_around;

        assert_eq!(wrap_around(10, -2), 8);
        assert_eq!(wrap_around(10, -1), 9);
        assert_eq!(wrap_around(10, 0), 0);
        assert_eq!(wrap_around(10, 9), 9);
        assert_eq!(wrap_around(10, 10), 0);
        assert_eq!(wrap_around(10, 11), 1);
    }

    #[test]
    fn test_lattice() {
        use crate::lattice::*;
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: false,
        };

        assert_eq!(
            Lattice::new(Location::new(0, 1), periodicity),
            Err(InitializationError::LatticeTooSmall)
        );
        assert_eq!(
            Lattice::new(Location::new(32, 33), periodicity),
            Err(InitializationError::LatticeTooLarge)
        );

        let lattice = Lattice::new(Location::new(10, 3), periodicity).unwrap();

        assert_eq!(lattice.location_to_site_index(&Location::new(0, 0)), 0);
        assert_eq!(lattice.location_to_site_index(&Location::new(0, 2)), 2);
        assert_eq!(lattice.location_to_site_index(&Location::new(1, 0)), 3);
        assert_eq!(lattice.location_to_site_index(&Location::new(1, 1)), 4);
        assert_eq!(lattice.location_to_site_index(&Location::new(9, 2)), 29);

        assert_eq!(lattice.site_index_to_location(29), Location::new(9, 2));
    }

    #[test]
    fn test_lattice_neighbors() {
        use super::{Lattice, Location, Periodicity};

        fn test_bonds(
            lattice: &Lattice,
            bond_index: usize,
            expected_site_index_0: u8,
            expected_site_index_1: u8,
        ) {
            let bond = lattice.bonds[bond_index];
            assert_eq!(bond.site_index_0, expected_site_index_0);
            assert_eq!(bond.site_index_1, expected_site_index_1);
        }

        // Lattice: 0-1-2-3-4
        for periodic_in_x in [false, true] {
            let periodicity = Periodicity {
                periodic_in_x,
                periodic_in_y: false,
            };
            let lattice = Lattice::new(Location::new(5, 1), periodicity).unwrap();
            if periodic_in_x {
                assert_eq!(lattice.bonds.len(), 5);
            } else {
                assert_eq!(lattice.bonds.len(), 4);
            }

            test_bonds(&lattice, 0, 0, 1);
            test_bonds(&lattice, 1, 1, 2);
            test_bonds(&lattice, 2, 2, 3);
            test_bonds(&lattice, 3, 3, 4);
            if periodic_in_x {
                test_bonds(&lattice, 4, 4, 0);
            }
        }

        // Lattice:
        // 2-5
        // | |
        // 1-4
        // | |
        // 0-3
        for periodic_in_y in [false, true] {
            let periodicity = Periodicity {
                periodic_in_x: false,
                periodic_in_y,
            };
            let lattice = Lattice::new(Location::new(2, 3), periodicity).unwrap();
            if periodic_in_y {
                assert_eq!(lattice.bonds.len(), 9);
                test_bonds(&lattice, 0, 0, 3);
                test_bonds(&lattice, 1, 0, 1);
                test_bonds(&lattice, 2, 1, 4);
                test_bonds(&lattice, 3, 1, 2);
                test_bonds(&lattice, 4, 2, 5);
                test_bonds(&lattice, 5, 2, 0);
                test_bonds(&lattice, 6, 3, 4);
                test_bonds(&lattice, 7, 4, 5);
                test_bonds(&lattice, 8, 5, 3);
            } else {
                assert_eq!(lattice.bonds.len(), 7);
                test_bonds(&lattice, 0, 0, 3);
                test_bonds(&lattice, 1, 0, 1);
                test_bonds(&lattice, 2, 1, 4);
                test_bonds(&lattice, 3, 1, 2);
                test_bonds(&lattice, 4, 2, 5);
                test_bonds(&lattice, 5, 3, 4);
                test_bonds(&lattice, 6, 4, 5);
            }
        }
    }

    #[test]
    fn test_find_overlapping_locations() {
        use crate::lattice::*;
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: false,
        };
        let lattice = Lattice::new(Location::new(10, 3), periodicity).unwrap();
        assert_eq!(
            lattice.find_overlapping_locations(&[Location::new(0, 1), Location::new(0, 2)]),
            None
        );
        assert_eq!(
            lattice.find_overlapping_locations(&[
                Location::new(0, 1),
                Location::new(0, 2),
                Location::new(0, 1)
            ]),
            Some(Location::new(0, 1))
        );
    }

    #[test]
    fn test_get_site_neighbor_map() {
        use crate::lattice::*;
        // Lattice:
        // 2-5
        // | |
        // 1-4
        // | |
        // 0-3
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: true,
        };
        let lattice = Lattice::new(Location::new(2, 3), periodicity).unwrap();
        let site_neighbor_map = lattice.get_site_neighbor_map();
        assert_eq!(lattice.number_of_sites, 6);
        assert_eq!(site_neighbor_map.len(), 6);

        fn assert_neighbors_of(site_neighbors: &super::SiteNeighbors, expected_neighbors: &[u8]) {
            let num_expected_neighbors = expected_neighbors.len();
            assert_eq!(
                usize::from(site_neighbors.number_of_neighbors),
                num_expected_neighbors
            );
            for i in 0..num_expected_neighbors {
                assert_eq!(
                    expected_neighbors[i],
                    site_neighbors.neighbor_site_indices[i]
                );
            }
        }

        assert_neighbors_of(&site_neighbor_map[0], &[1, 2, 3]);
        assert_neighbors_of(&site_neighbor_map[1], &[0, 2, 4]);
        assert_neighbors_of(&site_neighbor_map[2], &[0, 1, 5]);
        assert_neighbors_of(&site_neighbor_map[3], &[0, 4, 5]);
        assert_neighbors_of(&site_neighbor_map[4], &[1, 3, 5]);
        assert_neighbors_of(&site_neighbor_map[5], &[2, 3, 4]);
    }
}
