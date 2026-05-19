//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Hamiltonian construction and sparse matrix loading.

use crate::{basis::Basis, bits::BitCombination, lattice::Bond, sparse::SparseMatrix};

/// Defines the model of the system.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelType {
    /// Spinless fermion t/V model.
    TV,

    /// Particle-hole symmetric t/V model.
    TVSym,

    /// Spin-1/2 XXZ model.
    XXZ,
}

impl ModelType {
    /// Creates a model type from a unique identifier.
    pub fn from_identifier(id: &str) -> Option<ModelType> {
        match id {
            "tv" => Some(Self::TV),
            "tv-sym" => Some(Self::TVSym),
            "xxz" => Some(Self::XXZ),
            _ => None,
        }
    }

    /// Gets the unique identifier for this model type.
    pub fn to_identifier(&self) -> &'static str {
        match self {
            ModelType::TV => "tv",
            ModelType::TVSym => "tv-sym",
            ModelType::XXZ => "xxz",
        }
    }
}

/// Result of applying a Hamiltonian on a vector while using just a single bond.
pub struct HamiltonianBondTerm {
    /// Backward-hopping part of the kinetic term.
    pub hop_backward: Option<(usize, f64)>,

    /// Forward-hopping part of the kinetic term.
    pub hop_forward: Option<(usize, f64)>,

    /// The on-site interaction term.
    pub interaction: f64,
}

/// The Hamiltonian of a quantum system.
pub struct Hamiltonian {
    /// Basis used by the Hamiltonian.
    pub basis: Basis,
    v_offset: f64,
    interaction_offset: f64,
    fermionic: bool,
}

impl Hamiltonian {
    /// Creates a new Hamiltonian for the given basis.
    ///
    /// The initial interaction offset is zero.
    pub fn new(basis: Basis, model_type: ModelType) -> Self {
        let (v_offset, fermionic) = match model_type {
            ModelType::TV => (0.0, true),
            ModelType::TVSym => (-0.5, true),
            ModelType::XXZ => (-0.5, false),
        };
        let interaction_offset = 0.0;
        Hamiltonian {
            basis,
            v_offset,
            interaction_offset,
            fermionic,
        }
    }

    /// Gets the interaction offset.
    ///
    /// The interaction offset is an additional sum term for all interaction terms (i.e. the
    /// diagonal terms).
    pub fn get_interaction_offset(&self) -> f64 {
        self.interaction_offset
    }

    /// Sets the interaction offset.
    pub fn set_interaction_offset(&mut self, interaction_offset: f64) {
        self.interaction_offset = interaction_offset;
    }

    /// Gets the dimension of the Hamiltonian.
    pub fn dimension(&self) -> usize {
        self.basis.dimension()
    }

    /// Evaluates the diagonal interaction term for a single bond and basis state.
    pub fn evaluate_interaction_term(&self, bond: &Bond, state_j: BitCombination) -> f64 {
        let v_offset = self.v_offset;
        let site_0 = bond.site_index_0;
        let site_1 = bond.site_index_1;
        let bit_0 = state_j.bit_at(site_0);
        let bit_1 = state_j.bit_at(site_1);
        let spin_0 = (bit_0 as u8 as f64) + v_offset;
        let spin_1 = (bit_1 as u8 as f64) + v_offset;
        bond.v * spin_0 * spin_1
    }

    /// Evaluates all non-zero Hamiltonian terms caused by one bond.
    pub fn evaluate_bond(&self, bond: &Bond, state_j: BitCombination) -> HamiltonianBondTerm {
        let basis = &self.basis;
        let v_offset = self.v_offset;
        let mut result = HamiltonianBondTerm {
            hop_backward: None,
            hop_forward: None,
            interaction: 0.0,
        };
        // Find indices j for which H(i,j) is non-zero.
        let site_0 = bond.site_index_0;
        let site_1 = bond.site_index_1;
        let bit_0 = state_j.bit_at(site_0);
        let bit_1 = state_j.bit_at(site_1);
        let spin_0 = (bit_0 as u8 as f64) + v_offset;
        let spin_1 = (bit_1 as u8 as f64) + v_offset;
        result.interaction = bond.v * spin_0 * spin_1;
        if bit_0 != bit_1 {
            // This requires some explanation.
            //
            // Remember that the following Hamiltonian is used:
            //
            // H = -t (kinetic terms) + V (potential terms)
            //
            // This means that if we have either
            //     * a bosonic model or
            //     * a fermionic model but the hopped particles in between are even
            // then we have -t for the kinetic part.
            let mut sign_flip = false;
            if self.fermionic {
                let odd_jump = state_j.count_particles_between(site_0, site_1) & 1;
                sign_flip = odd_jump != 0;
            }
            let t = if sign_flip { bond.t } else { -bond.t };
            // Flipping is possible.
            let mut flipped_state = state_j;
            flipped_state.flip(site_0);
            flipped_state.flip(site_1);
            let i = basis.index_of(flipped_state);
            let h_ij = t;
            if h_ij != 0.0 {
                if bit_1 {
                    // Jump from site 1 to site 0.
                    result.hop_backward = Some((i, h_ij));
                } else {
                    // Jump from site 0 to site 1.
                    result.hop_forward = Some((i, h_ij));
                }
            }
        }
        result
    }

    /// Calculates the sum of all diagonal elements.
    ///
    /// Note that this trace also includes the interaction offset.
    pub fn trace(&self) -> f64 {
        let mut sum = 0.0;
        let dim = self.dimension();
        let basis = &self.basis;
        let bonds = &basis.lattice.bonds;
        for j in 0..dim {
            let state_j = self.basis.at(j);
            for bond in bonds {
                sum += self.evaluate_interaction_term(bond, state_j);
            }
        }
        let sum_interaction_offset = (dim as f64) * self.interaction_offset;
        sum + sum_interaction_offset
    }

    /// Minimizes the trace of the Hamiltonian by subtracting the average diagonal element value.
    ///
    /// This effectively sets the trace to zero.
    pub fn minimize_trace(&mut self) {
        if self.dimension() == 0 {
            return;
        }
        self.set_interaction_offset(0.0);
        let trace = self.trace();
        let avg_diag = trace / (self.dimension() as f64);
        self.set_interaction_offset(-avg_diag);
    }

    /// Loads a sparse column from a state.
    ///
    /// This also includes the interaction offset.
    pub fn load_sparse_column_with_state(
        &self,
        j: usize,
        state_j: BitCombination,
        duplets: &mut Vec<(usize, f64)>,
    ) {
        let basis = &self.basis;
        let bonds = &basis.lattice.bonds;
        // Find indices j for which H(i,j) is non-zero.
        let mut h_ii = 0.0;
        duplets.clear();
        for bond in bonds {
            let bond_term = self.evaluate_bond(bond, state_j);
            h_ii += bond_term.interaction;
            if let Some((i, h_ij)) = bond_term.hop_backward {
                duplets.push((i, h_ij));
            }
            if let Some((i, h_ij)) = bond_term.hop_forward {
                duplets.push((i, h_ij));
            }
        }
        h_ii += self.interaction_offset;
        if h_ii != 0.0 {
            duplets.push((j, h_ii));
        }
    }

    /// Loads the j-th column in sparse representation.
    ///
    /// This method effectively applies the j-th basis vector to the Hamiltonian
    /// `H = sum_i,j H(i,j) |i)(j|` to find the non-zero coefficients `H(i,j)`.
    ///
    /// The i-th component of the j-th column vector is `H(i,j) = (i|H|j)`.
    /// This method finds the j-th column of the
    /// Hamiltonian matrix and returns a sparse representation of it.
    ///
    /// The method will write the result to a list of duplets in no particular order.
    /// Each duplet consists of a column index `i` and the value of `H(i,j)`.
    ///
    /// The Hamiltonian is self-adjoint (`H(i,j)=H(j,i)`), so the j-th column is equivalent
    /// to the j-th row.
    pub fn load_sparse_column(&self, j: usize, duplets: &mut Vec<(usize, f64)>) {
        self.load_sparse_column_with_state(j, self.basis.at(j), duplets);
    }

    /// Counts the non-zero elements in the Hamiltonian.
    pub fn count_non_zeros(&self) -> usize {
        let dim = self.dimension();
        let mut acc = 0;
        let mut duplets = Vec::new();
        for i in 0..dim {
            self.load_sparse_column(i, &mut duplets);
            acc += duplets.len();
        }
        acc
    }

    /// Builds the full Hamiltonian as a sparse row matrix.
    pub fn to_sparse_matrix(&self) -> SparseMatrix {
        let dim = self.dimension();
        let nnz = self.count_non_zeros();
        if nnz == 0 {
            panic!("to_sparse_matrix: nnz == 0");
        }
        let mut matrix = SparseMatrix::with_capacity(dim, nnz);
        let mut duplets = Vec::with_capacity(nnz);
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for row in 0..dim {
            self.load_sparse_column(row, &mut duplets);
            duplets.sort_unstable_by_key(|x| x.0);
            columns.clear();
            values.clear();
            for duplet in &duplets {
                let col = duplet.0 as u32;
                let val = duplet.1;
                columns.push(col);
                values.push(val);
            }
            matrix.push_row(&columns, &values);
        }
        matrix
    }
}

mod tests {
    use super::Hamiltonian;

    #[allow(dead_code)]
    fn test_column(
        duplets: &mut Vec<(usize, f64)>,
        h: &Hamiltonian,
        j: usize,
        expected: &[(usize, f64)],
    ) {
        h.load_sparse_column(j, duplets);
        duplets.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(duplets, expected);
    }

    #[test]
    fn test_hamiltonian_xxz_2x1() {
        use super::{Hamiltonian, ModelType};
        use crate::basis::Basis;
        use crate::lattice::{Lattice, Location, Periodicity};

        let t = 0.5;
        let v = -2.0;
        let a = v / 4.0;
        let b = t;

        let lattice_size = Location::new(2, 1);
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: false,
        };
        let mut lattice = Lattice::new(lattice_size, periodicity).unwrap();
        lattice.set_translation_invariant_bonds(t, v, t, v);

        let basis_0 = Basis::new(lattice.clone(), 0).unwrap();
        let basis_1 = Basis::new(lattice.clone(), 1).unwrap();
        let basis_2 = Basis::new(lattice.clone(), 2).unwrap();

        assert_eq!(basis_0.dimension(), 1);
        assert_eq!(basis_0.at(0).bits[0], 0b00);
        assert_eq!(basis_0.at(0).bits[1], 0);

        assert_eq!(basis_1.dimension(), 2);
        assert_eq!(basis_1.at(0).bits[0], 0b01);
        assert_eq!(basis_1.at(0).bits[1], 0);
        assert_eq!(basis_1.at(1).bits[0], 0b10);
        assert_eq!(basis_1.at(1).bits[1], 0);

        assert_eq!(basis_2.dimension(), 1);
        assert_eq!(basis_2.at(0).bits[0], 0b11);
        assert_eq!(basis_2.at(0).bits[1], 0);

        let mut duplets = Vec::new();

        for model_type in [ModelType::XXZ, ModelType::TVSym] {
            let basis_0 = Basis::new(lattice.clone(), 0).unwrap();
            let basis_1 = Basis::new(lattice.clone(), 1).unwrap();
            let basis_2 = Basis::new(lattice.clone(), 2).unwrap();
            let h_0 = Hamiltonian::new(basis_0, model_type);
            let h_1 = Hamiltonian::new(basis_1, model_type);
            let h_2 = Hamiltonian::new(basis_2, model_type);
            test_column(&mut duplets, &h_0, 0, &[(0, a)]);
            test_column(&mut duplets, &h_1, 0, &[(0, -a), (1, -b)]);
            test_column(&mut duplets, &h_1, 1, &[(0, -b), (1, -a)]);
            test_column(&mut duplets, &h_2, 0, &[(0, a)]);
        }
    }

    #[test]
    fn test_hamiltonian_tv_2x1() {
        use super::{Hamiltonian, ModelType};
        use crate::basis::Basis;
        use crate::lattice::{Lattice, Location, Periodicity};

        let t = 0.5;
        let v = -2.0;

        let lattice_size = Location::new(2, 1);
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: false,
        };
        let mut lattice = Lattice::new(lattice_size, periodicity).unwrap();
        lattice.set_translation_invariant_bonds(t, v, t, v);

        let basis_0 = Basis::new(lattice.clone(), 0).unwrap();
        let basis_1 = Basis::new(lattice.clone(), 1).unwrap();
        let basis_2 = Basis::new(lattice, 2).unwrap();

        let mut duplets = Vec::new();
        let h_0 = Hamiltonian::new(basis_0, ModelType::TV);
        let h_1 = Hamiltonian::new(basis_1, ModelType::TV);
        let h_2 = Hamiltonian::new(basis_2, ModelType::TV);
        test_column(&mut duplets, &h_0, 0, &[]);
        test_column(&mut duplets, &h_1, 0, &[(1, -t)]);
        test_column(&mut duplets, &h_1, 1, &[(0, -t)]);
        test_column(&mut duplets, &h_2, 0, &[(0, v)]);

        // Now test this with an interaction offset.
        let mut h_2 = h_2;
        h_2.set_interaction_offset(-v);
        test_column(&mut duplets, &h_2, 0, &[]);
        h_2.set_interaction_offset(-0.5 * v);
        test_column(&mut duplets, &h_2, 0, &[(0, 0.5 * v)]);
    }

    #[test]
    fn test_hamiltonian_tv_256x1() {
        use super::{Hamiltonian, ModelType};
        use crate::basis::Basis;
        use crate::lattice::{Lattice, Location, Periodicity};

        let t = -0.5;
        let v = 2.0;

        let system_width: usize = 129;
        let lattice_size = Location::new(system_width as i32, 1);
        let periodicity = Periodicity {
            periodic_in_x: true,
            periodic_in_y: false,
        };
        let mut lattice = Lattice::new(lattice_size, periodicity).unwrap();
        dbg!(lattice.bonds.last());
        lattice.set_translation_invariant_bonds(t, v, t, v);

        let basis = Basis::new(lattice.clone(), 1).unwrap();
        let dim = basis.dimension();
        assert_eq!(dim, system_width);
        let hamiltonian = Hamiltonian::new(basis, ModelType::TV);
        assert_eq!(hamiltonian.trace(), 0.0);
        let mut duplets = Vec::new();
        // Test that this Hamiltonian is diagonal:
        // H = [  0 -t  0  0 ...  0 -t ]
        //     [ -t  0 -t  0 ...  0  0 ]
        //     [  0 -t  0 -t ...  0  0 ]
        //     [  0  0 -t  0 ...  0  0 ]
        //     ...
        //     [  0  0  0  0 ...  0 -t ]
        //     [ -t  0  0  0 ... -t  0 ]
        test_column(&mut duplets, &hamiltonian, 0, &[(1, -t), (dim - 1, -t)]);
        test_column(&mut duplets, &hamiltonian, 1, &[(0, -t), (2, -t)]);

        test_column(
            &mut duplets,
            &hamiltonian,
            dim - 2,
            &[(dim - 3, -t), (dim - 1, -t)],
        );
        test_column(
            &mut duplets,
            &hamiltonian,
            dim - 1,
            &[(0, -t), (dim - 2, -t)],
        );
    }

    #[test]
    fn test_hamiltonian_xxz_3x1_2_particles() {
        use super::{Hamiltonian, ModelType};
        use crate::basis::Basis;
        use crate::lattice::{Lattice, Location, Periodicity};

        let t = 0.5;
        let v = 16.0;

        let lattice_size = Location::new(3, 1);
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: false,
        };
        let mut lattice = Lattice::new(lattice_size, periodicity).unwrap();
        lattice.set_translation_invariant_bonds(t, v, t, v);

        let basis_2 = Basis::new(lattice.clone(), 2).unwrap();

        let mut duplets = Vec::new();
        // States = {Up-Up-Down, Up-Down-Up, Down-Up-Up}
        // Interaction = v * {0.25-0.25, -0.25-0.25, -0.25+0.25} = {0, -v/2, 0}
        // (   0   -t    0  )
        // (  -t   -v/2 -t  )
        // (   0   -t    0  )
        let mut h = Hamiltonian::new(basis_2, ModelType::XXZ);
        assert_eq!(h.dimension(), 3);
        assert_eq!(h.trace(), -0.5 * v);
        test_column(&mut duplets, &h, 0, &[(1, -t)]);
        test_column(&mut duplets, &h, 1, &[(0, -t), (1, -0.5 * v), (2, -t)]);
        test_column(&mut duplets, &h, 2, &[(1, -t)]);

        // Interaction = v * {0.25-0.25, -0.25-0.25, -0.25+0.25} = {0, -v/2, 0}
        // (   v/2 -t    0   )
        // (  -t    0   -t   )
        // (   0   -t    v/2 )
        h.set_interaction_offset(0.5 * v);
        assert_eq!(h.dimension(), 3);
        assert_eq!(h.trace(), v);
        test_column(&mut duplets, &h, 0, &[(0, 0.5 * v), (1, -t)]);
        test_column(&mut duplets, &h, 1, &[(0, -t), (2, -t)]);
        test_column(&mut duplets, &h, 2, &[(1, -t), (2, 0.5 * v)]);

        h.minimize_trace();
        // The interaction offset should now be v/6 = 2.666666...
        assert!((h.get_interaction_offset() - v / 6.0).abs() < 1E-20);
        // This should set the trace to approximately zero.
        assert!(h.trace().abs() < 1E-20);
    }
}
