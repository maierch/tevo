//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Case setup for the quantum bowling simulation.

use std::error::Error;

use crate::{
    args::Arguments,
    basis::Basis,
    complex::{Complex, SparseComplexVector},
    hamiltonian::Hamiltonian,
    lattice::{Lattice, Location, Periodicity},
};

/// Fully prepared data for one quantum bowling run.
pub struct QuantumBowlingCase {
    /// Name used for the output directory.
    pub name: String,

    /// Full Hamiltonian for the simulation.
    pub hamiltonian: Hamiltonian,

    /// Initial state in sparse form.
    pub initial_state: SparseComplexVector,

    /// One-column Hamiltonian used for the vertical Gaussian profile.
    pub vertical_hamiltonian: Hamiltonian,

    /// Groundstate amplitudes in the vertical direction.
    pub vertical_groundstate: Vec<f64>,
}

fn make_gaussian(x_start: i32, x_len: i32, x0: f64, k0: f64, sigma: f64) -> Vec<Complex> {
    if x_len == 0 {
        return Vec::new();
    }
    let mut v = Vec::with_capacity(x_len as usize);
    let x_end = x_start + x_len;
    let mut norm2 = 0.0;
    for x in x_start..x_end {
        let dx = (x as f64) - x0;
        let z = Complex::new(-dx * dx / (2.0 * sigma * sigma), dx * k0).exp();
        norm2 += z.abs_squared();
        v.push(z);
    }
    if norm2 == 0.0 {
        panic!("gaussian particle created with zero norm");
    }
    let s = 1.0 / norm2.sqrt();
    for z in v.iter_mut() {
        *z = *z * s;
    }
    v
}

impl QuantumBowlingCase {
    /// Creates a new case.
    ///
    /// This initializes what is needed to start the simulation.
    pub fn new(args: &Arguments) -> Result<Self, Box<dyn Error>> {
        let cs = &args.case_settings;
        let lattice_size = cs.lattice_size;
        let periodicity = cs.lattice_periodicity;
        let mut lattice = Lattice::new(lattice_size, periodicity).map_err(|err| err.boxed())?;
        lattice.set_translation_invariant_bonds(cs.tx, cs.vx, cs.ty, cs.vy);
        let mut particles = Vec::new();
        if cs.gaussian_size.x * cs.gaussian_size.y > 0 {
            particles.push(Location::new(1, 0));
        }
        let wall_end = cs.wall_start + cs.wall_size;
        for x in cs.wall_start.x..wall_end.x {
            for y in cs.wall_start.y..wall_end.y {
                particles.push(Location::new(x, y));
            }
        }
        for location in &cs.extra_wall_sites {
            particles.push(location.clone());
        }
        let num_particles = particles.len();
        let basis = Basis::new(lattice.clone(), num_particles)
            .ok_or("parameters lead to an invalid basis")?;

        let y_periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: periodicity.periodic_in_y,
        };
        if periodicity.periodic_in_y && lattice_size.y == 2 {
            return Err("periodicity in y-direction is not allowed in 2-chain systems".into());
        }
        let mut vertical_lattice =
            Lattice::new(Location::new(1, lattice_size.y), y_periodicity).unwrap();
        vertical_lattice.set_translation_invariant_bonds(cs.tx, cs.vx, cs.ty, cs.vy);
        // dbg!(&vertical_lattice.bonds);
        let vertical_basis =
            Basis::new(vertical_lattice, 1).expect("parameters lead to an invalid vertical basis");
        let vertical_hamiltonian = Hamiltonian::new(vertical_basis, cs.model_type);
        // dbg!(&vertical_hamiltonian.to_sparse_matrix());
        let periodic_in_y = periodicity.periodic_in_y;
        let vertical_groundstate = make_vertical_groundstate(&vertical_hamiltonian, periodic_in_y)?;
        let x0 = cs.gaussian_center_x as f64;
        let k0 = cs.gaussian_momentum_x as f64;
        let mut duplets = Vec::new();
        let gaussian_end = cs.gaussian_start + cs.gaussian_size;
        let sigma = cs.gaussian_sigma;
        let gaussian = make_gaussian(cs.gaussian_start.x, cs.gaussian_size.x, x0, k0, sigma);
        if gaussian.is_empty() {
            let state_index = basis
                .state_index_from_particle_locations(&particles)
                .map_err(|err| err.boxed())?;
            duplets.push((state_index, Complex::new(1.0, 0.0)));
        } else {
            for x in cs.gaussian_start.x..gaussian_end.x {
                let g_x = gaussian[(x - cs.gaussian_start.x) as usize];
                for y in cs.gaussian_start.y..gaussian_end.y {
                    let location = Location::new(x, y);
                    lattice
                        .expect_contains(&location)
                        .map_err(|err| err.boxed())?;
                    let g_y = vertical_groundstate[y as usize];
                    particles[0] = location;
                    let state_index = basis
                        .state_index_from_particle_locations(&particles)
                        .map_err(|err| err.boxed())?;
                    duplets.push((state_index, g_x * g_y));
                }
            }
        }
        let initial_state = SparseComplexVector::from_unsorted_duplets(basis.dimension(), duplets);
        let hamiltonian = Hamiltonian::new(basis, cs.model_type);
        let model = cs.model_type.to_identifier();
        let tx = cs.tx;
        let ty = cs.ty;
        let vx = cs.vx;
        let vy = cs.vy;
        let s = args.simulation_settings.time_step;
        let name = format!("{model}_tx{tx}_vx{vx}_ty{ty}_vy{vy}_s{s:.e}");
        Ok(QuantumBowlingCase {
            name,
            hamiltonian,
            initial_state,
            vertical_hamiltonian,
            vertical_groundstate,
        })
    }
}

fn make_vertical_groundstate(
    vertical_hamiltonian: &Hamiltonian,
    periodic_in_y: bool,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let len = vertical_hamiltonian.basis.lattice.lattice_size.y as usize;
    if periodic_in_y || len == 2 {
        // If there is periodicity in y-direction or if we only have 2 chains,
        // we choose a very simple distribution in y-direction.
        if len == 0 {
            return Err("make_vertical_groundstate: lattice_size.y == 0".into());
        }
        let s = (1.0_f64 / (len as f64)).sqrt();
        Ok(vec![s; len])
    } else if len == 1 {
        Ok(vec![1.0])
    } else {
        vertical_hamiltonian
            .to_sparse_matrix()
            .estimate_groundstate()
    }
}

mod test {
    #[test]
    fn test_gaussian() {
        use super::make_gaussian;
        use crate::complex::Complex;
        use std::f64::consts::PI;

        let x_start = 0;
        let x_len = 3;
        let x0 = 1.0;
        let k0 = 0.5 * PI;
        let sigma = 3.0;
        let actual = make_gaussian(x_start, x_len, x0, k0, sigma);
        let e = (-1.0_f64 / 18.0).exp();
        let s = 1.0 / (e * e + 1.0 + e * e).sqrt();
        let expected = vec![
            Complex::new(0.0, -e) * s,
            Complex::new(1.0, 0.0) * s,
            Complex::new(0.0, e) * s,
        ];
        let dist_sq = actual
            .iter()
            .zip(expected.iter())
            .fold(0.0_f64, |acc, (a, b)| acc + (*a - *b).abs_squared());
        assert!(dist_sq < 1.0E-8);
    }
}
