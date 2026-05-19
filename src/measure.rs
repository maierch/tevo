//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Measurement accumulation and output files.

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::{
    args::ImageFormat,
    cluster::ClusterSearch,
    complex::Complex,
    hamiltonian::Hamiltonian,
    image::{ColorMap, Image},
    lattice::{Lattice, Location},
};

/// Maximum cluster size tracked in measurement output.
pub const NUM_CLUSTERS: usize = 8;

/// Validates a cluster number.
fn validate_cluster_number(cluster_number: Option<usize>) -> io::Result<()> {
    if let Some(n) = cluster_number {
        if n == 0 || n > NUM_CLUSTERS {
            let msg = format!("write_measurement_svg got a bad cluster number: {n}");
            return Err(io::Error::new(io::ErrorKind::Other, msg));
        }
    }
    Ok(())
}

fn get_site_measurement_component(
    site_measurement: &SiteMeasurement,
    cluster_number: &Option<usize>,
) -> f64 {
    match cluster_number {
        None => site_measurement.spin_up_density,
        Some(n) => site_measurement.cluster_density[*n - 1],
    }
}

fn extension_for_image_format(image_format: &ImageFormat) -> &'static str {
    match image_format {
        ImageFormat::PNG => "png",
        ImageFormat::SVG => "svg",
    }
}

fn space_time_plot_file_name(cluster_number: &Option<usize>, image_format: &ImageFormat) -> String {
    let ext = extension_for_image_format(image_format);
    match cluster_number {
        None => format!("0-full-density-space-time.{ext}"),
        Some(n) => format!("{n}-cluster-density-space-time.{ext}"),
    }
}

fn two_dimensional_plot_file_name(measurement: usize, image_format: &ImageFormat) -> String {
    let ext = extension_for_image_format(image_format);
    format!("density-{measurement:03}.{ext}")
}

fn component_directory_name(output_dir: &Path, cluster_number: &Option<usize>) -> PathBuf {
    match cluster_number {
        None => output_dir.join("0-full-density-2D-plots"),
        Some(n) => output_dir.join(format!("{n}-cluster-density-2D-plots")),
    }
}

fn write_image_file(path: &Path, image: Image, image_format: &ImageFormat) -> io::Result<()> {
    match image_format {
        ImageFormat::PNG => image.upscale(8).write_png_file(path),
        ImageFormat::SVG => image.write_svg_file(path),
    }
}

/// Write measurement image for a space-time plot.
fn write_measurement_image_space_time(
    output_dir: &Path,
    lattice: &Lattice,
    site_measurements: &Vec<Vec<SiteMeasurement>>,
    colormap: &ColorMap,
    cluster_number: Option<usize>,
    image_format: &ImageFormat,
) -> io::Result<()> {
    let lattice_size = lattice.lattice_size;
    let width = lattice_size.x as usize;
    let height = site_measurements.len();
    let mut image = Image::new(width, height);
    for (measurement, site_measurements) in site_measurements.iter().enumerate() {
        let mut row_buffer: Vec<f64> = vec![0.0; width];
        for (site_index, site_measurement) in site_measurements.iter().enumerate() {
            let site_index = site_index.try_into().unwrap();
            let location = lattice.site_index_to_location(site_index);
            let x: usize = location.x.try_into().unwrap();
            let value = get_site_measurement_component(site_measurement, &cluster_number);
            row_buffer[x] += value;
        }
        for (x, value) in row_buffer.iter().enumerate() {
            let color = colormap.get_color(*value);
            let x: usize = x.try_into().unwrap();
            image.set_pixel(x, measurement, color);
        }
    }
    let fname = space_time_plot_file_name(&cluster_number, image_format);
    let path = &output_dir.join(fname);
    write_image_file(path, image, image_format)
}

/// Write measurement image for a 2D density plot.
fn write_measurement_image_two_dimensional(
    output_dir: &Path,
    lattice: &Lattice,
    site_measurements: &Vec<Vec<SiteMeasurement>>,
    colormap: &ColorMap,
    cluster_number: Option<usize>,
    image_format: &ImageFormat,
) -> io::Result<()> {
    let lattice_size = lattice.lattice_size;
    let width = lattice_size.x as usize;
    let height = lattice_size.y as usize;
    for (measurement, site_measurements) in site_measurements.iter().enumerate() {
        let mut image = Image::new(width, height);
        for (site_index, site_measurement) in site_measurements.iter().enumerate() {
            let site_index = site_index.try_into().unwrap();
            let Location { x, y } = lattice.site_index_to_location(site_index);
            let value = get_site_measurement_component(site_measurement, &cluster_number);
            let color = colormap.get_color(value);
            let x: usize = x.try_into().unwrap();
            let y: usize = y.try_into().unwrap();
            image.set_pixel(x, height - y - 1, color);
        }
        let fname = two_dimensional_plot_file_name(measurement, image_format);
        let path = &output_dir.join(fname);
        write_image_file(path, image, image_format)?;
    }
    Ok(())
}

/// Renders the measurement and writes it to a SVG file.
fn write_measurement_images(
    output_dir: &Path,
    lattice: &Lattice,
    site_measurements: &Vec<Vec<SiteMeasurement>>,
    colormap: &ColorMap,
    cluster_number: Option<usize>,
    image_format: &ImageFormat,
) -> io::Result<()> {
    validate_cluster_number(cluster_number)?;
    let component_dir = component_directory_name(output_dir, &cluster_number);
    let component_dir_exists = std::fs::exists(&component_dir)?;
    if !component_dir_exists {
        std::fs::create_dir(&component_dir)?;
    }
    write_measurement_image_space_time(
        &output_dir,
        lattice,
        site_measurements,
        colormap,
        cluster_number,
        image_format,
    )?;
    write_measurement_image_two_dimensional(
        &component_dir,
        lattice,
        site_measurements,
        colormap,
        cluster_number,
        image_format,
    )
}

/// A measurement for a given site at a given time.
#[derive(Clone)]
pub struct SiteMeasurement {
    /// Expectation value for occupation.
    ///
    /// This is the expectation value for the site being occupied by a (spin-up) particle.
    pub spin_up_density: f64,
    /// Expectation values for membership in a cluster.
    ///
    /// This is the expectation value for the site being occupied by a (spin-up) particle
    /// that is part of an M-cluster (a cluster of exactly M particles).
    ///
    /// Here `cluster_density[n]` is the density associated to an (n+1)-cluster.
    pub cluster_density: [f64; NUM_CLUSTERS],
}

impl SiteMeasurement {
    /// Creates an empty site measurement.
    pub fn zero() -> Self {
        Self {
            spin_up_density: 0.0,
            cluster_density: [0.0; 8],
        }
    }
}

/// A measurement for a given bond at a given time.
#[derive(Clone)]
pub struct BondMeasurement {
    /// Expectation value for hopping from site 1 to site 0.
    pub hop_backward: Complex,

    /// Expectation value for hopping from site 0 to site 1.
    pub hop_forward: Complex,

    /// Expectation value for the interaction term.
    pub interaction: f64,
}

impl BondMeasurement {
    /// Creates an empty bond measurement.
    pub fn zero() -> Self {
        Self {
            hop_backward: Complex::zero(),
            hop_forward: Complex::zero(),
            interaction: 0.0,
        }
    }
}

#[derive(Clone)]
struct MostOccupiedState {
    state_index: usize,
    amplitude: Complex,
}

impl MostOccupiedState {
    pub fn zero() -> Self {
        Self {
            state_index: 0,
            amplitude: Complex::zero(),
        }
    }
}

#[derive(Clone)]
struct IndexProbabilityTuple {
    state_index: usize,
    probability: f64,
}

impl IndexProbabilityTuple {
    pub fn zero() -> Self {
        Self {
            state_index: 0,
            probability: 0.0,
        }
    }
}

#[derive(Clone)]
struct MostOccupiedStates {
    most_occupied_states: Vec<MostOccupiedState>,
}

impl MostOccupiedStates {
    pub fn new(num_most_occup_states: usize) -> Self {
        let most_occupied_states = vec![MostOccupiedState::zero(); num_most_occup_states];
        Self {
            most_occupied_states,
        }
    }

    fn apply(&mut self, buffer: &mut MostOccupiedStatesBuffer, state: &[Complex]) {
        buffer
            .temp_buffer
            .sort_unstable_by(|a, b| f64::total_cmp(&b.probability, &a.probability));
        let hos = &mut self.most_occupied_states;
        for (dest, src) in hos.iter_mut().zip(buffer.temp_buffer.iter()) {
            let state_index = src.state_index;
            dest.state_index = state_index;
            dest.amplitude = state[state_index];
        }
    }
}

struct MostOccupiedStatesBuffer {
    temp_buffer: Vec<IndexProbabilityTuple>,
}

impl MostOccupiedStatesBuffer {
    fn new(basis_dimension: usize) -> Self {
        let temp_buffer = vec![IndexProbabilityTuple::zero(); basis_dimension];
        Self { temp_buffer }
    }

    fn is_enabled(&self) -> bool {
        self.temp_buffer.len() > 0
    }
}

fn measure_everything(
    hamiltonian: &Hamiltonian,
    state: &[Complex],
    site_measurements: &mut [SiteMeasurement],
    bond_measurements: &mut [BondMeasurement],
    most_occup_states: &mut MostOccupiedStates,
    most_occup_buffer: &mut MostOccupiedStatesBuffer,
) {
    let basis = &hamiltonian.basis;
    let lattice = &basis.lattice;
    let combinations = &basis.combinations;
    let mut site_indices = [0; 256];
    let mut cluster_search = ClusterSearch::new();
    let most_occup_enabled = most_occup_buffer.is_enabled();
    // Iterate over all basis states.
    for (index, ci) in state.iter().enumerate() {
        // index ... basis state index of the component
        // ci    ... probability amplitude
        let probability = ci.abs_squared();
        if most_occup_enabled {
            let ho = &mut most_occup_buffer.temp_buffer[index];
            ho.state_index = index;
            ho.probability = probability;
        }
        if probability == 0.0 {
            continue;
        }
        // Load the combination
        let state_bits = combinations.combination_at_index(index);
        let len = state_bits.locations_of_ones_into(&mut site_indices);
        cluster_search.find_clusters(lattice, &state_bits);
        let cluster_sizes = cluster_search.get_cluster_sizes();
        for site_index in &site_indices[0..len] {
            let site_index = *site_index as usize;
            let site_measurement = &mut site_measurements[site_index];

            // Density
            site_measurement.spin_up_density += probability;

            // Clusters
            let cluster_size = cluster_sizes[site_index];
            let cluster_density = &mut site_measurement.cluster_density;
            // Only works for clusters size from 1 to 8
            if cluster_size > 0 && cluster_size < 9 {
                let cluster_array_index: usize = (cluster_size - 1).into();
                cluster_density[cluster_array_index] += probability;
            }
        }
        for (bond_index, bond) in basis.lattice.bonds.iter().enumerate() {
            let b = &mut bond_measurements[bond_index];
            let bond_term = hamiltonian.evaluate_bond(bond, state_bits);
            if let Some((j, t)) = bond_term.hop_backward {
                b.hop_backward += ci.adj_mul(state[j]) * t;
            };
            if let Some((j, t)) = bond_term.hop_forward {
                b.hop_forward += ci.adj_mul(state[j]) * t;
            };
            b.interaction += probability * bond_term.interaction;
        }
    }
    if most_occup_enabled {
        most_occup_states.apply(most_occup_buffer, state);
    }
}

/// Stores all measurements collected during a simulation.
pub struct Measurements {
    hamiltonian: Hamiltonian,
    times: Vec<f64>,
    site_measurements: Vec<Vec<SiteMeasurement>>,
    bond_measurements: Vec<Vec<BondMeasurement>>,
    most_occup_states: Vec<MostOccupiedStates>,
    most_occup_buffer: MostOccupiedStatesBuffer,
}

impl Measurements {
    /// Creates an object that can hold `num_measurements` for the given system.
    pub fn new(
        hamiltonian: Hamiltonian,
        num_measurements: usize,
        num_most_occupied: usize,
    ) -> Self {
        let basis = &hamiltonian.basis;
        let lattice = &basis.lattice;
        let num_sites = lattice.number_of_sites as usize;
        let num_bonds = lattice.bonds.len();
        let times = vec![0.0; num_measurements];
        let site_measurements = vec![vec![SiteMeasurement::zero(); num_sites]; num_measurements];
        let bond_measurements = vec![vec![BondMeasurement::zero(); num_bonds]; num_measurements];
        let num_most_occupied = num_most_occupied.min(basis.dimension());
        let zero_most_occupied = MostOccupiedStates::new(num_most_occupied);
        let most_occup_states = vec![zero_most_occupied; num_measurements];
        let most_occup_buffer = MostOccupiedStatesBuffer::new(basis.dimension());
        Self {
            hamiltonian,
            times,
            site_measurements,
            bond_measurements,
            most_occup_states,
            most_occup_buffer,
        }
    }

    // pub fn get_hamiltonian(&self) -> &Hamiltonian {
    //     &self.hamiltonian
    // }

    /// Performs all measurements on the given state state using the provided Hamiltonian.
    pub fn measure(&mut self, measurement: usize, time: f64, state: &[Complex]) {
        self.times[measurement] = time;
        measure_everything(
            &self.hamiltonian,
            state,
            &mut self.site_measurements[measurement],
            &mut self.bond_measurements[measurement],
            &mut self.most_occup_states[measurement],
            &mut self.most_occup_buffer,
        );
    }

    /// Writes site measurements as CSV text
    pub fn write_site_measurements_as_csv(&self, writer: &mut impl Write) -> io::Result<()> {
        let times = &self.times;
        let lattice = &self.hamiltonian.basis.lattice;
        let site_measurements = &self.site_measurements;
        writeln!(
            writer,
            "\"measurement\",\"time\",\"x\",\"y\",\"spin_up_density\""
        )?;
        for (mt, measurements) in times.iter().enumerate().zip(site_measurements.iter()) {
            let (measurement, time) = mt;
            for (site_index, site_measurement) in measurements.iter().enumerate() {
                let site_index = site_index.try_into().unwrap();
                let Location { x, y } = lattice.site_index_to_location(site_index);
                let sud = site_measurement.spin_up_density;
                writeln!(writer, "{measurement},{time},{x},{y},{sud:.E}")?;
            }
        }
        Ok(())
    }

    /// Writes bond measurements as CSV text
    pub fn write_bond_measurements_as_csv(&self, writer: &mut impl Write) -> io::Result<()> {
        let times = &self.times;
        let bond_measurements = &self.bond_measurements;
        let lattice = &self.hamiltonian.basis.lattice;
        let bonds = &lattice.bonds;
        write!(writer, "\"measurement\",\"time\",")?;
        write!(writer, "\"x0\",\"y0\",\"x1\",\"y1\",\"t\",\"v\",")?;
        write!(writer, "\"hop_backward_real\",\"hop_backward_imag\",")?;
        write!(writer, "\"hop_forward_real\",\"hop_forward_imag\",")?;
        writeln!(writer, "\"interaction\"")?;
        for (mt, measurements) in times.iter().enumerate().zip(bond_measurements.iter()) {
            let (measurement, time) = mt;
            for (bond_index, bond_measurement) in measurements.iter().enumerate() {
                let bond = bonds[bond_index];
                let Location { x: x0, y: y0 } = lattice.site_index_to_location(bond.site_index_0);
                let Location { x: x1, y: y1 } = lattice.site_index_to_location(bond.site_index_1);
                let t = bond.t;
                let v = bond.v;
                let br = bond_measurement.hop_backward.real;
                let bi = bond_measurement.hop_backward.imag;
                let fr = bond_measurement.hop_forward.real;
                let fi = bond_measurement.hop_forward.imag;
                let i = bond_measurement.interaction;
                write!(writer, "{measurement},{time},")?;
                write!(writer, "{x0},{y0},{x1},{y1},{t},{v},")?;
                writeln!(writer, "{br:.E},{bi:.E},{fr:.E},{fi:.E},{i:.E}")?;
            }
        }
        Ok(())
    }

    /// Writes site cluster expectation values as CSV text.
    pub fn write_cluster_sizes_as_csv(&self, writer: &mut impl Write) -> io::Result<()> {
        let lattice = &self.hamiltonian.basis.lattice;
        let site_measurements = &self.site_measurements;
        let num_particles = self.hamiltonian.basis.number_of_particles();
        write!(writer, "\"measurement\",\"x\",\"y\"")?;
        let max_cluster_size = num_particles.min(9);
        for cluster_size in 1..=max_cluster_size {
            write!(writer, ",\"c{cluster_size}\"")?;
        }
        writeln!(writer, "")?;
        for (mi, measurements) in site_measurements.iter().enumerate() {
            assert!(measurements.len() <= 256); // This justifies the conversion (as u8) below.
            for (site_index, site_measurement) in measurements.iter().enumerate() {
                let site_index = site_index as u8;
                let Location { x, y } = lattice.site_index_to_location(site_index);
                let cluster_density = &site_measurement.cluster_density;
                write!(writer, "{mi},{x},{y}")?;
                for i in 0..max_cluster_size {
                    write!(writer, ",{0:.E}", cluster_density[i])?;
                }
                writeln!(writer, "")?;
            }
        }
        Ok(())
    }

    fn write_most_occupied_as_csv(&self, writer: &mut impl Write) -> io::Result<()> {
        let hamiltonian = &self.hamiltonian;
        let basis = &hamiltonian.basis;
        let lattice = &basis.lattice;
        let bonds = &lattice.bonds;
        let num_particles = basis.number_of_particles();
        write!(
            writer,
            "\"measurement\",\"c_real\",\"c_imag\",\"interaction\""
        )?;
        for particle_index in 1..=num_particles {
            write!(writer, ",\"x{particle_index}\",\"y{particle_index}\"")?;
        }
        writeln!(writer, "")?;
        let mut site_indices = [0; 256];
        for (mi, measurements) in self.most_occup_states.iter().enumerate() {
            let mos = &measurements.most_occupied_states;
            for m in mos {
                let real = m.amplitude.real;
                let imag = m.amplitude.imag;
                let state_bits = basis.at(m.state_index);
                let mut interaction = 0.0;
                for bond in bonds {
                    interaction += hamiltonian.evaluate_interaction_term(bond, state_bits);
                }
                let np = state_bits.locations_of_ones_into(&mut site_indices);
                write!(writer, "{mi},{real:.E},{imag:.E},{interaction}")?;
                for site_index in site_indices.iter().take(np) {
                    let location = basis.lattice.site_index_to_location(*site_index);
                    let x = location.x;
                    let y = location.y;
                    write!(writer, ",{x},{y}")?;
                }
                writeln!(writer, "")?;
            }
        }
        Ok(())
    }

    fn write_output_files(&self, output_dir: &Path, image_format: &ImageFormat) -> io::Result<()> {
        let colormap = ColorMap::rainbow();
        let lattice = &self.hamiltonian.basis.lattice;
        let num_particles = self.hamiltonian.basis.number_of_particles();
        let site_measurements = &self.site_measurements;
        write_measurement_images(
            output_dir,
            lattice,
            site_measurements,
            &colormap,
            None,
            image_format,
        )?;
        for cluster_number in 1..=NUM_CLUSTERS.min(num_particles) {
            let cl_num = Some(cluster_number);
            write_measurement_images(
                output_dir,
                lattice,
                site_measurements,
                &colormap,
                cl_num,
                image_format,
            )?;
        }
        Ok(())
    }

    /// Writes all measurements to the file system.
    ///
    /// Note: this method expects the output directory to already exist.
    /// Otherwise, it will likely panic.
    pub fn write_files(
        &self,
        output_dir: &Path,
        image_format: &Option<ImageFormat>,
    ) -> io::Result<()> {
        let site_measurements_file = File::create(output_dir.join("site-measurements.csv"))?;
        self.write_site_measurements_as_csv(&mut BufWriter::new(site_measurements_file))?;
        let bond_measurements_file = File::create(output_dir.join("bond-measurements.csv"))?;
        self.write_bond_measurements_as_csv(&mut BufWriter::new(bond_measurements_file))?;
        let cluster_measurements_file = File::create(output_dir.join("cluster-measurements.csv"))?;
        self.write_cluster_sizes_as_csv(&mut BufWriter::new(cluster_measurements_file))?;
        if self.most_occup_buffer.is_enabled() {
            let most_occupied_states_file =
                File::create(output_dir.join("most-occupied-states.csv"))?;
            self.write_most_occupied_as_csv(&mut BufWriter::new(most_occupied_states_file))?;
        }
        if let Some(image_format) = image_format {
            self.write_output_files(output_dir, image_format)?;
        }
        Ok(())
    }
}

mod test {

    #[test]
    fn test_measure_everything() {
        use super::{
            BondMeasurement, MostOccupiedStates, MostOccupiedStatesBuffer, SiteMeasurement,
            measure_everything,
        };
        use crate::basis::Basis;
        // use crate::complex::{normalize_complex_vector, Complex};
        use crate::complex::Complex;
        use crate::hamiltonian::{Hamiltonian, ModelType};
        use crate::lattice::{Lattice, Location, Periodicity};

        // Test with a 3x1 grid and two particles.
        let t = -0.5;
        let v = 2.0;
        let num_particles = 2;
        let lattice_size = Location::new(3, 1);
        let periodicity = Periodicity {
            periodic_in_x: false,
            periodic_in_y: false,
        };
        let mut lattice = Lattice::new(lattice_size, periodicity).unwrap();
        lattice.set_translation_invariant_bonds(t, v, t, v);
        let basis = Basis::new(lattice, num_particles).unwrap();
        let dim = basis.dimension();
        let num_sites = basis.lattice.number_of_sites as usize;
        let num_bonds = basis.lattice.bonds.len();
        assert_eq!(num_sites, 3);
        assert_eq!(num_bonds, 2);
        let mut site_measurements = vec![SiteMeasurement::zero(); num_sites];
        let mut bond_measurements = vec![BondMeasurement::zero(); num_bonds];
        assert_eq!(basis.dimension(), dim);
        assert_eq!(basis.at(0).bits[0], 0b011);
        assert_eq!(basis.at(1).bits[0], 0b101);
        assert_eq!(basis.at(2).bits[0], 0b110);
        let basis_dimension = basis.dimension();
        let hamiltonian = Hamiltonian::new(basis, ModelType::XXZ);

        let num_most_occupied = 3;
        let mut most_occup_states = MostOccupiedStates::new(num_most_occupied);
        let mut most_occup_buffer = MostOccupiedStatesBuffer::new(basis_dimension);

        // |110>
        let state = vec![Complex::new(1.0, 0.0), Complex::zero(), Complex::zero()];
        measure_everything(
            &hamiltonian,
            &state,
            &mut site_measurements,
            &mut bond_measurements,
            &mut most_occup_states,
            &mut most_occup_buffer,
        );

        assert_eq!(site_measurements[0].spin_up_density, 1.0); // Sz[0]
        assert_eq!(site_measurements[1].spin_up_density, 1.0); // Sz[1]
        assert_eq!(site_measurements[2].spin_up_density, 0.0); // Sz[2]
        assert_eq!(bond_measurements[0].hop_backward, Complex::zero()); // S+[0] S-[1]
        assert_eq!(bond_measurements[1].hop_backward, Complex::zero()); // S+[1] S-[2]
        assert_eq!(bond_measurements[0].hop_forward, Complex::zero()); // S-[1] S-[0]
        assert_eq!(bond_measurements[1].hop_forward, Complex::zero()); // S-[2] S-[1]
        assert_eq!(bond_measurements[0].interaction, 0.25 * v); // Sz[0] Sz[1]
        assert_eq!(bond_measurements[1].interaction, -0.25 * v); // Sz[1] Sz[2]

        let mos = &most_occup_states.most_occupied_states;
        assert_eq!(mos[0].state_index, 0);
        assert_eq!(mos[0].amplitude, Complex::new(1.0, 0.0));

        // (|110>-i|101>+|011>)/sqrt(3)
        let isq = (1.0 / 3.0_f64).sqrt();
        let state = vec![
            Complex::new(isq, 0.0),
            Complex::new(0.0, -isq),
            Complex::new(isq, 0.0),
        ];
        site_measurements.fill(SiteMeasurement::zero());
        bond_measurements.fill(BondMeasurement::zero());
        let mut most_occup_buffer = MostOccupiedStatesBuffer::new(0);
        measure_everything(
            &hamiltonian,
            &state,
            &mut site_measurements,
            &mut bond_measurements,
            &mut most_occup_states,
            &mut most_occup_buffer,
        );

        assert!((site_measurements[0].spin_up_density - 2.0 / 3.0).abs() < 1.0E-10); // Sz[0]
        assert!((site_measurements[1].spin_up_density - 2.0 / 3.0).abs() < 1.0E-10); // Sz[1]
        assert!((site_measurements[2].spin_up_density - 2.0 / 3.0).abs() < 1.0E-10); // Sz[2]

        // Hop from 1 to 0: |011> to i<101|
        // t/3 (<110|+i<101|+<011|) S+0 S-1 (|110>-i|101>+|011>) = i*t/3
        assert_eq!(
            bond_measurements[0].hop_backward,
            Complex::new(0.0, t / 3.0)
        );

        // Hop from 2 to 1: -i|101> to <100|
        // t/3 (<110|+i<101|+<011|) S+1 S-2 (|110>-i|101>+|011>) = -i*t/3
        assert_eq!(
            bond_measurements[1].hop_backward,
            Complex::new(0.0, -t / 3.0)
        );

        // Hop from 0 to 1: -i|101> to <011|
        // t/3 (<110|+i<101|+<011|) S+1 S-0 (|110>-i|101>+|011>) = -i*t/3
        assert_eq!(
            bond_measurements[0].hop_forward,
            Complex::new(0.0, -t / 3.0)
        );

        // Hop from 1 to 2: -|110> to i<010|
        // t/3 (<110|+i<101|+<011|) S+1 S-0 (|110>-i|101>+|011>) = i*t/3
        assert_eq!(bond_measurements[1].hop_forward, Complex::new(0.0, t / 3.0));

        // v/3 (<110|+i<101|+<011|) Sz0 Sz1 (|110>-i|101>+|011>) = v/3 (1/4 - 1/4 - 1/4)
        assert_eq!(bond_measurements[0].interaction, -v / 12.0);

        // v/3 (<110|+i<101|+<011|) Sz1 Sz2 (|110>-i|101>+|011>) = v/3 (-1/4 - 1/4 + 1/4)
        assert_eq!(bond_measurements[1].interaction, -v / 12.0);

        // (0*|110>-2*|101>+i*|011>)/sqrt(3)
        let c0 = Complex::zero();
        let c1 = Complex::new(-2.0 * isq, 0.0);
        let c2 = Complex::new(0.0, isq);
        let state = vec![c0, c1, c2];
        let mut most_occup_buffer = MostOccupiedStatesBuffer::new(basis_dimension);
        measure_everything(
            &hamiltonian,
            &state,
            &mut site_measurements,
            &mut bond_measurements,
            &mut most_occup_states,
            &mut most_occup_buffer,
        );
        let mos = &most_occup_states.most_occupied_states;
        assert_eq!(mos[0].state_index, 1);
        assert_eq!(mos[0].amplitude, c1);
        assert_eq!(mos[1].state_index, 2);
        assert_eq!(mos[1].amplitude, c2);
        assert_eq!(mos[2].state_index, 0);
        assert_eq!(mos[2].amplitude, c0);
    }
}
