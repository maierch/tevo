//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! File IO and reporting utilities.

use crate::basis::Basis;
use crate::complex::Complex;
use crate::hamiltonian::Hamiltonian;
use std::fs::{File, Metadata};
use std::io::{BufReader, BufWriter, Read, Write};
#[cfg(target_os = "linux")]
use std::os::unix::fs::MetadataExt;
#[cfg(target_os = "windows")]
use std::os::windows::fs::MetadataExt;
use std::path::Path;
use std::{fs, io};

/// The current program version
pub const PROGRAM_VERSION: &str = "0.7.6";

/// Creates a histogram from a state.
pub fn histogram_from_state(state: &[Complex]) -> Vec<(usize, f64)> {
    let n_bins = 30;
    let mut histogram = vec![(0, 0.0); n_bins];
    for (n, bin) in histogram.iter_mut().enumerate().skip(1) {
        let exponent = (n as f64) - (n_bins as f64);
        bin.1 = 10_f64.powf(exponent);
    }
    for c in state {
        let x = c.abs_squared();
        let index = match histogram.binary_search_by(|y| y.1.total_cmp(&x)) {
            Ok(index) => index,
            Err(index) => index - 1,
        };
        let index = index.min(n_bins - 1);
        histogram[index].0 += 1;
    }
    histogram
}

/// Subprogram for writing out a Hamiltonian as CSV data.
pub fn write_hamiltonian_to_csv(
    writer: &mut impl Write,
    hamiltonian: &Hamiltonian,
) -> io::Result<()> {
    let dim = hamiltonian.dimension();
    writeln!(writer, "\"row\",\"col\",\"value\"")?;
    let mut duplets = Vec::new();
    for i in 0..dim {
        hamiltonian.load_sparse_column(i, &mut duplets);
        for duplet in &duplets {
            writeln!(writer, "{},{},{}", i, duplet.0, duplet.1)?;
        }
    }
    Ok(())
}

/// Subprogram for writing out a Hamiltonian as CSV data.
pub fn write_hamiltonian_to_csv_file(path: &Path, hamiltonian: &Hamiltonian) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    write_hamiltonian_to_csv(&mut w, hamiltonian)?;
    Ok(())
}

/// Subprogram for writing out a basis as binary data.
pub fn write_basis_to_binary(writer: &mut impl Write, basis: &Basis) -> io::Result<()> {
    let magic: u64 = 0xBA515000;
    let version: u64 = 2;
    let dim = basis.dimension() as u64;
    let num_particles = basis.number_of_particles() as u64;
    let lattice_size = basis.lattice.lattice_size;
    let nx = lattice_size.x as u64;
    let ny = lattice_size.y as u64;
    let mut flags = 0 as u64;
    if basis.lattice.periodicity.periodic_in_x {
        flags |= 1;
    }
    if basis.lattice.periodicity.periodic_in_y {
        flags |= 2;
    }
    writer.write(&magic.to_le_bytes())?;
    writer.write(&version.to_le_bytes())?;
    writer.write(&dim.to_le_bytes())?;
    writer.write(&num_particles.to_le_bytes())?;
    writer.write(&nx.to_le_bytes())?;
    writer.write(&ny.to_le_bytes())?;
    writer.write(&flags.to_le_bytes())?;

    let num_particles = basis.number_of_particles();
    let mut dest: [u8; 256] = [0; 256];
    for index in 0..basis.dimension() {
        let state = basis.at(index);
        state.locations_of_ones_into(&mut dest);
        writer.write(&dest[0..num_particles])?;
    }
    Ok(())
}

/// Subprogram for writing out a state as binary data.
///
/// The state will be normalized before being written.
///
/// Panics if the state has a zero norm.
pub fn write_state_to_binary(writer: &mut impl Write, state: &[Complex]) -> io::Result<()> {
    let mut norm2 = 0.0;
    for c in state.iter() {
        let abs_sq = c.abs_squared();
        norm2 += abs_sq;
    }
    if norm2 == 0.0 {
        panic!("write_state_to_csv: state is the zero vector");
    }
    let factor = 1.0 / norm2.sqrt();
    for c in state {
        let z = *c;
        let real = z.real * factor;
        writer.write(&real.to_le_bytes())?;
        let imag = z.imag * factor;
        writer.write(&imag.to_le_bytes())?;
    }
    Ok(())
}

/// Writes a basis to a binary file.
pub fn write_basis_to_binary_file(path: &Path, basis: &Basis) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    write_basis_to_binary(&mut w, basis)?;
    Ok(())
}

/// Writes a state to a binary file.
pub fn write_state_to_binary_file(path: &Path, state: &[Complex]) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    write_state_to_binary(&mut w, state)?;
    Ok(())
}

#[inline]
fn read_f64(reader: &mut BufReader<File>) -> io::Result<f64> {
    let mut bytes = [0_u8; 8];
    reader.read(&mut bytes)?;
    return Ok(f64::from_le_bytes(bytes));
}

#[cfg(target_os = "linux")]
fn meta_data_size(meta: Metadata) -> usize {
    meta.size() as usize
}

#[cfg(target_os = "windows")]
fn meta_data_size(meta: Metadata) -> usize {
    meta.file_size() as usize
}

/// Reads a state file.
pub fn read_state_from_file(path: &Path) -> io::Result<Vec<Complex>> {
    let file = File::open(path)?;
    let meta = file.metadata()?;
    let file_size = meta_data_size(meta);
    let dim = file_size / 16;
    assert_eq!(file_size, dim * 16, "bad state file size");
    let mut coefficients = Vec::with_capacity(dim);
    let mut reader = BufReader::new(file);
    for _ in 0..dim {
        let real = read_f64(&mut reader)?;
        let imag = read_f64(&mut reader)?;
        coefficients.push(Complex { real, imag });
    }
    Ok(coefficients)
}

/// Tries to find out the hosts name by invoking the "hostname" command.
///
/// This will only be invoked if all other methods of getting the hostname fail.
fn get_hostname_via_command() -> Option<String> {
    let out = std::process::Command::new("hostname").output().ok()?;
    let hostname = String::from_utf8(out.stdout).ok()?;
    Some(hostname.trim().to_string())
}

/// Tries to find out the hosts name.
///
/// The method will read the environment variables `"HOST"` and `"HOSTNAME"` to find the host name
/// if possible and otherwise will return `None`.
///
/// Finally, the method will try to invoke the `hostname` command.
pub fn get_hostname() -> Option<String> {
    match std::env::var("HOST") {
        Err(_) => match std::env::var("HOSTNAME") {
            Err(_) => get_hostname_via_command(),
            Ok(s) => Some(s),
        },
        Ok(s) => Some(s),
    }
}

/// Information about the system and calculation.
///
/// This structure is for information about the system and the calculation that can
/// be gathered before the simulation starts.
pub struct InfoData {
    /// Basis dimension
    pub basis_dimension: usize,

    /// MiB of RAM used by the Hamilton matrix.
    pub hamilton_matrix_mib: f64,

    /// Number of non-zero elements in the Hamilton matrix.
    pub hamilton_matrix_nnz: usize,

    /// Hostname of the device this calculation is being performed on.
    pub hostname: Option<String>,

    /// Size of the lattice in x-direction
    pub lattice_width: i32,

    /// Size of the lattice in y-direction
    pub lattice_height: i32,

    /// Number of measurements in total
    pub num_measurements: usize,

    /// Total elapsed program time.
    pub total_time: f64,

    /// Elapsed seconds for building the Hamilton matrix
    pub setup_time: f64,

    /// Elapsed seconds for the simulation
    pub simulation_time: f64,

    /// Total number of perfored time steps.
    pub total_steps: usize,

    /// Timesteps performed per measurement
    pub steps_per_measurement: usize,

    /// Total norm of the final vector (without normalization).
    ///
    /// This is basically the product of all vector norms for each step.
    pub total_norm: Option<f64>,

    /// Largest norm that occured for a vector during a single step.
    pub max_step_norm: f64,

    /// Offset used on each diagonal elements of the Hamiltonian.
    pub interaction_offset: f64,

    /// Trace of the final sparse Hamiltonian matrix.
    pub hamiltonian_trace: f64,

    /// Norm after projection.
    pub projection_norm: f64,
}

impl InfoData {
    /// Writes this info struct as a CSV file to the specified output directory.
    pub fn write_csv_file(&self, file_path: &Path) -> io::Result<()> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);
        let hostname = match &self.hostname {
            None => "unknown",
            Some(s) => &s,
        };
        let max_step_norm_dev = self.max_step_norm - 1.0;
        writeln!(writer, "\"property\",\"value\"")?;
        writeln!(writer, "\"hostname\",\"{}\"", hostname)?;
        writeln!(writer, "\"basis_dimension\",{}", self.basis_dimension)?;
        let nnz = self.hamilton_matrix_nnz;
        writeln!(
            writer,
            "\"hamilton_matrix_mib\",{}",
            self.hamilton_matrix_mib
        )?;
        writeln!(writer, "\"hamilton_matrix_nnz\",{}", nnz)?;
        writeln!(writer, "\"lattice_width\",{}", self.lattice_width)?;
        writeln!(writer, "\"lattice_height\",{}", self.lattice_height)?;
        writeln!(writer, "\"num_measurements\",{}", self.num_measurements)?;
        writeln!(writer, "\"total_seconds\",{}", self.total_time)?;
        writeln!(writer, "\"setup_seconds\",{}", self.setup_time)?;
        writeln!(
            writer,
            "\"time_evolution_seconds\",{}",
            self.simulation_time
        )?;
        writeln!(writer, "\"total_steps\",{}", self.total_steps)?;
        writeln!(
            writer,
            "\"steps_per_measurement\",{}",
            self.steps_per_measurement
        )?;
        if let Some(total_norm) = self.total_norm {
            let deviation = total_norm - 1.0;
            writeln!(writer, "\"total_norm_deviation\",{deviation:.12e}")?;
        } else {
            writeln!(writer, "\"total_norm_deviation\",0")?;
        }
        writeln!(writer, "\"version\",{}", PROGRAM_VERSION)?;
        writeln!(writer, "\"max_step_norm\",{0:.12e}", self.max_step_norm)?;
        writeln!(
            writer,
            "\"max_step_norm_deviation\",{0:.12e}",
            max_step_norm_dev
        )?;
        writeln!(
            writer,
            "\"hamiltonian_trace\",{0:.12e}",
            self.hamiltonian_trace
        )?;
        writeln!(
            writer,
            "\"interaction_offset\",{0:.12e}",
            self.interaction_offset
        )?;
        writeln!(writer, "\"projection_norm\",{0:.12e}", self.projection_norm)?;
        Ok(())
    }
}

mod tests {

    #[test]
    fn test_histogram() {
        use super::histogram_from_state;
        use crate::complex::Complex;

        fn complex_vector_from(data: &[(f64, f64)]) -> Vec<Complex> {
            data.into_iter().map(|x| Complex::new(x.0, x.1)).collect()
        }

        let data = [
            (0.9, -0.01),
            (2.0, 1.0),
            (-1.4e-10, -9.1e-10),
            (0.0, 0.0),
            (1.0E-30, 1.0E-40),
        ];
        let hist = histogram_from_state(&complex_vector_from(&data));
        println!("{:?}", hist);
    }

    // #[test]
    // fn write_and_read_state_file() {
    //     use super::{read_state_from_file, write_state_to_binary_file};
    //     use crate::complex::{normalize_complex_vector, Complex};
    //     use std::path::PathBuf;

    //     let len = 128;
    //     let mut expected: Vec<Complex> = Vec::with_capacity(len);
    //     for i in 0..len {
    //         let real = i as f64;
    //         let imag = -0.3 * real;
    //         expected.push(Complex::new(real, imag));
    //     }
    //     let path: PathBuf = "write_and_read_state_file.bin".into();
    //     normalize_complex_vector(&mut expected);

    //     write_state_to_binary_file(&path, &expected).unwrap();
    //     let result = read_state_from_file(&path).unwrap();
    //     assert_eq!(result.len(), expected.len());
    //     for i in 0..len {
    //         let a = result[i];
    //         let b = expected[i];
    //         assert!((a.real - b.real).abs() < 1E-15);
    //         assert!((a.imag - b.imag).abs() < 1E-15);
    //     }
    // }
}
