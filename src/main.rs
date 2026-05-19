//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Binary entry point for running simulations from the command line.

use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tevo::args::Arguments;
use tevo::cases::QuantumBowlingCase;
use tevo::hamiltonian::Hamiltonian;

use tevo::complex::Complex;
use tevo::measure::Measurements;
use tevo::projection::perform_projection;
use tevo::sparse::parallel_sparse_time_evolution;
use tevo::utils::{
    InfoData, PROGRAM_VERSION, get_hostname, histogram_from_state, read_state_from_file,
    write_basis_to_binary_file, write_hamiltonian_to_csv_file, write_state_to_binary_file,
};

/// Subprogram for predicting the memory usage of this case.
fn predict_memory_usage(hamiltonian: &Hamiltonian) {
    let dim = hamiltonian.dimension();
    let mut num_duplets = 0;
    let mut duplets = Vec::new();
    for i in 0..dim {
        hamiltonian.load_sparse_column(i, &mut duplets);
        num_duplets += duplets.len();
    }
    let indptr_byte_size = (dim + 1) * 8;
    let duplets_byte_size = 12 * num_duplets;
    let matrix_mib_size = (indptr_byte_size + duplets_byte_size) as f64 / 1024.0 / 1024.0;

    let num_vectors = 4;
    let vector_byte_size = num_vectors * 16 * dim;
    let vector_mib_size = vector_byte_size as f64 / 1024.0 / 1024.0;

    let total_mib_size = vector_mib_size + matrix_mib_size;
    let nnz = num_duplets;
    print!(
        "Hamiltonian Matrix:
    Dimension: {dim},
    Non-zeros: {nnz},
    Size in bytes: {matrix_mib_size:.3} MiB,

State vectors:
    Size in bytes: {vector_mib_size:.3} MiB,

Total:
    Size in bytes: {total_mib_size:.3} MiB,
"
    );
    if dim > (u32::MAX as usize) {
        println!("WARNING: dimension > 32-bit integer maximum!");
    }
}

fn safe_divide(a: f64, b: f64) -> Result<usize, Box<dyn Error>> {
    if b == 0.0 {
        return Err("attempted division by zero".into());
    }
    if a < 0.0 || b < 0.0 {
        return Err("safe_divide: encountered negative numbers".into());
    }
    let d_f64 = a / b;
    let d_usize = d_f64.round() as usize;
    if ((d_usize as f64) - d_f64).abs() > 1.0E-10 {
        return Err(format!("{a} is not divisible by {b}").into());
    }
    Ok(d_usize)
}

fn sparse_time_evolution(
    args: &mut Arguments,
    case: QuantumBowlingCase,
) -> Result<(), Box<dyn Error>> {
    let mut hamiltonian = case.hamiltonian;
    if args.predict_memory_usage {
        predict_memory_usage(&hamiltonian);
        println!("\nExiting without performing a simulation");
        return Ok(());
    }
    let output_dir_name = &case.name;
    println!("\nOutput directory: {output_dir_name}");
    let output_dir = PathBuf::from(output_dir_name);
    let exists = output_dir.exists();
    let force_overwrite = args.simulation_settings.force_overwrite;
    if exists && !force_overwrite && !(args.save_basis || args.save_hamiltonian) {
        return Err(error_output_dir_already_exists(output_dir_name));
    }
    if !exists {
        std::fs::create_dir(&output_dir)?;
    }
    if args.save_hamiltonian {
        write_hamiltonian_to_csv_file(&output_dir.join("hamilton-matrix.csv"), &hamiltonian)?;
        println!("\nBasis has been saved");
    }
    if args.save_basis {
        let basis_file_path = output_dir.join("basis.bin");
        write_basis_to_binary_file(&basis_file_path, &hamiltonian.basis).map_err(Box::new)?;
        println!("\nBasis has been saved");
    }
    if args.save_basis || args.save_hamiltonian {
        println!("Exiting without performing a simulation");
        return Ok(());
    }

    let num_threads = args.simulation_settings.num_threads;
    println!("Using {num_threads} threads.");
    let state_0 = case.initial_state;
    let dim = hamiltonian.dimension();

    let lattice_size = hamiltonian.basis.lattice.lattice_size;
    println!("Dimension: {}", dim);
    let mut state = match &args.initial_state_file {
        None => state_0.to_dense(),
        Some(fname) => {
            println!("Loading initial state file: {}", fname);
            let path = PathBuf::from(fname);
            let state = read_state_from_file(&path)?;
            if state.len() != dim {
                return Err("the initial state file has the wrong size".into());
            }
            state
        }
    };

    let mut projection_norm = 1.0;
    if let Some((start, end)) = args.simulation_settings.projection_area {
        println!("Performing projection on state 0");
        projection_norm = perform_projection(&hamiltonian.basis, &mut state, start, end);
        println!("Norm after projection was {0:.4E}", projection_norm);
    }

    let save_states = args.simulation_settings.save_states.clone();
    if save_states.is_saved(0) {
        println!("Saving state 0...");
        let path = &output_dir.join("state-0.bin");
        write_state_to_binary_file(&path, &state).map_err(Box::new)?;
    }

    println!("Building matrix...");
    let begin_hamiltonian = Instant::now();
    if !args.simulation_settings.disable_interaction_offset {
        hamiltonian.minimize_trace();
    }
    let interaction_offset = hamiltonian.get_interaction_offset();
    let hamilton_matrix = hamiltonian.to_sparse_matrix();
    let end_hamiltonian = Instant::now();
    let setup_time = end_hamiltonian
        .duration_since(begin_hamiltonian)
        .as_secs_f64();
    println!("Building the matrix took {setup_time:.3}s");
    println!("Hamiltonian:");
    let nnz = hamilton_matrix.count_non_zeros();
    println!("    Non-zeros: {nnz}");
    let hamilton_matrix_mib = hamilton_matrix.estimate_byte_size() as f64 / 1024.0 / 1024.0;
    println!("    Memory:    {hamilton_matrix_mib:.3} MiB");

    let time_step = args.simulation_settings.time_step;
    let time_per_measurement = args.simulation_settings.time_per_measurement;
    let time_per_simulation = args.simulation_settings.time_per_simulation;
    let steps_per_measurement = safe_divide(time_per_measurement, time_step.abs())?;
    let num_measurements = 1 + safe_divide(time_per_simulation, time_per_measurement)?;
    let total_steps = steps_per_measurement * (num_measurements - 1);
    let save_most_occupied = args.save_most_occupied;

    let begin_simulation = Instant::now();

    let measurements = Arc::new(Mutex::new(Measurements::new(
        hamiltonian,
        num_measurements,
        save_most_occupied,
    )));
    let monitor_measurements = measurements.clone();
    let total_measurement_time = Arc::new(Mutex::new(0.0));
    let tmt_clone = total_measurement_time.clone();
    let output_dir_2 = output_dir.clone();
    let monitor = Box::new(move |measurement: usize, time: f64, state: &[Complex]| {
        let elapsed = begin_simulation.elapsed().as_secs_f64();
        let total = num_measurements - 1;
        println!("[{elapsed:6.3}]: Finished measurement {measurement}/{total}");
        let begin_measurement = Instant::now();
        monitor_measurements
            .lock()
            .unwrap()
            .measure(measurement, time, state);
        if save_states.is_saved(measurement) {
            let path = &output_dir_2.join(format!("state-{measurement}.bin"));
            write_state_to_binary_file(&path, &state)
                .map_err(Box::new)
                .unwrap();
        }
        let measurement_time = begin_measurement.elapsed().as_secs_f64();
        *tmt_clone.lock().unwrap() += measurement_time;
    });
    let (state, time_evo_report) = parallel_sparse_time_evolution(
        &hamilton_matrix,
        state,
        time_step,
        num_measurements,
        steps_per_measurement,
        num_threads,
        monitor,
    );
    let hamiltonian_trace = hamilton_matrix.trace();
    drop(hamilton_matrix);
    let total_measurement_time = *total_measurement_time.lock().unwrap();
    let simulation_time = begin_simulation.elapsed().as_secs_f64() - total_measurement_time;
    let total_time = begin_hamiltonian.elapsed().as_secs_f64();
    let sec_per_step = simulation_time / (total_steps as f64);
    let ns_per_step_per_mdim = 1.0e9 * simulation_time / (total_steps * dim) as f64;
    let image_format = &args.simulation_settings.image_format;
    let max_step_norm = time_evo_report.norm.get_max_step_norm_sq().sqrt();
    let total_norm = time_evo_report.norm.get_total_norm_sq().map(|x| x.sqrt());
    measurements
        .lock()
        .unwrap()
        .write_files(&output_dir, image_format)
        .map_err(Box::new)?;
    let info_data = InfoData {
        basis_dimension: dim,
        hamilton_matrix_nnz: nnz,
        hamilton_matrix_mib,
        hostname: get_hostname(),
        lattice_width: lattice_size.x,
        lattice_height: lattice_size.y,
        num_measurements,
        setup_time,
        simulation_time,
        steps_per_measurement,
        total_time,
        total_steps,
        total_norm,
        interaction_offset,
        hamiltonian_trace,
        max_step_norm,
        projection_norm,
    };
    info_data
        .write_csv_file(&output_dir.join("info.csv"))
        .map_err(Box::new)?;
    let histogram = histogram_from_state(&state);
    for entry in histogram {
        println!("# of coefficients >= {:5.0e}: {}", entry.1, entry.0)
    }
    let max_step_norm_minus_one = max_step_norm - 1.0;
    let max_step_norm_minus_one_str = if max_step_norm_minus_one < 0.0 {
        format!("(=1-{0:.e})", -max_step_norm_minus_one)
    } else {
        format!("(=1+{0:.e})", max_step_norm_minus_one)
    };
    if let Some(total_norm) = total_norm {
        let total_norm_deviation = total_norm - 1.0;
        println!("Deviation of the final vector norm ||v|| - 1: {total_norm_deviation:.e}");
    } else {
        println!("Warning: final vector norm is not a finite value.");
    }
    println!("Finished");
    println!(
        "Maximum step norm: {0:.e}  {1}",
        max_step_norm, max_step_norm_minus_one_str
    );
    println!("Measurements took {total_measurement_time:.3}s");
    println!("Simulation took {simulation_time:.3}s to simulate {total_steps} timesteps");
    println!("               ({sec_per_step:.3}s per timestep)");
    println!("               ({ns_per_step_per_mdim:.3}ns per timestep per dim)");
    Ok(())
}

fn error_output_dir_already_exists(output_dir_name: &str) -> Box<dyn Error> {
    return format!(
        "the output directory '{output_dir_name}' already exists (try \"--force-overwrite\")"
    )
    .into();
}

fn run_main(args: &mut Arguments) -> Result<(), Box<dyn Error>> {
    let case = QuantumBowlingCase::new(args)?;
    sparse_time_evolution(args, case)
}

const HELP_TEXT_HEADER: &'static str = "tevo - Quantum Simulation Program
Author: Christian Maier (2026)
License: MIT

";

#[cfg(target_feature = "avx2")]
fn has_avx2_support() -> bool {
    true
}

#[cfg(not(target_feature = "avx2"))]
fn has_avx2_support() -> bool {
    false
}

/// Main entry point for the program
fn main() -> ExitCode {
    let argv: Vec<String> = env::args().collect();
    let mut args = match Arguments::new(argv) {
        Ok(args) => args,
        Err(err) => {
            println!("Error: bad arguments: {err}");
            return ExitCode::FAILURE;
        }
    };
    let version = PROGRAM_VERSION;
    if args.show_version {
        println!("{}", version);
        println!("AVX2 enabled: {}", has_avx2_support());
        return ExitCode::SUCCESS;
    }
    if args.show_help {
        println!("{HELP_TEXT_HEADER}");
        Arguments::print_help_text();
        return ExitCode::SUCCESS;
    }
    let result = run_main(&mut args);
    if let Err(err) = result {
        println!("Error: {err}");
        return ExitCode::FAILURE;
    };
    ExitCode::SUCCESS
}
