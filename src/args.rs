//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Command line argument parsing and simulation settings.

use crate::hamiltonian::ModelType;
use crate::lattice::{Location, Periodicity};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::f64::consts::PI;
use std::num::{ParseFloatError, ParseIntError};

fn map_parse_int_error(string: &str, err: ParseIntError) -> Box<dyn Error> {
    format!("{} in \"{}\"", err.to_string(), string).into()
}

fn map_parse_float_error(string: &str, err: ParseFloatError) -> Box<dyn Error> {
    format!("{} in \"{}\"", err.to_string(), string).into()
}

struct Args {
    options: HashMap<String, String>,
}

fn bad_boolean(s: String) -> Box<dyn Error> {
    format!("expected \"false\", \"true\", or an empty string, but got {s}").into()
}

fn parse_location(string: &str) -> Result<Location, Box<dyn Error>> {
    let comma_index = match string.find(',') {
        None => {
            return Err(format!(
                "expected location argument like e.g. '7,3', but got '{}' instead",
                string
            )
            .into());
        }
        Some(i) => i,
    };
    let x_str = string.split_at(comma_index).0;
    let y_str = &string.split_at(comma_index).1[1..];
    let x = x_str
        .parse()
        .map_err(|err| map_parse_int_error(x_str, err))?;
    let y = y_str
        .parse()
        .map_err(|err| map_parse_int_error(y_str, err))?;
    Ok(Location::new(x, y))
}

fn parse_location_list(string: &str) -> Result<Vec<Location>, Box<dyn Error>> {
    let len = string.split(';').count();
    let mut vec = Vec::with_capacity(len);
    for word in string.split(';') {
        vec.push(parse_location(word)?);
    }
    Ok(vec)
}

impl Args {
    fn expect_empty(&self) -> Result<(), Box<dyn Error>> {
        if !self.options.is_empty() {
            let key: Vec<&String> = self.options.keys().take(1).collect();
            return Err(format!("unparsed options {}", key.first().unwrap()).into());
        }
        Ok(())
    }

    fn take_str<'a>(&'a mut self, names: &[&str]) -> Option<String> {
        if names.len() == 0 {
            return None;
        }
        for name in names {
            if let Some(s) = self.options.remove(*name) {
                return Some(s);
            }
        }
        None
    }

    fn take_flag(&mut self, default: bool, names: &[&str]) -> Result<bool, Box<dyn Error>> {
        match self.take_str(names) {
            None => Ok(default),
            Some(s) => {
                if s == "" || s == "true" {
                    Ok(true)
                } else if s == "false" {
                    Ok(false)
                } else {
                    Err(bad_boolean(s))
                }
            }
        }
    }

    fn get_string(&mut self, default: &str, names: &[&str]) -> String {
        let string = self.take_str(names);
        match string {
            None => String::from(default),
            Some(s) => s,
        }
    }

    fn get_usize(&mut self, default: usize, names: &[&str]) -> Result<usize, Box<dyn Error>> {
        let string = self.take_str(names);
        match string {
            None => Ok(default),
            Some(s) => s.parse().map_err(|err| map_parse_int_error(&s, err)),
        }
    }

    fn get_f64(&mut self, default: f64, names: &[&str]) -> Result<f64, Box<dyn Error>> {
        let string = self.take_str(names);
        match string {
            None => Ok(default),
            Some(s) => s.parse().map_err(|err| map_parse_float_error(&s, err)),
        }
    }

    fn get_location(
        &mut self,
        default_x: i32,
        default_y: i32,
        names: &[&str],
    ) -> Result<Location, Box<dyn Error>> {
        let string = self.take_str(names);
        let string = match string {
            None => return Ok(Location::new(default_x, default_y)),
            Some(s) => s,
        };
        parse_location(&string)
    }

    fn get_location_list(&mut self, names: &[&str]) -> Result<Vec<Location>, Box<dyn Error>> {
        let string = self.take_str(names);
        let string = match string {
            None => return Ok(Vec::new()),
            Some(s) => s,
        };
        parse_location_list(&string)
    }
}

fn arg_is_valid_key(arg: &str) -> Result<bool, String> {
    if arg.starts_with('-') {
        let mut key = &arg[1..];
        if key.starts_with('-') {
            key = &key[1..];
        }
        if key.is_empty() {
            return Err(String::from("encountered empty option"));
        }
        return Ok(true);
    } else {
        return Ok(false);
    }
}

fn parse_args(args: Vec<String>) -> Result<Args, Box<dyn Error>> {
    let mut args = args.into_iter().skip(1).peekable();
    let mut options = HashMap::new();
    loop {
        let arg = match args.next() {
            None => {
                break;
            }
            Some(arg) => arg,
        };
        let is_key = arg_is_valid_key(&arg)?;
        if is_key {
            let key = arg;
            match key.find('=') {
                Some(index) => {
                    let (key, value) = key.split_at(index);
                    options.insert(key.to_string(), value[1..].to_string());
                    continue;
                }
                None => {}
            }
            match args.peek() {
                None => {
                    options.insert(key.to_string(), String::new());
                }
                Some(value) => {
                    let is_key = arg_is_valid_key(value)?;
                    if is_key {
                        options.insert(key, String::new());
                    } else {
                        options.insert(key, value.to_string());
                        args.next();
                    }
                }
            }
            continue;
        } else {
            return Err(format!("positional argument \"{arg}\" is not allowed").into());
        }
    }
    Ok(Args { options })
}

/// Settings for the initialization of the quantum system.
pub struct CaseSettings {
    /// Size of the lattice
    pub lattice_size: Location,

    /// Periodicity of the lattice
    pub lattice_periodicity: Periodicity,

    /// First site of the wall
    pub wall_start: Location,

    /// Size of the wall
    pub wall_size: Location,

    /// Extra sites of the wall
    pub extra_wall_sites: Vec<Location>,

    /// First site of the gaussian particle
    pub gaussian_start: Location,

    /// Size of the gaussian particle
    pub gaussian_size: Location,

    /// Center of the gaussian particle
    pub gaussian_center_x: f64,

    /// Momentum of the gaussian particle
    pub gaussian_momentum_x: f64,

    /// Parameter to control the spread of the gaussian particle
    pub gaussian_sigma: f64,

    /// Type of the underlying model
    pub model_type: ModelType,

    /// Paramter to control the hopping strength in x-direction.
    pub tx: f64,

    /// Parameter to control the interaction strenght in x-direction.
    pub vx: f64,

    /// Paramter to control the hopping strength in y-direction.
    pub ty: f64,

    /// Parameter to control the interaction strenght in y-direction.
    pub vy: f64,
}

/// Settings for saving states.
#[derive(Clone)]
pub enum SaveStates {
    /// Save all states.
    SaveAllStates,

    /// Save only the states of the specified measurements.
    ///
    /// If the set is empty, no state are saved at all.
    SaveSpecificMeasurements(HashSet<usize>),
}

impl SaveStates {
    /// Whether the state for the given measurement number should be saved.
    pub fn is_saved(&self, measurement: usize) -> bool {
        match self {
            SaveStates::SaveAllStates => true,
            // A HashSet might be better, but this will do for now.
            SaveStates::SaveSpecificMeasurements(measurements) => {
                measurements.contains(&measurement)
            }
        }
    }
}

/// Output format for images.
pub enum ImageFormat {
    /// Use the PNG format.
    PNG,
    /// Use the SVG format.
    SVG,
}

/// Parses the image output format from the argument.
fn parse_image_output_format_arg(
    arg: Option<String>,
) -> Result<Option<ImageFormat>, Box<dyn Error>> {
    match arg {
        None => Ok(Some(ImageFormat::PNG)),
        Some(arg) => {
            let arg = arg.to_ascii_lowercase();
            if arg == "png" {
                return Ok(Some(ImageFormat::PNG));
            } else if arg == "svg" {
                return Ok(Some(ImageFormat::SVG));
            } else if arg == "none" {
                return Ok(None);
            } else {
                return Err(format!("bad image output format: '{arg}'").into());
            }
        }
    }
}

/// Settings for the simulation.
pub struct SimulationSettings {
    /// Time for a single step
    pub time_step: f64,

    /// Time between measurements
    pub time_per_measurement: f64,

    /// Time at which the simulation stops.
    pub time_per_simulation: f64,

    /// Number of threads to use.
    pub num_threads: usize,

    /// Verbosity level
    pub verbosity: usize,

    /// The image output format to use or `None` to not write images at all.
    pub image_format: Option<ImageFormat>,

    /// Whether to save the measured states to CSV files.
    pub save_states: SaveStates,

    /// Whether existing files should simply be overwritten.
    pub force_overwrite: bool,

    /// Whether the automatic interaction offset for trace minimization should be disabled.
    pub disable_interaction_offset: bool,

    /// Start point and (inclusive) end point of the projection area.
    pub projection_area: Option<(Location, Location)>,
}

/// Arguments for the program.
pub struct Arguments {
    /// Settings for the initialization.
    pub case_settings: CaseSettings,

    /// Settings for the simulation.
    pub simulation_settings: SimulationSettings,

    /// Whether to just show the help text and exit.
    pub show_help: bool,

    /// Whether to just show the version number and exit.
    pub show_version: bool,

    /// Whether to only predict memory usage and exit.
    pub predict_memory_usage: bool,

    /// Whether to save the Hamilton matrix and basis to a file.
    pub save_hamiltonian: bool,

    /// Whether to save the basis to a file.
    pub save_basis: bool,

    /// Path to the initial state file to load the state from.
    pub initial_state_file: Option<String>,

    /// Number of highest occupied states to save.
    pub save_most_occupied: usize,
}

/// Calculates `(a + b) / 2`.
fn middle_of(a: i32, b: i32) -> f64 {
    let a = a as f64;
    let b = b as f64;
    return 0.5 * (a + b);
}

fn detect_default_number_of_threads() -> usize {
    match std::thread::available_parallelism() {
        Err(_) => 1,
        Ok(num_threads) => {
            let n: usize = num_threads.into();
            let estimated_num_cores = n / 2;
            if estimated_num_cores <= 0 {
                1
            } else if estimated_num_cores >= 16 {
                16
            } else {
                estimated_num_cores
            }
        }
    }
}

fn parse_save_states(arg: Option<String>) -> Result<SaveStates, Box<dyn Error>> {
    match arg {
        None => Ok(SaveStates::SaveSpecificMeasurements(HashSet::new())),
        Some(arg) => {
            if arg.is_empty() {
                return Ok(SaveStates::SaveAllStates);
            }
            let mut result = HashSet::new();
            for word in arg.split(',') {
                match word.parse() {
                    Err(_) => {
                        return Err(
                            format!("bad number in argument '--save-states': '{word}'").into()
                        );
                    }
                    Ok(measurement) => {
                        result.insert(measurement);
                    }
                }
            }
            Ok(SaveStates::SaveSpecificMeasurements(result))
        }
    }
}

fn parse_projection_area(
    arg: Option<String>,
) -> Result<Option<(Location, Location)>, Box<dyn Error>> {
    match arg {
        None => Ok(None),
        Some(arg) => {
            let locations = parse_location_list(&arg)?;
            if locations.len() == 2 {
                Ok(Some((locations[0], locations[1])))
            } else {
                Err("projection area argument must be a list of two locations".into())
            }
        }
    }
}

const HELP_TEXT: &'static str = "tevo - Quantum Simulation Program
Author: Christian Maier (2026)
License: MIT

Usage: tevo [OPTIONS]

Available options:

    -h,   --help                Prints this help
          --version             Shows version information
    -v=N, --verbosity=N         Verbosity level of the console output (0=silent, 1=default, 2=all)
    -j=N, --threads=N           Number of threads to use, use estimate if argument is omitted
    -f,   --force-overwrite     Overwrite files in existing directories

    --time-step=<float>            Time step for the simulation (default: 5E-4)
    --time-per-measurement=<float> Time between measurements. Must be a multiple of the time step.
    -e --simulation-time=<float>   Time at which the simulation should end. Must be a multiple of
                                   the time per measurement.
    
    --tx=<float>                    Horizontal hopping paramter tx
    --vx=<float>                    Horizontal interaction parameter Vx
    --ty=<float>                    Vertical hopping parameter ty. Must be greater than zero
                                    if the lattice is not periodic in y-direction.
                                    This will be fixed in later versions.
    --vy=<float>                    Vertical interaction parameter Vy
    --lattice-size=<int>,<int>      Size (width, height) of the lattice
    --periodic-in-x                 Make the lattice periodic in horizontal direction
    --periodic-in-y                 Make the lattice periodic in vertical direction
    --wall-start=<int>,<int>        Start site (x, y) of the wall
    --wall-size=<int>,<int>         Size (width, height) of the wall
    --gaussian-start=<int>,<int>    Start site (x, y) of the gaussian
    --gaussian-size=<int>,<int>     Size (width, height) of the gaussian.
                                    Use --gaussian-size=0,0 to remove the gaussian.
    --gaussian-center=<float>       Center of the gaussian distribution
    --gaussian-momentum=<float>     Momentum of the gaussian (default=-pi/2)
    --gaussian-sigma=<float>        Sigma of the gaussian (default=3)

    --model=xxz     Use the Heisenberg spin-1/2 XXZ model
    --model=tv      Use the t/V model (spinless fermions)
    --model=tv-sym  Use the t/V model (spinless fermions) with particle-hole symmetry

    --save-states[=MEASUREMENT1,MEASUREMENT2,...]
                
                    Saves the states of the specified measurements as binary files.
                    The argument is a comma separated list of numbers and optional.
                    The list is optional. If it is empty, all states are saved.
                    Example: --save-states=23,45

    --save-most-occupied-states=<int>    Saves the N most occupied states (default=0).

    --disable-interaction-offset  Disables automatic offset for Hamiltonian trace minimization.
    --image-format=png      Write output images in PNG format (default)
    --image-format=svg      Write output images in SVG format (default)
    --image-format=none     Disable image output
    --save-hamiltonian      Saves the Hamiltonian to a CSV file without running the simulation.
    --save-basis            Saves the basis to a binary file without running the simulation.
    --predict-memory-usage  Predicts the memory usage without running the simulation.

    --extra-wall-sites=<int>,<int>;<int>,<int>;...

            Extra sites of the wall.
    --initial-state-file=<state-N.bin>  Loads the initial state from a binary state file.
    --projection-area=<int>,<int>;<int>,<int>
    
            x0,y0;x1;y1 specifies the start point and (inclusive) end point of the projection area.
";

impl Arguments {
    /// Parses program arguments from a list of strings.
    pub fn new(argv: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let mut args = parse_args(argv)?;
        let lattice_size = args.get_location(7, 2, &["--lattice-size"])?;
        let periodic_in_x = args.take_flag(false, &["--periodic-in-x"])?;
        let periodic_in_y = args.take_flag(false, &["--periodic-in-y"])?;
        let nx = lattice_size.x;
        let hx = nx / 2;
        let ny = lattice_size.y;
        let tx = args.get_f64(-0.5, &["--tx"])?;
        let ty = args.get_f64(-0.5, &["--ty"])?;
        let vx = args.get_f64(8.0, &["--vx"])?;
        let vy = args.get_f64(1.0, &["--vy"])?;
        let wall_start = args.get_location(hx, 0, &["--wall-start"])?;
        let wall_size = args.get_location(1, ny, &["--wall-size"])?;
        let extra_wall_sites = args.get_location_list(&["--extra-wall-sites"])?;
        let gaussian_start = args.get_location(0, 0, &["--gaussian-start"])?;
        let def_size = (wall_start.x - gaussian_start.x).min(5);
        let gaussian_size = args.get_location(def_size, ny, &["--gaussian-size"])?;
        let default_center = middle_of(gaussian_start.x, gaussian_start.x + gaussian_size.x);
        let gaussian_center_x = args.get_f64(default_center, &["--gaussian-center"])?;
        let gaussian_momentum_x = args.get_f64(-0.5 * PI, &["--gaussian-momentum"])?;
        let gaussian_sigma = args.get_f64(3.0, &["--gaussian-sigma"])?;
        let model_type_str = args.get_string("tv", &["--model"]);
        let model_type = ModelType::from_identifier(&model_type_str)
            .ok_or(format!("unknown model {model_type_str}"))?;

        let time_step = args.get_f64(5.0E-4, &["-s", "--time-step"])?;
        let time_to_end = args.get_f64(
            10.0,
            &["-e", "--time-of-last-measurement", "--simulation-time"],
        )?;
        let time_per_measurement = args.get_f64(time_to_end, &["--time-per-measurement"])?;
        let mut num_threads = args.get_usize(0, &["-j", "--threads"])?;
        if num_threads == 0 {
            num_threads = detect_default_number_of_threads();
        }
        let verbosity = args.get_usize(1, &["-v", "--verbosity"])?;
        let image_format = parse_image_output_format_arg(args.take_str(&["--image-format"]))?;
        let save_hamiltonian = args.take_flag(false, &["--save-hamiltonian"])?;
        let save_basis = args.take_flag(false, &["--save-basis"])?;
        let save_states = parse_save_states(args.take_str(&["--save-states"]))?;
        let save_most_occupied = args.get_usize(0, &["--save-most-occupied-states"])?;
        let disable_interaction_offset =
            args.take_flag(false, &["--disable-interaction-offset"])?;
        let force_overwrite = args.take_flag(false, &["-f", "--force-overwrite"])?;
        let initial_state_file = args.take_str(&["--initial-state-file"]);

        let show_help = args.take_flag(false, &["-h", "--help"])?;
        let show_version = args.take_flag(false, &["--version"])?;
        let predict_memory_usage = args.take_flag(false, &["--predict-memory-usage"])?;

        let projection_area = parse_projection_area(args.take_str(&["--projection-area"]))?;

        args.expect_empty()?;

        Ok(Self {
            case_settings: CaseSettings {
                lattice_size,
                lattice_periodicity: Periodicity {
                    periodic_in_x,
                    periodic_in_y,
                },
                tx,
                ty,
                vx,
                vy,
                wall_start,
                wall_size,
                extra_wall_sites,
                gaussian_start,
                gaussian_size,
                gaussian_center_x,
                gaussian_momentum_x,
                gaussian_sigma,
                model_type,
            },
            simulation_settings: SimulationSettings {
                time_step,
                time_per_simulation: time_to_end,
                time_per_measurement,
                num_threads,
                verbosity,
                image_format,
                save_states,
                force_overwrite,
                disable_interaction_offset,
                projection_area,
            },
            show_help,
            show_version,
            predict_memory_usage,
            save_hamiltonian,
            save_basis,
            initial_state_file,
            save_most_occupied,
        })
    }

    /// Prints the help text.
    pub fn print_help_text() {
        println!("{HELP_TEXT}");
    }
}
