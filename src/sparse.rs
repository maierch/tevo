//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Sparse matrices and parallel sparse time evolution.

use std::{
    error::Error,
    sync::{Arc, Barrier, Mutex, RwLock},
    thread,
};

use crate::complex::Complex;

/// A real symmetric sparse matrix.
#[derive(Debug)]
pub struct SparseMatrix {
    dim: usize,
    data: Vec<f64>,
    indices: Vec<u32>,
    indptr: Vec<usize>,
}

fn normalize_vec_f64(v: &mut [f64]) {
    let mut acc: f64 = 0.0;
    for x in v.iter_mut() {
        let x = *x;
        acc += x * x;
    }
    if acc == 0.0 {
        panic!("normalize_vec_f64: attempted to normalize a null vector");
    }
    let s = 1.0 / acc.sqrt();
    for x in v.iter_mut() {
        *x *= s;
    }
}

fn compare_vectors(u: &[f64], v: &mut [f64]) -> f64 {
    if u[0].signum() != v[0].signum() {
        for x in v.iter_mut() {
            *x *= -1.0;
        }
    }
    let mut acc = 0.0;
    for (x, y) in u.iter().zip(v.iter()) {
        let d = *x - *y;
        acc += d * d;
    }
    return acc;
}

impl SparseMatrix {
    /// Creates a new sparse matrix with the given data.
    pub fn with_capacity(dim: usize, nnz: usize) -> Self {
        let data = Vec::with_capacity(nnz);
        let indices = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(dim + 1);
        indptr.push(0);
        Self {
            dim,
            data,
            indices,
            indptr,
        }
    }

    /// Appends one row to the sparse matrix.
    pub fn push_row(&mut self, columns: &[u32], values: &[f64]) {
        let len = columns.len();
        if values.len() != len {
            panic!("SparseMatrix::insert_row: columns.len() != values.len()");
        }
        let row = self.indptr.len() - 1;
        if row >= self.dim {
            panic!("SparseMatrix::insert_row: attempted insert beyond matrix dimension");
        }
        for value in values {
            self.data.push(*value);
        }
        for col in columns {
            self.indices.push(*col);
        }
        self.indptr.push(self.indptr.last().unwrap() + len);
    }

    /// Estimates the size of the sparse matrix in bytes.
    pub fn estimate_byte_size(&self) -> usize {
        8 + 3 * 3 * 8 + 8 * self.data.len() + 4 * self.indices.len() + 8 * self.indptr.len()
    }

    /// Number of non-zero elements.
    pub fn count_non_zeros(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn accumulate(
        &self,
        v: &[Complex],
        start_duplet_index: usize,
        end_duplet_index: usize,
    ) -> Complex {
        let mut acc = Complex::zero();
        let range = start_duplet_index..end_duplet_index;
        for (i, x) in self.indices[range.clone()]
            .iter()
            .zip(self.data[range.clone()].iter())
        {
            let i = *i;
            acc += v[i as usize] * (*x);
        }
        acc
    }

    /// Computes `out = self * v - s * v`.
    pub fn multiply_with_real_vec_and_shift(&self, out: &mut [f64], v: &[f64], s: f64) {
        let indptr = &self.indptr;
        for (row, (out_row, v_row)) in out.iter_mut().zip(v.iter()).enumerate() {
            let mut acc = 0.0;
            let range = indptr[row]..indptr[row + 1];
            for (i, x) in self.indices[range.clone()]
                .iter()
                .zip(self.data[range.clone()].iter())
            {
                let i = *i;
                acc += &v[i as usize] * x;
            }
            *out_row = acc - s * (*v_row);
        }
    }

    /// Finds the groundstate via the power method.
    pub fn estimate_groundstate(&self) -> Result<Vec<f64>, Box<dyn Error>> {
        // Try to find the ground state with the power method.
        let first_value = self
            .data
            .first()
            .ok_or("estimate_groundstate: Hamilton matrix is zero")?;
        let dim = self.dim;
        if dim == 0 {
            return Err("estimate_groundstate: dimension is zero".into());
        }
        // TODO: This is not a very good way to make a pseudo-random state.
        // Let's hope this works!
        // Make sure to verify the results!

        let max_value = self.data.iter().fold(*first_value, |acc, v| acc.max(*v));
        let mut state = vec![0.0; dim];
        for (i, x) in state.iter_mut().enumerate() {
            let mut n = (i as u32).wrapping_add(0xfe0a713b);
            n = n.wrapping_add(max_value.to_bits() as u32);
            n = ((n >> 16) ^ n).wrapping_mul(0x29fc3aeb);
            let rand = (n as f64) / (u32::MAX as f64);
            *x = 2.0 * rand - 1.0;
        }
        let energy_shift = ((dim as f64) * max_value).abs();
        let max_iteration = 100000;
        normalize_vec_f64(&mut state);
        let mut temp = state.clone();
        let mut err_sq = 0.0;
        for _ in 0..max_iteration {
            self.multiply_with_real_vec_and_shift(&mut temp, &state, energy_shift);
            normalize_vec_f64(&mut temp);
            err_sq = compare_vectors(&state, &mut temp);
            if err_sq < 1.0E-40 {
                return Ok(temp);
            }
            state = temp.clone();
        }
        if err_sq < 1.0E-20 {
            Ok(state)
        } else {
            Err(format!("estimate_groundstate: too many iterations, remainder: {err_sq}").into())
        }
    }

    /// Computes the sum of diagonal matrix entries.
    pub fn trace(&self) -> f64 {
        let mut sum = 0.0;
        let indices = &self.indices;
        let data = &self.data;
        for (row, (ind0, ind1)) in self
            .indptr
            .iter()
            .zip(self.indptr.iter().skip(1))
            .enumerate()
        {
            for ind in *ind0..*ind1 {
                let column = indices[ind] as usize;
                if row == column {
                    sum += data[ind];
                }
            }
        }
        sum
    }

    // /// Adds a value to all diagonal elements.
    // pub fn add_to_diagonal(&mut self, x: f64) {
    //     let indices = &self.indices;
    //     let data = &mut self.data;
    //     for (row, (ind0, ind1)) in self
    //         .indptr
    //         .iter()
    //         .zip(self.indptr.iter().skip(1))
    //         .enumerate()
    //     {
    //         for ind in *ind0..*ind1 {
    //             let column = indices[ind] as usize;
    //             if row == column {
    //                 data[ind] += x;
    //             }
    //         }
    //     }
    // }

    // /// Subtracts the mean value of all diagonal elements from the diagonal elements.
    // ///
    // /// Panics if the matrix has dimension zero.
    // pub fn subtract_mean_diagonal_value(&mut self) {
    //     let dim = self.dim;
    //     if dim == 0 {
    //         panic!("invoked subtract_mean_diagonal_value on matrix with zero dimension");
    //     }
    //     let f_dim: f64 = dim as f64;
    //     let mean = self.trace() / f_dim;
    //     self.add_to_diagonal(-mean);
    // }
}

/// Tracks normalization during time evolution.
#[derive(Clone)]
pub struct NormStruct {
    step_norm_sq: f64,
    max_step_norm_sq: f64,
    total_norm_sq: f64,
}

impl NormStruct {
    fn new() -> Self {
        Self {
            step_norm_sq: 0.0,
            max_step_norm_sq: 0.0,
            total_norm_sq: 1.0,
        }
    }

    fn add_section_norm_sq(&mut self, section_norm_sq: f64) {
        self.step_norm_sq += section_norm_sq;
        self.max_step_norm_sq = self.max_step_norm_sq.max(section_norm_sq);
    }

    fn conclude_step(&mut self) {
        if self.total_norm_sq.is_finite() && self.step_norm_sq.is_finite() {
            self.total_norm_sq *= self.step_norm_sq;
        }
        self.step_norm_sq = 0.0;
    }

    fn get_norm_factor(&self) -> Option<f64> {
        // The norm factor is 1 / sqrt(step_norm_sq)
        let step_norm_sq = self.step_norm_sq;
        if step_norm_sq == 0.0 {
            return None;
        }
        if !step_norm_sq.is_finite() {
            return None;
        }
        Some(1.0 / self.step_norm_sq.sqrt())
    }

    /// Gets the largest squared norm observed during a single step.
    pub fn get_max_step_norm_sq(&self) -> f64 {
        self.max_step_norm_sq
    }

    /// Gets the accumulated squared norm if it stayed finite.
    pub fn get_total_norm_sq(&self) -> Option<f64> {
        if self.total_norm_sq.is_finite() {
            Some(self.total_norm_sq)
        } else {
            None
        }
    }
}

fn worker(
    is_leader: bool,
    start_row: usize,
    num_rows: usize,
    hamiltonian: &SparseMatrix,
    state: &RwLock<Vec<Complex>>,
    temp: &RwLock<Vec<Complex>>,
    time_step: f64,
    num_measurements: usize,
    steps_per_measurement: usize,
    barrier: &Barrier,
    norm_struct_mutex: Arc<Mutex<NormStruct>>,
    monitor: Option<Box<dyn FnMut(usize, f64, &[Complex]) + Send>>,
) {
    let mut monitor = monitor;
    let end_row = start_row + num_rows;
    // `buffer` is the thread local state data buffer.
    let mut buffer = vec![Complex::zero(); num_rows];
    let indptr = &hamiltonian.indptr;
    let time_per_measurement = time_step * (steps_per_measurement as f64);
    for measurement in 1..num_measurements {
        let time = (measurement as f64) * time_per_measurement;
        for _ in 0..steps_per_measurement {
            {
                // Computes `buffer = state - i*H*0.5*delta_t*state`
                let state = state.read().unwrap();
                for row in start_row..(start_row + num_rows) {
                    let start_duplet_index = indptr[row];
                    let end_duplet_index = indptr[row + 1];
                    let acc = hamiltonian.accumulate(&state, start_duplet_index, end_duplet_index);
                    buffer[row - start_row] = state[row] + acc.mul_minus_i_times(0.5 * time_step);
                }
            }
            // Computes `temp = buffer` on the relevent slice.
            // With some luck, this is fast so the write lock isn't taken too long.
            {
                temp.write().unwrap()[start_row..end_row].copy_from_slice(&buffer);
            }
            // Wait untill everyone wrote to `temp`.
            barrier.wait();
            let section_norm_sq = {
                // Computes `buffer = state - i*H*delta_t*temp`
                // Note: multiple read locks can be taken by different threads.
                let state = state.read().unwrap();
                let temp = temp.read().unwrap();
                // {
                //     let state = state.read().unwrap();
                //     buffer.copy_from_slice(&state.read().unwrap()[start_row..end_row]);
                // }
                let mut section_norm_sq = 0.0;
                for row in start_row..(start_row + num_rows) {
                    let start_duplet_index = indptr[row];
                    let end_duplet_index = indptr[row + 1];
                    let acc = hamiltonian.accumulate(&temp, start_duplet_index, end_duplet_index);
                    let state_coeff = state[row] + acc.mul_minus_i_times(time_step);
                    buffer[row - start_row] = state_coeff;
                    section_norm_sq += state_coeff.abs_squared();
                    // buffer[row - start_row] += acc;
                }
                section_norm_sq
            };
            // Make all threads report their norm before we continue.
            norm_struct_mutex
                .lock()
                .unwrap()
                .add_section_norm_sq(section_norm_sq);
            barrier.wait();
            // Now each thread can write its buffer to the final state.
            let norm_factor = norm_struct_mutex.lock().unwrap().get_norm_factor().unwrap();
            {
                // Computes `state = buffer`
                let local_state = &mut state.write().unwrap()[start_row..end_row];
                for (dest, src) in local_state.iter_mut().zip(buffer.iter()) {
                    *dest = *src * norm_factor;
                }
            }
            // Wait until all threads have written to the final state vector.
            barrier.wait();
            // One thread should reset the steps square norm to zero.
            // Wait for the leader to finish resetting the norm mutex.
            if is_leader {
                norm_struct_mutex.lock().unwrap().conclude_step();
            }
            barrier.wait();
        } // End of the current step.
        if is_leader {
            let state = &mut state.write().unwrap();
            // let norm = normalize_complex_vector(state);
            if let Some(ref mut monitor) = monitor {
                monitor(measurement, time, state);
            }
        }
        // Wait for the leader to finish.
        barrier.wait();
    } // End of the current measurement.
}

/// Summary data from a time evolution run.
pub struct TimeEvolutionReport {
    /// Norm diagnostics collected during time evolution.
    pub norm: NormStruct,
}

/// Evolves a state with a sparse Hamiltonian using multiple threads.
pub fn parallel_sparse_time_evolution(
    hamiltonian: &SparseMatrix,
    state: Vec<Complex>,
    time_step: f64,
    num_measurements: usize,
    steps_per_measurement: usize,
    num_threads: usize,
    monitor: Box<dyn FnMut(usize, f64, &[Complex]) + Send>,
) -> (Vec<Complex>, TimeEvolutionReport) {
    let mut monitor = monitor;
    monitor(0, 0.0, &state);
    let dim = state.len();
    let temp = RwLock::new(state.clone());
    let state = RwLock::new(state);
    let mut offsets = Vec::new();
    let chunk_size = dim.div_ceil(num_threads);
    for i in 0..num_threads {
        let start_row = i * chunk_size;
        let end_row = ((i + 1) * chunk_size).min(dim);
        let num_rows = end_row - start_row;
        offsets.push((start_row, num_rows));
    }
    let norm_sq_mutex = Arc::new(Mutex::new(NormStruct::new()));
    let barrier = Barrier::new(num_threads);
    thread::scope(|s| {
        let mut monitor = Some(monitor);
        for (start_row, num_rows) in offsets {
            let norm_sq_mutex = norm_sq_mutex.clone();
            let monitor = monitor.take();
            let barrier = &barrier;
            let state = &state;
            let temp = &temp;
            let is_leader = start_row == 0;
            s.spawn(move || {
                worker(
                    is_leader,
                    start_row,
                    num_rows,
                    hamiltonian,
                    state,
                    temp,
                    time_step,
                    num_measurements,
                    steps_per_measurement,
                    &barrier,
                    norm_sq_mutex,
                    monitor,
                );
            });
        }
    });
    let s = state.into_inner().unwrap();
    let norm = norm_sq_mutex.lock().unwrap().clone();
    (s, TimeEvolutionReport { norm })
}

mod tests {
    #[cfg(test)]
    use super::SparseMatrix;

    #[cfg(test)]
    fn assert_almost_equal(u: Vec<f64>, v: Vec<f64>, eps: f64) {
        let mut v = v;
        assert!(u.len() > 0);
        assert_eq!(u.len(), v.len());
        if u[0].signum() != v[0].signum() {
            for x in v.iter_mut() {
                *x *= -1.0;
            }
        }
        for (x, y) in u.iter().zip(v.iter()) {
            let dist = (*x - *y).abs();
            assert!(dist <= eps);
        }
    }

    // #[cfg(test)]
    // fn assert_almost_equal_sparse_matrices(a: &SparseMatrix, b: &SparseMatrix, eps: f64) {
    //     assert_eq!(a.dim, b.dim);
    //     assert_eq!(a.count_non_zeros(), b.count_non_zeros());
    //     let ma = a.indices.iter().zip(a.data.iter());
    //     let mb = b.indices.iter().zip(b.data.iter());
    //     for ((i, x), (j, y)) in ma.zip(mb) {
    //         assert_eq!(*i, *j);
    //         assert!((x - y).abs() < eps);
    //     }
    // }

    #[cfg(test)]
    pub fn from_rows(dim: usize, rows: &[&[(u32, f64)]]) -> SparseMatrix {
        let mut nnz = 0;
        for row in rows {
            nnz += row.len();
        }
        let mut matrix = SparseMatrix::with_capacity(dim, nnz);
        for row in rows {
            let columns: Vec<u32> = row.iter().map(|x| x.0).collect();
            let values: Vec<f64> = row.iter().map(|x| x.1).collect();
            matrix.push_row(&columns, &values);
        }
        matrix
    }

    #[test]
    fn test_groundstate_0() {
        let hamilton_matrix = from_rows(2, &[&[(0, 1.5), (1, -2.0)], &[(0, -2.0), (1, 3.0)]]);
        // In Python: np.linalg.eigh(np.array([[1.5, -2],[-2, 3]]))
        let groundstate = hamilton_matrix.estimate_groundstate().unwrap();

        let eps = 1.0E-7;
        assert_almost_equal(groundstate, vec![-0.82192562, -0.56959484], eps);
    }

    #[test]
    fn test_groundstate() {
        let rsqrt2 = (1.0_f64 / 2.0).sqrt();

        let t_list = [-0.3, -0.3, -0.05, -1.0, 1.1, 2.3, 8.0];
        let v_list = [0.5, -0.5, 2.0, 0.0, 2.0, -0.04, 8.0];

        for (t, v) in t_list.iter().zip(v_list.iter()) {
            let t = *t;
            let v = *v;
            let a = v / 4.0;
            let hamilton_matrix = from_rows(2, &[&[(0, -a), (1, -t)], &[(0, -t), (1, -a)]]);
            let groundstate = hamilton_matrix.estimate_groundstate().unwrap();
            let eps = 1.0E-14;

            let eval_0 = -t - a;
            let evec_0 = vec![rsqrt2, rsqrt2];
            let eval_1 = t - a;
            let evec_1 = vec![rsqrt2, -rsqrt2];
            let evec = if eval_0 <= eval_1 { evec_0 } else { evec_1 };
            assert_almost_equal(groundstate, evec, eps);
        }
    }

    #[test]
    fn test_trace() {
        let dim = 3;
        let nnz = 9;
        let mut matrix = SparseMatrix::with_capacity(dim, nnz);
        matrix.push_row(&[0, 1, 2], &[1.0, 2.0, 3.0]);
        matrix.push_row(&[0, 1, 2], &[5.0, -2.5, 4.0]);
        matrix.push_row(&[0, 1, 2], &[9.0, 8.0, 20.1]);
        let trace = matrix.trace();
        assert!((trace - (1.0 - 2.5 + 20.1)).abs() < 1E-30);

        let dim = 4;
        let nnz = 6;
        let mut matrix = SparseMatrix::with_capacity(dim, nnz);
        matrix.push_row(&[0], &[3.8]);
        matrix.push_row(&[1, 2], &[-2.5, 2.1]);
        matrix.push_row(&[2, 3], &[9.0, 2.2]);
        matrix.push_row(&[3], &[-23.2]);
        let trace = matrix.trace();
        assert!((trace - (3.8 - 2.5 + 9.0 - 23.2)).abs() < 1E-30);
    }

    // #[test]
    // fn test_add_to_diagonal() {
    //     // (  1.5  -2.0 )
    //     // ( -2.0   3.0 )
    //     let mut matrix = from_rows(2, &[&[(0, 1.5), (1, -2.0)], &[(0, -2.0), (1, 3.0)]]);
    //     matrix.add_to_diagonal(-0.5);

    //     // (  0.5  -2.0 )
    //     // ( -2.0   2.0 )
    //     let expected = from_rows(2, &[&[(0, 1.0), (1, -2.0)], &[(0, -2.0), (1, 2.5)]]);
    //     assert_almost_equal_sparse_matrices(&matrix, &expected, 1E-20);

    //     matrix.add_to_diagonal(-1.5);

    //     // (  0.5  -2.0 )
    //     // ( -2.0   2.0 )
    //     let expected = from_rows(2, &[&[(0, -0.5), (1, -2.0)], &[(0, -2.0), (1, 1.0)]]);
    //     assert_almost_equal_sparse_matrices(&matrix, &expected, 1E-20);
    // }
}
