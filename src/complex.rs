//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Complex numbers and sparse complex vectors.

/// A complex number.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(16))]
pub struct Complex {
    /// Real component.
    pub real: f64,

    /// Imaginary component.
    pub imag: f64,
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.imag < 0.0 {
            write!(f, "{}-{}j", self.real, -self.imag)
        } else {
            write!(f, "{}+{}j", self.real, self.imag)
        }
    }
}

impl std::ops::Add<Complex> for Complex {
    type Output = Complex;

    fn add(self, rhs: Complex) -> Self::Output {
        Complex {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}

impl std::ops::Sub<Complex> for Complex {
    type Output = Complex;

    fn sub(self, rhs: Complex) -> Self::Output {
        Complex {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Complex;

    fn mul(self, rhs: f64) -> Self::Output {
        Complex {
            real: self.real * rhs,
            imag: self.imag * rhs,
        }
    }
}

impl std::ops::Mul<Complex> for Complex {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Self::Output {
        Complex {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }
}

impl Complex {
    /// Creates the complex number zero.
    #[inline]
    pub fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }

    /// Creates a complex number from real and imaginary parts.
    #[inline]
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    /// Returns the squared absolute value.
    #[inline]
    pub fn abs_squared(&self) -> f64 {
        let real = self.real;
        let imag = self.imag;
        return real * real + imag * imag;
    }

    /// Computes the complex exponential.
    #[inline]
    pub fn exp(&self) -> Complex {
        let exp_real = self.real.exp();
        let (s, c) = self.imag.sin_cos();
        Complex {
            real: exp_real * c,
            imag: exp_real * s,
        }
    }

    /// Computes `self.adjoint() * rhs`
    #[inline]
    pub fn adj_mul(&self, rhs: Complex) -> Complex {
        Complex {
            real: self.real * rhs.real + self.imag * rhs.imag,
            imag: self.real * rhs.imag - self.imag * rhs.real,
        }
    }

    /// Multiplies by `-i * t`.
    #[inline]
    pub fn mul_minus_i_times(&self, t: f64) -> Complex {
        Complex {
            real: t * self.imag,
            imag: -t * self.real,
        }
    }
}

/// Computes the squared norm of a dense complex vector.
pub fn norm_squared(v: &[Complex]) -> f64 {
    v.iter().fold(0.0, |acc, z| acc + z.abs_squared())
}

/// A sparse vector of complex numbers.
#[derive(Debug)]
pub struct SparseComplexVector {
    /// Dimension of the vector.
    dim: usize,

    /// A vector of value/index tuples where the indices are sorted in ascending order.
    duplets: Vec<(usize, Complex)>,
}

impl SparseComplexVector {
    /// Creates a sparse complex vector of the given dimension from a vector of duplets.
    pub fn from_unsorted_duplets(dim: usize, duplets: Vec<(usize, Complex)>) -> Self {
        let mut duplets = duplets;
        duplets.sort_by_key(|k| k.0);
        let mut v = Self { dim, duplets };
        v.normalize();
        v.normalize();
        v
    }

    // /// The dimension of the vector.
    // pub fn dimension(&self) -> usize {
    //     self.dim
    // }

    /// Gets the states as a list of duplets.
    // pub fn get_duplets(&self) -> &[(usize, Complex)] {
    //     &self.duplets
    // }

    /// Returns a dense representation of this vector.
    pub fn to_dense(&self) -> Vec<Complex> {
        let mut v = vec![Complex::zero(); self.dim];
        for duplet in self.duplets.iter() {
            v[duplet.0] = duplet.1;
        }
        v
    }

    /// Calculates the squared norm of the vector.
    pub fn norm_squared(&self) -> f64 {
        self.duplets
            .iter()
            .fold(0.0, |acc, d| acc + d.1.abs_squared())
    }

    /// Calculates the squared norm of the vector.
    pub fn normalize(&mut self) {
        let norm = self.norm_squared().sqrt();
        if norm != 0.0 {
            let s = 1.0 / norm;
            for duplet in self.duplets.iter_mut() {
                duplet.1.real *= s;
                duplet.1.imag *= s;
            }
        } else {
            panic!("attempted to normalize a vector of norm zero");
        }
    }
}

/// Normalizes a vector of complex numbers in-place.
pub fn normalize_complex_vector(state: &mut [Complex]) -> f64 {
    let mut norm2 = 0.0;
    for z in state.iter() {
        norm2 += z.abs_squared();
    }
    if norm2 == 0.0 {
        panic!("attempted to normalize a zero vector");
    }
    let norm = norm2.sqrt();
    let recip_norm = 1.0 / norm;
    for z in state {
        *z = *z * recip_norm;
    }
    norm
}

mod test {

    #[cfg(test)]
    use super::Complex;

    #[cfg(test)]
    fn almost_equal_f64(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[cfg(test)]
    fn almost_equal_complex(a: Complex, b: Complex, eps: f64) -> bool {
        (a - b).abs_squared() < (eps * eps)
    }

    #[test]
    fn test_complex_operators() {
        let eps = 1.0E-16;
        let a = Complex::new(2.0, 1.5);
        let aa = a.adj_mul(a);
        assert!(almost_equal_f64(a.abs_squared(), aa.real, eps));
        assert!(almost_equal_f64(0.0, aa.imag, eps));

        let a_mul_3i = a.mul_minus_i_times(3.0);
        assert!(almost_equal_complex(a_mul_3i, Complex::new(4.5, -6.0), eps));
    }
}
