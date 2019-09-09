use super::FloatRT;
use std::ops::Mul;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix4X4 {
    pub m: [[FloatRT; 4]; 4],
}

impl Matrix4X4 {
    pub fn new(
        t00: FloatRT,
        t01: FloatRT,
        t02: FloatRT,
        t03: FloatRT,
        t10: FloatRT,
        t11: FloatRT,
        t12: FloatRT,
        t13: FloatRT,
        t20: FloatRT,
        t21: FloatRT,
        t22: FloatRT,
        t23: FloatRT,
        t30: FloatRT,
        t31: FloatRT,
        t32: FloatRT,
        t33: FloatRT,
    ) -> Self {
        Self {
            m: [
                [t00, t01, t02, t03],
                [t10, t11, t12, t13],
                [t20, t21, t22, t23],
                [t30, t31, t32, t33],
            ],
        }
    }

    pub fn from_rows(
        r1: [FloatRT; 4],
        r2: [FloatRT; 4],
        r3: [FloatRT; 4],
        r4: [FloatRT; 4],
    ) -> Self {
        Self {
            m: [r1, r2, r3, r4],
        }
    }

    pub fn identity() -> Self {
        Self::default()
    }

    //TODO: Is this matrix inverse numerically stable enough?
    /// Computes and returns the inverse of a Matrix4X4.
    /// Uses Gauss Jordan with partial pivoting
    pub fn inverse(&self) -> Self {
        let mut temp = self.clone();
        let mut inverse = Self::default();
        let mut permutation = [0, 1, 2, 3];

        for r1 in 0..4 {
            // Find pivot
            let pivot = Matrix4X4::pivot_idx(&temp, r1, &permutation);
            permutation.swap(r1, pivot);
            let pivot_row = permutation[r1];

            // Scale pivot row such that leading element is 1.0
            let pivot_inverse = 1.0 / temp.m[pivot_row][r1];
            for i in 0..4 {
                temp.m[pivot_row][i] *= pivot_inverse;
                inverse.m[pivot_row][i] *= pivot_inverse;
            }

            // Subtract pivot row to eliminate other rows
            for r2 in 0..4 {
                if r2 != r1 {
                    let target_row = permutation[r2];
                    let s = temp.m[target_row][r1] / temp.m[pivot_row][r1];
                    Self::add_row(&mut temp, -s, pivot_row, target_row);
                    Self::add_row(&mut inverse, -s, pivot_row, target_row);
                }
            }
        }

        Matrix4X4::from([
            inverse.m[permutation[0]],
            inverse.m[permutation[1]],
            inverse.m[permutation[2]],
            inverse.m[permutation[3]],
        ])
    }

    fn pivot_idx(m: &Self, column: usize, permutation_indexes: &[usize]) -> usize {
        let mut max_idx = column;
        for i in column..4 {
            let permuted_index = permutation_indexes[i];
            if m.m[permuted_index][column].abs() > m.m[permutation_indexes[max_idx]][column].abs() {
                max_idx = i;
            }
        }
        max_idx
    }

    /// Add s*r1 to r2 in place
    fn add_row(mat: &mut Matrix4X4, s: FloatRT, r1: usize, r2: usize) -> () {
        for i in 0..4 {
            mat.m[r2][i] += mat.m[r1][i] * s;
        }
    }

    pub fn transpose(&self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0], self.m[3][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1], self.m[3][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2], self.m[3][2]],
                [self.m[0][3], self.m[1][3], self.m[2][3], self.m[3][3]],
            ],
        }
    }

    /// Determinant of upper-left 3x3 submatrix
    pub fn determinant(&self) -> FloatRT {
        self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1])
            - self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0])
            + self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0])
    }
}

impl Default for Matrix4X4 {
    fn default() -> Self {
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

impl From<[[FloatRT; 4]; 4]> for Matrix4X4 {
    fn from(a: [[FloatRT; 4]; 4]) -> Self {
        Self { m: a }
    }
}

impl Mul for Matrix4X4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = Self::default();
        for i in 0..4 {
            for j in 0..4 {
                out.m[i][j] = self.m[i][0] * rhs.m[0][j]
                    + self.m[i][1] * rhs.m[1][j]
                    + self.m[i][2] * rhs.m[2][j]
                    + self.m[i][3] * rhs.m[3][j]
            }
        }
        out
    }
}

impl<'a, 'b> Mul<&'b Matrix4X4> for &'a Matrix4X4 {
    type Output = Matrix4X4;
    fn mul(self, rhs: &'b Matrix4X4) -> Self::Output {
        let mut out = Matrix4X4::default();
        for i in 0..4 {
            for j in 0..4 {
                out.m[i][j] = self.m[i][0] * rhs.m[0][j]
                    + self.m[i][1] * rhs.m[1][j]
                    + self.m[i][2] * rhs.m[2][j]
                    + self.m[i][3] * rhs.m[3][j]
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, AbsDiffEq, RelativeEq, UlpsEq};

    #[test]
    fn default() {
        let m = Matrix4X4::default();
        assert_eq!(m.m[0][0], 1.0);
        assert_eq!(m.m[1][1], 1.0);
        assert_eq!(m.m[2][2], 1.0);
        assert_eq!(m.m[3][3], 1.0);
    }

    #[test]
    fn equals() {
        let m1 = Matrix4X4::default();
        let m2 = Matrix4X4::default();
        assert_eq!(m1, m2);
    }

    #[test]
    fn multiply() {
        let m1 = Matrix4X4::from_rows(
            [1., 3., 4., 2.],
            [2., 3., -2., 5.],
            [3., 1., 4., -5.],
            [2., 7., 3., -2.],
        );
        let m2 = Matrix4X4::from_rows(
            [-3., 5., 1., 2.],
            [4., 1., -3., 4.],
            [6., 3., 1., 1.],
            [-2., 4., 7., 2.],
        );

        let res = Matrix4X4::from([
            [29., 28., 10., 22.],
            [-16., 27., 26., 24.],
            [29., 8., -31., 4.],
            [44., 18., -30., 31.],
        ]);
        assert_eq!(m1 * m2, res);
    }

    #[test]
    fn inverse_find_pivot() {
        let m = Matrix4X4::from([
            [2.0, 3.0, 4.0, 3.0],
            [1.2, 3.4, -4.3, 11.9],
            [8.2, 7.0, 0.5, 5.9],
            [12.0, -3.4, 2.3, 5.6],
        ]);

        let permutation = [0, 1, 2, 3];
        let pivot = Matrix4X4::pivot_idx(&m, 0, &permutation);
        assert_eq!(pivot, 3);
        let pivot = Matrix4X4::pivot_idx(&m, 1, &permutation);
        assert_eq!(pivot, 2);
        let pivot = Matrix4X4::pivot_idx(&m, 2, &permutation);
        assert_eq!(pivot, 3);
        let pivot = Matrix4X4::pivot_idx(&m, 3, &permutation);
        assert_eq!(pivot, 3);

        let permutation = [2, 0, 3, 1];
        let pivot = Matrix4X4::pivot_idx(&m, 0, &permutation);
        assert_eq!(pivot, 2);
        let pivot = Matrix4X4::pivot_idx(&m, 1, &permutation);
        assert_eq!(pivot, 2);
        let pivot = Matrix4X4::pivot_idx(&m, 2, &permutation);
        assert_eq!(pivot, 3);
        let pivot = Matrix4X4::pivot_idx(&m, 3, &permutation);
        assert_eq!(pivot, 3);
    }
    #[test]
    fn inverse_add_rows() {
        let mut m = Matrix4X4::from([
            [2.0, 3.0, 4.0, 3.0],
            [1.2, 3.4, -4.3, 11.9],
            [8.2, 7.0, 0.5, 5.9],
            [12.0, -3.4, 2.3, 5.6],
        ]);
        let result = Matrix4X4::from([
            [2.0, 3.0, 4.0, 3.0],
            [5.2, 9.4, 3.7, 17.9],
            [8.2, 7.0, 0.5, 5.9],
            [12.0, -3.4, 2.3, 5.6],
        ]);
        Matrix4X4::add_row(&mut m, 2.0, 0, 1);
        assert_relative_eq!(m, result);
    }

    #[test]
    fn inverse() {
        let m = Matrix4X4::from([
            [2.0, 3.0, 4.0, 3.0],
            [1.2, 3.4, -4.3, 11.9],
            [8.2, 7.0, 0.5, 5.9],
            [12.0, -3.4, 6.3, 5.6],
        ]);
        assert_relative_eq!(m.inverse() * m, Matrix4X4::identity());
    }

    /// Approximate equality implementations for testing purposes
    impl AbsDiffEq for Matrix4X4 {
        type Epsilon = <FloatRT as AbsDiffEq>::Epsilon;
        fn default_epsilon() -> Self::Epsilon {
            FloatRT::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            for i in 0..4 {
                for j in 0..4 {
                    if <FloatRT as AbsDiffEq>::abs_diff_ne(&self.m[i][j], &other.m[i][j], epsilon) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    impl RelativeEq for Matrix4X4 {
        fn default_max_relative() -> <FloatRT as AbsDiffEq>::Epsilon {
            FloatRT::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: <FloatRT as AbsDiffEq>::Epsilon,
        ) -> bool {
            for i in 0..4 {
                for j in 0..4 {
                    if FloatRT::relative_ne(&self.m[i][j], &other.m[i][j], epsilon, max_relative) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    impl UlpsEq for Matrix4X4 {
        fn default_max_ulps() -> u32 {
            FloatRT::default_max_ulps()
        }

        fn ulps_eq(
            &self,
            other: &Self,
            epsilon: <FloatRT as AbsDiffEq>::Epsilon,
            max_ulps: u32,
        ) -> bool {
            for i in 0..4 {
                for j in 0..4 {
                    if FloatRT::ulps_ne(&self.m[i][j], &other.m[i][j], epsilon, max_ulps) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
}
