use super::vector::Vec3;
use super::{Cross, Dot, FloatRT, Scalar};
use assert_approx_eq::assert_approx_eq;
use num_traits::{Float, Num, NumCast, PrimInt, Signed};
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

type Normal3f = Normal3<FloatRT>;
type Normal3i = Normal3<i32>;

// Convenience factories
pub fn normal3f(x: FloatRT, y: FloatRT, z: FloatRT) -> Normal3f {
    Normal3f::new(x, y, z)
}

pub fn normal3i(x: i32, y: i32, z: i32) -> Normal3i {
    Normal3i::new(x, y, z)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Normal3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Scalar> Normal3<T> {
    pub fn new(x: T, y: T, z: T) -> Normal3<T> {
        debug_assert!(x.is_finite() && y.is_finite() && z.is_finite());
        Normal3 { x, y, z }
    }

    pub fn cast<U: Scalar>(self) -> Normal3<U> {
        Normal3::new(
            U::from(self.x).unwrap(),
            U::from(self.y).unwrap(),
            U::from(self.z).unwrap(),
        )
    }

    pub fn length_squared(self) -> FloatRT {
        NumCast::from(self.x * self.x + self.y * self.y + self.z * self.z)
            .expect("Failed cast in Normal3<T>::length_squared")
    }

    pub fn length(self) -> FloatRT {
        self.length_squared().sqrt()
    }

    pub fn normalize(self) -> Self {
        self / T::from(self.length()).unwrap()
    }

    pub fn dist(v1: Normal3<T>, v2: Normal3<T>) -> FloatRT {
        Normal3::length(v2 - v1)
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    //Component wise max
    pub fn max(v1: Self, v2: Self) -> Self {
        Self::new(v1.x.max(v2.x), v1.y.max(v2.y), v1.z.max(v2.z))
    }

    //Component wise max
    pub fn min(v1: Self, v2: Self) -> Self {
        Self::new(v1.x.min(v2.x), v1.y.min(v2.y), v1.z.min(v2.z))
    }

    pub fn min_component(self) -> T {
        self.x.min(self.z).min(self.y)
    }

    pub fn max_component(self) -> T {
        self.x.max(self.z).max(self.y)
    }

    pub fn max_dim(self) -> usize {
        if self.x > self.y {
            if self.x > self.z {
                0
            } else {
                2
            }
        } else {
            if self.y > self.z {
                1
            } else {
                2
            }
        }
    }
}

//Convert a Vec3 to a Normal3
impl<T: Scalar> From<Vec3<T>> for Normal3<T> {
    fn from(p: Vec3<T>) -> Self {
        Self::new(p.x, p.y, p.z)
    }
}

impl<T: Scalar> Dot for Normal3<T> {
    type Output = T;
    fn dot(n1: Self, n2: Self) -> T {
        n1.x * n2.x + n1.y * n2.y + n1.z * n2.z
    }

    fn abs_dot(n1: Self, n2: Self) -> T {
        Self::dot(n1, n2).abs()
    }

    fn face_forward(self, v: Self) -> Self {
        if Self::dot(self, v) < T::zero() {
            -self
        } else {
            self
        }
    }
}

impl<T: Scalar> Dot<Vec3<T>> for Normal3<T> {
    type Output = T;
    fn dot(n1: Self, v2: Vec3<T>) -> T {
        n1.x * v2.x + n1.y * v2.y + n1.z * v2.z
    }

    fn abs_dot(n1: Self, v2: Vec3<T>) -> T {
        Self::dot(n1, v2).abs()
    }

    fn face_forward(self, v: Vec3<T>) -> Self {
        if Self::dot(self, v) < T::zero() {
            -self
        } else {
            self
        }
    }
}

impl<T: Scalar> Index<usize> for Normal3<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Normal3: Bad index op!"),
        }
    }
}

impl<T: Scalar> Add for Normal3<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Scalar> Sub for Normal3<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Scalar> Neg for Normal3<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T: Scalar> Mul<T> for Normal3<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

//TODO: This may not be necesarry once a spectrum model is in use
impl<T: Scalar> Mul<Normal3<T>> for Normal3<T> {
    type Output = Self;
    fn mul(self, rhs: Normal3<T>) -> Self::Output {
        Normal3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl<T: Scalar> Div<T> for Normal3<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        Normal3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn normal_equals() {
        // Floats
        let v1 = normal3f(1.0, 1.0, 1.0);
        let v2 = normal3f(1.0, 1.0, 1.0);
        assert_eq!(v1, v2);

        // Integers
        let v1 = normal3i(1, 1, 1);
        let v2 = normal3i(1, 1, 1);
        assert_eq!(v1, v2);
    }

    #[test]
    fn normal_not_equals() {
        // Floats
        let v1 = normal3f(1.0, 1.0, 1.0);
        let v2 = normal3f(1.0, 2.0, 1.0);
        assert_ne!(v1, v2);

        // Integers
        let v1 = normal3i(1, 1, 1);
        let v2 = normal3i(0, 3, 2);
        assert_ne!(v1, v2);
    }

    #[test]
    #[should_panic]
    fn normal_no_nans() {
        normal3f(FloatRT::nan(), FloatRT::nan(), FloatRT::nan());
    }

    #[test]
    fn normal_add() {
        let v1 = normal3f(1.0, -2.0, 5.0);
        let v2 = normal3f(12.0, 3.0, -1.0);
        assert_eq!(v1 + v2, normal3f(13.0, 1.0, 4.0));
        let v1 = normal3i(3, -2, 5);
        let v2 = normal3i(11, 4, -4);
        assert_eq!(v1 + v2, normal3i(14, 2, 1));
    }

    #[test]
    fn normal_sub() {
        let v1 = normal3f(1.0, -2.0, 5.0);
        let v2 = normal3f(12.0, 3.0, -1.0);
        assert_eq!(v1 - v2, normal3f(-11.0, -5.0, 6.0));

        let v1 = normal3i(1, -2, 5);
        let v2 = normal3i(12, 3, -1);
        assert_eq!(v1 - v2, normal3i(-11, -5, 6));
    }

    #[test]
    fn normal_negate() {
        let v1 = normal3f(1.0, -2.0, 5.0);
        assert_eq!(-v1, normal3f(-1.0, 2.0, -5.0));

        let v1 = normal3i(1, -2, 5);
        assert_eq!(-v1, normal3i(-1, 2, -5));
    }

    #[test]
    fn normal_mul_by_scalar() {
        let v1 = normal3f(2.0, 3.0, 4.0);
        assert_eq!(v1 * 2.0, normal3f(4.0, 6.0, 8.0));
        assert_eq!(v1 * 2.0, normal3f(4.0, 6.0, 8.0));

        let v1 = normal3i(2, 3, 4);
        assert_eq!(v1 * 2, normal3i(4, 6, 8));
        assert_eq!(v1 * 2, normal3i(4, 6, 8));
    }

    #[test]
    fn normal_div_by_scalar() {
        let v1 = normal3f(2.0, 3.0, 4.0);
        assert_eq!(v1 / 2.0, normal3f(1.0, 1.5, 2.0));

        let v1 = normal3i(2, 3, -5);
        assert_eq!(v1 / 2, normal3i(1, 1, -2));
    }

    #[test]
    fn normal_dot() {
        let v1 = normal3f(2.0, 3.0, 4.0);
        let v2 = normal3f(1.0, -2.0, 5.0);
        assert_eq!(Normal3::dot(v1, v2), 16.0);

        let v1 = normal3i(2, 5, 3);
        let v2 = normal3i(1, -5, 2);
        assert_eq!(Normal3::dot(v1, v2), -17);

        // With vector
        let v1 = normal3f(2.0, 3.0, 4.0);
        let v2 = Vec3::new(1.0, -2.0, 5.0);
        assert_eq!(Normal3::dot(v1, v2), 16.0);
    }

    #[test]
    fn normal_cross_with_vec() {
        let n1 = normal3f(2.0, 3.0, 4.0);
        let v2 = Vec3::new(1.0, -2.0, 5.0);
        assert_eq!(Vec3::cross(n1, v2), Vec3::new(23.0, -6.0, -7.0));

        let v1 = Vec3::new(2, 3, 4);
        let n2 = normal3i(1, -2, 5);
        assert_eq!(Vec3::cross(v1, n2), Vec3::new(23, -6, -7));
    }

    #[test]
    fn normal_length() {
        let v = normal3f(2.0, -3.0, 4.0);
        assert_approx_eq!(5.3851648, v.length());

        let v = normal3i(-3, 5, 2);
        assert_approx_eq!(6.164414, v.length());
    }

    #[test]
    fn normal_normalize() {
        let v = normal3f(2.0, -3.0, 4.0);
        let v_normal = v.normalize();
        assert_approx_eq!(v_normal.x, 0.3713906);
        assert_approx_eq!(v_normal.y, -0.55708601);
        assert_approx_eq!(v_normal.z, 0.74278135);
    }

    #[test]
    fn normal_abs() {
        let v = normal3f(-2.2, 4.5, -10.0);
        assert_eq!(v.abs(), normal3f(2.2, 4.5, 10.0));
        let v = normal3i(-2, 4, -10);
        assert_eq!(v.abs(), normal3i(2, 4, 10));
    }

    #[test]
    fn normal_cast() {
        let p1 = normal3f(2.6, 3.3, -4.7);
        let p2: Normal3i = p1.cast();
        assert_eq!(p2, normal3i(2, 3, -4));
    }

    #[test]
    fn normal_index() {
        let v1 = normal3f(1.2, 8.2, -9.4);
        assert_eq!(v1[0], 1.2);
        assert_eq!(v1[1], 8.2);
        assert_eq!(v1[2], -9.4);
    }

    #[test]
    fn normal_max_min() {
        let v1 = normal3f(1.4, -5.5, 8.9);
        let v2 = normal3f(3.4, -7.0, 6.0);
        assert_eq!(v1.max_component(), 8.9);
        assert_eq!(v2.min_component(), -7.0);
        assert_eq!(Normal3::max(v1, v2), normal3f(3.4, -5.5, 8.9));
        assert_eq!(Normal3::min(v1, v2), normal3f(1.4, -7.0, 6.0));
        assert_eq!(v1.max_dim(), 2);
    }

    #[test]
    fn normal_face_forward() {
        let n = normal3f(1.0, 1.0, 1.0);
        let v = Vec3::new(-0.5, 1.0, -1.0);
        assert_eq!(n.face_forward(v), normal3f(-1.0, -1.0, -1.0));

        let v = Vec3::new(0.5, 1.0, 0.5);
        assert_eq!(n.face_forward(v), normal3f(1.0, 1.0, 1.0));
    }

}
