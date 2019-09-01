use super::{FloatRT, Scalar};
use super::vector::Vec3;
use assert_approx_eq::assert_approx_eq;
use num_traits::{Float, Num, NumCast, PrimInt, Signed};
use std::ops::{Add, Div, Index, Mul, Neg, Sub};


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

    pub fn dot(v1: Normal3<T>, v2: Normal3<T>) -> T {
        v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    }

    pub fn normalize(self) -> Normal3<T> {
        self / T::from(self.length()).unwrap()
    }

    pub fn dist(v1: Normal3<T>, v2: Normal3<T>) -> FloatRT {
        Normal3::length(v2 - v1)
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    pub fn abs_dot(v1: Self, v2: Self) -> T {
        Self::dot(v1, v2).abs()
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

    pub fn permute(self, x: usize, y: usize, z: usize) -> Self {
        Self::new(self[x], self[y], self[z])
    }

    // pub fn reflect(v: Normal3<T>, n: Normal3<T>) -> Normal3<T> {
    //     v - n * 2.0 * Normal3::dot(v, n)
    // }

    // pub fn lerp(v1: Normal3<T>, v2: Normal3<T>, t: f32) -> Normal3<T> {
    //     v1 * (1.0 - t)  + v2 * t
    // }
}

//Convert a Vec3 to a Normal3
impl<T: Scalar> From<Vec3<T>> for Normal3<T> {
    fn from(p: Vec3<T>) -> Self {
        Self::new(p.x, p.y, p.z)
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