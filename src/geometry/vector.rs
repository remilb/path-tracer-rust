use super::normal::Normal3;
use super::point::{Point2, Point3};
use super::{Cross, Dot, FloatRT, Scalar};
use assert_approx_eq::assert_approx_eq;
use num_traits::{Float, Num, NumCast, PrimInt, Signed};
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

// Switching this alias between f32 and f64 will switch precision of all vector math

// Convenience aliases
pub type Vec3f = Vec3<FloatRT>;
pub type Vec3i = Vec3<i32>;
pub type Vec2f = Vec2<FloatRT>;
pub type Vec2i = Vec2<i32>;

// Convenience factories
pub fn vec3f(x: FloatRT, y: FloatRT, z: FloatRT) -> Vec3f {
    Vec3f::new(x, y, z)
}

pub fn vec3i(x: i32, y: i32, z: i32) -> Vec3i {
    Vec3i::new(x, y, z)
}

pub fn vec2f(x: FloatRT, y: FloatRT) -> Vec2f {
    Vec2f::new(x, y)
}

pub fn vec2i(x: i32, y: i32) -> Vec2i {
    Vec2i::new(x, y)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Scalar> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Vec3<T> {
        debug_assert!(x.is_finite() && y.is_finite() && z.is_finite());
        Vec3 { x, y, z }
    }

    pub fn zeros() -> Vec3<T> {
        Vec3 {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    pub fn ones() -> Vec3<T> {
        Vec3 {
            x: T::one(),
            y: T::one(),
            z: T::one(),
        }
    }

    pub fn cast<U: Scalar>(self) -> Vec3<U> {
        Vec3::new(
            U::from(self.x).unwrap(),
            U::from(self.y).unwrap(),
            U::from(self.z).unwrap(),
        )
    }

    pub fn length_squared(self) -> FloatRT {
        NumCast::from(self.x * self.x + self.y * self.y + self.z * self.z)
            .expect("Failed cast in Vec3<T>::length_squared")
    }

    pub fn length(self) -> FloatRT {
        self.length_squared().sqrt()
    }

    pub fn normalize(self) -> Vec3<T> {
        self / T::from(self.length()).unwrap()
    }

    pub fn dist(v1: Vec3<T>, v2: Vec3<T>) -> FloatRT {
        Vec3::length(v2 - v1)
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

    pub fn permute(self, x: usize, y: usize, z: usize) -> Self {
        Self::new(self[x], self[y], self[z])
    }

    // Orthonormal basis from single vector
    pub fn coordinate_system(v: Self) -> (Self, Self, Self) {
        if v.x.abs() > v.y.abs() {
            let v2 = Self::new(-v.z, T::zero(), v.x).normalize();
            let v3 = Self::cross(v, v2);
            (v, v2, v3)
        } else {
            let v2 = Self::new(T::zero(), v.z, -v.y).normalize();
            let v3 = Self::cross(v, v2);
            (v, v2, v3)
        }
    }
}

//Convert a Point3 to a Vec3
impl<T: Scalar> From<Point3<T>> for Vec3<T> {
    fn from(p: Point3<T>) -> Self {
        Self::new(p.x, p.y, p.z)
    }
}

// Dot products
impl<T: Scalar> Dot for Vec3<T> {
    type Output = T;
    fn dot(v1: Self, v2: Self) -> T {
        v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    }

    fn abs_dot(v1: Self, v2: Self) -> T {
        Self::dot(v1, v2).abs()
    }

    fn face_forward(self, v: Self) -> Self {
        if Self::dot(self, v) < T::zero() {
            -self
        } else {
            self
        }
    }
}

impl<T: Scalar> Dot<Normal3<T>> for Vec3<T> {
    type Output = T;
    fn dot(v1: Self, n2: Normal3<T>) -> T {
        v1.x * n2.x + v1.y * n2.y + v1.z * n2.z
    }

    fn abs_dot(v1: Self, n2: Normal3<T>) -> T {
        Self::dot(v1, n2).abs()
    }

    fn face_forward(self, v: Normal3<T>) -> Self {
        if Self::dot(self, v) < T::zero() {
            -self
        } else {
            self
        }
    }
}

impl<T: Scalar> From<Normal3<T>> for Vec3<T> {
    fn from(p: Normal3<T>) -> Self {
        Self::new(p.x, p.y, p.z)
    }
}

// Cross products
impl<T: Scalar> Cross for Vec3<T> {
    fn cross(v1: Self, v2: Self) -> Self {
        // Lift to double to prevent catastrophic cancellation, as per PBRT
        let v1: Vec3<f64> = v1.cast();
        let v2: Vec3<f64> = v2.cast();
        Vec3::new(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x,
        )
        .cast()
    }
}

impl<T: Scalar> Cross<Vec3<T>, Normal3<T>> for Vec3<T> {
    fn cross(v1: Self, v2: Normal3<T>) -> Self {
        // Lift to double to prevent catastrophic cancellation, as per PBRT
        let v1: Vec3<f64> = v1.cast();
        let v2: Normal3<f64> = v2.cast();
        Vec3::new(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x,
        )
        .cast()
    }
}

impl<T: Scalar> Cross<Normal3<T>, Vec3<T>> for Vec3<T> {
    fn cross(v1: Normal3<T>, v2: Self) -> Self {
        // Lift to double to prevent catastrophic cancellation, as per PBRT
        let v1: Normal3<f64> = v1.cast();
        let v2: Vec3<f64> = v2.cast();
        Vec3::new(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x,
        )
        .cast()
    }
}

impl<T: Scalar> Index<usize> for Vec3<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3: Bad index op!"),
        }
    }
}

impl<T: Scalar> Add for Vec3<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Scalar> Sub for Vec3<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Scalar> Neg for Vec3<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T: Scalar> Mul<T> for Vec3<T> {
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
impl<T: Scalar> Mul<Vec3<T>> for Vec3<T> {
    type Output = Self;
    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl<T: Scalar> Div<T> for Vec3<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

//Vec2
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T: Scalar> Vec2<T> {
    pub fn new(x: T, y: T) -> Vec2<T> {
        debug_assert!(x.is_finite() && y.is_finite());
        Vec2 { x, y }
    }

    pub fn cast<U: Scalar>(self) -> Vec2<U> {
        Vec2::new(U::from(self.x).unwrap(), U::from(self.y).unwrap())
    }

    pub fn zeros() -> Vec2<T> {
        Vec2 {
            x: T::zero(),
            y: T::zero(),
        }
    }

    pub fn ones() -> Vec2<T> {
        Vec2 {
            x: T::one(),
            y: T::one(),
        }
    }

    pub fn length_squared(self) -> FloatRT {
        NumCast::from(self.x * self.x + self.y * self.y)
            .expect("Failed cast in Vec2<T>::length_squared")
    }

    pub fn length(self) -> FloatRT {
        self.length_squared().sqrt()
    }

    pub fn dot(v1: Vec2<T>, v2: Vec2<T>) -> T {
        v1.x * v2.x + v1.y * v2.y
    }

    pub fn normalize(self) -> Vec2<T> {
        self / T::from(self.length()).unwrap()
    }

    pub fn dist(v1: Vec2<T>, v2: Vec2<T>) -> FloatRT {
        Vec2::length(v2 - v1)
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }

    pub fn abs_dot(v1: Self, v2: Self) -> T {
        Self::dot(v1, v2).abs()
    }

    //Component wise max
    pub fn max(v1: Self, v2: Self) -> Self {
        Self::new(v1.x.max(v2.x), v1.y.max(v2.y))
    }

    //Component wise max
    pub fn min(v1: Self, v2: Self) -> Self {
        Self::new(v1.x.min(v2.x), v1.y.min(v2.y))
    }

    pub fn min_component(self) -> T {
        self.x.min(self.y)
    }

    pub fn max_component(self) -> T {
        self.x.max(self.y)
    }

    pub fn max_dim(self) -> usize {
        if self.x > self.y {
            0
        } else {
            1
        }
    }

    pub fn permute(self, x: usize, y: usize) -> Self {
        Self::new(self[x], self[y])
    }
}

//Convert a Point2 to a Vec2
impl<T: Scalar> From<Point2<T>> for Vec2<T> {
    fn from(p: Point2<T>) -> Self {
        Self::new(p.x, p.y)
    }
}

impl<T: Scalar> Index<usize> for Vec2<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Vec3: Bad index op!"),
        }
    }
}

impl<T: Scalar> Add for Vec2<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<T: Scalar> Sub for Vec2<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<T: Scalar> Neg for Vec2<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl<T: Scalar> Mul<T> for Vec2<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

//TODO: This may not be necesarry once a spectrum model is in use
impl<T: Scalar> Mul<Vec2<T>> for Vec2<T> {
    type Output = Self;
    fn mul(self, rhs: Vec2<T>) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl<T: Scalar> Div<T> for Vec2<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn vector_equals() {
        // Floats
        let v1 = vec3f(1.0, 1.0, 1.0);
        let v2 = vec3f(1.0, 1.0, 1.0);
        assert_eq!(v1, v2);
        let v1 = vec2f(1.0, 1.0);
        let v2 = vec2f(1.0, 1.0);
        assert_eq!(v1, v2);

        // Integers
        let v1 = vec3i(1, 1, 1);
        let v2 = vec3i(1, 1, 1);
        assert_eq!(v1, v2);
        let v1 = vec2i(1, 1);
        let v2 = vec2i(1, 1);
        assert_eq!(v1, v2);
    }

    #[test]
    fn vector_not_equals() {
        // Floats
        let v1 = vec3f(1.0, 1.0, 1.0);
        let v2 = vec3f(1.0, 2.0, 1.0);
        assert_ne!(v1, v2);
        let v1 = vec2f(2.0, 1.0);
        let v2 = vec2f(1.0, 1.0);
        assert_ne!(v1, v2);

        // Integers
        let v1 = vec3i(1, 1, 1);
        let v2 = vec3i(0, 3, 2);
        assert_ne!(v1, v2);
        let v1 = vec2i(1, 1);
        let v2 = vec2i(0, 3);
        assert_ne!(v1, v2);
    }

    #[test]
    #[should_panic]
    fn vector_no_nans() {
        vec3f(FloatRT::nan(), FloatRT::nan(), FloatRT::nan());
    }

    #[test]
    fn vector_add() {
        let v1 = vec3f(1.0, -2.0, 5.0);
        let v2 = vec3f(12.0, 3.0, -1.0);
        assert_eq!(v1 + v2, vec3f(13.0, 1.0, 4.0));
        let v1 = vec3i(3, -2, 5);
        let v2 = vec3i(11, 4, -4);
        assert_eq!(v1 + v2, vec3i(14, 2, 1));
        let v1 = vec2f(1.0, -2.0);
        let v2 = vec2f(12.0, 3.0);
        assert_eq!(v1 + v2, vec2f(13.0, 1.0));
        let v1 = vec2i(3, -2);
        let v2 = vec2i(11, 4);
        assert_eq!(v1 + v2, vec2i(14, 2));
    }

    #[test]
    fn vector_sub() {
        let v1 = vec3f(1.0, -2.0, 5.0);
        let v2 = vec3f(12.0, 3.0, -1.0);
        assert_eq!(v1 - v2, vec3f(-11.0, -5.0, 6.0));

        let v1 = vec3i(1, -2, 5);
        let v2 = vec3i(12, 3, -1);
        assert_eq!(v1 - v2, vec3i(-11, -5, 6));

        let v1 = vec2f(1.0, -2.0);
        let v2 = vec2f(12.0, 3.0);
        assert_eq!(v1 - v2, vec2f(-11.0, -5.0));

        let v1 = vec2i(1, -2);
        let v2 = vec2i(12, 3);
        assert_eq!(v1 - v2, vec2i(-11, -5));
    }

    #[test]
    fn vector_negate() {
        let v1 = vec3f(1.0, -2.0, 5.0);
        assert_eq!(-v1, vec3f(-1.0, 2.0, -5.0));

        let v1 = vec3i(1, -2, 5);
        assert_eq!(-v1, vec3i(-1, 2, -5));

        let v1 = vec2f(1.0, -2.0);
        assert_eq!(-v1, vec2f(-1.0, 2.0));

        let v1 = vec2i(1, -2);
        assert_eq!(-v1, vec2i(-1, 2));
    }

    #[test]
    fn vector_mul_by_scalar() {
        let v1 = vec3f(2.0, 3.0, 4.0);
        assert_eq!(v1 * 2.0, vec3f(4.0, 6.0, 8.0));
        assert_eq!(v1 * 2.0, vec3f(4.0, 6.0, 8.0));

        let v1 = vec3i(2, 3, 4);
        assert_eq!(v1 * 2, vec3i(4, 6, 8));
        assert_eq!(v1 * 2, vec3i(4, 6, 8));

        let v1 = vec2f(2.0, 3.0);
        assert_eq!(v1 * 2.0, vec2f(4.0, 6.0));
        assert_eq!(v1 * 2.0, vec2f(4.0, 6.0));

        let v1 = vec2i(2, 3);
        assert_eq!(v1 * 2, vec2i(4, 6));
        assert_eq!(v1 * 2, vec2i(4, 6));
    }

    #[test]
    fn vector_div_by_scalar() {
        let v1 = vec3f(2.0, 3.0, 4.0);
        assert_eq!(v1 / 2.0, vec3f(1.0, 1.5, 2.0));

        let v1 = vec3i(2, 3, -5);
        assert_eq!(v1 / 2, vec3i(1, 1, -2));

        let v1 = vec2f(2.0, 3.0);
        assert_eq!(v1 / 2.0, vec2f(1.0, 1.5));

        let v1 = vec2i(2, 3);
        assert_eq!(v1 / 2, vec2i(1, 1));
    }

    #[test]
    fn vector_dot() {
        let v1 = vec3f(2.0, 3.0, 4.0);
        let v2 = vec3f(1.0, -2.0, 5.0);
        assert_eq!(Vec3::dot(v1, v2), 16.0);

        let v1 = vec3i(2, 5, 3);
        let v2 = vec3i(1, -5, 2);
        assert_eq!(Vec3::dot(v1, v2), -17);

        let v1 = vec2f(2.0, 3.0);
        let v2 = vec2f(1.0, -2.0);
        assert_eq!(Vec2::dot(v1, v2), -4.0);

        let v1 = vec2i(2, 5);
        let v2 = vec2i(1, -5);
        assert_eq!(Vec2::dot(v1, v2), -23);
    }

    #[test]
    fn vector_cross() {
        let v1 = vec3f(2.0, 3.0, 4.0);
        let v2 = vec3f(1.0, -2.0, 5.0);
        assert_eq!(Vec3::cross(v1, v2), vec3f(23.0, -6.0, -7.0));

        let v1 = vec3i(2, 3, 4);
        let v2 = vec3i(1, -2, 5);
        assert_eq!(Vec3::cross(v1, v2), vec3i(23, -6, -7));
    }

    #[test]
    fn vector_length() {
        let v = vec3f(2.0, -3.0, 4.0);
        assert_approx_eq!(5.3851648, v.length());

        let v = vec3i(-3, 5, 2);
        assert_approx_eq!(6.164414, v.length());

        let v = vec2f(2.0, -3.0);
        assert_approx_eq!(3.6055512, v.length());

        let v = vec2i(-3, 5);
        assert_approx_eq!(5.8309519, v.length())
    }

    #[test]
    fn vector_normalize() {
        let v = vec3f(2.0, -3.0, 4.0);
        let v_normal = v.normalize();
        assert_approx_eq!(v_normal.x, 0.3713906);
        assert_approx_eq!(v_normal.y, -0.55708601);
        assert_approx_eq!(v_normal.z, 0.74278135);
    }

    #[test]
    fn vector_abs() {
        let v = vec3f(-2.2, 4.5, -10.0);
        assert_eq!(v.abs(), vec3f(2.2, 4.5, 10.0));
        let v = vec3i(-2, 4, -10);
        assert_eq!(v.abs(), vec3i(2, 4, 10));
        let v = vec2f(-2.2, 4.5);
        assert_eq!(v.abs(), vec2f(2.2, 4.5));
        let v = vec2i(-2, 4);
        assert_eq!(v.abs(), vec2i(2, 4));
    }

    #[test]
    fn vector_cast() {
        let p1 = vec3f(2.6, 3.3, -4.7);
        let p2: Vec3i = p1.cast();
        assert_eq!(p2, vec3i(2, 3, -4));

        let p1 = vec2i(2, -3);
        let p2: Vec2f = p1.cast();
        assert_eq!(p2, vec2f(2.0, -3.0));
    }

    #[test]
    fn vector_index() {
        let v1 = vec3f(1.2, 8.2, -9.4);
        assert_eq!(v1[0], 1.2);
        assert_eq!(v1[1], 8.2);
        assert_eq!(v1[2], -9.4);

        let v1 = vec2i(3, 4);
        assert_eq!(v1[0], 3);
        assert_eq!(v1[1], 4);
    }

    #[test]
    fn vector_max_min() {
        let v1 = vec3f(1.4, -5.5, 8.9);
        let v2 = vec3f(3.4, -7.0, 6.0);
        assert_eq!(v1.max_component(), 8.9);
        assert_eq!(v2.min_component(), -7.0);
        assert_eq!(Vec3::max(v1, v2), vec3f(3.4, -5.5, 8.9));
        assert_eq!(Vec3::min(v1, v2), vec3f(1.4, -7.0, 6.0));
        assert_eq!(v1.max_dim(), 2);

        let v1 = vec2i(1, -5);
        let v2 = vec2i(3, -7);
        assert_eq!(v1.max_component(), 1);
        assert_eq!(v2.min_component(), -7);
        assert_eq!(Vec2::max(v1, v2), vec2i(3, -5));
        assert_eq!(Vec2::min(v1, v2), vec2i(1, -7));
        assert_eq!(v1.max_dim(), 0);
    }

    #[test]
    fn vector_permute() {
        let v1 = vec3f(1.3, 4.2, 6.7);
        assert_eq!(v1.permute(1, 2, 0), vec3f(4.2, 6.7, 1.3));
    }

    #[test]
    fn vector_coordinate_system() {
        let v1 = vec3f(1.3, 4.2, 6.7);
        let (v1, v2, v3) = Vec3::coordinate_system(v1.normalize());
        assert_approx_eq!(Vec3::dot(v1, v2), 0.0);
        assert_approx_eq!(Vec3::dot(v3, v2), 0.0);
        assert_approx_eq!(Vec3::dot(v1, v3), 0.0);
        assert_approx_eq!(v1.length(), 1.0);
        assert_approx_eq!(v2.length(), 1.0);
        assert_approx_eq!(v3.length(), 1.0);
    }
}
