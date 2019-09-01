use super::vector::{vec2f, vec2i, vec3f, vec3i, Vec2, Vec3};
use super::FloatRT;
use super::Scalar;
use assert_approx_eq::assert_approx_eq;
use num_traits::{Float, Num, NumCast, PrimInt, Signed};
use std::ops::{Add, Index, Mul, Sub};

// Convenience aliases
type Point3f = Point3<FloatRT>;
type Point3i = Point3<i32>;
type Point2f = Point2<FloatRT>;
type Point2i = Point2<i32>;

// Convenience factories
pub fn point3f(x: FloatRT, y: FloatRT, z: FloatRT) -> Point3f {
    Point3f::new(x, y, z)
}

pub fn point3i(x: i32, y: i32, z: i32) -> Point3i {
    Point3i::new(x, y, z)
}

pub fn point2f(x: FloatRT, y: FloatRT) -> Point2f {
    Point2f::new(x, y)
}

pub fn point2i(x: i32, y: i32) -> Point2i {
    Point2i::new(x, y)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Scalar> Point3<T> {
    pub fn new(x: T, y: T, z: T) -> Point3<T> {
        debug_assert!(x.is_finite() && y.is_finite() && z.is_finite());
        Point3 { x, y, z }
    }

    pub fn cast<U: Scalar>(self) -> Point3<U> {
        Point3 {
            x: U::from(self.x).unwrap(),
            y: U::from(self.y).unwrap(),
            z: U::from(self.z).unwrap(),
        }
    }

    pub fn zeros() -> Point3<T> {
        Point3 {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    pub fn ones() -> Point3<T> {
        Point3 {
            x: T::one(),
            y: T::one(),
            z: T::one(),
        }
    }

    pub fn lerp(t: FloatRT, p1: Point3<T>, p2: Point3<T>) -> Point3<T> {
        (Point3::lift_to_float(p1) * (1.0 - t) + Point3::lift_to_float(p2) * t).cast()
    }

    fn lift_to_float(p: Point3<T>) -> Point3<FloatRT> {
        Point3 {
            x: NumCast::from(p.x).unwrap(),
            y: NumCast::from(p.y).unwrap(),
            z: NumCast::from(p.z).unwrap(),
        }
    }

    //Component wise max
    pub fn max(p1: Point3<T>, p2: Point3<T>) -> Point3<T> {
        Self {
            x: p1.x.max(p2.x),
            y: p1.y.max(p2.y),
            z: p1.z.max(p2.z),
        }
    }

    pub fn truncate(self) -> Point2<T> {
        Point2::new(self.x, self.y)
    }

    pub fn dist(p1: Point3<T>, p2: Point3<T>) -> FloatRT {
        Vec3::length(p2 - p1)
    }

    pub fn floor(self) -> Point3<T> {
        Self::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    pub fn ceil(self) -> Point3<T> {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    pub fn abs(self) -> Point3<T> {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    pub fn permute(self, x: usize, y: usize, z: usize) -> Self {
        Self::new(self[x], self[y], self[z])
    }
}

impl<T: Scalar> Index<usize> for Point3<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Point3: Bad index op!"),
        }
    }
}

//Create Point3 from Vec3
impl<T: Scalar, U: Scalar> From<Vec3<U>> for Point3<T> {
    fn from(v: Vec3<U>) -> Point3<T> {
        Self::new(
            T::from(v.x).unwrap(),
            T::from(v.y).unwrap(),
            T::from(v.z).unwrap(),
        )
    }
}

//Add a vec3 to a point3
impl<T> Add<Vec3<T>> for Point3<T>
where
    T: Scalar,
{
    type Output = Self;
    fn add(self, rhs: Vec3<T>) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

//Subtract a vec3 from a point3
impl<T> Sub<Vec3<T>> for Point3<T>
where
    T: Scalar,
{
    type Output = Self;
    fn sub(self, rhs: Vec3<T>) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

//Add a point3 to a point3, for the purpose of weighted sums
impl<T> Add for Point3<T>
where
    T: Scalar,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

//Multiply a point3 by a scalar, for the purpose of weighted sums
impl<T> Mul<T> for Point3<T>
where
    T: Scalar,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
//Subtract a point3 from a point3
impl<T> Sub for Point3<T>
where
    T: Scalar,
{
    type Output = Vec3<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

// Point2
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

impl<T: Scalar> Point2<T> {
    pub fn new(x: T, y: T) -> Point2<T> {
        debug_assert!(x.is_finite() && y.is_finite());
        Point2 { x, y }
    }

    pub fn cast<U: Scalar>(self) -> Point2<U> {
        Point2 {
            x: U::from(self.x).unwrap(),
            y: U::from(self.y).unwrap(),
        }
    }

    pub fn zeros() -> Point2<T> {
        Point2 {
            x: T::zero(),
            y: T::zero(),
        }
    }

    pub fn ones() -> Point2<T> {
        Point2 {
            x: T::one(),
            y: T::one(),
        }
    }

    pub fn lerp(t: FloatRT, p1: Point2<T>, p2: Point2<T>) -> Point2<T> {
        (Point2::lift_to_float(p1) * (1.0 - t) + Point2::lift_to_float(p2) * t).cast()
    }

    fn lift_to_float(p: Point2<T>) -> Point2<FloatRT> {
        Point2 {
            x: NumCast::from(p.x).unwrap(),
            y: NumCast::from(p.y).unwrap(),
        }
    }

    //Component wise max
    pub fn max(p1: Point2<T>, p2: Point2<T>) -> Point2<T> {
        Self {
            x: p1.x.max(p2.x),
            y: p1.y.max(p2.y),
        }
    }

    pub fn extend(self) -> Point3<T> {
        Point3::new(self.x, self.y, T::zero())
    }

    pub fn dist(p1: Point2<T>, p2: Point2<T>) -> FloatRT {
        Vec2::length(p2 - p1)
    }

    pub fn floor(self) -> Point2<T> {
        Self::new(self.x.floor(), self.y.floor())
    }

    pub fn ceil(self) -> Point2<T> {
        Self::new(self.x.ceil(), self.y.ceil())
    }

    pub fn abs(self) -> Point2<T> {
        Self::new(self.x.abs(), self.y.abs())
    }

    pub fn permute(self, x: usize, y: usize) -> Self {
        Self::new(self[x], self[y])
    }
}

impl<T: Scalar> Index<usize> for Point2<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Point3: Bad index op!"),
        }
    }
}

//Create Point2 from Vec2
impl<T: Scalar, U: Scalar> From<Vec2<U>> for Point2<T> {
    fn from(v: Vec2<U>) -> Point2<T> {
        Self::new(T::from(v.x).unwrap(), T::from(v.y).unwrap())
    }
}

//Add a Vec2 to a Point2
impl<T> Add<Vec2<T>> for Point2<T>
where
    T: Scalar,
{
    type Output = Self;
    fn add(self, rhs: Vec2<T>) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

//Subtract a Vec2 from a Point2
impl<T> Sub<Vec2<T>> for Point2<T>
where
    T: Scalar,
{
    type Output = Self;
    fn sub(self, rhs: Vec2<T>) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

//Add a Point2 to a Point2, for the purpose of weighted sums
impl<T> Add for Point2<T>
where
    T: Scalar,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

//Multiply a Point2 by a scalar, for the purpose of weighted sums
impl<T> Mul<T> for Point2<T>
where
    T: Scalar,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}
//Subtract a Point2 from a Point2
impl<T> Sub for Point2<T>
where
    T: Scalar,
{
    type Output = Vec2<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.x - rhs.x, self.y - rhs.y)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn point_equals() {
        // Floats
        let p1 = point3f(1.0, 1.0, 1.0);
        let p2 = point3f(1.0, 1.0, 1.0);
        assert_eq!(p1, p2);
        let p3 = point2f(2.0, 2.0);
        let p4 = point2f(2.0, 2.0);
        assert_eq!(p3, p4);

        // Integers
        let p1 = point3i(1, 1, 1);
        let p2 = point3i(1, 1, 1);
        assert_eq!(p1, p2);
        let p3 = point2i(2, 2);
        let p4 = point2i(2, 2);
        assert_eq!(p3, p4);
    }

    #[test]
    fn point_not_equals() {
        // Floats
        let p1 = point3f(1.0, 1.0, 1.0);
        let p2 = point3f(1.0, 2.0, 1.0);
        assert_ne!(p1, p2);
        let p3 = point2f(2.2, 2.2);
        let p4 = point2f(2.0, 2.0);
        assert_ne!(p3, p4);

        // Integers
        let p1 = point3i(1, 1, 1);
        let p2 = point3i(0, 3, 2);
        assert_ne!(p1, p2);
        let p3 = point2i(2, 3);
        let p4 = point2i(3, 2);
        assert_ne!(p3, p4);
    }

    #[test]
    #[should_panic]
    fn point_no_nans() {
        point3f(FloatRT::nan(), FloatRT::nan(), FloatRT::nan());
    }

    #[test]
    fn point_extend_truncate() {
        let p = point3f(2.0, 3.0, 4.0);
        let result = point2f(2.0, 3.0);
        assert_eq!(p.truncate(), result);
        let p = p.truncate();
        let result = point3f(2.0, 3.0, 0.0);
        assert_eq!(p.extend(), result);
    }

    #[test]
    fn point_add() {
        let p1 = point3f(1.0, -2.0, 5.0);
        let p2 = point3f(12.0, 3.0, -1.0);
        assert_eq!(p1 + p2, point3f(13.0, 1.0, 4.0));

        let p1 = point3i(3, -2, 5);
        let p2 = point3i(11, 4, -4);
        assert_eq!(p1 + p2, point3i(14, 2, 1));

        let p1 = point2f(1.0, -2.0);
        let p2 = point2f(12.0, 3.0);
        assert_eq!(p1 + p2, point2f(13.0, 1.0));

        let p1 = point2i(3, -2);
        let p2 = point2i(11, 4);
        assert_eq!(p1 + p2, point2i(14, 2));
    }

    #[test]
    fn point_add_vector() {
        let p1 = point3f(1.0, -2.0, 5.0);
        let v2 = vec3f(12.0, 3.0, -1.0);
        assert_eq!(p1 + v2, point3f(13.0, 1.0, 4.0));

        let p1 = point3i(3, -2, 5);
        let v2 = vec3i(11, 4, -4);
        assert_eq!(p1 + v2, point3i(14, 2, 1));

        let p1 = point2f(1.0, -2.0);
        let v2 = vec2f(12.0, 3.0);
        assert_eq!(p1 + v2, point2f(13.0, 1.0));

        let p1 = point2i(3, -2);
        let v2 = vec2i(11, 4);
        assert_eq!(p1 + v2, point2i(14, 2));
    }

    #[test]
    fn point_sub() {
        let p1 = point3f(1.0, -2.0, 5.0);
        let p2 = point3f(12.0, 3.0, -1.0);
        assert_eq!(p1 - p2, vec3f(-11.0, -5.0, 6.0));

        let p1 = point3i(1, -2, 5);
        let p2 = point3i(12, 3, -1);
        assert_eq!(p1 - p2, vec3i(-11, -5, 6));

        let p1 = point2f(1.0, -2.0);
        let p2 = point2f(12.0, 3.0);
        assert_eq!(p1 - p2, vec2f(-11.0, -5.0));

        let p1 = point2i(1, -2);
        let p2 = point2i(12, 3);
        assert_eq!(p1 - p2, vec2i(-11, -5));
    }

    #[test]
    fn point_mul_by_scalar() {
        let p1 = point3f(2.0, 3.0, 4.0);
        assert_eq!(p1 * 2.0, point3f(4.0, 6.0, 8.0));

        let p1 = point3i(2, 3, 4);
        assert_eq!(p1 * 2, point3i(4, 6, 8));

        let p1 = point2f(2.0, 3.0);
        assert_eq!(p1 * 2.0, point2f(4.0, 6.0));

        let p1 = point2i(2, 3);
        assert_eq!(p1 * 2, point2i(4, 6));
    }

    #[test]
    fn point_dist() {
        let p1 = point3f(2.0, 3.0, 4.0);
        let p2 = point3f(1.0, -2.0, 5.0);
        assert_eq!(Point3f::dist(p1, p2), 5.1961524);

        let p1 = point3i(2, 3, 4);
        let p2 = point3i(1, -2, 5);
        assert_eq!(Point3i::dist(p1, p2), 5.1961524);

        let p1 = point2f(10.2, 3.0);
        let p2 = point2f(1.0, -4.0);
        assert_eq!(Point2f::dist(p1, p2), 11.5602768);

        let p1 = point2i(3, -4);
        let p2 = point2i(1, -9);
        assert_eq!(Point2i::dist(p1, p2), 5.3851648);
    }

    #[test]
    fn point_cast() {
        let p1 = point3f(2.6, 3.3, -4.7);
        let p2: Point3i = p1.cast();
        assert_eq!(p2, point3i(2, 3, -4));

        let p1 = point2i(2, -3);
        let p2: Point2f = p1.cast();
        assert_eq!(p2, point2f(2.0, -3.0));
    }

    #[test]
    fn point_lerp() {
        let p1 = point3f(2.0, 3.0, -4.0);
        let p2 = point3f(4.0, 9.0, 2.0);
        let p3 = Point3f::lerp(0.6, p1, p2);
        assert_approx_eq!(p3.x, 3.2);
        assert_approx_eq!(p3.y, 6.6);
        assert_approx_eq!(p3.z, -0.4);

        let p1 = point2f(2.0, 3.0);
        let p2 = point2f(4.0, 9.0);
        let p3 = Point2f::lerp(0.6, p1, p2);
        assert_approx_eq!(p3.x, 3.2);
        assert_approx_eq!(p3.y, 6.6);

        let p1 = point3i(2, 3, -4);
        let p2 = point3i(4, 9, 2);
        let p3 = Point3i::lerp(0.6, p1, p2);
        assert_eq!(p3, point3i(3, 6, 0));

        let p1 = point2i(2, 3);
        let p2 = point2i(4, 9);
        let p3 = Point2i::lerp(0.6, p1, p2);
        assert_eq!(p3, point2i(3, 6));
    }

    #[test]
    fn point_max() {
        let p1 = point3f(2.0, 11.0, -4.0);
        let p2 = point3f(4.0, 9.0, -6.0);
        let p3 = Point3f::max(p1, p2);
        assert_eq!(p3, point3f(4.0, 11.0, -4.0));

        let p1 = point2f(0.0, 3.0);
        let p2 = point2f(0.0, 9.0);
        let p3 = Point2f::max(p1, p2);
        assert_eq!(p3, point2f(0.0, 9.0));

        let p1 = point3i(24, 3, -4);
        let p2 = point3i(4, 11, 2);
        let p3 = Point3i::max(p1, p2);
        assert_eq!(p3, point3i(24, 11, 2));

        let p1 = point2i(2, 3);
        let p2 = point2i(7, -9);
        let p3 = Point2i::max(p1, p2);
        assert_eq!(p3, point2i(7, 3));
    }

    #[test]
    fn point_floor_ceil_abs() {
        let p1 = point3f(-2.3, 11.7, -4.4);
        assert_eq!(p1.floor(), point3f(-3.0, 11.0, -5.0));
        assert_eq!(p1.ceil(), point3f(-2.0, 12.0, -4.0));
        assert_eq!(p1.abs(), point3f(2.3, 11.7, 4.4));

        let p1 = point2f(-2.3, 11.7);
        assert_eq!(p1.floor(), point2f(-3.0, 11.0));
        assert_eq!(p1.ceil(), point2f(-2.0, 12.0));
        assert_eq!(p1.abs(), point2f(2.3, 11.7));
    }
}
