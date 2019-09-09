use super::point::{Point2, Point2f, Point3, Point3f};
use super::vector::{Vec2, Vec3};
use super::{FloatRT, Scalar};
use crate::utils::math::lerp;
use std::ops::Index;

// Convenience aliases
pub type Bounds3f = Bounds3<FloatRT>;
pub type Bounds3i = Bounds3<u32>;
pub type Bounds2f = Bounds2<FloatRT>;
pub type Bounds2i = Bounds2<u32>;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bounds3<T> {
    pub p_min: Point3<T>,
    pub p_max: Point3<T>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Bounds2<T> {
    p_min: Point2<T>,
    p_max: Point2<T>,
}

impl<T: Scalar> Bounds3<T> {
    pub fn new(p1: Point3<T>, p2: Point3<T>) -> Self {
        Bounds3 {
            p_min: Point3::min(p1, p2),
            p_max: Point3::max(p1, p2),
        }
    }

    pub fn from_point(p: Point3<T>) -> Self {
        Bounds3 { p_min: p, p_max: p }
    }

    pub fn corner(&self, corner: u8) -> Point3<T> {
        assert!(corner >= 0 && corner < 8);
        let x = self[(corner & 1)].x;
        let y = self[if (corner & 2) > 0 { 1 } else { 0 }].y;
        let z = self[if (corner & 4) > 0 { 1 } else { 0 }].z;
        Point3::new(x, y, z)
    }

    pub fn diagonal(self) -> Vec3<T> {
        self.p_max - self.p_min
    }

    pub fn surface_area(self) -> T {
        let d = self.diagonal();
        T::from(2).unwrap() * (d.x * d.y + d.y * d.z + d.x * d.z)
    }

    pub fn volume(self) -> T {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    pub fn maximum_extent(self) -> usize {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }

    pub fn lerp(self, t: Point3f) -> Point3<T> {
        let p1 = Point3::lift_to_float(self.p_min);
        let p2 = Point3::lift_to_float(self.p_max);
        Point3::new(
            lerp(t.x, p1.x, p2.x),
            lerp(t.y, p1.y, p2.y),
            lerp(t.z, p1.z, p2.z),
        )
        .cast()
    }

    pub fn offset(self, p: Point3<T>) -> Vec3<T> {
        let mut o = p - self.p_min;
        if self.p_max.x > self.p_min.x {
            o.x = o.x / (self.p_max.x - self.p_min.x);
        }
        if self.p_max.y > self.p_min.y {
            o.y = o.y / (self.p_max.y - self.p_min.y);
        }
        if self.p_max.z > self.p_min.z {
            o.z = o.z / (self.p_max.z - self.p_min.z);
        }
        return o;
    }

    pub fn bounding_sphere(self) -> (Point3<T>, FloatRT) {
        let center = (self.p_min + self.p_max) / T::from(2).unwrap();
        let radius = if Self::inside(center, self) {
            Point3::dist(center, self.p_max)
        } else {
            0.0
        };
        (center, radius)
    }

    pub fn union_point(b: Self, p: Point3<T>) -> Self {
        Bounds3 {
            p_min: Point3::min(b.p_min, p),
            p_max: Point3::max(b.p_max, p),
        }
    }

    pub fn union(b1: Self, b2: Self) -> Self {
        Bounds3 {
            p_min: Point3::min(b1.p_min, b2.p_min),
            p_max: Point3::max(b1.p_max, b2.p_max),
        }
    }

    pub fn intersect(b1: Self, b2: Self) -> Self {
        Bounds3 {
            p_min: Point3::max(b1.p_min, b2.p_min),
            p_max: Point3::min(b1.p_max, b2.p_max),
        }
    }

    pub fn overlaps(b1: Self, b2: Self) -> bool {
        let x_overlap = (b1.p_max.x >= b2.p_min.x) && (b1.p_min.x <= b2.p_max.x);
        let y_overlap = (b1.p_max.y >= b2.p_min.y) && (b1.p_min.y <= b2.p_max.y);
        let z_overlap = (b1.p_max.z >= b2.p_min.z) && (b1.p_min.z <= b2.p_max.z);
        x_overlap && y_overlap && z_overlap
    }

    pub fn inside(p: Point3<T>, b: Self) -> bool {
        (p.x >= b.p_min.x && p.x <= b.p_max.x)
            && (p.y >= b.p_min.y && p.y <= b.p_max.y)
            && (p.z >= b.p_min.z && p.z <= b.p_max.z)
    }

    // Doesn't consider points on the upper boundary to be inside the bounds
    pub fn inside_exclusive(p: Point3<T>, b: Self) -> bool {
        (p.x >= b.p_min.x && p.x < b.p_max.x)
            && (p.y >= b.p_min.y && p.y < b.p_max.y)
            && (p.z >= b.p_min.z && p.z < b.p_max.z)
    }

    pub fn expand(b: Self, delta: T) -> Self {
        let v = Vec3::new(delta, delta, delta);
        Bounds3::new(b.p_min - v, b.p_max + v)
    }
}

impl<T: Scalar> Index<u8> for Bounds3<T> {
    type Output = Point3<T>;
    fn index(&self, index: u8) -> &Self::Output {
        match index {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => panic!("Bad index for Bounds3!"),
        }
    }
}

impl<T: Scalar> Bounds2<T> {
    pub fn new(p1: Point2<T>, p2: Point2<T>) -> Self {
        Self {
            p_min: Point2::min(p1, p2),
            p_max: Point2::max(p1, p2),
        }
    }

    pub fn from_point(p: Point2<T>) -> Self {
        Self { p_min: p, p_max: p }
    }

    pub fn diagonal(self) -> Vec2<T> {
        self.p_max - self.p_min
    }

    pub fn perimeter(self) -> T {
        let d = self.diagonal();
        (d.x + d.x + d.y + d.y)
    }

    pub fn area(self) -> T {
        let d = self.diagonal();
        d.x * d.y
    }

    pub fn maximum_extent(self) -> usize {
        let d = self.diagonal();
        if d.x > d.y {
            0
        } else {
            1
        }
    }

    pub fn lerp(self, t: Point2f) -> Point2<T> {
        let p1 = Point2::lift_to_float(self.p_min);
        let p2 = Point2::lift_to_float(self.p_max);
        Point2::new(lerp(t.x, p1.x, p2.x), lerp(t.y, p1.y, p2.y)).cast()
    }

    pub fn offset(self, p: Point2<T>) -> Vec2<T> {
        let mut o = p - self.p_min;
        if self.p_max.x > self.p_min.x {
            o.x = o.x / (self.p_max.x - self.p_min.x);
        }
        if self.p_max.y > self.p_min.y {
            o.y = o.y / (self.p_max.y - self.p_min.y);
        }
        return o;
    }

    pub fn bounding_circle(self) -> (Point2<T>, FloatRT) {
        let center = (self.p_min + self.p_max) / T::from(2).unwrap();
        let radius = if Self::inside(center, self) {
            Point2::dist(center, self.p_max)
        } else {
            0.0
        };
        (center, radius)
    }

    pub fn union_point(b: Self, p: Point2<T>) -> Self {
        Self {
            p_min: Point2::min(b.p_min, p),
            p_max: Point2::max(b.p_max, p),
        }
    }

    pub fn union(b1: Self, b2: Self) -> Self {
        Self {
            p_min: Point2::min(b1.p_min, b2.p_min),
            p_max: Point2::max(b1.p_max, b2.p_max),
        }
    }

    pub fn intersect(b1: Self, b2: Self) -> Self {
        Self {
            p_min: Point2::max(b1.p_min, b2.p_min),
            p_max: Point2::min(b1.p_max, b2.p_max),
        }
    }

    pub fn overlaps(b1: Self, b2: Self) -> bool {
        let x_overlap = (b1.p_max.x >= b2.p_min.x) && (b1.p_min.x <= b2.p_max.x);
        let y_overlap = (b1.p_max.y >= b2.p_min.y) && (b1.p_min.y <= b2.p_max.y);
        x_overlap && y_overlap
    }

    pub fn inside(p: Point2<T>, b: Self) -> bool {
        (p.x >= b.p_min.x && p.x <= b.p_max.x) && (p.y >= b.p_min.y && p.y <= b.p_max.y)
    }

    // Doesn't consider points on the upper boundary to be inside the bounds
    pub fn inside_exclusive(p: Point2<T>, b: Self) -> bool {
        (p.x >= b.p_min.x && p.x < b.p_max.x) && (p.y >= b.p_min.y && p.y < b.p_max.y)
    }

    pub fn expand(b: Self, delta: T) -> Self {
        let v = Vec2::new(delta, delta);
        Self::new(b.p_min - v, b.p_max + v)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::{AbsDiffEq, RelativeEq, UlpsEq};
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn bounds_from_point() {
        let p = Point3f::new(1.0, 1.2, 1.4);
        let b = Bounds3f::from_point(p);
        assert_eq!(b.p_max, p);
        assert_eq!(b.p_min, p);
    }

    #[test]
    fn bounds_index() {
        let p1 = Point3f::new(-1.0, 0.0, -2.0);
        let p2 = Point3f::new(2.0, 3.0, 2.0);
        let b1 = Bounds3f::new(p2, p1);
        assert_eq!(b1[0], p1);
        assert_eq!(b1[1], p2);
    }

    #[test]
    fn bounds_corners() {
        let p1 = Point3f::new(-1.0, 0.0, -2.0);
        let p2 = Point3f::new(2.0, 3.0, 2.0);
        let b1 = Bounds3f::new(p2, p1);
        assert_eq!(b1.corner(0), p1);
        assert_eq!(b1.corner(7), p2);
        assert_eq!(b1.corner(1), Point3f::new(2.0, 0.0, -2.0));
        assert_eq!(b1.corner(2), Point3f::new(-1.0, 3.0, -2.0));
        assert_eq!(b1.corner(3), Point3f::new(2.0, 3.0, -2.0));
        assert_eq!(b1.corner(4), Point3f::new(-1.0, 0.0, 2.0));
        assert_eq!(b1.corner(5), Point3f::new(2.0, 0.0, 2.0));
        assert_eq!(b1.corner(6), Point3f::new(-1.0, 3.0, 2.0));
    }

    #[test]
    fn bounds_diagonal() {
        let p1 = Point3f::new(-1.0, 0.0, -2.0);
        let p2 = Point3f::new(2.0, 3.0, 2.0);
        let b1 = Bounds3f::new(p2, p1);
        // switched parameter order
        let b2 = Bounds3f::new(p1, p2);
        assert_eq!(b1.diagonal(), Vec3::new(3.0, 3.0, 4.0));
        assert_eq!(b2.diagonal(), Vec3::new(3.0, 3.0, 4.0));

        let p1 = Point2f::new(-1.0, 0.0);
        let p2 = Point2f::new(2.0, 3.0);
        let b1 = Bounds2f::new(p2, p1);
        // switched parameter order
        let b2 = Bounds2f::new(p1, p2);
        assert_eq!(b1.diagonal(), Vec2::new(3.0, 3.0));
        assert_eq!(b2.diagonal(), Vec2::new(3.0, 3.0));
    }

    #[test]
    fn bounds_surface_area() {
        let p1 = Point3f::new(0.0, 0.0, 0.0);
        let p2 = Point3f::new(2.0, 3.0, 2.0);
        let b1 = Bounds3f::new(p2, p1);
        assert_eq!(b1.surface_area(), 32.0);
    }

    #[test]
    fn bounds_volume() {
        let p1 = Point3f::new(0.0, 0.0, 0.0);
        let p2 = Point3f::new(2.0, 3.0, 2.0);
        let b1 = Bounds3f::new(p2, p1);
        assert_eq!(b1.volume(), 12.0);
    }

    #[test]
    fn bounds_perimeter() {
        let p1 = Point2f::new(-1.0, 0.0);
        let p2 = Point2f::new(2.0, 3.0);
        let b1 = Bounds2f::new(p2, p1);
        assert_eq!(b1.perimeter(), 12.0);
    }

    #[test]
    fn bounds_area() {
        let p1 = Point2f::new(-1.0, 0.0);
        let p2 = Point2f::new(2.0, 3.0);
        let b1 = Bounds2f::new(p2, p1);
        assert_eq!(b1.area(), 9.0);
    }

    #[test]
    fn bounds_maximum_extent() {
        let p1 = Point3f::new(-1.0, 0.0, 1.0);
        let p2 = Point3f::new(2.0, 3.0, 4.5);
        let b1 = Bounds3f::new(p2, p1);
        assert_eq!(b1.maximum_extent(), 2);
    }

    #[test]
    fn bounds_lerp() {
        let t = Point3f::new(0.5, 0.2, 0.6);
        let p1 = Point3f::new(0.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b = Bounds3f::new(p1, p2);
        let p_lerp = b.lerp(t);
        assert_approx_eq!(p_lerp.x, 2.0);
        assert_approx_eq!(p_lerp.y, 2.0);
        assert_approx_eq!(p_lerp.z, 2.0);
    }

    #[test]
    fn bounds_offset() {
        let p1 = Point3f::new(0.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b = Bounds3f::new(p1, p2);
        let p_offset = b.offset(Point3f::new(2.0, 2.0, 2.0));
        assert_approx_eq!(p_offset.x, 0.5);
        assert_approx_eq!(p_offset.y, 0.2);
        assert_approx_eq!(p_offset.z, 0.6);
    }

    #[test]
    fn bounds_bounding_sphere() {
        let p1 = Point3f::new(2.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b = Bounds3f::new(p1, p2);
        assert_eq!(
            b.bounding_sphere(),
            (Point3f::new(3.0, 5.0, 0.0), 11.224972)
        )
    }
    #[test]
    fn bounds_union_point() {
        let p1 = Point3f::new(2.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b = Bounds3f::new(p1, p2);
        let p3 = Point3f::new(5.0, 9.0, 11.0);
        let b2 = Bounds3f::union_point(b, p3);
        assert_eq!(b2.p_max, Point3f::new(5.0, 10.0, 11.0));
        assert_eq!(b2.p_min, p1);
    }

    #[test]
    fn bounds_union() {
        let p1 = Point3f::new(2.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b1 = Bounds3f::new(p1, p2);
        let p3 = Point3f::new(5.0, 9.0, 11.0);
        let b2 = Bounds3f::new(p2, p3);
        let b3 = Bounds3f::union(b1, b2);
        assert_eq!(b3.p_max, Point3f::new(5.0, 10.0, 11.0));
        assert_eq!(b3.p_min, p1);
    }

    #[test]
    fn bounds_intersect() {
        let p1 = Point3f::new(2.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b1 = Bounds3f::new(p1, p2);
        let p3 = Point3f::new(5.0, 9.0, 11.0);
        let b2 = Bounds3f::new(p2, p3);
        let b3 = Bounds3f::intersect(b1, b2);
        assert_eq!(b3.p_max, Point3f::new(4.0, 10.0, 10.0));
        assert_eq!(b3.p_min, Point3f::new(4.0, 9.0, 10.0));
    }

    #[test]
    fn bounds_overlaps() {
        let p1 = Point3f::new(2.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b1 = Bounds3f::new(p1, p2);
        let p3 = Point3f::new(5.0, 9.0, 11.0);
        let b2 = Bounds3f::new(p2, p3);
        assert!(Bounds3f::overlaps(b1, b2));
        let p4 = Point3f::new(6.0, 12.0, 12.0);
        let b3 = Bounds3f::new(p3, p4);
        assert!(!Bounds3f::overlaps(b1, b3));
    }

    #[test]
    fn bounds_inside() {
        let p1 = Point3f::new(2.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b1 = Bounds3f::new(p1, p2);
        assert!(Bounds3f::inside(Point3f::new(3.0, 4.0, 0.0), b1));
        assert!(!Bounds3f::inside(Point3f::new(-3.0, -4.0, 0.0), b1));
        assert!(Bounds3f::inside(Point3f::new(4.0, 10.0, 10.0), b1));
    }

    #[test]
    fn bounds_inside_exclusive() {
        let p1 = Point3f::new(2.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b1 = Bounds3f::new(p1, p2);
        assert!(Bounds3f::inside_exclusive(Point3f::new(3.0, 4.0, 0.0), b1));
        assert!(!Bounds3f::inside_exclusive(
            Point3f::new(4.0, 10.0, 10.0),
            b1
        ));
    }

    #[test]
    fn bounds_expand() {
        let p1 = Point3f::new(2.0, 0.0, -10.0);
        let p2 = Point3f::new(4.0, 10.0, 10.0);
        let b1 = Bounds3f::new(p1, p2);
        let b2 = Bounds3f::expand(b1, 2.0);
        assert_eq!(b2.p_max, Point3f::new(6.0, 12.0, 12.0));
        assert_eq!(b2.p_min, Point3f::new(0.0, -2.0, -12.0));
    }

    /// Approximate equality implementations for testing purposes
    impl<T: AbsDiffEq> AbsDiffEq for Bounds3<T>
    where
        T::Epsilon: Copy,
    {
        type Epsilon = T::Epsilon;
        fn default_epsilon() -> T::Epsilon {
            T::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            Point3::abs_diff_eq(&self.p_min, &other.p_min, epsilon)
                && Point3::abs_diff_eq(&self.p_max, &other.p_max, epsilon)
        }
    }

    impl<T: RelativeEq> RelativeEq for Bounds3<T>
    where
        T::Epsilon: Copy,
    {
        fn default_max_relative() -> T::Epsilon {
            T::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: T::Epsilon,
        ) -> bool {
            Point3::relative_eq(&self.p_min, &other.p_min, epsilon, max_relative)
                && Point3::relative_eq(&self.p_max, &other.p_max, epsilon, max_relative)
        }
    }

    impl<T: UlpsEq> UlpsEq for Bounds3<T>
    where
        T::Epsilon: Copy,
    {
        fn default_max_ulps() -> u32 {
            T::default_max_ulps()
        }

        fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
            Point3::ulps_eq(&self.p_min, &other.p_min, epsilon, max_ulps)
                && Point3::ulps_eq(&self.p_max, &other.p_max, epsilon, max_ulps)
        }
    }

    impl<T: AbsDiffEq> AbsDiffEq for Bounds2<T>
    where
        T::Epsilon: Copy,
    {
        type Epsilon = T::Epsilon;
        fn default_epsilon() -> T::Epsilon {
            T::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            Point2::abs_diff_eq(&self.p_min, &other.p_min, epsilon)
                && Point2::abs_diff_eq(&self.p_max, &other.p_max, epsilon)
        }
    }

    impl<T: RelativeEq> RelativeEq for Bounds2<T>
    where
        T::Epsilon: Copy,
    {
        fn default_max_relative() -> T::Epsilon {
            T::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: T::Epsilon,
        ) -> bool {
            Point2::relative_eq(&self.p_min, &other.p_min, epsilon, max_relative)
                && Point2::relative_eq(&self.p_max, &other.p_max, epsilon, max_relative)
        }
    }

    impl<T: UlpsEq> UlpsEq for Bounds2<T>
    where
        T::Epsilon: Copy,
    {
        fn default_max_ulps() -> u32 {
            T::default_max_ulps()
        }

        fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
            Point2::ulps_eq(&self.p_min, &other.p_min, epsilon, max_ulps)
                && Point2::ulps_eq(&self.p_max, &other.p_max, epsilon, max_ulps)
        }
    }
}
