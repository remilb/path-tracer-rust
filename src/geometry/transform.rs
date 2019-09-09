//TODO: Rigorously deal with round off error in transforms
use super::bounds::{Bounds2, Bounds3f};
use super::matrix::Matrix4X4;
use super::normal::Normal3;
use super::point::{Point3, Point3f};
use super::ray::Ray;
use super::vector::{Vec3, Vec3f};
use super::Cross;
use super::{FloatRT, Scalar};
use approx::relative_ne;
use num_traits::NumCast;
use std::ops::Mul;

#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    m: Matrix4X4,
    m_inv: Matrix4X4,
}

pub trait Transformer<T: Copy> {
    fn apply(&self, target: T) -> T;
}

impl Transform {
    /// New Transform from provided mat. Inverse is computed.
    pub fn new(mat: [[FloatRT; 4]; 4]) -> Self {
        let m = Matrix4X4::from(mat);
        let m_inv = m.inverse();
        Self { m, m_inv }
    }

    /// New Transform from matrix m with explicitly provided inverse m_inv
    pub fn with_inverse(m: Matrix4X4, m_inv: Matrix4X4) -> Self {
        Self { m, m_inv }
    }

    pub fn translate(delta: Vec3f) -> Self {
        let m = Matrix4X4::from([
            [1.0, 0.0, 0.0, delta.x],
            [0.0, 1.0, 0.0, delta.y],
            [0.0, 0.0, 1.0, delta.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = Matrix4X4::from([
            [1.0, 0.0, 0.0, -delta.x],
            [0.0, 1.0, 0.0, -delta.y],
            [0.0, 0.0, 1.0, -delta.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        Self { m, m_inv }
    }

    pub fn scale(factor: Vec3f) -> Self {
        let m = Matrix4X4::from([
            [factor.x, 0.0, 0.0, 0.0],
            [0.0, factor.y, 0.0, 0.0],
            [0.0, 0.0, factor.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = Matrix4X4::from([
            [1.0 / factor.x, 0.0, 0.0, 0.0],
            [0.0, 1.0 / factor.y, 0.0, 0.0],
            [0.0, 0.0, 1.0 / factor.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        Self { m, m_inv }
    }

    pub fn rotate_x(deg: FloatRT) -> Self {
        let rads = deg.to_radians();
        let cos_theta = rads.cos();
        let sin_theta = rads.sin();
        let m = Matrix4X4::from([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta, 0.0],
            [0.0, sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = m.transpose();
        Self { m, m_inv }
    }

    pub fn rotate_y(deg: FloatRT) -> Self {
        let rads = deg.to_radians();
        let cos_theta = rads.cos();
        let sin_theta = rads.sin();
        let m = Matrix4X4::from([
            [cos_theta, 0.0, sin_theta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = m.transpose();
        Self { m, m_inv }
    }

    pub fn rotate_z(deg: FloatRT) -> Self {
        let rads = deg.to_radians();
        let cos_theta = rads.cos();
        let sin_theta = rads.sin();
        let m = Matrix4X4::from([
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta, cos_theta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = m.transpose();
        Self { m, m_inv }
    }

    pub fn rotate(deg: FloatRT, axis: &Vec3f) -> Self {
        let a = axis.normalize();
        let cos_theta = deg.to_radians().cos();
        let sin_theta = deg.to_radians().sin();
        let mut m = Matrix4X4::default();
        // First basis vector rotation
        m.m[0][0] = a.x * a.x + (1.0 - a.x * a.x) * cos_theta;
        m.m[0][1] = a.x * a.y * (1.0 - cos_theta) - a.z * sin_theta;
        m.m[0][2] = a.x * a.z * (1.0 - cos_theta) + a.y * sin_theta;
        m.m[0][3] = 0.0;
        // Second basis vector rotation
        m.m[1][0] = a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta;
        m.m[1][1] = a.y * a.y + (1.0 - a.y * a.y) * cos_theta;
        m.m[1][2] = a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta;
        m.m[1][3] = 0.0;
        // Third basis vector rotation
        m.m[2][0] = a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta;
        m.m[2][1] = a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta;
        m.m[2][2] = a.z * a.z + (1.0 - a.z * a.z) * cos_theta;
        m.m[2][3] = 0.0;

        let m_inv = m.transpose();

        Self { m, m_inv }
    }

    pub fn look_at(pos: Point3f, subject: Point3f, up: Vec3f) -> Self {
        let w = (subject - pos).normalize();
        let u = Vec3f::cross(up.normalize(), w).normalize();
        let v = Vec3f::cross(w, u);
        let m = Matrix4X4::from([
            [u.x, v.x, w.x, pos.x],
            [u.y, v.y, w.y, pos.y],
            [u.z, v.z, w.z, pos.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = m.inverse();
        Self { m, m_inv }
    }

    /// Returns a new Transform that is the inverse of self
    pub fn inverse(&self) -> Self {
        Self {
            m: self.m_inv.clone(),
            m_inv: self.m.clone(),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.m.m[0][0] == 1.0
            && self.m.m[0][1] == 0.0
            && self.m.m[0][2] == 0.0
            && self.m.m[0][3] == 0.0
            && self.m.m[1][0] == 0.0
            && self.m.m[1][1] == 1.0
            && self.m.m[1][2] == 0.0
            && self.m.m[1][3] == 0.0
            && self.m.m[2][0] == 0.0
            && self.m.m[2][1] == 0.0
            && self.m.m[2][2] == 1.0
            && self.m.m[2][3] == 0.0
            && self.m.m[3][0] == 0.0
            && self.m.m[3][1] == 0.0
            && self.m.m[3][2] == 0.0
            && self.m.m[3][3] == 1.0
    }

    pub fn has_scale(&self) -> bool {
        let u = self.apply(Vec3f::new(1.0, 0.0, 0.0));
        let v = self.apply(Vec3f::new(0.0, 1.0, 0.0));
        let w = self.apply(Vec3f::new(0.0, 0.0, 1.0));

        relative_ne!(u.length_squared(), 1.0)
            || relative_ne!(v.length_squared(), 1.0)
            || relative_ne!(w.length_squared(), 1.0)
    }

    pub fn swaps_handedness(&self) -> bool {
        self.m.determinant() < 0.0
    }
}

/// Default transform is the identity transformation
impl Default for Transform {
    fn default() -> Self {
        Self {
            m: Matrix4X4::identity(),
            m_inv: Matrix4X4::identity(),
        }
    }
}

impl Mul for Transform {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::with_inverse(self.m * rhs.m, rhs.m_inv * self.m_inv)
    }
}

impl<T: Scalar> Transformer<Vec3<T>> for Transform {
    fn apply(&self, v: Vec3<T>) -> Vec3<T> {
        let (x, y, z): (FloatRT, FloatRT, FloatRT) = (
            NumCast::from(v.x).unwrap(),
            NumCast::from(v.y).unwrap(),
            NumCast::from(v.z).unwrap(),
        );
        let xv =
            NumCast::from(self.m.m[0][0] * x + self.m.m[0][1] * y + self.m.m[0][2] * z).unwrap();
        let yv =
            NumCast::from(self.m.m[1][0] * x + self.m.m[1][1] * y + self.m.m[1][2] * z).unwrap();
        let zv =
            NumCast::from(self.m.m[2][0] * x + self.m.m[2][1] * y + self.m.m[2][2] * z).unwrap();
        Vec3::new(xv, yv, zv)
    }
}

impl<T: Scalar> Transformer<Point3<T>> for Transform {
    fn apply(&self, v: Point3<T>) -> Point3<T> {
        let (x, y, z): (FloatRT, FloatRT, FloatRT) = (
            NumCast::from(v.x).unwrap(),
            NumCast::from(v.y).unwrap(),
            NumCast::from(v.z).unwrap(),
        );
        let xp = self.m.m[0][0] * x + self.m.m[0][1] * y + self.m.m[0][2] * z + self.m.m[0][3];
        let yp = self.m.m[1][0] * x + self.m.m[1][1] * y + self.m.m[1][2] * z + self.m.m[1][3];
        let zp = self.m.m[2][0] * x + self.m.m[2][1] * y + self.m.m[2][2] * z + self.m.m[2][3];
        let wp = self.m.m[3][0] * x + self.m.m[3][1] * y + self.m.m[3][2] * z + self.m.m[3][3];
        if wp == 1.0 {
            Point3::new(
                NumCast::from(xp).unwrap(),
                NumCast::from(yp).unwrap(),
                NumCast::from(zp).unwrap(),
            )
        } else {
            (Point3::new(xp, yp, zp) / wp).cast()
        }
    }
}

impl<T: Scalar> Transformer<Normal3<T>> for Transform {
    fn apply(&self, n: Normal3<T>) -> Normal3<T> {
        let (x, y, z): (FloatRT, FloatRT, FloatRT) = (
            NumCast::from(n.x).unwrap(),
            NumCast::from(n.y).unwrap(),
            NumCast::from(n.z).unwrap(),
        );

        let xn =
            NumCast::from(self.m_inv.m[0][0] * x + self.m_inv.m[1][0] * y + self.m_inv.m[2][0] * z)
                .unwrap();
        let yn =
            NumCast::from(self.m_inv.m[0][1] * x + self.m_inv.m[1][1] * y + self.m_inv.m[2][1] * z)
                .unwrap();
        let zn =
            NumCast::from(self.m_inv.m[0][2] * x + self.m_inv.m[1][2] * y + self.m_inv.m[2][2] * z)
                .unwrap();

        Normal3::new(xn, yn, zn)
    }
}

//TODO: Some round-off error management for transforms of Rays
impl Transformer<Ray> for Transform {
    fn apply(&self, r: Ray) -> Ray {
        let o = self.apply(r.o);
        let d = self.apply(r.d);
        Ray::new(o, d, r.tMax, r.time)
    }
}

impl Transformer<Bounds3f> for Transform {
    fn apply(&self, b: Bounds3f) -> Bounds3f {
        // TODO: Remove this after fleshing out more passing tests
        // let min_point = b.p_min;
        // let bbox = Bounds3f::from_point(self.apply(min_point));
        // let bbox = Bounds3f::union_point(
        //     bbox,
        //     self.apply(Point3f::new(b.p_max.x, b.p_min.y, b.p_min.z)),
        // );
        // let bbox = Bounds3f::union_point(
        //     bbox,
        //     self.apply(Point3f::new(b.p_min.x, b.p_max.y, b.p_min.z)),
        // );
        // let bbox = Bounds3f::union_point(
        //     bbox,
        //     self.apply(Point3f::new(b.p_min.x, b.p_min.y, b.p_max.z)),
        // );
        // let bbox = Bounds3f::union_point(
        //     bbox,
        //     self.apply(Point3f::new(b.p_min.x, b.p_max.y, b.p_max.z)),
        // );
        // let bbox = Bounds3f::union_point(
        //     bbox,
        //     self.apply(Point3f::new(b.p_max.x, b.p_max.y, b.p_min.z)),
        // );
        // let bbox = Bounds3f::union_point(
        //     bbox,
        //     self.apply(Point3f::new(b.p_max.x, b.p_min.y, b.p_max.z)),
        // );
        // let bbox = Bounds3f::union_point(
        //     bbox,
        //     self.apply(Point3f::new(b.p_max.x, b.p_max.y, b.p_max.z)),
        // );
        // bbox

        //Alternate implementation
        let xa = Vec3f::new(self.m.m[0][0], self.m.m[1][0], self.m.m[2][0]) * b.p_min.x;
        let xb = Vec3f::new(self.m.m[0][0], self.m.m[1][0], self.m.m[2][0]) * b.p_max.x;

        let ya = Vec3f::new(self.m.m[0][1], self.m.m[1][1], self.m.m[2][1]) * b.p_min.y;
        let yb = Vec3f::new(self.m.m[0][1], self.m.m[1][1], self.m.m[2][1]) * b.p_max.y;

        let za = Vec3f::new(self.m.m[0][2], self.m.m[1][2], self.m.m[2][2]) * b.p_min.z;
        let zb = Vec3f::new(self.m.m[0][2], self.m.m[1][2], self.m.m[2][2]) * b.p_max.z;

        let translation = Point3f::new(self.m.m[0][3], self.m.m[1][3], self.m.m[2][3]);

        Bounds3f::new(
            translation + Vec3f::min(xa, xb) + Vec3f::min(ya, yb) + Vec3f::min(za, zb),
            translation + Vec3f::max(xa, xb) + Vec3f::max(ya, yb) + Vec3f::max(za, zb),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::normal::Normal3f;
    use crate::geometry::ray::Ray;
    use approx::assert_relative_eq;

    #[test]
    fn translate() {
        let delta = Vec3f::new(1.0, 4.0, -2.0);
        let t = Transform::translate(delta);

        // Can't translate a vector
        assert_eq!(t.apply(delta), delta);

        // Can't translate normals
        let n = Normal3f::new(5.5, -600.0, 7.2);
        assert_relative_eq!(t.apply(n), n);

        // Translate points
        let p = Point3f::new(2.5, -3.0, 4.2);
        let result = Point3f::new(3.5, 1.0, 2.2);
        assert_relative_eq!(t.apply(p), result);

        // Translate rays
        let r = Ray::new(
            Point3f::new(1.2, 2.3, -10.0),
            Vec3f::new(2.0, -1.0, 3.0),
            0.0,
            0.0,
        );
        let result = Ray::new(
            Point3f::new(2.2, 6.3, -12.0),
            Vec3f::new(2.0, -1.0, 3.0),
            0.0,
            0.0,
        );
        assert_relative_eq!(t.apply(r), result);

        // Translate bounding box
        let b = Bounds3f::new(Point3f::new(0.0, 0.0, 0.0), Point3f::new(2.0, 5.0, 9.0));
        let result = Bounds3f::new(Point3f::new(1.0, 4.0, -2.0), Point3f::new(3.0, 9.0, 7.0));
        assert_relative_eq!(t.apply(b), result);
    }

    #[test]
    fn scale() {
        let t = Transform::scale(Vec3f::new(2.0, 3.0, 0.5));
        let t_collapse = Transform::scale(Vec3f::new(0.0, 0.0, 0.0));
        let t_flip_x = Transform::scale(Vec3f::new(-2.0, 3.0, 0.5));
        let t_flip_y = Transform::scale(Vec3f::new(2.0, -3.0, 0.5));
        let t_flip_z = Transform::scale(Vec3f::new(2.0, 3.0, -0.5));

        // Scale vectors
        let v = Vec3f::new(3.0, 4.0, 7.0);
        assert_relative_eq!(t.apply(v), Vec3f::new(6.0, 12.0, 3.5));
        assert_relative_eq!(t_collapse.apply(v), Vec3f::new(0.0, 0.0, 0.0));
        assert_relative_eq!(t_flip_x.apply(v), Vec3f::new(-6.0, 12.0, 3.5));

        // Scale points
        let p = Point3f::new(3.0, 4.0, 7.0);
        assert_relative_eq!(t.apply(p), Point3f::new(6.0, 12.0, 3.5));
        assert_relative_eq!(t_collapse.apply(p), Point3f::new(0.0, 0.0, 0.0));
        assert_relative_eq!(t_flip_y.apply(p), Point3f::new(6.0, -12.0, 3.5));

        // Scale normals, a bit trickier
        let n = Normal3f::new(1.0, 1.0, 1.0);
        assert_relative_eq!(t.apply(n), Normal3f::new(0.5, 0.3333333333, 2.0));
        //Can't meaninfully transform normal of a squashing scale as inverse vanishes
        //assert_relative_eq!(t_collapse.apply(n), Normal3f::new(0.0, 0.0, 0.0));
        assert_relative_eq!(t_flip_z.apply(n), Normal3f::new(0.5, 0.3333333333, -2.0));

        // Scale rays
        let r = Ray::new(
            Point3f::new(1.2, 2.3, -10.0),
            Vec3f::new(2.0, -1.0, 3.0),
            0.0,
            0.0,
        );
        assert_relative_eq!(
            t.apply(r),
            Ray::new(
                Point3f::new(2.4, 6.9, -5.0),
                Vec3f::new(4.0, -3.0, 1.5),
                0.0,
                0.0
            )
        );
        assert_relative_eq!(
            t_collapse.apply(r),
            Ray::new(
                Point3f::new(0.0, 0.0, 0.0),
                Vec3f::new(0.0, 0.0, 0.0),
                0.0,
                0.0
            )
        );
        assert_relative_eq!(
            t_flip_z.apply(r),
            Ray::new(
                Point3f::new(2.4, 6.9, 5.0),
                Vec3f::new(4.0, -3.0, -1.5),
                0.0,
                0.0
            )
        );

        // Scale bounding boxes
        let b = Bounds3f::new(Point3f::new(-1.0, -2.0, 0.0), Point3f::new(2.0, 5.0, 9.0));
        let result = Bounds3f::new(Point3f::new(-2.0, -6.0, 0.0), Point3f::new(4.0, 15.0, 4.5));
        assert_relative_eq!(t.apply(b), result);
        assert_eq!(
            t_collapse.apply(b),
            Bounds3f::new(Point3f::zeros(), Point3f::zeros())
        );
        let result = Bounds3f::new(Point3f::new(2.0, -6.0, 0.0), Point3f::new(-4.0, 15.0, 4.5));
        assert_relative_eq!(t_flip_x.apply(b), result);
    }

    #[test]
    fn has_scale() {
        let t_scale = Transform::scale(Vec3f::new(2.0, 3.0, 0.5));
        //let t_scale_flip = Transform::scale(Vec3f::new(-1.0, 1.0, 1.0));
        let t_identity = Transform::default();
        let t_rot = Transform::rotate_x(85.0);
        assert!(t_scale.has_scale());
        assert!(!t_identity.has_scale());
        assert!(!t_rot.has_scale());
        assert!((t_rot * t_scale).has_scale());
        //assert!(t_scale_flip.has_scale());
    }

    #[test]
    fn swaps_handedness() {
        let t_scale = Transform::scale(Vec3f::new(2.0, 3.0, 0.5));
        let t_flip = Transform::scale(Vec3f::new(-2.0, 3.0, 0.5));
        let t_flip_unit = Transform::scale(Vec3f::new(-1.0, 1.0, 1.0));
        let t_rot = Transform::rotate_x(85.0);
        assert!(!t_scale.swaps_handedness());
        assert!(t_flip.swaps_handedness());
        assert!(!t_rot.swaps_handedness());
        assert!((t_rot * t_flip).swaps_handedness());
        assert!(t_flip_unit.swaps_handedness());
    }
}
