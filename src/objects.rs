use crate::ray::{Collider, Collision, Ray};
use crate::vec3::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub r: f32,
}

impl Collider for Sphere {
    fn get_collision(&self, r: Ray) -> Option<Collision> {
        let oc = r.origin() - self.center;
        let a = Vec3::dot(r.dir(), r.dir());
        let b = 2.0 * Vec3::dot(oc, r.dir());
        let c = Vec3::dot(oc, oc) - self.r * self.r;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant > 0.0 {
            let t = (-b - discriminant.sqrt()) / (2.0 * a);
            let point = r.position_at_time(t);
            let normal = Vec3::normalize(point - self.center);
            Some(Collision { point, normal, t })
        } else {
            None
        }
    }
}
