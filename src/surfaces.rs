use crate::ray::{Collider, Collision, Ray};
use crate::vec3::Vec3;

//TODO: This is kind of ugly boilerplate-y right now but apparently this is how to do it
pub enum Surface {
    Sphere(Sphere),
    Plane(Plane),
    Parallelogram(Parallelogram),
}

impl Collider for Surface {
    fn get_collision(&self, r: Ray, t_min: f32, t_max: f32) -> Option<Collision> {
        match self {
            Surface::Sphere(s) => s.get_collision(r, t_min, t_max),
            Surface::Plane(p) => p.get_collision(r, t_min, t_max),
            Surface::Parallelogram(p) => p.get_collision(r, t_min, t_max),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub r: f32,
}

impl Collider for Sphere {
    fn get_collision(&self, r: Ray, t_min: f32, t_max: f32) -> Option<Collision> {
        let oc = r.origin() - self.center;
        let a = Vec3::dot(r.dir(), r.dir());
        let b = 2.0 * Vec3::dot(oc, r.dir());
        let c = Vec3::dot(oc, oc) - self.r * self.r;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0. {
            None
        } else {
            let t = (-b - discriminant.sqrt()) / (2.0 * a);
            if t < t_min || t > t_max {
                None
            } else {
                let point = r.position_at_time(t);
                let normal = Vec3::normalize(point - self.center);
                Some(Collision { point, normal, t })
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    origin: Vec3,
    normal: Vec3,
}

impl Collider for Plane {
    fn get_collision(&self, r: Ray, t_min: f32, t_max: f32) -> Option<Collision> {
        let t = Vec3::dot(self.normal, self.origin - r.origin()) / Vec3::dot(self.normal, r.dir());
        if t < t_min || t > t_max {
            None
        } else {
            let point = r.position_at_time(t);
            Some(Collision {
                point,
                normal: self.normal,
                t,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Parallelogram {
    origin: Vec3,
    normal: Vec3,
    u: Vec3,
    v: Vec3,
    u_len: f32,
    v_len: f32,
    u_dot_u: f32,
    u_dot_v: f32,
    v_dot_v: f32,
    det: f32,
}

impl Parallelogram {
    pub fn new(origin: Vec3, u: Vec3, v: Vec3) -> Parallelogram {
        let normal = Vec3::normalize(Vec3::cross(u, v));
        let (u_len, v_len) = (Vec3::len(u), Vec3::len(v));
        let (u_dot_u, u_dot_v, v_dot_v) = (Vec3::dot(u, u), Vec3::dot(u, v), Vec3::dot(v, v));
        let det = u_dot_u * v_dot_v - u_dot_v * u_dot_v;
        Parallelogram {
            origin,
            normal,
            u,
            v,
            u_len,
            v_len,
            u_dot_u,
            u_dot_v,
            v_dot_v,
            det,
        }
    }
}

impl Collider for Parallelogram {
    fn get_collision(&self, r: Ray, t_min: f32, t_max: f32) -> Option<Collision> {
        let t = Vec3::dot(self.normal, self.origin - r.origin()) / Vec3::dot(self.normal, r.dir());
        if t < t_min || t > t_max {
            return None;
        }
        let point = r.position_at_time(t);
        let rhs = point - self.origin;
        let (u_dot_rhs, v_dot_rhs) = (Vec3::dot(self.u, rhs), Vec3::dot(self.v, rhs));
        let w1 = (self.v_dot_v * u_dot_rhs - self.u_dot_v * v_dot_rhs) / self.det;
        let w2 = (-self.u_dot_v * u_dot_rhs + self.u_dot_u * v_dot_rhs) / self.det;
        if 0. <= w1 && w1 <= 1. && 0. <= w2 && w2 <= 1. {
            Some(Collision {
                point,
                normal: self.normal,
                t,
            })
        } else {
            None
        }
    }
}

// pub struct AABB {
//     center: Vec3,
//     w: f32,
//     h: f32,
//     d: f32,
// }

// impl AABB {
//     pub fn new(center: Vec3, w: f32, h: f32, d: f32) -> AABB {
//         AABB { center, w, h, d }
//     }

//     pub fn cube(center: Vec3, l: f32) -> AABB {
//         AABB {center, w: l, h: l, d: l}
//     }
// }

// impl Collider for AABB {
//     fn get_collision(&self, r: Ray) -> Option<Collision> {

//     }
// }
