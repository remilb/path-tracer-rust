use crate::vec3::Vec3;
use crate::materials::Material;
use std::option::Option;
//use assert_approx_eq::assert_approx_eq;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    origin: Vec3,
    dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Ray {
        Ray {
            origin,
            dir: Vec3::normalize(dir),
        }
    }

    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    pub fn dir(&self) -> Vec3 {
        self.dir
    }

    pub fn position_at_time(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

// Dumb struct for handling collisions
pub struct Collision {
    pub point: Vec3,
    pub normal: Vec3,
    pub t: f32,
    pub mat: Material,
}

// It will be the responsibility of the implementor to return the single relevant collision point if multiple are found
pub trait Collider {
    fn get_collision(&self, r: Ray, t_min: f32, t_max: f32) -> Option<Collision>;
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     #[test]
//     fn ray_position_at_time() {
//         let o = Vec3::new(0., 0., 0.);
//         let dir = Vec3::new(2.0, 1.0, 3.0);
//         let r = Ray::new(o, dir);
//         let p_at_t = r.position_at_time(4.0);
//         let true_p_at_t = Vec3::new(0.57142857, 0.28571428, 0.85714285);
//         assert_approx_eq!(p_at_t, true_p_at_t);
//     }
// }
