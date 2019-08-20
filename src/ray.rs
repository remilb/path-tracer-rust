use crate::vec3::Vec3;
//use assert_approx_eq::assert_approx_eq;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    origin: Vec3,
    dir: Vec3,
}

impl Ray {
    fn new(origin: Vec3, dir: Vec3) -> Ray {
        Ray {
            origin,
            dir: Vec3::normalize(dir),
        }
    }

    fn position_at_time(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

trait Collider {
    fn get_intersection_point(r: Ray) -> Vec3;
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
