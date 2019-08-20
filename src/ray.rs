use crate::vec3::Vec3;

#[derive(Debug)]
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
