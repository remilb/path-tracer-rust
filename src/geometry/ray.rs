use super::point::Point3f;
use super::vector::Vec3f;
use super::{FloatRT, INFINITY};
use std::ops::FnOnce;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub o: Point3f,
    pub d: Vec3f,
    pub tMax: FloatRT,
    pub time: FloatRT,
}

impl Ray {
    pub fn new(o: Point3f, d: Vec3f, tMax: FloatRT, time: FloatRT) -> Ray {
        Ray { o, d, tMax, time }
    }

    pub fn point_at_t(self, t: FloatRT) -> Point3f {
        self.o + self.d * t
    }
}

impl Default for Ray {
    fn default() -> Ray {
        Ray {
            o: Point3f::zeros(),
            d: Vec3f::zeros(),
            tMax: INFINITY,
            time: 0.0,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ray_point_at_t() {
        let d = Vec3f::new(1.0, 1.0, 1.0);
        let r = Ray {
            d,
            ..Ray::default()
        };
        let p = Point3f::new(4.0, 4.0, 4.0);
        assert_eq!(r.point_at_t(4.0), p);
    }
}
