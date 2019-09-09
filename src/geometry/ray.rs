use super::point::Point3f;
use super::vector::Vec3f;
use super::{FloatRT, INFINITY};
use std::ops::FnOnce;

#[derive(Debug, Clone, Copy, PartialEq)]
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
mod tests {
    use super::*;
    use approx::{AbsDiffEq, RelativeEq, UlpsEq};

    #[test]
    fn point_at_t() {
        let d = Vec3f::new(1.0, 1.0, 1.0);
        let r = Ray {
            d,
            ..Ray::default()
        };
        let p = Point3f::new(4.0, 4.0, 4.0);
        assert_eq!(r.point_at_t(4.0), p);
    }

    /// Approximate equality implementations for testing purposes
    impl AbsDiffEq for Ray {
        type Epsilon = <FloatRT as AbsDiffEq>::Epsilon;
        fn default_epsilon() -> Self::Epsilon {
            Self::Epsilon::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            Point3f::abs_diff_eq(&self.o, &other.o, epsilon)
                && Vec3f::abs_diff_eq(&self.d, &other.d, epsilon)
        }
    }

    impl RelativeEq for Ray {
        fn default_max_relative() -> <FloatRT as AbsDiffEq>::Epsilon {
            FloatRT::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: <FloatRT as AbsDiffEq>::Epsilon,
        ) -> bool {
            Point3f::relative_eq(&self.o, &other.o, epsilon, max_relative)
                && Vec3f::relative_eq(&self.d, &other.d, epsilon, max_relative)
        }
    }

    impl UlpsEq for Ray {
        fn default_max_ulps() -> u32 {
            FloatRT::default_max_ulps()
        }

        fn ulps_eq(
            &self,
            other: &Self,
            epsilon: <FloatRT as AbsDiffEq>::Epsilon,
            max_ulps: u32,
        ) -> bool {
            Point3f::ulps_eq(&self.o, &other.o, epsilon, max_ulps)
                && Vec3f::ulps_eq(&self.d, &other.d, epsilon, max_ulps)
        }
    }
}
