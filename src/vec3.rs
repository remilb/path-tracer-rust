use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
//TODO: pub is sloppy methinks
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn zeros() -> Vec3 {
        Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }

    pub fn ones() -> Vec3 {
        Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }

    pub fn dot(v1: Vec3, v2: Vec3) -> f32 {
        v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    }

    pub fn cross(v1: Vec3, v2: Vec3) -> Vec3 {
        Vec3 {
            x: v1.y * v2.z - v1.z * v2.y,
            y: v1.z * v2.x - v1.x * v2.z,
            z: v1.x * v2.y - v1.y * v2.x,
        }
    }

    pub fn squared_len(v: Vec3) -> f32 {
        Vec3::dot(v, v)
    }

    pub fn len(v: Vec3) -> f32 {
        Vec3::squared_len(v).sqrt()
    }

    pub fn normalize(v: Vec3) -> Vec3 {
        v / Vec3::len(v)
    }

    pub fn dist(v1: Vec3, v2: Vec3) -> f32 {
        Vec3::len(v2 - v1)
    }

    pub fn lerp(v1: Vec3, v2: Vec3, t: f32) -> Vec3 {
        if t < 0. || t > 1. {
            panic!(format!(
                "t in lerp should be in [0.0, 1.0], value was {}",
                t
            ))
        }
        (1.0 - t) * v1 + t * v2
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f32) -> Vec3 {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

// ** Not sure if I really want these yet
// impl AddAssign for Vec3 {
//     fn add_assign(&mut self, other: Self) {
//         *self = *self + other
//     }
// }

// impl SubAssign for Vec3 {
//     fn sub_assign(&mut self, other: Self) {
//         *self = *self - other
//     }
// }

// impl MulAssign<f32> for Vec3 {
//     fn mul_assign(&mut self, other: f32) {
//         *self = *self * other
//     }
// }

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn vector_equals() {
        let v1 = Vec3::new(1.0, 1.0, 1.0);
        let v2 = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn vector_add() {
        let v1 = Vec3::new(1.0, -2.0, 5.0);
        let v2 = Vec3::new(12.0, 3.0, -1.0);
        assert_eq!(v1 + v2, Vec3::new(13.0, 1.0, 4.0));
    }

    #[test]
    fn vector_sub() {
        let v1 = Vec3::new(1.0, -2.0, 5.0);
        let v2 = Vec3::new(12.0, 3.0, -1.0);
        assert_eq!(v1 - v2, Vec3::new(-11.0, -5.0, 6.0));
    }

    #[test]
    fn vector_negate() {
        let v1 = Vec3::new(1.0, -2.0, 5.0);
        assert_eq!(-v1, Vec3::new(-1.0, 2.0, -5.0));
    }

    #[test]
    fn vector_mul_by_scalar() {
        let v1 = Vec3::new(2.0, 3.0, 4.0);
        assert_eq!(v1 * 2.0, Vec3::new(4.0, 6.0, 8.0));
        assert_eq!(2.0 * v1, Vec3::new(4.0, 6.0, 8.0));
    }

    #[test]
    fn vector_div_by_scalar() {
        let v1 = Vec3::new(2.0, 3.0, 4.0);
        assert_eq!(v1 / 2.0, Vec3::new(1.0, 1.5, 2.0));
    }

    #[test]
    fn vector_dot() {
        let v1 = Vec3::new(2.0, 3.0, 4.0);
        let v2 = Vec3::new(1.0, -2.0, 5.0);
        assert_eq!(Vec3::dot(v1, v2), 16.0);
    }

    #[test]
    fn vector_cross() {
        let v1 = Vec3::new(2.0, 3.0, 4.0);
        let v2 = Vec3::new(1.0, -2.0, 5.0);
        assert_eq!(Vec3::cross(v1, v2), Vec3::new(23.0, -6.0, -7.0));
    }
    // #[test]
    // fn vector_length() {
    //     let v = Vec3::new(2.0, 3.0, 4.0);
    //     ass

    //#[test]
    // fn vector_compound_assign() {
    //     let v1 = Vec3::new(2.0, 3.0, 4.0);
    //     // Add
    //     v1 += Vec3::new(1.0, 1.0, 2.0);
    //     assert_eq!(v1, Vec3::new(3.0, 4.0, 6.0));
    //     // Sub
    // }
}
