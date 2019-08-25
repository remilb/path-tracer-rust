use crate::ray::Ray;
use crate::vec3::Vec3;
use rand::random;

//Probably useful initially to be able to specify width and height of
//our plane of projection (in world coordinates), since it will effectively be the viewport as well.
//This combined with a horizontal fov (or alternatively )
pub struct Camera {
    pos: Vec3,
    vfov: f32,
    lens_radius: f32,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
}

impl Camera {
    pub fn new(
        pos: Vec3,
        lookat: Vec3,
        up: Vec3,
        vfov: f32,
        aspect: f32,
        aperture: f32,
        flen: f32,
    ) -> Camera {
        let vfov = vfov * std::f32::consts::PI / 180.0;
        let half_height = (vfov / 2.0).tan();
        let half_width = half_height * aspect;
        let w = Vec3::normalize(pos - lookat);
        let u = Vec3::normalize(Vec3::cross(up, w));
        let v = Vec3::cross(w, u);
        Camera {
            pos,
            vfov,
            lens_radius: aperture / 2.0,
            lower_left_corner: pos - flen * (half_width * u + half_height * v + w),
            horizontal: 2.0 * half_width * u * flen,
            vertical: 2.0 * half_height * v * flen,
            u,
            v,
            w,
        }
    }

    pub fn default_camera() -> Camera {
        let pos = Vec3::new(13., 2., 3.);
        let lookat = Vec3::new(0., 0., 0.);
        let up = Vec3::new(0., 1., 0.);
        Camera::new(pos, lookat, up, 20.0, 2.0, 0.1, 10.0)
    }

    pub fn cast_ray(&self, s: f32, t: f32) -> Ray {
        let rd = self.lens_radius * Camera::random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(
            self.pos + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.pos - offset,
        )
    }

    fn random_in_unit_disk() -> Vec3 {
        loop {
            let p = 2.0 * Vec3::new(random(), random(), 0.0) - Vec3::new(1.0, 1.0, 0.0);
            if Vec3::squared_len(p) < 1.0 {
                return p;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn camera_corners() {
        // Trivial case with axis-aligned projection plane
        let pos = Vec3::zeros();
        let dir = Vec3::new(0., 0., -1.);
        let up = Vec3::new(0., 1., 0.);
        let c = Camera::new(pos, dir, up, 4., 2., 2.);
        let corners = c.corners();
        assert_eq!(corners, (Vec3::new(-2., -1., -2.), Vec3::new(2., 1., -2.)));

        // Case with axis-unaligned projection plane
        // let dir = Vec3::new(1., 0., -1.);
        // let c = Camera::new(pos, dir, up, 4., 2., 2.);
        // let corners = c.corners();
        // assert_eq!(corners, (Vec3::new(-2., -1., -2.), Vec3::new(2., 1., -2.)));
    }
}
