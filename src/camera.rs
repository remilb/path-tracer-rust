use crate::ray::Ray;
use crate::vec3::Vec3;

//Probably useful initially to be able to specify width and height of
//our plane of projection (in world coordinates), since it will effectively be the viewport as well.
//This combined with a horizontal fov (or alternatively )
pub struct Camera {
    pos: Vec3,
    dir: Vec3,
    up: Vec3,
    pub w: f32,
    pub h: f32,
    f_len: f32,
    corners: (Vec3, Vec3, Vec3, Vec3),
}

impl Camera {
    pub fn new(pos: Vec3, dir: Vec3, up: Vec3, w: f32, h: f32, f_len: f32) -> Camera {
        Camera {
            pos,
            dir: Vec3::normalize(dir),
            up: Vec3::normalize(up),
            w,
            h,
            f_len,
            corners: Camera::projection_corners(pos, up, dir, w, h, f_len),
        }
    }

    pub fn pos(&self) -> Vec3 {
        self.pos
    }

    pub fn default_camera() -> Camera {
        Camera::new(
            Vec3::zeros(),
            Vec3::new(0., 0., -1.),
            Vec3::new(0., 1., 0.),
            4.,
            2.,
            2.,
        )
    }

    pub fn cast_ray(&self, u: f32, v: f32) -> Ray {
        let (ul, _ur, lr, ll) = self.corners;
        let (hor, vert) = (lr - ll, ul - ll);
        Ray::new(self.pos, ll + u * hor + v * vert - self.pos)
    }

    //TODO: This is super brittle right now, up and dir are only orthogonal by construction.
    //I'll get back to it with a more general transform system and all that jazz
    fn projection_corners(
        pos: Vec3,
        up: Vec3,
        dir: Vec3,
        w: f32,
        h: f32,
        f_len: f32,
    ) -> (Vec3, Vec3, Vec3, Vec3) {
        let right = Vec3::normalize(Vec3::cross(up, -dir));
        let half_diagonal_up = (w / 2.) * right + (h / 2.) * up;
        let half_diagonal_down = (w / 2.) * right + (h / 2.) * -up;
        let center = pos + dir * f_len;
        let upper_left = center - half_diagonal_down;
        let upper_right = center + half_diagonal_up;
        let lower_right = center + half_diagonal_down;
        let lower_left = center - half_diagonal_up;
        (upper_left, upper_right, lower_right, lower_left)
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
