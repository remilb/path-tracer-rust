use crate::vec3::Vec3;

//Probably useful initially to be able to specify width and height of
//our plane of projection (in world coordinates), since it will effectively be the viewport as well.
//This combined with a horizontal fov (or alternatively )
pub struct Camera {
    pos: Vec3,
    pub w: u32,
    pub h: u32,
    hfov: f32,
    f_len: f32
}

impl Camera {
    pub fn new(pos: Vec3, w: u32, h: u32, f_len: f32) -> Camera {
        Camera {pos, w, h, f_len, hfov: 0.0}
    }
}