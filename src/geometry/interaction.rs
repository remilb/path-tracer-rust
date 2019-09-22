use super::normal::Normal3f;
use super::point::{Point2f, Point3f};
use super::ray::Ray;
use super::vector::Vec3f;
use super::{FloatRT, Cross, Dot};
use num_traits::Float;

pub enum InteractionType {
    SurfaceInteraction(SurfaceInteraction),
    MediumInteraction(MediumInteraction),
}

//TODO: Rigorous round off error handling
pub trait Interaction {
    fn p(&self) -> Point3f;
    fn n(&self) -> Normal3f;
    fn time(&self) -> FloatRT;

    fn spawn_ray(&self, d: Vec3f) -> Ray {
        Ray::new(self.p(), d, FloatRT::infinity(), self.time())
    }

    fn spawn_ray_to_point(&self, p: Point3f) -> Ray {
        let d = p - self.p();
        Ray::new(self.p(), d, 1.0, self.time())
    }

    fn spawn_ray_to_interaction<T: Interaction>(&self, it: T) -> Ray {
        let d = it.p() - self.p();
        Ray::new(self.p(), d, 1.0, self.time())
    }
}


pub struct SurfaceInteraction {
    pub p: Point3f,
    pub time: FloatRT,
    pub wo: Vec3f,
    pub n: Normal3f,
    pub uv: Point2f,
    pub dp: (Vec3f, Vec3f),
    pub dn: (Normal3f, Normal3f),
    pub shading: ShadingData
}

// TODO: Implement mediums at some point
pub struct MediumInteraction {
    p: Point3f,
    time: FloatRT,
    wo: Vec3f,
}

pub struct ShadingData {
    pub n: Normal3f,
    pub dp: (Vec3f, Vec3f),
    pub dn: (Normal3f, Normal3f),
}

impl SurfaceInteraction {
    pub fn new(p: Point3f, uv:  Point2f, wo: Vec3f, dp: (Vec3f, Vec3f), dn: (Normal3f, Normal3f), time: FloatRT, flip_normal: bool) -> Self {
        let n = if flip_normal {-Normal3f::from(Vec3f::cross(dp.0, dp.1).normalize())} else {Normal3f::from(Vec3f::cross(dp.0, dp.1).normalize())};
        let shading = ShadingData {n, dp, dn};
        Self {p, time, wo, n, uv, dp, dn, shading}
    }

    pub fn set_shading_geometry(&mut self, dp: (Vec3f, Vec3f), dn: (Normal3f, Normal3f), flip_normal: bool, orientation_is_authoritative: bool) {
        let n = if flip_normal {-Normal3f::from(Vec3f::cross(dp.0, dp.1).normalize())} else {Normal3f::from(Vec3f::cross(dp.0, dp.1).normalize())};
        let mut shading = ShadingData {n, dp, dn};
        if orientation_is_authoritative {
            self.n = self.n.face_forward(n);
        } else {
            shading.n = shading.n.face_forward(self.n);
        }
        self.shading = shading;
    }
}