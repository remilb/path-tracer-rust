use crate::ray::Ray;
use crate::vec3::Vec3;
use rand::random;

pub trait Shader {
    fn shade(&self, in_ray: Ray, point: Vec3, normal: Vec3) -> Option<ShaderResult>;
}

pub struct ShaderResult {
    pub out_ray: Ray,
    pub attenuation: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub enum Material {
    Lambert(Lambert),
    Metal(Metal),
    Dielectric(Dielectric),
}

impl Shader for Material {
    fn shade(&self, in_ray: Ray, point: Vec3, normal: Vec3) -> Option<ShaderResult> {
        match self {
            Material::Lambert(l) => l.shade(in_ray, point, normal),
            Material::Metal(m) => m.shade(in_ray, point, normal),
            Material::Dielectric(d) => d.shade(in_ray, point, normal),
        }
    }
}

// Different shaders as their own types
#[derive(Debug, Clone, Copy)]
pub struct Lambert {
    pub albedo: Vec3,
}

impl Shader for Lambert {
    fn shade(&self, in_ray: Ray, point: Vec3, normal: Vec3) -> Option<ShaderResult> {
        let target = point + normal + random_point_in_unit_sphere();
        let out_ray = Ray::new(point, target - point);
        let attenuation = self.albedo;
        Some(ShaderResult {
            out_ray,
            attenuation,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Metal {
    pub albedo: Vec3,
    pub fuzz: f32,
}

impl Shader for Metal {
    fn shade(&self, in_ray: Ray, point: Vec3, normal: Vec3) -> Option<ShaderResult> {
        let out_ray_dir =
            Vec3::reflect(in_ray.dir(), normal) + self.fuzz * random_point_in_unit_sphere();
        let out_ray = Ray::new(point, out_ray_dir);
        let attenuation = self.albedo;
        if Vec3::dot(out_ray_dir, normal) > 0. {
            Some(ShaderResult {
                out_ray,
                attenuation,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Dielectric {
    pub ior: f32,
    pub albedo: Vec3,
}

impl Dielectric {
    pub fn refract(vin: Vec3, n: Vec3, ni_over_nt: f32) -> Option<Vec3> {
        let vin = Vec3::normalize(vin);
        let dt = Vec3::dot(vin, n);
        let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
        if discriminant > 0.0 {
            let refracted = ni_over_nt * (vin - n * dt) - n * discriminant.sqrt();
            Some(refracted)
        } else {
            None
        }
    }

    pub fn schlick(cosine: f32, ior: f32) -> f32 {
        let r0 = (1.0 - ior) / (1.0 + ior);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * ((1.0 - cosine).powi(5))
    }
}

impl Shader for Dielectric {
    fn shade(&self, in_ray: Ray, point: Vec3, normal: Vec3) -> Option<ShaderResult> {
        let reflected = Vec3::reflect(in_ray.dir(), normal);
        let is_internal_ray = Vec3::dot(in_ray.dir(), normal) > 0.0;

        let (outward_normal, ni_over_nt, cosine) = if is_internal_ray {
            let cosine = self.ior * Vec3::dot(in_ray.dir(), normal);
            (-normal, self.ior, cosine)
        } else {
            let cosine = -Vec3::dot(in_ray.dir(), normal);
            (normal, 1.0 / self.ior, cosine)
        };

        let mut reflect_prob: f32 = 1.0;
        let mut refracted = Vec3::zeros();

        if let Some(r) = Dielectric::refract(in_ray.dir(), outward_normal, ni_over_nt) {
            reflect_prob = Dielectric::schlick(cosine, self.ior);
            refracted = r;
        }

        if random::<f32>() < reflect_prob {
            Some(ShaderResult {
                out_ray: Ray::new(point, reflected),
                attenuation: self.albedo,
            })
        } else {
            Some(ShaderResult {
                out_ray: Ray::new(point, refracted),
                attenuation: self.albedo,
            })
        }
    }
}

fn random_point_in_unit_sphere() -> Vec3 {
    loop {
        let p = 2.0 * Vec3::new(random(), random(), random()) - Vec3::ones();
        if Vec3::squared_len(p) < 1.0 {
            return p;
        }
    }
}
