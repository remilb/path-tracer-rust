use crate::camera::Camera;
use crate::image::Image;
use crate::materials::Shader;
use crate::ray::{Collider, Ray};
use crate::scene::Scene;
use crate::vec3::Vec3;
use rand::prelude::*;

pub struct RayTracer {
    camera: Camera,
}

impl RayTracer {
    pub fn new(camera: Camera) -> RayTracer {
        RayTracer { camera }
    }
    pub fn render(&self) -> Image {
        let scene = Scene::random_scene();
        let (nx, ny) = (2160, 1080);
        let ns = 100;
        let mut img = Image::new(nx, ny);

        for y in 0..ny {
            for x in 0..nx {
                let mut color = Vec3::zeros();
                for _s in 0..ns {
                    let (jx, jy): (f32, f32) = (random(), random());
                    let (xt, yt) = ((x as f32 + jx) / nx as f32, (y as f32 + jy) / ny as f32);
                    let ray = self.camera.cast_ray(xt, yt);
                    color = color + self.color(ray, &scene);
                }
                color = color / (ns as f32);
                img.set_pixel(x, ny - 1 - y, color);
            }
        }

        img
    }

    fn color(&self, ray: Ray, scene: &Scene) -> Vec3 {
        // Check objects in scene
        if let Some(collision) = scene.get_collision(ray, 0.001, std::f32::MAX) {
            let shader_res = collision.mat.shade(ray, collision.point, collision.normal);
            if let Some(res) = shader_res {
                //return 0.5 * (collision.normal + Vec3::new(1., 1., 1.));
                return res.attenuation * self.color(res.out_ray, scene);
            } else {
                //Absorbed
                return Vec3::zeros();
            }
        } else {
            // Or return skybox
            let white = Vec3::new(1.0, 1.0, 1.0);
            let blue = Vec3::new(0.5, 0.7, 1.0);
            let t = 0.5 * (ray.dir().y + 1.0);
            Vec3::lerp(white, blue, t)
        }
    }
}
