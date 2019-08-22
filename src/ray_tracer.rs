use crate::camera::Camera;
use crate::image::Image;
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
        let scene = Scene::default_scene();
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

    //TODO: Background doesn't really look how I'd like due to unit rays
    fn color(&self, ray: Ray, scene: &Scene) -> Vec3 {
        // Check objects in scene
        if let Some(collision) = scene.get_collision(ray, 0.001, std::f32::MAX) {
            let out_ray = Ray::new(
                collision.point,
                (collision.point + collision.normal + RayTracer::random_point_in_unit_sphere())
                    - collision.point,
            );
            //return 0.5 * (collision.normal + Vec3::new(1., 1., 1.));
            return 0.5 * self.color(out_ray, scene);
        } else {
            // Or return skybox
            let white = Vec3::new(1.0, 1.0, 1.0);
            let blue = Vec3::new(0.5, 0.7, 1.0);
            let t = (ray.dir().y + (self.camera.h / 2.)) / self.camera.h;
            Vec3::lerp(white, blue, t)
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
}
