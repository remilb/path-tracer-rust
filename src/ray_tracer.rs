use crate::camera::Camera;
use crate::image::Image;
use crate::objects::Sphere;
use crate::ray::{Collider, Ray};
use crate::vec3::Vec3;

pub struct RayTracer {
    camera: Camera,
}

impl RayTracer {
    pub fn new(camera: Camera) -> RayTracer {
        RayTracer { camera }
    }
    pub fn render(&self) -> Image {
        let (nx, ny) = (2160 * 2, 1080 * 2);
        let mut img = Image::new(nx, ny);
        let (ul, _ur, lr, ll) = self.camera.corners();
        let sphere = Sphere {
            center: Vec3::new(0., 0., -1.0),
            r: 0.5,
        };

        for y in 0..ny {
            for x in 0..nx {
                let (xt, yt) = (x as f32 / nx as f32, y as f32 / ny as f32);
                let (u, v) = (Vec3::lerp(ll, lr, xt), Vec3::lerp(ll, ul, yt));
                let ray = Ray::new(
                    self.camera.pos(),
                    Vec3::new(u.x, v.y, v.z) - self.camera.pos(),
                );
                let color = self.color(ray, sphere);
                img.set_pixel(x, ny - 1 - y, color);
            }
        }

        img
    }

    //TODO: Background doesn't really look how I'd like due to unit rays
    fn color(&self, ray: Ray, sphere: Sphere) -> Vec3 {
        if let Some(collision) = sphere.get_collision(ray) {
            return 0.5 * (collision.normal + Vec3::new(1., 1., 1.));
        }

        let white = Vec3::new(1.0, 1.0, 1.0);
        let blue = Vec3::new(0.5, 0.7, 1.0);
        let t = (ray.dir().y + (self.camera.h / 2.)) / self.camera.h;
        Vec3::lerp(white, blue, t)
    }
}
