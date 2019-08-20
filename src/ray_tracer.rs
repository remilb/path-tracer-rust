use crate::camera::Camera;
use crate::image::Image;
use std::convert::TryInto;

pub struct RayTracer {
    camera: Camera,
}

impl RayTracer {
    pub fn new(camera: Camera) -> RayTracer {
        RayTracer { camera }
    }
    pub fn render(&self) -> Image {
        let mut img = Image::new(self.camera.w, self.camera.h);
        RayTracer::draw_background(&mut img);
        return img;
    }

    // Background is a special case for now
    fn draw_background(img: &mut Image) {
        let white = (1.0, 1.0, 1.0);
        let blue = (0.5, 0.7, 1.0);
        for y in 0..img.h() {
            let t = (y as f32) / (img.h() as f32);
            for x in 0..img.w() {
                let color = (
                    t * white.0 + (1.0 - t) * blue.0,
                    t * white.1 + (1.0 - t) * blue.1,
                    t * white.2 + (1.0 - t) * blue.2,
                );
                img.set_pixel(
                    x,
                    y,
                    (color.0 * 255.99) as u8,
                    (color.1 * 255.99) as u8,
                    (color.2 * 255.99) as u8,
                );
            }
        }
    }
}
