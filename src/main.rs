mod camera;
mod image;
mod ppm;
mod ray;
mod ray_tracer;
mod vec3;

use ray_tracer::RayTracer;
use camera::Camera;
use vec3::Vec3;

fn main() -> std::io::Result<()> {
    let width = 1920;
    let height = 1080;
    let camera = Camera::new(Vec3::new(0.0, 0.0, 0.0), width, height, 10.0);
    let rt = RayTracer::new(camera);
    let img = rt.render();
    ppm::write_ppm("fooey.ppm", img)
}
