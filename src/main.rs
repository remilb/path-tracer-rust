mod camera;
mod image;
mod scene;
mod surfaces;
mod ppm;
mod ray;
mod ray_tracer;
mod vec3;

use camera::Camera;
use ray_tracer::RayTracer;

fn main() -> std::io::Result<()> {
    let camera = Camera::default_camera();
    let rt = RayTracer::new(camera);
    let img = rt.render();
    ppm::write_ppm("fooey.ppm", img)
}
