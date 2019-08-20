mod image;
mod ppm;
mod vec3;
mod ray;

fn main() -> std::io::Result<()> {
    ppm::write_ppm("fooey.ppm", image::Image::new_test_image(1920, 1080))
}
