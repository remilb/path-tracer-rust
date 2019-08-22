use crate::ray::{Collider, Collision, Ray};
use crate::surfaces::{Parallelogram, Plane, Sphere, Surface};
use crate::vec3::Vec3;

pub struct Scene {
    surfaces: Vec<Surface>,
}

impl Scene {
    pub fn default_scene() -> Scene {
        let s = Sphere {
            center: Vec3::new(0., 0., -2.),
            r: 0.5,
        };

        let s2 = Sphere {
            center: Vec3::new(0., -100.5, -3.),
            r: 100.0,
        };

        let p = Parallelogram::new(
            Vec3::new(1., 1., -8.),
            Vec3::new(2., 0., 0.),
            Vec3::new(1., 1., 0.),
        );

        let surfaces = vec![Surface::Sphere(s), Surface::Sphere(s2)];
        Scene { surfaces }
    }
}

impl Collider for Scene {
    fn get_collision(&self, r: Ray, t_min: f32, t_max: f32) -> Option<Collision> {
        let mut hit_something = false;
        let mut closest_so_far = t_max;
        let mut temp_col: Option<Collision> = None;
        for s in &self.surfaces {
            if let Some(collision) = s.get_collision(r, t_min, closest_so_far) {
                hit_something = true;
                closest_so_far = collision.t;
                temp_col = Some(collision)
            }
        }

        if hit_something {
            temp_col
        } else {
            None
        }
    }
}

pub struct IterHelper<'a> {
    iter: std::slice::Iter<'a, Surface>,
}

impl<'a> IntoIterator for &'a Scene {
    type Item = &'a Surface;
    type IntoIter = IterHelper<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IterHelper {
            iter: self.surfaces.iter(),
        }
    }
}

impl<'a> Iterator for IterHelper<'a> {
    type Item = &'a Surface;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
