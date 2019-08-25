use crate::materials::{Dielectric, Lambert, Material, Metal};
use crate::ray::{Collider, Collision, Ray};
use crate::surfaces::{Parallelogram, Plane, Sphere, Surface};
use crate::vec3::Vec3;
use rand::random;

pub struct Scene {
    surfaces: Vec<Surface>,
}

impl Scene {
    pub fn default_scene() -> Scene {
        let mat1 = Material::Lambert(Lambert {
            albedo: Vec3::new(0.5, 0.3, 0.2),
        });
        let mat2 = Material::Lambert(Lambert {
            albedo: Vec3::new(0.2, 0.5, 0.2),
        });
        let mat3 = Material::Metal(Metal {
            albedo: Vec3::new(0.8, 0.6, 0.2),
            fuzz: 0.4,
        });
        let mat4 = Material::Metal(Metal {
            albedo: Vec3::new(0.75, 0.76, 0.76),
            fuzz: 0.,
        });
        let glass = Material::Dielectric(Dielectric {
            albedo: Vec3::ones(),
            ior: 1.5,
        });

        let s1 = Sphere {
            center: Vec3::new(0., 0., -2.),
            r: 0.5,
            mat: glass,
        };

        let s2 = Sphere {
            center: Vec3::new(0., -100.5, -3.),
            r: 100.0,
            mat: mat2,
        };

        let s3 = Sphere {
            center: Vec3::new(-1., 0., -3.),
            r: 0.3,
            mat: mat3,
        };

        let s4 = Sphere {
            center: Vec3::new(1.3, -0.1, -2.5),
            r: 0.4,
            mat: mat4,
        };

        let surfaces = vec![
            Surface::Sphere(s1),
            Surface::Sphere(s2),
            Surface::Sphere(s3),
            Surface::Sphere(s4),
        ];
        Scene { surfaces }
    }

    pub fn random_scene() -> Scene {
        let mut surfaces = Vec::new();
        let randf = random::<f32>;

        //World sphere
        let mat = Material::Lambert(Lambert {
            albedo: Vec3::new(0.5, 0.5, 0.5),
        });
        surfaces.push(Surface::Sphere(Sphere {
            center: Vec3::new(0.0, -1000.0, 0.0),
            r: 1000.0,
            mat,
        }));

        for a in -11..11 {
            for b in -11..11 {
                let choose_mat: f32 = random();
                let center = Vec3::new((a as f32) + 0.9 * randf(), 0.2, (b as f32) + 0.9 * randf());
                if Vec3::len(center - Vec3::new(4.0, 0.2, 0.0)) > 0.9 {
                    if choose_mat < 0.8 {
                        //diffuse
                        let color =
                            Vec3::new(randf() * randf(), randf() * randf(), randf() * randf());
                        let mat = Material::Lambert(Lambert { albedo: color });
                        surfaces.push(Surface::Sphere(Sphere {
                            center,
                            r: 0.2,
                            mat,
                        }));
                    } else if choose_mat < 0.95 {
                        //metal
                        let color = Vec3::new(
                            0.5 * (randf() + 1.0),
                            0.5 * (randf() + 1.0),
                            0.5 * (randf() + 1.0),
                        );
                        let mat = Material::Metal(Metal {
                            albedo: color,
                            fuzz: randf() * 0.5,
                        });
                        surfaces.push(Surface::Sphere(Sphere {
                            center,
                            r: 0.2,
                            mat,
                        }));
                    } else {
                        // glass
                        let mat = Material::Dielectric(Dielectric {
                            ior: 1.5,
                            albedo: Vec3::ones(),
                        });
                        surfaces.push(Surface::Sphere(Sphere {
                            center,
                            r: 0.2,
                            mat,
                        }));
                    }
                }
            }
        }

        // Some big guys
        let glass = Material::Dielectric(Dielectric {
            ior: 1.5,
            albedo: Vec3::ones(),
        });
        let metal = Material::Metal(Metal {
            albedo: Vec3::new(0.7, 0.6, 0.5),
            fuzz: 0.0,
        });
        let diffuse = Material::Lambert(Lambert {
            albedo: Vec3::new(0.4, 0.2, 0.1),
        });
        surfaces.push(Surface::Sphere(Sphere {
            center: Vec3::new(0.0, 1.0, 0.0),
            r: 1.0,
            mat: glass,
        }));
        surfaces.push(Surface::Sphere(Sphere {
            center: Vec3::new(-4.0, 1.0, 0.0),
            r: 1.0,
            mat: diffuse,
        }));
        surfaces.push(Surface::Sphere(Sphere {
            center: Vec3::new(4.0, 1.0, 0.0),
            r: 1.0,
            mat: metal,
        }));

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
