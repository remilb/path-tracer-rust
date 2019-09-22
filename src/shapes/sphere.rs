use super::{Shape, ShapeBaseData};
use crate::geometry::{
    Bounds3f, Cross, Dot, FloatRT, Normal3f, Point2f, Point3f, Ray, SurfaceInteraction, Transform,
    Transformer, Vec3f,
};
use crate::utils::math::quadratic;
use num_traits::clamp;

struct Sphere {
    sbd: ShapeBaseData,
    radius: FloatRT,
    z_min: FloatRT,
    z_max: FloatRT,
    theta_min: FloatRT,
    theta_max: FloatRT,
    phi_max: FloatRT,
}

impl Sphere {
    pub fn new(
        object_to_world: Transform,
        world_to_object: Transform,
        reverse_orientation: bool,
        radius: FloatRT,
        z_min: FloatRT,
        z_max: FloatRT,
        phi_max: FloatRT,
    ) -> Self {
        let transform_swaps_handedness = object_to_world.swaps_handedness();
        let sbd = ShapeBaseData {
            object_to_world,
            world_to_object,
            reverse_orientation,
            transform_swaps_handedness,
        };
        let z_min = clamp(z_min.min(z_max), -radius, radius);
        let z_max = clamp(z_max.max(z_min), -radius, radius);
        let theta_min = clamp(z_min / radius, -1.0, 1.0).acos();
        let theta_max = clamp(z_max / radius, -1.0, 1.0).acos();
        let phi_max = clamp(phi_max, 0.0, 360.0).to_radians();
        Self {
            sbd,
            radius,
            z_min,
            z_max,
            theta_min,
            theta_max,
            phi_max,
        }
    }

    fn point_is_valid_partial_sphere(&self, p: Point3f, phi: FloatRT) -> bool {
        let z_valid = (self.z_min == -self.radius && self.z_max == self.radius)
            || (p.z >= self.z_min && p.z <= self.z_max);
        let phi_valid = phi <= self.phi_max;
        z_valid && phi_valid
    }

    /// Returns the relevant partial derivatives of the surface ((dpdu, dpdv), (dndu, dndv))
    fn partial_derivatives(&self, point: Point3f) -> ((Vec3f, Vec3f), (Normal3f, Normal3f)) {
        // Position derivatives
        let theta = clamp(point.z / self.radius, -1.0, 1.0).acos();
        let z_radius = (point.x * point.x + point.y * point.y).sqrt();
        let inv_z_radius = 1.0 / z_radius;
        let cos_phi = point.x * inv_z_radius;
        let sin_phi = point.y * inv_z_radius;
        let dpdu = Vec3f::new(-self.phi_max * point.y, self.phi_max * point.x, 0.0);
        let dpdv = Vec3f::new(
            point.z * cos_phi,
            point.z * sin_phi,
            -self.radius * theta.sin(),
        ) * (self.theta_max - self.theta_min);

        // Normal derivatives
        let d2pduu = Vec3f::new(point.x, point.y, 0.0) * -self.phi_max * self.phi_max;
        let d2pduv = Vec3f::new(-sin_phi, cos_phi, 0.0)
            * (self.theta_max - self.theta_min)
            * point.z
            * self.phi_max;
        let d2pdvv = Vec3f::from(point)
            * -(self.theta_max - self.theta_min)
            * (self.theta_max - self.theta_min);

        let (E, F, G) = (
            Vec3f::dot(dpdu, dpdu),
            Vec3f::dot(dpdu, dpdv),
            Vec3f::dot(dpdv, dpdv),
        );
        let n = Vec3f::cross(dpdu, dpdv).normalize();
        let (e, f, g) = (
            Vec3f::dot(n, d2pduu),
            Vec3f::dot(n, d2pduv),
            Vec3f::dot(n, d2pdvv),
        );

        let inv_EGF2 = 1.0 / (E * G - F * F);
        let dndu = Normal3f::from(
            (dpdu * (f * F - e * G) * inv_EGF2) + (dpdv * (e * F - f * E) * inv_EGF2),
        );
        let dndv = Normal3f::from(
            (dpdu * (g * F - f * G) * inv_EGF2) + (dpdv * (f * F - g * E) * inv_EGF2),
        );

        ((dpdu, dpdv), (dndu, dndv))
    }
}

impl Shape for Sphere {
    fn transform_swaps_handedness(&self) -> bool {
        self.sbd.transform_swaps_handedness
    }
    fn orientation_is_reversed(&self) -> bool {
        self.sbd.reverse_orientation
    }
    fn object_to_world(&self) -> &Transform {
        &self.sbd.object_to_world
    }

    // TODO: Tighten this up for partially swept spheres
    fn object_bound(&self) -> Bounds3f {
        Bounds3f::new(
            Point3f::new(-self.radius, -self.radius, self.z_min),
            Point3f::new(self.radius, self.radius, self.z_max),
        )
    }

    fn intersect(&self, ray: Ray) -> Option<(FloatRT, SurfaceInteraction)> {
        // Transform ray to object space
        let ray = self.sbd.world_to_object.apply(ray);

        // Quadratic coefficients
        let a = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        let b = 2.0 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y + ray.d.z * ray.o.z);
        let c =
            ray.o.x * ray.o.x + ray.o.y * ray.o.y + ray.o.z * ray.o.z - self.radius * self.radius;

        // Check for solutions
        let (t0, t1) = match quadratic(a, b, c) {
            Some((t0, t1)) => (t0, t1),
            None => return None,
        };

        // Hit times are within bounds?
        if t0 > ray.tMax || t1 <= 0.0 {
            return None;
        }

        // Get nearest hit
        let mut t_hit = if t0 <= 0.0 { t1 } else { t0 };

        // One more check
        if t_hit > ray.tMax {
            return None;
        }

        // Get hit point and refine it a smidge
        let mut point_hit = ray.point_at_t(t_hit);
        point_hit = point_hit * (self.radius / Point3f::dist(point_hit, Point3f::zeros()));

        // Jitter point_hit a touch if it is at either pole of sphere
        point_hit = if point_hit.x == 0.0 && point_hit.y == 0.0 {
            Point3f::new(self.radius * 1e-5, point_hit.y, point_hit.z)
        } else {
            point_hit
        };

        // Get phi for point hit
        let mut phi = point_hit.y.atan2(point_hit.x);
        // Remap [-pi, pi] to [0, 2*pi] to fit our parameterization
        phi = if phi < 0.0 {
            phi + 2.0 * std::f32::consts::PI
        } else {
            phi
        };

        // So gross
        if !self.point_is_valid_partial_sphere(point_hit, phi) {
            if t_hit == t1 {
                return None;
            }
            if t1 > ray.tMax {
                return None;
            }
            t_hit = t1;
            point_hit = ray.point_at_t(t_hit);
            point_hit = point_hit * (self.radius / Point3f::dist(point_hit, Point3f::zeros()));
            point_hit = if point_hit.x == 0.0 && point_hit.y == 0.0 {
                Point3f::new(self.radius * 1e-5, point_hit.y, point_hit.z)
            } else {
                point_hit
            };
            phi = point_hit.y.atan2(point_hit.x);
            phi = if phi < 0.0 {
                phi + 2.0 * std::f32::consts::PI
            } else {
                phi
            };
            if !self.point_is_valid_partial_sphere(point_hit, phi) {
                return None;
            }
        }

        // Find parametric representation
        let u = phi / self.phi_max;
        let theta = clamp(point_hit.z / self.radius, -1.0, 1.0).acos();
        let v = (theta - self.theta_min) / (self.theta_max - self.theta_min);

        // Partial derivatives
        let (dp, dn) = self.partial_derivatives(point_hit);

        // TODO: What is up with the discrepancy between t_hit and the time that gets passed to the SurfaceInteraction?
        let interaction = self.object_to_world().apply(SurfaceInteraction::new(
                point_hit,
                Point2f::new(u, v),
                -ray.d,
                dp,
                dn,
                ray.time,
                self.should_flip_normal(),
            ));

        Some((t_hit, interaction))
    }

    fn intersect_test(&self, ray: Ray) -> bool {
        // Transform ray to object space
        let ray = self.sbd.world_to_object.apply(ray);

        // Quadratic coefficients
        let a = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        let b = 2.0 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y + ray.d.z * ray.o.z);
        let c =
            ray.o.x * ray.o.x + ray.o.y * ray.o.y + ray.o.z * ray.o.z - self.radius * self.radius;

        // Check for solutions
        let (t0, t1) = match quadratic(a, b, c) {
            Some((t0, t1)) => (t0, t1),
            None => return false,
        };

        // Hit times are within bounds?
        if t0 > ray.tMax || t1 <= 0.0 {
            return false;
        }

        // Get nearest hit
        let mut t_hit = if t0 <= 0.0 { t1 } else { t0 };

        // One more check
        if t_hit > ray.tMax {
            return false;
        }

        // Get hit point and refine it a smidge
        let mut point_hit = ray.point_at_t(t_hit);
        point_hit = point_hit * (self.radius / Point3f::dist(point_hit, Point3f::zeros()));

        // Jitter point_hit a touch if it is at either pole of sphere
        point_hit = if point_hit.x == 0.0 && point_hit.y == 0.0 {
            Point3f::new(self.radius * 1e-5, point_hit.y, point_hit.z)
        } else {
            point_hit
        };

        // Get phi for point hit
        let mut phi = point_hit.y.atan2(point_hit.x);
        // Remap [-pi, pi] to [0, 2*pi] to fit our parameterization
        phi = if phi < 0.0 {
            phi + 2.0 * std::f32::consts::PI
        } else {
            phi
        };

        // So gross
        if !self.point_is_valid_partial_sphere(point_hit, phi) {
            if t_hit == t1 {
                return false;
            }
            if t1 > ray.tMax {
                return false;
            }
            t_hit = t1;
            point_hit = ray.point_at_t(t_hit);
            point_hit = point_hit * (self.radius / Point3f::dist(point_hit, Point3f::zeros()));
            point_hit = if point_hit.x == 0.0 && point_hit.y == 0.0 {
                Point3f::new(self.radius * 1e-5, point_hit.y, point_hit.z)
            } else {
                point_hit
            };
            phi = point_hit.y.atan2(point_hit.x);
            phi = if phi < 0.0 {
                phi + 2.0 * std::f32::consts::PI
            } else {
                phi
            };
            if !self.point_is_valid_partial_sphere(point_hit, phi) {
                return false;
            }
        }
        return true;
    }

    fn surface_area(&self) -> FloatRT {
        self.phi_max * self.radius * (self.z_max - self.z_min)
    }
}

mod tests {
    use super::*;
    use num_traits::Float;
    use approx::assert_relative_eq;

    #[test]
    fn intersect() {
        // Place a sphere down the z-axis
        let o2w = Transform::translate(Vec3f::new(0.0, 0.0, 10.0));
        let w2o = o2w.inverse();
        let sphere = Sphere::new(o2w, w2o, false, 4.0, -4.0, 4.0, 360.0);

        // Should miss
        let ray = Ray::new(
            Point3f::zeros(),
            Vec3f::new(6.0, 1.0, 1.0),
            FloatRT::infinity(),
            0.0,
        );
        let hit = sphere.intersect(ray);
        assert!(hit.is_none());

        // Should hit
        let ray = Ray::new(
            Point3f::zeros(),
            Vec3f::new(0.0, 0.0, 1.0),
            FloatRT::infinity(),
            0.0,
        );
        let hit = sphere.intersect(ray);
        match hit {
            Some((t, _)) => assert_eq!(t, 6.0),
            None => panic!(),
        }

        // Grazing hit
        let ray = Ray::new(
            Point3f::new(4.0, 0.0, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
            FloatRT::infinity(),
            0.0,
        );
        let hit = sphere.intersect(ray);
        assert!(hit.is_some());
    }

    #[test]
    fn intersect_partial_spheres() {
        let o2w = Transform::translate(Vec3f::new(0.0, 0.0, 10.0));
        let o2w = o2w * Transform::rotate_x(-90.0);
        let w2o = o2w.inverse();
        let sphere = Sphere::new(o2w, w2o, false, 4.0, -3.0, 3.0, 90.0);

        let ray = Ray::new(
            Point3f::new(2.0, 0.0, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
            FloatRT::infinity(),
            0.0,
        );
        let hit = sphere.intersect(ray);
        assert!(hit.is_some());

        let ray = Ray::new(
            Point3f::new(-2.0, 0.0, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
            FloatRT::infinity(),
            0.0,
        );
        let hit = sphere.intersect(ray);
        assert!(hit.is_none());

        // Pass over clipped z
                let ray = Ray::new(
            Point3f::new(1.0, 3.5, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
            FloatRT::infinity(),
            0.0,
        );
        let hit = sphere.intersect(ray);
        assert!(hit.is_none());
    }

    #[test]
    fn partial_derivatives() {
        let sphere = Sphere::new(Transform::identity(), Transform::identity(), false, 4.0, -4.0, 4.0, 360.0);
        let ((dpdu, dpdv), (dndu, dndv)) = sphere.partial_derivatives(Point3f::new(4.0, 0.0, 0.0));

        assert_relative_eq!(dpdu.normalize(), Vec3f::new(0.0, 1.0, 0.0));
        assert_relative_eq!(dpdv.normalize(), Vec3f::new(0.0, 0.0, 1.0));

    }
}
