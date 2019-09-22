use crate::geometry::{Bounds3f, FloatRT, SurfaceInteraction, Ray, Transform, Transformer};

mod sphere;

// TODO: Borrow Transforms for use with a Transform pool
struct ShapeBaseData {
    object_to_world: Transform,
    world_to_object: Transform,
    reverse_orientation: bool,
    transform_swaps_handedness: bool,
}

pub trait Shape {
    fn transform_swaps_handedness(&self) -> bool;
    fn orientation_is_reversed(&self) -> bool;
    fn object_bound(&self) -> Bounds3f;
    fn object_to_world(&self) -> &Transform;

    /// Allows shape to be used as an area light
    fn surface_area(&self) -> FloatRT;

    /// Intersect methods should return the time of the closest intersection point as well as the resulting Interaction
    fn intersect(&self, ray: Ray) -> Option<(FloatRT, SurfaceInteraction)>;

    /// This method should be overridden if the shape knows how to provide a tighter transformed world bound
    fn world_bound(&self) -> Bounds3f {
        self.object_to_world().apply(self.object_bound())
    }

    /// Shapes that can should override this method with a more efficient one
    fn intersect_test(&self, ray: Ray) -> bool {
        self.intersect(ray).is_some()
    }

    /// Used to let Interactions know whether or not they should flip their computed normal
    fn should_flip_normal(&self) -> bool {
        self.orientation_is_reversed() ^ self.transform_swaps_handedness()
    }
}
