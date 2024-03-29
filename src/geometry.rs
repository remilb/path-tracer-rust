mod bounds;
mod interaction;
mod matrix;
mod normal;
mod point;
mod ray;
mod transform;
mod vector;

use num_traits::{NumCast, Signed, ToPrimitive};

pub type FloatRT = f32;
pub const INFINITY: FloatRT = std::f32::INFINITY;

// Reexports
pub use bounds::{Bounds2, Bounds2f, Bounds2i, Bounds3, Bounds3f, Bounds3i};
pub use interaction::{InteractionType, SurfaceInteraction};
pub use normal::{Normal3, Normal3f, Normal3i};
pub use point::{Point2, Point2f, Point2i, Point3, Point3f, Point3i};
pub use ray::Ray;
pub use transform::{Transform, Transformer};
pub use vector::{Vec2, Vec2f, Vec2i, Vec3, Vec3f, Vec3i};

// Dot product trait allows for dot between various types (Vec3, Normal3)
pub trait Dot<RHS = Self> {
    type Output;
    fn dot(v1: Self, v2: RHS) -> Self::Output;
    fn abs_dot(v1: Self, v2: RHS) -> Self::Output;
    fn face_forward(self, vs: RHS) -> Self;
}

// Cross product trait allows for dot between various types (Vec3, Normal3)
pub trait Cross<LHS = Self, RHS = Self> {
    fn cross(v1: LHS, v2: RHS) -> Self;
}

// Want to unify integer and floats in a single bound for generics
// Mostly for NaN checks at the moment
// TODO: Look for a more elegant way to deal with all these float/int incongruities
// These vector classes are meant to be super generic, and are pretty limited in scope,
// so this should be fine for now
pub trait Scalar: Signed + NumCast + PartialOrd + Copy {
    fn is_finite(self) -> bool;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        NumCast::from(n)
    }
}

//Foreign implementations
impl Scalar for f32 {
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    fn floor(self) -> Self {
        self.floor()
    }
    fn ceil(self) -> Self {
        self.ceil()
    }
}

impl Scalar for f64 {
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    fn floor(self) -> Self {
        self.floor()
    }
    fn ceil(self) -> Self {
        self.ceil()
    }
}

impl Scalar for i32 {
    fn is_finite(self) -> bool {
        true
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }

    fn floor(self) -> Self {
        self
    }
    fn ceil(self) -> Self {
        self
    }
}

impl Scalar for i16 {
    fn is_finite(self) -> bool {
        true
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn floor(self) -> Self {
        self
    }
    fn ceil(self) -> Self {
        self
    }
}

impl Scalar for i8 {
    fn is_finite(self) -> bool {
        true
    }
    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
    fn floor(self) -> Self {
        self
    }
    fn ceil(self) -> Self {
        self
    }
}
