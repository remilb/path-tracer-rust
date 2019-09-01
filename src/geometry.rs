mod normal;
mod point;
mod vector;

use num_traits::{NumCast, Signed};

type FloatRT = f32;

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