type FloatRT = f32;

pub mod math {
    use super::*;
    pub fn lerp(t: FloatRT, v1: FloatRT, v2: FloatRT) -> FloatRT {
        (1.0 - t) * v1 + t * v2
    } 
}