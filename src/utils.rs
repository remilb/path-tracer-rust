type FloatRT = f32;

pub mod math {
    use super::*;
    pub fn lerp(t: FloatRT, v1: FloatRT, v2: FloatRT) -> FloatRT {
        (1.0 - t) * v1 + t * v2
    }

    /// Solve a quadratic equation of for ax^2 + bx + c = 0. If solution exists, return a tuple
    /// containing the two solutions or, if only one solution, that solution in both tuple elements
    pub fn quadratic(a: FloatRT, b: FloatRT, c: FloatRT) -> Option<(FloatRT, FloatRT)> {
        let discrim = f64::from(b) * f64::from(b) - 4.0 * f64::from(a) * f64::from(c);
        if discrim < 0.0 {
            return None;
            }      
        let root_discrim = discrim.sqrt();
        let q = if b < 0.0 {-0.5 * (f64::from(b) - root_discrim)} else {-0.5 * (f64::from(b) + root_discrim)};
        let (t0, t1) = (q / f64::from(a), f64::from(c) / q);
        if t0 > t1 {return Some((t1 as FloatRT, t0 as FloatRT));} else {return Some((t0 as FloatRT, t1 as FloatRT));}
    }
}
