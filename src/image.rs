use crate::vec3::Vec3;
use std::convert::TryInto;

//TODO: Get rid of pub and decide if color type rather than Vec3 is appropriate
pub struct Image {
    w: u32,
    h: u32,
    pub buf: Vec<Vec3>,
}

impl Image {
    pub fn new(w: u32, h: u32) -> Image {
        Image {
            w,
            h,
            buf: vec![Vec3::new(0., 0., 0.); (w * h).try_into().unwrap()],
        }
    }

    pub fn w(&self) -> u32 {
        self.w
    }

    pub fn h(&self) -> u32 {
        self.h
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, color: Vec3) {
        let index = self.get_index(x, y);
        self.buf[index] = color;
    }

    fn get_index(&self, x: u32, y: u32) -> usize {
        (self.w * y + x).try_into().unwrap()
    }

    // pub fn new_test_image(w: u32, h: u32) -> Image {
    //     let mut buf = vec![];
    //     for y in 0..h {
    //         for x in 0..w {
    //             let r = y as f32 / h as f32;
    //             let g = x as f32 / w as f32;
    //             let b = 0.2;
    //             buf.push((255.99 * r) as u8);
    //         }
    //     }
    //     Image { w, h, buf }
    // }
}
