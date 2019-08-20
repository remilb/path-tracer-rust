use std::convert::TryInto;

pub struct Image {
    w: u32,
    h: u32,
    //TODO: Get rid of this pub
    pub buf: Vec<u8>,
}

impl Image {
    pub fn new(w: u32, h: u32) -> Image {
        Image {
            w,
            h,
            buf: vec![0; (w * h * 3).try_into().unwrap()],
        }
    }

    pub fn w(&self) -> u32 {
        self.w
    }

    pub fn h(&self) -> u32 {
        self.h
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8) {
        let index = self.get_index(x, y);
        self.buf[index] = r;
        self.buf[index + 1] = g;
        self.buf[index + 2] = b;
    }

    fn get_index(&self, x: u32, y: u32) -> usize {
        (self.w * 3 * y + x * 3).try_into().unwrap()
    }

    pub fn new_test_image(w: u32, h: u32) -> Image {
        let mut buf = vec![];
        for y in 0..h {
            for x in 0..w {
                let r = y as f32 / h as f32;
                let g = x as f32 / w as f32;
                let b = 0.2;
                buf.push((255.99 * r) as u8);
                buf.push((255.99 * g) as u8);
                buf.push((255.99 * b) as u8);
            }
        }
        Image { w, h, buf }
    }
}
