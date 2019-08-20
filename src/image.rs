pub struct Image {
    pub w: u32,
    pub h: u32,
    pub buf: Vec<u8>,
}

impl Image {
    pub fn new() -> Image {
        Image {
            w: 0,
            h: 0,
            buf: vec![],
        }
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
