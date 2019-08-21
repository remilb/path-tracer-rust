use std::fs;

use crate::image::Image;

pub fn write_ppm(path: &str, img: Image) -> std::io::Result<()> {
    let header = format!("P3\n{} {}\n255\n", img.w(), img.h());
    let data: String = img
        .buf
        .iter()
        .flat_map(|color| {
            vec![
                color_float_to_u8(color.x).to_string(),
                color_float_to_u8(color.y).to_string(),
                color_float_to_u8(color.z).to_string(),
            ]
        })
        .collect::<Vec<String>>()
        .join(" ");
    let ppm = format!("{}{}", header, data);
    fs::write(path, ppm)?;
    Ok(())
}

fn color_float_to_u8(c: f32) -> u8 {
    (c * 255.99) as u8
}
