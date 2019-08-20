use std::fs;

use crate::image::Image;

pub fn write_ppm(path: &str, img: Image) -> std::io::Result<()> {
    let header = format!("P3\n{} {}\n255\n", img.w(), img.h());
    let data: String = img
        .buf
        .iter()
        .map(|num| num.to_string())
        .collect::<Vec<String>>()
        .join(" ");
    let ppm = format!("{}{}", header, data);
    fs::write(path, ppm)?;
    Ok(())
}
