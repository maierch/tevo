//! Copyright (c) 2026 Christian Maier
//! SPDX-License-Identifier: MIT
//! Minimal image and color map utilities for measurement output.

use std::io::{BufWriter, Write};
use std::path::Path;

const SVG_HEADER: &'static [u8] = "<?xml version=\"1.0\" standalone=\"no\"?>\n".as_bytes();

const MAX_IMAGE_WIDTH: usize = u32::MAX as usize;

/// Pixel color with 3 channels and 8-bit color depth.
#[derive(Clone, Copy)]
pub struct ColorRGB8 {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
}

/// Converts a 64-bit floating point value into a valid byte.
fn quantize(x: f64) -> u8 {
    let i = (256.0 * x).floor();
    if i <= 0.0 {
        0
    } else if i >= 255.0 {
        255
    } else {
        i as u8
    }
}

impl ColorRGB8 {
    /// Creates a pixel color from floating point values.
    pub fn from_f64(r: f64, g: f64, b: f64) -> Self {
        let r = quantize(r);
        let g = quantize(g);
        let b = quantize(b);
        ColorRGB8 { r, g, b }
    }
}

/// A very simple raster image.
pub struct Image {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Image {
    /// Creates a new image.
    pub fn new(width: usize, height: usize) -> Self {
        let data = vec![0; 3 * width * height];
        if width > MAX_IMAGE_WIDTH || height > MAX_IMAGE_WIDTH {
            panic!("width > MAX_IMAGE_WIDTH || height > MAX_IMAGE_WIDTH");
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Sets the color at a given position.
    pub fn set_pixel(&mut self, x: usize, y: usize, color: ColorRGB8) {
        let offset = 3 * (x + self.width * y);
        self.data[offset] = color.r;
        self.data[offset + 1] = color.g;
        self.data[offset + 2] = color.b;
    }

    /// Writes the image to a PPM file.
    // pub fn write_ppm_file(&self, path: &Path) -> Result<(), std::io::Error> {
    //     let w = self.width;
    //     let h = self.height;
    //     let header = format!("P6\n{w} {h}\n255\n");
    //     let mut file = std::fs::File::create(path)?;
    //     file.write(header.as_bytes())?;
    //     file.write(&self.data)?;
    //     Ok(())
    // }

    /// Writes the image to an SVG file.
    pub fn write_svg_file(&self, path: &Path) -> Result<(), std::io::Error> {
        let mut out = BufWriter::new(std::fs::File::create(path)?);
        let width = self.width;
        let height = self.height;
        let data = &self.data;
        out.write(SVG_HEADER)?;
        write!(out, "<svg viewBox=\"0 0 {width} {height}\"")?;
        write!(out, " version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\"")?;
        write!(out, " xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n")?;
        let mut src_pos = 0;
        for y in 0..height {
            for x in 0..width {
                let r = data[src_pos];
                src_pos += 1;
                let g = data[src_pos];
                src_pos += 1;
                let b = data[src_pos];
                src_pos += 1;
                write!(out, "<rect x=\"{x}\" y=\"{y}\" width=\"1\" height=\"1\"")?;
                write!(out, " fill=\"#{r:02x}{g:02x}{b:02x}\"/>\n")?;
            }
        }
        write!(out, "</svg>\n")?;
        Ok(())
    }

    /// Writes the image to a PNG file.
    pub fn write_png_file(&self, path: &Path) -> Result<(), std::io::Error> {
        let file = std::fs::File::create(path)?;
        let ref mut w = BufWriter::new(file);
        let width: u32 = self.width.try_into().unwrap();
        let height: u32 = self.height.try_into().unwrap();
        let mut encoder = png::Encoder::new(w, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));
        let source_chromaticities = png::SourceChromaticities::new(
            (0.31270, 0.32900),
            (0.64000, 0.33000),
            (0.30000, 0.60000),
            (0.15000, 0.06000),
        );
        encoder.set_source_chromaticities(source_chromaticities);
        encoder.set_compression(png::Compression::Best);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&self.data)?;
        Ok(())
    }

    /// Returns a nearest-neighbor upscaled copy of the image.
    pub fn upscale(self, scale: usize) -> Self {
        let width = self.width;
        let height = self.height;
        let data = self.data;
        let mut image = Image::new(width * scale, height * scale);
        let mut src_pos = 0;
        for y in 0..height {
            for x in 0..width {
                let r = data[src_pos];
                src_pos += 1;
                let g = data[src_pos];
                src_pos += 1;
                let b = data[src_pos];
                src_pos += 1;
                let color = ColorRGB8 { r, g, b };
                for sy in 0..scale {
                    let new_y = y * scale + sy;
                    for sx in 0..scale {
                        let new_x = x * scale + sx;
                        image.set_pixel(new_x, new_y, color)
                    }
                }
            }
        }
        image
    }
}

/// Maps numbers to colors.
pub struct ColorMap {
    steps: Vec<(f64, f64, f64, f64)>,
}

#[inline]
fn color_from_step(step: &(f64, f64, f64, f64)) -> ColorRGB8 {
    ColorRGB8::from_f64(step.1, step.2, step.3)
}

#[inline]
fn lerp(t: f64, x1: f64, x2: f64) -> f64 {
    x1 + t * (x2 - x1)
}

fn interpolate_color_from_steps(
    t: f64,
    step1: &(f64, f64, f64, f64),
    step2: &(f64, f64, f64, f64),
) -> ColorRGB8 {
    let r = quantize(lerp(t, step1.1, step2.1));
    let g = quantize(lerp(t, step1.2, step2.2));
    let b = quantize(lerp(t, step1.3, step2.3));
    ColorRGB8 { r, g, b }
}

impl ColorMap {
    /// Creates a new color map from the given color steps.
    pub fn new(steps: Vec<(f64, f64, f64, f64)>) -> Self {
        let mut steps = steps;
        if steps.len() < 1 {
            steps.push((0.0, 0.0, 0.0, 0.0));
        }
        Self { steps }
    }

    /// Gets the color for a given number.
    pub fn get_color(&self, value: f64) -> ColorRGB8 {
        if !value.is_finite() {
            panic!("get_color: got non-finite value: {value}");
        }
        let first = self.steps[0];
        if value <= first.0 {
            return color_from_step(&first);
        }
        let last = self.steps[self.steps.len() - 1];
        if value >= last.0 {
            return color_from_step(&last);
        }
        let index = match self.steps.binary_search_by(|t| t.0.total_cmp(&value)) {
            Err(idx) => idx - 1,
            Ok(idx) => idx,
        };
        let step1 = self.steps[index];
        let step2 = self.steps[index + 1];
        let t = (value - step1.0) / (step2.0 - step1.0);
        interpolate_color_from_steps(t, &step1, &step2)
    }

    /// Creates the rainbow color map.
    pub fn rainbow() -> Self {
        Self::new(vec![
            (0.000, 1.0, 1.0, 1.0), // white
            (0.125, 0.5, 0.5, 0.7), // dark-blue 1
            (0.250, 0.0, 0.0, 0.5), // dark-blue 2
            (0.375, 0.0, 0.0, 1.0), // blue
            (0.500, 0.0, 1.0, 1.0), // cyan
            (0.625, 0.0, 1.0, 0.0), // green
            (0.750, 1.0, 1.0, 0.0), // orange
            (0.875, 1.0, 0.0, 0.0), // red
            (1.000, 0.5, 0.0, 0.0), // dark-red
        ])
    }
}

// Test for writing images
mod test {
    #[test]
    fn write_image_files() {
        use super::{ColorMap, Image};
        let rainbow = ColorMap::rainbow();
        let mut img = Image::new(30, 10);
        for x in 0..30 {
            for y in 0..10 {
                let value = (x as f64 / 30.0) * (y as f64 / 10.0);
                img.set_pixel(x, y, rainbow.get_color(value));
            }
        }
        let svg_path = std::path::Path::new("test-output.svg");
        let png_path = std::path::Path::new("test-output.png");
        img.write_svg_file(svg_path).unwrap();
        img.upscale(8).write_png_file(png_path).unwrap();
    }
}
