// src/models/glm_ocr/input.rs
//! Image + Prompt processor for GLM-OCR.
//! Resizes, normalizes, and extracts 3D patches from images.
use crate::utils::image::{
    to_tensor, ImageProcessConfig, ImageProcessTrait, ToFilter,
};
use crate::utils::image::{IMAGE_PLACEHOLDER, PLACEHOLDER};
use candle_core::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};

/// Image processor for GLM-OCR.
/// Follows HF transformers' GLM-OCR preprocessing:
/// 1. Smart resize to factor-aligned dimensions
/// 2. Normalize (mean/std)
/// 3. Extract 3D patches (temporal × spatial)
#[derive(Clone)]
pub struct GlmOcrImageProcessor {
    cfg: ImageProcessConfig,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    min_pixels: usize,
    max_pixels: usize,
    fixed_width: Option<usize>,
    fixed_height: Option<usize>,
}

impl GlmOcrImageProcessor {
    pub const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    pub const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    pub const VISION_START: &str = "<|begin_of_image|>";
    pub const VISION_END: &str = "<|end_of_image|>";
    pub const IMAGE_PAD: &str = "<|image_pad|>";

    pub fn default(cfg: &ImageProcessConfig) -> Self {
        let max_row = std::cmp::max(cfg.max_height, cfg.max_width);
        Self {
            cfg: cfg.clone(),
            patch_size: cfg.patch_size,
            merge_size: cfg.spatial_merge_size,
            temporal_patch_size: cfg.temporal_patch_size.unwrap_or(2),
            min_pixels: 56 * 56,
            max_pixels: max_row * max_row,
            fixed_width: None,
            fixed_height: None,
        }
    }

    /// Resize the image to dimensions divisible by factor (patch_size * merge_size).
    /// Guarantees nh >= factor and nw >= factor (at least 1 grid cell).
    fn smart_resize(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let factor = self.patch_size * self.merge_size;
        // Use ceil to always round up, and enforce minimum of factor
        let mut nh = ((h as f64 / factor as f64).ceil() as usize * factor).max(factor);
        let mut nw = ((w as f64 / factor as f64).ceil() as usize * factor).max(factor);

        let pixels = nh * nw;
        // Scale down if too large, scale up if too small (always preserving >= factor)
        if pixels > self.max_pixels {
            let beta = (pixels as f64 / self.max_pixels as f64).sqrt();
            nh = (((nh as f64 / beta) as usize / factor) * factor).max(factor);
            nw = (((nw as f64 / beta) as usize / factor) * factor).max(factor);
            eprintln!("[GLM-OCR smart_resize] scaled down: nh={}, nw={}", nh, nw);
        } else if pixels < self.min_pixels {
            let beta = (self.min_pixels as f64 / pixels as f64).sqrt();
            nh = (((nh as f64 * beta) as usize / factor) * factor).max(factor);
            nw = (((nw as f64 * beta) as usize / factor) * factor).max(factor);
            eprintln!("[GLM-OCR smart_resize] scaled up: nh={}, nw={}", nh, nw);
        }

        Ok((nh, nw))
    }

    /// Preprocess a single image and extract patches.
    /// Returns (flattened_patches [N, C*T*pH*pW], (grid_h, grid_w)).
    fn prepreprocess(&mut self, image: &DynamicImage, target_hw: (u32, u32)) -> Result<(Tensor, (usize, usize))> {
        let (th, tw) = target_hw;
        eprintln!("[GLM-OCR prepreprocess] target_hw=({}, {}), fixed_hw=({:?}, {:?})", th, tw, self.fixed_height, self.fixed_width);

        // Smart resize
        let (mut nh, mut nw) = self.smart_resize(th as usize, tw as usize)?;
        eprintln!("[GLM-OCR prepreprocess] after smart_resize: nh={}, nw={}", nh, nw);

        // Use fixed size if already set (for batch consistency)
        if let (Some(h), Some(w)) = (self.fixed_height, self.fixed_width) {
            nh = h;
            nw = w;
            eprintln!("[GLM-OCR prepreprocess] using cached fixed: nh={}, nw={}", nh, nw);
        } else {
            self.fixed_height = Some(nh);
            self.fixed_width = Some(nw);
        }

        // Ensure at least 1 grid cell in each dimension
        if nh < self.patch_size {
            nh = self.patch_size;
        }
        if nw < self.patch_size {
            nw = self.patch_size;
        }
        eprintln!("[GLM-OCR prepreprocess] final: nh={}, nw={}", nh, nw);

        // Resize and convert to RGB
        let image = image
            .resize_exact(nw as u32, nh as u32, image::imageops::FilterType::Triangle)
            .to_rgb8();

        let image_mean = Some(self.cfg.image_mean.unwrap_or(Self::DEFAULT_MEAN));
        let image_std = Some(self.cfg.image_std.unwrap_or(Self::DEFAULT_STD));

        // Normalize to tensor [C, H, W]
        let (mut patches, _) = to_tensor(
            &vec![DynamicImage::ImageRgb8(image)],
            image_mean,
            image_std,
        )?;
        eprintln!("[GLM-OCR prepreprocess] to_tensor output shape: {:?}", patches.dims());

        // For a single image, duplicate to match temporal_patch_size (T=1 -> T=2)
        if patches.dim(0)? == 1 {
            patches = patches.repeat((self.temporal_patch_size, 1, 1, 1))?;
        }

        let c = patches.dim(1)?;
        let grid_t = patches.dim(0)? / self.temporal_patch_size;
        let grid_h = nh / self.patch_size;
        let grid_w = nw / self.patch_size;
        eprintln!("[GLM-OCR prepreprocess] c={}, grid_t={}, grid_h={}, grid_w={}, num_patches={}", c, grid_t, grid_h, grid_w, grid_t * grid_h * grid_w);

        // Reshape [T, C, H, W] -> [grid_t, T, C, grid_h/merge, merge, pH, grid_w/merge, merge, pW]
        patches = patches.reshape(&[
            grid_t,
            self.temporal_patch_size,
            c,
            grid_h / self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w / self.merge_size,
            self.merge_size,
            self.patch_size,
        ])?;

        // Permute -> [grid_t, grid_h/merge, grid_w/merge, merge, merge, C, T, pH, pW]
        patches = patches.permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?;

        // Flatten -> [grid_t * grid_h * grid_w, C * T * pH * pW]
        let patches = patches.reshape((
            grid_t * grid_h * grid_w,
            c * self.temporal_patch_size * self.patch_size * self.patch_size,
        ))?;

        Ok((patches, (grid_h, grid_w)))
    }
}

impl ImageProcessTrait for GlmOcrImageProcessor {
    fn process_inputs(
        &mut self,
        prompt: &mut String,
        images: &Vec<DynamicImage>,
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        // Find max dimensions for batch consistency
        let (max_w, max_h) = images
            .iter()
            .map(|i| i.dimensions())
            .fold((0, 0), |(mw, mh), (w, h)| (mw.max(w), mh.max(mh)));

        let mut pixel_values = Vec::new();
        let mut grid_thw = Vec::new();

        for image in images {
            let (patches, (h, w)) = self.prepreprocess(image, (max_h, max_w))?;
            pixel_values.push(patches);
            grid_thw.push((h, w));
        }

        // Stack patches: [num_images, num_patches, C*T*pH*pW]
        let pixel_values = Tensor::stack(&pixel_values, 0)?;

        // Prompt expansion
        let merge_len = self.merge_size * self.merge_size;
        let mut image_idx = 0;
        let mut replace_strings = Vec::new();

        while prompt.contains(IMAGE_PLACEHOLDER) {
            let grid = grid_thw[image_idx];
            let num_patches: usize = (grid.0 * grid.1) as usize / merge_len;
            let mut replace_tokens = vec![Self::VISION_START];
            replace_tokens.extend(vec![Self::IMAGE_PAD; num_patches]);
            replace_tokens.push(Self::VISION_END);

            replace_strings.push(replace_tokens.join(""));
            *prompt = prompt.replace(IMAGE_PLACEHOLDER, PLACEHOLDER);
            image_idx += 1;
        }

        while prompt.contains(PLACEHOLDER) {
            if let Some(replace_str) = replace_strings.pop() {
                *prompt = prompt.replace(PLACEHOLDER, &replace_str);
            } else {
                break;
            }
        }

        Ok((pixel_values, grid_thw))
    }
}
