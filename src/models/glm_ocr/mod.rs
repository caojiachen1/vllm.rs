// src/models/glm_ocr/mod.rs
//! GLM-OCR: GlmOcrForConditionalGeneration
//! A 0.9B multimodal OCR model combining CogViT vision encoder (0.4B) + GLM text decoder (0.5B).
//! Reference: https://huggingface.co/zai-org/GLM-OCR

pub mod config;
pub mod input;
pub mod vision;

use crate::models::glm4::GLM4ForCausalLM;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::image::ImageData;
use crate::utils::progress::ProgressLike;
use crate::{models::layers::distributed::Comm, utils::config::RopeScalingValue};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use config::GlmOcrConfig;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use vision::GlmOcrVisionModel;

// ---------------------------------------------------------------------------
// Config parsing helpers
// ---------------------------------------------------------------------------

fn try_parse_glm_ocr_config(config: &Config) -> Option<GlmOcrConfig> {
    let raw = config.extra_config_json.as_ref()?;
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    if v.get("vision_config").is_none() {
        return None;
    }
    serde_json::from_value(v).ok()
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct GlmOcrForConditionalGeneration {
    text_model: GLM4ForCausalLM,
    vision_model: Option<GlmOcrVisionModel>,
    spatial_merge_size: usize,
    image_token_id: Option<u32>,
}

impl GlmOcrForConditionalGeneration {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let mut text_config = config.clone();
        let mut vision_model = None;
        let mut spatial_merge_size = 2usize;
        let mut image_token_id = None;

        if let Some(cfg) = try_parse_glm_ocr_config(config) {
            spatial_merge_size = cfg.vision_config.spatial_merge_size;
            image_token_id = Some(cfg.image_token_id);

            // Build text config from the nested text_config + rope_parameters
            text_config = cfg.text_config.clone();
            if let Ok(raw_json) =
                serde_json::from_str::<serde_json::Value>(config.extra_config_json.as_deref().unwrap_or("{}"))
            {
                let tc = &raw_json["text_config"];
                if let Some(rp) = tc.get("rope_parameters") {
                    if let Some(theta) = rp.get("rope_theta").and_then(|v| v.as_f64()) {
                        text_config.rope_theta = Some(theta);
                    }
                    if let Some(section) = rp.get("mrope_section") {
                        if let Ok(sec) = serde_json::from_value::<Vec<f64>>(section.clone()) {
                            let mut map = HashMap::new();
                            map.insert(
                                "mrope_section".to_string(),
                                RopeScalingValue::NumberArray(sec),
                            );
                            map.insert(
                                "rope_type".to_string(),
                                RopeScalingValue::String("default".to_string()),
                            );
                            text_config.rope_scaling = Some(map);
                        }
                    }
                }
                if let Some(eos) = tc.get("eos_token_id") {
                    text_config.eos_token_id = serde_json::from_value(eos.clone()).ok();
                }
            }

            // Load vision encoder from model.visual.*
            crate::log_info!("Loading GLM-OCR vision tower...");
            match GlmOcrVisionModel::new(&cfg.vision_config, &vb.pp("model.visual"), dtype, device)
            {
                Ok(vm) => vision_model = Some(vm),
                Err(e) => {
                    crate::log_error!(
                        "Failed to load GLM-OCR vision tower: {}. Running text-only.",
                        e
                    );
                }
            }
        }

        crate::log_info!("Loading GLM-OCR language model...");
        // Language model weights live under "model.language_model.*"
        // lm_head is at the top level.
        let text_model = GLM4ForCausalLM::new_with_prefix(
            vb,
            comm,
            &text_config,
            dtype,
            is_rope_i,
            device,
            progress_reporter,
            Some("model.language_model.".to_string()),
        )?;

        Ok(Self {
            text_model,
            vision_model,
            spatial_merge_size,
            image_token_id,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        let mut input_embeds = self.text_model.embed_forward(input_ids)?;
        let device = input_embeds.device().clone();

        if let Some(images) = images {
            let vision_model = match self.vision_model.as_ref() {
                Some(vm) => vm,
                None => {
                    crate::log_warn!("GLM-OCR: ignoring image inputs, vision tower not loaded.");
                    return self.text_model.forward(
                        &input_embeds,
                        positions,
                        kv_caches,
                        input_metadata,
                        true,
                    );
                }
            };
            let image_token_id = match self.image_token_id {
                Some(id) => id,
                None => {
                    crate::log_warn!("GLM-OCR: ignoring image inputs, image_token_id not set.");
                    return self.text_model.forward(
                        &input_embeds,
                        positions,
                        kv_caches,
                        input_metadata,
                        true,
                    );
                }
            };

            let dtype = input_embeds.dtype();

            // pixel_values: [N_images, C * T * pH * pW] already flat
            let pixel_values = images.to_tensor_f32(&device)?.to_dtype(dtype)?;

            // grid_thw: [N_images, 3] with T=1
            let grid_thw = {
                let num_images = images.patches.len();
                let mut data: Vec<u32> = Vec::with_capacity(num_images * 3);
                for &(h, w) in &images.patches {
                    data.push(1); // T
                    data.push(h as u32);
                    data.push(w as u32);
                }
                Tensor::from_vec(data, (num_images, 3), &device)?
            };

            // Handle batching offset (prefill may cover a subset of images)
            let num_images = images.patches.len();
            let (pixel_values, grid_thw) = if images.image_idx > 0 {
                let start = (images.image_idx as usize).min(num_images);
                let len = num_images - start;
                if len == 0 {
                    return self.text_model.forward(
                        &input_embeds,
                        positions,
                        kv_caches,
                        input_metadata,
                        true,
                    );
                }
                (
                    pixel_values.narrow(0, start, len)?,
                    grid_thw.narrow(0, start, len)?,
                )
            } else {
                (pixel_values, grid_thw)
            };

            let image_embeds = vision_model.forward(&pixel_values, &grid_thw)?;
            let image_embeds = image_embeds.to_device(&device)?.to_dtype(dtype)?;

            // Scatter image embeddings into positions marked with image_token_id
            let image_mask = input_ids.eq(image_token_id)?;
            let image_mask_expanded = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;
            use attention_rs::ops::NonZeroOp;
            let indices = image_mask_expanded.flatten_all()?.nonzero()?.squeeze(1)?;

            if indices.shape().dim(0)? > 0 {
                let hidden = input_embeds.dim(D::Minus1)?;
                let indices_len = indices.shape().dim(0)?;
                if indices_len % hidden != 0 {
                    candle_core::bail!(
                        "GLM-OCR: image indices {} not divisible by hidden {}",
                        indices_len,
                        hidden
                    );
                }
                let tokens_in_chunk = indices_len / hidden;
                let total_tokens = image_embeds.dim(0)?;
                let start = images.image_token_offset.min(total_tokens);
                let end = start + tokens_in_chunk;
                if end > total_tokens {
                    candle_core::bail!(
                        "GLM-OCR: image token slice out of range: start {}, len {}, total {}",
                        start,
                        tokens_in_chunk,
                        total_tokens
                    );
                }
                let image_embeds_slice = if start > 0 || end < total_tokens {
                    image_embeds.narrow(0, start, tokens_in_chunk)?
                } else {
                    image_embeds
                };

                let mut x_flat = input_embeds.flatten_all()?;
                let img_flat = image_embeds_slice.flatten_all()?;
                x_flat = x_flat.scatter_add(
                    &indices,
                    &(img_flat - x_flat.gather(&indices, 0)?)?,
                    0,
                )?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        self.text_model.forward(
            &input_embeds,
            positions,
            kv_caches,
            input_metadata,
            true, // already embedded
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        _embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.text_model
            .forward_embedding(input_ids, positions, kv_caches, input_metadata, false)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.text_model.get_vocab_size()
    }
}
