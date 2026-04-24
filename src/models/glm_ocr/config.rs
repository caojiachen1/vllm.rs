// src/models/glm_ocr/config.rs
use crate::{serde_default, utils::config::Config};

serde_default!(usize, default_vision_hidden_size, 1024);
serde_default!(usize, default_vision_depth, 24);
serde_default!(usize, default_vision_num_heads, 16);
serde_default!(usize, default_vision_intermediate_size, 4096);
serde_default!(usize, default_vision_out_hidden_size, 1536);
// merge_intermediate_size = 4 * out_hidden_size by default (1536 * 4 = 6144)
serde_default!(usize, default_merge_intermediate_size, 6144);
serde_default!(usize, default_vision_patch_size, 14);
serde_default!(usize, default_vision_spatial_merge_size, 2);
serde_default!(usize, default_vision_temporal_patch_size, 2);
serde_default!(usize, default_vision_in_channels, 3);
serde_default!(bool, default_attention_bias, true);
fn default_vision_hidden_act() -> candle_nn::Activation {
    candle_nn::Activation::Silu
}
serde_default!(u32, default_image_token_id, 151859);
serde_default!(u32, default_image_start_token_id, 151857);
serde_default!(u32, default_image_end_token_id, 151858);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GlmOcrVisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_depth")]
    pub depth: usize,
    #[serde(default = "default_vision_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_out_hidden_size")]
    pub out_hidden_size: usize,
    /// Intermediate size for the patch merger gate MLP
    #[serde(default = "default_merge_intermediate_size")]
    pub merge_intermediate_size: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_vision_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_vision_in_channels")]
    pub in_channels: usize,
    pub rms_norm_eps: Option<f64>,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: candle_nn::Activation,
}

impl GlmOcrVisionConfig {
    pub fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps.unwrap_or(1e-5)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GlmOcrConfig {
    pub architectures: Option<Vec<String>>,
    pub text_config: Config,
    pub vision_config: GlmOcrVisionConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_image_start_token_id")]
    pub image_start_token_id: u32,
    #[serde(default = "default_image_end_token_id")]
    pub image_end_token_id: u32,
}
