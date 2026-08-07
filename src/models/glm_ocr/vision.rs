// src/models/glm_ocr/vision.rs
//! CogViT vision encoder for GLM-OCR.
//! Implements GlmOcrVisionModel matching HuggingFace transformers modeling_glm_ocr.py.
use super::config::GlmOcrVisionConfig;
use crate::models::layers::{
    distributed::ReplicatedLinear,
    others::{layer_norm, rms_norm, Conv3dConfig, Conv3dNoBias, NormX},
    VarBuilderX,
};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Module;
use either::Either;

// ---------------------------------------------------------------------------
// Patch embedding – Conv3d: kernel=[T, pH, pW], stride=[T, pH, pW]
// ---------------------------------------------------------------------------
struct PatchEmbed {
    proj: Conv3dNoBias,
    in_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilderX) -> Result<Self> {
        let proj = Conv3dNoBias::new(
            cfg.in_channels,
            cfg.hidden_size,
            [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size],
            Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;
        Ok(Self {
            proj,
            in_channels: cfg.in_channels,
            temporal_patch_size: cfg.temporal_patch_size,
            patch_size: cfg.patch_size,
            hidden_size: cfg.hidden_size,
        })
    }

    /// hidden_states: [total_patches, in_ch * temporal_patch_size * patch_size^2]
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let n = hidden_states.dim(0)?;
        // Reshape: [N, in_ch, temporal_patch_size, patch_size, patch_size]
        let xs = hidden_states
            .reshape((
                n,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ))?
            .to_dtype(DType::BF16)?;
        // Conv3dNoBias: splits at dim=2 (temporal), applies two Conv2d, sums, unsqueezes(2)
        // Output: [N, hidden_size, 1, 1, 1]
        let out = self.proj.forward(&xs)?;
        out.reshape((n, self.hidden_size))
    }
}

// ---------------------------------------------------------------------------
// Rotary position embedding for vision
// ---------------------------------------------------------------------------
struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    fn new(half_head_dim: usize, device: &Device) -> Result<Self> {
        // inv_freq[i] = 1 / (10000 ^ (2i / dim))
        // dim = half_head_dim * 2 = head_dim
        let dim = half_head_dim * 2;
        let inv_freq: Vec<f32> = (0..half_head_dim)
            .map(|i| 1.0f32 / 10000_f32.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (len,), device)?,
        })
    }

    /// Returns freqs table [seqlen, half_head_dim]
    fn forward(&self, seqlen: usize) -> Result<Tensor> {
        let device = self.inv_freq.device();
        let seq = Tensor::arange(0f32, seqlen as f32, device)?;
        seq.unsqueeze(1)?.broadcast_mul(&self.inv_freq.unsqueeze(0)?)
    }
}

// ---------------------------------------------------------------------------
// Rotary helpers (vision – uses `rotate_half` where first half negated)
// ---------------------------------------------------------------------------
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, last - half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // cos/sin: [seq, head_dim]; q/k: [seq, num_heads, head_dim]
    let cos = cos.unsqueeze(D::Minus2)?.to_dtype(DType::F32)?; // [seq, 1, head_dim]
    let sin = sin.unsqueeze(D::Minus2)?.to_dtype(DType::F32)?;
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

// ---------------------------------------------------------------------------
// Vision Attention
// ---------------------------------------------------------------------------
struct VisionAttention {
    qkv: ReplicatedLinear,
    q_norm: NormX,
    k_norm: NormX,
    proj: ReplicatedLinear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilderX) -> Result<Self> {
        let dim = cfg.hidden_size;
        let head_dim = dim / cfg.num_heads;
        let eps = cfg.rms_norm_eps();
        Ok(Self {
            qkv: ReplicatedLinear::load_b(
                dim,
                dim * 3,
                cfg.attention_bias,
                vb.pp("qkv"),
                &None,
                &None,
                DType::BF16,
            )?,
            q_norm: rms_norm(head_dim, eps, vb.pp("q_norm"), DType::F32, false)?,
            k_norm: rms_norm(head_dim, eps, vb.pp("k_norm"), DType::F32, false)?,
            proj: ReplicatedLinear::load_b(
                dim,
                dim,
                cfg.attention_bias,
                vb.pp("proj"),
                &None,
                &None,
                DType::BF16,
            )?,
            num_heads: cfg.num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[i64],
        cos: &Tensor, // [seq, head_dim]
        sin: &Tensor, // [seq, head_dim]
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let orig_dtype = xs.dtype();
        let qkv = self.qkv.forward(xs)?;
        // [seq, 3, num_heads, head_dim]
        let qkv = qkv.reshape((seq_len, 3, self.num_heads, self.head_dim))?;
        let q = qkv.i((.., 0, .., ..))?; // [seq, num_heads, head_dim]
        let k = qkv.i((.., 1, .., ..))?;
        let v = qkv.i((.., 2, .., ..))?;

        let q = self.q_norm.forward(&q.to_dtype(DType::F32)?)?.to_dtype(DType::F32)?;
        let k = self.k_norm.forward(&k.to_dtype(DType::F32)?)?.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        // Process each image/video segment separately
        let scaling = (self.head_dim as f64).sqrt().recip();
        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0] as usize;
            let end = window[1] as usize;
            if end <= start {
                continue;
            }
            let seg = end - start;
            // [seg, heads, head_dim] -> [heads, seg, head_dim]
            let q_s = q.narrow(0, start, seg)?.transpose(0, 1)?.contiguous()?;
            let k_s = k.narrow(0, start, seg)?.transpose(0, 1)?.contiguous()?;
            let v_s = v.narrow(0, start, seg)?.transpose(0, 1)?.contiguous()?;
            // [heads, seg, seg]
            let att = (q_s.matmul(&k_s.t()?)? * scaling)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // [heads, seg, head_dim] -> [seg, heads, head_dim] -> [seg, heads*head_dim]
            let out = att
                .matmul(&v_s)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((seg, self.num_heads * self.head_dim))?;
            outputs.push(out.to_dtype(orig_dtype)?);
        }
        let out = Tensor::cat(&outputs, 0)?;
        self.proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// Vision MLP (SiLU gate, with bias per config.attention_bias)
// ---------------------------------------------------------------------------
struct VisionMlp {
    gate_proj: ReplicatedLinear,
    up_proj: ReplicatedLinear,
    down_proj: ReplicatedLinear,
}

impl VisionMlp {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilderX) -> Result<Self> {
        let bias = cfg.attention_bias;
        Ok(Self {
            gate_proj: ReplicatedLinear::load_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                bias,
                vb.pp("gate_proj"),
                &None,
                &None,
                DType::BF16,
            )?,
            up_proj: ReplicatedLinear::load_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                bias,
                vb.pp("up_proj"),
                &None,
                &None,
                DType::BF16,
            )?,
            down_proj: ReplicatedLinear::load_b(
                cfg.intermediate_size,
                cfg.hidden_size,
                bias,
                vb.pp("down_proj"),
                &None,
                &None,
                DType::BF16,
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// Vision Block: pre-norm, add residual (no post-norms)
// ---------------------------------------------------------------------------
struct VisionBlock {
    norm1: NormX,
    norm2: NormX,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilderX) -> Result<Self> {
        let eps = cfg.rms_norm_eps();
        Ok(Self {
            norm1: rms_norm(cfg.hidden_size, eps, vb.pp("norm1"), DType::F32, false)?,
            norm2: rms_norm(cfg.hidden_size, eps, vb.pp("norm2"), DType::F32, false)?,
            attn: VisionAttention::new(cfg, vb.pp("attn"))?,
            mlp: VisionMlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[i64],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let dtype = xs.dtype();
        let normed1 = self.norm1.forward(xs)?.to_dtype(dtype)?;
        let attn_out = self.attn.forward(&normed1, cu_seqlens, cos, sin)?;
        let xs = (xs + attn_out.to_dtype(dtype)?)?;
        let normed2 = self.norm2.forward(&xs)?.to_dtype(dtype)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        xs + mlp_out.to_dtype(dtype)?
    }
}

// ---------------------------------------------------------------------------
// Spatial downsample: nn.Conv2d(hidden_size, out_hidden_size, kernel=ms, stride=ms)
// No sub-key "proj" — weights are at "downsample.weight"
// ---------------------------------------------------------------------------
struct SpatialDownsample {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    in_channels: usize,
    out_channels: usize,
}

impl SpatialDownsample {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilderX) -> Result<Self> {
        let ms = cfg.spatial_merge_size;
        let (weight, bias) = match &vb.0 {
            Either::Left(v) => {
                let w = v
                    .get(
                        (cfg.out_hidden_size, cfg.hidden_size, ms, ms),
                        "weight",
                    )?
                    .to_dtype(DType::BF16)?;
                // Conv2d bias is optional; try to load
                let b = v.get(cfg.out_hidden_size, "bias").ok().map(|t| t.to_dtype(DType::BF16).unwrap());
                (w, b)
            }
            Either::Right(_) => candle_core::bail!("GGUF not supported for GLM-OCR"),
        };
        Ok(Self {
            weight,
            bias,
            stride: ms,
            in_channels: cfg.hidden_size,
            out_channels: cfg.out_hidden_size,
        })
    }

    /// xs: [N, hidden_size] where N is divisible by spatial_merge_size^2
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let n = xs.dim(0)?;
        let ms = self.stride;
        let grouped = n / (ms * ms);
        // Permute to [grouped, hidden_size, ms, ms] for conv2d
        let xs = xs
            .reshape((grouped, ms, ms, self.in_channels))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        let out = xs.conv2d(&self.weight, 0, self.stride, 1, 1)?;
        // out: [grouped, out_hidden_size, 1, 1]
        let out = out.reshape((grouped, self.out_channels))?;
        if let Some(b) = &self.bias {
            out.broadcast_add(b)
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// VisionPatchMerger: proj -> GELU(LayerNorm) -> SiLU-gate MLP
// context_dim = out_hidden_size * in_channels  (all no bias)
// ---------------------------------------------------------------------------
struct VisionPatchMerger {
    proj: ReplicatedLinear,
    post_projection_norm: NormX,
    gate_proj: ReplicatedLinear,
    up_proj: ReplicatedLinear,
    down_proj: ReplicatedLinear,
    hidden_act: candle_nn::Activation,
}

impl VisionPatchMerger {
    fn new(cfg: &GlmOcrVisionConfig, vb: VarBuilderX) -> Result<Self> {
        let hidden_act = cfg.hidden_act;
        let dim = cfg.out_hidden_size;
        let context_dim = cfg.out_hidden_size * cfg.in_channels;
        Ok(Self {
            proj: ReplicatedLinear::load_no_bias(
                dim,
                dim,
                vb.pp("proj"),
                &None,
                &None,
                DType::BF16,
            )?,
            // LayerNorm (with learnable weight + bias)
            post_projection_norm: layer_norm(dim, 1e-5, true, vb.pp("post_projection_norm"), DType::F32)?,
            gate_proj: ReplicatedLinear::load_no_bias(
                dim,
                context_dim,
                vb.pp("gate_proj"),
                &None,
                &None,
                DType::BF16,
            )?,
            up_proj: ReplicatedLinear::load_no_bias(
                dim,
                context_dim,
                vb.pp("up_proj"),
                &None,
                &None,
                DType::BF16,
            )?,
            down_proj: ReplicatedLinear::load_no_bias(
                context_dim,
                dim,
                vb.pp("down_proj"),
                &None,
                &None,
                DType::BF16,
            )?,
            hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // proj
        let xs_proj = self.proj.forward(xs)?;
        // GELU(LayerNorm(proj))
        let xs_norm = self.post_projection_norm.forward(&xs_proj.to_dtype(DType::F32)?)?;
        let xs_act = xs_norm.gelu()?.to_dtype(DType::BF16)?;
        // SiLU-gate MLP
        let gate = self.hidden_act.forward(&self.gate_proj.forward(&xs_act)?)?;
        let up = self.up_proj.forward(&xs_act)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// Main vision model
// ---------------------------------------------------------------------------
pub struct GlmOcrVisionModel {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    post_layernorm: NormX,
    downsample: SpatialDownsample,
    merger: VisionPatchMerger,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
}

impl GlmOcrVisionModel {
    pub fn new(
        cfg: &GlmOcrVisionConfig,
        vb: &VarBuilderX,
        _dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(cfg, vb.pp("patch_embed"))?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(cfg, vb.pp(&format!("blocks.{i}")))?);
        }

        let eps = cfg.rms_norm_eps();
        let post_layernorm =
            rms_norm(cfg.hidden_size, eps, vb.pp("post_layernorm"), DType::F32, false)?;
        let downsample = SpatialDownsample::new(cfg, vb.pp("downsample"))?;
        let merger = VisionPatchMerger::new(cfg, vb.pp("merger"))?;

        // head_dim = hidden_size / num_heads; rotary dim = head_dim / 2
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, device)?;

        Ok(Self {
            patch_embed,
            blocks,
            post_layernorm,
            downsample,
            merger,
            rotary_pos_emb,
            spatial_merge_size: cfg.spatial_merge_size,
        })
    }

    /// Compute rotary position embeddings for all vision tokens.
    /// Returns (cos, sin) each of shape [total_tokens, head_dim].
    ///
    /// Follows HF Python implementation's rot_pos_emb exactly:
    /// For each (t, h, w) grid, produce hpos_ids and wpos_ids with reshape+permute
    /// so that tokens within a merge window are interleaved.
    fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<(Tensor, Tensor)> {
        let device = self.rotary_pos_emb.inv_freq.device();
        let grid = grid_thw.to_vec2::<u32>()?;
        let ms = self.spatial_merge_size;

        let max_grid = grid
            .iter()
            .flat_map(|v| [v[1] as usize, v[2] as usize])
            .max()
            .unwrap_or(1);

        let freq_table = self.rotary_pos_emb.forward(max_grid)?; // [max_grid, half_head_dim]
        let _half_dim = freq_table.dim(1)?;

        let mut all_pos_ids: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();

        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;

            // Build hpos_ids: shape [h, w] then reshape/permute/flatten
            // hpos_ids[i, j] = i (row index)
            let mut hpos_flat = vec![0i64; h * w];
            for r in 0..h {
                for c in 0..w {
                    hpos_flat[r * w + c] = r as i64;
                }
            }

            // wpos_ids[i, j] = j (col index)
            let mut wpos_flat = vec![0i64; h * w];
            for r in 0..h {
                for c in 0..w {
                    wpos_flat[r * w + c] = c as i64;
                }
            }

            // Reshape to [h/ms, ms, w/ms, ms] then permute to [h/ms, w/ms, ms, ms] then flatten
            // Python: hpos_ids.reshape(h//ms, ms, w//ms, ms).permute(0, 2, 1, 3).flatten()
            let mh = h / ms;
            let mw = w / ms;
            let mut hpos_reordered = vec![0i64; h * w];
            let mut wpos_reordered = vec![0i64; h * w];
            let mut idx = 0;
            for mr in 0..mh {
                for mc in 0..mw {
                    for ir in 0..ms {
                        for ic in 0..ms {
                            let orig_r = mr * ms + ir;
                            let orig_c = mc * ms + ic;
                            hpos_reordered[idx] = hpos_flat[orig_r * w + orig_c];
                            wpos_reordered[idx] = wpos_flat[orig_r * w + orig_c];
                            idx += 1;
                        }
                    }
                }
            }

            // Repeat for t temporal frames
            let mut hpos_t = Vec::with_capacity(t * h * w);
            let mut wpos_t = Vec::with_capacity(t * h * w);
            for _ in 0..t {
                hpos_t.extend_from_slice(&hpos_reordered);
                wpos_t.extend_from_slice(&wpos_reordered);
            }
            all_pos_ids.push((hpos_t, wpos_t));
        }

        // Concatenate all images
        let total: usize = all_pos_ids.iter().map(|(v, _)| v.len()).sum();
        let mut hpos_all: Vec<i64> = Vec::with_capacity(total);
        let mut wpos_all: Vec<i64> = Vec::with_capacity(total);
        for (h, w) in all_pos_ids {
            hpos_all.extend(h);
            wpos_all.extend(w);
        }

        let hpos_t = Tensor::from_vec(hpos_all, (total,), device)?;
        let wpos_t = Tensor::from_vec(wpos_all, (total,), device)?;

        // Look up frequencies: [total, half_head_dim]
        let h_emb = freq_table.index_select(&hpos_t, 0)?;
        let w_emb = freq_table.index_select(&wpos_t, 0)?;

        // Concatenate: [total, head_dim]
        let emb = Tensor::cat(&[&h_emb, &w_emb], 1)?;

        Ok((emb.cos()?, emb.sin()?))
    }

    /// cu_seqlens as Vec<i64> (prefix 0)
    fn build_cu_seqlens(&self, grid_thw: &Tensor) -> Result<Vec<i64>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = vec![0i64];
        let mut acc = 0i64;
        for g in &grid {
            let area = (g[1] * g[2]) as i64;
            for _ in 0..(g[0] as usize) {
                acc += area;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    /// pixel_values: [total_patches, C * T * pH * pW]
    /// grid_thw: [num_images, 3] (T, H, W) in patch units
    /// Returns: [total_merged_tokens, out_hidden_size]
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.patch_embed.forward(pixel_values)?;

        let (cos, sin) = self.rot_pos_emb(grid_thw)?;
        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
        }

        hidden_states = self
            .post_layernorm
            .forward(&hidden_states)?
            .to_dtype(DType::BF16)?;

        // Spatial downsample: [N, hidden_size] -> [N/ms^2, out_hidden_size]
        // Reshape: [-1, ms, ms, hidden_size] then permute and conv2d
        hidden_states = self.downsample.forward(&hidden_states)?;

        // Merger
        self.merger.forward(&hidden_states)
    }
}
