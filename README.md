<p align="center">
  <img src="logo.png" width="640" alt="DDRK Omega Logo">
</p>

# DDRK Omega Sampler
### **D**omain-adaptive **D**iffusion **R**obust **K**ernel

> A production-ready universal sampler for **ComfyUI**.  
> Flow Matching · EDM · Adaptive Phase Routing · Momentum Denoising · Perceptual Sharpening · Architecture Auto-Detection

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What does DDRK stand for?

**D**omain-adaptive — automatically detects Flow Matching (Anima/Flux/SD3/Krea/Qwen/HiDream/Chroma/Lumina) vs EDM (SDXL/SD1.5) and recalibrates all internal parameters (schedules, integrators, SDE, sharpening, stabilization) on the fly.

**D**iffusion — obviously, this is a diffusion sampler.

**R**obust — built-in stabilization: momentum-corrected integrators, SABER temporal/spatial filtering, dynamic thresholding (Imagen-style), entropy-gated sharpening, and adaptive SDE noise.

**K**ernel — the core engine. Compact, self-contained, no external dependencies beyond PyTorch and ComfyUI itself.

---

## Why DDRK Omega?

Most samplers are either **fast but low-quality** (Euler) or **high-quality but slow** (DPM++ 3M SDE). They also force you to manually pick different samplers for Flow Matching and EDM models.

DDRK Omega solves both problems:

| Problem | DDRK Solution |
|---------|--------------|
| FM vs EDM incompatibility | Auto-detection by `sigma_max` + domain-specific calibration |
| RK4 is too slow (4× model calls) | Adaptive order: RK4 only when beneficial, Heun/Euler otherwise |
| Plastic / smeared fur or hair on FM | Per-step soft clamp removed for FM — preserves natural texture chaos |
| Wrong parameters for distilled/turbo LoRAs | `guidance_embed` detection + hint text warns about CFG≈1 variants |
| Oscillation / grain in textures | Velocity EMA (momentum) on final direction vector only |
| Color burn on EDM high CFG | Dynamic thresholding + std-matching CFG rescale (phi-blend — restores contrast instead of clamping) |
| Plastic / smeared look on photorealistic FM | Content-aware SABER — edge-gated fusion scales blur per-pixel; SABER fades in on Phase 2 |
| Video frame flickering | SABER 2.0 with temporal EMA + chaos metric |
| Over-sharpening noisy regions | Multi-scale perceptual sharpen with inverted entropy gate (3×3 texture + 5×5 structure, sharpens edges/textures, protects flat walls) |
| Memory leaks in long sessions | LRU-bounded caches for kernels and SABER buffers |
| Inpainting / img2img denoise bugs | `fix_empty_latent_channels` + correct `denoise<1.0` sigma handling matches stock KSampler |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1  (~55-65% steps)  │  Adaptive-order integrator + SDE     │
│                            │  RK4/Heun/Euler by trajectory        │
│                            │  curvature; EDM churn (Karras)       │
│                            │  SDE: entropy-gated + quadratic mask │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2  (~25-35% steps)  │  Euler + conditional stabilization   │
│                            │  EDM: SABER OFF (mid-step blur       │
│                            │  causes anatomical shifts)           │
│                            │  FM images: content-aware SABER      │
│                            │  (edge-gated fusion, fades in)       │
│                            │  FM video (5D): SABER temporal EMA   │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3  (~10% steps)     │  Euler + final sharpening            │
│                            │  Multi-scale unsharp (3×3+5×5)       │
│                            │  Phase-fade: sharpness ramps in      │
│                            │  over 1–2 steps, no hard switch      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Adaptive order by curvature.** RK4/Heun/Euler chosen per-step by trajectory curvature (||Δdenoised|| / σ-step), not static sigma thresholds. Honest 4-eval RK4, no fake FSAL.
- **Momentum on final direction only.** Smoothing intermediate RK stages destroys 4th-order accuracy. We apply EMA only to the finalized `d` vector between steps.
- **Phase boundary fade.** SABER and sharpening ramp over 1–2 steps at phase transitions instead of hard binary switches — eliminates "seam" micro-artifacts.
- **Sigma-adaptive SDE.** Noise scales as `√dt · σ^0.25`, applied only to truly flat regions (`(1−edge)² × entropy`) to avoid boundary artifacts and grain in mid-tones.
- **EDM churn (Karras Algorithm 2).** Classic stochastic injection for EDM models at high-sigma steps — compensates discretization error and boosts micro-detail on SDXL/SD1.5.
- **Multi-scale sharpening.** Perceptual unsharp mask uses both 3×3 (fine texture) and 5×5 (mid-frequency structure) kernels — sharpens without halo artifacts.
- **FM soft clamp removed.** Stock KSampler never clamps between steps. We match that for Flow Matching to preserve natural texture (fur, hair, water). EDM still clamped per Karras.
- **Photo-aware FM path.** Flow Matching models get auto-capped SABER (`≤0.15`), reduced momentum (`≤0.15`), and SDE gated by local entropy to prevent plastic skin and sand-like noise.
- **Universal dynamic thresholding.** Imagen-style clamping is active for both EDM and FM whenever `dyn_thresh_percentile < 1.0`, preventing CFG blowouts across all architectures.
- **True Karras schedule.** `ρ = 7` polynomial schedule for EDM, not a simple exponential masquerading as Karras.
- **Architecture introspection.** Detects model family via `unet_config["image_model"]` — Flux, SD3, Qwen, Krea, HiDream, Chroma, Lumina, SDXL, SD1.5, SD2.

---

## Installation

1. Clone or download this repo into your `ComfyUI/custom_nodes/` folder:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/HVOSTOVSKY/DDRK-Omega-Sampler.git
```

2. Restart ComfyUI. No additional Python packages required.

---

## Nodes

### 1. DDRK Omega Scheduler
Generates sigma schedules. Auto-detects model family and picks the optimal curve.

| Scheduler | Best for |
|-----------|----------|
| `ddrk_auto` | Let the sampler decide (recommended) |
| `ddrk_cosine` | Flow Matching — smooth cosine decay |
| `ddrk_beta` | Flow Matching — Beta-distributed steps |
| `ddrk_anima` | Flow Matching — with optional warmup |
| `ddrk_fewstep` | Flow Matching — 4-8 steps, aggressive shift |
| `ddrk_flow_cosmos` | Flow Matching — cosmos-style tail adjustment |
| `ddrk_flow_linear` | Flow Matching — linear with shift |
| `ddrk_edm_karras` | EDM — true Karras (ρ=7) |
| `ddrk_edm_poly` | EDM — polynomial tail |
| `ddrk_edm_simple` | EDM — pure exponential |

### 2. DDRK Omega Sampler
The sampler engine. Connect to a `SamplerCustom` or use standalone.

**Key parameters:**
- `integrator`: `auto` (recommended), `rk4`, `heun`, `euler`. On FM photo models `auto` prefers `heun`.
- `sde_strength`: 0.0 = deterministic. For FM photo realism use 0.02–0.04; for FM arts up to 0.08. EDM ignores SDE automatically.
- `sharpness`: final unsharp mask strength. Safe to raise to 0.35 on FM photo since entropy gate prevents halos.
- `saber_fusion`: stabilization weight (0 = off). Auto-capped to 0.15 on FM images to avoid smearing.
- `saber_mode`: `auto` detects video by 5D latent shape (`F > 1`).
- `use_ema_saber`: temporal EMA for video; spatial edge-aware blur for images.
- `ema_decay`: temporal smoothing factor for video SABER.
- `momentum_beta`: velocity EMA (0 = disabled). Auto-capped to 0.15 on FM to preserve fine detail.
- `dyn_thresh_percentile`: Imagen-style clamping for **both** EDM and FM (1.0 = off).
- `cfg_rescale`: soft compression of extreme latent values.
- `sde_seed`: deterministic SDE noise (-1 = random).
- `s_churn`: EDM churn strength (Karras Algorithm 2). Injects noise at high-sigma steps to compensate discretization error. 0 = off. Try 5–15 for SDXL detail boost.
- `s_noise`: EDM churn noise multiplier. 1.0 = standard Karras noise scaling.
- `content_aware`: Content-aware SABER fusion — edge-gated blur that preserves detail. Default True. Disable for pixel-art, flat-shading, or heavily stylized LoRAs where edge detection aliases against the pattern frequency.
- `auto_optimize`: Auto-disable SABER/SDE/momentum and force euler for FM few-step (≤10 steps). Does NOT set steps/cfg.

### 3. DDRK Omega Unified KSampler
All-in-one node. Drop-in replacement for ComfyUI's native KSampler.

**Additional features:**
- `smart_defaults`: Auto-detect architecture and set scheduler/integrator/shift/saber/sharpness. **Steps and CFG are checkpoint-specific** (depend on LoRA, distillation, turbo) and must be tuned manually. Use the hint output from SmartConfigNode as guidance.
- `saber_fusion`: Now exposed as a regular widget (not hidden forceInput).
- Live preview support with graceful fallback for older ComfyUI builds.
- Correct `denoise<1.0` handling matching stock KSampler (no double latent scaling).

### 4. DDRK Omega Smart Config
Debug / introspection node. Feed it a model + latent and it returns:
- `family`: detected architecture (flux, sd3, qwen, krea, hidream, chroma, lumina, sdxl, sd15, sd2, edm, fm)
- `hint`: human-readable recommendation for steps/cfg ranges
- `guidance_embed`: Boolean — True if the model has built-in guidance (usually distilled variants, CFG≈1)
- `scheduler_type`, `flow_shift`, `integrator`, `sde_strength`, `sharpness`, `saber_fusion`, `momentum_beta`: recommended sampler settings

> **Note:** SmartConfig does **not** guess steps or CFG. Those depend on training recipe, LoRA, and distillation — not on topology. Always check your model card.

---

## Recommended Settings

### General rule of thumb
- **For texture-heavy content** (fur, hair, water, fabric): `integrator=euler`, `saber_fusion=0`, `momentum_beta=0`, `sde_strength=0`
- **For smooth / stylized art**: `integrator=auto`, `saber_fusion=0.15–0.30`, `sde_strength=0.04–0.08`
- **For text / UI / graphics**: `integrator=euler`, `saber_fusion=0`, `sde_strength=0`, `sharpness=0.40+`
- **Distilled / turbo / CFG≈1 models**: check `guidance_embed` via SmartConfigNode — if True, start with CFG 1.0 and 4-8 steps

### Anima / Flux / SD3 / Krea / Qwen / HiDream / Chroma / Lumina — Arts & Stylized (Flow Matching)
| Parameter | Value |
|-----------|-------|
| Steps | 8–20 (checkpoint-dependent; turbo LoRA = 4-8, base = 20-50) |
| CFG | 1.0–4.0 (checkpoint-dependent; distilled = 1.0, dev = 3.5–4.0) |
| Scheduler | `ddrk_auto` or `ddrk_cosine` |
| Flow shift | 1.0 |
| Integrator | `euler` for texture; `auto` for smooth art |
| SDE strength | 0.00–0.04 (photo) / 0.04–0.08 (art) |
| Sharpness | 0.20–0.30 |
| SABER fusion | 0.00–0.15 (photo) / 0.15–0.30 (art) |
| Momentum beta | 0.00–0.15 |

### SDXL / SD1.5 (EDM)
| Parameter | Value |
|-----------|-------|
| Steps | 20–30 |
| CFG | 7.0–8.0 (base); 1.0–2.0 (turbo/distilled) |
| Scheduler | `ddrk_auto` or `ddrk_edm_karras` |
| Integrator | `auto` (will force Heun) |
| SDE strength | 0.00 (ignored automatically) |
| S churn | 5–15 (Karras Algorithm 2 — adds micro-detail) |
| S noise | 1.0 |
| Sharpness | 0.20 |
| SABER fusion | 0.15 (auto-capped) |
| Dyn threshold | 0.995 |

### Video / AnimateDiff
| Parameter | Value |
|-----------|-------|
| SABER mode | `video` (or `auto` if latent is 5D) |
| SABER fusion | 0.35–0.50 |
| EMA decay | 0.7 |

### Few-step / LCM-like / Turbo
| Parameter | Value |
|-----------|-------|
| Steps | 4–8 |
| CFG | 1.0 (distilled) |
| Scheduler | `ddrk_auto` or `ddrk_fewstep` |
| Integrator | `euler` (auto-forced) |
| Sharpness | 0.10–0.15 |
| SABER fusion | 0.00 |
| Momentum beta | 0.00 |
| auto_optimize | True |

---

## Changelog

| Version | Changes |
|---------|---------|
| **v1.0** | Original prototype by author. Hybrid RK4/Heun + SABER + SWT sharpening. |
| **v1.1** | Production pass by Kimi (Moonshot AI). Removed broken FSAL-RK4, fixed momentum scope, true Karras schedule, inverted SDE mask, adaptive dynamic thresholding. |
| **v1.2** | Fixed `local_entropy_mask` 5D reshape, safe sigma-to-float conversion, LRU cache bounds, exposed `sde_seed` in UI, expanded Unified KSampler parameters. |
| **v1.3** | Fixed SABER2 crash on 5D latents with `F=1` (single-frame video format). Reflect-pad safety for `avg_pool3d`. |
| **v1.4** | EDM path fixes — less blur, less clamp-aggression, more accurate steps. |
| **v1.4.1** | Adaptive Balance. Soft variance clamp, adaptive DT skip, SABER for EDM only on extreme early noise. |
| **v1.4.3** | Consolidated Quality & Audit Pass. EDM churn, multi-scale sharpen, std-matching CFG rescale, adaptive order by curvature, phase-boundary fade, content-aware SABER. |
| **v1.5** | **Architecture Introspection & Quality Hardening.** SmartConfigNode with `image_model` detection (Flux/SD3/Qwen/Krea/HiDream/Chroma/Lumina/SDXL/SD15/SD2), `guidance_embed` signal for distilled variants, `auto_optimize` for FM few-step, `smart_defaults` for one-click architecture setup. Fixed `fix_empty_latent_channels` compatibility, corrected `denoise<1.0` double-scaling bug, unified device source, live preview support, removed per-step soft clamp for FM (fixes plastic fur/hair), exposed `saber_fusion` as regular widget. |

---

## Credits & Provenance

1. **Original concept & prototype (v1.0)**  
   Created by **HVOSTOVSKY** — the core idea of a hybrid phase-based sampler with high-order integrators, SDE noise masking, and SABER temporal stabilization.

2. **Production hardening (v1.1–v1.5)**  
   Iterative development and multi-pass audits by **Kimi** (Moonshot AI). This included:
   - Mathematical audit: removal of invalid FSAL-RK4 caching, correct momentum application (final-direction only)
   - Schedule correctness: true Karras ρ=7, guaranteed `0.0` terminators for all FM curves
   - Tensor safety: correct 5D↔4D reshape ordering, `F=1` reflect-pad crash fix
   - Memory safety: LRU-bounded caches for LoG kernels and SABER buffers
   - API compliance: full ComfyUI `denoise<1.0`, `noise_mask`, `fix_empty_latent_channels`, and dtype contract adherence
   - Architecture introspection: `image_model` / `guidance_embed` detection via `model.model.model_config`
   - Quality tuning: FM clamp removal, auto_optimize, smart_defaults

---

## Known Limitations

- **RK4 is expensive.** ~2.5× model calls vs Euler. Use `auto` integrator to let the sampler decide.
- **Video mode requires 5D latents.** Standard AnimateDiff output works; single-frame 5D tensors (`[B,C,1,H,W]`) are handled but offer no temporal benefit.
- **Few-step (≤6) forces Euler.** High-order integrators need step budget to show advantage.
- **No built-in TeaCache / caching.** Each model call is fresh; speedups require external acceleration nodes.
- **Steps and CFG are not auto-detected.** They depend on checkpoint training, LoRA, and distillation. SmartConfigNode gives hints, but you must tune manually.
- **FM photorealism vs art trade-off.** The auto_optimize path auto-caps stabilization on FM images while preserving structure. If you generate heavily stylized FM art and see noise in flat color regions, manually raise `saber_fusion` to 0.20–0.30.

---

## Issues

Found a bug? Open an [Issue](https://github.com/HVOSTOVSKY/DDRK-Omega-Sampler/issues) with:
- Model name (Anima / Flux / SDXL / Krea / Qwen / HiDream / etc.)
- Steps and settings
- Error traceback (if crash) or comparison images (if quality issue)

---

## License

MIT License — free for personal and commercial use. Attribution appreciated but not required.

---

*«One sampler to rule them all — from Flux to SDXL.»*
