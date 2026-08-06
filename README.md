<p align="center">
  <img src="ddrk_logo.png" width="400" alt="DDRK Omega Logo">
</p>

# DDRK Omega Sampler
### **D**omain-adaptive **D**iffusion **R**obust **K**ernel

> A production-ready universal sampler for **ComfyUI**.  
> Flow Matching · EDM · Adaptive Phase Routing · Momentum Denoising · Perceptual Sharpening

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What does DDRK stand for?

**D**omain-adaptive — automatically detects Flow Matching (Anima/Flux/SD3) vs EDM (SDXL/SD1.5/Illugen) and recalibrates all internal parameters (schedules, integrators, SDE, sharpening) on the fly.

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
| Oscillation / grain in textures | Velocity EMA (momentum) on final direction vector only |
| Color burn on EDM high CFG | Dynamic thresholding + soft CFG rescale |
| Video frame flickering | SABER 2.0 with temporal EMA + chaos metric |
| Over-sharpening noisy regions | Perceptual sharpen with entropy gate (bypassed on final step) |
| Memory leaks in long sessions | LRU-bounded caches for kernels and SABER buffers |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1  (~65% steps)  │  High-order integrator + SDE noise │
│                         │  RK4 (FM) / Heun (EDM)             │
├─────────────────────────────────────────────────────────────┤
│  PHASE 2  (~25% steps)  │  Euler + SABER stabilization       │
│                         │  Temporal EMA (video) or bilateral │
│                         │  spatial smoothing (image)         │
├─────────────────────────────────────────────────────────────┤
│  PHASE 3  (~10% steps)  │  Euler + final sharpening          │
│                         │  Perceptual unsharp mask           │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **No fake FSAL.** Classical RK4 does not support First-Same-As-Last caching. We use honest 4-eval RK4 when the step budget allows, and fall back to Heun/Euler when it doesn't.
- **Momentum on final direction only.** Smoothing intermediate RK stages destroys 4th-order accuracy. We apply EMA only to the finalized `d` vector between steps.
- **Sigma-adaptive SDE.** Noise scales as `√dt · σ^0.25`, applied only to flat regions (inverted edge mask) to avoid boundary artifacts.
- **True Karras schedule.** `ρ = 7` polynomial schedule for EDM, not a simple exponential masquerading as Karras.

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
| `ddrk_edm_karras` | EDM — true Karras (ρ=7) |
| `ddrk_edm_poly` | EDM — polynomial tail |
| `ddrk_edm_simple` | EDM — pure exponential |

### 2. DDRK Omega Sampler
The sampler engine. Connect to a `SamplerCustom` or use standalone.

**Key parameters:**
- `integrator`: `auto` (recommended), `rk4`, `heun`, `euler`
- `sde_strength`: 0.0 = deterministic, 0.08 = subtle diversity (FM only)
- `sharpness`: final unsharp mask strength
- `saber_fusion`: stabilization weight (0 = off)
- `saber_mode`: `auto` detects video by 5D latent shape
- `momentum_beta`: velocity EMA (0 = disabled)
- `dyn_thresh_percentile`: Imagen-style clamping for EDM (1.0 = off)
- `cfg_rescale`: soft compression of extreme latent values
- `sde_seed`: deterministic SDE noise (-1 = random)

### 3. DDRK Omega Unified KSampler
All-in-one node. Drop-in replacement for ComfyUI's native KSampler.

---

## Recommended Settings

### Anima / Flux / SD3 (Flow Matching)
| Parameter | Value |
|-----------|-------|
| Steps | 20–30 |
| Scheduler | `ddrk_auto` or `ddrk_cosine` |
| Flow shift | 3.0 |
| Integrator | `auto` |
| SDE strength | 0.08 |
| Sharpness | 0.30 |
| SABER fusion | 0.30 |

### SDXL / SD1.5 (EDM)
| Parameter | Value |
|-----------|-------|
| Steps | 25–40 |
| Scheduler | `ddrk_auto` or `ddrk_edm_karras` |
| Integrator | `auto` (will force Heun) |
| SDE strength | 0.00 (ignored automatically) |
| Sharpness | 0.20 |
| SABER fusion | 0.15 (auto-capped) |
| Dyn threshold | 0.995 |

### Video / AnimateDiff
| Parameter | Value |
|-----------|-------|
| SABER mode | `video` (or `auto` if latent is 5D) |
| SABER fusion | 0.35–0.50 |
| EMA decay | 0.7 |

### Few-step / LCM-like
| Parameter | Value |
|-----------|-------|
| Steps | 4–8 |
| Scheduler | `ddrk_fewstep` |
| Integrator | `euler` (auto-forced) |
| Sharpness | 0.10 |
| SABER fusion | 0.10 |

---

## Changelog

| Version | Changes |
|---------|---------|
| **v1.0** | Original prototype by author. Hybrid RK4/Heun + SABER + SWT sharpening. |
| **v1.1** | Production pass by Kimi (Moonshot AI). Removed broken FSAL-RK4, fixed momentum scope, true Karras schedule, inverted SDE mask, adaptive dynamic thresholding. |
| **v1.2** | Fixed `local_entropy_mask` 5D reshape, safe sigma-to-float conversion, LRU cache bounds, exposed `sde_seed` in UI, expanded Unified KSampler parameters. |
| **v1.3** | Fixed SABER2 crash on 5D latents with `F=1` (single-frame video format). Reflect-pad safety for `avg_pool3d`. |
| **v1.4** | EDM path fixes — less blur, less clamp-aggression, more accurate steps. |
| **v1.4.1** | Adaptive Balance. |

---

## Credits & Provenance

This project has an unusual but fully transparent development lineage:

1. **Original concept & prototype (v1.0)**  
   Created by **HVOSTOVSKY** — the core idea of a hybrid phase-based sampler with high-order integrators, SDE noise masking, and SABER temporal stabilization.

2. **Local AI agent audit loop**  
   The codebase was iteratively reviewed by a local AI assistant, catching architectural inconsistencies and proposing mathematical improvements (adaptive order, sigma-aware schedules, FM/EDM branching).

3. **Production hardening (v1.1–v1.3)**  
   Final multi-pass static analysis, mathematical verification of integrator correctness, and edge-case hardening were performed by **Kimi** (Moonshot AI). This included:
   - Mathematical audit: removal of invalid FSAL-RK4 caching, correct momentum application (final-direction only)
   - Schedule correctness: true Karras ρ=7, guaranteed `0.0` terminators for all FM curves
   - Tensor safety: correct 5D↔4D reshape ordering, `F=1` reflect-pad crash fix
   - Memory safety: LRU-bounded caches for LoG kernels and SABER buffers
   - API compliance: full ComfyUI `denoise<1.0`, `noise_mask`, and dtype contract adherence

---

## Known Limitations

- **RK4 is expensive.** ~2.5× model calls vs Euler. Use `auto` integrator to let the sampler decide.
- **Video mode requires 5D latents.** Standard AnimateDiff output works; single-frame 5D tensors (`[B,C,1,H,W]`) are handled but offer no temporal benefit.
- **Few-step (<6) forces Euler.** High-order integrators need step budget to show advantage.
- **No built-in TeaCache / caching.** Each model call is fresh; speedups require external acceleration nodes.

---

## Issues

Found a bug? Open an [Issue](https://github.com/HVOSTOVSKY/DDRK-Omega-Sampler/issues) with:
- Model name (Anima / Flux / SDXL / etc.)
- Steps and settings
- Error traceback (if crash) or comparison images (if quality issue)

---

## License

MIT License — free for personal and commercial use. Attribution appreciated but not required.


---

*«One sampler to rule them all — from Flux to SDXL.»*
