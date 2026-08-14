<div align="center">

<img src="logo.png" width="620" alt="DDRK Omega Sampler">

# DDRK Omega Sampler

**D**omain-adaptive **D**iffusion **R**obust **K**ernel

*One sampler node for Flow Matching and EDM models alike.*

[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-1f6feb?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-3fb950?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.6.0-8957e5?style=flat-square)](#changelog)

</div>

---

## Overview

DDRK Omega is a single sampler node that handles **Flow Matching** models (Flux, SD3, Qwen, Krea, HiDream, Chroma, Lumina) and **EDM** models (SDXL, SD 1.5, SD 2) without switching samplers or relearning parameters per family. It detects the family from the sigma schedule and recalibrates internally.

The goal is a node you drop in and use — not one you tune for an hour per checkpoint.

Version 1.6.0 introduces **HC2**, a new integrator that reaches second-order accuracy at one model call per step, and completes a correctness pass driven by the sampler's own telemetry.

<div align="center">

| | |
|:--|:--|
| **HC2 integrator** | Second order at 1 model call/step — Heun quality at roughly half the compute on Flow Matching |
| **Adaptive phase routing** | Three phases, each with its own integrator policy and post-processing |
| **Ancestral SDE** | Calibrated noise split (Karras et al. 2022), gated to flat regions |
| **Per-step telemetry** | Opt-in JSON + CSV + summary logs of every internal decision |
| **Family auto-detection** | Schedules, integrator and enhancer defaults resolved per model family |

</div>

---

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/HVOSTOVSKY/DDRK-Omega-Sampler.git
```

Restart ComfyUI. No dependencies beyond what ComfyUI already requires.

---

## Quick start

Drop in **DDRK Omega Unified KSampler** in place of the stock KSampler, leave `smart_defaults` and `auto_optimize` on, and set steps and CFG as your checkpoint's model card recommends. That is the whole setup.

To try HC2 explicitly, set `integrator` to `hc2`.

---

## What is actually verified

This section exists because it is easy for a project like this to accumulate claims. Everything below was measured with the built-in telemetry on real generations across five checkpoints (Anima, Krea2, Qwen, Illustrious, PonyXL).

### Measured

**HC2 order of accuracy.** On a semi-linear test ODE with a known solution, error falls 4x per halving of the step (second order) against Euler's 2x. At an equal number of model evaluations it was ~33x more accurate at 40 steps.

**HC2 vs Heun on Flow Matching, equal compute.** HC2 at 20 steps (20 calls) vs Heun at 10 steps (19 calls), three seeds, every enhancer at zero: HC2's final latent held a **16-25% wider dynamic range on all three seeds**, mean +21%. The same metric had previously separated Euler from Heun/RK4 and matched blind visual judgement both times.

**HC2 vs Heun on Flow Matching, equal step count.** Both at 8 steps on a cosine schedule, three seeds: HC2 matched or beat Heun on 2 of 3 seeds (mean +5% range) while using **9 model calls against 15** and finishing roughly 40% faster.

**HC2 does not lean on the enhancer stack.** Turning sharpening, SDE and SABER off moved HC2's result by +0.2% while Heun's dropped 4.6%. HC2's detail comes from the integration, not from filters applied afterwards.

**Integrator ordering on Flow Matching at equal compute.** Euler came last on all three seeds, both by eye and by dynamic range, which ran 12-17% narrower than Heun/RK4.

**Stability and reproducibility.** No NaN or Inf across every logged run. Identical inputs reproduce identical output; residual variation between runs is ~1e-7 and comes from non-deterministic GPU kernels, not from the sampler.

### Measured, and negative

**HC2 shows no advantage on EDM.** Illustrious, three seeds, equal settings: mean range difference +1.2%, and HC2 used 10% more model calls to get it. On EDM, Heun or `auto` remains the sensible choice.

A plausible explanation: HC2 assumes the denoiser varies smoothly in lambda = -log(sigma). The Karras EDM schedule is already close to uniform in lambda, so it does much of that work already. Flow Matching schedules cannot be uniform in lambda — sigma reaches exactly zero, so lambda diverges — which is where an exponential integrator has the most to offer. **This explanation is plausible but not proven**, and it cannot be tested directly on Flow Matching for the same structural reason.

**The corrector and adaptive step placement did not show a measurable effect.** Both remain in the code, opt-in and off by default, documented as experimental.

### Not verified

- Individual enhancers (SABER, sharpening, AB2 extrapolation) have not been ablated against each other. Their defaults are starting points, not tuned optima.
- HC2's third-order mode is mathematically verified on synthetic ODEs but rarely accepted on real trajectories at low step counts, and has not been shown to improve images.
- Dynamic range is a proxy metric. It tracked visual quality in every comparison where the difference was obvious, and stopped discriminating on subtle ones.
- No automated tests.

---

## The HC2 integrator

### What it is

The core is an **exponential (semi-linear) multistep method** — the same family as DPM-Solver++(2M) and UniPC. That part is published mathematics, not invented here. The slope limiter and the measured-error order selection built on top are specific to this sampler.

### Why it beats Heun per model call

The sampling ODE is `dx/dsigma = (x - D(x, sigma)) / sigma`. It is *linear in x*, with all the difficulty living in the denoiser `D`. Euler on it already integrates the linear part exactly — the update reduces to DDIM. Euler's only error is treating `D` as constant across the step.

Heun and RK4 spend their extra model calls re-approximating the whole right-hand side, including the linear part that needed no approximating, and they evaluate `D` at intermediate points on a *predicted* trajectory, so those evaluations carry their own error.

HC2 keeps the exact linear solve and approximates `D` as linear in lambda = -log(sigma), using the previous step's evaluation — already exact, already paid for, no prediction required:

```
x_next = e^(-h) * x  +  (1 - e^(-h)) * D_n  +  limit( r * [(h - 1) + e^(-h)] )

    h = log(sigma_n / sigma_next)        r = (D_n - D_prev) / h_prev
```

The first two terms are Euler/DDIM. The third is the correction, whose coefficient behaves as `h^2/2` for small `h` — second order, at **one** model call per step.

### The slope limiter

`r` is an extrapolation from past data. Under high CFG the denoiser swings hard between steps and an unlimited extrapolation overshoots — the classic oscillation of high-order schemes near a sharp feature.

Finite-volume CFD solved this decades ago with slope limiters: keep full accuracy where the solution is smooth, drop toward first order where it is not, decided per element. Here the correction is capped elementwise at `limiter_kappa x` the magnitude of the first-order step. Unlike a clamp on latent values, this never touches the output range, so it costs no contrast.

In practice the limiter engages on 1-3% of elements on a typical step, rising to ~28% on the first step with history. It intervenes, it does not dominate.

### Order selection

With `hc2_max_order = 3`, HC2 fits a quadratic through three past evaluations and compares the third-order term against the second. If each successive term is smaller, the expansion is behaving and the extra order is used; if the third rivals the second, the fit is being driven by noise in `D` rather than real curvature, and HC2 falls back to second order. Both the ratio and the order chosen are logged.

Third order also pays one extra model call on the first step. A multistep method's first step is first-order, and that single step otherwise caps the entire run at second order — measured: cold start converged at 4x per halving, seeded start at 7.7x toward the theoretical 8x.

---

## Architecture

```
PHASE 1  (~55% FM / ~65% EDM of steps)
    Integrator chosen per step in auto mode, or pinned
    FM   ancestral SDE injection, gated to flat regions
    EDM  optional churn (Karras Alg. 2), SABER at very high sigma

PHASE 2  (~35% FM / ~22% EDM)
    Integrator per phase policy
    FM   SABER spatial fusion
    EDM  no SABER - mid-step blur shifts anatomy

PHASE 3  (remainder)
    Euler in auto mode
    FM   final perceptual sharpen on the last step
    EDM  no sharpen
```

**Adaptive integrator order.** In `auto` mode, phase 1 picks per step from a curvature measure: the fractional change of the denoiser output over the previous step. Because it is dimensionless, one pair of thresholds is valid on both families — an earlier version divided by the sigma step, which made the same quantity read ~0.6 on EDM and ~17 on FM.

**Ancestral SDE split.** A step from sigma to sigma_next decomposes into a shorter deterministic step to sigma_down plus noise of standard deviation sigma_up, calibrated so the combined variance reproduces sigma_next's marginal exactly (Karras et al. 2022). The spatial gating on top — noise only into flat, low-detail regions — is this sampler's own texture-preservation heuristic, not part of that derivation.

**Soft clamp.** EDM only, bound tied to the noise level (`max(4*sigma, 10)`) rather than a constant. A constant bound was clipping ~39% of the tensor at high sigma — ordinary early noise, not divergence. Flow Matching gets no per-step clamp at all; both families keep a wide final guard against actual blowups.

**Edge detection.** The LoG mask flags pixels whose local curvature exceeds the background level, estimated robustly from the median. A percentile cutoff was used before and was tautological — it marked a fixed 20% of every tensor as "edge" regardless of content, including at step 0 where the latent is still pure noise.

**Reproducibility.** SDE noise and EDM churn draw from one seeded generator. With `sde_seed = -1` the seed derives from the global RNG, which ComfyUI seeds from the workflow seed, so runs repeat from the seed widget alone.

---

## Nodes

| Node | Purpose |
|:--|:--|
| **DDRK Omega Unified KSampler** | All-in-one drop-in replacement for the stock KSampler. Start here. |
| **DDRK Omega Sampler** | Returns a `SAMPLER` object for use with `SamplerCustom`. |
| **DDRK Omega Scheduler** | Returns a `SIGMAS` schedule only. Also exposes `beta_a` / `beta_b`. |
| **DDRK Omega Smart Config** | Introspection: detected family, hint text, `guidance_embed`, recommended settings. Deliberately does not guess steps or CFG. |

### Schedulers

| Scheduler | Family | Notes |
|:--|:--|:--|
| `ddrk_auto` | both | Picks per family and step count. Recommended. |
| `ddrk_cosine` | FM | Cosine decay |
| `ddrk_beta` | FM | Shaped by `beta_a` / `beta_b`; defaults 2.0 / 1.0 give Karras-like shrinking steps |
| `ddrk_flow_linear` | FM | Linear with shift |
| `ddrk_flow_cosmos` | FM | Cosmos-style tail |
| `ddrk_fewstep` | FM | Aggressive shift for 4-8 steps |
| `ddrk_edm_karras` | EDM | Karras rho=7 |
| `ddrk_edm_poly` | EDM | Polynomial tail |
| `ddrk_edm_simple` | EDM | Exponential |

---

## Parameters

### Core

| Parameter | Effect |
|:--|:--|
| `integrator` | `auto` / `hc2` / `rk4` / `heun` / `euler`. At <=6 steps this is forced to euler unless HC2 is selected; the console reports any override. |
| `sde_strength` | Ancestral SDE amount. **FM only** — silently ignored on EDM, which uses `s_churn`. |
| `sharpness` | Final-step perceptual sharpen. **FM only** — sharpening is disabled on EDM by design. |
| `saber_fusion` | Spatial/temporal stabilization weight. 0 disables the module entirely. |
| `momentum_beta` | AB2 derivative extrapolation. 0 = plain integrator. Ignored by HC2, which does this analytically. |
| `dyn_thresh_percentile` | Percentile latent limiter. 1.0 = off. Engages only below 40% (FM) / 30% (EDM) of sigma_max. |
| `latent_rescale` | Attenuates values beyond ~2 std from the per-image mean. **Not** classical CFG-rescale. 0 = off. |

### HC2

| Parameter | Effect |
|:--|:--|
| `limiter_kappa` | Slope limiter strength. 1.0 means the correction may at most double or cancel the step, never reverse it. Lower to 0.5-0.7 if high CFG still blows out highlights. |
| `hc2_max_order` | 2 (default) or 3. Third order uses two past evaluations and one extra call to bootstrap. |
| `hc2_corrector` | *Experimental.* 0 = off. Otherwise spends a second call on steps whose correction exceeds this fraction of the first-order step. No measurable effect in testing. |
| `sigma_adapt` | *Experimental.* 0 = off. Moves intermediate sigmas to equalise estimated error; step count and endpoints unchanged. No measurable effect in testing. |

### EDM

| Parameter | Effect |
|:--|:--|
| `s_churn`, `s_tmin`, `s_tmax`, `s_noise` | Karras Alg. 2 churn. EDM only; `s_churn = 0` disables. |

### Other

| Parameter | Effect |
|:--|:--|
| `saber_mode` | `auto` treats a 5D latent with F>1 as video. |
| `use_ema_saber`, `ema_decay` | Temporal EMA for video SABER. |
| `sde_seed` | -1 derives the seed from the global RNG, reproducible from the workflow seed. |
| `content_aware` | Edge-gated SABER fusion. Turn off for pixel art and flat-shaded styles. |
| `auto_optimize` | Caps or disables enhancers by compute budget on FM. Does not touch steps or CFG. |
| `smart_defaults` | Sets scheduler, shift, integrator and enhancer levels from the detected family. Does not touch steps or CFG. |
| `debug_mode`, `debug_tag` | Per-step telemetry to disk. Off by default, zero overhead when off. |

> **A note on two names.** `dynamic_threshold` clips peaks but deliberately omits the renormalization step of Imagen-style dynamic thresholding, because dividing the whole latent by the percentile shifts global contrast on every step it fires. `latent_rescale` cannot be classical CFG-rescale: ComfyUI combines the conditional and unconditional predictions before the sampler is called, so the two tensors that method needs are not available here.

---

## Telemetry

Enable `debug_mode` and the sampler writes three files to your ComfyUI output folder: a JSON with everything, a CSV with one row per step, and a plain-text summary.

Per step it records the phase, the integrator chosen, curvature, model calls, which of SDE / SABER / sharpen / churn / clamp / thresholding fired and how far each moved the tensor, HC2's order, limiter activity and error ratio, latent statistics, NaN/Inf flags and wall time. The run header records every resolved parameter, the full sigma schedule, and — from the Unified node — cfg, seed, denoise and scheduler.

This is the most useful part of the project for anyone modifying it. Most of what 1.6.0 fixes was found by reading these logs, not by reading the code: a parameter that never fired, a mask reporting the same value on every step, a step count one lower than requested.

---

## Suggested starting points

Starting points, not tuned optima. Only the integrator guidance comes from a controlled comparison.

**Flow Matching**

| Parameter | Value |
|:--|:--|
| Steps / CFG | Checkpoint-dependent. Distilled: 4-8 / ~1. Base: 20-40 / 1-4. |
| Scheduler | `ddrk_auto` |
| Integrator | `hc2`, or `auto` |
| SDE strength | 0.00-0.08 |
| Sharpness | 0.10-0.15 |
| SABER fusion | 0.00-0.15 |

**EDM**

| Parameter | Value |
|:--|:--|
| Steps / CFG | 20-30 / 7-8 base; 4-8 / 1-2 turbo |
| Scheduler | `ddrk_auto` -> `ddrk_edm_karras` |
| Integrator | `auto` or `heun` |
| `s_churn` | 0, or 5-15 to try |
| Sharpness | No effect on EDM |
| SABER fusion | 0.20 |

**Video (5D latents)** — `saber_mode` = `video` or `auto`, `saber_fusion` 0.20-0.35, `ema_decay` 0.7.

---

## Known limitations

- **RK4 costs 4 model calls per step, Heun 2, HC2 and Euler 1.** Compare at equal call count, not equal steps.
- **<=6 steps forces euler** unless HC2 is selected. The console reports the override.
- **`sharpness` does nothing on EDM.** The parameter is shared across families; the feature is not.
- **HC2 shows no advantage on EDM** at equal compute.
- **Steps and CFG are never auto-detected.** They depend on training, LoRA and distillation.
- **Enhancers are not individually ablated.** SABER, SDE and sharpening are on by reputation, not by measurement.
- **No unit tests.** Schedulers, the ancestral split and the AB2 extrapolation have all been exercised on real generations, but none has an automated test.

---

## Changelog

### v1.6.0

**New — HC2 integrator.** Exponential multistep, second order at one model call per step, with a slope limiter and optional measured-error third order. See [The HC2 integrator](#the-hc2-integrator).

**Crashes and dead features**
- `ddrk_beta` raised `NotImplementedError` — PyTorch's Beta distribution has no `icdf`. Replaced with the closed-form Kumaraswamy inverse CDF. Default shape changed from (2, 5), whose last step covered 42% of the sigma range, to (2, 1), which shrinks monotonically. `beta_a` / `beta_b` are now reachable from the UI; previously no caller passed them.
- `denoise = 0.0` was selectable and divided by zero before sampling started.
- `ddrk_anima` appeared in both scheduler dropdowns with no implementation behind it. Removed from the UI, still accepted from saved workflows.
- `sharpness` never applied on any model — the final-step test compared the loop index against `total_steps - 1`, which the loop never reached.

**Schedules and stepping**
- Flow Matching schedules ended with a duplicated zero, so a 20-step request ran 19 steps and an 8-step request ran 7. Phase 3's budget was computed including the phantom step, which cost it a step at 20 and eliminated it entirely at 8.

**Numerics**
- Curvature is now dimensionless. It previously carried units of 1/sigma, averaging 0.60 on EDM and 17.2 on FM while being compared against fixed cutoffs.
- `rk4` no longer divides by the sigma floor on the terminal step; Euler is exact at that endpoint anyway.
- EDM soft clamp bound is `max(4*sigma, 10)` instead of a constant 10.
- The LoG edge mask thresholds against median-estimated background curvature instead of a fixed percentile, which had reported exactly 0.1976 mean coverage on every step of every model.
- `dynamic_threshold`'s early-out is now the exact algebraic no-op condition rather than a hard-coded magnitude.
- Fixed-gain sigmoid gates replaced with a scale-invariant z-score gate.

**Reproducibility**
- EDM churn drew from the global RNG while the SDE had its own generator — and with `sde_seed = -1` that generator was created unseeded, so SDE runs were not reproducible at all. Both now share one generator, seeded from the global RNG.
- The LoG subsample for large latents now uses a fixed-seed generator.

**Naming and honesty**
- `cfg_rescale` -> `latent_rescale`; the old key is still accepted.
- `dynamic_threshold` documented as a percentile limiter, not Imagen dynamic thresholding.
- Integrator overrides print what they changed instead of silently discarding an explicit choice.
- SABER is skipped, and reported as not fired, when `saber_fusion` is 0.
- EDM profiles no longer recommend a `sharpness` value.
- FM profile integrator changed from a hard `euler` to `auto`.
- Enhancer caps key off the estimated model-call budget rather than the step index, so equal-cost configurations get equal treatment.

**Performance**
- Curvature is computed only when the integrator is `auto` and only in phase 1, removing a GPU->CPU sync per step otherwise.
- Triplicated integrator dispatch collapsed into one function.

### Earlier

| Version | Summary |
|:--|:--|
| v1.5.2 | AB2 extrapolation replacing EMA momentum, ancestral SDE split, z-score gating, `s_tmin`/`s_tmax` |
| v1.5.0 | Architecture introspection, `smart_defaults`, `auto_optimize`, per-step FM clamp removed |
| v1.4.x | EDM path tuning, churn, multi-scale sharpen, content-aware SABER |
| v1.3 | SABER 5D crash fix for single-frame video latents |
| v1.2 | 5D reshape fix, LRU cache bounds, `sde_seed` exposed |
| v1.1 | Removed invalid FSAL-RK4, Karras rho=7 schedule, momentum scope fix |
| v1.0 | Original prototype: hybrid RK4/Heun, SABER, SWT sharpening |

---

## Credits

**Concept, prototype and direction** — [HVOSTOVSKY](https://github.com/HVOSTOVSKY). The phase-based sampler design, SDE noise masking and SABER stabilization are his.

**Production hardening (v1.1-v1.5.2)** — iterative audits and implementation by **Kimi** (Moonshot AI).

**Telemetry, correctness pass and HC2 (v1.6.0)** — instrumentation, log analysis, integrator design and fixes by **Claude** (Anthropic).

HC2's core follows the exponential-integrator line of work — Lu et al., *DPM-Solver++* (2022) and Zhao et al., *UniPC* (2023). The ancestral noise split follows Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models* (2022).

---

## Issues

Open an [issue](https://github.com/HVOSTOVSKY/DDRK-Omega-Sampler/issues) with the model name, your settings, and either the traceback or comparison images. For quality problems rather than crashes, enable `debug_mode` and attach the JSON — it makes most problems diagnosable without guesswork.

---

<div align="center">

**MIT License**

**«One sampler to rule them all — from Flux to SDXL.»**

</div>


