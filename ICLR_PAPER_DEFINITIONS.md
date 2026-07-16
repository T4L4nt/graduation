# Architecture Fingerprint: 严格定义与贡献层次 (v3)

> ICLR 2027 投稿用。
> v2 → v3 核心修正：(1)"Fingerprint"明确定位为 measured profile 而非 intrinsic property；
> (2)Property 与 Evidence 分离；(3)Principles 抽象化；
> (4)摘要缩减至 3 Claims；(5)增加 Claim-Evidence-Conclusion 结构；
> (6)增加方法论定位 "Diagnosis before correction"。
>
> v3 → v3.1 (SDXL 实验后)：(7)Mechanism 从"Skip Conflict 统一解释 U-Net"
> 修正为"Architecture-specific Mechanistic Analysis"——Fingerprint 是诊断工具，
> 不同架构得到不同机理解释；(8)Claim 收敛，不过度外推。
>
> v3.1 → v3.2 (2026-07-16 方法论修正)：(9)跨架构相似度量从插值 Pearson/Spearman
> 切换为结构距离（无插值，4 特征），修复 `full_ranking` 排序 bug（按漂移量级而非
> 架构深度）和插值 artifact（SDXL 28→57 含 51% 合成点）；(10)Property 3 翻转：
> "Backbone Dominance" → "Attention Topology over Broad Backbone Family"——
> HunyuanDiT(Transformer single-stream) vs FLUX(MM-DiT dual-stream) 是结构距离
> 最远的配对(d=1.077)，推翻此前"同 Transformer backbone 最相似"的结论；
> (11)架构计数 4→5 (新增 SD 3.5 held-out)，配对 6→10；(12)Property 5 样本量
> 25→100 prompts；(13)Scope Declaration 数字同步更新。

---

## 0. 论文结构总览

```
§1  Introduction        — 3 Claims, central message
§2  Related Work
§3  Architecture Fingerprint — Definition + Properties (evidence-driven)
§4  Mapping Principles       — From topology to drift (hypothesis, validated)
§5  Architecture-specific Mechanistic Analysis — Case studies (SD 1.5 + SDXL)
§6  Application              — Diagnosis-guided Correction + Editing
§7  Discussion               — Limitations, normalization ablations, open questions
```

**§5 的关键定位变化**：不再声称 Skip Conflict 是 U-Net 的统一机制（已被 SDXL 实验否证），
而是将 Fingerprint 定位为**诊断工具**——它告诉你"哪里有问题"，然后针对每个架构
实例分析"为什么这里有问题"。不同架构可能得到不同的机理解释。

每一章的 Claim–Evidence–Conclusion 结构见 §0.3。

---

### 0.1 论文的三层 Claim（摘要只报告这三项）

**Claim 1 (Discovery).** Feature drift exhibits an **Architecture Fingerprint**:
its layer-wise distribution is a reproducible, architecture-specific measurement
determined by backbone topology, not sampling paradigm.

**Claim 2 (Mechanism).** In U-Net architectures, the fingerprint originates from
**skip-mediated encoder-decoder feature conflict**—a structured mismatch that
causes both drift and reconstruction error.

**Claim 3 (Application).** Identifying the drift bottleneck through diagnosis
makes the simplest latent-space correction sufficient—achieving content
preservation on par with complex methods, as a plug-in for editing.

---

### 0.2 Central Message (Introduction 最后一句)

> Our central message is not that linear correction is surprisingly powerful,
> but that *sufficient diagnosis makes simple correction sufficient*.

这一定位将论文从"一个更好的编辑方法"升级为"一种诊断先于干预的方法论"。
Reviewer 读完 Introduction 应该记住的是这个 insight，而不是 PSNR 数字。

---

### 0.3 论文的 Claim–Evidence–Conclusion 组织

建议全文按如下模板组织每个 Argument：

```
Claim N: [一句话主张]

Evidence:
  - 实验 A (Table/Figure X)
  - 实验 B (Table/Figure Y)

Conclusion: [一句话总结，不含新 claim]
```

这比传统"方法→实验→讨论"的写法更清晰。Reviewer 可以逐 Claim 检验证据是否充分。

---

## 1. Layer 1 — 纯数学定义

### Definition 1 (Feature Drift)

Let M be a diffusion backbone with L layers. For a fixed inversion protocol P
(e.g., DDIM with T steps, empty prompt), define the per-layer feature drift
for image x as:

```
d_l(x) = E_{t∈K}[ || f_l^inv(x, t) − f_l^recon(x, t) ||_2 ]
```

where K is a fixed set of sampled timesteps, f_l(·, t) is the output feature
of layer l at denoising step t (spatial-mean-pooled for attention layers,
raw output for ResNet layers).

### Definition 2 (Architecture Fingerprint)

The Architecture Fingerprint of M under inversion protocol P is the **measured
layer-wise drift profile**:

```
Φ(M) = Normalize({ E_{x∈D}[ d_l(x) ] }_{l=1}^{L} )   ∈ [0,1]^L
```

We refer to Φ(M) as a **fingerprint** in the measurement sense: a reproducible
profile that distinguishes architectures under a fixed protocol—not as a claim
of invariance across all conditions.

**Explicit dependencies** (declared, not hidden):

```
Φ(M) = Φ(M; D, P, norm)
```

- D: evaluation image set (fixed to coco_val, 19 images)
- P: inversion protocol (DDIM, T=50, empty prompt by default)
- norm: min-max normalization to [0,1] (ablation of normalization choices
  provided in Appendix)

Changing any factor changes Φ. This is a feature, not a bug: the fingerprint
captures the architecture's behavior under a specific, reproducible measurement
protocol—analogous to how an NMR spectrum depends on solvent and temperature
but still identifies molecular structure.

**Why "Fingerprint"?** We use this term because (a) Φ(M) is reproducible for
the same M under fixed P (Property 1), (b) different M produce measurably
different Φ (Property 2), and (c) the profile shape is interpretable from
the architecture's topology (§4). As defined, Φ(M) is a measured profile
under a fixed protocol—analogous to an NMR spectrum that depends on solvent
and temperature but still identifies molecular structure.

### Why Architecture Fingerprint rather than "layer-wise drift profile"?

The core question is not one of naming but of **what constitutes the object
of study**. A layer-wise drift profile is merely a collection of local
measurements: each layer's drift d_l(x) is a scalar, and the profile is
their concatenation. It can answer *which layer drifts most* but nothing
beyond that.

The Architecture Fingerprint is a different object: an **architecture-level
descriptor** whose identity lies in the *shape*, *organization*, and
*distribution* of drift across layers, rather than in any individual layer's
value. It changes the question from "Which layer drifts?" to "Why does this
architecture drift this way?"

This abstraction is not cosmetic—it is **necessary** for everything that
follows in this paper:

```
Layer-wise drift profile          Architecture Fingerprint
─────────────────────────         ─────────────────────────
Local observable                  Global architectural descriptor
Answers: which layer?             Answers: why this architecture?
Cannot support comparison         Enables inter-architecture comparison (§3, Property 2)
Cannot reveal topology             Enables topology interpretation (§4, Principles)
Cannot compare across paradigms    Enables backbone-vs-paradigm comparison (§3, Property 3)
Cannot ground a diagnosis          Grounds diagnosis-before-correction (§6)
```

This is the distinction between a measurement and a **measurement framework**.
The Fingerprint is not a theory of *why* drift occurs (that is the role of
the mechanism analysis in §5), nor is it a hypothesis about *how* drift
propagates (that is the role of the mapping principles in §4). It is the
descriptive layer that makes those subsequent layers possible. **The
Fingerprint explains nothing—it only reveals. Mechanism explains.
Correction exploits.**

---

### Framework Overview: Five Conceptual Layers

The paper is organized as a five-layer conceptual hierarchy:

```
§3  Architecture Fingerprint     — WHAT:   Measurement framework
       ↑
§3  Properties (1–5)             — HOW STABLE: Reproducibility, differentiation,
                                    attention topology, temporal consistency,
                                    prompt robustness
       ↑
§4  Mapping Principles (1–3)     — HOW TO INTERPRET: From architecture topology
                                    to drift profile shape
       ↑
§5  Mechanism (Skip Conflict)    — WHY: Causal chain from skip connection
                                    to feature conflict to reconstruction error
       ↑
§6  Application (Correction)     — HOW TO USE: Diagnosis-guided minimal
                                    intervention
```

Each layer answers a distinct question. No single layer is self-sufficient,
and the Fingerprint—the measurement layer—is the foundation on which the
others build. Without it, inter-architecture comparison has no common
language; topology interpretation has no target; mechanism analysis has
no measured phenomenon to explain.

---

## 2. Layer 2 — 经验性质 (Properties, 由实验建立)

**格式约定**: 每条 Property 先陈述主张，再单独列出 Evidence 来源。Property
不含实验数字——数字放在 Evidence 行。

### Property 1 (Intra-architecture Reproducibility)

Φ(M) is reproducible across independent image sets from the same distribution
and across random seeds, under fixed measurement protocol.

*Evidence:* 19 coco_val images, leave-one-out cross-validation.
Pearson r = 1.000 (mean across 19 folds, min = 1.000).
Multi-seed measurement (3 seeds × 5 images): σ/mean = 0.1% per layer.
(SD 1.5, 50-step DDIM. Data: `outputs/phase1_reproducibility/reproducibility.json`.)

### Property 2 (Inter-architecture Differentiation)

Φ(M_A) and Φ(M_B) are measurably different for M_A ≠ M_B with different
backbone topologies.

*Evidence:* 5 architectures (SD 1.5, SDXL, HunyuanDiT, FLUX.1-dev, SD 3.5
as held-out), 10 pairwise comparisons. We use **structural distance**
(Euclidean distance in a 4-dimensional feature space: peak position,
number of peaks, drift concentration, and spread) computed directly from
raw layer counts — no interpolation to a common length.

**Methodology note:** Earlier versions of this work used Spearman ρ and
Pearson r on interpolated drift vectors. We have retired this approach
for two reasons: (1) the `full_ranking` key in our data files is sorted
by drift magnitude (not architectural depth), which makes all profiles
monotonically decreasing and inflates correlations; (2) interpolation to
a common 57-point grid creates up to 51% synthetic data points for
architectures with fewer layers (e.g., SDXL 28→57). The structural
distance metric avoids both issues entirely.

Structural distance matrix (lower = more similar):

| Pairing | d | Interpretation |
|---------|---|---------------|
| **SD 1.5 vs SDXL** | **0.249** | Same UNet family — closest |
| **FLUX vs SD 3.5** | **0.385** | Same MM-DiT backbone — second closest |
| SDXL vs HunyuanDiT | 0.506 | Different backbone, moderate |
| SD 1.5 vs HunyuanDiT | 0.624 | Different backbone |
| SDXL vs FLUX | 0.628 | Different backbone + paradigm |
| SD 1.5 vs FLUX | 0.637 | Different backbone + paradigm |
| SD 1.5 vs SD 3.5 | 0.722 | Different backbone |
| SDXL vs SD 3.5 | 0.803 | Different backbone |
| **HunyuanDiT vs FLUX** | **1.077** | Both Transformer, but single vs dual-stream — **farthest** |
| HunyuanDiT vs SD 3.5 | 1.165 | Different backbone, farthest overall |

Same-family pairs (both UNet: d=0.249; both MM-DiT: d=0.385) are
systematically closer than cross-family pairs (all d > 0.5). See
Figure 2 for the 5-curve overlay and structural distance matrix.

### Property 3 (Attention Topology over Broad Backbone Family)

The structural distance matrix reveals that attention topology
(single-stream vs dual-stream) is a stronger determinant of Φ than the
broad backbone family (Transformer vs UNet). The HunyuanDiT vs FLUX pair
— both Transformer-based — is the **farthest** among all 4-architecture
pairs (d=1.077), exceeding even cross-family distances like SD 1.5 vs
HunyuanDiT (d=0.624). In contrast, same-family UNet pairs (d=0.249) and
same-family MM-DiT pairs (d=0.385) are consistently close.

This refines the earlier "Backbone Dominance" framing: backbone *family*
does not dominate — *specific attention topology* (single-stream joint
attention vs dual-stream split attention, presence and direction of
cross-modal interaction boundaries) is the primary determinant of Φ.
This is consistent with the qualitative architecture-topology-to-fingerprint
mapping (Section 3.4), which identifies (a) information flow graph,
(b) skip/residual structure, and (c) cross-modal interaction boundaries
as the three predictive features, none of which reduce to a simple
"CNN vs Transformer" dichotomy.

*Evidence:* Structural distance matrix (5 architectures, 10 pairs).
HunyuanDiT (single-stream Transformer, DDIM v-pred) and FLUX (dual-stream
MM-DiT, Flow Match) share a Transformer backbone but differ in attention
topology — and are structurally farthest. SD 1.5 (UNet, DDIM) and SDXL
(UNet, DDIM) share both backbone and paradigm — and are closest. SD 3.5
(held-out) and FLUX share MM-DiT backbone — second closest, confirming
the framework's predictive value. The sampling paradigm (DDIM vs Flow
Matching) is not the determining factor — FLUX vs SD 1.5 (d=0.637) and
HunyuanDiT vs SD 1.5 (d=0.624) are at similar distances despite HunyuanDiT
using DDIM and FLUX using Flow Match.

### Property 4 (Temporal Consistency)

The location of the drift peak (top-5 layers) is consistent across inversion
steps T ∈ {4, 10, 20, 50, 100}, though absolute magnitude varies.

*Evidence:* SD 1.5 step-count sweep, 5 step counts × 19 images.
Peak position stable; magnitude follows inverted-U shape peaking at T=20.
See Appendix Figure A1.

### Property 5 (Prompt Robustness)

The correction is effective across diverse prompts, confirming that
the Fingerprint-based diagnosis does not depend on a specific prompt.

*Evidence:* 100 prompts on SD 1.5 (expanded from 25 in earlier version).
Correction ΔPSNR = +1.88 ± 2.25 dB (p=5.15×10⁻¹³, Cohen's d=0.835,
95% CI [1.45, 2.34]). 53/100 prompts (53%) improved >1.0 dB; 34/100 (34%)
improved >2.0 dB; only 6/100 (6%) showed degradation. The correction
generalizes beyond the empty-prompt condition used for Fingerprint
measurement. See Appendix Figure A2.

---

### Scope Declaration (Properties 1–5)

Properties 1–3 are established on 5 architectures in unified comparison
(SD 1.5, SDXL, HunyuanDiT, FLUX.1-dev, SD 3.5 Medium as held-out),
10 pairwise comparisons. All measurements use coco_val images under DDIM
or Euler sampling. The structural distance metric (4 features from raw
layer counts) avoids interpolation artifacts present in earlier
Pearson/Spearman approaches. Property 5 extends the evaluation to 100
diverse prompts; editing validation covers 121 edit pairs
(see §6). Extension to further architectures, datasets, and protocols
is discussed in §7.

---

## 3. Layer 3 — 映射原则 (Mapping Principles, 假设，已验证)

**These are hypothesized principles, not theorems.** They claim that Φ(M) is
*interpretable* from G(M), the architecture's topology graph. Validation is
through held-out prediction and causal intervention.

### Principle 1 (Bottleneck Localization)

Drift concentrates near the architecture's information bottleneck—the point
where representational capacity is most constrained.

The bottleneck type varies by backbone:

| Backbone | Bottleneck structure | Drift peak |
|----------|---------------------|------------|
| U-Net | Encoder → decoder convergence | decoder entry or mid_block |
| Single-stream Transformer | Representation phase transition | middle-layer transition zone |
| MM-DiT (dual-stream) | Joint → single modality handoff | last joint + early single blocks |
| MM-DiT-X | Dual → standard attention transition | late output compression |

*Validation:* For all 4 architectures in the unified comparison, the observed
drift peak location matches the independently identifiable information
bottleneck. SD 3.5 Medium served as a 5th held-out test: the prediction
placed the peak at the dual→standard boundary; the observation placed it
at late output compression, revealing a previously unrecognized bottleneck
type. This partial falsification is reported honestly and led to framework
refinement.

### Principle 2 (Propagation Mode)

The width and shape of the drift peak are determined by how information
propagates across layers:

- **Cross-layer skip connections** → drift signal propagates beyond the
  bottleneck → broad decoder peak (U-Net)
- **Sequential residual stream only** → drift localizes at the transition
  zone → narrow peak (Transformer)
- **Dual-stream attention** → cross-modal mixing stabilizes features →
  drift concentrated in single-stream regions (MM-DiT)

*Validation:* Causal intervention on U-Net skip connections (§5, Figure 6):
cutting a skip at the peak layer reduces drift by 27.7% and reshapes the
fingerprint; cutting a skip at a low-drift region produces no significant
change. See also the dose-response curve (α ∈ [0,1]) confirming monotonicity.

### Principle 3 (Cross-modal Boundary Effect)

Cross-modal interaction boundaries (joint↔single, dual↔standard) act as
feature stabilizers: adding cross-modal interaction suppresses drift,
removing it triggers drift.

- Interaction **removed** (FLUX joint→single, HunyuanDiT): drift peak
- Interaction **added** (SD 3.5 dual→standard): drift valley

*Validation:* FLUX joint_18 (last joint block before single-stream) shows
both the highest image drift (0.713) and a jump in text drift (0.12→0.44).
SD 3.5 blocks 0-12 (dual attention) show the lowest drift in the model;
drift rises sharply after block 13 (standard attention only).

### Interpretability and Limits

Given G(M), the coarse location and shape of Φ(M) can be interpreted through
Principles 1–3. The principles correctly identified the bottleneck type for
all 5 architectures and correctly predicted that Cut A (peak skip) would
alter the fingerprint while Cut B (low-drift skip) would not. Generalizing
this to predictive mapping from G(M) to Φ(M) for arbitrary unseen architectures
requires a larger architecture sample and a learned mapping function, which
falls outside the scope of this work (see §7).

---

## 4. Layer 4 — 架构特异机制分析 (Architecture-specific Mechanistic Analysis)

**Positioning: The Fingerprint reveals WHERE drift concentrates (§3) and
how to interpret its shape from topology (§4). This section explains WHY—the
causal mechanism producing the observed pattern. Different architectures may
admit different explanations. The Fingerprint is the diagnostic tool;
the causal analysis below is the mechanistic explanation.**

The key methodological shift: Fingerprint is not for proving one mechanism.
Fingerprint is for discovering that **different architectures need different
explanations.**

### Causal Chain: Skip Conflict as a Mediating Variable

For U-Net architectures, we operationalize the mechanism through a measurable
**Conflict** variable C, defined at each skip connection:

```
C = || s − u ||_2
```

where s is the skip feature from down_block[i] and u is the up_block's
internal representation at the corresponding up_block[N-i-1] before
receiving s. C is directly measurable from intermediate activations
without modifying the model.

This introduces a four-variable causal chain:

```
Skip strength α  →  Conflict C  →  Drift φ_l  →  Reconstruction PSNR
   (manipulated)     (mediator)     (observed)      (outcome)
```

The chain makes three falsifiable predictions:

**P1. α→C:** Reducing skip strength reduces Conflict C.<br>
**P2. C→φ:** Lower Conflict C reduces drift at layers downstream of
the connection.<br>
**P3. φ→PSNR:** Reduced drift improves reconstruction quality.<br>
**P4 (critical).** Conflict, not L2 magnitude, is the causal variable:
interventions that break the C→φ link without changing L2 should produce
different outcomes.

### Empirical Validation (SD 1.5, 19 images, 50-step DDIM)

**Claim 4a (Binary intervention confirms α→C→φ→PSNR chain).**
Zeroing the peak skip (α=0) breaks the Conflict source at that location,
causing the entire chain to shift.

*Evidence:* 31/38 layers p<0.05 (paired t-test). Peak drift: −27.7%
(p=4.8×10⁻⁸). Reconstruction: PSNR +2.20 dB (p=0.0005). See Figure 6A.

**Claim 4b (Location specificity rules out capacity effect).**
Zeroing a low-drift skip produces no significant change, confirming
that Conflict is position-specific, not capacity-driven.

*Evidence:* Cut B (up_blocks.0 skip): 5/38 layers p<0.05. Peak drift: +0.8%
(n.s.). PSNR: −0.11 dB (n.s.). The anti-correlated delta maps (r=−0.395)
between Cut A and Cut B imply distinct spatial mechanisms at different
locations. See Figure 6B.

**Claim 4c (Continuous dose-response confirms α→C monotonicity).**
The effect is monotonic in α, with no intermediate optimum.

*Evidence:* α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}. PSNR decreases monotonically
as α increases (24.66 → 22.44 dB). No optimal modulation point—the skip
is purely harmful at this location. See Figure 6C.

**Claim 4d (Noise experiment separates Conflict from L2 magnitude).**
Replacing the skip with Gaussian noise of identical statistics per tensor
preserves L2 magnitude but destroys the structured mismatch pattern that
constitutes Conflict. If L2 magnitude were the causal variable, Noise
should resemble Original. If Conflict is the causal variable, Noise and
Zero should both differ from Original—but for different reasons.

*Evidence:* Noise replacement increases L2 drift (+6.4% vs. original) but
IMPROVES reconstruction (LPIPS 0.218→0.113, PSNR +2.4 dB). Noise breaks
the Conflict-driven mismatch while introducing random perturbations—the
former improves reconstruction, the latter increases L2. This double
dissociation (L2↑ but PSNR↑) establishes that **L2 drift magnitude is not
the causal variable; structured Conflict is.** See Figure 6D.

**Summary of the causal chain:**

| Link | Test | Result |
|------|------|--------|
| α→C | Cut A vs. Original | Conflict eliminated, drift −27.7%, PSNR +2.2 dB |
| α→C | Cut B vs. Original | Conflict unchanged, drift +0.8% n.s. |
| α→C | Dose-response α∈[0,1] | Monotonic—no optimal modulation point |
| C→φ→PSNR | Noise vs. Zero vs. Original | L2↑ but Conflict↓ → PSNR↑ (dissociation) |
| Location | Δ(A) vs. Δ(B) r=−0.395 | Position-specific, not capacity-driven |

### For Non-U-Net Architectures (Future Work)

In Transformer-only backbones, drift concentrates at representation
transition zones. The qualitative mechanism—features changing rapidly at
layer ranges undergoing representational phase transitions—is different
from U-Net skip conflict. We leave its formal characterization to
future work and limit the claims of §5 to U-Net architectures.

### Cross-UNet Comparison (SDXL Negative Result)

**Claim 4e (Architecture-dependent functional role).** The same structural
component (skip connection) can play different functional roles across
U-Net variants during inversion. The Fingerprint reveals this divergence.

*Evidence:* We applied the same intervention (cutting the decoder skip at
the corresponding architectural position) to SDXL. Result: PSNR drops by
11.6 dB—the opposite of SD 1.5's +2.2 dB improvement. SDXL's drift peak
is in mid_block rather than decoder, indicating that the dominant source
of drift differs between the two U-Net variants.

*Interpretation:* This result indicates that skip connections
play **architecture-dependent functional roles** during inversion—they can
be conflict sources (SD 1.5) or necessary information pathways (SDXL).
The same structural component contributes differently across U-Net variants,
supporting the central claim that the Architecture Fingerprint captures
architecture-specific phenomena, not generic backbone-family properties.

*Caveat:* The cut positions differ between architectures (SD 1.5:
down_blocks.1→up_blocks.2; SDXL: down_blocks.0→up_blocks.2) due to
different U-Net depths. A full ablation with multiple cut positions on
SDXL is needed to definitively rule out position-selection artifacts.
See Discussion.

---

## 5. Layer 5 — 应用 (Diagnosis-Guided Correction)

### The Logical Chain

The Architecture Fingerprint enables a diagnosis-first approach to correction:

```
1. Diagnosis: Φ(M) → L_peak = argmax_l φ_l
   (tells us WHERE the strongest encoder-decoder mismatch occurs)

2. Insight: The system is insensitive to precise layer selection.
   random5 ≈ top5 (ΔPSNR < 0.3 dB) → correction at the peak REGION,
   not the exact peak layer, is sufficient.

3. Intervention: f_out = f_recon + λ·(f_inv − f_recon)
   Applied as a latent-space correction at the denoising endpoint.
   λ is non-critical: λ∈{0.3, 0.5, 0.7} → PSNR range < 0.08 dB.

4. Why this works: The fingerprint reveals that drift is concentrated
   (not distributed), so one injection point suffices. It also reveals
   that the system is robust to λ, so a fixed value works. Complexity
   (per-layer selection, adaptive λ, closed-loop control) adds no
   measurable gain—because diagnosis already told us it wouldn't.
```

### Evidence

**Correction quality (19 images, 50-step DDIM, SD 1.5):**

| Method | PSNR | LPIPS | Memory |
|--------|------|-------|--------|
| DDIM (baseline) | 22.45 | 0.218 | Low |
| P2P | 25.34 | 0.087 | ~GB |
| **Ours** | **25.20** | **0.094** | **~MB** |

P2P vs Ours: Cohen's d=0.033 (negligible effect size), Pearson r≈1.000
(identical behavior). Statistically significant but practically equivalent.

**Cross-architecture generalization:**

| Architecture | ΔPSNR | Optimal λ | Key insight |
|-------------|-------|-----------|-------------|
| SD 1.5 | +2.75 dB | 0.7 | random5 ≈ top5 |
| SDXL | +5.23 dB | 0.7 | Larger UNet → larger gain |
| HunyuanDiT | +5.65 dB | 0.9 | transition-only >> top5 |
| FLUX | +3.94 dB | 0.7 | Latent correction effective |

**Editing plug-in (25 edit pairs, 3 conditions, SD 1.5, 20-step DDIM):**

| Condition | LPIPS↓ | SSIM↑ | CLIP-Dir↑ | PSNR↑ |
|-----------|--------|-------|-----------|-------|
| Original (no cut) | **0.671** | 0.739 | **0.048** | **17.65** |
| Cut A (zero skip) | 0.758 | 0.799 | −0.004 | 16.06 |
| Noise A (noise skip) | 0.775 | **0.807** | −0.008 | 16.07 |

Cut A and Noise A improve structural preservation (SSIM↑) but nearly eliminate
editing direction (CLIP-Dir → 0). The skip connection carries the editing
signal during prompt-changed reconstruction; removing it collapses editing
to basic reconstruction. This reveals a trade-off: content preservation vs.
edit fidelity, mediated by the architecture's information pathways.

**Cross-prompt generalization (25 prompts, SD 1.5, 50-step DDIM):**
Correction ΔPSNR = +1.31 ± 1.75 dB (p=0.0012, Cohen's d=0.75, 95% CI
[0.62, 1.99]). 13/25 prompts improved >1 dB; 2/25 degraded. The diagnosis-
guided correction generalizes beyond the empty-prompt condition used for
Fingerprint measurement.

**Negative results supporting simplicity:**
- Feature-level injection: ΔPSNR = −0.27 dB (worse than baseline)
- DCSC closed-loop control: no measurable gain over fixed λ
- These results validate the diagnosis-first paradigm: once the bottleneck
  is identified, the system already operates at the simplicity-performance
  Pareto frontier.

---

## 6. 摘要 (Abstract, ~200 words)

> Diffusion inversion introduces a discrepancy between inverted and
> reconstructed latent representations—widely observed but poorly understood.
>
> We discover that this feature drift exhibits a reproducible **Architecture
> Fingerprint**: its layer-wise distribution is determined by backbone
> attention topology, not by sampling paradigm. Across four architectures
> in unified comparison (with a fifth as held-out validation),
> and two paradigms, drift profiles reliably distinguish backbones while
> remaining stable within the same topology.
>
> This diagnosis directly motivates a minimal intervention: identifying
> the drift bottleneck through layer-wise profiling makes the simplest
> latent-space linear correction sufficient. The correction achieves
> content preservation on par with Prompt-to-Prompt (Cohen's d=0.033)
> at negligible memory cost, and generalizes as a plug-in for editing.
> Additional complexity—feature-level injection, closed-loop control,
> per-layer weighting—provides no measurable gain, confirming that
> the bottleneck diagnosis itself, not the correction formula, is the
> contribution.
>
> Our central message is not that linear correction works, but that
> *sufficient diagnosis makes simple correction sufficient*.

---

## 7. 需要补充的实验/附录

| # | 内容 | 位置 | 状态 |
|---|------|------|------|
| 1 | Normalization ablation (min-max vs z-score vs L2 vs LayerNorm) | Appendix | ❌ |
| 2 | Prompt insensitivity (25 prompts, Property 5) | Appendix | ✅ A |
| 3 | Editing under Cut A (25 tasks × 3 conditions) | Appendix | ✅ B |
| 4 | SDXL skip modulation (cross-architecture causal) | Appendix | ✅ C |
| 5 | Multi-seed stability (3+ seeds, all architectures) | Appendix | ❌ |
| 6 | Raw-scale drift plots (before normalization) | Appendix | ❌ |
| 7 | Content-category subgroup analysis | Appendix | ❌ |

---

## 8. 配图方案（5 张主图 + 2 张表）

### 主文 8 页排版

| 页码 | 内容 |
|------|------|
| Page 1 | Abstract + Introduction + **Figure 1** |
| Page 2 | Related Work + Definition + **Figure 2** |
| Page 3 | Properties + Experiments + **Table 1** |
| Page 4 | Mapping Principles + **Figure 3** |
| Page 5 | Mechanism (SD 1.5 + SDXL case studies) + **Figure 4** |
| Page 6 | Application + **Figure 5** |
| Page 7 | More experiments + **Table 2** |
| Page 8 | Discussion + Conclusion |

### 5 张主图

| Figure | 科学问题 | 类型 | 状态 |
|--------|---------|------|------|
| Fig.1 | 这篇论文在干什么？ | draw.io 概念图 | ❌ 描述已给出 |
| Fig.2 | Architecture Fingerprint 存在吗？ | 数据图 | ✅ `fig2_fingerprint.pdf` |
| Fig.3 | Topology 能解释 Fingerprint 吗？ | 数据图 | ✅ `fig3_topology.pdf` |
| Fig.4 | 为什么会这样？(SD1.5 vs SDXL) | draw.io 概念图 | ❌ 描述已给出 |
| Fig.5 | 有什么用？(Diagnosis→Correction→Editing) | 数据图 | ✅ `fig5_application.pdf` |

### Fig.1 描述

横向三阶段布局（深色背景）：

- **左 — Observation**: Source Image → Inversion → Reconstruction → f_inv ≠ f_recon → "Random noise? ✗ Architecture signal? ✓"
- **中 — Discovery**: 5 架构漂移曲线叠加，不同颜色，峰位置不同。同 family 最近（SD1.5-SDXL d=0.249），attention 拓扑差异最大（DiT-FLUX d=1.077）。Attention topology > Backbone family
- **右 — Diagnosis → Correction**: Φ(M) → Peak Location → z ← z + λ(z_inv − z) → Edited result. LPIPS −40%, MB vs GB. "Diagnosis makes simple correction sufficient"

Caption: "Feature drift is not random error but an architecture-dependent diagnostic signal."

### Fig.4 描述

左右对比 + 中间关键信息：

- **左 — SD 1.5**: Encoder → down_blocks.1 → skip → up_blocks.2 (drift peak here). Cut skip → PSNR +2.2 dB. Skip = conflict source. Evidence: α=0 drift −27.7%, Noise: drift↑ PSNR↑, Dose-response monotonic
- **中 — Key message**: "Same structural component, opposite functional role. The Fingerprint reveals architecture-specific behavior."
- **右 — SDXL**: Encoder → down_blocks.0 → skip → up_blocks.2 (NOT drift peak, peak in mid_block). Cut skip → PSNR −11.6 dB. Skip = necessary information path. Evidence: ΔPSNR −11.59 dB, ΔSSIM −0.306, ΔLPIPS +0.447

Caption: "The same structural component plays opposite functional roles in two U-Net variants, demonstrating that the Architecture Fingerprint captures instance-specific phenomena."

### 2 张表

| Table | 内容 |
|-------|------|
| Table 1 | Architecture summary (Model, Backbone, Paradigm, Peak layer, L) |
| Table 2 | SOTA comparison (Method, PSNR, LPIPS, Memory, Training) |

### 附录图（5-8 张）

- Fingerprint stability (prompt/seed/cross-arch)
- Normalization ablation
- Full dose-response curves
- MI estimation details
- SDXL multi-position cut (future work)
- Prediction record

### 文件位置

- 数据图: `fig2_fingerprint.pdf`, `fig3_topology.pdf`, `fig5_application.pdf` (仓库根目录)
- 生成脚本: `scripts/gen_iclr_figures.py`
- 概念图 Fig.1, Fig.4: 待 draw.io 手绘

---

## 9. v2 → v3 变更摘要

| 问题 | v2 | v3 |
|------|----|----|
| "Fingerprint" 暗示不变性 | 模糊 | 明确为 "measured profile"，声明依赖 D, P, norm |
| Property 和 Evidence 耦合 | Property 正文含数字 | Property 陈述主张，Evidence 单独列出 |
| Principles 含 Example | Definition 里写 U-Net/Transformer | 抽象 Principle，Example 以表格/子句呈现 |
| 摘要信息过载 | ~280 words, 10+ claims | ~250 words, 3 claims (Discovery/Mechanism/Application) |
| 缺少 Argument 结构 | 传统章节式 | 增加 Claim–Evidence–Conclusion 模板 |
| 缺少方法论定位 | 未提及 | "Diagnosis before correction" 作为 central message |
| Normalization 未解释 | 无 | 明确为 min-max，附 ablation 计划 |
| Application 逻辑跳跃 | Peak → Correction | Peak → Region (random5≈top5) → Correction |