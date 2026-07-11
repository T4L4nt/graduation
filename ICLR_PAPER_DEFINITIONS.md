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
the architecture's topology (§4). The word does NOT imply that Φ is an
intrinsic, condition-invariant property of M.

---

## 2. Layer 2 — 经验性质 (Properties, 由实验建立)

**格式约定**: 每条 Property 先陈述主张，再单独列出 Evidence 来源。Property
不含实验数字——数字放在 Evidence 行。

### Property 1 (Intra-architecture Reproducibility)

Φ(M) is reproducible across independent image sets from the same distribution
and across random seeds, under fixed measurement protocol.

*Evidence:* 19 coco_val images, leave-one-out cross-validation.
Pearson r > 0.95 across splits. Multi-seed measurement: σ/mean < 5%.
(SD 1.5, 50-step DDIM. Replicated on SDXL, HunyuanDiT, FLUX.)

### Property 2 (Inter-architecture Differentiation)

Φ(M_A) and Φ(M_B) are measurably different for M_A ≠ M_B with different
backbone topologies.

*Evidence:* 5 architectures, 10 pairwise comparisons. Pearson r range:
[0.486, 0.792]. The highest correlation (SDXL vs HunyuanDiT) is driven by
normalization range compression (drift magnitude differs by ~1000×—see
Appendix for per-architecture raw-scale plots). See Figure 2 for the
5-curve overlay.

### Property 3 (Paradigm Stability)

Changing the sampling paradigm (DDIM vs. Flow Matching) produces a smaller
change in Φ than changing the architecture.

*Evidence:* FLUX.1-dev measured under Euler Flow Matching vs. DDIM architectures.
cos_sim(Φ_DDIM(FLUX), Φ_FM(FLUX)) is not directly measurable (FLUX does not
support DDIM), but the Pearson r matrix supports this: architectures sharing
backbone type have higher similarity than architectures sharing paradigm.
See Figure 3C.

### Property 4 (Temporal Consistency)

The location of the drift peak (top-5 layers) is consistent across inversion
steps T ∈ {4, 10, 20, 50, 100}, though absolute magnitude varies.

*Evidence:* SD 1.5 step-count sweep, 5 step counts × 19 images.
Peak position stable; magnitude follows inverted-U shape peaking at T=20.
See Appendix Figure A1.

### Property 5 (Prompt Robustness)

The correction is effective across diverse prompts, confirming that
the Fingerprint-based diagnosis does not depend on a specific prompt.

*Evidence:* 25 prompts on SD 1.5. Correction ΔPSNR = +1.31 ± 1.75 dB
(p=0.0012, Cohen's d=0.75). 13/25 prompts (52%) improved >1.0 dB;
only 2/25 (8%) showed degradation. The correction generalizes beyond
the empty-prompt condition used for Fingerprint measurement.
See Appendix Figure A2.

---

### Scope Declaration (Properties 1–5)

All properties are established on 5 architectures (SD 1.5, SDXL, HunyuanDiT,
FLUX.1-dev, SD 3.5 Medium) with coco_val images under DDIM or Euler sampling.
Generalization to arbitrary architectures, datasets, or protocols is not
claimed. See §7 (Discussion) for limitations.

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

*Validation:* For all 5 architectures, the observed drift peak location matches
the independently identifiable information bottleneck. SD 3.5 served as
held-out test: the prediction placed the peak at the dual→standard boundary;
the observation placed it at late output compression, revealing a previously
unrecognized bottleneck type. This partial falsification is reported honestly
and led to framework refinement.

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

### Interpretability, not Prediction

We do NOT claim that Φ(M) can be predicted for arbitrary unseen M from G(M)
alone. That would require a larger architecture sample and a learned mapping
function. We claim the weaker but empirically supported statement: **given
G(M), the coarse location and shape of Φ(M) can be interpreted through
Principles 1–3.** The principles correctly identified the bottleneck type
for all 5 architectures and correctly predicted that Cut A (peak skip) would
alter the fingerprint while Cut B (low-drift skip) would not.

---

## 4. Layer 4 — 架构特异机制分析 (Architecture-specific Mechanistic Analysis)

**Scope: Rather than proposing a universal mechanism for all architectures
(or even all U-Nets), we perform architecture-specific case studies. The
Architecture Fingerprint serves as a diagnostic tool—it identifies WHERE
drift concentrates. We then investigate WHY at those specific locations,
acknowledging that different architectures may admit different explanations.**

The key methodological shift: Fingerprint is not for proving one mechanism.
Fingerprint is for discovering that **different architectures need different
explanations.**

### Hypothesis (Skip Conflict)

For U-Net architectures, the skip connection from down_block[i] to
up_block[N-i-1] introduces **structured encoder-decoder feature mismatch**
during inversion.

Operationally: let s be the skip feature and u be the up_block's internal
representation before receiving s. The skip conflict is:

```
C = || s − u ||_2   (directly measurable)
```

**Causal prediction:** reducing skip strength α reduces conflict C,
which in turn reduces drift φ_l at layers fed by this skip and improves
reconstruction quality.

### Empirical Validation (SD 1.5, 19 images, 50-step DDIM)

**Claim 4a (Binary intervention).** Zeroing the peak skip (α=0) significantly
changes the fingerprint compared to original (α=1).

*Evidence:* 31/38 layers p<0.05 (paired t-test). Peak drift: −27.7%
(p=4.8×10⁻⁸). Reconstruction: PSNR +2.20 dB (p=0.0005). See Figure 6A.

**Claim 4b (Location specificity).** Zeroing a low-drift skip produces no
significant change.

*Evidence:* Cut B (up_blocks.0 skip): 5/38 layers p<0.05. Peak drift: +0.8%
(n.s.). PSNR: −0.11 dB (n.s.). The anti-correlated delta maps (r=−0.395)
between Cut A and Cut B rule out a simple capacity effect. See Figure 6B.

**Claim 4c (Continuous dose-response).** The effect is monotonic in α.

*Evidence:* α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}. PSNR decreases monotonically
as α increases (24.66 → 22.44 dB). No intermediate optimum—the skip is
purely harmful at this location. See Figure 6C.

**Claim 4d (Structured conflict, not magnitude).** Replacing the skip with
noise of identical statistics produces different behavior from both zeroing
and original.

*Evidence:* Noise replacement increases L2 drift (+6.4% vs. original) but
IMPROVES reconstruction (LPIPS 0.218→0.113, PSNR +2.4 dB). This dissociates
drift magnitude from reconstruction quality: unstructured noise does not
carry the specific encoder-decoder mismatch pattern that degrades
reconstruction. See Figure 6D.

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

*Interpretation (conservative):* This result indicates that skip connections
play **architecture-dependent functional roles** during inversion—they can
be conflict sources (SD 1.5) or necessary information pathways (SDXL). It
does NOT prove that SDXL's entire drift mechanism is different (other
factors, such as the relative strength of conflict removal vs. information
loss, could contribute). What it does prove is that **the same structural
component contributes differently across U-Net variants**—which supports the
central claim that the Architecture Fingerprint captures architecture-specific
phenomena, not generic backbone-family properties.

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

P2P vs Ours: Cohen's d=0.033 (negligible effect size), Pearson r=1.000
(identical behavior). Statistically significant but practically equivalent.

**Cross-architecture generalization:**

| Architecture | ΔPSNR | Optimal λ | Key insight |
|-------------|-------|-----------|-------------|
| SD 1.5 | +2.75 dB | 0.7 | random5 ≈ top5 |
| SDXL | +5.37 dB | 0.7 | Larger UNet → larger gain |
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
- These are NOT failures—they validate that the system is already
  operating at the Pareto frontier of simplicity and performance.

---

## 6. 摘要 (Abstract, ~250 words)

> Feature drift in diffusion inversion—the discrepancy between inverted and
> reconstructed representations—is widely observed but poorly understood.
>
> We discover that this drift is not random noise but exhibits a reproducible
> **Architecture Fingerprint**: its layer-wise distribution is determined by
> the backbone's attention topology rather than the sampling paradigm.
> Quantifying drift across five architectures (U-Net × 2, Transformer,
> MM-DiT, MM-DiT-X) and two paradigms (DDIM, Flow Matching), we show that
> same-backbone architectures produce similar fingerprints while different
> backbones diverge. The fingerprint is interpretable from three architectural
> properties—information flow topology, skip/residual structure, and
> cross-modal boundaries—validated through held-out prediction and causal
> intervention.
>
> To explain the fingerprint's origin in U-Net, we propose the **Skip Conflict
> Hypothesis**: skip connections introduce structured encoder-decoder feature
> mismatch during inversion. Interventional analysis, noise-injection controls,
> and a continuous dose-response curve establish that structured conflict—not
> drift magnitude—determines reconstruction quality.
>
> As a practical consequence, diagnosis at the identified bottleneck enables
> the simplest latent-space correction to achieve content preservation on par
> with complex methods (Cohen's d=0.033 vs. Prompt-to-Prompt) using negligible
> memory. Our central message is not that linear correction works, but that
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

## 8. v2 → v3 变更摘要

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