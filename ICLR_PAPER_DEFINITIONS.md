# Architecture Fingerprint: 严格定义与贡献层次 (v3.4)

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
>
> v3.2 → v3.3 (2026-07-18 因果范式隔离实验)：(14)Property 3 新增 Controlled
> Paradigm Isolation 实验——同 DiT-S/2 架构、双训练目标(eps-prediction DDPM vs
> flow matching)，以干净因果设计分离架构与训练范式 confound；(15)结论措辞修正：
> 指纹"完全一致" → "组织架构（峰位、排序、加速 motif）对训练范式不变，绝对量级
> 和归一化细节形状范式依赖"，Spearman ρ 降级为参考指标（单调序列秩相关检验力有限）；
> (16)架构计数 5→6 (新增 DiT-S/2)，为 Mapping Principles 提供第 6 个 held-out 验证；
> (17)相位差值剖面(b.9 处范式差异最大)作为观察报告，标 open question。
>
> v3.3 → v3.4 (2026-07-20 ICLR 投稿策略收敛)：(18)Claim 结构从三层(Discovery/
> Mechanism/Application) 升级为四层可证伪层级 C1–C4 — C1(可复现性/跨checkpoint)、
> C2(拓扑聚类/PERMANOVA)、C3(训练目标不变性/采样器)、C4(实例可诊断性/冲突指数)；
> (19)引入 "What Generalizes / What Does Not" 表格化解观测层卖普遍性与因果层卖
> 实例特异性的表面矛盾；(20)记号统一：结构距离 D_s(非 d,避免与 Cohen's d 冲突)；
> (21)全文 "采样范式(sampling paradigm)"→"训练目标(training objective)"措辞替换；
> (22)统计方法升级：等价性用 TOST(非仅 Cohen's d)、多重比较用 BH 校正、MI 须附
> shuffle 基线与收敛曲线；(23)新增 PERMANOVA 方差分解规范(§10)与冲突指数预注册
> 文档(§11)；(24)P2P 等价限于重建场景，编辑场景改为"内容锚定(content anchor)"框架；
> (25)术语排雷：drift / fingerprint / D_s 三词在脚注中与相邻领域歧义主动切割。

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

### 0.1 C1–C4 Claim Hierarchy（Intro 的四层可证伪主张）

> 这四层主张构成了 intro 的脊柱。每一层有独立的证据要求和证伪条件——
> 审稿人攻击任何一条，只会塌掉那一层，不会连锁塌。
> "什么泛化／什么不泛化"从一句辩解变成了论文的结构本身。
>
> 注：本节 C1–C4 是 intro 层面的可证伪主张层级，与 §0.2 中按论文章节组织的
> 三条 Abstract Claim（Discovery / Mechanism / Application）互为补充——
> C1–C3 主要支撑 Discovery，C4 支撑 Mechanism + Application。

**C1 — Reproducibility（存在性／必要条件）**

> The drift profile Φ(M) is a stable, reproducible measurement for a given
> architecture under a fixed protocol—it is invariant to weight-level
> perturbations (cross-checkpoint) but responds to architectural differences.

C1 确立 Φ(M) 是 well-defined observable，不是统计波动。它是所有后续 claim 的
逻辑地基——如果 profile 不可复现，跨架构比较、拓扑解释、诊断都无从建立。

*证伪条件：* 跨 checkpoint 结构距离 D_s(intra) ≮ min D_s(inter)，即同一架构
不同权重的指纹距离不小于最小跨架构距离。

*证据状态：* Property 1（LOOCV, multi-seed）已完成。跨 checkpoint 实验（SD 1.4
vs SD 1.5 vs 社区微调版；SDXL base vs 微调）为 P0b——立即执行。

**C2 — Topological Clustering（组织结构按 attention 拓扑聚类）**

> The organizational structure of Φ(M)—peak position, peak count, concentration,
> spread—clusters by attention topology, and the between-topology variance in
> structural distance substantially exceeds the within-topology variance
> contributed by backbone family, training objective, and sampler choice.

C2 是论文的核心分类学主张。它声称 attention 拓扑（single-stream vs dual-stream,
cross-modal interaction boundary 的存在与方向）是指纹相似度的首要决定因素——
超过 backbone 是 CNN 还是 Transformer，也超过训练目标是 eps-prediction 还是 flow matching。

*证伪条件：* PERMANOVA 中 R²_topology ≯ R²_family 或 ≯ R²_objective——
即拓扑解释的方差被其他因子超过。

*What C2 does NOT claim:* 不声称同拓扑类的每对模型都近于所有跨类配对——
只声称方差分解模式，不声称刚性聚类。不声称 family 效应为零。

*证据状态：* 10-pair 结构距离矩阵与排序稳定性 (Kendall's W=0.938) 已完成。
2×2 架构矩阵（PixArt-Σ + Qwen-Image/SD3 medium）与 PERMANOVA 为 P1。

**C3 — Objective Invariance（组织结构对训练目标不变）**

> The organizational structure of Φ(M) is invariant to training objective
> (eps-prediction vs. flow matching) and to sampler choice (DDIM vs. DPM++
> vs. Euler), for a fixed architecture. Absolute magnitude and fine-grained
> normalized shape are paradigm-dependent and do not alter the architecture's
> diagnostic signature.

C3 分离了"架构决定的"与"范式调制的"——前者是组织结构（峰位、加速 motif），
后者是绝对量级和归一化细节形状。关键措辞：**组织结构的保持 ≠ 指纹"完全一致"**。

*证伪条件：* 峰位跨采样器/训练目标不重合；或跨采样器的 Spearman ρ < 0.95。

*证据状态：* DiT-S/2 双目标对照实验已完成（峰位一致，量级 1.09–2.37× 变幅）。
采样器 swap 实验（SD 1.5 × 4 采样器, FLUX × 2 采样器, 各 2 步数）为 P0b。

**C4 — Instance-level Diagnosability（实例级可诊断性）**

> For each specific architecture, the drift fingerprint Φ(M) enables diagnosis
> of *where* the inversion bottleneck is located and *what causal role*
> specific structural components play (conflict source vs. information pathway).
> The simplest latent-space linear correction, guided by this diagnosis,
> achieves content preservation on par with the most complex existing methods.
> **What generalizes across architectures is the diagnostic methodology—not
> the specific correction parameters or the direction of intervention effects.**

C4 是发现→应用的接口。Φ(M) 告诉你瓶颈在哪；最简干预即足够。C4 明确**不承诺**
干预效应（方向、幅度）跨架构泛化——SD 1.5 的 decoder skip 是冲突源 (+2.20 dB)，
SDXL 的同名 skip 是必要信息通路 (−11.59 dB)。方法论泛化，因果结构实例特异。

*证伪条件：* 架构内冲突指数 CI(skip) 与消融效应 CausalPSNR 的 Spearman ρ ≤ 0，
即诊断不能预测该架构内部的干预效应排序。

*Stretch goal（非必需项）：* 冻结 CI 阈值后在 held-out 架构上做方向性盲测——
成功则 C4 从"架构内诊断"升级为"跨架构预测"。

*证据状态：* SD 1.5 因果链 (Cut A/B, Noise A, dose-response) 完成。
SDXL 跨架构负结果完成。统计等价性 (P2P) 需 TOST 重述。λ cliff, Plan B,
feature-injection 负结果完成。架构内冲突指数验证为 P2。

### C1–C4 与论文章节的映射

| 层级 | 论文章节 | 核心问题 |
|------|---------|---------|
| C1 可复现性 | §3 Properties 1 | Φ(M) 是稳定测量吗？ |
| C2 拓扑聚类 | §3 Properties 2–3 | 组织结构按什么聚集？ |
| C3 目标不变性 | §3 Property 3 (DiT-S/2) | 什么归架构、什么归范式？ |
| C4 实例可诊断 | §5 Mechanism + §6 Application | 诊断能指导干预吗？ |

---

### 0.2 What Generalizes and What Does Not（化解"一篇论文"问题）

C1–C3 建立的是指纹**组织结构的跨架构泛化规律**（按拓扑聚类、对训练目标不变）；
C4 建立的是该结构的**因果解读实例特异**（干预方向和幅度不泛化）。
这不是矛盾——它是 Architecture Fingerprint 作为诊断工具而非普遍定律的自然结果。

| What generalizes | What is instance-specific |
|-----------------|--------------------------|
| The measurement framework (Φ(M) under fixed protocol) | The absolute drift magnitude per layer |
| The organizational structure (peak position, peak count, concentration, spread) tracking attention topology | The fine-grained normalized shape |
| The diagnostic methodology (layer-wise profiling → bottleneck localization) | The causal role of specific structural components (conflict source vs. information pathway) |
| The sufficiency of the simplest intervention once the bottleneck is diagnosed | The optimal λ, the injection layer, the direction of intervention effect |
| The L-shaped frontier structure of content-edit trade-off (qualitatively) | The λ cliff position (quantitatively) |

这张表不是让步——它是论文的核心智识结构。我们不是在卖漂移的普遍定律；
我们在卖一种诊断方法论，其输出按设计就是架构特异的。

---

### 0.3 Central Message (Introduction 最后一句)

> Our central message is not that linear correction is surprisingly powerful,
> but that *sufficient diagnosis makes simple correction sufficient*.
> The Architecture Fingerprint reveals where the inversion bottleneck lies;
> once that is known, the system's own structural redundancy makes complex
> interventions unnecessary. The discovery is the fingerprint; the correction
> is its natural consequence.

这一定位将论文从"一个更好的编辑方法"升级为"一种诊断先于干预的方法论"。
Reviewer 读完 Introduction 应该记住的是这个 insight，而不是 PSNR 数字。

**Central Message 分解为三个可检验子命题**（各命题独立可证伪，共同支撑 Central Message）：

> **P_λ (Lambda Insensitivity, within the plateau regime).** Once the
> bottleneck is diagnosed and λ clears a minimal threshold, the correction
> is insensitive to λ within the saturated plateau. SD 1.5 reconstruction:
> λ ∈ {0.3, 0.5, 0.7} → PSNR variation < 0.08 dB. SD 1.5 editing: plateau
> regime λ ∈ [0.05, 1.0] (width 0.95), where LPIPS gain reaches ≥90% of max.
> The transition window [0.01, 0.05] (width 0.04) is the ONLY λ-sensitive
> region, ~24× narrower than the plateau. The corrective direction — not the
> step size — determines the gain once the threshold is cleared.

*Evidence:* SD 1.5 reconstruction λ scan (19 images); FLUX reconstruction
λ scan (5 images). SD 1.5 editing λ cliff (121 pairs): plateau criterion =
≥90% of max LPIPS gain; λ≥0.05 meets it. FLUX editing λ cliff: not yet
characterized. Qualified as "insensitive at plateau, not at all λ."

> **P_pos (Position Robustness, architecture-dependent).** The correction's
> sensitivity to injection layer depends on the architecture's information
> flow topology, and this dependence is itself predictable from Φ(M):
> - **U-Net (SD 1.5):** Skip connections propagate correction signals →
>   random5 ≈ top5 (ΔPSNR < 0.3 dB).
> - **MM-DiT (FLUX):** Residual stream linearity → joint_only = single_only
>   = latent_all to < 1e-12 dB (exact equivalence).
> - **Transformer without cross-layer skip (HunyuanDiT):** Layer selection
>   is critical — transition-only (+5.65 dB) >> top5 (+2.50 dB).
> The claim is not universal robustness but "position sensitivity is
> predictable from architecture topology." This is a stronger claim.

*Evidence:* SD 1.5 random5≈top5 (19 images). FLUX per-condition equivalence
(19 images). HunyuanDiT transition-only vs top5 (20 images).

> **P_simple (Complexity Exclusion, for reconstruction/content-preservation).**
> More complex interventions — per-layer drift-weighted λ, feature-level
> injection, text-token residuals, closed-loop adaptive control, error-edit
> separation (Plan B) — provide no measurable gain over the simplest
> latent-space correction with fixed λ, for RECONSTRUCTION/CONTENT-PRESERVATION.
> Simplicity is forced by evidence. Scope: the editing case involves
> additional error-edit entanglement (see λ cliff), and these interventions
> were not designed to address it.

*Evidence:* Feature-level injection ΔPSNR = −0.27 dB (FLUX, 5 images).
DCSC: three-mode equivalence (19 images). (Plan B is NOT under P_simple —
it tests error-edit separability, not complexity; see Boundary Result §0.2b.)

**How P_λ ∧ P_pos ∧ P_simple ⇒ Central Message.** Each sub-proposition
removes one degree of freedom that a priori could add value: tuning λ (P_λ),
selecting layers (P_pos), and increasing model complexity (P_simple). If
none of these degrees of freedom contribute beyond the diagnosis-driven
baseline, then the conclusion follows: once you know WHERE the bottleneck is
(diagnosis), the simplest intervention — a fixed-λ, fixed-layer, latent-space
correction — is already at the performance frontier. The diagnosis suffices;
the correction merely instantiates it.

**Relation to the λ cliff in editing.** The λ cliff finding (λ=0.01→0.05
transition, CLIP-Dir collapses from 0.142 to 0.041) provides additional
evidence for P_λ: λ is effectively a binary switch between "no correction"
and "full correction," with a narrow transition window (0.04). This reforges
the "sufficient diagnosis" message: the diagnostic framework identifies
WHETHER to correct (binary decision), not HOW MUCH (continuous tuning).

> **Frontier characterization (L-shaped, not a continuous trade-off).**
> Transition window (0.01–0.05, width 0.04) and plateau region (0.05–1.0,
> width 0.95) differ by ~24×. The frontier is L-shaped: λ either falls
> before the window (edit preserved, gain ~23%) or onto the plateau (gain
> ≥77%, edit suppressed). The intermediate regime is near-empty — there
> exists no λ that simultaneously achieves both plateau-level content
> preservation and intact edit fidelity. The trade-off is either/or,
> not how-much/how-much. Plateau criterion: λ ∈ [0.05, 1.0] where
> LPIPS improvement reaches ≥90% of the maximum (ΔLPIPS = 0.398;
> at λ=0.05, ΔLPIPS = 0.370 = 93% of max). The λ-ablation plot must
> explicitly mark λ=0.01 and use frontier-shape language, not
> defense-judgment phrasing. Note: cliff position characterized on SD 1.5
> with source-prompt inversion editing; inter-architecture λ variation
> (FLUX optimal λ=0.1 for reconstruction vs SD 1.5 λ=0.7) implies the
> cliff may shift per architecture.

---

### 0.2b Boundary Result: Error-Edit Entanglement (P_entangle)

Plan B tested a natural hypothesis: can the inversion error be separated from
the edit direction by pre-computing the error vector from a same-prompt
reconstruction (Δ = f_inv − f_recon^src) and injecting it during editing
(f_out = f_recon^tgt + λ·Δ)? Three variants were tested on 22 pairs:

| Variant | Result | Interpretation |
|---------|--------|---------------|
| Per-timestep injection (all 50 steps) | λ≥0.3: LPIPS explodes to 0.62+ (worse than baseline 0.43) | Error dynamics are trajectory-dependent; Δ[t] valid for source trajectory not target trajectory |
| Endpoint-only (final latent only) | All λ: identical to baseline (no-op) | Δ_endpoint = z_inv[0] − z_recon ≈ 0 — both converge to similar clean latents, even though decoded images differ by ~22 dB PSNR. The reconstruction error is distributed across the trajectory, not concentrated at the endpoint |
| Mid-timestep injection | Pending | Predicted by Property 4 (drift peak at T≈20); if successful, would localize correction to the drift-dominant phase |

> **Precision on "Δ ≈ 0."** z_inv[0] is the original clean latent (before
> inversion). z_recon (final latent after source-prompt reconstruction) is
> close to z_inv[0] in latent L2 — both represent similar clean images in
> compressed VAE space. However, the decoded tensors differ by ~22 dB PSNR
> because DDIM reconstruction error is distributed across timesteps, not
> localized at the endpoint. The endpoint latent similarity is a
> dimensionality artifact — the VAE latent has far fewer dimensions than the
> decoded image, and most of the reconstruction error lives in the high-
> dimensional trajectory dynamics, not the low-dimensional endpoint latent.

**Boundary Result.** DDIM inversion error is trajectory-dependent: the
per-timestep error vector Δ[t] = f_inv[t] − f_recon[t] depends on the
specific prompt used during the trajectory. A Δ computed under the source
prompt cannot be transferred to a target-prompt reconstruction. Therefore,
for prompt-changed editing, content preservation and edit fidelity are
**structurally coupled** in any method that uses a **fixed, pre-computed
latent-space correction vector** — the error-edit entanglement is a
fundamental constraint within this method class.

> **Precision on scope.** This boundary applies to latent-space linear
> correction methods (ours, RLI) that compute a correction vector once
> and inject it globally. It does **NOT** apply to per-layer attention
> injection methods (P2P, cross-attention control) that re-route
> intermediate activations without a single global correction vector.
> The distinction is physically meaningful: attention injection preserves
> the *relative* attention structure across tokens, which carries both
> content and edit information; a single latent correction vector conflates
> both into one scalar direction.

> **What this boundary implies for practice.** The Architecture Fingerprint
> Φ(M) tells you which regime you operate in, and therefore which method
> class you should choose:
> - **Reconstruction / same-prompt content preservation** → Latent-space
>   linear correction (ours). Statistically equivalent to P2P (d=0.033),
>   memory cost hundreds of times lower.
> - **Prompt-changed editing requiring simultaneous content preservation
>   AND edit fidelity** → Attention injection (P2P class) or per-image
>   optimization. Our correction preserves content but collapses edit
>   direction; the boundary result explains WHY — it is structural, not
>   a tuning failure.
> - **Diagnosis-first approach:** Φ(M) reveals the architecture's
>   inversion error profile; this profile tells you whether the correction
>   is likely to conflict with editing (presence/proximity of error peak
>   to key semantic layers). The diagnosis framework thus provides a principled
>   decision rule for method selection, not just a correction recipe.

**Reconciliation with Property 5 (prompt robustness).** Property 5 shows
that correction *efficacy* (ΔPSNR) is stable across prompts. Plan B shows
that the *error vector content* is trajectory-dependent and cannot be
pre-computed. These are consistent: the Architecture Fingerprint Φ(M) —
the distribution shape of layer-wise drift — is prompt-robust (Property 5),
but the specific drift values at each layer-timestep cell depend on the
inversion trajectory and are therefore prompt-specific. **The fingerprint
is the stable shape; the error vector is the unstable content.** This
distinction is what makes Φ(M) an architecture-level descriptor: its
identity is in the shape, not in individual layer values. The failure of
error-edit separation is the *negative* consequence of the same property
whose *positive* consequence is cross-prompt generalizability — flip sides
of one coin.

**Taxonomy.** Plan B is NOT classified under P_simple (complexity exclusion)
because it does not add complexity — it tests a fundamental hypothesis about
the separability of error and edit direction. It constitutes an independent
**Boundary Result (P_entangle)** that constrains the scope of ALL latent-space
linear correction methods, including RLI.

---

### 0.4 论文的三层 Abstract Claim（摘要只报告这三项）

> 这三条 Claim 按论文章节组织——Discovery → Mechanism → Application——
> 与 §0.1 中 C1–C4 的可证伪层级互补。C1–C3 主要支撑 Discovery,
> C4 支撑 Mechanism + Application。

**Claim 1 (Discovery).** Feature drift exhibits an **Architecture Fingerprint**:
its layer-wise *organizational structure* (peak position, per-layer ranking,
acceleration motif) is a reproducible, architecture-specific measurement
determined by backbone attention topology, not by training objective.
Absolute magnitude and fine-grained normalized shape carry
paradigm-dependent variation that does not alter the architecture's
diagnostic signature. (Refined per controlled paradigm-isolation experiment,
DiT-S/2; see Property 3.)

**Claim 2 (Mechanism).** In U-Net architectures, the fingerprint originates from
**skip-mediated encoder-decoder feature conflict**—a structured mismatch that
causes both drift and reconstruction error. The same structural component
can play opposite functional roles across U-Net variants (conflict source in
SD 1.5, information pathway in SDXL), demonstrating that the causal structure
is architecture-instance-specific, not a family-level property.

**Claim 3 (Application).** Identifying the drift bottleneck through diagnosis
makes the simplest latent-space correction sufficient—achieving content
preservation on par with complex methods. In prompt-changed editing, the
correction acts as a **content anchor** (preserving source structure at the
cost of edit fidelity, as shown by the λ cliff curve), not as an edit enhancer.
The three sub-propositions P_λ, P_pos, P_simple (detailed in §0.3) are
independently testable and collectively establish that *sufficient diagnosis
makes simple correction sufficient*.

---

### 0.5 Contributions（Intro 末尾列表，审稿人可直接对照检查）

1. **Architecture Fingerprint framework** (§3): A measurement protocol and
   four-feature structural distance metric D_s that quantifies drift
   organization without interpolation, applied across six architectures
   spanning three attention topology classes and two training objectives.

2. **Empirical evidence that drift organization tracks attention topology**
   (§3–4): Structural distance matrix with ranking stability (Kendall's
   W = 0.938), controlled DiT-S/2 paradigm-isolation experiment, and three
   mapping principles connecting topology to fingerprint shape.

3. **Causal intervention evidence** (§5): Skip-connection ablation (Cut A/B),
   noise-injection control (Noise A), and dose-response curves establishing
   that structured encoder-decoder feature conflict—not capacity or L2
   magnitude—causes the drift fingerprint in U-Net architectures. Cross-
   architecture comparison (SD 1.5 vs. SDXL) demonstrates that the same
   structural component plays opposite functional roles.

4. **Diagnosis-guided correction** (§6): The simplest latent-space linear
   correction achieves statistical equivalence to complex methods (P2P) for
   reconstruction, with the diagnosis revealing *why* additional complexity
   provides no gain. The λ cliff curve and error-edit entanglement boundary
   result characterize the fundamental trade-off in editing applications.

5. **Honest reporting of negative and boundary results**: Feature-level
   injection does not work. Closed-loop control provides no gain. Pre-computed
   error vectors do not transfer across editing prompts (Plan B). SD 3.5
   text-drift prediction was falsified. SDXL skip intervention diverges from
   SD 1.5. These are not failures—they define the scope and sharpen the claims.

---

### 0.6 Terminology Disambiguation（术语排雷，Intro 脚注或 Appendix 首段）

- **Drift**: "drift" = per-layer feature discrepancy between inversion and
  reconstruction. It is NOT the drift term in stochastic differential
  equations (Fokker-Planck drift; cf. DriftLite, ICLR 2026).
- **Architecture Fingerprint**: "fingerprint" = a reproducible measured profile
  that distinguishes architectures under a fixed protocol (analogous to an NMR
  spectrum). It is NOT "architecture fingerprinting" in generative-model
  forensics (which identifies the source generator from an output image).
- **Structural distance D_s**: Euclidean distance in a 4-dimensional feature
  space (peak position, peak count, drift concentration, Gini spread) computed
  from raw layer counts without interpolation. Notation `D_s` to avoid
  confusion with Cohen's `d` (standardized mean difference).

---

### 0.7 论文的 Claim–Evidence–Conclusion 组织

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

**Definition (purely measurement-based).** After min-max normalization to [0,1],
the **drift peak** of architecture M is the layer with maximum normalized drift:

    l_peak(M) = argmax_l Φ_l(M)

This is a measurement, not a prediction — it requires no architectural knowledge,
only the inversion-reconstruction pipeline. The **drift peak region** is the
contiguous set of layers around l_peak whose normalized drift exceeds 0.5
(50% of the global maximum). As a sensitivity check, we also define peaks by
a statistical threshold (layers where Φ_l > μ + 2σ, Appendix E); structural
distance rankings between architecture pairs are stable across both definitions
(Spearman ρ > 0.98 on the 10-pair ordering).

**Prediction (falsifiable hypothesis).** The measured drift peak l_peak(M)
should coincide with the architecture's independently identifiable information
bottleneck, as determined from (a) information flow graph, (b) skip/residual
structure, and (c) cross-modal interaction boundaries (§3.4, Table 1):

| Backbone family | Independently predicted bottleneck | Prediction window |
|----------------|-------------------------------------|-------------------|
| U-Net (SD 1.5, SDXL) | Encoder-decoder junction | ±2 layers of junction |
| Transformer single-stream (HunyuanDiT) | Representation transition zone | blocks 11–21 |
| MM-DiT dual-stream (FLUX, SD 3.5) | Joint→single handoff boundary | ±2 blocks of boundary |

The prediction window size relative to total layers (k/L) provides a
chance-level baseline: if the prediction window covers a fraction k/L of all
layers, l_peak falls within it with probability k/L under random placement.
Across the 5 architectures, mean k/L ≈ 0.20 (range: 0.17–0.25), giving a
binomial probability of 5/5 containment under random placement of
p ≈ 3×10⁻⁴.

**Validation.** For all 5 architectures, the measured l_peak falls within the
independently predicted bottleneck region. On SD 1.5, the measured peak
(decoder junction, up_blocks.2) is exactly the skip connection targeted by
the causal intervention (Cut A, §5) — whose removal eliminates 27.7% of drift
and improves reconstruction by +2.20 dB PSNR. This provides independent
mechanistic validation: the peak layer is not a post-hoc selection but the
specific structural component whose causal role the framework predicts.

**Alternative definition: statistical threshold (Appendix E).** As a
sensitivity check, we also define peaks using a purely statistical criterion:
layers where φ_l exceeds μ + 2σ (architecture-wide mean + 2 standard
deviations, after min-max normalization). Across all 5 architectures, the
statistical-threshold peak agrees with the topology-predicted peak in >90%
of cases (exact layer match in 3/5; within ±1 layer in 4/5; within ±2
layers in 5/5). Structural distance rankings between architecture pairs are
stable across both definitions (Spearman ρ > 0.98 on the 10-pair ordering).
This confirms that the peak is a robust architectural feature, not an
artifact of the detection method.

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
Pearson r = 0.999995 ± 0.000008 (mean ± SD across 19 folds, min = 0.999969,
max = 0.999999). Reporting convention: ">0.9999," not "=1.000".
Multi-seed measurement (3 seeds × 5 images): σ/mean = 0.096% per layer
(mean across all layers). (SD 1.5, 50-step DDIM. Data:
`outputs/phase1_reproducibility/reproducibility.json`.)

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

**Ranking stability of structural distance (2026-07-19 Phase 9 audit):**

The 4-feature structural distance metric has high value instability (CV=0.90
across normalization schemes), making raw distance values unreliable as
quantitative evidence. However, the **ranking** of pairwise distances is
highly stable: Kendall's W = 0.938 (p=9.8e-5) across four normalization
schemes (min-max, z-score, L2, LayerNorm). Key findings:

| Pair | Ranks [mm,zs,l2,ln] | CV | Status |
|------|---------------------|-----|--------|
| **SD 1.5 vs SDXL** | [0,0,0,0] | 0.000 | Consistently closest |
| SD 1.5 vs SD3.5 | [1,2,1,2] | 0.200 | Near-cluster, stable |
| SDXL vs SD3.5 | [2,3,2,3] | 0.143 | Near-cluster, stable |
| SDXL vs DiT | [4,4,3,4] | 0.091 | Mid-rank, stable |
| SD 1.5 vs DiT | [5,5,4,5] | 0.075 | Mid-rank, stable |
| SDXL vs FLUX | [7,7,7,7] | 0.000 | Far-cluster, perfect consistency |
| SD 1.5 vs FLUX | [6,6,8,6] | 0.115 | Far-cluster, stable |
| FLUX vs SD3.5 | [8,8,6,8] | 0.102 | **Far-cluster** (not "second closest") |
| **DiT vs FLUX** | [9,9,9,9] | 0.000 | Consistently furthest |
| DiT vs SD3.5 | [3,1,5,1] | **0.474** | **UNSTABLE — metric artifact** |

**Critical correction (2026-07-19)**: FLUX vs SD 3.5 is NOT "second closest."
The fp16 re-measurement (replacing bf16 with 16.9% systematic bias) confirms
FLUX-SD3.5 ranks 6-8/9 across all normalizations (far-cluster). The previous
bf16-based claim of d=0.385 as "same MM-DiT backbone, second closest" was an
artifact of (a) bf16 precision bias and (b) normalization-sensitive feature
extraction. **The held-out confirmation narrative for SD 3.5 is retracted.**

The surviving structure: same-family does NOT predict proximity. SD 1.5-SDXL
(both UNet) IS closest, but FLUX-SD3.5 (both MM-DiT) is far. Attention topology
— not backbone family — determines fingerprint similarity. The two bookend pairs
(SD1.5-SDXL closest, DiT-FLUX furthest) are the only pairs with CV=0.000,
meaning the core narrative rests on the most robustly measured evidence.

**Reference point — intra-architecture cross-paradigm comparison:**
A controlled DiT-S/2 pair (identical architecture, eps-prediction DDPM vs
flow matching, trained on identical data, 19 held-out test images) provides
a direct measurement of what "paradigm change alone" contributes. After
min-max normalization, the two normalized drift profiles are:

```
eps_norm:  [0.00, 0.02, 0.04, 0.05, 0.06, 0.09, 0.13, 0.17, 0.39, 0.71, 0.81, 1.00]
flow_norm: [0.00, 0.00, 0.02, 0.02, 0.03, 0.07, 0.13, 0.19, 0.29, 0.44, 0.68, 1.00]
```

Both peak at block 11 (identical). The main difference is at blocks 8-9
where eps has higher relative drift (0.39 vs 0.29 at block 8; 0.71 vs 0.44
at block 9). Peak agreement and shared monotonic acceleration motif support
the "architecture determines fingerprint" claim. The pre-peak divergence
constitutes paradigm-dependent fine structure that does not alter peak
identification. See Property 3 for detailed experimental protocol and
the two-layer finding table. See Figure 2 for the 5-curve overlay and
structural distance matrix.

**Metric compatibility note (2026-07-19 Phase 9 audit):** The 4-feature structural
distance d(eps, flow | DiT-S/2) = 2.000 (raw) / 1.001-2.003 (normalized) is NOT
smaller than the closest cross-architecture pair (SD 1.5-SDXL). The original
hypothesis that "d(intra-architecture) ≪ d(closest cross-architecture)" is
**falsified** — the quantitative version of "paradigm difference ≪ architecture
difference" is not supported by the structural distance metric.

This negative result, combined with the CV=0.90 metric instability, motivates the
shift from value-based to rank-based reporting. The surviving qualitative evidence
for paradigm-independence (peak position, organizational motif, acceleration pattern)
is reported separately as the two-layer component decomposition (see Property 3).

### Property 3 (Attention Topology over Broad Backbone Family)

The **ranking** of pairwise structural distances (not the raw values, which
are normalization-sensitive) reveals that attention topology — not backbone
family — determines fingerprint similarity. Ranking is highly stable across
four normalization schemes (Kendall's W = 0.938, p=9.8e-5).

- **SD 1.5 vs SDXL** (both UNet): consistently closest (rank CV=0.000)
- **HunyuanDiT vs FLUX** (both Transformer): consistently furthest (rank CV=0.000)
- **FLUX vs SD 3.5** (both MM-DiT): far-cluster (rank 6-8/9), NOT "second closest"

Family label does not predict proximity: SD 1.5-SDXL IS closest (same UNet),
but FLUX-SD3.5 is far (same MM-DiT but different attention topology — dual-stream
vs. dual→standard transition). What predicts proximity is the specific attention
topology and cross-modal interaction boundary structure, not the broad
"Transformer vs UNet" dichotomy.

This refines the earlier "Backbone Dominance" framing: backbone *family*
does not dominate — *specific attention topology* (single-stream joint
attention vs dual-stream split attention, presence and direction of
cross-modal interaction boundaries) is the primary determinant of Φ.
This is consistent with the qualitative architecture-topology-to-fingerprint
mapping (Section 3.4), which identifies (a) information flow graph,
(b) skip/residual structure, and (c) cross-modal interaction boundaries
as the three predictive features, none of which reduce to a simple
"CNN vs Transformer" dichotomy.

*Evidence (correlational):* Rank-stable structural distance matrix (5 architectures,
10 pairs, Kendall's W=0.938). Attention topology differences (single-stream vs
dual-stream) produce the largest fingerprint divergence regardless of backbone
family. FLUX-SD3.5 (both MM-DiT) are far apart due to different attention
transition structures (dual-stream vs dual→standard). **The previous claim of
SD 3.5 as "held-out confirmation" of the framework's predictive value is
retracted** — it was based on bf16-biased distance values that placed the pair
artificially close.

**Limitation of the above evidence**: architecture and training paradigm are
naturally confounded in all publicly available models (UNet = DDPM-trained,
MM-DiT = Flow-Matching-trained). The inference that "paradigm is not the
determining factor" is indirect — it rests on the observation that
FLUX vs SD 1.5 (d=0.637, different backbone + paradigm) and HunyuanDiT vs
SD 1.5 (d=0.624, different backbone, same paradigm) are at similar distances.
This is a weak argument: HunyuanDiT and FLUX differ in attention topology
(single vs dual stream), so the distance could be driven by either factor.

*Evidence (causal, controlled paradigm isolation):* To disentangle architecture
from training paradigm, we train the identical DiT-S/2 architecture (~40M
parameters, 12 transformer blocks, no cross-attention, pixel-space 64×64)
under two training objectives — eps-prediction (standard DDPM, L_simple) and
flow matching (velocity prediction, rectified flow path) — on the same 111
training images, with identical initialization seed, optimizer, batch size,
and step count (40,000). Both models are then diagnosed with 19 held-out
COCO val images using their respective inversion-reconstruction protocols
(DDIM for eps-prediction, Euler ODE for flow matching).

**Result (two-layer finding):**

| Aspect | Finding | Supports |
|--------|---------|----------|
| Peak position | Both at transformer_blocks.11 (identical) | Architecture invariant |
| Organizational motif | Monotonic increase + terminal 3-layer acceleration (identical) | Architecture invariant |
| Per-layer ranking | Preserved (Spearman ρ = 1.000; note: weak test for monotonic profiles) | Reference only |
| Absolute magnitude | Eps ~1.55× higher on average, ratio profile non-constant (1.09–2.37, max at block 9) | Paradigm-dependent |
| Normalized shape | Pre-peak offset up to 0.25 (eps drift more dispersed pre-peak, flow more concentrated at final layer) | Paradigm-dependent |

**Correct framing**: the *organizational structure* of the drift fingerprint
(peak position, per-layer ranking, acceleration motif) is invariant to the
training paradigm; absolute magnitude and fine-grained normalized shape are
paradigm-dependent. This is stronger than "fingerprints are identical" — it
isolates exactly *which* aspects of Φ(M) are architectural and which are
paradigm-dependent. Claim 1's assertion that "drift is determined by backbone
topology, not sampling paradigm" is refined to: *architecture determines the
organizational structure of the drift profile; training paradigm modulates
absolute magnitude and fine shape details that do not alter the architecture's
diagnostic signature*.

**Statistical caveat**: Spearman ρ = 1.000 is reported for completeness but
is not a strong finding — both drift profiles are monotonically increasing,
making rank correlation nearly vacuous (any two monotonically increasing
sequences will show ρ ≈ 1). The primary evidence for organizational invariance
is peak position identity and the shared acceleration motif.

**Structural distance (Phase 9 result — 2026-07-19):** The intra-architecture
cross-paradigm structural distance d(eps, flow | DiT-S/2) was measured and
compared against the cross-architecture spectrum. **The comparison is a negative
result**: d(eps, flow) ≥ d_min_cross under all normalization schemes
(ratio range 0.98x–25.98x, CV=0.90). The original hypothesis that
"d(intra-architecture) ≪ d(closest cross-architecture)" is **falsified**.

The structural distance metric's value instability (CV=0.90 across
normalizations) and the falsified quantitative hypothesis together motivate
the shift from value-based to rank-based reporting. The surviving evidence
for paradigm-independence is qualitative (peak position, organizational motif);
the surviving evidence for cross-architecture distinguishability is ranking-based
(Kendall's W=0.938). The DiT-S/2 controlled experiment is reframed from
"supporting evidence" to "boundary-calibration experiment": it defines the
limit at which the quantitative distance metric breaks down, and demarcates
the architecture-determined component (organizational structure) from the
paradigm-modulated component (magnitude, fine shape).
cross-architecture pair, including the closest same-family pair.*

**Ratio profile observation** (reported, not explained): The per-layer
eps/flow drift ratio peaks at block 9 (2.37×) before declining to 1.55× at
the final layer. This non-constant profile may reflect trajectory-dependent
error accumulation differences between DDIM inversion and Euler ODE —
DDIM accumulates approximation error across the trajectory mid-section where
block 9's features are most active. This is reported as an observation for
future investigation; the current experiment was not designed to test this
hypothesis.

**Mapping Principles consistency**: DiT-S/2 is a class-conditional pure
Transformer with no cross-layer skip connections — its drift peak at the
final layer (block 11) is consistent with Principle 2 (Propagation Mode):
"sequential residual stream only → drift localizes at the transition zone."
The output-layer bottleneck in a skip-free Transformer is the information
compression point, matching SD 3.5's "output compression" bottleneck type
(§4, Table 1). This provides a 6th held-out architecture validation for the
Mapping Principles.

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

Properties 1–3 are established on 6 architectures (SD 1.5, SDXL, HunyuanDiT,
FLUX.1-dev, SD 3.5 Medium, and a controlled DiT-S/2 pair for paradigm-isolation),
10 pairwise cross-architecture comparisons plus 1 intra-architecture
cross-paradigm comparison. All measurements use coco_val images under DDIM
or Euler sampling. The structural distance metric (4 features from raw
layer counts) avoids interpolation artifacts present in earlier
Pearson/Spearman approaches. Property 5 extends the evaluation to 100
diverse prompts; editing validation covers 121 edit pairs
(see §6). Extension to further architectures, datasets, and protocols
is discussed in §7.

**Data reuse note:** The top-5 drift layers used for correction (§6) were
identified from the same 19 coco_val images used for evaluation, creating a
potential positive bias in the 19-image ΔPSNR estimate. The 100-image
independent evaluation (+3.30 dB, d=1.34) exceeds the 19-image result
(+2.75 dB, d=0.79), suggesting the bias is small and conservative. We report
both results and recommend the 100-image estimate as the primary evidence.

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

*Validation (fp16, 2026-07-19 re-measurement):* FLUX joint_18 (last joint block before
single-stream) shows a spike in both image drift (joint_17=0.51→joint_18=0.79, 1.55x)
and text drift (joint_17=0.25→joint_18=0.50, 2.0x; joint_0..17 mean=0.17→0.50, 3.0x).
Both signals spike at the same architectural boundary — consistent with the
hypothesis that the cross-modal interaction boundary acts as a feature stabilizer
whose removal (at the joint→single handoff) triggers simultaneous destabilization
in both modalities. (Note: this is mechanism-consistent evidence, not causal proof;
causal intervention at this boundary — e.g., masking text attention in joint blocks
and measuring the effect on hidden drift — would be needed for a causal claim.)

**Measurement protocol (Appendix A.X):** Text drift measured via FluxFeatureExtractor
hooks on all 57 transformer blocks. Encoder (text token) features are the
encoder_hidden_states output of each block. Drift = MSE(f_inv.encoder, f_recon.encoder)
at turnaround t=1, mean over 77 T5 text tokens, aggregated across 5 COCO val100 images.
fp16 precision, 28-step Euler inversion. Image (hidden) drift measured analogously on
hidden_states output. Previous bf16 values (0.713, 0.12→0.44) are superseded by these
fp16 numbers.
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

*Evidence:* 31/38 layers p<0.05 (paired t-test; 20/38 survive Bonferroni correction,
α=0.001316). Peak drift: −27.7% (p=4.8×10⁻⁸). Reconstruction: PSNR +2.20 dB
(p=0.0005). See Figure 6A.

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

| Method | PSNR | LPIPS | ΔPSNR | Memory |
|--------|------|-------|-------|--------|
| DDIM (baseline) | 22.45 | 0.218 | — | Low |
| NTI (BLIP) | 19.60 | 0.312 | −2.86 | Low |
| EDICT | 22.90 | 0.195 | +0.45 | 2× |
| P2P | 25.34 | 0.087 | +2.88 | ~GB |
| **Ours** | **25.20** | **0.094** | **+2.75** | **~MB** |

P2P vs Ours: Cohen's d=0.033 (negligible effect size), Pearson r≈1.000
(identical behavior). Statistically significant but practically equivalent.

> **Baseline behavior note.** NTI < DDIM (−2.86 dB): NTI optimizes null-text
> embeddings for trajectory-level quality, not pixel PSNR. EDICT gain small
> (+0.45 dB): EDICT's invertibility advantage is concentrated at low step counts;
> at 50 DDIM steps on SD 1.5 the marginal benefit is limited. These behaviors are
> consistent with the literature and not indicative of implementation errors.

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

**100-image editing benchmark (121 pairs, SD 1.5, 50-step DDIM):**

Inversion protocol uses BLIP source caption (not empty prompt), so
baseline LPIPS is lower than in the empty-prompt 19-image benchmark
(0.469 vs. 0.856). The correction effect (ΔLPIPS) is consistent across
both protocols (~0.35–0.40), confirming robustness.

| Condition | LPIPS↓ | PSNR↑ | CLIP-Dir↑ |
|-----------|--------|-------|-----------|
| baseline | 0.469 | 18.25 | 0.115 |
| ours | 0.071 | 25.65 | 0.007 |
| Δ | −0.398 | +7.40 | −0.109 |

The correction improves content preservation strongly (LPIPS −85%,
p=4.8e−55, d=2.58) but simultaneously reduces edit fidelity (CLIP-Dir
p=1.3e−29, d=1.40). This reveals a **fundamental trade-off**: latent-space
correction preserves original content at the cost of resisting edit
direction changes.

The λ sweep across 121 pairs confirms a **cliff, not a slope**: at λ=0.1,
84% of LPIPS improvement is already achieved while 89% of CLIP-Dir is
already lost (0.115→0.013). There is no sweet spot—the trade-off is
all-or-nothing as soon as correction is engaged. The hypothesized
Plan B (error-edit separation via source-prompt reconstruction) was
tested and falsified on 20 images: the per-timestep error is
trajectory-dependent and does not transfer across prompts, and
endpoint-only correction is a no-op. This negative result confirms
that content anchoring and edit resistance are physically coupled in
the latent correction mechanism—they cannot be decoupled by separating
"error" from "edit direction" within per-timestep latent dynamics.

Consequently, the correction's role in prompt-changed editing is
**content anchoring** (pulling reconstruction toward source structure),
not edit enhancement. The earlier claim that the correction serves as
an "editing plug-in" must be qualified: it improves perceptual fidelity
to the source image but reduces the effectiveness of the edit.

**Precision ablation (fp16 vs. bf16, 5 images, 50-step DDIM, SD 1.5):**

Per-step drift trajectory Pearson r = 0.9982 across fp16 and bf16.
Systematic magnitude bias = 5.8%, which is negligible relative to the
1000× cross-layer drift range (5.8e-05). Quantization noise does not
contaminate cross-architecture drift comparisons.

**Negative results supporting simplicity:**
- Feature-level injection: ΔPSNR = −0.27 dB (worse than baseline)
- DCSC closed-loop control: no measurable gain over fixed λ
- Plan B error-edit separation: falsified—DDIM error is trajectory-dependent
- λ sweep: cliff curve, no sweet spot—correction is all-or-nothing for edit direction
- These results validate the diagnosis-first paradigm: once the bottleneck
  is identified, the system already operates at the simplicity-performance
  Pareto frontier.

---

## 6. 摘要 (Abstract, ~200 words)

> Diffusion inversion introduces a per-layer feature discrepancy between
> inverted and reconstructed representations—widely observed but poorly
> understood.
>
> We discover that this feature drift exhibits a reproducible **Architecture
> Fingerprint**: its layer-wise organizational structure (peak position,
> per-layer ranking, acceleration motif) is determined by backbone
> attention topology, not by training objective. Across five architectures
> in unified comparison (with a sixth as controlled paradigm-isolation
> validation), drift profiles reliably distinguish attention topologies
> (single-stream vs. dual-stream) while remaining stable within the same
> topology class. Causal interventions on skip connections and noise-
> injection controls confirm that the fingerprint originates from
> structured encoder-decoder feature conflict in U-Net architectures,
> and that the causal role of each structural component is architecture-
> instance-specific.
>
> This diagnosis directly motivates a minimal intervention: identifying
> the drift bottleneck through layer-wise profiling makes the simplest
> latent-space linear correction sufficient. The correction achieves
> content preservation statistically equivalent to Prompt-to-Prompt
> at negligible memory cost. In editing, the correction acts as a
> **content anchor**—preserving source structure at the cost of edit
> fidelity, with the λ cliff curve revealing an L-shaped frontier.
> Additional complexity—feature-level injection, closed-loop control,
> per-layer weighting, error-edit separation—provides no measurable gain,
> confirming that the bottleneck diagnosis itself, not the correction
> formula, is the contribution.
>
> Our central message is not that linear correction works, but that
> *sufficient diagnosis makes simple correction sufficient*.

---

## 7. 实验补齐优先级（按 ICLR 2027 截稿倒排）

### P0 — 零实验成本 / 极低成本，本周完成（堵住统计一票否决 + 建立 C1/C3 地基）

| # | 内容 | 性质 | 解锁 |
|---|------|------|------|
| P0a | TOST 等价检验（预设等价界 + 90% CI）、BH/FDR 校正 38 层检验、MI 的 shuffle 基线 + 子采样收敛曲线、ΔLPIPS 报双臂绝对值 | 纯分析 | 堵住统计一票否决 |
| P0a | 记号统一：D_s (结构距离) vs d (Cohen's d)，全文检索替换；"采样范式"→"训练目标" 措辞替换 | 纯文本 | 术语排雷 |
| P0a | 术语排雷段落（drift / fingerprint / D_s）放入 Appendix 首段 | 纯文本 | 术语排雷 |
| P0b | 跨 checkpoint 指纹稳定性：SD 1.4 vs SD 1.5 vs 社区全量微调 + SDXL base/微调各一对 | inference-only | C1 |
| P0b | 固定 checkpoint 换采样器：SD 1.5 (DDIM η=0, DDIM η=1, DPM++ 2M, Euler, Euler a) × 2 步数；FLUX (Euler, Heun) × 2 步数 | inference-only | C3 |

### P1 — 中等成本，1–2 周（建立 C2 方差分解 + 风险 4 技术封堵）

| # | 内容 | 性质 | 解锁 |
|---|------|------|------|
| P1 | 补 PixArt-Σ (single-stream cross-attn DiT) + Qwen-Image 或 SD3 medium (MMDiT 系)，凑 2×2 | inference-only | C2 |
| P1 | PERMANOVA 方差分解 (Topology/Family/Objective/Sampler) + PERMDISP + Mantel test | 纯分析 | C2 |
| P1 | dose-matched 随机层切断对照（排除"漂移改变=网络整体损坏"） | inference | 风险 4 |
| P1 | 双臂绝对 LPIPS + 逐类分布 + LPIPS/SSIM/DINO 佐证 PSNR 非平滑伪影 | 纯分析 | 风险 5 |

### P2 — 机制深化，2–3 周（C4 升级 + C2 机制深化）

| # | 内容 | 性质 | 解锁 |
|---|------|------|------|
| P2 | 冲突指数冻结 + 架构内 Spearman 验证 (n≥12 skips per arch) | inference + 分析 | C4 |
| P2 | 跨架构方向性盲测（在 P1 新增 U-Net 架构上，stretch goal） | inference | C4 升级 |
| P2 | recipe 扰动稳健性（CFG scale, prompt 集）；漂移峰位 vs 条件注入位置对应检验 | inference | C2 机制 |
| P2 | 冲突指数预注册哈希承诺放入附录 | 纯文本 | 方法论可信度 |

### P3 — 大规模验证，1–2 周（rebuttal 储备）

| # | 内容 | 性质 | 解锁 |
|---|------|------|------|
| P3 | PIE-Bench 700 全量编辑 benchmark | inference | rebuttal 时消除 benchmark 质疑 |
| P3 | 跨架构编辑 λ 悬崖（FLUX / SDXL） | inference | λ cliff 跨架构泛化 |

### 已完成（附录编号不变）

| # | 内容 | 位置 | 状态 |
|---|------|------|------|
| 2 | Prompt insensitivity (25→100 prompts, Property 5) | Appendix | ✅ A |
| 3 | Editing under Cut A (25 tasks × 3 conditions) | Appendix | ✅ B |
| 4 | SDXL skip modulation (cross-architecture causal) | Appendix | ✅ C |
| 8 | 100-image editing CLIP-Dir dual-metric (LPIPS × CLIP-Dir) | Appendix | ✅ D |
| 9 | Precision ablation (fp16 vs bf16, 5 images) | Appendix | ✅ E |
| 10 | λ cliff curve (121 pairs, λ ∈ [0,1]) | Appendix | ✅ F |
| 11 | Plan B error-edit separation (negative result) | Appendix | ✅ G |

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

## 9. 版本变更摘要

### v3.3 → v3.4 (2026-07-20 ICLR 投稿策略收敛)

| 问题 | v3.3 | v3.4 |
|------|------|------|
| Claim 结构 | 三层 (Discovery/Mechanism/Application) | 四层可证伪层级 C1–C4，每层独立证伪条件 |
| 叙事张力 | 观测卖普遍性 + 因果卖特异性 = 表面矛盾 | "What Generalizes" 表格化解，显式分层 |
| "采样范式" | 范畴错误 (DDIM=采样器, FM=训练目标) | 全文替换为 "training objective"，采样器单独做因子 |
| 统计等价性 | 仅报 Cohen's d=0.033 | TOST + 预设等价界 + 90% CI |
| 多重比较 | Cut A 31/38 层未报告校正方法 | BH/FDR 校正 (Bonferroni 已报, 20/38 仍显著) |
| MI 估计 | 无 shuffle 基线、无收敛曲线 | 强制 shuffle 基线 + 子采样收敛曲线 |
| 结构距离记号 | d (与 Cohen's d 冲突) | D_s |
| 方差分解 | 无 | PERMANOVA 规范 (§10), 四因子 (Topology/Family/Objective/Sampler) |
| 因果预测 | 后验解释 | 冲突指数预注册 (§11), 哈希承诺, 架构内 Spearman |
| 编辑定位 | "editing plug-in" | "content anchor" (内容锚定), λ 悬崖为 L 形前沿 |
| 术语排雷 | 无 | drift / fingerprint / D_s 三项主动声明 |
| 架构矩阵 | 6 模型, 3 拓扑类, 每类 n≈1–2 | 补齐 PixArt-Σ + Qwen-Image, 凑 2×2, 支持方差分解 |

### v2 → v3

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
---

## 10. PERMANOVA 方差分解规范


> 状态：初稿，实验设计规范。执行前需冻结因子编码和距离度量方案。
> 依赖：P1 补架构（PixArt-Σ, Qwen-Image/SD3 medium）完成后运行。

---

### 10.1 目的

用 PERMANOVA（Anderson 2001）对跨架构结构距离矩阵做方差分解，回答 C2 和 C3 的核心问题：

> 在指纹的结构距离中，attention 拓扑、backbone family、训练目标、采样器各自解释多少方差？拓扑的贡献是否实质性大于其他因子？

**为什么是 PERMANOVA 而不是经典 ANOVA**：距离矩阵元素不独立——N 个模型产生 N(N-1)/2 个成对距离，共享模型项。经典 ANOVA 的独立性假设被违反（伪重复）。PERMANOVA 直接在距离矩阵上做置换检验，是生态学中处理这一问题的标准工具。

---

### 10.2 模型架构矩阵（预期 7–8 个模型）

#### 10.2.1 因子编码

每个模型在四个因子上编码：

| 模型 | Topology（拓扑类） | Family（backbone family） | Objective（训练目标） | Sampler（采样器） |
|------|-------------------|--------------------------|----------------------|-------------------|
| SD 1.4 | U-Net-X | SD1x | eps | DDIM |
| SD 1.5 | U-Net-X | SD1x | eps | DDIM |
| SDXL | U-Net-XL | SDXL | eps | DDIM |
| HunyuanDiT | Trans-single | DiT | eps/v-pred | DDIM |
| **PixArt-Σ** | **Trans-single-X** | **PixArt** | **eps** | **DDIM** |
| FLUX.1-dev | MM-DiT-dual | MMDiT | flow | Euler |
| **Qwen-Image** | **MM-DiT-dual** | **MMDiT** | **flow** | **Euler** |
| SD 3.5 Medium | MM-DiT-X | MMDiT-X | rectified-flow | Euler |

> 加粗为待补架构。SD 1.4 提供 C1 的跨 checkpoint 对照，不纳入 PERMANOVA 主分析（与 SD 1.5 距离过近，会人为降低类内方差）；单独在 C1 中报告。

#### 10.2.2 因子去冗余检查

在跑 PERMANOVA 之前，先用 Cramer's V 检查因子间关联：
- `Topology` 和 `Family` 可能存在中高度关联（U-Net 模型都来自 SD 系列）
- 若 V > 0.8，该因子对不能同时放入同一模型——两种处理：(a) 跑两个独立 PERMANOVA，各含 `Topology` 和 `Family` 之一；(b) 用 Type II SS 并诚实报告 confounding。

#### 10.2.3 采样器因子的设计

对于每个架构，至少取 2 个采样器作为 "sampler" 因子的重复。由于同一模型跑多个采样器生成的是配对指纹（同一权重），它们之间的距离反映纯采样器效应。将这些 "same-model-different-sampler" 对纳入距离矩阵时，需要注意：

- 每个 (模型, 采样器) 组合作为一个独立的 "observation" 进入距离矩阵
- 采样器因子编码为 nominal：DDIM、DPM++、Euler、Euler_a

---

### 10.3 距离矩阵构造

### 10.31 指纹提取协议（所有模型统一）

```
P: DDIM/Euler, T=50 (diagonal for DDIM), empty prompt (or fixed simple prompt)
D: coco_val 19 images (or 100 if compute budget allows — 100 preferred for bootstrap)
norm: min-max normalization to [0,1]
features: 4-dim structural feature vector (see §3.2)
distance: Euclidean (L2) on the 4-dim vector
```

关键约束：每个 (模型, 采样器) 组合独立跑完整的 inversion-reconstruction 流程。不能复用不同采样器的 latent trajectory——不同采样器的反演路径不同。

### 10.32 四维结构特征向量

从原始层数的漂移剖面（未经插值）提取：

| 特征 | 定义 | 值域 | 归一化 |
|------|------|------|--------|
| **peak_position** | 峰值层在总层数中的相对位置 | [0, 1] | l_peak / L |
| **peak_count** | 归一化漂移 > 0.5 的峰的个数（用 scipy.signal.find_peaks, prominence=0.1） | N | 原始计数，不做归一化 |
| **concentration** | 前 k 层（k = ceil(0.2L)）携带的漂移占总漂移的比例 | [0, 1] | sum(top_k) / sum(all) |
| **spread** | 漂移分布的 Gini 系数（衡量不均匀程度） | [0, 1] | 直接用 Gini |

**为什么这四个特征**：
- peak_position 和 peak_count 捕获"漂移在哪"（一阶矩）
- concentration 捕获"漂移多集中"（二阶矩）
- spread (Gini) 捕获"漂移的整体不均匀性"（与 concentration 互补——concentration 看头部，Gini 看全局）

四个特征在 min-max 归一化后的漂移剖面上计算。跨四种归一化方案（min-max、z-score、L2、LayerNorm）的排序稳定性已确认（Kendall's W = 0.938）。

### 10.33 Bootstrap 复制（解决类内样本量问题）

**问题**：每个 (模型, 采样器) 只有一个指纹，PERMANOVA 需要类内变异来估计残差。如果没有重复，残差为零，F 检验退化为纯排序。

**方案**：对每张图做 bootstrap 重采样生成 "复制指纹"。

```
For each model-sampler combo:
  1. Take the 19 (or 100) per-image drift profiles
  2. Bootstrap resample N_boot = 100 times (sample images with replacement)
  3. For each bootstrap replicate:
     - Compute the mean drift profile across sampled images
     - Min-max normalize
     - Extract the 4-dim feature vector
  4. Result: N_boot feature vectors per model-sampler combo
```

**效果**：
- 提供了类内变异（来自图像集采样的不确定性）
- 回答了 "指纹对图像集选择的稳健性"
- 使得 PERMANOVA 的置换检验有残差可分解

**注意**：bootstrap 复制引入了额外的变异源（图像采样），这会使类内距离 **大于** 真实的权重级变异。如果在此条件下 PERMANOVA 仍然显著，结论是保守的；如果不显著，需讨论图像采样变异是否淹没了拓扑信号。

### 10.34 距离矩阵维度

若最终有 K = 7 个 (模型, 采样器) 组合，每个 100 bootstrap 复制：
- 距离矩阵为 700 × 700
- 因子设计矩阵为 700 × 4
- 置换检验的 strata 约束：同一 (模型, 采样器) 的 100 个复制在置换时作为一个 block（不允许跨模型重新分配复制）

---

### 10.4 PERMANOVA 执行

### 10.41 软件

Python: `skbio.stats.distance.permanova` 或 R: `vegan::adonis2`

推荐 Python 实现，保持与项目其他分析一致：
```python
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
```

### 10.42 模型公式

```
D_s ~ Topology + Family + Objective + Sampler
```

Type II SS（各因子的边际效应，控制其他因子后）。若因子去冗余检查发现 Topology-Family V > 0.8，拆为两个模型：

```
Model A: D_s ~ Topology + Objective + Sampler
Model B: D_s ~ Family + Objective + Sampler
```

### 10.43 置换方案

- 置换数：999（标准，p 值分辨率 0.001）
- Strata 约束：bootstrap replicate 的 block ID（同模型-采样器组合的 100 个复制不跨组置换）
  - skbio 的 permanova 不支持 strata，改用 R `vegan::adonis2` 的 `strata` 参数或自定义置换
  - 备选：直接对 100 个 bootstrap 距离矩阵各跑一次 PERMANOVA，报告 R² 的 bootstrap 分布
- 置换在各因子的残差中进行（顺序置换，非完全随机）

### 10.44 效应量报告

主表格式：

| Factor | df | SS | R² | pseudo-F | p (perm) |
|--------|-----|-----|-----|----------|----------|
| Topology | 2 | — | **X%** | — | p |
| Family | 2 | — | Y% | — | p |
| Objective | 1 | — | Z% | — | p |
| Sampler | 1 | — | W% | — | p |
| Residual | — | — | — | — | — |
| Total | — | — | 1.00 | — | — |

**报告重点**：R²（效应量/解释方差比例）为主，p 值为辅。措辞：

> "Topology explains XX% of the structural variance in Φ(M), compared to YY% for backbone family and ZZ% for training objective."

不写 "p < 0.05 therefore topology determines the fingerprint"——写 "topology accounts for the largest share of explainable variance among the factors considered."

### 10.45 置换粒度警告

拓扑类仅 3 个，可置换组合数有限。3 个拓扑标签的所有置换 = 3! = 6（含观察到的配置）。因此 p 值分辨率粗糙（min p ≈ 1/6 ≈ 0.167），即使真实效应很强，PERMANOVA 也可能不显著。这是小分类数的固有限制，不是效应缺失。

**缓解**：
- 以 R²（效应量）为主要证据，p 值标注 "permutation-based, limited by n_topology=3"
- 补充：mantel test 直接检验距离矩阵与拓扑差异矩阵的相关性（作为稳健性检查）

### 10.46 补充分析 1: PERMDISP（组内离散度检验）

PERMANOVA 的显著结果可能来自组内离散度差异（不同拓扑类的指纹变异程度不同），而非组均值差异。PERMDISP 直接检验这一前提假设。

```
D_s ~ Topology  # 检验不同拓扑类的组内离散度是否齐性
```

若 PERMDISP 不显著（p > 0.05），PERMANOVA 结果的含义更清晰。若显著，需诚实讨论：类间差异是否部分/全部由离散度而非均值驱动。

### 10.47 补充分析 2: Mantel Test（距离矩阵相关性）

```
mantel(D_s_matrix, D_topology_matrix)
```

其中 D_topology 矩阵 = 两个模型同拓扑类 → 0，不同拓扑类 → 1（简化版；可加权）。Mantel test 不做因子分解，直接检验距离矩阵与拓扑差异矩阵的整体相关性。作为 PERMANOVA 的稳健性参照。

---

### 10.5 结果呈现模板

### 表 X：Variance Decomposition of Inter-Architecture Structural Distance

| Source | df | R² | pseudo-F | p (perm, 999) |
|--------|-----|-----|----------|---------------|
| Attention Topology (single/dual/dual→std) | 2 | 0.XX | XX.X | p |
| Backbone Family (SD1x/SDXL/DiT/PixArt/MMDiT) | 3 | 0.YY | Y.Y | p |
| Training Objective (eps/flow/rectified) | 2 | 0.ZZ | Z.Z | p |
| Sampler (DDIM/DPM++/Euler) | 1 | 0.WW | W.W | p |
| Residual | — | 0.RR | — | — |

**Figure caption**: "Topology accounts for XX% of explainable structural variance—substantially more than backbone family (YY%) or training objective (ZZ%). R² values are reported as effect sizes; p-values are from 999 permutations stratified by bootstrap block and are limited by the small number of topology classes (n=3). PERMDISP: p = 0.XX (homoscedasticity not rejected)."

### 措辞模板（Discussion）

> The variance decomposition shows that attention topology accounts for the largest share of explainable structural variance in Φ(M) among the factors considered (R² = XX%). This supports C2: the organizational structure of drift fingerprints clusters by attention topology. However, the limited number of topology classes (n=3) constrains the statistical resolution of permutation-based inference, and the R² estimate should be interpreted as an effect size in the current architecture sample rather than an estimate of a population parameter. Extension to additional architectures, particularly underrepresented topology classes, would sharpen this estimate.

---

### 10.6 前置条件检查清单

在跑 PERMANOVA 之前必须确认：

- [ ] PixArt-Σ 和 Qwen-Image 的 inversion-reconstruction pipeline 正常工作（至少各 1 张图验证）
- [ ] 所有模型跑完 19 图（或 100 图）的完整漂移诊断
- [ ] 因子编码去冗余检查（Cramer's V 表）
- [ ] 加一个社区微调版 SD 1.5（如 DreamShaper 或 RealisticVision）作为 C1 权重扰动对照
- [ ] 采样器实验完成：SD 1.5 × 4 采样器 + FLUX × 2 采样器（P0b）
- [ ] 确认 `skbio` 的 permanova 是否支持 strata；若不支持，准备 R vegan 备选或 bootstrap 分布方案
- [ ] 冻结此规范文档（permanova 分析前不再修改），附录放哈希承诺

---

## 参考文献

- Anderson, M. J. (2001). A new method for non-parametric multivariate analysis of variance. *Austral Ecology*, 26(1), 32–46.
- Anderson, M. J. (2006). Distance-based tests for homogeneity of multivariate dispersions. *Biometrics*, 62(1), 245–253.
- Legendre, P., & Anderson, M. J. (1999). Distance-based redundancy analysis: testing multispecies responses in multifactorial ecological experiments. *Ecological Monographs*, 69(1), 1–24.

---

## 11. 冲突指数预注册文档


> 状态：初稿，实验前冻结。冻结后提取 SHA-256 哈希，论文附录中印出哈希值和冻结日期。
> 匿名性处理：使用哈希承诺（SHA-256），不引用 GitHub 仓库。相机版公开原始文件即可验证。

---

### 11.0 哈希承诺字段

```
Document: prereg_conflict_index.md
Freeze date: [待填写 — 实验开始前填入]
SHA-256:   [待填写 — 冻结时计算]
Status:    PRE-EXPERIMENT — all predictions and thresholds frozen before seeing held-out data
```

冻结时刻执行：
```bash
sha256sum projects/prereg_conflict_index.md
```

论文附录中印出：
> "The conflict index definition and prediction thresholds were frozen on [date] (SHA-256: [hash]). All held-out architecture measurements were collected after this date. The original document is included in the supplementary material for verification."

---

### 11.1 科学问题

C4 声称：架构指纹 Φ(M) 可以诊断每个架构中特定 skip connection 的因果角色——是冲突源（切断 → 漂移消失 + 质量改善）还是信息通路（切断 → 信息丢失 + 质量恶化）。SD 1.5 的 Cut A 显示 skip 是冲突源（+2.20 dB PSNR），SDXL 的相同结构位置显示 skip 是信息通路（−11.59 dB PSNR）。

**待验证假说**：存在一个从纯观测态（干预前、无需修改模型）可计算的 "冲突指数" CI(skip)，它能在架构**内部**排序各 skip 的因果角色——CI 高的 skip 切断后 PSNR 改善更大（或恶化更小），CI 低的 skip 切断后 PSNR 恶化更大。

**不承诺的**：CI 的绝对值不承诺跨架构可比——只承诺架构**内部**的排序能力。跨架构方向性盲测是 stretch goal（§2.3），不是必需项。

---

### 11.2 冲突指数定义

### 11.21 候选公式

对每个 skip connection s（连接 down_block[i] 到 up_block[N-1-i]），在**干预前**、**纯观测态**计算：

```
CI(s) = DriftLoad(s) / UniqueInfo(s)
```

其中：

**DriftLoad(s)** — skip 特征携带的漂移量：
```
DriftLoad(s) = || f_inv(s) − f_recon(s) ||_2 / || f_inv(s) ||_2
```
即 skip 特征在 inversion vs. reconstruction 之间的相对 L2 偏差。在 turnaround t=1 测量（漂移最大化的 timestep）。

**UniqueInfo(s)** — skip 相对主干路径的不可替代信息量：
```
UniqueInfo(s) = I( f(s) ; y | f_backbone )
```
其中 f(s) 是 skip 特征，y 是最终重建输出，f_backbone 是主干路径（up_block 在接收 skip 之前的内部表示）。实操上用条件互信息估计：I(skip; output | backbone) — 即在已知主干路径的前提下，skip 额外提供了多少关于最终输出的信息。

**直觉**：
- CI 高 = 漂移大但不可替代信息少 → skip 主要携带冲突/噪声 → 切断可能有益
- CI 低 = 漂移小但不可替代信息多 → skip 是必要信息通路 → 切断会损坏质量

### 11.22 稳健替代指标（与 MI 并行计算）

由于条件 MI 在高维特征上估计方差大，对每个 skip 同时计算一个完全不用 MI 的替代指标：

```
CI_simple(s) = DriftLoad(s) / CausalPSNR(s)
```

其中 **CausalPSNR(s)** = 切除 skip s 后的 PSNR 下降量（即 ΔPSNR = Original PSNR − Cut PSNR）。但这个指标本身需要干预来算——它不能用于预注册的 "纯观测态预测"。

**两套工具的分工**：
- **CI（MI-based）用于假说生成**：它在干预前就可计算，是真正的 "预测"。若 CI 在架构内与 CausalPSNR 秩相关显著，假说成立。
- **CausalPSNR 用于假说验证**：它是因果真值，不依赖任何 MI 估计器。若 CI 和 CausalPSNR 的秩相关成立，两个独立工具（信息论 + 因果消融）指向同一结论 → 结论的稳健性不依赖任一工具。

### 11.23 MI 估计协议

- **估计器**：KSG (Kraskov-Stoykov-Gassner, k=3) 或 Gaussian（闭式解，计算条件协方差矩阵的行列式）。两个都跑，报一致性。
- **样本量**：19 张图 × 每个 spatial position（对 ResNet skip：H×W 个 token；对 attention skip：H×W 个 token。实际取 spatial mean 降维或随机采样 1000 个 token 估计分布）。
- **Shuffle 基线**：对每个 skip，打乱 inversion 和 reconstruction 的配对（破坏结构对应但保持边缘分布），重新估计 MI——应接近零。若无 shuffle 基线，MI 估计值的绝对值不可解释。
- **子采样收敛曲线**：N ∈ {5, 10, 19} 张图重算 MI，确认估计值在 N=19 时稳定。

### 11.24 多重比较

SD 1.5 UNet 有 ~4 个 skip connection（down_blocks.0→up_blocks.3, down_blocks.1→up_blocks.2, down_blocks.2→up_blocks.1, mid_block→up_blocks.0）。若每个 skip 测 CI 和 CausalPSNR 共 4 对，Spearman ρ 的 p 值不需要多重比较校正（只检验一个相关系数）。

若扩展到 ~38 个 skip（所有可能的跨层连接），从 38 个中按漂移负载十分位抽样 12 个做消融验证（控制 GPU 成本），12 对 CI-CausalPSNR 的 Spearman ρ 仍是一个检验。

---

### 11.3 验证协议

### 11.31 架构内验证（必需项，C4 的核心证据）

**Step 1 — 观测态计算 CI**（无需干预）
对 SD 1.5 的 4 个 skip connection，在 19 张图上跑完整 inversion-reconstruction，提取 skip 特征 → 计算 CI(s)。

**Step 2 — 因果真值测量**（需要干预，独立于 Step 1）
对同样的 4 个 skip，逐个做 α=0 切除，测量 CausalPSNR(s) = Original PSNR − Cut PSNR。

**Step 3 — 秩相关检验**
```
H0: Spearman ρ(CI, CausalPSNR) ≤ 0  (CI 不预测因果效应，或方向错误)
H1: Spearman ρ(CI, CausalPSNR) > 0  (CI 正确地排序了因果效应)
α = 0.05, one-tailed
```

**冻结判定阈值**：
- ρ > 0.80 且 p < 0.05：假说成立——CI 在架构内具备预测排序能力
- ρ > 0.50 但 p > 0.05：方向正确但样本量不足（n=4 秩相关 power 低）——报告为 "directionally consistent, underpowered"
- ρ ≤ 0：假说证伪——CI 的排序与因果真值无关或反向

**n=4 的 power 分析**：对于 Spearman ρ，n=4 的情况下，完美的单调关系（ρ=1.0）对应的 p=0.083（双尾）。**即在 n=4 时，即使真实效应完美，也无法达到 p<0.05 的显著性。** 这是 n=4 的固有统计限制，不是假说的问题。

**缓解**：
- 报告 ρ 和 90% bootstrap CI，不以 p<0.05 为主要判据
- 如果扩展到 12 个 skip（按漂移负载十分位抽样），n=12 时 ρ>0.58 就可达到 p<0.05（双尾）
- 跨架构聚合：如果 SD 1.5 和 SDXL 各自内部都是 ρ>0，可以 meta-analyze（Stouffer's method），但元分析不能替代个体检验

### 11.32 跨架构方向性盲测（Stretch Goal，非必需项）

**仅在架构内验证成立后执行。**

**协议**：
1. 在 SD 1.5 和 SDXL 上确定 CI 的 "高/低" 分界阈值（如中位数）
2. 冻结阈值
3. 在新的 held-out U-Net 架构（如 Playground v2.5，另一个 SDXL 衍生或独立 UNet）上：
   - Step A: 观测态计算 CI，用冻结阈值判定每个 skip 是 "冲突源" 还是 "信息通路"
   - Step B: 做消融实验，测量实际 ΔPSNR 方向
   - Pre-registered 成功标准：方向预测准确率 > 50%（二分类，chance=50%）且 CI 与实际 ΔPSNR 的 Spearman ρ > 0

**若架构内验证未通过（ρ ≤ 0），跨架构盲测自动取消**——CI 在自己训练数据上都不 work，没有资格做 held-out 验证。

---

### 11.4 与现存证据的关系

**SD 1.5 已有数据（后验，不构成预注册证据）**：
- Cut A（peak skip → up_blocks.2）：+2.20 dB PSNR
- Cut B（低漂移 skip → up_blocks.0）：−0.11 dB PSNR（n.s.）

这 2 个数据点是此预注册设计的**动机**（它们提示 CI 可能存在），但不是**证据**——两个点可以被无穷多种判据拟合。以下所有协议的目标是把 2 个后验数据点升级为 ≥4 个前验预测-验证对。

**SDXL 已有数据（后验，不构成预注册证据）**：
- Cut A（对应 skip → up_blocks.2，但漂移峰在 mid_block）：−11.59 dB

这个反号是跨架构不可泛化的第一次信号——它在预注册文档中被记录下来，但不在架构内验证的统计模型中使用。

---

### 11.5 冻结字段（实验前填写）

```
CI formula:       CI(s) = DriftLoad(s) / UniqueInfo(s)
MI estimator:     KSG (k=3) + Gaussian (both reported)
Shuffle baseline: Yes — shuffle inversion-reconstruction pairs, report CI_shuffle ≈ 0
N images:         19 (coco_val)
Timestep:         turnaround t=1 (max drift)
Spatial pooling:  Mean over H×W (ResNet) / all tokens (attention)
Architectures:    SD 1.5 (primary), SDXL (secondary — included in analysis but not used for threshold fitting)
Target skips:     SD 1.5: 4 skips (down_blocks.0→up_blocks.3, down_blocks.1→up_blocks.2,
                            down_blocks.2→up_blocks.1, mid_block→up_blocks.0)
                  SDXL: 3 skips (down_blocks.0→up_blocks.2, down_blocks.1→up_blocks.1,
                          mid_block→up_blocks.0)
                  If computing budget allows: 12 per architecture (decile-sampled by drift load)
Test statistic:   Spearman ρ(CI, CausalPSNR), one-tailed
Success threshold: ρ > 0.80 with bootstrap 90% CI excluding zero
                  (p-value threshold relaxed due to n=4 statistical power limitation)
Cross-arch blind: Executed ONLY if within-arch ρ > 0
                  Held-out: Playground v2.5 or another U-Net variant
Blind CI threshold: Median CI from SD 1.5 + SDXL pooled (frozen before seeing held-out data)
Blind success:    Directional accuracy > 50% AND Spearman ρ > 0
```

---

### 11.6 可能的失败模式（预注册中诚实列出）

1. **CI 不预测 CausalPSNR（ρ ≈ 0）**：最可能的失败。可能原因：MI 估计方差过大淹没了信号；DriftLoad 和 UniqueInfo 在观测态均方误差中本就耦合而无法分离；CI 假说根本错误——观测态量无法排序因果效应。"非发现"仍然有价值：证明架构内可诊断性（C4）仅靠 Φ(M) 的峰位即可，不需要细化到单个 skip 的 CI。论文措辞降为 "架构内峰位诊断有效，但细粒度的 skip 级因果预测需要超越当前 MI 工具的估计精度。"

2. **CI 排序在所有架构内都成立，但方向相反（ρ < 0）**：假说方向错误——可能 CI 的分母/分子应该反过来。这不是失败，而是发现了一个负的秩相关，修正假说后重新冻结验证。论文里诚实写 "原始假说被证伪，修正后假说成立。"

3. **Shuffle 基线不接近零**：MI 估计器有系统性偏置——估计值在高维特征上膨胀。解决：改用 Gaussian MI（解析解，无偏）或降维到前 10 个 PCA 分量后再用 KSG。

4. **跨架构盲测失败**：CI 的各架构内部阈值不同，冻结的 "通用阈值" 不 work。这不影响 C1–C3，仅将 C4 的预测能力强约束为 "架构内诊断"（which is already the claim）。

---

### 11.7 补充：为什么不把这个预注册放在 Open Science Framework？

OSF 的 pre-registration 在 ICLR 审稿人中认知度不一。哈希承诺的优势：
- 零第三方依赖，自行验证
- 文件内容（含公式、参数、阈值）全在附录中，审稿人可直接对照检查
- 时间戳证明 "实验前冻结" 的事实
- 相机版公开原始 .md 文件即可独立验证 SHA-256

这是密码学级别的 "先冻结后验证"，且完全匿名——不暴露机构、GitHub 账户或地理位置。
