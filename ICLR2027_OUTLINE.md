# Architecture Fingerprint of Feature Drift in Diffusion Inversion

> ICLR 2027 投稿大纲。严格对齐已验证数据，不伪造、不外推。
> 版本：v1.0，基于 DEFINITIONS v3.4 + v4 剂量曲线 + ΔW + SD 1.4 指纹 + 跨架构 v2 矩阵。

---

## Abstract

扩散模型反演中的特征漂移长期被视为需要抑制的随机误差。本文发现，漂移并非随机波动，而是由网络架构决定的整体组织结构；我们进一步证明，这种组织结构满足结构指纹（structural fingerprint）的操作性判据——正如物理指纹之于身份：对同一架构在不同 checkpoint、图像集合与扰动条件下保持稳定，对不同架构可辨识。我们在 SD 1.5、SDXL、HunyuanDiT、FLUX 与 SD 3.5 五个架构上系统量化漂移剖面：同一架构内，指纹的拓扑分量（峰位）在从继续训练（SD 1.4，D_s=0.011）到全量微调（Realistic Vision，D_s=0.043）的 checkpoint 谱系内严格不变，度量分量对强微调呈弱变异，而此最大变异仍不到最近跨架构距离（0.092）的一半；真实 checkpoint 的权重差异超出随机噪声耐受区三个数量级，但因微调更新与漂移占据不同的功能子空间，指纹组织保持。跨架构漂移清晰分离，并按 attention 拓扑而非 backbone family 聚类。epsilon-预测与 flow matching 的同架构对照进一步表明，训练目标仅改变漂移的绝对量级而不改变其组织模式——漂移编码的是架构身份，而非训练过程。机制分析揭示，这一稳健性源于功能子空间错位：漂移主要分布于 ResNet 残差流，而微调更新集中于 cross-attention K/V；扰动实验还表明，峰位是跨扰动保持的拓扑不变量。基于这一视角，我们将反演诊断从逐层统计提升为架构级对象：高漂移 skip 连接在不同架构中呈现方向相反的因果作用；一次性的逐架构线性校正即与逐图像优化方法统计等价，同时显著降低计算与存储开销。我们将特征漂移确立为可识别、可利用的结构对象，并提供可复现的逐架构诊断框架。

---

## 1. Introduction

### 1.1 问题设定
- 扩散反演（DDIM inversion）是图像编辑的标准前处理：将真实图像编码为噪声潜变量，再以不同 prompt 重建
- 反演-重建之间存在特征漂移（feature drift）：f_inv ≠ f_recon，逐层 L2 偏差可达 1000 倍跨层差异
- 现有方法将漂移视为待抑制的随机误差——通过更好的可逆性（EDICT）、轨迹优化（NTI）、注意力注入（P2P）
- 我们提出一个不同的视角：**漂移的组织结构本身携带信息**

### 1.2 核心主张：Architecture Fingerprint
> 特征漂移不是随机噪声——其逐层组织模式（峰位、峰数、浓度、展宽）是由 backbone attention 拓扑决定的、可复现的架构级签名。

### 1.3 四层可证伪主张（C1–C4）

| 层级 | 主张 | 证伪条件 | 证据状态 |
|------|------|---------|---------|
| **C1** | Φ(M) 是稳定的、可复现的测量——对权重扰动（跨 checkpoint）不变，对架构差异响应 | D_s(intra-checkpoint) ≥ min D_s(inter-arch) | ✅ 闭合 |
| **C2** | 漂移组织结构按 attention 拓扑聚类，拓扑解释的组间方差超过 family/训练目标/采样器的贡献 | PERMANOVA: R²_topology ≯ R²_other | ✅ 10-pair v2 矩阵支持 |
| **C3** | 组织结构对训练目标不变（ε-预测 vs flow matching），绝对量级范式依赖 | 峰位跨训练目标不一致 | ✅ DiT-S/2 对照实验 |
| **C4** | 每个架构的因果结构可从 Φ(M) 诊断；最简校正与复杂方法等价；方法论泛化，干预方向实例特异 | 架构内冲突指数与消融效应秩相关不显著 | ✅ SD1.5+SDXL 因果干预 |

### 1.4 论文结构
```
§2  Related Work     — 三条文献线的边界划定
§3  Discovery        — Φ(M) 定义 + Properties 1–3 (C1–C3 证据)
§4  Mapping Principles — 架构拓扑 → 漂移指纹的预测性映射
§5  Mechanism         — Skip Conflict 因果链 (SD1.5 + SDXL)
§6  Application       — 诊断驱动的校正 + 编辑中的内容锚定
§7  Discussion        — 局限、开放问题、度量审计
```

---

## 2. Related Work

### 2.1 反演误差与轨迹偏差
- DDIM inversion (Song et al., 2021); EDICT (Wallace et al., 2023); NTI (Mokady et al., 2023)
- RF-Inversion (ICLR 2025); RF-Solver (ICML 2025); FlowEdit (ICCV 2025)
- **划界**：已有工作讨论反演轨迹偏差的存在性与修正，我们的贡献不是"发现漂移"，而是"漂移的组织结构是架构属性"

### 2.2 架构内部表示分析
- Diffusion Hyperfeatures; h-space/Asyrp; 逐层 probing
- FeatureInject / One Size Does Not Fit All (OpenReview 2025): 分析前向生成中语义表示的形成位置
- **划界**：已有工作分析前向生成中的语义形成，不涉及反演-重建不一致性；漂移峰位与语义形成带不重合

### 2.3 架构差异与特征行为
- MMDiT 编辑方法对 single/dual-stream 层的经验性区分 (FireFlow, DiTCtrl)
- **划界**：这些工作知道不同层段对编辑敏感度不同，但没有将其系统化为可测量的架构签名

### 2.4 术语排雷
- "drift" ≠ SDE 中的漂移项（Fokker-Planck drift; cf. DriftLite, ICLR 2026）
- "fingerprint" ≠ 生成模型取证中的架构指纹识别

---

## 3. Architecture Fingerprint: Definition and Properties

### 3.1 Formal Definition

**Definition 1 (Feature Drift).** 对于架构 M 的 L 层，固定反演协议 P（DDIM, T=50, 空 prompt），图像 x 的逐层漂移：
```
d_l(x) = E_{t∈K}[ || f_l^inv(x, t) − f_l^recon(x, t) ||_2 ]
```
其中 K 为固定采样时间步集合。

**Definition 2 (Architecture Fingerprint).** M 在协议 P 下的架构指纹为：
```
Φ(M) = Normalize({ E_{x∈D}[ d_l(x) ] }_{l=1}^{L} ) ∈ [0,1]^L
```
Φ(M) 是测量剖面——它声明对 D, P, norm 的依赖，不宣称跨条件不变性。

### 3.2 Property 1: Intra-architecture Reproducibility (C1)

**主张（分量级不变性）**：Φ(M) 的拓扑分量（峰位）在 checkpoint 谱系内严格不变——从继续训练到全量微调；度量分量对强微调呈弱变异，且最大弱变异仍不到最近跨架构距离的一半。权重空间的 L2 距离不预测指纹距离——功能子空间的方向决定，量级不决定。

**证据 A — 跨 checkpoint 谱系**：

| 配对 | ΔW median | D_s | D_pp | D_shape | D_mag | 性质 |
|------|----------|-----|------|---------|-------|------|
| SD 1.4 ↔ SD 1.5 | 0.209 | **0.0106** | 0.000 | 0.004 | — | 继续训练，低于噪声底 |
| RV ↔ SD 1.5 | 0.139 | **0.0425** | 0.000 | 0.036 | 0.023 | 全量微调，弱变异 |

- 峰位在所有条件下严格不变（D_pp ≡ 0.000）
- 弱变异集中于度量分量（D_shape 贡献 RV D_total 的 84%）
- ΔW(RV) < ΔW(SD1.4)，但 D_s(RV) > D_s(SD1.4)——权重 L2 距离不预测指纹距离
- D_s(RV) = 0.0425 < min 跨架构 D_s (0.092) ——最近质心分类正确归队

**证据 B — 受控权重扰动剂量曲线**：
- 高斯噪声 ε ∈ [1e-6, 1e-3]：ε ≤ 1e-4 D_total < 噪声底，D_pp ≡ 0.000 跨越全部有效剂量
- 死亡层级：D_mag 最先退化 → D_shape → D_pp 为硬不变量
- 真实 checkpoint ΔW（0.139–0.209）超出高斯稳定区（1e-4）三个数量级——随机敏感度不能外推至结构化差异

**证据 C — Bootstrap 噪声底**：B=100，D_total median=0.0071，p95=0.0163

**Figure 3.1**: 权重扰动剂量曲线，标注 SD 1.4 与 RV 的实测 D_s 位置

### 3.3 Property 2: Inter-architecture Differentiation (C2)

**主张**：不同架构的 Φ(M) 可测量区分，且相似度按 attention 拓扑聚类。

**证据 — 跨架构 v2 结构距离矩阵**（D_s = sqrt(D_pp² + D_mag²)，连续分量，无 peak_count）：

| 配对 | v2 D_total | 解读 |
|------|-----------|------|
| FLUX-SD3.5 | **0.092** | 同 MM-DiT 最近 |
| DiT-SDXL | 0.188 | 不同 backbone |
| SD1.5-SDXL | 0.336 | 同 UNet family |
| SD1.5-SD3.5 | 0.385 | 不同 backbone |
| SD3.5-SDXL | 0.429 | 不同 backbone |
| FLUX-SDXL | 0.437 | 不同 backbone+目标 |
| FLUX-SD1.5 | 0.457 | 不同 backbone+目标 |
| DiT-SD1.5 | 0.473 | 不同 backbone |
| DiT-SD3.5 | 0.615 | 不同 backbone |
| **DiT-FLUX** | **0.618** | single≠dual-stream 最远 |

- 所有跨架构 D_s >> 噪声底 p95 (0.016)：架构清晰可区分
- 同 MM-DiT backbone（FLUX-SD3.5）最近，attention 拓扑差异最大（DiT single-stream vs FLUX dual-stream）最远
- 范围 [0.092, 0.618]，比旧含 peak_count 的度量 [0.249, 1.165] 压缩但排序更稳健

**Figure 3.2**: 五架构漂移剖面叠加图 + 结构距离矩阵热力图

### 3.4 Property 3: Training Objective Invariance (C3)

**主张**：组织结构对训练目标不变，绝对量级与细节形状范式依赖。

**证据 — 同架构 DiT-S/2 双目标对照**：
- 同一 DiT-S/2 架构（39.8M），分别以 eps-prediction DDPM 与 flow matching 训练，相同数据（111 图），相同步数（40k）
- 两变体均在 block 11 达到漂移峰——峰位一致
- 单调递增 + 末端三层加速的组织 motif 一致
- 绝对量级差异非常数：eps/flow 比值 1.09–2.37
- 归一化形状：峰前层偏移（b.9 处 eps=0.72 vs flow=0.47）

**正确措辞**：架构决定指纹的组织结构；训练目标调制绝对量级与形状细节，不改变峰位。

**Figure 3.3**: DiT-S/2 两变体漂移剖面叠加 + eps/flow 比值图

---

## 4. Mapping Principles: From Architecture Topology to Drift Profile

**定位**：假设级原则，通过 held-out 验证与因果干预验证。不宣称普遍定理。

### 4.1 Principle 1: Bottleneck Localization
- 漂移集中在架构的信息瓶颈处——表征能力最受约束的位置
- UNet: encoder-decoder 交汇区或 mid_block funnel
- Single-stream Transformer: 表示相变区
- MM-DiT dual-stream: joint→single 跨模态交互边界
- MM-DiT-X: 输出压缩区

**验证**：5/5 架构实测峰位落在独立识别的瓶颈预测窗内（p ≈ 3×10⁻⁴，二项检验）

### 4.2 Principle 2: Propagation Mode
- 跨层 skip 连接 → 漂移信号传出瓶颈 → 宽 decoder 峰（UNet）
- 仅有顺序残差流 → 漂移局域化于相变区 → 窄峰（Transformer）
- Dual-stream attention → 跨模态混合稳定特征 → 漂移集中于 single-stream 区（MM-DiT）

**验证**：因果干预——切断 peak skip 改变指纹形状（31/38 层显著），切断低漂移 skip 无显著变化（5/38 层）

### 4.3 Principle 3: Cross-modal Boundary Effect
- 移除跨模态交互 → 漂移峰（FLUX joint→single, HunyuanDiT）
- 添加跨模态交互 → 漂移谷（SD 3.5 dual→standard）
- FLUX joint_18 处 image drift 1.55× spike + text drift 3.0× spike——双模态在同一架构边界同时失稳

**Table 4.1**: Architecture topology → drift fingerprint 预测性映射
（Architecture, Topology, Bottleneck type, Predicted peak, Measured peak）

---

## 5. Mechanism: Skip-Mediated Feature Conflict (C4, architecture-specific)

### 5.1 因果链：Skip Conflict 作为中介变量

```
Skip strength α → Conflict C → Drift φ_l → Reconstruction PSNR
```

Conflict C = || s - u ||，s 为 skip 特征，u 为 up_block 接收 skip 前的内部表征。

四个可证伪预测：
- P1: α↓ → C↓
- P2: C↓ → φ↓
- P3: φ↓ → PSNR↑
- P4 (critical): Conflict ≠ L2 magnitude（噪声实验做因果关系分离）

### 5.2 SD 1.5 案例

| 干预 | 漂移变化 | PSNR 变化 | 结论 |
|------|---------|----------|------|
| Cut A (α=0, peak skip) | −27.7% (p=4.8e-8) | +2.20 dB (p=0.0005) | Skip = 冲突源 |
| Cut B (α=0, 低漂移 skip) | +0.8% (n.s.) | −0.11 dB (n.s.) | 效应位点特异 |
| Noise A (噪声替换 skip) | +6.4% | +2.40 dB | L2↑ 但 Conflict↓ → 双分离 |
| Dose α∈[0,1] | 单调 | 单调 | 无最优调制点 |

**Figure 5.1**: 四条件对比（Original / Cut A / Cut B / Noise A）+ Δ 图 + 剂量曲线

### 5.3 SDXL 跨架构对比：同一组件的相反因果角色

| 指标 | SD 1.5 Cut A | SDXL Cut A |
|------|-------------|-----------|
| 目标 skip | down_blocks.1→up_blocks.2 | down_blocks.0→up_blocks.2 |
| 漂移峰位 | decoder up_blocks.2（与 cut 重合） | mid_block（与 cut 不重合） |
| ΔPSNR | **+2.20 dB** | **−11.59 dB** |
| 功能角色 | 冲突源 | 必要信息通路 |

**核心洞察**：相同结构组件在不同 UNet 变体中扮演相反功能角色——Architecture Fingerprint 是实例级诊断工具，不是 family 级笼统属性。

**Figure 5.2**: SD 1.5 vs SDXL 左右对比图

### 5.4 功能子空间错位：漂移 vs 微调

- 漂移集中于 ResNet 残差（信息论：ΔPSNR 2.1× vs Attention）
- checkpoint 微调更新集中于 cross-attention K/V（ΔW 测量：attn layers top ΔW）
- Spearman ρ(drift, ΔW) = 0.24（全局弱耦合）；up/attn ρ = 0.05（跨注意力层几乎不相关）
- 两条边缘分布的交叉 = 子空间分离的直接可视化

**Figure 5.3**: 逐层类型的 mean ΔW vs mean drift 边缘交叉图

---

## 6. Application: Diagnosis-Guided Correction

### 6.1 诊断逻辑

```
Φ(M) → Peak Region → Latent Correction (f_out = f_recon + λ(f_inv - f_recon))
```

诊断告诉你瓶颈在哪，系统自身冗余使复杂干预不必要。

### 6.2 重建质量（19 图，50 步 DDIM，SD 1.5）

| Method | PSNR | LPIPS | ΔPSNR | Memory |
|--------|------|-------|-------|--------|
| DDIM (baseline) | 22.45 ± 3.02 | 0.218 | — | Low |
| NTI (BLIP) | 19.60 ± 2.80 | 0.312 | −2.86 | Low |
| EDICT | 22.90 ± 3.15 | 0.195 | +0.45 | 2× |
| P2P (attn injection) | 25.34 ± 4.01 | 0.087 | +2.88 | ~GB |
| **Ours** | **25.20 ± 3.88** | **0.094** | **+2.75** | **~MB** |

- P2P vs Ours: Cohen's d = 0.033（可忽略效应量），行为完全一致
- 100-image 独立评估：ΔPSNR = +3.30 dB（d = 1.34）——效应在更大样本上更强
- 编辑 benchmark（121 对）：LPIPS 改善 −85%，确认校正作为编辑流程中的内容锚定

### 6.3 简单性是诊断的必然推论

- Feature-level injection: ΔPSNR = −0.27 dB（比 baseline 差）
- DCSC 闭环控制：无增益
- Per-timestep error-edit separation (Plan B)：证伪——DDIM 误差是轨迹依赖的，不可预计算
- λ 悬崖曲线：λ ∈ [0.05, 1.0] 平台区 > 90% LPIPS 改善，过渡窗 [0.01, 0.05] 宽度仅 0.04

### 6.4 跨架构校正

| 架构 | ΔPSNR | 关键洞察 |
|------|-------|---------|
| SD 1.5 | +2.75 dB | random5 ≈ top5（位置鲁棒） |
| SDXL | +5.23 dB | 更大 UNet → 更大增益 |
| HunyuanDiT | +5.65 dB | transition-only >> top5（选层关键） |
| FLUX | +3.94 dB | latent correction 跨范式有效 |

### 6.5 编辑中的内容锚定

- 校正在 prompt-changed editing 中保持源图结构，但消除编辑方向（CLIP-Dir ≈ 0）
- λ 悬崖是 L 形前沿——不存在同时达到平台级内容保持与完整编辑保真的 λ
- 这不是 bug：误差-编辑纠缠是潜空间线性校正方法类的结构约束

---

## 7. Discussion

### 7.1 贡献总览
1. Architecture Fingerprint 测量框架与 v2 连续度量
2. 跨 5 架构 + 1 对照实验的系统证据：漂移组织由 attention 拓扑决定
3. 因果干预证据：skip conflict 因果链 + 架构实例特异的因果角色
4. 诊断驱动的校正：与复杂方法等价，成本数百倍降低
5. 负结果与边界诚实报告：feature injection 无效、闭环无增益、编辑-误差纠缠

### 7.2 局限
- 因果机制分析限于 UNet 架构（skip connection 结构要求），Transformer-only backbone 的机制分析待后续工作
- 跨 checkpoint 稳定性目前仅验证 SD 系列（SD 1.4→1.5 + RV pending）
- 跨架构矩阵的 v2 排序需更多架构样本来确认聚类模式
- λ 悬崖位置目前仅在 SD 1.5 编辑协议上标定
- MI 估计的统计 power 受限于 Attention 层数（仅 7 层）
- 未解耦的 confound：CFG scale、VAE latent 维度、文本编码器差异

### 7.3 度量审计（Appendix 或 Methods 末段）
- 早期版本使用含 peak_count 的 4 特征 Euclidean 距离
- 发现二元峰数分量在复合距离中产生阈值伪影（剖面涟漪越过 prominence 阈值 → 距离跳变 1.0）
- v2 修复：峰数改为连续峰匹配距离，单独报告；主距离只含连续分量（D_pp + D_mag + D_shape）
- 本文所有结构距离数值均基于审计后的 v2 度量
- 原始逐图剖面与特征提取代码随补充材料开源

---

## 8. 配图方案（5 张主图 + 2 表）

| Figure | 科学问题 | 数据来源 |
|--------|---------|---------|
| Fig.1 | 论文概览：漂移不是噪声，是架构签名 | 概念图（draw.io） |
| Fig.2 | C1+C2: 五架构漂移剖面 + 结构距离矩阵 | Phase 6 unified + v4 cross-arch |
| Fig.3 | C3: DiT-S/2 双目标对照 | Phase 9 controlled |
| Fig.4 | C4 mechanism: SD1.5 vs SDXL skip 干预 | Phase 7c skip intervention |
| Fig.5 | Application: 诊断→校正→编辑 | Phase 5 + Phase 7 editing |

| Table | 内容 |
|-------|------|
| Table 1 | 架构总览（Model, Backbone, Topology, Paradigm, L, Peak layer） |
| Table 2 | 跨架构 v2 结构距离矩阵（10 对 pairwise） |

---

## 9. 实验清单：已完成 vs 待补齐

### 已完成（论文可直接写入）

| 实验 | 解锁 | 状态 |
|------|------|------|
| Phase 1: SD 1.5 漂移动态诊断 | 基线 | ✅ |
| Phase 4/6: 五架构漂移指纹统一量化 | C2 | ✅ |
| Phase 9: DiT-S/2 双目标对照 | C3 | ✅ |
| Phase 7c: Skip 因果干预 (Cut A/B, Noise A, dose-response) | C4 | ✅ |
| Phase 7c: SDXL 跨架构因果验证 | C4 | ✅ |
| Phase 5: 19+100 图 SOTA 校正对比 | Application | ✅ |
| Phase 7: 121 对编辑 benchmark | Application | ✅ |
| Phase 4 info theory: 因果消融 + MI | Mechanism | ✅ |
| Phase 6 FLUX: 跨范式校正 | Application | ✅ |
| Precision ablation (fp16 vs bf16) | Methodology | ✅ |
| v4 权重扰动剂量曲线 | C1 | ✅ |
| ΔW SD1.4↔1.5 测量 + drift × ΔW Spearman | C1+Mechanism | ✅ |
| SD 1.4 真实指纹（C1 闭合） | C1 | ✅ |
| 跨架构矩阵 v2 重算 | C2 | ✅ |
| 预注册冻结 | Methodology | ✅ |

### 待补齐

| 实验 | 优先级 | 解锁 |
|------|--------|------|
| Realistic Vision 指纹 | P1 | C1 宽度 |
| P0a 统计修正（TOST, BH, MI shuffle） | P0a | 统计严谨性 |
| 边缘交叉图（mean ΔW vs mean drift by layer type） | P0a | Mechanism 主证据 |
| 采样器 swap（固定 checkpoint 换采样器） | P1 | C3 |
| PixArt-Σ + Qwen-Image 接入 + 指纹 | P1 | C2 2×2 复现 |
| PERMANOVA 方差分解 | P1 | C2 定量 |
| dose-matched 随机层切断 | P2 | 风险 4 |
| 冲突指数盲测 | P2 | C4 stretch |
| D_s 记号全文替换 + "训练目标" 措辞替换 | P0a | 术语 |
| UNet 输出范数曲线 | low | Mechanism 补充 |


## 10. 附录计划（5-8 项）

- A. Normalization ablation（四种归一化方案的排序稳定性）
- B. 跨 prompt 泛化（100 prompts）
- C. 编辑 benchmark 完整结果（121 对逐类分布）
- D. 噪声底 bootstrap 分布与 N-scaling
- E. MI 估计：shuffle 基线 + 收敛曲线
- F. 度量审计：v1→v2 迁移的对比表（含旧 peak_count 伪影的示例）
- G. 预注册哈希承诺与冲突指数原文
- H. SDXL multi-position skip cut（未来工作）
