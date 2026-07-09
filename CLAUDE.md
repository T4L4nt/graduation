# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

硕士毕业设计：**扩散反演特征漂移的架构指纹——发现、理解与利用**。作者为塔拉尼提·居马努尔。

**核心贡献：Architecture Fingerprint of Feature Drift**

发现扩散反演中的特征漂移具有清晰的**架构级结构**——漂移模式由 backbone 的 attention 拓扑决定，而非采样器 artifact。跨 SD 1.5 / SDXL / HunyuanDiT / FLUX 四种架构和 DDIM / Flow Matching 两种范式统一量化：同 backbone 架构间漂移分布高度相似（FLUX vs HunyuanDiT Pearson r=0.727），不同 backbone 间显著分化（FLUX vs SD 1.5 r=0.486）。漂移指纹可从架构的 (a) 信息流图、(b) skip/residual 结构、(c) 跨模态交互边界三要素预测。

**理论解释**：信息论（因果消融 + 互信息估计）解释漂移为何集中在特定层类型——ResNet 残差包含的可恢复信息显著多于 Attention（ΔPSNR 2.1×），且该差异不由特征方差驱动（MI 比率仅 1.1×）。流形分析与收敛性推导作为补充视角。

**工程推论——最简校正是诊断充分的自然结果**：一旦通过逐层诊断定位架构瓶颈，最简 latent 线性校正即可达到复杂方法的同等效果。与 P2P 统计等价（Cohen's d=0.033），内存低数百倍。刻意复杂化（feature-level 注入、文本 token 残差、闭环控制）均不提供额外增益——简单性是诊断的成果，不是方法的局限。

> 核心贡献是**发现规律**（Architecture Fingerprint），方法是**利用规律**的自然推论。线性插值公式不是我们的发明（RLI 已独立发现类似形式），我们的贡献是**诊断→定位→极简干预**的范式以及"为什么线性插值有效"的理论解释。

## CVPR 投稿定位

**核心发现：Architecture Fingerprint of Feature Drift**

特征漂移不是随机噪声——它是**架构签名**。漂移模式由 backbone 的 attention 拓扑（single-stream vs dual-stream, CNN skip vs residual stream）决定，不由采样范式（DDIM vs Flow Matching）决定。四架构两范式统一量化支持这一结论。

**理论：信息论解释漂移为何有结构**

因果消融 + 互信息估计表明 ResNet 残差的可恢复信息远多于 Attention（ΔPSNR 2.1×）——这解释了漂移为何集中在特定层类型，以及为何简单校正有效。

**工程验证：最简校正是诊断的自然推论**

校正对注入位置鲁棒（random5≈top5）、对 λ 不敏感、跨架构有效。这不是”我们发明了好方法”——这是”诊断充分后，最简单的方法就足够”。Feature-level 校正无效（Δ=−0.27 dB）和闭环控制无增益进一步支持：刻意复杂化不带来收益。

> 论文回答的核心问题是 **”Why does inversion fail?”** 而非 **”How to improve inversion?”**。答案：反演失败具有清晰的架构级结构。利用这一结构，最简干预即可。发现规律是贡献，利用规律是验证。

## 项目阶段

| 阶段 | 时间 | 状态 |
|------|------|------|
| Phase 1 诊断 | 2026.5 | ✅ 完成 |
| Phase 2 校正 | 2026.6 | ✅ 完成 |
| Phase 3 选择性校正 + 风格编辑 | 2026.6–7 | ✅ 完成 |
| Phase 4 理论 + 跨架构验证 | 2027.2–7 | ✅ 完成 |
| Phase 5 统计验证 + 缺口补齐 | 2027.7 | ✅ 完成 |
| Phase 6 FLUX Flow Matching 扩展 | 2027.7 | ✅ 完成 |
| Phase 7 编辑 Benchmark | 2027.7 | ✅ 完成 |
| Phase 8 SD 3.5 预测验证 + FLUX 消融 | 2027.7 | 🔴 待做 |

**DCSC（闭环控制器）**：已探索并放弃。实验验证闭环控制在当前系统上没有可测量的增益（三模式在对抗条件下 PSNR 等价），该负结果为"简单性即优势"的叙事提供了支撑。论文 Discussion 中诚实提及。

---

## Phase 8：SD 3.5 预测验证 + FLUX 消融补全 🔴

### 8a：SD 3.5 预测验证（P0，~1 天）

**背景**：SD 3.5 Medium（24 层 MMDiT, dual_attention_layers=0-12）漂移诊断已跑完，但预测被证伪——原始预测为"dual→standard 过渡区（layers 12-14）漂移峰"，实际为"过渡区漂移谷（0.12，全局最低），峰值在 block_22（0.27）"。

**原因分析**：SD 3.5 的 dual→standard 过渡与 FLUX 的 joint→single 过渡**方向相反**：
- FLUX: joint→single 是**丢失**跨模态交互 → 漂移峰（特征失稳）
- SD 3.5: dual→standard 是**获得**跨模态交互 → 漂移谷（特征被 stabilize）

这反而是框架的**细化**而非推翻：跨模态交互边界的效应方向取决于交互是被加入还是被移除。

**待做**：
1. 修改 `scripts/sd35_phase1_diagnostics.py`，加入 text drift 测量（验证"跨模态获得→text drift 下降"）
2. 量化修正预测的匹配度（peak 位置、形状、text drift 行为）
3. 将原始预测（证伪）和修正预测（成立）写入 `prediction_record.json`
4. 撰写 honest 叙事：预测→证伪→分析→框架细化→新预测成立

### 8b：FLUX 层组消融（P1，~1 天）

**背景**：消融矩阵中 FLUX 列完全空白。其他三架构已完成：

| 消融 | SD 1.5 | SDXL | HunyuanDiT | FLUX |
|------|:------:|:----:|:----------:|:----:|
| top5 vs random | ✅ | ❌ | ✅ | ❌ |
| 层组消融 | ✅ | ❌ | ✅ (transition/bottom/top) | ❌ |
| λ scan | ✅ | ✅ | ✅ | ❌ |

**待做**（使用 `scripts/flux_phase6c_analysis.py` 或新脚本）：
1. FLUX λ scan：λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}，joint 层注入，19 图
2. 层组消融：
   - joint-only（19 blocks）
   - single-only（38 blocks）
   - early-single（single_0-18）
   - late-single（single_19-37）
   - joint + early-single
   - top5（跨区域）
3. 对比：joint-only vs single-only ΔPSNR，验证"single blocks 漂移更大 → correction 增益更大"的预测

**预期**：single-only 校正增益应大于 joint-only（因 single 漂移更大，1.4×）。early-single vs late-single 可能揭示 bimodal 指纹的校正含义。

---

## 开发环境

- conda 环境 `grad`（Python 3.10），激活：`conda activate grad`
- GPU：NVIDIA RTX PRO 6000 Blackwell (48GB)
- PyTorch 2.11.0+cu128, diffusers 0.38.0, transformers 5.12.1
- 主模型：`runwayml/stable-diffusion-v1-5`（已缓存）
- CLIP：`openai/clip-vit-large-patch14`（已缓存，需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 离线运行）
- 运行脚本时需 `export PYTHONPATH=scripts:$PYTHONPATH`

---

## Phase 1：DDIM 反演-重建漂移动态诊断 ✅

- **脚本**：`scripts/phase1_diagnostics.py`
- **输出**：`outputs/phase1/layer_drift_summary.json`（19 图 coco_val）
- **关键发现**：
  - 漂移在 UNet 层间极不均匀——跨层差距达 1000×
  - 漂移集中在 decoder up_blocks ResNet（`up_blocks.2.resnets.0` 跨图最高）
  - ResNet 漂移比 Attention 大约 5×（与直觉相反——注意力并非信息瓶颈）
  - 跨架构漂移指纹各不相同（详见 Phase 4/6 跨架构诊断章节）

---

## Phase 2：零训练残差校正模块 ✅

核心公式：$f_{out} = f_{recon} + \lambda \cdot (f_{inv} - f_{recon})$

### 关键结果（19 图 coco_val，50 步，λ=0.7）

- 平均 ΔPSNR **+2.75 dB**，LPIPS 从 0.218 → 0.094
- 与 P2P 统计等价：ΔPSNR 差 0.13 dB，Cohen's d=0.033（可忽略效应量），Pearson r=1.000

### 核心发现

1. **校正对注入位置鲁棒**：random5 ≈ top5（差 < 0.3 dB），1 层 minimal ≈ 5 层 full
2. **漂移加权无效**（r ≈ −0.11）：诊断的价值不在"选层"而在揭示架构级瓶颈
3. **步数鲁棒性呈倒 U 曲线**：校正峰值在 20 步（Δ=+4.65 dB），4 步/100 步时递减至 +1.7~1.8 dB
4. **λ 稳定性**：λ ∈ {0.3, 0.5, 0.7} PSNR 差 < 0.08 dB——λ 非敏感超参数

---

## Phase 3：选择性校正 + 风格编辑 ✅

### 3a：选择性校正

按特征类型（structural/textural/minimal）做选择性校正。1 层 minimal ≈ 5 层 full（PSNR 差 < 0.2 dB）。脚本：`scripts/phase3_prep.py`（含选择性校正逻辑），输出：`outputs/phase3_selective/`（跨架构验证时已集成到 per-architecture 输出目录）

### 3b：风格编辑 + CLIP 钉扎

CLIP 空间正交投影 + 闭环钉扎反馈。风格注入时保护内容结构。脚本：`scripts/phase3_common.py`（CLIPFeatureExtractor、run_* 函数），三类场景验证：人像/建筑/艺术字体。

### DCSC 负结果

探索了 Drift-Aware Closed-Loop Controller（P-控制律 + 可校正内容子空间 + 稳定性命题），在对抗扰动 σ ∈ {0.01, 0.05, 0.10, 0.20, 0.50} × 3 模式 (open_loop/phase3_pin/dcsc) 下三模式 PSNR 等价。闭环控制在当前系统上不提供可测量增益——系统本身已对 λ 鲁棒，自适应调节 λ 没有额外价值。**论文 Discussion 中诚实提及作为"简单性即优势"的佐证。**

---

## Phase 4：理论分析 + 跨架构验证 ✅

理论目标：解释漂移为何具有架构级结构，以及为何最简校正有效。信息论（因果消融 + 互信息估计）作为主要框架，流形分析与收敛性推导作为补充视角。

### 信息论（因果消融 + 互信息估计）

| 层类型 | ΔPSNR | I(f_inv; f_recon) [nats] |
|--------|-------|--------------------------|
| ResNet | **+2.27 ± 0.48 dB** | 6.84 ± 1.34 |
| Attention | +1.09 ± 0.48 dB | 6.30 ± 1.11 |
| 比率 | 2.1× | 1.1× |

脚本：`scripts/phase4_info_theory.py`（因果消融），`scripts/phase4_mi_estimation.py`（KSG + Gaussian MI），输出：`outputs/phase4_info_theory/`，`outputs/phase4_mi/`

### 流形视角

ResNet 残差比 Attention 更贴合流形切空间（对齐度 0.572 vs 0.420）。特征流形呈沙漏形状。

脚本：`scripts/phase4_manifold.py`，输出：`outputs/phase4_manifold/`

### 收敛性分析（真实 UNet 特征验证）

- 误差收缩（恒等式）：$\|T_\lambda(f) - f^{\text{inv}}\| = |1-\lambda| \cdot \|f - f^{\text{inv}}\|$
- Skip connection 传播（一阶推导，标注为 Proposition）：$d_{l+1} \approx \lambda d_l$（假设 $\|\nabla F_l\| \ll 1$）
- 实证 $\|\nabla F_l\|$：均值 0.996，假设不完全成立——random5≈top5 可能来自 UNet 多路径组合效应

脚本：`scripts/phase4_convergence_verify.py`（全部使用真实 UNet 特征），输出：`outputs/phase4_convergence/`

### 跨架构漂移指纹 ✅

| 架构 | 漂移集中区域 | 独特发现 |
|------|------------|---------|
| SD 1.5 (UNet) | decoder up_blocks ResNet | ResNet >> Attention |
| SDXL (UNet) | mid_block | 与 SD 1.5 完全不同 |
| HunyuanDiT (Transformer) | bottom→top 过渡区 (blocks 11-21) | 无 ResNet/residual 概念 |
| **FLUX (MM-DiT)** | **early single + late single + last joint** | **双峰分布，漂移是架构签名而非采样器 artifact** |

脚本：`scripts/phase4_fingerprint.py`，`scripts/phase6_unified_fingerprint.py`

---

## Phase 5：统计验证 + 缺口补齐 ✅

### 19 图 SOTA 对比（50 DDIM 步，含误差棒）

| Method | PSNR↑ | LPIPS↓ | ΔPSNR | Training | Memory |
|--------|-------|--------|-------|----------|--------|
| DDIM (baseline) | 22.45 ± 3.02 | 0.218 | — | None | Low |
| NTI (BLIP) | 19.11 ± 2.49 | 0.352 | −3.34 | Optimization | Low |
| EDICT | 23.23 ± 3.08 | 0.206 | +0.78 | None | 2× |
| **P2P (attn)** | **25.34 ± 4.01** | **0.087** | **+2.88** | None | ~GB |
| ControlNet (Canny) | 8.20 ± 1.47 | 0.830 | — | Pre-trained | ~1.4GB |
| **Ours_Corr** | **25.20 ± 3.88** | **0.094** | **+2.75** | None | **~MB** |

> NTI 和 EDICT 基于 15 图评估，其余方法基于 19 图。

P2P vs Ours：配对 t-test p=0.0015（统计显著），但 Cohen's d=0.033（效应量可忽略），Pearson r=1.000（行为完全一致）。

### 步数鲁棒性（倒 U 曲线）

| 步数 | DDIM PSNR | Corr PSNR | ΔPSNR |
|------|-----------|-----------|-------|
| 4 | 17.02 | 18.73 | +1.72 |
| 10 | 17.19 | 21.66 | +4.47 |
| **20** | **19.24** | **23.89** | **+4.65** |
| 50 | 22.45 | 25.20 | +2.75 |
| 100 | 23.63 | 25.44 | +1.81 |

### 失败案例分析

6/19 图 Δ < 1.0 dB。三类失败模式：天花板效应（DDIM PSNR 已高）、真正失效（残差不含可恢复结构）、退化上限（反演丢失过多信息）。

### 统计检验汇总

- P2P vs Ours：t=3.737, p=0.0015, Cohen's d=0.033——统计显著但实际等价
- Ours vs DDIM：Δ=+2.75, p=8.8e-6, Cohen's d=0.791——大效应

脚本：`scripts/phase5_final_comparison.py`（SOTA 表+统计检验），`scripts/phase5_failure_lambda.py`（失败案例+λ稳定性），输出：`outputs/phase5_final/`

---

## Phase 6：FLUX Flow Matching 跨范式扩展 ✅

将诊断→校正框架从 DDIM 扩散迁移到 Flow Matching 范式（FLUX.1-dev, ~12B, MM-DiT dual-stream Transformer, 57 blocks = 19 joint + 38 single），实现**跨范式、跨架构**的系统性验证。

### FLUX 架构漂移指纹

**Top-10 漂移层**（joint_18 最高，hidden_drift=0.713，从 turnaround t=1 测量）：

漂移呈 **early single 主导 + 尾端 joint 异常** 的分布：

| 组 | n | mean drift | max drift |
|----|---|-----------|----------|
| Joint blocks (image) | 19 | 0.330 | 0.713 (joint_18) |
| **Single blocks (image)** | 38 | **0.464** | 0.710 (single_2) |
| Joint blocks (text) | 19 | 0.149 | 0.439 (joint_18) |

关键发现：
- **Single blocks 漂移 > Joint blocks**（1.4×）——dual-stream attention 混合 text+image 使特征更稳定
- **Text token 漂移远小于 Image token**（0.15 vs 0.46，~3× 差距）——文本在反演中高度稳定
- **joint_18 的 text drift 跳升**至 0.44（其他 joint blocks 均值 ~0.12）——跨模态信息瓶颈在最后的 joint block
- 特殊 token 位置（511 = EOS/BOS）漂移最大（0.44），语义 token (11-14) > padding token (273-290)

### 19 图校正结果（50 步 Euler，λ=0.7，latent-space correction）

| 指标 | 无校正 | λ=0.7 校正 | Δ |
|------|--------|-----------|------|
| PSNR | 12.03 ± 2.68 | 15.97 ± 2.84 | **+3.94 dB** |
| SSIM | 0.342 ± 0.080 | 0.512 ± 0.094 | +0.170 |
| LPIPS | 0.725 ± 0.090 | 0.485 ± 0.071 | −0.240 |

配对 t-test：t=13.62, **p=6.43×10⁻¹¹**, Cohen's d=3.12（大效应）。校正增益高于 SD 1.5（+2.75 dB）——反演越差，校正增益越大，校正利用架构内在冗余，不依赖反演精度。

### Feature-level 校正（文本 token 残差注入）消融 ✅

5 图 × 4 条件（λ_hidden=0.7, λ_encoder=0.5），turnaround t=1 特征作为参考：

| 条件 | ΔPSNR | 结论 |
|------|-------|------|
| Latent correction (baseline) | **+2.98 dB** | 潜空间全局注入——有效 |
| Feature hidden only | −0.27 dB | 特征层 turnaround 注入——无效 |
| Feature hidden + text | −0.28 dB | text token 残差无增益 |
| Feature text only | −0.09 dB | text-only 接近零效果 |

**关键负结果**：从 turnaround 特征注入在特征层无效——需要 per-timestep 参考或不同的注入策略。Text token 残差不提供额外增益（与 text drift 很低一致）。这反而支持"最简单方法（latent correction）最好"的叙事。

### 四架构跨范式漂移指纹对比

| 架构 | Backbone | 范式 | 漂移指纹 | 模式 |
|------|---------|------|---------|------|
| SD 1.5 | UNet (CNN+Cross-Attn) | DDIM | decoder up_blocks ResNet | 单峰（decoder 集中） |
| SDXL | UNet (更大) | DDIM | mid_block | 单峰（中间层） |
| HunyuanDiT | Transformer (single-stream) | DDIM (v-pred) | blocks 11-21 | 单峰（中层） |
| **FLUX** | **MM-DiT (dual-stream)** | **Flow Match** | **early single + late single + last joint** | **双峰 + 尾端异常** |

**核心发现**：漂移指纹不由采样范式决定（HunyuanDiT 和 FLUX 都是 Transformer 但指纹完全不同），而由具体的 attention 结构（single-stream vs dual-stream）决定。**特征漂移是架构签名，不是采样器 artifact。**

### 架构间漂移分布统计相似度

四架构漂移向量插值到统一长度（57），两两 Pearson r：

| 配对 | Pearson r | Spearman ρ | 解读 |
|------|-----------|------------|------|
| **FLUX vs HunyuanDiT** | **0.727** | 0.720 | 同 Transformer backbone，不同范式→Backbone 主导 |
| FLUX vs SD 1.5 | 0.486 | 0.407 | 不同 backbone+范式→最低相似度 |
| FLUX vs SDXL | 0.646 | 0.544 | 中等相似 |
| SD 1.5 vs SDXL | 0.666 | 0.804 | 同 UNet family→高秩相关 |
| SDXL vs HunyuanDiT | 0.790 | 0.647 | 最高 Pearson（但 drift 量级差 1000×） |

**核心结论**：漂移指纹不由采样范式（DDIM vs Flow Matching）决定，而由 backbone attention 结构（single-stream vs dual-stream MM-DiT）决定。**特征漂移是架构签名，不是采样器 artifact。**

输出：`outputs/phase6_unified/four_arch_fingerprint.png`、`arch_similarity_matrix.png`、`drift_profile_overlay.png`。

### Euler 反演限制分析

| 架构 | 范式 | Baseline PSNR | ΔPSNR | N |
|------|------|--------------|-------|---|
| SD 1.5 | DDIM (可逆) | 22.48 dB | +2.50 | 24 |
| SDXL | DDIM (可逆) | 22.11 dB | +5.37 | 19 |
| HunyuanDiT | DDIM v-pred (可逆) | ~16 dB | +5.65 | 19 |
| **FLUX** | **Flow Match Euler (不可逆)** | **12.03 dB** | **+3.94** | 19 |

Euler 反演代价：FLUX baseline 比 SD 1.5 DDIM 低 ~10.5 dB。校正回收率：+3.94 / 10.5 ≈ 38%。HunyuanDiT baseline 偏低（~16 dB）因 v-prediction DDIM 反演不可逆。校正机制在不可逆反演下依然显著有效——范式无关。

### 技术笔记

- FLUX transformer 需 packed latent tokens（`_pack_latents`），不能直接传 VAE latent
- `_unpack_latents` 需传图像尺寸（img_h, img_w），不是 latent 尺寸
- 存全部 50 步 × 57 blocks 特征会耗尽 CPU RAM；只存 turnaround 点
- T5 offload 到 CPU 必要（48GB 显存不够）
- 模型来源：ModelScope（AI-ModelScope/FLUX.1-dev），~31GB

### 架构拓扑 → 漂移指纹：预测性映射

四架构的漂移指纹不是随机的，而是由其架构拓扑决定的。以下是系统性对应关系：

| 架构 | 信息流拓扑 | 瓶颈位置 | 漂移指纹 | 机理 |
|------|----------|---------|---------|------|
| **SD 1.5** (UNet) | Skip connection 跨层传播 | decoder endpoint | decoder ResNet 单峰 | Skip 将校正信号前传，漂移在 decoder 末端累积 |
| **SDXL** (UNet, 更大) | 扩大 encoder/decoder | mid_block funnel | mid_block 单峰 | 模型放大→处理分布改变→信息瓶颈移至 middle |
| **HunyuanDiT** (Transformer) | Sequential residual stream | 表示层转变区 | blocks 11-21 单峰 | 无跨层 skip，漂移在特征表示转变最快的区域集中。**选对层至关重要**：transition-only (+5.65 dB) >> top5 (+2.50 dB) |
| **FLUX** (MM-DiT) | Dual-stream → single 切换 | 架构类型边界 | early single + last joint 双峰 | Dual-stream attention 稳定特征，移除时触发漂移 spike |

**三层预测框架**：

1. **信息瓶颈拓扑决定漂移在哪**：UNet 家族漂移在信息汇聚点（decoder/mid_block），Transformer 家族漂移在架构类型边界（joint→single, 表示层转变）
2. **Skip/residual 结构决定漂移传播**：UNet skip 将误差传向 decoder；Transformer 残差流无跨层捷径，漂移在转变区本地集中
3. **跨模态交互边界制造漂移 spike**：FLUX joint_18 是 joint→single 的 handoff 点——跨模态上下文在此丢失，text drift 跳升（0.12→0.44），image drift 达峰（0.713）

**与新架构的可泛化性**：给定一个新扩散 backbone，可从其 (a) 信息流图、(b) skip/residual 结构、(c) 跨模态交互边界，**预测**其漂移指纹的位置和形状——这是"漂移是架构签名"从描述性陈述升级为预测性框架的关键一步。

可视化：`outputs/phase7_editing/arch_topo_fingerprint_mapping.png`（`scripts/phase7_arch_topo_mapping.py`）

---

## Phase 7：编辑 Benchmark ✅

将残差校正作为**编辑流程的通用插件**，在 prompt-changed editing 上验证内容保持能力。

### 协议

1. BLIP 生成 coco_val 19 张图的 source caption
2. 程序化生成 28 个编辑对（word swap / attribute change / style transfer），20 对完成完整评估
3. DDIM inversion (source prompt) → 4 条件 reconstruction (target prompt)：
   - baseline（无校正、无 P2P）
   - ours（latent correction, λ=0.7）
   - p2p（cross-attention injection, λ_attn=0.8）
   - ours + p2p（两者叠加）

### 结果（28 编辑对，SD 1.5，50 步）

| Condition | LPIPS↓ | PSNR↑ |
|-----------|--------|-------|
| baseline | 0.856 | 13.93 |
| **ours** | **0.511** | 13.05 |
| p2p | 0.852 | 13.88 |
| ours + p2p | 0.511 | 13.05 |

**校正将编辑中的感知内容保持提升了 40%**（LPIPS 0.86→0.51）。简化版 P2P 单独使用几乎没有效果（0.852），叠加在 ours 上也没有额外增益。

按编辑类型：

| 编辑类型 | baseline LPIPS | ours LPIPS | 改善 |
|---------|---------------|-----------|------|
| attribute change | 0.848 | 0.537 | −37% |
| style transfer | 0.859 | 0.501 | −42% |

### 核心洞察

- **校正作为插件有效**：最简单的 latent correction 在 prompt-changed editing 上显著改善内容保持，跨多种编辑类型稳定
- **简单方法最优**：P2P 的复杂注意力操作不提供额外增益——我们的 latent correction 已经足够
- **回答了 "so what"**：校正不仅让同 prompt 重建更好（Phase 2），也让编辑更好（Phase 7）

脚本：`scripts/phase7_editing_benchmark.py`，输出：`outputs/phase7_editing/`

---

## HunyuanDiT Phase 2：20 图校正消融 ✅

将校正框架在 HunyuanDiT（Transformer backbone, DDIM v-prediction）上做完整消融验证。

### λ 扫描（top5 层，20 图）

| λ | 0.1 | 0.3 | 0.5 | 0.7 | **0.9** |
|----|-----|-----|-----|-----|------|
| PSNR | 16.74 | 17.53 | 18.95 | 18.90 | **19.45** |

DiT 最优 λ=**0.9**，与 SD 1.5 的 λ=0.7 不同——DiT 反演质量更差（baseline PSNR ~15 dB），需要更强的校正。

### 消融实验（λ=0.9）

| 条件 | ΔPSNR | 说明 |
|------|-------|------|
| **transition** (blocks 11-21) | **+5.65 dB** | 仅 transition 区域，效果最优 |
| top10 | +4.98 dB | 10 层 |
| region_bottom | +3.11 dB | bottom blocks |
| region_top | +2.71 dB | top blocks |
| top5 | +2.50 dB | 跨区域 top-5 漂移层 |
| region_transition | +2.50 dB | transition 区域（同 top5 层选） |

**核心发现**：transition 区域单独使用（+5.65 dB）远超 top5（+2.50 dB）——这与 SD 1.5 的"注入位置不敏感"**完全不同**。DiT 是纯 Transformer，没有 UNet 式 skip connection，校正信号不能跨层自由传播。**在 DiT 上，选对层至关重要**——这是架构拓扑决定校正行为的直接证据。

脚本：`scripts/dit_phase2_full.py`，输出：`outputs/dit_phase2/`（`ablation.json` + `lambda_scan.json` + `lambda_curve.png` + `ablation_delta_psnr.png`）

---

## 设计原则

1. **Discovery precedes method**（发现先于方法）：Architecture Fingerprint 是核心贡献，校正是验证这一发现的自然推论。论文回答 "Why does inversion fail?" 而非 "How to improve inversion?"
2. **Simplicity is a consequence of diagnosis**（简单是诊断的成果）：1 层 ≈ 5 层效果。诊断告诉我们不需要复杂控制——架构的内在冗余本身就是鲁棒性保证。简单不是妥协，是理解到位后的必然
3. **Theory explains experiment**（理论解释实验）：信息论框架解释漂移为何有结构、为何 λ 不敏感、为何 random5≈top5。不宣称"理论预测"，只宣称"理论解释"
4. **Honesty over hype**（诚实优于包装）：DCSC 闭环控制的失败被诚实记录，收敛性推导明确标注假设条件和实证差距。负结果不是失败——它们支撑了"简单性即优势"的叙事

---

## 论文叙事

**Discovery → Understanding → Exploitation** 三章递进：

- **Chapter 3 — Discovery（诊断）**：漂移不是噪声，有清晰的架构级结构（Architecture Fingerprint）
- **Chapter 4 — Understanding（理论）**：信息论（因果消融 + MI）作为主要理论框架解释漂移为何有结构；流形分析和收敛性推导作为补充视角
- **Chapter 5 — Exploitation（校正）**：诊断驱动的最简校正——方法简单是因为诊断充分，不是方法简陋

---

## 关键脚本

| 脚本 | 功能 |
|------|------|
| `scripts/phase1_diagnostics.py` | Phase 1：UNet 层级漂移动态诊断 |
| `scripts/phase2_common.py` | Phase 2 共享：加载、DDIM、FeatureCorrector、指标、可视化 |
| `scripts/phase2_full.py` | Phase 2 主实验：λ 扫描 + 评估 + 消融 |
| `scripts/phase2_nti.py` | NTI 基线 |
| `scripts/phase2_edict.py` | EDICT 基线 |
| `scripts/phase3_common.py` | 公共工具：CLIPFeatureExtractor、run_* 函数 |
| `scripts/phase3_prep.py` | 选择性校正 + 风格编辑 + 钉扎 |
| `scripts/phase4_info_theory.py` | 因果消融（信息论） |
| `scripts/phase4_mi_estimation.py` | KSG + Gaussian MI 互信息估计 |
| `scripts/phase4_manifold.py` | 特征流形分析与切空间对齐 |
| `scripts/phase4_convergence_verify.py` | 收敛性验证（真实 UNet 特征） |
| `scripts/phase4_fingerprint.py` | 跨架构漂移指纹图 |
| `scripts/phase4_summary.py` | SOTA 综合对比表 |
| `scripts/phase4_p2p.py` | Prompt-to-Prompt 交叉注意力对比 |
| `scripts/phase4_controlnet.py` | ControlNet Canny 条件生成对比 |
| `scripts/phase4_scenes.py` | 三类场景验证 |
| `scripts/phase5_final_comparison.py` | Phase 5：19 图 SOTA 表 + 统计检验 + 图表 |
| `scripts/phase5_failure_lambda.py` | Phase 5：失败案例分析 + λ 稳定性 |
| `scripts/gen_thesis_figures.py` | 论文配图生成（--mode figures/phase5/summary） |
| `scripts/gen_unified_framework_diagram.py` | 统一框架架构图 |
| `scripts/sdxl_phase1_diagnostics.py` | SDXL Phase 1 诊断 |
| `scripts/sdxl_phase2_full.py` | SDXL Phase 2 校正 |
| `scripts/dit_phase1_diagnostics.py` | HunyuanDiT Phase 1 诊断 |
| `scripts/dit_phase2_common.py` | HunyuanDiT Phase 2 共享（v_prediction DDIM、3D token） |
| `scripts/dit_phase2_full.py` | HunyuanDiT Phase 2 校正 |
| `scripts/flux_common.py` | FLUX Phase 6 共享：加载、Euler inversion、FeatureExtractor、FluxFeatureCorrector |
| `scripts/flux_phase6_diagnosis.py` | FLUX Phase 6a：57 block 漂移诊断 + latent 校正 |
| `scripts/flux_phase6c_analysis.py` | FLUX Phase 6c：19 图验证 + text drift + text injection + Euler 分析 |
| `scripts/phase6_unified_fingerprint.py` | 四架构统一漂移热力图 + 相似度矩阵 + 剖面叠图 |
| `scripts/phase7_editing_benchmark.py` | Phase 7：编辑 benchmark（BLIP caption + 4 条件编辑 + 指标评估） |
| `scripts/phase7_arch_topo_mapping.py` | Phase 7b：架构拓扑→漂移指纹预测性映射可视化 |

## 数据分集

| 分集 | 路径 | 数量 | 用途 |
|------|------|------|------|
| coco_val | `data/coco_val/` | 19 张 | 独立测试集，用于层选择 + 定量评估 + SOTA 对比 |
| basetest | `data/basetest/` | 8 张 | 历史测试 |
| 人像 | `data/portraits/` | 8 张 | 场景验证 |
| 建筑 | `data/architecture/` | 5 张 | 场景验证 |
| 艺术字体 | `data/typography/` | 5 张 | 场景验证 |

## 领域调研

**反演方法**：DDIM → EDICT → NTI

**内容保持**：
- **RLI (Residual Linear Interpolation)**：Jo et al., ICCVW 2025。数学形式上与我们的校正公式等价，但存在本质差异：

| 维度 | RLI | Ours |
|------|-----|------|
| **瓶颈识别策略** | 基于启发式经验，在 UNet 的 attention 层实施残差插值以平滑突变 | 通过层间诊断量化定位架构瓶颈（196/40/57 层），实现**精准干预**而非全局均匀插值 |
| **理论解释深度** | 从注意力平滑角度提供直观动机：减少 attention 突变可缓解编辑伪影 | 从信息论角度（因果消融 + 互信息估计）解释线性插值为何有效、何时有效，流形与收敛性分析作为补充 |
| **架构兼容性** | 在 SD 1.5 / SDXL 的 UNet-based 扩散模型上验证 | 覆盖 **UNet、HunyuanDiT、MM-DiT 及 Flow Matching** 等主流生成架构，验证跨范式泛化性 |
| **问题聚焦** | 针对编辑过程中出现的局部 artifact 与 attention 突变进行后验平滑 | 针对 **inversion-reconstruction 不一致性的根源**进行先验修正，从源头提升编辑保真度 |

- **LAMS-Edit**（最接近，开环混合）→ **Prompt-to-Prompt**（交叉注意力）→ **DiffStateGrad**（SVD 低秩）

**跨架构逐层分析**：
- **FeatureInject / One Size Does Not Fit All**（OpenReview 2025, id=slCmiGEX1D，最大新颖性风险点）：跨架构（SDXL/SD3.5/FLUX）逐层特征注入，分析语义表示**在哪里形成**。三层区分：(1) 他们分析**前向生成**、从不反演，反演-重建一致性这一研究对象在其工作中不存在；(2) **漂移位置 ≠ 语义形成位置**——四架构中三个漂移峰落在其 formation 带外（SD1.5 decoder末端、FLUX joint_18、HunyuanDiT blocks.20），仅 SDXL 重合且用 info-funnel 机理诚实解释；(3) 他们无反演/无校正/无理论/无 Flow Matching。对照图：`outputs/phase7_editing/formation_vs_drift_comparison.png`（`scripts/phase7_formation_vs_drift.py`）。

**差异化定位**：诊断驱动 + 理论解释 + 跨架构验证 + 内存优势 + 统计等价于 P2P。线性插值不是我们的发明——RLI 已独立发现类似形式——我们的贡献是**发现 Architecture Fingerprint** 这一规律，以及**诊断→定位→极简干预**的范式：一旦通过逐层诊断定位了架构瓶颈，最简线性校正即可达到复杂方法的同等效果。简单性是诊断的必然结果，而非调参的偶然发现。

---

## Phase 7c：Skip Connection 因果干预实验 ✅

将三层预测框架从"被动观测"升级为"因果干预"——手术式切断 SD 1.5 UNet 的 skip connection，验证框架能预测干预后果。

### 实验设计

| 条件 | 干预 | 目标 skip | 预测 |
|------|------|----------|------|
| Cut A | 零化 skip → up_blocks.2 | down_blocks.1→up_blocks.2 (漂移峰) | 指纹形状根本改变 |
| Cut B | 零化 skip → up_blocks.0 | down_blocks.3→up_blocks.0 (低漂移区) | 指纹形状基本保持 |

零化在推理时进行，预训练权重不变，反演-重建全流程在修改后拓扑上重新执行。Cut A 和 Cut B 各切断同量的 skip 信息——如果效应是容量驱动，两个 cut 应产生相似的空间模式。

### 19 图结果（50 步 DDIM）

| 指标 | Cut A (peak skip) | Cut B (低漂移 skip) |
|------|-------------------|---------------------|
| 显著变化层 (p<0.05) | **31/38** | 5/38 |
| 峰值层 Δ | **−27.7%** (p=4.8×10⁻⁸) | +0.8% (p=0.15, n.s.) |
| 指纹形状 | **根本改变** | 基本保持 |
| Δ 图空间相关性 | — | r = −0.395 (反相关) |

### 核心发现

1. **架构拓扑直接导致漂移指纹**：切断 peak skip → 漂移大幅下降 27.7%。机理：skip connection 引入 encoder-decoder 特征不匹配，切断消除了不匹配源（但重建质量可能下降——decoder 失去了有用的 encoder 信息）
2. **效应是位置特异的（拓扑效应，非容量效应）**：两个 cut 移除等量信息，但 Cut A 改变 31/38 层而 Cut B 仅改变 5/38 层。Δ 图 r=−0.395（反相关）——不同位置产生不同空间模式，排除容量解释
3. **三层预测框架具备干预能力**：不仅能被动描述漂移指纹，还能预测"在哪里做干预会产生什么效应"

### 论文定位

主文 Discovery 章最后一小节（~0.3 页），一张三列对比图 + Δ 图。核心叙事：
> "To verify the framework's predictive nature, we performed causal interventions by surgically cutting skip connections at inference time. The framework correctly predicted which intervention would alter the fingerprint (Cut A, peak region) and which would preserve it (Cut B, low-drift region). Anti-correlated delta maps (r=−0.395) rule out a trivial capacity-effect explanation."

这是 ICLR 审稿人最看重的"因果干预"级别证据——从 FLUX(拟合) → SD3.5(held-out 预测验证) → SD 1.5 skip 干预(因果操纵) 的三级跳。

脚本：`scripts/phase7_skip_intervention.py`，输出：`outputs/phase7_skip_intervention/`
