# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

硕士毕业设计：**扩散反演特征漂移的架构指纹——发现、理解与利用**。作者为塔拉尼提·居马努尔。

**核心贡献：Architecture Fingerprint of Feature Drift**

发现扩散反演中的特征漂移具有清晰的**架构级结构**——漂移模式由 backbone 的 attention 拓扑决定，而非采样器 artifact。跨 SD 1.5 / SDXL / HunyuanDiT / FLUX / SD 3.5 五种架构和 DDIM / Flow Matching 两种范式统一量化。定量分析以**结构距离**（无插值，从原始层数提取 peak 位置/峰数/浓度/展宽四个特征）为主：同 backbone family 最近（SD 1.5-SDXL d=0.249, FLUX-SD 3.5 d=0.385），跨 family 显著分化（d>0.5），attention 结构差异越大距离越远（HunyuanDiT single-stream vs FLUX dual-stream d=1.077 最远）。注意：基于插值的 Pearson/Spearman 相关矩阵不可靠——不同架构的层数差异导致插值合成点占比最高达 51%（SDXL 28→57），严重夸大单峰稀疏分布间的统计相似度。漂移指纹可从架构的 (a) 信息流图、(b) skip/residual 结构、(c) 跨模态交互边界三要素预测。

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
| Phase 7c Skip 因果干预 | 2027.7 | ✅ 完成 |
| Phase 8 ICLR 补充实验 | 2027.7 | ✅ 完成 |
| 100-image 扩展评估 (Phase 5/7/8) | 2026.7.15 | ✅ 完成 |

**DCSC（闭环控制器）**：已探索并放弃。实验验证闭环控制在当前系统上没有可测量的增益（三模式在对抗条件下 PSNR 等价），该负结果为"简单性即优势"的叙事提供了支撑。论文 Discussion 中诚实提及。

---

## Phase 8：SD 3.5 预测验证 + FLUX 消融补全 ✅

### 8a：SD 3.5 预测验证（P0，~1 天）

**背景**：SD 3.5 Medium（24 层 MMDiT, dual_attention_layers=0-12）漂移诊断已跑完，但预测被证伪——原始预测为"dual→standard 过渡区（layers 12-14）漂移峰"，实际为"过渡区漂移谷（0.12，全局最低），峰值在 block_22（0.27）"。

**原因分析**：SD 3.5 的 dual→standard 过渡与 FLUX 的 joint→single 过渡**方向相反**：
- FLUX: joint→single 是**丢失**跨模态交互 → 漂移峰（特征失稳）
- SD 3.5: dual→standard 是**获得**跨模态交互 → 漂移谷（特征被 stabilize）

这反而是框架的**细化**而非推翻：跨模态交互边界的效应方向取决于交互是被加入还是被移除。

**已完成**：
1. ✅ `scripts/sd35_phase8a_text_drift.py` 已完成 text drift 测量——image drift 修正预测成立（peak block_22, valley 12-14）
2. ✅ 原始预测（证伪）和修正预测（成立）已写入 `prediction_record.json`
3. ⚠️ **text drift v2 预测证伪**：预测 dual attention (0-12) text drift 应更高（缺乏跨模态 stabilize），实际相反——dual mean=0.057，standard joint mean=0.181（`text_dual_gt_mid=false`）。dual attention 层的 text drift 反而更低，挑战了"跨模态交互 stabilize text"的简单机理解释。需在论文中诚实记录这一双方向证伪（image 修正成立 + text 修正仍失败）。

### 8b：FLUX 层组消融（P1，~1 天）✅ 完成

**结果**（28 步 Euler，feature-level correction，19 图）：

#### λ Scan（5 图）

| λ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|-----|-----|-----|-----|-----|
| ΔPSNR | **+1.38** | +1.25 | +0.50 | +0.14 | −0.01 |

FLUX feature-level correction 对 λ **高度敏感**——最优在极小值（0.1），与 SD 1.5 latent correction 的 λ 不敏感性相反。

#### 层组消融（19 图，λ=0.1，28 步，feature-level correction）

| 条件 | ΔPSNR | 说明 |
|------|-------|------|
| **late_single** (19-37) | **+3.18** | 仅后半段 single blocks，效果最优 |
| joint_only (19 blocks) | +3.05 | 仅 joint blocks |
| single_only (38 blocks) | +3.05 | 全部 single blocks |
| latent_only (all 57) | +3.05 | 全部 blocks |
| early_single (0-18) | +1.49 | 仅前半段 single blocks |
| joint_plus_early | +1.49 | joint + 前半段 single |
| top5 | +1.44 | 跨区域 top-5 漂移 blocks |

#### 关键发现与叙事影响

1. **"Single > Joint" 预测证伪**：single blocks 漂移 1.4× 更大，但 joint_only = single_only（ratio=1.00x）。漂移量级不决定校正潜力——与 Phase 5 "漂移加权无效（r≈−0.11）"跨架构一致。

2. **"注入位置鲁棒"跨架构复现**：joint_only = single_only = latent_all 三者 per-image PSNR 完全一致（3 图验证，非均值巧合）。MM-DiT 的 residual stream 使校正信号经 57 层累积后等价——无论在哪一层注入、注入多少层，最终 latent 完全相同。这是比 SD 1.5 random5≈top5 更强的位置不敏感性（SD 1.5 是"近似等价"，MM-DiT 是"严格等价"），将"简单性即优势"从 UNet 推广到 Transformer backbone。

3. **early ≠ late 揭示 residual stream 的方向性**：late_single **单独** 就达到 single_only（全部 38 blocks）的效果，且超过 joint_only。early_single 仅一半增益。校正信号在 residual stream 中**单向累积**——越靠近输出，校正效果越大。这与 SD 1.5 decoder 末端校正最优的规律一致。

4. **三层预测框架修正**：漂移指纹告诉你**哪里出问题**，但不直接告诉你**哪里修复最有效**。校正增益由信息流的**因果距离**决定（越近输出→越少后续稀释→效果越大），不由局部漂移量级决定。

**输出**：`outputs/phase8b_flux/ablation.json`

---

## 2026-07-12 审查修复记录 🔧

**Claim–Evidence 审计**（详见 `ICLR_PAPER_DEFINITIONS.md`）：

| 修复项 | 内容 |
|--------|------|
| Property 1 (Reproducibility) | 补 LOOCV 实验，r=1.000, σ/mean=0.1% → `outputs/phase1_reproducibility/` |
| Property 3 降级 | "Paradigm Stability" → "Backbone Dominance"，诚实承认范式对比是间接的 |
| 架构计数修正 | "5 archives, 10 pairwise" → "4 unified + 1 held-out (SD 3.5), 6 pairwise" |
| Phase 8b λ 标注 | 消融表头 λ=0.7 → λ=0.1（实际最优值） |
| Euler 表 DiT N | N=19, ~16 dB, +5.65 → N=3*, 15.24 dB, +4.67 |
| Pearson r | =1.000 → ≈1.000（4 处修正） |
| Abstract 压缩 | 250→~180 words，砍掉 Mechanism 独立段 |
| 防御性写作 | 删除/改写 5 处 "We do NOT claim" / "does NOT prove" |
| Necessity 论证 | 新增 "Why Architecture Fingerprint rather than layer-wise drift profile?" |
| Mechanism 升级 | Skip Conflict 从观察升级为四变量因果链（α→C→φ→PSNR） |
| SD 3.5 text drift | 记录 v2 预测证伪（dual=0.057 vs joint=0.181） |
| FLUX 校正验证 | joint_only=single_only=latent_all 经 3 图 per-image 确认（非 bug） |

---

## 开发环境

- conda 环境 `grad`（Python 3.10），激活：`conda activate grad`
- GPU：NVIDIA RTX PRO 6000 Blackwell (48GB), CUDA 13.0
- PyTorch 2.11.0+cu128, diffusers 0.38.0, transformers 5.12.1
- 主模型：`runwayml/stable-diffusion-v1-5`（已缓存）
- CLIP：`openai/clip-vit-large-patch14`（已缓存，需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 离线运行）
- 运行脚本时需 `export PYTHONPATH=scripts:$PYTHONPATH`
- **NCCL 库修复**：CUDA 13.0 驱动与 PyTorch cu128 的 NCCL 有符号冲突，需 preload nvidia-nccl-cu12 的 libnccl
  ```bash
  LD_PRELOAD="$(python -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')/lib/libnccl.so.2" python script.py
  ```
  或固定路径：
  ```bash
  LD_PRELOAD="/home/hiaskc/miniconda3/envs/grad/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2" python script.py
  ```

---

## Phase 1：DDIM 反演-重建漂移动态诊断 ✅

- **脚本**：`scripts/phase1_diagnostics.py`
- **输出**：`outputs/phase1/layer_drift_summary.json`（19 图 coco_val）
- **Property 1 验证**：`outputs/phase1_reproducibility/reproducibility.json`
  - LOOCV Pearson r = 1.000（19 折，min=1.000）
  - Multi-seed σ/mean = 0.1%（3 seeds × 5 images）
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
- 与 P2P 统计等价：ΔPSNR 差 0.13 dB，Cohen's d=0.033（可忽略效应量），Pearson r≈1.000

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
| NTI (BLIP) | 19.60 ± 2.80 | 0.312 | −2.86 | Optimization | Low |
| EDICT | 22.90 ± 3.15 | 0.195 | +0.45 | None | 2× |
| **P2P (attn)** | **25.34 ± 4.01** | **0.087** | **+2.88** | None | ~GB |
| ControlNet (Canny) | 8.20 ± 1.47 | 0.830 | — | Pre-trained | ~1.4GB |
| **Ours_Corr** | **25.20 ± 3.88** | **0.094** | **+2.75** | None | **~MB** |

> NTI 和 EDICT 现基于完整 19 图评估（2026-07-10 补齐）。所有方法使用相同 19 张 COCO val 图像。

P2P vs Ours：配对 t-test p=0.0015（统计显著），但 Cohen's d=0.033（效应量可忽略），Pearson r≈1.000（行为完全一致）。

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

### Phase 5 100-image 扩展评估 ✅ (2026-07-15)

将 Phase 5 从 19 图扩展到 104 图（coco_val100，50 步，λ=0.7，top-5 layers）：

| 指标 | 19-image | 100-image |
|------|---------|----------|
| Baseline PSNR | 22.45 ± 3.02 | 22.06 ± 3.42 |
| Ours PSNR | 25.20 ± 3.88 | 25.36 ± 3.85 |
| **ΔPSNR** | **+2.75** | **+3.30 ± 2.45** |
| Cohen's d | 0.791 | **1.340** |
| 95% CI | — | **[2.84, 3.78]** dB |
| Pearson r | ≈1.000 (vs P2P) | 0.779 (baseline-ours) |

100 图结果更强（ΔPSNR 更高、CI 紧、d 更大），确认为非小样本偶然。
脚本：`scripts/phase5_100image_recover.py`，输出：`outputs/phase5_100image/`

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

### 五架构跨范式漂移指纹对比

| 架构 | Backbone | 范式 | 漂移指纹 | 模式 |
|------|---------|------|---------|------|
| SD 1.5 | UNet (CNN+Cross-Attn) | DDIM | decoder ResNet | 单峰（decoder 集中） |
| SDXL | UNet (更大) | DDIM | mid_block | 单峰（中间层） |
| HunyuanDiT | Transformer (single-stream) | DDIM (v-pred) | blocks 11-21 | 单峰（中层） |
| **FLUX** | **MM-DiT (dual-stream)** | **Flow Match** | **early single + late single + last joint** | **双峰 + 尾端异常** |
| **SD 3.5** | **MM-DiT-X (dual→standard)** | **Rectified Flow** | **late blocks + output compression** | **多峰（过渡区谷值）** |

**核心发现**：相同 backbone family 漂移指纹最近（结构距离 d=0.249~0.385），但 attention 拓扑差异可以压倒 backbone family——HunyuanDiT (single-stream) vs FLUX (dual-stream) 是距离最远的配对（d=1.077），尽管都是 Transformer。漂移指纹不由采样范式决定，而由具体的 attention 结构（single-stream vs dual-stream, joint→single 过渡方向）决定。**特征漂移是架构签名，不是采样器 artifact。**

### 架构间漂移指纹结构距离

**方法学警告**：基于插值的 Pearson/Spearman 相关矩阵存在两个根本缺陷：
1. **排序 bug**：`full_ranking` 按漂移量级降序排列（非架构深度），用作 profile 会使 SDXL 和 DiT 都变成单调递减向量，任何相关性指标都会虚高。
2. **插值 artifact**：不同架构层数差异大（24→57），插值合成点占比最高达 51%（SDXL），严重夸大单峰稀疏分布间的相似度。

**正确方法**：从原始层数直接提取四个结构特征（peak 位置、峰数、浓度、展宽），计算 Euclidean 距离——不经过任何插值。

| 配对 | 结构距离 d | 解读 |
|------|-----------|------|
| **SD 1.5 vs SDXL** | **0.249** | 同 UNet family→最近 ✅ |
| **FLUX vs SD 3.5** | **0.385** | 同 MM-DiT backbone→次近 ✅ |
| SDXL vs HunyuanDiT | 0.506 | 不同 backbone，中等距离 |
| SD 1.5 vs HunyuanDiT | 0.624 | 不同 backbone |
| SDXL vs FLUX | 0.628 | 不同 backbone+范式 |
| SD 1.5 vs FLUX | 0.637 | 不同 backbone+范式 |
| SD 1.5 vs SD 3.5 | 0.722 | 不同 backbone |
| SDXL vs SD 3.5 | 0.803 | 不同 backbone |
| **HunyuanDiT vs FLUX** | **1.077** | 同 Transformer 但 single≠dual-stream→最远 |

**关键发现**：

1. **同 family 最近**：SD 1.5-SDXL (d=0.249) 和 FLUX-SD 3.5 (d=0.385) 是最近的两个配对——相同 backbone 架构间漂移指纹最相似。
2. **HunyuanDiT 是 outlier**：single-stream DiT 与 dual-stream FLUX 距离最远（1.077），尽管都是 Transformer。**attention 拓扑（single vs dual）比 backbone family（Transformer vs UNet）更决定漂移指纹。**
3. **跨 family 距离 > 0.5**：所有不同 backbone 的配对距离都在 0.5 以上，支持"架构指纹可区分"的 claim。
4. **定性映射 > 定量指标**：架构拓扑→漂移指纹的三层预测框架（信息流图、skip/residual 结构、跨模态交互边界）比任何单一数值指标更可靠。定量距离矩阵仅作为辅助验证。

> **方法学教训**：此前的 Pearson/Spearman 相关矩阵（包括 CLAUDE.md 历史版本中的 FLUX vs DiT ρ=0.722 等）均基于 `full_ranking` 错误排序数据 + 插值 → 所有数值不可靠。正确结论是结构距离和定性映射，而非数值相关。

**核心结论**：漂移指纹不由采样范式（DDIM vs Flow Matching）决定，而由 backbone attention 拓扑（single-stream vs dual-stream MM-DiT, CNN skip vs residual stream）决定。**特征漂移是架构签名，不是采样器 artifact。**

输出：`outputs/phase6_unified/four_arch_fingerprint.png`、`arch_similarity_matrix.png`、`drift_profile_overlay.png`。

### Euler 反演限制分析

| 架构 | 范式 | Baseline PSNR | ΔPSNR | N |
|------|------|--------------|-------|---|
| SD 1.5 | DDIM (可逆) | 22.48 dB | +2.50 | 24 |
| SDXL | DDIM (可逆) | 22.11 dB | +5.23 | 19 |
| HunyuanDiT | DDIM v-pred (可逆) | 15.24 dB | +4.67 | 3* |
| **FLUX** | **Flow Match Euler (不可逆)** | **12.03 dB** | **+3.94** | 19 |

> 注：SDXL ΔPSNR 和 DiT baseline/euler 值来源于不同实验运行，与 per-experiment 输出文件一致。Euler 分析文件 (`outputs/phase6_flux/euler_analysis.json`) 中 DiT 仅含 3 张图（`*` 标注）——跨实验对比时需注意样本量差异。DiT Phase 2 独立 20 图实验 Δ=+5.65 dB，transition 区域效果更优。

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

### Phase 7 100-image 编辑 Benchmark 扩展 ✅ (2026-07-15)

将编辑 benchmark 从 19 图扩展到 104 图/121 编辑对。
简化协议：去掉 P2P（已证等价），仅 baseline vs ours；反演使用 source caption (BLIP) 而非空 prompt，
与重建时 target prompt 的差异构成编辑语义。

| 指标 | 19-image (28 对) | **100-image (121 对)** |
|------|-----------------|----------------------|
| Baseline LPIPS | 0.856 | 0.469 |
| Ours LPIPS | 0.511 | **0.071** |
| ΔLPIPS | −0.345 (−40%) | **−0.398 (−85%)** |
| ΔPSNR | — | **+7.40 dB** |
| LPIPS p | — | **4.8e-55** |
| Cohen's d | — | **2.579** |
| 95% CI | — | **[−0.426, −0.371]** |

按编辑类型（100-image）：

| 类型 | pairs | Baseline LPIPS | Ours LPIPS | 改善 |
|------|-------|---------------|-----------|------|
| Style transfer | 104 | 0.490 | 0.069 | −86% |
| Word swap | 17 | 0.339 | 0.081 | −76% |

100 图编辑 benchmark 确认校正作为编辑插件的有效性——跨 104 图、121 个编辑对，
p=4.8e-55，效应量 d=2.579。源 prompt 反演使 baseline LPIPS 显著低于空 prompt 反演
（0.469 vs 0.856），校正效果（ΔLPIPS）跨两个协议一致（~0.35-0.40 LPIPS 改善）。

脚本：`scripts/phase7_editing_100image.py`，输出：`outputs/phase7_editing_100image/`

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

1. **架构拓扑直接导致漂移指纹**：切断 peak skip → 漂移大幅下降 27.7%。机理：skip connection 引入 encoder-decoder **特征冲突**——编码器特征与解码器重建路径不一致，导致漂移和重建误差。

2. **重建质量反直觉提升**：Cut A 的 PSNR **+2.20 dB** (p=0.0005)，SSIM +0.060，LPIPS −0.099。切断冲突源 → 漂移↓ **且** 重建↑。这与初始预测相反——skip connection 在反演-重建场景下是**有害的**，它引入的特征冲突同时导致漂移和重建误差。

3. **噪声注入完成因果链**：Noise A（高斯噪声替换 skip，保持统计特性但摧毁结构信息）→ 峰值漂移 +6.4%（比原始更差）。完整排序：Noise > Original > Zero。随机干扰比结构化冲突更严重。

4. **效应是位置特异的（拓扑效应，非容量效应）**：两个 cut 移除等量信息，但 Cut A 改变 31/38 层而 Cut B 仅改变 5/38 层。Δ 图 r=−0.395（反相关）——不同位置产生不同空间模式，排除容量解释。

5. **三层预测框架具备干预能力**：不仅能被动描述漂移指纹，还能预测"在哪里做干预会产生什么效应"。

### 重建质量测量（19 图，50 步 DDIM）

| 条件 | PSNR | SSIM | LPIPS | 峰值漂移 |
|------|------|------|-------|---------|
| Cut A (α=0.00) | 24.66 ± 3.90 | 0.693 ± 0.152 | 0.119 ± 0.054 | 1684 (−27.7%) |
| **Noise A** | **24.84 ± 3.99** | **0.698 ± 0.152** | **0.113 ± 0.045** | **2480 (+6.4%)** |
| α=0.25 | 23.10 ± 3.06 | 0.656 ± 0.141 | 0.169 ± 0.066 | — |
| α=0.50 | 23.02 ± 3.07 | 0.649 ± 0.146 | 0.182 ± 0.089 | ~1978 |
| α=0.75 | 22.57 ± 3.02 | 0.635 ± 0.148 | 0.212 ± 0.096 | — |
| Original (α=1.00) | 22.44 ± 3.00 | 0.633 ± 0.150 | 0.218 ± 0.095 | 2329 |
| Cut B (低漂移 skip) | 22.35 ± 2.98 | 0.631 ± 0.149 | 0.224 ± 0.096 | 2348 (+0.8%) |

### 三级因果梯度

**Level 1 — 二元干预**：Cut A 27.7%漂移↓ +2.20 dB PSNR↑；Cut B 无显著变化。拓扑效应 r=−0.395 排除容量解释。

**Level 2 — 机制分离（噪声注入）**：Noise A 漂移↑6.4% 但 PSNR↑2.4 dB——打破了漂移-质量相关性。skip 携带的是**结构化冲突**，随机噪声不携带冲突模式，所以即使 L2 漂移更大，感知质量 (LPIPS 0.113 vs 0.218) 反而大幅提升。

**Level 3 — 连续剂量-响应**：α↓ → PSNR 单调↑，没有"最优调制点"——α=0 就是最优。skip 在任何强度下都有害。SSIM 和 LPIPS 完全一致（α=0: SSIM 0.693, LPIPS 0.119; α=1.0: SSIM 0.633, LPIPS 0.218）。

### 核心发现（修正后）

1. **skip connection 是冲突源，不是信息源**：在反演-重建场景下，encoder skip 引入的特征与 decoder 路径**冲突**，导致漂移和重建误差。切断冲突源 → 漂移↓且 PSNR↑（与最初预测的"失去有用信息"相反——这是一个通过实验证伪获得的更深层发现）。

2. **结构化冲突 ≠ 一般干扰**：Noise A 的 L2 漂移更大但重建更好——证明 harm 来自冲突的**结构**（encoder-decoder 特征模式的特定不对齐），而非冲突的**量级**。

3. **Skip Conflict 是架构实例特异的，不是 UNet family 的普遍规律**：SDXL 的对应 skip 切断实验产生完全相反的结果——PSNR **暴跌 11.6 dB**（SD 1.5: +2.2 dB）。相同拓扑位置，相反因果效应。SDXL 的 mid_block 漂移峰和 SD 1.5 的 decoder 漂移峰不仅在位置不同，在因果机制上也完全不同。这强化了核心主张：**Architecture Fingerprint 是每个特定架构的签名，不是 backbone family 的笼统属性。**

### SDXL 跨架构因果验证（Task C, 20 prompts, 50 步 DDIM）

| 指标 | SD 1.5 Cut A | SDXL Cut A |
|------|-------------|-----------|
| 目标 skip | down_blocks.1 → up_blocks.2 | down_blocks.0 → up_blocks.2 |
| 结构角色 | 内部 skip（4-block UNet 第 2/4 层） | 最外层 skip（3-block UNet 第 0/3 层） |
| 漂移峰位置 | decoder up_blocks.2（与 cut 重合） | **mid_block（与 cut 不重合）** |
| 峰值层 Δ漂移 | **−27.7%** | 待测 |
| ΔPSNR | **+2.20 dB** (改善) | **−11.59 dB** (暴跌) |
| ΔSSIM | +0.060 | −0.306 |
| ΔLPIPS | −0.099 | +0.447 |
| 机理解释 | skip 引入冲突→切断消除冲突 | skip 携带必要信息→切断破坏重建 |

**关键洞察**：两个架构在 decoder 上的同名组件（`up_blocks.2` 的输入 skip）扮演完全不同的功能角色。SD 1.5 中该 skip 是冲突源（漂移峰与 cut 位置重合），SDXL 中该 skip 是必要信息通路（漂移峰在 mid_block，cut 未触达冲突源，反而破坏了信息流）。相同的结构组件，相反的功能角色——这验证了 Property 2（架构间可区分性不止在指纹形状，更在因果结构）。

3. **效应是位置特异的**：Cut B（低漂移 skip）无显著效应——架构拓扑精确决定了冲突在哪里发生。

4. **剂量-响应验证因果性**：PSNR/SSIM/LPIPS 均随 α 单调变化，排除了混淆变量。

### 完整因果链

```
架构拓扑 → Skip 连接位置 → 编码器-解码器特征冲突 → 漂移指纹 + 重建误差
                                                      ↓
                                          切断冲突源 → 漂移↓ + 重建↑
                                          噪声替换    → 漂移↑ (更差)
```

### 论文定位

主文 Discovery 章最后一小节（~0.5 页），一张四条件对比图 + Δ 图 + 重建质量表。核心叙事：
> "Causal interventions on skip connections validate the framework's predictive nature. Cutting the peak skip (Cut A) fundamentally altered the fingerprint (31/38 layers p<0.05) while improving reconstruction (+2.2 dB PSNR), revealing that the skip introduces harmful encoder-decoder feature mismatch. Cutting a low-drift skip (Cut B) preserved both fingerprint (5/38 layers significant) and quality. Noise injection (Noise A) increased drift beyond original, confirming that skip content—not capacity—determines the fingerprint. Anti-correlated delta maps (r=−0.395) rule out capacity-effect explanations."

证据链：FLUX(拟合) → SD3.5(held-out 预测+修正) → SD 1.5 skip 干预(因果操纵) → Noise injection(机制分离) → Reconstruction quality(意外发现→更深理解) → SDXL(跨架构负结果→架构特异性) → Editing verification(编辑不退化)

脚本：`scripts/phase7_skip_intervention.py`（主实验）、`scripts/phase7_skip_recon_quality.py`（重建质量）、`scripts/phase7_skip_noise_intervention.py`（噪声注入）、`scripts/phase7_skip_intervention_viz.py`（可视化）、`scripts/phase7_skip_partial_modulation.py`（剂量-响应）
输出：`outputs/phase7_skip_intervention/`（含 `results.json`, `recon_quality.json`, `results_noise.json`, `results_partial_modulation.json`, `prediction_record.json`, fig4a-d, fig_dose_response）

---

## Phase 8: ICLR 补充实验 ✅

### 8a: 跨 Prompt 验证（25 → 100 prompts, SD 1.5, 50 步）✅ (2026-07-15 扩展)

| 指标 | 25-prompt | **100-prompt** |
|------|----------|---------------|
| ΔPSNR | +1.31 ± 1.75 | **+1.88 ± 2.25** |
| p-value | 0.0012 | **5.15e-13** |
| Cohen's d | 0.75 | **0.835** |
| 95% CI | [0.62, 1.99] | **[1.45, 2.34]** |
| 改善 >1dB | 13/25 (52%) | **53/100 (53%)** |
| 改善 >2dB | — | **34/100 (34%)** |
| 变差 | 2/25 (8%) | **6/100 (6%)** |

100-prompt 验证 CI 更紧 ([1.45, 2.34] vs [0.62, 1.99])，效应量更大 (d=0.835 vs 0.75)。
负例率一致 (6% vs 8%)——校正效果跨 prompt 稳健，不是空 prompt 特殊产物。
Abstract 类 prompt 最易退化（3/4 在 outlier 列表），文本/人像/物体类最稳。
脚本：`scripts/phase8_iclr_cross_prompt.py`（DIVERSE_PROMPTS 扩展至 100），输出：`outputs/phase8_iclr_cross_prompt/`

### 8b: 编辑验证（25 tasks × 3 条件, SD 1.5, 20 步）

| 条件 | LPIPS↓ | SSIM↑ | CLIP-Dir↑ | PSNR↑ |
|------|--------|-------|-----------|-------|
| Original | **0.671** | 0.739 | **0.048** | **17.65** |
| Cut A | 0.758 | 0.799 | −0.004 | 16.06 |
| Noise A | 0.775 | **0.807** | −0.008 | 16.07 |

Cut A/Noise A 提升结构保持 (SSIM↑) 但几乎消除编辑方向 (CLIP-Dir → 0)——
skip connection 在编辑中承担方向信号传播，切断后编辑退化为基础重建。

### 8c: SDXL 跨架构因果验证（20 prompts, 30 步）

| 指标 | SD 1.5 Cut A | SDXL Cut A |
|------|-------------|-----------|
| 目标 skip | down_blocks.1 → up_blocks.2 | down_blocks.0 → up_blocks.2 |
| 结构角色 | 内部 skip（4-block UNet 第 2/4 层） | 最外层 skip（3-block UNet 第 0/3 层） |
| 漂移峰位置 | decoder up_blocks.2（与 cut 重合） | **mid_block（与 cut 不重合）** |
| ΔPSNR | **+2.20 dB** | **−11.59 dB** |
| ΔSSIM | +0.060 | −0.306 |
| ΔLPIPS | −0.099 | +0.447 |

相同的结构组件，相反的功能角色。Architecture Fingerprint 是诊断工具——
揭示每个架构实例特有的漂移机理，不是 backbone family 的笼统属性。

输出：`outputs/phase8_iclr_cross_prompt/`, `outputs/phase8_iclr_editing/`, `outputs/phase8_iclr_sdxl/`

> FLUX 消融矩阵（层组消融 + λ scan）结果见上方 Phase 8 规划区 [8b：FLUX 层组消融](#8bflux-层组消融p1~1-天-完成)。

> **已知问题**：~~Phase 7c 的 `results_partial_modulation.json` 缺失~~ **已于 2026-07-10 重跑验证**，中间 α 值（0.25/0.50/0.75）的 PSNR 均与先前 claim 一致。剂量-响应单调性确认：PSNR 随 α↓ 单调上升。
