# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

硕士毕业设计：**基于诊断驱动的扩散模型特征漂移分析与校正**。作者为塔拉尼提·居马努尔。

**核心贡献**：

1. **Diagnosis-driven diffusion feature analysis.** 系统量化并定位 DDIM 反演-重建过程中的逐层特征漂移，揭示 UNet 各层的信息分工结构（ResNet vs Attention，encoder vs decoder vs bottleneck）。将扩散反演从黑盒过程转变为可诊断的结构系统。

2. **Theoretical understanding of feature drift.** 三理论统一框架：信息论（残差的可校正信息含量）、流形几何（残差对齐于特征流形切空间）、收敛性分析（skip connection 传播与误差收缩）。三个理论从不同角度解释同一现象。

3. **Minimal correction with cross-architecture validation.** 基于诊断结果的最简残差校正机制，跨 SD 1.5 / SDXL / DiT 三种架构验证有效。与 P2P 内容保持能力相当（ΔPSNR +2.92 vs +2.98），但内存低数百倍。诊断驱动的层选择使方法高效——校正对注入位置鲁棒（skip connection 传播定理）。

**核心发现**：诊断揭示架构瓶颈（ResNet 漂移 >> Attention），校正利用瓶颈特征做鲁棒注入（不依赖精细层选择），skip connections 使任意 ResNet 层校正效果等价。

## 项目阶段

| 阶段 | 时间 | 状态 |
|------|------|------|
| Phase 1 诊断 | 2026.5 | ✅ 完成 |
| Phase 2 校正 | 2026.6 | ✅ 完成 |
| Phase 4 理论 + SOTA | 2027.2–7 | ✅ 实验完成，论文撰写中 |
| 跨架构验证 | 2026.7 | ✅ SDXL / DiT 全部通过 |

Phase 3（风格编辑/DCSC）已删除。校正机制本身足够强——不需要加控制层。

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
- **输出**：`outputs/phase1/layer_drift_summary.json`
- **关键发现**：漂移集中在 decoder up_blocks ResNet 层（`up_blocks.2.resnets.0` 跨图最高）。ResNet 漂移比 Attention 大约一个数量级。早期 5 图结果受人脸图影响偏向 decoder，独立 coco_val（19 图）验证确认 mid_block 贡献显著。

---

## Phase 2：零训练残差校正模块 ✅

核心公式：$f_{out} = f_{recon} + \lambda \cdot (f_{inv} - f_{recon})$

### 关键结果

- **测试集**（24 图，50 步，λ=0.7）：平均 Δ PSNR **+2.50 dB**，LPIPS 从 0.190 → 0.086（↓55%）
- **coco_val only**（19 图，λ=0.7）：平均 Δ PSNR **+2.76 dB**，LPIPS 从 0.218 → 0.094
- **基线对比**（17 图，50 步）：DDIM+Corr (+3.90 dB) >> EDICT (+0.37 dB) > DDIM > NTI (−2.43 dB)
- **消融**（3 图，λ=0.7）：latent_interp ≈ random5 ≈ top5 ≈ encoder5 > attention5

### 核心发现

1. **校正对注入位置鲁棒**：random5 ≈ top5（差 < 0.3 dB），任何 decoder ResNet 层组效果接近。原因：UNet skip connections 传播校正信号
2. **漂移加权无效**：per-layer 权重 w_i ∝ drift_i 不能拉开 top5 与 random5 差距
3. **latent 空间插值优于特征空间校正**，但 latent 空间无法做内容/风格解耦
4. **诊断的价值不在"选层"**而在揭示架构级瓶颈（decoder ResNets 贡献 >99% 漂移，Attention 可忽略）

---

## Phase 4：理论深化 + SOTA 对比 ✅

### 信息论分析

**方法**：逐层边际校正收益。对每一层单独注入残差校正，ΔPSNR 测量该层残差的可校正信息含量。

| 层类型 | ΔPSNR |
|--------|-------|
| **ResNet** | **+2.27 ± 0.48 dB** |
| Attention | +1.09 ± 0.48 dB |
| **比率** | **2.1×** |

Top-5 ΔPSNR 层：`down_blocks.0.resnets.0` (+2.79), `up_blocks.3.resnets.1` (+2.78), `down_blocks.0.resnets.1` (+2.75), `up_blocks.3.resnets.0` (+2.75), `up_blocks.2.resnets.2` (+2.70)。

Bottom：`up_blocks.0.attentions.0` ΔPSNR = **0.00**——残差与像素重建完全正交。

脚本：`scripts/phase4_info_theory.py`，输出：`outputs/phase4_info_theory/`

### 流形视角

ResNet 残差比 Attention 更贴合流形切空间（对齐度 0.572 vs 0.420）。特征流形呈沙漏形状：encoder 浅层 dim=4 → bottleneck dim=35 → decoder 深层 dim=2。

最高对齐层：`down_blocks.0.resnets.0` (0.908, dim=4), `up_blocks.3.resnets.2` (0.904, dim=2)。

脚本：`scripts/phase4_manifold.py`，输出：`outputs/phase4_manifold/`

### 收敛性分析

**误差收缩**：$\|T_\lambda(f) - f^{\text{inv}}\| = |1-\lambda| \cdot \|f - f^{\text{inv}}\|$

**Skip connection 传播**：$d_{l+1} \approx (I + \nabla F_l) \cdot \lambda d_l \approx \lambda d_l$（当 $\|\nabla F_l\| \ll 1$）

直接解释 random5 ≈ top5：校正信号以 ≈ 单位增益通过 skip connections 传播。

*注意：上述推导基于一阶 Taylor 展开和 $\|\nabla F_l\| \ll 1$ 的假设。论文中应标注为"推导/命题"而非"定理"，并明确假设条件。*

脚本：`scripts/phase4_convergence_verify.py`，输出：`outputs/phase4_convergence/`

### 跨架构漂移指纹 ✅

| 架构 | 漂移集中区域 | 独特发现 |
|------|------------|---------|
| SD 1.5 (UNet) | decoder up_blocks ResNet | ResNet >> Attention |
| SDXL (UNet) | mid_block | 与 SD 1.5 完全不同 |
| DiT (Transformer) | bottom→top 过渡区 (blocks 11-21) | 无 ResNet/residual 概念 |

脚本：`scripts/phase4_fingerprint.py`，输出：`outputs/phase4_sota/cross_arch_fingerprint.png`

### SOTA 横向对比（19 图 coco_val，50 步）

| Method | PSNR↑ | LPIPS↓ | ΔPSNR | Training | Memory |
|--------|-------|--------|-------|----------|--------|
| DDIM (baseline) | 20.78 | 0.269 | — | None | Low |
| EDICT | 21.15 | 0.256 | +0.37 | None | 2× |
| NTI (BLIP) | 18.35 | 0.353 | −2.43 | Optimization | Low |
| **P2P (attn)** | **23.77** | **0.089** | **+2.98** | None | ~GB |
| ControlNet (Canny) | 9.55 | 0.781 | — | Pre-trained | ~1.4GB |
| **Ours_Corr** | **23.70** | **0.097** | **+2.92** | None | **~MB** |

P2P 与 Ours_Corr 效果相当（差 0.06 dB），但 Ours 内存低数百倍。诊断驱动 + 零训练 + 跨架构验证是差异化优势。

### 三类场景验证（50 步，λ=0.7）

| 场景 | Baseline PSNR | Correction Δ |
|------|-------------|-------------|
| 人像 (8张) | 26.39 | **+4.93 dB** |
| 建筑 (5张) | 21.24 | **+6.46 dB** |
| 艺术字体 (5张) | 22.03 | **+5.14 dB** |

脚本：`scripts/phase4_scenes.py`，输出：`outputs/phase4_sota/scenes/`

---

## 设计原则

1. **Diagnosis precedes intervention**（诊断先于干预）：Phase 1 的逐层漂移诊断先于 Phase 2 的校正，确保干预有依据
2. **Correction is geometry-aware**（校正利用几何结构）：信息论 + 流形分析证明残差是流形切方向的有意义信号
3. **Simplicity over complexity**（简单优于复杂）：1 层校正 ≈ 5 层效果。诊断告诉我们不需要复杂控制——skip connections 本身就是天然的鲁棒性保证

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
| `scripts/phase4_info_theory.py` | 逐层可校正信息含量分析 |
| `scripts/phase4_manifold.py` | 特征流形分析与校正几何解释 |
| `scripts/phase4_convergence_verify.py` | 收敛性数值验证 |
| `scripts/phase4_fingerprint.py` | 跨架构漂移指纹图 |
| `scripts/phase4_summary.py` | SOTA 综合对比表 |
| `scripts/phase4_p2p.py` | Prompt-to-Prompt 交叉注意力对比 |
| `scripts/phase4_controlnet.py` | ControlNet Canny 条件生成对比 |
| `scripts/phase4_scenes.py` | 三类场景验证 |
| `scripts/gen_thesis_figures.py` | 论文配图生成 |
| `scripts/gen_unified_framework_diagram.py` | 统一框架架构图 |
| `scripts/gen_unified_ablation_table.py` | 统一消融汇总表 |
| `scripts/gen_failure_case_figure.py` | 失败案例分析图 |
| `scripts/sdxl_phase1_diagnostics.py` | SDXL Phase 1 诊断 |
| `scripts/sdxl_phase2_full.py` | SDXL Phase 2 校正 |
| `scripts/sdxl_phase3_prep.py` | SDXL Phase 3 风格注入（可选） |
| `scripts/dit_phase1_diagnostics.py` | DiT Phase 1 诊断 |
| `scripts/dit_phase2_common.py` | DiT Phase 2 共享（v_prediction DDIM、3D token） |
| `scripts/dit_phase2_full.py` | DiT Phase 2 校正 |
| `scripts/dit_phase3_prep.py` | DiT Phase 3 风格注入（可选） |

## 数据分集

| 分集 | 路径 | 数量 | 用途 |
|------|------|------|------|
| coco_val | `data/coco_val/` | 19 张 | 层选择 + 定量评估 |
| basetest | `data/basetest/` | 8 张 | 历史测试 |
| 人像 | `data/portraits/` | 8 张 | 场景验证 |
| 建筑 | `data/architecture/` | 5 张 | 场景验证 |
| 艺术字体 | `data/typography/` | 5 张 | 场景验证 |

## 领域调研

**反演方法**：DDIM → EDICT → NTI

**内容保持**：LAMS-Edit（最接近，开环混合）→ Plug-and-Play RLI → DiffStateGrad（SVD 低秩）

**差异化定位**：诊断驱动 + 理论闭环 + 跨架构验证 + 内存优势。不是新方法，是新认识。
