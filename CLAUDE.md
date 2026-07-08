# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

硕士毕业设计：**诊断驱动的扩散模型反演特征漂移分析与零训练校正**。作者为塔拉尼提·居马努尔。

**核心贡献**：

1. **Diagnosis-driven diffusion feature analysis.** 系统量化并定位 DDIM 反演-重建过程中的逐层特征漂移，揭示 UNet 各层的信息分工结构（ResNet vs Attention，encoder vs decoder vs bottleneck）。将扩散反演从黑盒过程转变为可诊断的结构系统。

2. **Theoretical understanding of feature drift.** 三理论互补框架：信息论（因果消融 + 互信息估计）、流形几何（残差的切空间对齐）、收敛性分析（skip connection 传播与误差收缩）。三个理论从不同角度解释同一现象，相互印证。

3. **Minimal correction with cross-architecture validation.** 基于诊断结果的最简残差校正机制，跨 SD 1.5 / SDXL / DiT / FLUX 四种架构两种采样范式验证有效。与 P2P 内容保持能力统计等价（19 图，ΔPSNR 差 0.13 dB，Cohen's d=0.033），内存低数百倍。

**核心发现**：诊断揭示架构瓶颈（ResNet 漂移 >> Attention），校正利用瓶颈特征做鲁棒注入（1 层 ≈ 5 层效果）。简单性是诊断的成果，不是方法的局限。

## CVPR 目标创新点

投稿级创新点：

1. **跨架构 / 跨范式的扩散反演特征漂移指纹**  
   证明特征漂移是**架构签名（architecture signature）**，而非采样器 artifact。在 UNet (SD 1.5 / SDXL) / Transformer (DiT) / MM-DiT (FLUX) 三种 backbone 以及 DDIM / Flow Matching 两种范式下统一量化。FLUX vs DiT Pearson r=0.727（同 backbone）> FLUX vs SD 1.5 r=0.486（不同 backbone）——**backbone attention 结构决定漂移模式，范式影响次要**。

2. **诊断驱动的零训练残差校正——简单性即优势**  
   诊断揭示注入位置不重要（random5≈top5）且 λ 不敏感（0.3-0.7 差 <0.08 dB）——**最简单的全局 latent 注入就是最优解**。四架构校正均显著有效（SD 1.5 +2.50 dB, SDXL +5.23 dB, DiT +4.67 dB, FLUX +3.94 dB）。在编辑 benchmark 上将内容保持提升 40%（LPIPS 0.86→0.51），远超简化版 P2P。与 P2P 统计等价（Cohen's d=0.033），内存低数百倍。Feature-level 校正反而无效（Δ=−0.27 dB），文本 token 残差零增益——刻意复杂化会降低性能。

> 这两个创新点构成”先发现规律，再利用规律设计方法”的完整主线。线性插值不是我们的发明（RLI 已独立发现），但**诊断→定位→极简干预**的范式以及”为什么线性插值有效”的理论解释是我们的核心贡献。

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

**DCSC（闭环控制器）**：已探索并放弃。实验验证闭环控制在当前系统上没有可测量的增益（三模式在对抗条件下 PSNR 等价），该负结果为"简单性即优势"的叙事提供了支撑。论文 Discussion 中诚实提及。

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
  - 跨架构漂移指纹各不相同：SD 1.5→decoder / SDXL→mid_block / DiT→blocks 11-21 / FLUX→early single + late single + last joint（双峰）

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

按特征类型（structural/textural/minimal）做选择性校正。1 层 minimal ≈ 5 层 full（PSNR 差 < 0.2 dB）。脚本：`scripts/phase3_prep.py`（含选择性校正逻辑），输出：`outputs/phase3_selective/`

### 3b：风格编辑 + CLIP 钉扎

CLIP 空间正交投影 + 闭环钉扎反馈。风格注入时保护内容结构。脚本：`scripts/phase3_common.py`（CLIPFeatureExtractor、run_* 函数），三类场景验证：人像/建筑/艺术字体。

### DCSC 负结果

探索了 Drift-Aware Closed-Loop Controller（P-控制律 + 可校正内容子空间 + 稳定性命题），在对抗扰动 σ ∈ {0.01, 0.05, 0.10, 0.20, 0.50} × 3 模式 (open_loop/phase3_pin/dcsc) 下三模式 PSNR 等价。闭环控制在当前系统上不提供可测量增益——系统本身已对 λ 鲁棒，自适应调节 λ 没有额外价值。**论文 Discussion 中诚实提及作为"简单性即优势"的佐证。**

---

## Phase 4：理论深化 + 跨架构验证 ✅

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
| DiT (Transformer) | bottom→top 过渡区 (blocks 11-21) | 无 ResNet/residual 概念 |
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
| DiT | Transformer (single-stream) | DDIM (v-pred) | blocks 11-21 | 单峰（中层） |
| **FLUX** | **MM-DiT (dual-stream)** | **Flow Match** | **early single + late single + last joint** | **双峰 + 尾端异常** |

**核心发现**：漂移指纹不由采样范式决定（DiT 和 FLUX 都是 Transformer 但指纹完全不同），而由具体的 attention 结构（single-stream vs dual-stream）决定。**特征漂移是架构签名，不是采样器 artifact。**

### 架构间漂移分布统计相似度

四架构漂移向量插值到统一长度（57），两两 Pearson r：

| 配对 | Pearson r | Spearman ρ | 解读 |
|------|-----------|------------|------|
| **FLUX vs DiT** | **0.727** | 0.720 | 同 Transformer backbone，不同范式→Backbone 主导 |
| FLUX vs SD 1.5 | 0.486 | 0.407 | 不同 backbone+范式→最低相似度 |
| FLUX vs SDXL | 0.646 | 0.544 | 中等相似 |
| SD 1.5 vs SDXL | 0.666 | 0.804 | 同 UNet family→高秩相关 |
| SDXL vs DiT | 0.790 | 0.647 | 最高 Pearson（但 drift 量级差 1000×） |

**核心结论**：漂移指纹不由采样范式（DDIM vs Flow Matching）决定，而由 backbone attention 结构（single-stream vs dual-stream MM-DiT）决定。**特征漂移是架构签名，不是采样器 artifact。**

输出：`outputs/phase6_unified/four_arch_fingerprint.png`、`arch_similarity_matrix.png`、`drift_profile_overlay.png`。

### Euler 反演限制分析

| 架构 | 范式 | Baseline PSNR | ΔPSNR | N |
|------|------|--------------|-------|---|
| SD 1.5 | DDIM (可逆) | 22.48 dB | +2.50 | 24 |
| SDXL | DDIM (可逆) | 22.11 dB | +5.23 | 19 |
| DiT | DDIM v-pred (可逆) | 15.24 dB | +4.67 | 3 |
| **FLUX** | **Flow Match Euler (不可逆)** | **12.03 dB** | **+3.94** | 19 |

Euler 反演代价：FLUX baseline 比 SD 1.5 DDIM 低 ~10.5 dB。校正回收率：+3.94 / 10.5 ≈ 38%。校正机制在不可逆反演下依然显著有效——范式无关。

### 技术笔记

- FLUX transformer 需 packed latent tokens（`_pack_latents`），不能直接传 VAE latent
- `_unpack_latents` 需传图像尺寸（img_h, img_w），不是 latent 尺寸
- 存全部 50 步 × 57 blocks 特征会耗尽 CPU RAM；只存 turnaround 点
- T5 offload 到 CPU 必要（48GB 显存不够）
- 模型来源：ModelScope（AI-ModelScope/FLUX.1-dev），~31GB

---

## Phase 7：编辑 Benchmark ✅

将残差校正作为**编辑流程的通用插件**，在 prompt-changed editing 上验证内容保持能力。

### 协议

1. BLIP 生成 coco_val 19 张图的 source caption
2. 程序化生成 28 个编辑对（word swap / attribute change / style transfer）
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

## 设计原则

1. **Diagnosis precedes intervention**（诊断先于干预）：Phase 1 的逐层漂移诊断先于 Phase 2 的校正
2. **Correction is geometry-aware**（校正利用几何结构）：因果消融 + 互信息 + 流形分析互补证明残差是有意义的信号
3. **Simplicity over complexity**（简单优于复杂）：1 层校正 ≈ 5 层效果。诊断告诉我们不需要复杂控制——skip connections 本身就是天然的鲁棒性保证
4. **Honesty over hype**（诚实优于包装）：DCSC 闭环控制的失败被诚实记录，收敛性推导明确标注假设条件和实证差距

---

## 论文叙事

What → Why → How 三章递进：

- **What（第 3 章 诊断）**：漂移不是噪声，有清晰的架构级结构
- **Why（第 4 章 理论）**：三视角互补解释漂移成因
- **How（第 5 章 校正）**：诊断驱动的最简校正 + SOTA 对比

叙事文件：`THESIS_NARRATIVE.md`

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
| `scripts/dit_phase1_diagnostics.py` | DiT Phase 1 诊断 |
| `scripts/dit_phase2_common.py` | DiT Phase 2 共享（v_prediction DDIM、3D token） |
| `scripts/dit_phase2_full.py` | DiT Phase 2 校正 |
| `scripts/flux_common.py` | FLUX Phase 6 共享：加载、Euler inversion、FeatureExtractor、FluxFeatureCorrector |
| `scripts/flux_phase6_diagnosis.py` | FLUX Phase 6a：57 block 漂移诊断 + latent 校正 |
| `scripts/flux_phase6c_analysis.py` | FLUX Phase 6c：19 图验证 + text drift + text injection + Euler 分析 |
| `scripts/phase6_unified_fingerprint.py` | 四架构统一漂移热力图 + 相似度矩阵 + 剖面叠图 |
| `scripts/phase7_editing_benchmark.py` | Phase 7：编辑 benchmark（BLIP caption + 4 条件编辑 + 指标评估） |

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
- **RLI (Residual Linear Interpolation)**：Jo et al., ICCVW 2025。在 self-attention 层做线性残差插值以稳定 editing。数学形式上与我们的校正公式等价，但存在本质差异：(1) RLI 经验驱动（凭直觉选 attention 层），我们诊断驱动（逐层量化后定位瓶颈）；(2) RLI 面向 editing artifacts 平滑，我们面向 inversion-reconstruction 不一致性的系统校正；(3) RLI 无理论分析，我们有三视角互补框架；(4) RLI 仅覆盖 UNet 架构（SD 1.4/2.0/2.1/SDXL），我们覆盖 DiT（Transformer）+ FLUX（MM-DiT + Flow Matching）。RLI 的经验发现支持了我们的核心论点——线性插值在特征校正中是有效的，但只有诊断才能定位最优注入位置并解释为什么。
- **LAMS-Edit**（最接近，开环混合）→ **Prompt-to-Prompt**（交叉注意力）→ **DiffStateGrad**（SVD 低秩）

**差异化定位**：诊断驱动 + 理论闭环 + 跨架构验证 + 内存优势 + 统计等价于 P2P。线性插值不是我们的发明——RLI 已独立发现类似形式——我们的贡献是**诊断→定位→极简干预**这个范式：一旦通过逐层诊断定位了架构瓶颈，最简线性校正即可达到复杂方法的同等效果。简单性是诊断的必然结果，而非调参的偶然发现。
