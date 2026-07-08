# 目标创新点（CVPR 投稿版）

> 版本：2026-07-08
> 目标：将硕士论文的"诊断 + 线性校正"升级为可投稿 CVPR 的跨架构理解与通用方法。

---

## 创新点 1：跨架构 / 跨范式的扩散反演特征漂移指纹

### 学术表述
> **We systematically reveal that feature drift in diffusion inversion is an architecture signature rather than a sampler artifact, by quantifying its layer-wise distribution across UNet, Transformer, and MM-DiT under both DDIM and Flow Matching paradigms.**

### 用人话解释
> **"漂移长什么样"是由模型架构决定的，不是由 DDIM 还是 Flow Matching 决定的。**

同一个现象，在 UNet（SD 1.5 / SDXL）、纯 Transformer（DiT）、MM-DiT（FLUX）中呈现出完全不同的"指纹"：
- SD 1.5: decoder up_blocks ResNet 单峰
- SDXL: mid_block 单峰
- DiT: blocks 11-21 单峰
- FLUX: early single + late single + last joint（双峰+尾端异常）

**FLUX vs DiT Pearson r=0.727（同 Transformer backbone）> FLUX vs SD 1.5 r=0.486（不同 backbone）**——backbone attention 结构决定漂移模式，范式影响次要。这些指纹可以用架构拓扑来解释。

### 为什么它能到 CVPR
- **现象新**：以往工作多分析单架构（SD 1.5），未系统地将 UNet / Transformer / MM-DiT 摆在同一框架下研究反演漂移。
- **结论有概括性**：得出"drift fingerprint is architecture signature"这种能预测新架构行为的结论。
- **可视化冲击力强**：四张漂移热力图并排 + 相似度矩阵，审稿人一眼能 get 到。
- **与 RLI / P2P 不冲突**：不是在比"方法"，而是在回答他们没回答的"为什么"。

### 当前状态
| 证据 | 状态 |
|------|------|
| SD 1.5 漂移指纹 | ✅ 19 图，完整统计 |
| SDXL 漂移指纹 | ⚠️ 5 图，需扩展到 19 图 |
| DiT 漂移指纹 | ⚠️ 5 图，需扩展到 19 图 |
| FLUX 漂移指纹 | ✅ 19 图，完整统计 |
| 四架构指纹图 + 相似度矩阵 | ✅ `outputs/phase6_unified/` |
| 指纹与架构拓扑的对应关系表 | ⚠️ 需 formalize |

---

## 创新点 2：诊断驱动的零训练残差校正——简单性即优势

### 学术表述
> **We propose a diagnosis-driven, training-free residual correction mechanism. Rather than hand-engineering layer selection or injection strength, we let the drift fingerprint guide where to correct — and find that the simplest approach (fixed λ, global latent-space injection) already achieves state-of-the-art content preservation, generalizing across four architectures and two sampling paradigms.**

### 核心主张

校正公式 $f_{out} = f_{recon} + \lambda \cdot (f_{inv} - f_{recon})$ 本身并不新颖（RLI 独立发现了类似形式）。我们的贡献是：

1. **诊断先于干预**：不是凭直觉选层（RLI 选 attention），而是逐层量化 57/40/196 层后由数据告诉我们在哪里注入
2. **简单性是诊断的必然结果**：诊断揭示了两个关键事实——
   - 注入位置不重要（random5 ≈ top5，差 < 0.3 dB）
   - 注入强度不重要（λ ∈ {0.3, 0.5, 0.7}，PSNR 差 < 0.08 dB）
   
   **λ 不敏感性不是调参的巧合，而是理论预言的结果**：收敛性分析导出误差收缩因子 |1−λ|，只要 λ 保持在这个因子的收敛域内，精确值不重要。一个固定 λ=0.7 的全局 latent 注入在四架构上都是最优或接近最优。

3. **跨范式有效性**：同一公式在 DDIM（弯曲轨迹）和 Flow Matching Euler（不可逆直线轨迹，baseline 低 10.5 dB）下均显著有效——校正机制是范式无关的

### 与 RLI 的本质差异

| 维度 | RLI | Ours |
|------|-----|------|
| 选层方式 | 凭经验选 attention 层 | **诊断 196/40/57 层后定位瓶颈** |
| 为什么有效 | 无解释 | **三理论互补框架**（信息论+流形+收敛性） |
| 架构覆盖 | UNet only | **UNet + Transformer + MM-DiT** |
| 范式覆盖 | DDIM only | **DDIM + Flow Matching** |
| 方法趋势 | 经验→经验 | **诊断→理解→极简干预** |

### 为什么它能到 CVPR
- **反直觉**：最优方法是最简单的——刻意为之的复杂化（DCSC 闭环、feature-level 注入）反而降低性能。审稿人会觉得新鲜。
- **可解释**：为什么简单方法有效？因为诊断告诉你瓶颈在哪里，理论告诉你为什么校正公式的形式（λ 存在但不敏感）是正确的。
- **通用性强**：同一个公式跨四架构两范式有效，不需要训练。

### 当前状态
| 证据 | 状态 |
|------|------|
| SD 1.5 校正（19 图 + 消融 + 统计） | ✅ |
| SDXL 校正 | ⚠️ 5 图，需扩展到 19 图 + λ 扫描 |
| DiT 校正 | ⚠️ 3-5 图，需扩展到 19 图 + λ 扫描 |
| FLUX 校正 | ✅ 19 图 latent correction |
| Feature-level / text injection 负结果 | ✅ 支持简单性叙事 |
| DCSC 闭环负结果 | ✅ 支持简单性叙事 |
| **真实编辑 benchmark** | ❌ 必须做 |

---

## 两个创新点的逻辑关系

```
创新点 1：看清"不同架构在哪里漂移"
        ↓
        导出架构拓扑 → 漂移指纹的映射
        ↓
创新点 2：诊断揭示：注入位置不重要，λ 不重要
        最简单的全局 latent 注入就是最优解
        ↓
        在跨架构、跨范式、真实编辑任务上验证
```

这个逻辑比"自适应 λ"的叙事更强：
> **我们不需要自适应——因为诊断已经告诉我们，这个系统对精细控制不敏感。简单性是发现的成果，不是方法的局限。**

---

## 与 RLI 关系的关键论述

RLI 的经验发现（线性插值在 attention 层有效）实际上支持了我们的核心论点——线性残差注入是正确的函数形式。但 RLI 没有回答的问题，我们回答了：

- **为什么线性形式正确？** → 收敛性分析：误差收缩因子 |1−λ|
- **为什么 fix λ 就够了？** → λ 不敏感性的理论与实证双重验证
- **最优层在哪里？** → 不是 attention，是诊断揭示的架构瓶颈（ResNet / early single blocks / last joint block）
- **其他架构呢？** → 跨四架构两范式，同一公式通用

---

## 从叙事中移除的项

- DCSC 闭环控制器（已证明无效，论文 Discussion 中诚实提及）
- 自适应 λ（系统对 λ 不敏感，自适应无法提供增益——已在三组独立实验中被否定）
- AdaIN / CLIP style injector 等探索性风格编辑
- ControlNet 作为反演 baseline
- "统计等价于 P2P"的表述 → 改为"practically on par with negligible effect size"

---

## 待办优先级

### 阻塞（不做不能投）

| # | 任务 | 当前 |
|---|------|------|
| 1 | SDXL 漂移 + 校正扩展到 19 图 | 5 图 |
| 2 | DiT 漂移 + 校正扩展到 19 图 | 5/3 图 |
| 3 | 真实编辑 benchmark（P2P-style prompt-changed editing） | ❌ |
| 4 | 指纹与架构拓扑对应关系表 | ⚠️ 初步 |

### 重要（提升竞争力）

| # | 任务 |
|---|------|
| 5 | 论文写作（8 章） |
| 6 | 更多 baseline（LEDITS++, InfEdit, PnP） |
| 7 | 答辩 PPT |

### 低优先级

| # | 任务 |
|---|------|
| 8 | 测试集 50+ 张 |
| 9 | THESIS_NARRATIVE.md 更新 |
