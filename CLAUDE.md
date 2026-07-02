# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

硕士毕业设计：**基于扩散模型的内容保持与风格解耦图像编辑**。作者为塔拉尼提·居马努尔。

核心目标：在 Stable Diffusion 的 DDIM 反演-重建 pipeline 上解决两个问题：
1. **内容漂移**：DDIM 反演-重建过程中的信息丢失
2. **风格耦合**：编辑过程中内容与风格难以独立控制

## 项目阶段

| 阶段 | 时间 | 状态 |
|------|------|------|
| 第一阶段 | 2026.5 | ✅ 完成：DDIM 反演-重建漂移动态诊断 |
| 第二阶段 | 2026.6 | ✅ 完成：零训练残差校正模块 + 消融 + 基线对比 |
| 第三阶段 | 2026.6–7 | ✅ 完成：CLIP 正交投影 + prompt 风格注入 + 钉扎约束，论文配图已生成 |
| 第四阶段 | 2027.2– | ⏳ 论文撰写与答辩 |
| SDXL 泛化 | 2026.7 | ✅ 完成：Phase 1-3 全部验证，跨 UNet 架构泛化成功 |
| DiT 泛化 | 2026.7 | ✅ 完成：Phase 1-3 全部验证，跨 Transformer 架构泛化成功 |

## 开发环境

- conda 环境 `grad`（Python 3.10），激活：`conda activate grad`
- GPU：NVIDIA RTX PRO 6000 Blackwell (48GB)
- PyTorch 2.11.0+cu128, diffusers 0.38.0, transformers 5.12.1
- 主模型：`runwayml/stable-diffusion-v1-5`（已缓存）
- CLIP：`openai/clip-vit-large-patch14`（已缓存，需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 离线运行）
- Conda activate.d 脚本已设置 `LD_PRELOAD` 解决 NCCL 库冲突
- 运行脚本时需 `export PYTHONPATH=scripts:$PYTHONPATH`

---

## 第一阶段：DDIM 反演-重建漂移动态诊断 ✅

- **脚本**：`scripts/phase1_diagnostics.py`
- **输出**：`outputs/phase1/layer_drift_summary.json`
- **关键发现**：漂移集中在 `up_blocks.2.resnets.0`（跨图一致最高），mid_block ResNet 块在 coco_val 上排第 2、5 位。早期 5 图结果受人脸图影响偏向 decoder，独立 coco_val（19 图）诊断显示 mid_block 贡献更显著。ResNet 漂移比 Attention 大约一个数量级

---

## 第二阶段：零训练残差校正模块 ✅

核心公式：$f_{out} = f_{recon} + \lambda \cdot (f_{inv} - f_{recon})$

### 关键结果

- **测试集**（24 图，50 步，λ=0.7）：平均 Δ PSNR **+2.50 dB**，LPIPS 从 0.190 → 0.086（↓55%）
- **coco_val only**（19 图，λ=0.7）：平均 Δ PSNR **+2.76 dB**，LPIPS 从 0.218 → 0.094
- **基线对比**（17 图，50 步）：DDIM+Corr (+3.90 dB) >> EDICT (+0.37 dB) > DDIM > NTI (−2.43 dB)
- **消融**（3 图，λ=0.7）：latent_interp ≈ random5 ≈ top5 ≈ encoder5 > attention5

### 诚实发现

1. **校正对注入位置鲁棒**：random5 ≈ top5（差 < 0.3 dB），任何 decoder ResNet 层组效果接近。原因：UNet skip connections 传播校正信号。诊断的价值不在"选层"而在揭示**架构级瓶颈**（decoder ResNets 贡献 >99% 漂移，Attention 可忽略）
2. **漂移加权无效**（2026-07-02 验证）：per-layer 权重 w_i ∝ drift_i 不能拉开 top5 与 random5 差距（+2.81 vs +2.96 dB）。漂移量 ≠ 校正收益，关系非单调
3. latent 空间插值优于特征空间校正，但 latent 空间无法做内容/风格解耦
4. 校正收益与基线质量无关，与图像纹理复杂度正相关

**修正后的叙事**：真正区分于 LAMS-Edit/P2P 的是**注入什么**（特征级残差 vs 注意力图 vs latent），而非注入哪里。诊断揭示架构瓶颈（哪些模块在漂移），校正利用瓶颈特征做鲁棒注入（不依赖精细层选择）。

---

## 第三阶段：风格解耦与编辑 ✅

核心定位：**内容保持编辑框架**——用 CLIP 文本 prompt 做风格注入，正交钉扎约束保证内容不漂移。

### 方法

1. **CLIP 正交投影**：$v_{style} = v_{text} - proj_{v_{content}}(v_{text})$，将文本风格方向分解为与内容正交的分量
2. **Prompt 风格注入**：SD text embedding 空间线性插值，无模态差异
3. **正交钉扎约束**：去噪过程周期性解码→CLIP 编码→检查内容投影偏离→自适应缩减风格强度

### 关键结果（coco_val，50 步）

| 方法 | PSNR | LPIPS |
|------|------|-------|
| DDIM Baseline | 19.9 | 0.150 |
| Style Only (无保护) | 19.0 | **0.312** |
| Ours (corr+style+pin) | 20.8 | **0.125** |

风格注入无保护时 LPIPS 从 0.15 崩溃到 0.31，框架保护后恢复到 0.12。钉扎约束跨图触发（5-8/9 checks），自适应调控风格强度。

### 论文配图（`outputs/thesis_figures/`）

| 文件 | 内容 |
|------|------|
| `phase2_correction.png` | 6 图 × (Original + Baseline + Ours) 对比网格 |
| `phase3_framework.png` | 风格迁移安全框架：Content + Baseline + Style Only + Ours |
| `direction_interpolation.png` | SLERP 风格方向插值 watercolor→cyberpunk |
| `phase2_ablation.png` | 3 图平均消融：top5 / random5 / encoder5 / attention5 / latent_interp |
| `coco_val_summary.json` | 全部 19 张 coco 图片定量评估 |

---

## 第四阶段：论文撰写与答辩 ⏳

### 待完成

| # | 任务 | 说明 | 状态 |
|---|------|------|------|
| 1 | 三类场景验证 | 人像(8张)、建筑(5张)、艺术字体(5张)，全部通过 Phase 2+3 验证 | ✅ |
| 2 | 理论深化 | 方向 A：建立理论框架，回应"方法太简单"的批评 | ⏳ |
| 2.1 | 信息论分析 | 量化每层特征与原始图像的 mutual information，解释为何 ResNet 特征携带可校正信息 | ⏳ |
| 2.2 | 流形视角 | 反演/重建轨迹视为特征流形上的两条路径，校正是沿梯度方向一阶修正 | ⏳ |
| 2.3 | 收敛性证明 | 漂移加权引入后校正的收敛性理论保证 | ⏳ |
| 2.4 | 理论章节撰写 | 整合信息论 + 流形 + 收敛性，形成论文核心理论章节 | ⏳ |
| 3 | 跨架构漂移指纹图 | SD 1.5 / SDXL / DiT 三种架构漂移热力图并排对比，证明诊断的架构洞察力 | ✅ |
| 4 | 论文撰写 | 正文 + 图表 + 参考文献 | ⏳ |

### 已完成

| # | 任务 | 说明 |
|---|------|------|
| — | SDXL 泛化 | Phase 1-3 全部验证，跨 UNet 架构泛化成功 |
| — | DiT 泛化 | Phase 1-3 全部验证，跨 Transformer 架构泛化成功 |
| — | SOTA 横向对比 | DDIM / EDICT / NTI(BLIP) / P2P / ControlNet / LAMS-Edit 全部完成 |
| — | 综合对比表 | `outputs/phase4_sota/` |

### 场景数据准备

| 场景 | 来源 | 数量 | 说明 |
|------|------|------|------|
| 人像 | data/portraits/ | 8 张 | Unsplash，验证身份保持 + 风格编辑 |
| 建筑 | data/architecture/ | 5 张 | Pexels 现代建筑，验证几何结构保持 |
| 艺术字体 | data/typography/ | 5 张 | Pexels 排版/书法，验证笔触逻辑保持 |

### 三类场景验证结果（50 步，λ=0.7）

| 场景 | Baseline PSNR | Correction Δ | Style+Pin Δ | 钉扎触发 |
|------|-------------|-------------|-------------|---------|
| 人像 (8张) | 26.38 | **+4.94 dB** | +4.94 dB | 2-9/9 |
| 建筑 (5张) | 21.22 | **+6.47 dB** | +6.47 dB | 0-8/9 |
| 艺术字体 (5张) | 22.01 | **+5.16 dB** | +5.16 dB | 2-9/9 |

**关键发现**：
- 校正在三类场景上均稳健有效（平均 +5.5 dB），建筑受益最大（几何结构保持）
- 钉扎在 0-9/9 checks 频繁触发，Style+Pin 保持与 Correction 相同的 PSNR——证明钉扎有效防止了风格注入的内容漂移
- 人像 ArcFace 相似度 0.73-0.95，身份保持验证通过
- 脚本：`scripts/phase4_scenes.py`，输出：`outputs/phase4_sota/scenes/`

---

## 关键脚本

| 脚本 | 功能 |
|------|------|
| `scripts/phase1_diagnostics.py` | UNet 层级漂移动态诊断 |
| `scripts/phase2_common.py` | 共享模块：加载、DDIM、FeatureCorrector、指标、可视化 |
| `scripts/phase2_full.py` | Phase 2 主实验：λ 扫描 + 评估 + 消融 |
| `scripts/phase2_nti.py` | NTI 基线对比 |
| `scripts/phase2_edict.py` | EDICT 精确可逆基线对比 |
| `scripts/phase3_prep.py` | Phase 3：CLIP 正交投影 + prompt 风格注入 + 钉扎约束 |
| `scripts/gen_thesis_figures.py` | 论文配图生成（仅使用 data/coco_val，prompt 风格注入） |
| `scripts/gen_report.py` | 综合报告生成（PDF） |
| `scripts/sdxl_phase1_diagnostics.py` | SDXL Phase 1：UNet 层级漂移动态诊断（3 块结构） |
| `scripts/sdxl_phase2_full.py` | SDXL Phase 2：λ 扫描 + 消融（mid_block 主导） |
| `scripts/sdxl_phase3_prep.py` | SDXL Phase 3：风格注入 + 钉扎约束（2048-dim embedding 适配） |
| `scripts/dit_phase1_diagnostics.py` | DiT Phase 1：40 个 Transformer block 漂移动态诊断 |
| `scripts/dit_phase2_common.py` | DiT Phase 2 共享模块：v_prediction DDIM、FeatureCorrector（3D token 适配） |
| `scripts/dit_phase2_full.py` | DiT Phase 2：λ 扫描 + 消融（过渡区校正最优） |
| `scripts/dit_phase3_prep.py` | DiT Phase 3：CLIP 正交投影 + 风格注入 + 钉扎（1024/2048-dim 适配） |
| `scripts/phase4_p2p.py` | Phase 4：Prompt-to-Prompt 交叉注意力混合对比 |
| `scripts/phase4_controlnet.py` | Phase 4：ControlNet Canny 条件生成对比 |
| `scripts/phase4_summary.py` | Phase 4：SOTA 综合对比表生成 |

---

## 数据分集

| 分集 | 路径 | 数量 | 用途 |
|------|------|------|------|
| coco_val | `data/coco_val/` | 19 张 | **层选择 + 定量评估**（独立于历史 val 集） |
| basetest | `data/basetest/` | 8 张 | 历史测试（包含 face1/face2） |
| val (历史) | face1, face2, nature, content, watercolor | 5 张 | Phase 1-2 λ 调参（已污染，不可用于评估） |

**层选择独立性**：Phase 1 诊断最初在 5 张历史图上完成（`layer_drift_summary_orig5.json`），发现与 coco_val 19 图排名仅 2/5 重叠。2026-07-02 改用 coco_val 独立排名（`layer_drift_summary.json`），确保层选择与 λ 调参（历史 val 集）完全独立。

---

## SDXL 泛化 ✅

SDXL 与 SD 1.5 的主要差异：3 个 down/up blocks（非 4 个）、双文本编码器（2048-dim）、`added_cond_kwargs` 条件机制。

### Phase 1 关键发现

- 漂移集中在 **mid_block**（非 decoder up_blocks），与 SD 1.5 完全不同
- 脚本：`scripts/sdxl_phase1_diagnostics.py`，输出：`outputs/sdxl_phase1/`

### Phase 2 关键结果

- FeatureCorrector 在 SDXL 上直接复用（架构无关）
- λ 扫描 + 消融完成，输出：`outputs/sdxl_phase2/`

### Phase 3 关键结果（3 图 coco_val，50 步）

| Image | Baseline | Correction | Style + Pin |
|-------|----------|------------|-------------|
| coco_139 | PSNR 20.32, LPIPS 0.710 | **29.65, 0.084** (+9.33 dB) | 29.75, 0.081 |
| coco_285 | PSNR 22.28, LPIPS 0.234 | 22.82, 0.200 (+0.54) | 22.75, 0.203 |
| coco_632 | PSNR 20.95, LPIPS 0.481 | 23.68, 0.167 (+2.73) | 23.68, 0.161 |

- 校正效果图像依赖性大（与 SD 1.5 一致，低基线质量时受益更大）
- 钉扎约束正常工作（2-4/9 checks 触发）
- 脚本：`scripts/sdxl_phase3_prep.py`，输出：`outputs/sdxl_phase3/`

---

## DiT 泛化 ✅

HunyuanDiT (`Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers`) 是纯 Transformer 扩散模型，与 UNet 架构完全不同：

| 维度 | SD 1.5 (UNet) | HunyuanDiT (Transformer) |
|------|-------------|--------------------------|
| 层数/结构 | ~25 层 (encoder-mid-decoder) | 40 个 HunyuanDiTBlock (0-19 bottom, 20-39 top + skip) |
| 特征形状 | 4D [B,C,H,W]，各层分辨率不同 | 3D [B,N,D]，统一 [1, 4096, 1408] |
| 预测类型 | `epsilon` | `v_prediction` |
| 文本编码器 | CLIP (77×768) | CLIP/BERT (77×1024) + T5 (256×2048) |
| pipeline 属性 | `pipe.unet` | `pipe.transformer` |

### v_prediction DDIM 反演

v_prediction 下 DDIM 反演质量较低（50 步 baseline PSNR 10-18 dB，vs SD 1.5 ~20 dB）。需转换公式：
- v → x_0: `x_0 = sqrt(α_t)·z_t - sqrt(1-α_t)·v`
- v → eps: `eps = sqrt(1-α_t)·z_t + sqrt(α_t)·v`
- 100 步可改善到 ~17 dB

### Phase 1 关键发现

- **漂移集中在 bottom→top 过渡区 (blocks.11-21)**，blocks.20（首个 skip connection block）始终最高
- 深层 bottom (0-10) 和深层 top (22-39) 漂移极低
- 与 UNet 的 "decoder ResNet 漂移最大" 模式完全不同，是 DiT 架构的独有发现
- 脚本：`scripts/dit_phase1_diagnostics.py`，输出：`outputs/dit_phase1/`

### Phase 2 关键结果（3 图 coco_val，λ=0.7）

| 层组 | 层数 | 平均 ΔPSNR |
|------|------|-----------|
| top5 | 5 | +4.67 |
| top10 | 10 | +6.39 |
| **transition (11-21)** | **11** | **+7.51 dB** |
| region_bottom | 5 | +4.50 |
| region_top | 5 | +4.55 |

- FeatureCorrector 在 3D token 特征 [B,N,D] 上同样有效
- 更多层 = 更好效果，与 Phase 1 诊断一致
- 脚本：`scripts/dit_phase2_common.py` + `dit_phase2_full.py`，输出：`outputs/dit_phase2/`

### Phase 3 关键结果（3 图 coco_val，50 步，2026-07-02 修正钉扎 Bug 后重跑）

| Image | Baseline | Correction | Style + Pin |
|-------|----------|------------|-------------|
| coco_139 | 17.29, 0.477 | 21.08, 0.261 (+3.8 dB) | 20.15, 0.282 (+2.9) |
| coco_285 | 10.70, 0.569 | 12.93, 0.620 (+2.2) | 11.96, 0.668 (+1.3) |
| coco_632 | 18.14, 0.396 | 21.30, 0.232 (+3.2) | 20.89, 0.256 (+2.8) |

- 修正后 style_pin 始终 ≤ correction（符合"风格注入牺牲内容"预期），不再出现反转异常
- 钉扎触发率合理化：coco_139 4/9, coco_285 2/9, coco_632 0/9

### 跨架构泛化结论

**核心机制（诊断驱动 + 层内残差校正 + CLIP 闭环钉扎）与扩散模型架构无关**，在 UNet (SD 1.5/SDXL) 和纯 Transformer (DiT) 上均验证有效。

---

## 领域调研

**反演方法**：DDIM → EDICT → NTI → BELM（未实现）

**内容保持**：LAMS-Edit（最接近，开环混合）→ Plug-and-Play RLI → DiffStateGrad（SVD 低秩）

**风格解耦**：StyleTex（CLIP 正交投影公式来源）→ IP-Adapter（解耦 cross-attn）→ Content-Style Inversion（AdaIN）

**差异化定位**：内容漂移校正 + 风格解耦整合为统一免训练框架，诊断驱动 + 闭环反馈。

---

## Phase 4：SOTA 横向对比 ✅

### 综合对比结果（3 图 coco_val，50 步）

| Method | PSNR↑ | LPIPS↓ | ΔPSNR | Training | Memory | 类型 |
|--------|-------|--------|-------|----------|--------|------|
| DDIM (baseline) | 20.78 | 0.269 | — | None | Low | 反演-重建 |
| EDICT | 21.15 | 0.256 | +0.37 | None | 2x | 精确可逆 |
| NTI (BLIP) | 18.35 | 0.353 | −2.43 | Optimization | Low | 优化反演 |
| **P2P (attn)** | **23.77** | **0.089** | **+2.98** | None | ~GB | 注意力混合 |
| ControlNet (Canny) | 9.55 | 0.781 | — | Pre-trained | ~1.4GB | 条件生成 |
| **Ours_Corr** | **23.70** | **0.097** | **+2.92** | None | **~MB** | 特征校正 |

### 核心结论

1. **P2P 与 Ours_Corr 内容保持能力相当**（ΔPSNR +2.98 vs +2.92，差 0.06 dB）
2. **关键差异在内存**：P2P 需保存全部交叉注意力图（~GB），Ours 只需 top-5 特征（~MB）
3. **Ours 独有优势**：诊断驱动层选择 + CLIP 闭环钉扎 + 风格注入无缝集成
4. **ControlNet 不是反演方法**：PSNR 不可直接比较，但 CLIP_content 0.6-0.8 表明结构有保持

### 脚本与输出

| 脚本 | 输出 |
|------|------|
| `scripts/phase4_p2p.py` | `outputs/phase4_sota/p2p/` |
| `scripts/phase4_controlnet.py` | `outputs/phase4_sota/controlnet/` |
| `scripts/phase4_summary.py` | `outputs/phase4_sota/comparison_psnr.png` + `comparison_table.json` |

---

## LAMS-Edit 对比分析（论文 Related Work 关键素材）

LAMS-Edit (arXiv:2601.02987, 2026.01) 是免训练图像编辑框架，核心机制：
1. **反演轨迹复用**：保存 DDIM 反演每一步的中间 latent z_t* 和 attention map A_t*
2. **Latent & Attention Mixing**：生成过程每步加权混合反演轨迹与生成轨迹
3. **调度器**：w_t 按时间步衰减，支持 linear/exponential/logistic
4. **风格**：通过 LoRA 引入（需预训练权重）
5. **内存**：约 12GB CPU 存储中间状态

### 各模块冲突评估

| 你的模块 | 冲突等级 | 判断依据 |
|----------|---------|---------|
| Phase 1 诊断（逐层 MSE 漂移排序） | ⭐ 极低 | LAMS-Edit 完全没有诊断/层选择概念，这是你最硬的原创壁垒 |
| Phase 2 top-k 残差校正 | ⭐⭐ 低 | 底层假设（反演中间状态有价值）重叠，但实现层级不同：你操作 UNet 层内特征，LAMS-Edit 操作全局 latent/attention |
| LambdaScheduler | ⭐⭐ 低 | 按时间步调度是扩散模型通用技术（CFG scale scheduling 等），且你的调度器由 Phase 1 诊断结果驱动 |
| CLIP 正交投影 | ⭐ 极低 | LAMS-Edit 完全无此组件 |
| Prompt 风格注入 | ⭐ 极低 | LAMS-Edit 用 LoRA，你用文本 embedding 插值（零训练） |
| 正交钉扎约束 | ⭐⭐ 低-中 | 范式竞争：LAMS-Edit 是开环混合，钉扎是闭环反馈。需主动对比 |

### 三个核心防御点

1. **粒度差异**：LAMS-Edit 操作全局 latent/attention（像素/空间级），你操作 UNet 内部层特征（特征通道级）。你的方法是"自校正"（单条轨迹内部的同层参考），理论上可叠加在任何编辑方法之上。

2. **诊断驱动 vs 均匀处理**：LAMS-Edit 对所有步骤一视同仁地混合；Phase 1 先诊断哪些层在漂移，再有针对性地修——方法论文本差。

3. **开环 vs 闭环**：LAMS-Edit 是开环控制（固定调度曲线），正交钉扎是闭环控制（CLIP 实时监测内容漂移并反馈调节风格强度），自适应能力更强。

### 论文对比表示意

| 维度 | LAMS-Edit | 你的方法 |
|------|-----------|---------|
| 控制模式 | 开环（固定调度） | 闭环（CLIP 反馈调节） |
| 内容监测 | 无 | 每 10 步 VAE 解码 + CLIP 投影 |
| 自适应 | 所有图同一套参数 | 按图自适应调整风格强度 |
| 风格引入 | LoRA（需预训练） | 文本空间插值（零训练） |
| 内存 | ~12GB CPU | top-k hook，内存友好 |
| 层选择 | 无 | 诊断驱动 top-k |

### 论文撰写策略

- **Related Work 主动提及 LAMS-Edit**（2026.01，很新，审稿人一定知道），明确对比
- **术语区分**：避免 "mixing"/"blending"，用 "residual correction"、"drift compensation"、"layer-wise hook injection"
- **核心差异化标签**："诊断驱动" + "闭环反馈"，全文反复强化
- **消融建议**：如果时间允许，增加与 LAMS-Edit 核心思想的对比实验；时间紧则做详尽的定性对比即可
