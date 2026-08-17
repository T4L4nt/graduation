# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## 项目概述

ICLR 2027 投稿：**Architecture Fingerprint of Feature Drift in Diffusion Inversion**。

**核心贡献**：发现扩散反演中的特征漂移具有由 backbone attention 拓扑决定的**分量级架构指纹**——拓扑分量（峰位）为跨扰动的不变量，度量分量（浓度/展宽/形状）呈条件依赖的弱变异。

**当前证据状态**（P-multi@104 canonical，v3.6 形态二分框架）：

| Claim | 状态 | 关键证据 |
|-------|------|---------|
| C1 分量级不变性（内层峰类） | ✅ | Band1 四变体 D_pp 全 0，D_mag ∈ [0.005, 0.023]；SDXL-Turbo D_pp=0/D_mag=0.0022（噪声底内） |
| C1 弱主张（斜坡类） | ✅ | 末层 argmax 稳定；形状参数实例级（PixArt-α vs Σ 0.096） |
| Finding 2 形态二分 | ✅ | 内层局域×3（SD1.5/SDXL/H-DiT）vs 末端累积×3（FLUX/PixArt/SD3.5），Spearman ≥ +0.987 |
| 可分辨性极限 | ✅ | FLUX–PixArt D_mag=0.0011 < 噪声底 p95 0.0042 |
| C3 双目标对照 | ⚠️ P-t0 时代数据，P-multi 审计进行中 | DiT-S/2 eps/flow 峰位同归 block 11（P-t0 值） |
| C4 实例可诊断 | ✅ | Cut A +2.2dB vs −11.6dB（干预，与协议无关） |
| 统计 | ✅ | TOST≤0.2dB；ρ(drift,Δ)=−0.395 为历史值，存盘为 −0.395（待统一） |

**形态二分（P-multi@104）**：

| 类 | 成员 | pp | conc/sp |
|----|------|------|---------|
| 内层局域型 | SD1.5 / SDXL / H-DiT | 0.684 / 0.429 / 0.500 | ~0.57–0.61 |
| 末端累积型 | FLUX / PixArt-Σ / SD3.5 | 0.983 / 0.964 / 0.958 | 斜坡，argmax 退化 |

**剩余待补**：Finding 2 英文成稿 · Fig.2 重画 · 随机初始化 P-multi 重跑（5 种子，运行中）· DiT-S/2 结晶 P-multi 审计（运行中，eps 仅存 10k/50k）· PERMANOVA · MI shuffle 基线（预注册缺口，见审查报告）· HunyuanDiT 组件拆分 · weight_perturb v1-v3 按 v2 度量重做（peak_count 已废除）· phase8b FLUX 消融表重跑（空列表 bug 已修）

## 开发环境

- conda 环境 `grad`（Python 3.10）：`conda activate grad`
- GPU：NVIDIA RTX PRO 6000 Blackwell (96GB), CUDA 13.0
- PyTorch 2.11.0+cu128, diffusers 0.38.0
- 离线运行：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- 运行脚本：`export PYTHONPATH=scripts:$PYTHONPATH`
- NCCL 修复：`LD_PRELOAD="$(python -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')/lib/libnccl.so.2"`
- HF 镜像：`HF_ENDPOINT=https://hf-mirror.com`

## 数据分集

| 分集 | 路径 | N | 用途 |
|------|------|---|------|
| coco_val | `data/coco_val/` | 19 | 诊断+评估 |
| coco_val100 | `data/coco_val100/` | 104 | 100-image 扩展 + DiT 训练 |
| basetest | `data/basetest/` | 8 | 历史测试 |

## 已缓存模型

| 模型 | 大小 | 用途 |
|------|------|------|
| SD 1.5 | 5.2G | baseline |
| SD 1.4 (CompVis) | 3.3G | C1 checkpoint |
| SDXL base | 14G | C2 跨架构 |
| HunyuanDiT | 14G | C2 跨架构 |
| CLIP ViT-L/14 | 1.6G | SD 1.5 文本编码器 |
| OpenCLIP ViT-H/14 | ~3G | #13 编码器 swap（1024-dim，与 SD1.5 不兼容） |
| LCM LoRA | ~300M | #11 边界模型 |
| DiT-S/2 eps ×5 | 3G | #10 结晶曲线 |
| DiT-S/2 flow ×5 | 3G | #10 结晶曲线 |

## v2 结构距离度量

三连续分量，无 peak_count：
- D_pp = |peak_position_a − peak_position_b|
- D_shape = 1 − Spearman ρ(profile_a, profile_b)
- D_mag = L2(concentration, spread)
- D_s = sqrt(D_pp² + D_mag²)

噪声底（B=100 bootstrap）：median=0.0071, p95=0.0163

## 关键脚本（P0b 含）

| 脚本 | 功能 |
|------|------|
| `scripts/p0b_cross_checkpoint_v4.py` | C1 v4 剂量曲线（v2 度量，per-image 聚合） |
| `scripts/p0b_sd14_fingerprint.py` | SD 1.4/RV 指纹诊断（--unet-path 通用） |
| `scripts/p0b_measure_delta_w.py` | 逐层 ΔW 测量 |
| `scripts/p0b_recalc_cross_arch_v2.py` | 跨架构 v2 矩阵重算 |
| `scripts/p0b_sampler_swap.py` | SD 1.5 多采样器对照（C3） |
| `scripts/p0b_lcm_lora_fingerprint.py` | LCM LoRA 边界模型指纹 |
| `scripts/p0b_rand_text_control.py` | 随机文本嵌入对照 |
| `scripts/p0b_dit_crystallization.py` | DiT 结晶曲线诊断 |
| `scripts/p0b_conflict_bakeoff.py` | conflict 度量烘焙赛（骨架） |
| `scripts/p0b_dit_decompose.py` | DiT 组件拆分（骨架） |
| `scripts/p0b_plot_c1_master.py` | C1 完整剂量曲线图 |
| `scripts/p0b_plot_boundary.py` | 边界谱系柱状图 |
| `scripts/p0b_plot_dose_response.py` | C1 早期剂量图 |
| `scripts/phase1_diagnostics.py` | Phase 1 SD 1.5 诊断 |
| `scripts/phase2_common.py` | Phase 2 共享工具 |
| `scripts/dit_controlled_shared.py` | DiT 训练+诊断共享 |
| `scripts/train_dit_epsilon.py` | DiT eps 训练 |
| `scripts/train_dit_flow.py` | DiT flow 训练 |

## 关键输出

| 路径 | 内容 |
|------|------|
| `outputs/p0b_cross_checkpoint/` | P0b 全部实验数据 |
| `outputs/train_controlled/crystallization/` | 结晶曲线 |
| `outputs/phase1/layer_drift_summary.json` | SD 1.5 逐层漂移 |
| `outputs/sdxl_phase1/layer_drift_summary.json` | SDXL 逐层漂移 |
| `outputs/dit_phase1/layer_drift_summary.json` | DiT 逐层漂移 |
| `outputs/phase9_flux_fp16/` | FLUX 逐层漂移 (fp16) |
| `outputs/sd35_phase1/` | SD 3.5 逐层漂移 |
| `outputs/phase7_editing_100image/` | 编辑 benchmark (121 pairs) |
| `outputs/phase7_skip_intervention/` | Skip 因果干预 |
| `outputs/phase5_final/` | 19-image SOTA 对比 |

## 论文文档

| 文件 | 内容 |
|------|------|
| `ICLR2027_OUTLINE.md` | 论文大纲 + 中文摘要 |
| `ICLR_PAPER_DEFINITIONS.md` | 严格定义与贡献层次 v3.4 |
| `projects/prereg_freezed.md` | 预注册冻结记录 |
| `projects/permanova_spec.md` | PERMANOVA 方差分解规范 |
| `projects/iclr_intro_c1c4.md` | C1-C4 Intro 段落 |
| `projects/ICLR2027_OUTLINE.md` | 大纲副本 |
