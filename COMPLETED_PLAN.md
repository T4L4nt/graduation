# 已完成工作计划回顾

> 此文件记录项目全部已完成内容，用于核对完成度、发现遗漏项。

---

## Phase 1：DDIM 反演-重建漂移动态诊断 ✅

| # | 任务 | 状态 | 产出 |
|---|------|------|------|
| 1.1 | 实现 UNet 196 层逐层特征提取 | ✅ | `scripts/phase1_diagnostics.py` |
| 1.2 | 19 张 coco_val 逐层漂移量化 | ✅ | `outputs/phase1/layer_drift_summary.json` |
| 1.3 | 漂移分布分析（跨层 1000× 差距） | ✅ | ResNet >> Attention, decoder 集中 |
| 1.4 | Top-10 漂移层排名 | ✅ | `up_blocks.2.resnets.0` 最高 |
| 1.5 | ResNet vs Attention 对比分析 | ✅ | ~5× 差距 |

**关键发现**：漂移不是均匀噪声，有清晰的架构级结构。层间漂移跨 1000×，ResNet 漂移约 5× Attention，集中在 decoder up_blocks。

---

## Phase 2：零训练残差校正模块 ✅

| # | 任务 | 状态 | 产出 |
|---|------|------|------|
| 2.1 | FeatureCorrector 模块实现 | ✅ | `scripts/phase2_common.py` |
| 2.2 | λ 扫描实验（0.1–1.0） | ✅ | `scripts/phase2_full.py` |
| 2.3 | 19 图 coco_val 评估（PSNR/LPIPS/SSIM） | ✅ | `outputs/phase2_full/` |
| 2.4 | 消融实验：注入位置（top5/random5/encoder5/attention5/latent_interp） | ✅ | random5 ≈ top5 (差 < 0.3 dB) |
| 2.5 | 漂移加权消融 | ✅ | 加权无效 (r ≈ −0.11) |
| 2.6 | λ 稳定性验证 | ✅ | λ ∈ {0.3,0.5,0.7} PSNR 差 < 0.08 dB |
| 2.7 | NTI 基线 | ✅ | `scripts/phase2_nti.py`, `outputs/phase2_nti/` |
| 2.8 | EDICT 基线 | ✅ | `scripts/phase2_edict.py`, `outputs/phase2_edict/` |

**关键发现**：校正公式 $f + \lambda \cdot (f_{inv} - f_{recon})$ 在 19 图上平均 ΔPSNR +2.75 dB。注入位置鲁棒（random5 ≈ top5）。

---

## Phase 3：选择性校正 + 风格编辑 ✅

| # | 任务 | 状态 | 产出 |
|---|------|------|------|
| 3.1 | 选择性校正（structural/textural/minimal） | ✅ | `scripts/phase3_prep.py` |
| 3.2 | CLIP 特征提取器 | ✅ | `scripts/clip_utils.py` |
| 3.3 | 风格编辑 + CLIP 正交投影钉扎 | ✅ | `scripts/phase3_prep.py` |
| 3.4 | 三类场景验证（人像/建筑/艺术字体） | ✅ | `outputs/phase3_selective/` |
| 3.5 | DCSC 闭环控制器探索 | ✅（负结果） | 三模式 PSNR 等价，闭环无增益 |

**关键发现**：1 层 minimal ≈ 5 层 full。DCSC 负结果支撑"简单性即优势"叙事。

---

## Phase 4：理论深化 + 跨架构验证 ✅

| # | 任务 | 状态 | 产出 |
|---|------|------|------|
| 4.1 | 因果消融实验 | ✅ | `scripts/phase4_info_theory.py`, `outputs/phase4_info_theory/` |
| 4.2 | 互信息估计（KSG + Gaussian MI） | ✅ | `scripts/phase4_mi_estimation.py`, `outputs/phase4_mi/` |
| 4.3 | 特征流形分析与切空间对齐 | ✅ | `scripts/phase4_manifold.py`, `outputs/phase4_manifold/` |
| 4.4 | 收敛性验证（真实 UNet 特征） | ✅ | `scripts/phase4_convergence_verify.py`, `outputs/phase4_convergence/` |
| 4.5 | 跨架构漂移指纹（SD 1.5 / SDXL / HunyuanDiT） | ✅ | `scripts/phase4_fingerprint.py` |
| 4.6 | P2P 交叉注意力对比 | ✅ | `scripts/phase4_p2p.py` |
| 4.7 | ControlNet Canny 对比 | ✅ | `scripts/phase4_controlnet.py` |
| 4.8 | 三类场景汇总 | ✅ | `scripts/phase4_scenes.py` |
| 4.9 | SOTA 综合对比表 | ✅ | `scripts/phase4_summary.py` |

**关键发现**：三理论互补——信息论解释"丢失了多少信息"，流形解释"残差在什么结构上"，收敛性解释"信号如何传播"。

---

## Phase 5：统计验证 + 缺口补齐 ✅

| # | 任务 | 状态 | 产出 |
|---|------|------|------|
| 5.1 | 19 图 SOTA 对比表（含误差棒） | ✅ | `scripts/phase5_final_comparison.py`, `outputs/phase5_final/` |
| 5.2 | P2P vs Ours 统计检验（t-test, Cohen's d, Pearson r） | ✅ | t=3.737, p=0.0015, d=0.033 |
| 5.3 | Ours vs DDIM 统计检验 | ✅ | p=8.8e-6, d=0.791 |
| 5.4 | 步数鲁棒性（倒 U 曲线：4/10/20/50/100 步） | ✅ | 峰值 20 步 Δ=+4.65 dB |
| 5.5 | 失败案例分析（6/19 图 Δ < 1.0 dB） | ✅ | `scripts/phase5_failure_lambda.py` |
| 5.6 | λ 稳定性系统验证 | ✅ | λ ∈ {0.1–0.9} PSNR 稳定 |
| 5.7 | 论文配图生成 | ✅ | `scripts/gen_thesis_figures.py`, `outputs/thesis_figures/` |
| 5.8 | NTI/EDICT 缺失数据补齐 | ✅ | `scripts/fill_missing_edict_nti.py` |
| 5.9 | 统一消融表 | ✅ | `scripts/gen_unified_ablation_table.py` |

---

## 跨架构验证 ✅

| # | 架构 | 诊断 | 校正 | 产出 |
|---|------|------|------|------|
| 6.1 | SD 1.5 (UNet) | ✅ | ✅ | decoder up_blocks 集中, Δ=+2.50 dB |
| 6.2 | SDXL (UNet) | ✅ `sdxl_phase1_diagnostics.py` | ✅ `sdxl_phase2_full.py` | mid_block 集中, Δ=+5.37 dB |
| 6.3 | HunyuanDiT (Transformer) | ✅ `dit_phase1_diagnostics.py` | ✅ `dit_phase2_full.py` | blocks 11-21 集中, Δ=+5.65 dB |
| 6.4 | FLUX (MM-DiT) | ✅ `flux_phase6_diagnosis.py` | ✅ `flux_phase6c_analysis.py` | dual-peak, Δ=+3.94 dB |

### Phase 6：FLUX Flow Matching ✅

| 子任务 | 状态 | 产出 |
|--------|------|------|
| 6.4a 57-block 漂移诊断 | ✅ | `outputs/phase6_flux/diagnosis_summary.json` |
| 6.4b 19 图 latent correction | ✅ | ΔPSNR=+3.94, p=6.4e-11, d=3.12 |
| 6.4c Text token 漂移分析 | ✅ | mean=0.144, 3×低于 image drift |
| 6.4d Text token injection 消融 | ✅ | 负结果: feature-level Δ=−0.27 dB, text-only Δ=−0.09 |
| 6.4e 四架构统一指纹图 | ✅ | `outputs/phase6_unified/` |
| 6.4f Euler 反演限制分析 | ✅ | Euler cost ~10.5 dB, correction recovers ~38% |

---

## 文档与叙事 ✅

| # | 任务 | 状态 | 产出 |
|---|------|------|------|
| 7.1 | 论文叙事框架（What→Why→How） | ✅ | `THESIS_NARRATIVE.md` |
| 7.2 | 项目 CLAUDE.md | ✅ | 持续维护，已含全部关键数据 |
| 7.3 | RLI 文献对比分析 | ✅ | 已 commit (`57ba59f`) |
| 7.4 | 统一框架架构图 | ✅ | `scripts/gen_unified_framework_diagram.py` |

---

## 待办：论文写作

| # | 章节 | 状态 | 依赖 |
|---|------|------|------|
| 8.1 | 第 1 章 引言 | ⬜ 未开始 | — |
| 8.2 | 第 2 章 相关工作 | ⬜ 未开始 | RLI 分析已完成 |
| 8.3 | 第 3 章 诊断 | ⬜ 未开始 | Phase 1 + Phase 4 跨架构数据齐全 |
| 8.4 | 第 4 章 理论 | ⬜ 未开始 | 因果消融 + MI + 流形 + 收敛性数据齐全 |
| 8.5 | 第 5 章 校正 | ⬜ 未开始 | Phase 2 + Phase 5 数据齐全 |
| 8.6 | 第 6 章 应用 | ⬜ 未开始 | 人像/建筑/艺术字体场景数据 |
| 8.7 | 第 7 章 讨论 | ⬜ 未开始 | — |
| 8.8 | 第 8 章 结论 | ⬜ 未开始 | — |
| 8.9 | 答辩 PPT | ⬜ 未开始 | — |
| 8.10 | THESIS_NARRATIVE.md 更新（加入 FLUX） | ✅ | Phase 6 + 7c + 8 内容已加入 |
| 8.11 | INNOVATION_POINTS.md | ✅ | 2026-07-08 重写，移除自适应 λ |
| 8.12 | ICLR_PAPER_DEFINITIONS.md | ✅ | v3.1, 包含 Claim-Evidence-Conclusion |

---

## 待办：阻塞项（CVPR 投稿前必须完成）

| # | 任务 | 优先级 | 说明 |
|---|------|--------|------|
| 9.1 | SDXL 漂移 + 校正扩展到 19 图 | ✅ 已完成 | 22 图，Δ=+5.08 dB |
| 9.2 | HunyuanDiT 漂移 + 校正扩展到 19 图 | ✅ 已完成 | 20 图，λ 扫描 + 消融，transition-only +5.65 dB |
| 9.3 | 真实编辑 benchmark（P2P-style prompt-changed） | ✅ 已完成 | Phase 7，20 对评估 |
| 9.4 | 指纹与架构拓扑对应关系表 | ✅ 已完成 | `arch_topo_fingerprint_mapping.png` |

## 待办：提升竞争力

| # | 任务 | 优先级 | 说明 |
|---|------|--------|------|
| 10.1 | 论文写作（8 章） | 高 | 实验数据齐全 |
| 10.2 | 更多 baseline（LEDITS++, InfEdit, PnP, MasaCtrl） | 中 | 现有对比已覆盖主要方法 |
| 10.3 | 扩充测试集至 50+ 张 | 低 | 19 张已有统计检验，若盲审要求可追加 |

---

## Phase 7c：Skip Connection 因果干预 ✅

| # | 任务 | 状态 | 产出 |
|---|------|------|------|
| 7c.1 | Cut A (peak skip) 干预实验 | ✅ | 31/38 层 p<0.05, 漂移 −27.7%, PSNR +2.20 dB |
| 7c.2 | Cut B (低漂移 skip) 对照 | ✅ | 5/38 层显著, 无重建影响 |
| 7c.3 | Noise A 噪声注入（机制分离） | ✅ | 漂移 +6.4%, PSNR +2.40 dB (打破漂移-质量相关性) |
| 7c.4 | 部分调制剂量-响应 (α ∈ [0,1]) | ✅ | PSNR 随 α↓ 单调上升, 2026-07-10 重跑验证 |
| 7c.5 | SDXL 跨架构因果验证 | ✅ | −11.59 dB（相反效应） |

---

## Phase 8：ICLR 补充实验 ✅

| # | 任务 | 状态 | 产出 |
|---|------|------|------|
| 8a.1 | 跨 prompt 验证 (25 prompts) | ✅ | ΔPSNR +1.31, p=0.0012 |
| 8b.1 | 编辑验证 (25 tasks × 3 conditions) | ✅ | CLIP-Dir → 0 for Cut A/Noise A |
| 8c.1 | SDXL 跨架构因果验证 | ✅ | −11.59 dB（见 7c.5） |
| 8d.1 | FLUX feature-level λ scan (5 图) | ✅ | 最优 λ=0.1, +1.38 dB |
| 8d.2 | FLUX 层组消融 (19 图) | ✅ | joint=single=latent=+3.05, late_single=+3.18 |
| 8d.3 | EDICT/NTI 补齐 4 缺失图 | ✅ | 所有方法现基于相同 19 图 |

**关键发现**："Single > Joint" 预测证伪, "注入位置鲁棒" 跨架构复现, late_single 揭示 residual stream 方向性。
