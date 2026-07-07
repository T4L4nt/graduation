# Phase 6: Flow Matching 扩展

> 目标：将诊断→校正框架从 DDIM 扩散迁移到 Flow Matching 范式，实现跨范式、跨架构的系统性验证，达到顶会级别贡献。

---

## 一、科学问题

扩散反演的特征漂移是 DDIM 的特异性现象，还是生成模型特征空间的普遍性质？

**阶段性结论**：Flow Matching 模型中同样存在显著的架构级特征漂移，但其指纹结构与 UNet/DiT 完全不同——**漂移指纹是架构签名，不由采样范式决定**。

---

## 二、FLUX 架构分析

### 2.1 模型选型：`black-forest-labs/FLUX.1-dev`（已下载）

| 属性 | 值 |
|------|---|
| 参数量 | ~12B |
| Backbone | MM-DiT (Dual-stream Transformer) |
| 文本编码器 | CLIP-L/14 + T5-XXL |
| VAE latent | 16 通道（SD 为 4 通道） |
| 范式 | Rectified Flow（velocity prediction） |
| 轨迹 | 直线：x_t = (1-t)·x_0 + t·ε |
| 下载来源 | ModelScope（AI-ModelScope/FLUX.1-dev） |
| 磁盘占用 | ~31GB（删除重复的 flux1-dev.safetensors 后） |

### 2.2 Block 结构

| Block 类型 | 数量 | 功能 |
|-----------|------|------|
| **Joint blocks**（双流） | 19 | text + image tokens 联合 attention |
| **Single blocks**（单流） | 38 | 仅 image tokens，text 通过 modulation 注入 |

总计 57 个 transformer block。

### 2.3 与已有架构的对比

| 维度 | SD 1.5 (UNet) | HunyuanDiT | FLUX.1-dev |
|------|-------------|-----------|-----------|
| Backbone | CNN + Cross-Attn | Transformer | MM-DiT (Dual-stream) |
| 层数 | 196 layers | 40 blocks | 57 blocks (19j+38s) |
| Skip connection | UNet skip | Residual only | Residual only |
| Text 注入 | Cross-attention | AdaLN + Cross-attn | Joint-attn + Modulation |
| 范式 | DDIM (ε-pred) | DDIM (v-pred) | Flow Match (v-pred) |
| 轨迹 | 弯曲 | 弯曲 | **直线** |

### 2.4 关键差异点

1. **直线轨迹**：flow matching 的 x_t 是 x_0 和噪声的线性插值。**实际发现**：Euler 正反向会产生累积误差，基线 PSNR 偏低（~13.5 dB vs SD 1.5 的 ~22.5 dB）
2. **Joint blocks**：text tokens 和 image tokens 在同一个 attention 里混合——可以单独分析 text token 漂移
3. **Single blocks**：text 只通过 MLP modulation 注入
4. **无 UNet skip**：纯 residual 架构

---

## 三、Phase 6a 实际结果 ✅（2026-07-07）

### 3.1 实验设置

- 模型：FLUX.1-dev（ModelScope 下载，BF16，T5 offload 到 CPU）
- 测试集：coco_val 3 张（pilot），计划扩展到 19 张
- 步数：50 步
- Inversion 方法：Euler forward (t=0→1) + Euler backward (t=1→0)
- 校正：latent-space residual correction, λ=0.7
- 输出：`outputs/phase6_flux/diagnosis_summary.json`

### 3.2 漂移指纹

**Top-10 漂移层**：

| 排名 | Block | hidden_drift | encoder_drift | 类型 |
|------|-------|-------------|---------------|------|
| 1 | joint_18 | 0.713 | 0.439 | 最后的 joint block |
| 2 | single_2 | 0.710 | 0.438 | 早期 single |
| 3 | single_0 | 0.708 | 0.440 | 最早的 single |
| 4 | single_1 | 0.706 | 0.439 | 早期 single |
| 5 | single_3 | 0.704 | 0.438 | 早期 single |
| 6 | single_4 | 0.696 | 0.438 | 早期 single |
| 7 | single_5 | 0.691 | 0.438 | 早期 single |
| 8 | single_6 | 0.668 | 0.438 | 早期 single |
| 9 | single_7 | 0.630 | 0.437 | 早期 single |
| 10 | single_8 | 0.589 | 0.437 | 早期 single |

**漂移呈双峰分布**：early single blocks (0-11) 主导 + late single blocks (34-37) 次级峰值 + last joint block (joint_18) 异常高。

### 3.3 组统计

| 组 | n | mean drift | max drift |
|----|---|-----------|----------|
| Joint blocks (image) | 19 | 0.330 | 0.713 |
| **Single blocks (image)** | 38 | **0.464** | 0.710 |
| Joint blocks (text) | 19 | 0.149 | 0.439 |

关键发现：
- **Single blocks 漂移 > Joint blocks**（1.4×）
- **Text token 漂移远小于 Image token**（0.15 vs 0.46，~3× 差距）
- **joint_18** 的 text drift 跳升至 0.44——跨模态信息瓶颈在最后的 joint block

### 3.4 与已有架构漂移指纹对比

| 架构 | Backbone | 范式 | 漂移指纹 | 模式 |
|------|---------|------|---------|------|
| SD 1.5 | UNet (CNN+Cross-Attn) | DDIM | decoder up_blocks ResNet | 单峰（decoder 集中） |
| SDXL | UNet (更大) | DDIM | mid_block | 单峰（中间层） |
| DiT | Transformer (single-stream) | DDIM (v-pred) | blocks 11-21 | 单峰（中层） |
| **FLUX** | **MM-DiT (dual-stream)** | **Flow Match** | **early single + late single + last joint** | **双峰 + 尾端异常** |

**核心发现**：漂移指纹不由采样范式决定（DiT 和 FLUX 都是 Transformer 但指纹完全不同），而由具体的 attention 结构（single-stream vs dual-stream）决定。

### 3.5 校正结果

| 指标 | 无校正 | λ=0.7 校正 | Δ |
|------|--------|-----------|------|
| PSNR | 13.49 ± 0.95 | 16.98 ± 1.21 | **+3.49 dB** |
| SSIM | 0.358 ± 0.080 | 0.511 ± 0.094 | +0.153 |
| LPIPS | 0.687 ± 0.086 | 0.429 ± 0.036 | -0.258 |

**反直觉发现**：FLUX 基线 PSNR（13.49 dB）远低于 SD 1.5（22.45 dB），但校正增益（+3.49 dB）反而大于 SD 1.5（+2.75 dB）。反演越差，校正增益越大——校正利用的是架构内在冗余，不依赖反演精度。

### 3.6 技术笔记

- FLUX transformer 需要 packed latent tokens（`_pack_latents`），不能直接传 VAE latent
- `_unpack_latents` 需要传入图像尺寸（img_h, img_w），不是 latent 尺寸
- 存储全部 50 步的特征（57 blocks × 50 steps）会耗尽 CPU RAM（~42GB）；只存 turnaround 点的特征即可
- T5 offload 到 CPU 是必要的，否则 48GB 显存不够

---

## 四、待完成实验

### 4.1 Phase 6a 收尾

- [ ] 19 张图全量诊断（当前 3 张 pilot 已完成）
- [ ] 生成 FLUX 57 block 漂移热力图
- [ ] 与 SD 1.5 / DiT 的统计相似度分析

### 4.2 Phase 6b：FLUX 残差校正完整消融

- [ ] λ 扫描（0.1, 0.3, 0.5, 0.7, 0.9）
- [ ] 注入位置消融（joint_only / single_only / top5 / random5）
- [ ] 步数鲁棒性（10, 20, 50, 100）
- [ ] Text token 残差注入实验（仅 joint blocks）

### 4.3 Phase 6c：四架构跨范式对比

- [ ] 统一可视化（四张漂移热力图并排）
- [ ] 补充 SDXL/DiT 校正数据（如缺失）
- [ ] 统计检验（FLUX vs 其他架构的漂移分布差异）

### 4.4 Phase 6d：Text token 漂移深度分析

- [ ] 单个 text token 级别的漂移分布（哪些 token 漂移最大？）
- [ ] Text drift 与语义内容的关联
- [ ] joint_18 为何 text drift 跳升？

---

## 五、时间线（修订）

```
✅ 第 1 周  │ FLUX 下载 + 环境搭建 + inversion 实现 + feature hook
           │ Phase 6a pilot（3 张图）完成
           │
第 2-3 周  │ Phase 6a 收尾：19 张图全量 + 热力图
           │ Phase 6b：校正消融 + λ 扫描
           │
第 4-5 周  │ Phase 6c：四架构对比 + 统一可视化
           │ Phase 6d：Text token 深挖
           │
第 6-7 周  │ 写作：Phase 6 实验章节
           │ 更新 THESIS_NARRATIVE.md + CLAUDE.md
```

---

## 六、风险更新

| 风险 | 原概率 | 实际结果 |
|------|--------|---------|
| FLUX 下载失败 | 中 | ✅ 已解决：ModelScope 下载成功，删重复文件 |
| Flow matching inversion 不标准 | 中 | ⚠️ Euler 正反向基线 PSNR 偏低（13.5 dB），但校正仍有效 |
| 漂移太小不出彩 | 中 | ✅ 漂移信号清晰，双峰模式明显 |
| 48G 显存不足 | 中 | ✅ 已解决：T5 offload + 只存 turnaround 点特征 |
| 校正增益小 | 低 | ✅ 反直觉：ΔPSNR=+3.49 反而大于 SD 1.5 |

---

## 七、交付物（更新）

1. ✅ FLUX 57 block 漂移指纹数据（`outputs/phase6_flux/diagnosis_summary.json`）
2. ✅ FLUX common utilities（`scripts/flux_common.py`）
3. ✅ Phase 6a 诊断脚本（`scripts/flux_phase6_diagnosis.py`）
4. ⬜ FLUX 漂移热力图
5. ⬜ 四架构并排漂移热力图
6. ⬜ FLUX 校正完整消融表
7. ⬜ Text token 漂移深度分析
8. ⬜ 更新后的论文叙事
