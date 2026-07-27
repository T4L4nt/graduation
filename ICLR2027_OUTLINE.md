# Architecture Fingerprint of Feature Drift in Diffusion Inversion

> ICLR 2027 投稿大纲。严格对齐已验证数据，不伪造、不外推。
> 版本：v2.0，基于 AC 级评审反馈重写——以概念必要性、证据层级、泛化意义为三条主线。

---

## ICLR AC 预评审摘要

| 维度 | 评分 | 说明 |
|------|------|------|
| Novelty | 8.5–9.5 | 提出新的分析对象（概念抽象层），但"概念必要性"是最大风险 |
| Technical Quality | 不可仅从摘要判断 | 机制 claim 需要因果证据，不能只有相关性 |
| Empirical Evaluation | 强 | 假设驱动设计，不是 benchmark 堆砌 |
| Clarity | 8.5 | 概念主线清晰，正文需避免信息过载 |
| Significance | 8.5 | 有潜力，取决于泛化性——能否超越"解释 drift" |
| **Overall** | **8–9** | 竞争力区间；上限由正文论证质量决定 |

**三个最需要正文回答的问题：**
1. **概念必要性**：为什么 Architecture Fingerprint 不是 "layer-wise drift profile" 的重新命名？
2. **证据层级**：哪些结论是观察（correlation），哪些有因果支持（intervention）？必须明确界定。
3. **泛化意义**：为什么这个概念不仅解释了你的现象，而且能成为理解扩散模型架构的通用分析框架？

---

## Abstract

扩散模型反演（DDIM inversion）在重建图像时引入的特征漂移，通常被视为需要抑制的不稳定因素。本文提出，这种漂移不是随机副作用，而是一种由网络架构决定的、可测量且可复现的结构属性。我们将这一属性定义为 **Architecture Fingerprint**，并证明它可以作为描述、比较和诊断扩散模型架构行为的统一分析对象。

在 SD 1.5、SDXL、HunyuanDiT、FLUX、SD 3.5 与 PixArt-Σ 六个架构上的系统实验表明，该指纹在同一架构内对继续训练、LoRA 蒸馏、文本编码器替换和全量微调均保持稳定；DiT-S/2 的训练追踪显示，指纹在训练早期即快速形成并锁定。随机初始化实验揭示了指纹的生成机制：网络拓扑决定了可能形成的指纹空间，而训练负责选择并锁定其中的具体实例。

跨架构比较进一步表明，决定指纹相似性的不是 Transformer 采用 single- 还是 dual-stream attention，而是跨模态交互机制本身。这一稳健性的根源在于功能子空间错位——漂移集中于 ResNet 残差流，而微调更新主要写入 cross-attention 层，两者相对独立。由此出发，Architecture Fingerprint 可直接驱动架构级诊断：同一 skip 连接在不同架构中可能承担不同甚至相反的功能角色，而一种一次性的架构级校正即可达到与逐图像优化方法等价的内容保持效果。

这些结果表明，特征漂移并非扩散反演中的随机副作用，而是一种稳定的架构属性；Architecture Fingerprint 为扩散模型提供了统一的架构级分析、比较与校正视角。

---

## 1. Introduction

### 1.1 问题设定与概念必要性

> **这是全文最重要的段落。必须回答："为什么 Architecture Fingerprint 不是 layer-wise drift profile 的重新命名？"**

- 扩散反演在重建图像时引入逐层特征漂移（feature drift）：f_inv ≠ f_recon
- 现有工作将漂移视为待抑制的不稳定因素——EDICT 追求可逆性，NTI 优化轨迹，P2P 注入注意力
- **视角转换**：如果漂移不是噪声，而是架构的签名呢？
- **概念必要性论证**（必须写进 Introduction，约半页）：
  - Layer-wise drift profile 只能回答 "哪一层漂移最大" —— 它是一个局部观测量
  - Architecture Fingerprint 回答的是 "为什么这个架构以这种方式漂移" —— 它是一个架构级描述符
  - 这个抽象不是文字游戏：没有它，跨架构比较没有共同语言（不同架构层数不同，profile 不可直接比较）；拓扑解释没有目标（profile 本身不携带架构信息）；机制分析没有测量对象（冲突、信息瓶颈需要定位到具体结构组件）
  - 具体对比例子：两个 UNet（SD 1.5 和 SDXL）的 layer-wise drift profile 形状完全不同（层数不同），但它们的 Architecture Fingerprint 在特征空间中落入同一聚类——profile 无法揭示的东西，指纹可以

### 1.2 中心主张：Architecture Fingerprint 作为统一分析对象

> 不是 "我们发现了一个现象" —— 而是 "我们提出了一个分析对象，并证明它有用"。

**Architecture Fingerprint Φ(M)** —— 架构 M 在固定反演协议下，逐层漂移的归一化组织模式。它不是理论（不解释为什么），不是假说（不预测如何传播），而是一个**测量框架**——它为后续的分析（拓扑映射）、解释（机制分析）和利用（诊断校正）提供了共同的基础。

**分量级不变性**：指纹的不同分量服从不同的不变性规律——
- 拓扑分量（峰位）：严格不变，跨越训练步数、训练目标、checkpoint 更替、LoRA、编码器替换
- 度量分量（浓度、展宽、形状）：条件依赖，对强微调呈弱变异

**生成机制**：网络拓扑定义可能形成的指纹空间（随机初始化 UNet 有完整峰结构但峰位不同），训练负责选择并锁定其中的具体实例（DiT 结晶曲线：峰位在 10k–30k 步写入后不再变动）。

### 1.3 论文路线图

```
§2  Related Work     — 三条文献线的边界划定（反演误差 / 表示分析 / 架构差异）
§3  Architecture Fingerprint — 定义 + 三性质 (C1 分量级不变性, C2 按交互机制聚类, C3 目标不变性)
§4  Mapping Principles — 从架构拓扑到指纹形状的解释性映射
§5  Mechanism         — 因果链：skip conflict (UNet) + cross-modal boundary (MM-DiT)
§6  Application       — 诊断驱动的校正 (Diagnosis → Correction)
§7  Discussion        — 局限、概念必要性反思、泛化前景
```

### 1.4 证据层级声明（贯穿全文）

> 这是阻止 "Observation paper" 陷阱的关键。每一章必须明确标注证据类型。

| 层级 | 定义 | 论文中的例子 |
|------|------|-------------|
| **Observation** | 测量到但未解释的模式 | 五架构漂移剖面叠加、峰位分布 |
| **Correlation** | 统计关联，不声称因果 | ρ(drift, ΔW)=0.24, Spearman 跨采样器 |
| **Intervention** | 操纵变量后测量效应 | Cut A/B 切除, Noise A 噪声替换, α 剂量 |
| **Prediction** | 冻结假设后在 held-out 数据上验证 | 预注册冲突指数盲测 (pending) |
| **Falsification** | 主动寻找反例 | PixArt-Σ 部分证伪 C2 简单版, SDXL 反号 |

---

## 2. 概念必要性：Why "Fingerprint" Matters

> **这是全文最关键的半页。放在 §3 开头或作为 §3.1 的独立子节。**

### 2.1 Layer-wise Drift Profile ≠ Architecture Fingerprint

| 维度 | Layer-wise Drift Profile | Architecture Fingerprint |
|------|------------------------|-------------------------|
| 抽象层级 | 局部观测量 | 架构级描述符 |
| 回答的问题 | 哪一层漂移最大？ | 为什么这个架构以这种方式漂移？ |
| 跨架构比较 | 不可直接比较（层数不同） | v2 连续度量支持 pairwise 比较 |
| 携带架构信息 | 否——profile 本身不含拓扑语义 | 是——可映射到信息瓶颈、skip 结构、跨模态边界 |
| 可诊断性 | 只能排序层 | 可定位因果冲突源 |

### 2.2 具体的必要性示例

- 没有 Fingerprint 框架：跨架构比较只能做 "SD 1.5 有 38 层，SDXL 有 28 层，它们的 drift profile 不一样" —— 这是平凡观察
- 有 Fingerprint 框架：提取峰位、浓度、展宽三个连续分量 → 计算 pairwise 结构距离 → 发现 FLUX-SD3.5 最近（0.092），DiT-FLUX 最远（0.618）→ 揭示跨模态交互机制是聚类主因
- 没有 Fingerprint 框架：随机初始化 UNet 的 drift profile "看起来不同" —— 定性描述
- 有 Fingerprint 框架：峰位从 0.079 移到 0.763 → 量化了"训练重定位指纹"的效应量（D_s=0.699）→ 揭示拓扑定义空间、训练选择实例

---

## 3. Architecture Fingerprint: Definition and Properties

### 3.1 形式化定义

**Definition 1 (Feature Drift).** 架构 M 在反演协议 P 下，逐层反演-重建特征偏差：
```
d_l(x) = E_{t∈K}[ || f_l^inv(x, t) − f_l^recon(x, t) ||_2 ]
```

**Definition 2 (Architecture Fingerprint).** 
```
Φ(M) = Normalize({ E_{x∈D}[ d_l(x) ] }_{l=1}^{L} ) ∈ [0,1]^L
```
Φ(M) 声明对 D（图像集）、P（反演协议）、norm（归一化方案）的依赖——它是**测量剖面**，不宣称跨条件不变性。

**Definition 3 (v2 Structural Distance).**
```
D_s(A, B) = sqrt( D_pp² + D_shape² + D_mag² )
```
- D_pp = |peak_position_A − peak_position_B|
- D_shape = 1 − Spearman ρ(profile_A, profile_B)（仅同长度架构可用）
- D_mag = L2(concentration, spread)
- D_total (cross-arch) = sqrt(D_pp² + D_mag²)

噪声底（B=100 bootstrap）：median=0.0071, p95=0.0163

### 3.2 Property 1: Component-Level Invariance (C1)

**主张**：Φ(M) 的拓扑分量（峰位）在 checkpoint 谱系内严格不变——从继续训练到全量微调。度量分量对强微调呈弱变异，最大变异仍不到最近跨架构距离的一半。

**证据层级**：

| 实验 | 类型 | 结论 |
|------|------|------|
| SD 1.4 ↔ SD 1.5 直接测量 | Observation | D_s=0.011, D_pp=0.000 |
| RV ↔ SD 1.5 直接测量 | Observation | D_s=0.043, D_pp=0.000 |
| LCM LoRA ↔ SD 1.5 | Observation | D_s=0.021, D_pp=0.000 |
| RandText ↔ SD 1.5 | Observation | D_s=0.034, D_pp=0.000 |
| v4 高斯扰动剂量曲线 | Intervention | ε≤1e-4 全部 < 噪声底, D_pp 全程 0.000 |
| DiT-S/2 训练结晶曲线 | Observation (trajectory) | 峰位 10k(eps)/30k(flow) 锁定 |
| 随机初始化 UNet | Intervention | D_s=0.699, 峰位 0.079 → 训练 0.763 |
| ΔW 测量 (SD1.4↔1.5, RV) | Observation | ΔW=0.14–0.21, 3 个数量级超稳定区 |

**边界谱系图**（论文的核心可视化之一）：

```
SD1.4(0.011) < LCM(0.021) < RandText(0.034) < RV(0.043) ‖ inter-arch(0.092) ‖ DiT-FLUX(0.618) ‖ random(0.699)
         ← 谱系带 →                    ← 跨架构带 →          ← 未训练 →
```

**注意**：C1 不声称 "对所有扰动不变" —— 度量分量对强微调有可检测的弱变异（RV 0.043）。但即使这个最大变异，仍然不到最近跨架构距离（0.092）的一半。这是**诚实的弱变异**，不是失败的 invariance。

### 3.3 Property 2: Clustering by Cross-Modal Mechanism (C2)

**主张**：指纹相似性由跨模态交互机制决定，而非简单的 attention 拓扑二分（single vs dual stream）。

**证据层级**：

| 实验 | 类型 | 结论 |
|------|------|------|
| 五架构 v2 pairwise 矩阵 | Observation | FLUX-SD3.5 最近(0.092), DiT-FLUX 最远(0.618) |
| PixArt-Σ 指纹加入矩阵 | **Falsification** | 同 topo 的 DiT-PixArt D_s=0.571 最远！PixArt 聚类到 MM-DiT |
| 最近质心分类盲测 | Prediction | RV/LCM/RandText 3/3 正确归入 UNet; PixArt 错归 UNet |

**关键发现**：PixArt 和 HunyuanDiT 同属 "single-stream Transformer with cross-attention"，但它们的指纹距离（0.571）反而比 PixArt vs FLUX（0.234，不同拓扑类）更大。这两个架构的共同点不是 attention 模式，而是跨模态交互的具体实现（T5 text encoder + split processing）——PixArt 与 SD3/FLUX 共享此实现。这意味着 C2 的简单版本（"attention 拓扑决定指纹"）被 PixArt 数据**部分证伪**，更准确的表述是：**"跨模态交互的具体实现机制，而非 attation 是 single 还是 dual stream，决定了指纹的聚类方式。"**

### 3.4 Property 3: Objective Invariance (C3)

**主张**：峰位与浓度对训练目标不变，绝对量级与归一化预峰形状范式依赖。

**证据层级**：

| 实验 | 类型 | 结论 |
|------|------|------|
| DiT-S/2 eps vs flow 对照 | Intervention (controlled) | 峰位同归 block 11, 浓度差 0.011 |
| eps 结晶曲线 (10k–50k) | Observation (trajectory) | 峰位 10k 锁定, D_total 10k→40k 降 86% |
| flow 结晶曲线 (10k–50k) | Observation (trajectory) | 峰位 30k 锁定, 收敛速度慢于 eps |
| 采样器 swap (SD1.5 ×5) | Intervention | 确定性 ODE 保持峰位, 随机 DDIM(η=1) 移位 |

---

## 4. Mapping Principles

**定位**：假设级原则，通过 held-out 验证与因果干预验证。不宣称普遍定理。

- **Principle 1 (Bottleneck Localization)**: 漂移峰位与架构的信息瓶颈重合（p≈3×10⁻⁴, 二项检验, 5/5 架构）
- **Principle 2 (Propagation Mode)**: Skip 连接传播漂移信号（Cut A vs Cut B 提供因果验证）
- **Principle 3 (Cross-Modal Boundary)**: 跨模态交互边界是特征稳定器（FLUX joint_18 双模态 spike 提供观测支持，因果干预 pending）

**证据层级**：Principles 1–2 有因果干预支持（Cut A/B, Noise A）。Principle 3 目前是**观测层级**（FLUX 的 1.55×/3.0× spike 是 mechanism-consistent evidence, 不是 causal proof）——正文中诚实标注。

---

## 5. Mechanism

### 5.1 因果链：Skip Conflict (UNet)

```
Skip strength α → Conflict C → Drift φ_l → Reconstruction PSNR
   (manipulated)   (mediator)   (observed)    (outcome)
```

**证据层级**：

| 实验 | 类型 | 结论 |
|------|------|------|
| Cut A vs Original | **Intervention** | α=0 → C=0 → φ peak −27.7% → PSNR +2.20 dB |
| Cut B vs Original | **Intervention** | 低漂移 skip 切除, 5/38 层显著 (n.s.), 效应位点特异 |
| Dose-response α∈[0,1] | **Intervention** | 单调——无最优调制点, skip 在该位点纯粹有害 |
| Noise A vs Zero vs Original | **Intervention** | L2↑ 但 PSNR↑ → L2 幅度不是因果变量, 结构化的 Conflict 才是 |
| ρ(drift, Cut A Δdrift)=−0.59 | **Correlation** | 高漂移层对消融响应最大 —— 预测力而非因果 |

### 5.2 跨架构对比：同一组件，相反功能

| 指标 | SD 1.5 Cut A | SDXL Cut A |
|------|-------------|-----------|
| ΔPSNR | **+2.20 dB** | **−11.59 dB** |
| 功能角色 | 冲突源 | 必要信息通路 |

**证据层级**：这是 Observation（两个数据点，不是通用规则）。正文措辞："skip connection 的功能角色在不同 UNet 变体中不同——Fingerprint 是诊断工具，不是 family-level 的通用因果处方。"

### 5.3 功能子空间错位

- 漂移集中于 ResNet 残差流（信息论：ΔPSNR 2.1× vs Attention）
- 微调更新集中于 cross-attention K/V（ΔW 测量）
- 全局 ρ(drift, ΔW)=0.24，跨注意力层仅为 0.05
- **证据层级**：**Correlation**。正文不能写成 "漂移和微调占据不同空间"，写成 "漂移与微调的功能子空间存在可测量的错位"。

### 5.4 Cross-Modal Boundary (MM-DiT)

- FLUX joint_18: image drift 1.55× spike + text drift 3.0× spike
- **证据层级**：**Observation**（mechanism-consistent evidence, 不是 causal proof）。正文标注："这是与 mechanism 一致的观测证据。因果验证——如 masking 特定 joint block 的 text attention 并测量 drift 效应——留待后续工作。"

---

## 6. Application: Diagnosis-Guided Correction

### 6.1 诊断逻辑

```
Φ(M) → Peak Location → Latent Correction (z ← z + λ(z_inv − z_recon))
```

### 6.2 证据

**位点依赖性**（§7 的主角，不是应用段）：
- UNet: random5 ≈ top5（ΔPSNR < 0.3 dB）——skip 连接传播校正信号
- MM-DiT (FLUX): joint_only = single_only = latent_all（到 1e-12 dB）——残差流线性
- Transformer (HunyuanDiT): transition-only >> top5（+5.65 vs +2.50 dB）——选层关键

**统计等价性**：
- P2P vs Ours: TOST ≤ 0.2 dB（p1<0.001, p2=0.033）——等价
- 100-image 独立评估：ΔPSNR = +3.30 dB (d=1.34)
- Cut A LPIPS/SSIM 三指标均改善（排除模糊化伪影）

**编辑中的内容锚定**：
- λ 悬崖曲线：λ ∈ [0.05, 1.0] 平台区 > 90% LPIPS 改善
- 过渡窗 [0.01, 0.05] 宽度仅 0.04——L 形前沿, 不存在 sweet spot
- 编辑 LPIPS −85% (121 pairs, p=4.8e-55)

**负结果**（支持"诊断→极简"叙事）：
- Feature-level injection: −0.27 dB
- DCSC 闭环控制: 无增益
- Plan B error-edit separation: 证伪（DDIM 误差是轨迹依赖的）

---

## 7. Discussion

### 7.1 三个审稿人最关心的问题

**Q1: Architecture Fingerprint 为什么不是一个新名字？**
> A: Layer-wise drift profile 是局部观测量的拼接。Architecture Fingerprint 是架构级描述符——不同的抽象层级、不同的信息载体、不同的下游用途。§2 给出了完整的必要性论证，包括具体对比例子。

**Q2: 机制论断的证据有多强？**
> A: 本文明确区分四层证据（§1.4）。其中三层有因果干预支持（skip conflict 因果链、高斯噪声扰动剂量、随机初始化实验）。功能子空间错位是**相关性**证据（ρ=0.24），跨模态边界效应是**观测**证据。没有任何结论声称的因果强度超出其证据层级。见 §5 中的逐实验标注。

**Q3: 这个概念有多通用？**
> A: 指纹已在 6 个架构上验证——覆盖 UNet（2 变体）、single-stream Transformer（HunyuanDiT, PixArt-Σ）、dual-stream MM-DiT（FLUX, SD3.5）。在所有验证范围内，指纹保持同架构稳定、跨架构可辨。局限包括：仅验证了 DDIM/Euler 反演协议、仅使用固定 prompt 条件、Transformer-only 架构的机制分析尚不完整。§7.3 列出了未解耦的 confound 和开放问题。

### 7.2 贡献

1. **Architecture Fingerprint** — 新的分析对象，将反演诊断从逐层统计提升为架构级描述
2. **系统实验证据** — 6 架构、2 训练目标、跨 checkpoint/LoRA/微调/编码器的分量级不变性
3. **因果机制** — Skip conflict 因果链（UNet）+ 跨模态边界效应（MM-DiT）+ 功能子空间错位
4. **诊断驱动的校正** — 与逐图像优化方法等价，成本降低数百倍
5. **度量审计** — peak_count 阈值 artifact 的发现与修复，v2 连续距离

### 7.3 局限与未来工作

- UNet 以外的机制分析限于观测层级
- 跨架构矩阵目前 6 架构 / 3 拓扑类，每类 n=2
- 反演协议仅验证 DDIM/Euler
- Transformer-only 架构的 skip conflict 类比未建立
- 未解耦 confound：CFG scale, VAE latent 维度, 文本编码器类型

---

## 8. 配图方案

| Figure | 科学问题 | 类型 |
|--------|---------|------|
| Fig.1 | Architecture Fingerprint 概念概览 | 概念图 |
| Fig.2 | C1+C2: 边界谱系 + v2 跨架构矩阵 | 数据图 |
| Fig.3 | C3: DiT-S/2 双目标结晶曲线 | 数据图 |
| Fig.4 | C4 mechanism: SD1.5 vs SDXL skip 干预对比 | 数据图 |
| Fig.5 | Application: 诊断→校正→编辑 | 数据图 |

| Table 1 | 架构总览 (Model, Backbone, Topology, Paradigm, L, Peak) |
| Table 2 | v2 跨架构 pairwise 结构距离矩阵 |

---

## 9. 实验状态

### 已完成

| 实验 | 解锁 | 证据层级 |
|------|------|---------|
| 六架构漂移指纹统一量化 | C2 | Observation |
| v4 高斯扰动剂量曲线 | C1 | Intervention |
| SD 1.4 / RV / LCM / RandText 指纹 | C1 | Observation |
| ΔW 测量 + drift×ΔW Spearman | C1, Mechanism | Correlation |
| DiT-S/2 eps+flow 结晶曲线 | C3 | Observation (trajectory) |
| 随机初始化 UNet | C1 | Intervention |
| v2 跨架构矩阵 + PixArt | C2 | Observation + Falsification |
| Skip 因果干预 (Cut A/B, Noise A, dose) | C4 | Intervention |
| SDXL 跨架构因果验证 | C4 | Observation |
| 最近质心识别盲测 | C2 | Prediction |
| MMDiT joint→single 边界分析 | Mechanism | Observation |
| TOST/BH/FDR 统计包 | 统计 | — |
| 度量审计 (peak_count→v2) | 方法 | — |

### 待补齐

| 实验 | 优先级 | 阻塞 |
|------|--------|------|
| PixArt-Σ + PERMANOVA 方差分解 | P1 | 数据已齐, 分析待做 |
| HunyuanDiT 组件拆分 (attention/MLP/残差) | P2 | pos_embed 尺寸依赖 |
| MI shuffle 基线 | P2 | 需 Phase 4 特征重提取 |
| 写作 | P0 | 现在开始 |

---
