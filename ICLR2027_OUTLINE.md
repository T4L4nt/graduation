# Architecture Fingerprint of Feature Drift in Diffusion Inversion

> ICLR 2027 投稿大纲。v2.1 — 论证驱动，信息密度收束。
> 三条主线：概念必要性（融入 Intro）→ 证据层级（贯穿 Findings）→ 泛化意义（Discussion）。

---

## Abstract

扩散模型反演（DDIM inversion）在重建图像时引入的特征漂移，通常被视为需要抑制的不稳定因素。本文提出，这种漂移不是随机副作用，而是一种由网络架构决定的、可测量且可复现的结构属性。我们将这一属性定义为 **Architecture Fingerprint**，并证明它可以作为描述、比较和诊断扩散模型架构行为的统一分析对象。

在 SD 1.5、SDXL、HunyuanDiT、FLUX、SD 3.5 与 PixArt-Σ 六个架构上的系统实验表明，该指纹在同一架构内对继续训练、LoRA 蒸馏、文本编码器替换和全量微调均保持稳定；DiT-S/2 的训练追踪显示，指纹在训练早期即快速形成并锁定。随机初始化实验揭示了指纹的生成机制：网络拓扑决定了可能形成的指纹空间，而训练负责选择并锁定其中的具体实例。

跨架构比较进一步表明，决定指纹相似性的不是 Transformer 采用 single- 还是 dual-stream attention，而是跨模态交互机制本身。这一稳健性的根源在于功能子空间错位——漂移集中于 ResNet 残差流，而微调更新主要写入 cross-attention 层，两者相对独立。由此出发，Architecture Fingerprint 可直接驱动架构级诊断：同一 skip 连接在不同架构中可能承担不同甚至相反的功能角色，而一种一次性的架构级校正即可达到与逐图像优化方法等价的内容保持效果。

这些结果表明，特征漂移并非扩散反演中的随机副作用，而是一种稳定的架构属性；Architecture Fingerprint 为扩散模型提供了统一的架构级分析、比较与校正视角。

---

## Figure Storyline

| Figure | 一句话 | 承载的论证 |
|--------|--------|-----------|
| Fig.1 | 为什么 profile 不够，为什么需要 Fingerprint | 概念必要性 |
| Fig.2 | Architecture Fingerprint 的定义与边界谱系 | C1 分量级不变性 |
| Fig.3 | DiT-S/2 结晶曲线（eps+flow） | C3 目标不变性 + "拓扑×训练"生成机制 |
| Fig.4 | SD 1.5 vs SDXL skip 干预对比 | C4 机制 (UNet实例) + MMDiT boundary spike |
| Fig.5 | 诊断→校正：random5≈top5 / transition-only≫top5 | Application |

> 五张图自己就是完整故事。审稿人不读正文也能理解论文讲了什么。

---

## 1. Introduction

### 1.1 问题

扩散反演（DDIM inversion）在重建图像时在 UNet/Transformer 的每一层产生特征偏差（feature drift）。现有方法一致将这种漂移视为需要抑制的不稳定因素——EDICT 追求可逆性，NTI 优化轨迹，P2P 注入注意力。

### 1.2 为什么逐层剖面不够

逐层漂移剖面（layer-wise drift profile）是局部观测量的拼接。它回答"哪一层漂移最大"，但不能回答"为什么这个架构以这种方式漂移"。具体地：
- **不可比较**：不同架构层数不同（SD 1.5 有 38 层，SDXL 有 28 层），profile 不能直接做 pairwise 比较
- **不携带架构信息**：profile 本身是数字序列，不含拓扑语义（瓶颈在哪、skip 连接如何传播、跨模态交互边界在何处）
- **不能支撑诊断**：从 profile 只能知道"哪层漂移大"，不能知道"这个结构组件是冲突源还是信息通路"

一个具体的例子：PixArt-Σ 和 SDXL 均采用不同的 backbone 设计（DiT with cross-attention vs UNet），层数完全相同（28 层），逐层 profile 可比性受 VAE 维度、量级差异等因素干扰——但它们在 Architecture Fingerprint 空间中峰位重合（D_pp=0.000，D_total=0.142），是全部 10 对跨架构比较中最近的一对。相比之下，PixArt-Σ 与同属 single-stream Transformer 的 HunyuanDiT 距离 0.468（3.3× 更远）。Profile 无法揭示这种结构组织层面的亲缘关系。

### 1.3 Architecture Fingerprint

本文提出，漂移的逐层组织模式不是随机波动，而是一种由网络架构决定的、可测量且可复现的结构属性。我们将这一属性定义为 **Architecture Fingerprint** Φ(M)——架构 M 在固定反演协议下，逐层漂移的归一化组织模式。

Φ(M) 不是理论（不解释为什么漂移发生），不是假说（不预测如何传播），而是一个**测量框架**——它为后续的跨架构比较、拓扑映射、机制分析和诊断校正提供共同基础。它和 profile 的关系类似于 NMR 谱和分子中原子坐标的关系：同一个分子在不同溶剂和温度下产生不同的 NMR 谱，但谱的峰位和分裂模式仍能识别分子结构；同理，Φ(M) 在不同的 checkpoint、训练目标和采样器下变化，但其组织模式仍能识别架构身份。

### 1.4 本文的核心主张

1. **分量级不变性**：Φ(M) 的拓扑分量（峰位）在同一架构内对 checkpoint 更替、LoRA、微调和编码器替换保持严格不变；度量分量呈条件依赖的弱变异，但最大变异仍远小于跨架构距离。
2. **生成机制**：网络拓扑定义可形成的指纹空间，训练选择并锁定具体实例（"拓扑×训练"）。
3. **按跨模态交互机制聚类**：指纹相似性由跨模态交互的具体实现机制决定，而非简单的 single/dual-stream attention 二分。
4. **架构级诊断**：Φ(M) 可直接定位架构瓶颈，驱动一次性的架构级校正，效果与逐图像优化方法等价。

### 1.5 证据层级

本文明确区分四层证据，每个结论的措辞严格匹配其支撑证据的层级：

| 层级 | 定义 | 论文中的典型实例 |
|------|------|----------------|
| **Observation** | 测量到的模式，未声称因果 | 五架构漂移剖面、PixArt 聚类 |
| **Correlation** | 统计关联 | ρ(drift, ΔW)=0.24 |
| **Intervention** | 操纵变量后测量效应 | Cut A/B、噪声替换、剂量曲线 |
| **Prediction** | 冻结假设后 held-out 验证 | 最近质心分类盲测 |

没有声称因果的结论会标注为 "consistent with the hypothesis" 或 "observed association"，不会使用 "causes"、"drives"、"determines"。

---

## 2. Related Work

三条文献线，划清边界：

- **反演误差与轨迹偏差**：DDIM inversion~\citep{song2021ddim} 的反演-重建不一致性已被多项工作关注。NTI~\citep{mokady2023nti} 通过空文本优化改善重建，EDICT~\citep{wallace2023edict} 提出耦合变换实现精确反演，P2P~\citep{hertz2023p2p} 注入交叉注意力保持内容。近期，RF-Inversion~\citep{hong2024rfinversion}、RF-Solver~\citep{wu2024rfsolver}、FlowEdit~\citep{kulikov2024flowedit}、FireFlow~\citep{zhong2024fireflow} 和 DiTCtrl~\citep{chen2024ditctrl} 分别针对 rectified flow 和 DiT 架构优化轨迹偏差。RLI~\citep{rout2024rli} 独立发现了与我们相同的校正公式。上述工作的一致前提是"漂移需要被抑制"——我们的贡献不是发现漂移，而是将漂移的组织结构重新定义为架构的可测量属性。
- **架构内部表示分析**：Diffusion Hyperfeatures~\citep{luo2024diffusionhyperfeatures}、h-space~\citep{kwon2023hspace}、Asyrp~\citep{haas2023asyrp}、FeatureInject~\citep{basu2024featureinject} 等分析前向生成中的语义表示——不涉及反演-重建不一致性。Splice~\citep{tumanyan2023splice} 分析了 DiT 内部表示但不涉及漂移的可比性。
- **架构差异与特征行为**：DiTCtrl~\citep{chen2024ditctrl} 已知 MMDiT 不同层段对编辑敏感度不同——没有将其系统化为可测量的架构签名。从更广的视角看，机械可解释性工作如 Circuits~\citep{olah2020zoom,elhage2021transformer} 和 ACDC~\citep{conmy2023causal} 提供了跨架构分析特征行为的先例，但均未涉及扩散反演中的特征漂移组织。

---

## 3. Architecture Fingerprint: Definition and Empirical Findings

### 3.1 形式化定义

**Feature Drift:** d_l(x) = E_{t∈K}[ ‖f_l^inv − f_l^recon‖_2 ]

**Architecture Fingerprint:** Φ(M) = Normalize({E_{x∈D}[d_l(x)]}_{l=1}^L) ∈ [0,1]^L

Φ(M) 声明对图像集 D、反演协议 P、归一化方案 norm 的依赖。

**v2 Structural Distance:** D_s(A,B) = sqrt(D_pp² + D_shape² + D_mag²)
- D_pp = |peak_pos_A − peak_pos_B|
- D_shape = 1 − Spearman ρ (same-length only)
- D_mag = L2(concentration, Gini spread)

噪声底（B=100 bootstrap）：median=0.007, p95=0.016。最小跨架构距离（FLUX-SD3.5）= 0.092。

### 3.2 Finding 1: Component-Level Invariance

**Φ(M) 呈现分量级不变性：峰位在 checkpoint 谱系内严格不变，度量分量条件依赖。**

**证据**：

| 模型变体 | D_s vs SD 1.5 | D_pp | 类型 |
|----------|-------------|------|------|
| SD 1.4 (继续训练) | 0.011 | 0.000 | Observation |
| LCM LoRA (蒸馏) | 0.021 | 0.000 | Observation |
| RandText (编码器替换) | 0.034 | 0.000 | Intervention |
| RV (全量微调) | 0.043 | 0.000 | Observation |

- v4 高斯扰动剂量曲线（Intervention）：ε ≤ 1e-4 D_total < 噪声底，D_pp 全程 0.000
- ΔW 测量：真实 checkpoint ΔW (0.14–0.21) 超出高斯稳定区三个数量级——L2 权重量级不预测指纹距离

**边界谱系**：
```
SD1.4(0.011) < LCM(0.021) < RandText(0.034) < RV(0.043) ‖ inter-arch(0.092) ‖ DiT-FLUX(0.618) ‖ random(0.699)
```

### 3.3 Finding 2: Clustering by Cross-Modal Mechanism

**指纹相似性由跨模态交互机制决定，而非简单的 attention 拓扑二分。**

**v2 pairwise matrix** (D_s = sqrt(D_pp² + D_mag²), 104 images):

| Pair | D_total | D_pp | D_mag | 注 |
|------|---------|------|-------|-----|
| PixArt-Σ–SDXL | 0.142 | 0.000 | 0.142 | 峰位完全相同 |
| SD1.5–SDXL | 0.207 | 0.201 | 0.047 | 同 UNet |
| FLUX–SD1.5 | 0.264 | 0.219 | 0.148 | |
| PixArt-Σ–SD1.5 | 0.275 | 0.201 | 0.188 | |
| HunyuanDiT–SD1.5 | 0.308 | 0.263 | 0.160 | |
| FLUX–HunyuanDiT | 0.310 | 0.044 | 0.306 | |
| HunyuanDiT–PixArt-Σ | 0.468 | 0.464 | 0.059 | 同 single-stream，远 |
| FLUX–SDXL | 0.463 | 0.420 | 0.195 | |
| HunyuanDiT–SDXL | 0.478 | 0.464 | 0.114 | |
| FLUX–PixArt-Σ | **0.538** | 0.420 | 0.335 | 最远 |

PixArt-Σ 与 HunyuanDiT 同属 "single-stream Transformer with cross-attention"（D_total=0.468，D_pp=0.464），而 PixArt-Σ 与 SDXL 的距离仅 0.142，且峰位完全相同（D_pp=0.000）。值得注意的是，PixArt-Σ 和 SDXL 共享 VAE 架构（SDXL VAE），这一 confound 尚未与跨模态交互机制完全解耦——正文需明确讨论。HunyuanDiT-FLUX 同为 cross-modal-rich 架构，D_pp 仅 0.044，但 D_mag=0.306 使总距离处于中档——说明度量分量（浓度/展宽）携带了独立于峰位的信息。

最近质心分类盲测：RV、LCM、RandText 均正确归入 UNet 质心；PixArt 未能归入同拓扑的 DiT 簇。

### 3.4 Finding 3: Objective Invariance

**峰位与浓度对训练目标不变，绝对量级与预峰形状范式依赖。**

- DiT-S/2 eps vs flow 对照（Intervention）：峰位同归 block 11，浓度差 0.011
- 结晶曲线（Observation）：eps 在 10k 步锁定峰位，flow 在 30k 步锁定。最终稳态一致，收敛速度范式依赖
- 采样器 swap（Intervention）：确定性 ODE（DPM++, Euler, Euler a）保持峰位；随机 DDIM (η=1) 移位

---

## 4. Mechanism: Architectures Localize Information Conflict Differently

> **中心叙事**：Architecture Fingerprint 的形态差异源于不同架构将信息冲突局域化在不同结构组件的不同方式。UNet 通过 skip connection 将冲突集中在上采样段；MM-DiT 通过 cross-modal attention 将冲突释放于 joint→single 边界；子空间错位解释了为什么微调不改变指纹的组织形态。

### 4.1 The Conflict Variable

对每个 skip connection，定义 Conflict：C = ‖s − u‖_2，其中 s 为 skip 特征，u 为 up_block 接收 s 前的内部表征。

因果链：Skip strength α → Conflict C → Drift φ_l → Reconstruction PSNR

### 4.2 UNet Instance: Skip Connection as Conflict Conduit

| 干预 | 结果 | 类型 |
|------|------|------|
| Cut A (α=0, peak skip) | drift −27.7%, PSNR +2.20 dB | Intervention |
| Cut B (α=0, low-drift skip) | 无显著变化 | Intervention |
| Noise A (replace skip with noise) | L2↑ but PSNR↑ | Intervention |
| Dose α∈[0,1] | monotonic loss | Intervention |
| ρ(drift, Cut A Δdrift) = −0.59 | peak layer = peak response | Correlation |

### 4.3 Cross-Architecture Contrast: Same Component, Opposite Role

| | SD 1.5 Cut A | SDXL Cut A |
|---|---|---|
| ΔPSNR | **+2.20 dB** | **−11.59 dB** |
| Fingerprint | Peak at decoder | Peak at mid_block |
| Role | Conflict source | Information pathway |

> 证据层级：Observation（两个数据点）。措辞："同一结构组件在不同 UNet 变体中扮演不同功能角色——Fingerprint 是实例级诊断工具。"

### 4.4 Why Fine-Tuning Doesn't Alter the Fingerprint: Functional Subspace Misalignment

- 漂移集中于 ResNet 残差流
- ΔW 集中于 cross-attention K/V
- Global ρ(drift, ΔW)=0.24; up/attn ρ=0.05
- 证据层级：Correlation。措辞："漂移与微调的功能子空间呈现可测量的错位，这为 checkpoint 不变性提供了一种可能的机制解释。"

### 4.5 MM-DiT Instance: Cross-Modal Boundary as Feature Stabilizer

- FLUX joint_18: image drift 1.55× spike + text drift 3.0× spike
- 证据层级：Observation。措辞："与跨模态交互边界作为特征稳定器的假说一致。因果验证留待后续工作。"

---

## 5. Application: Diagnosis-Guided Correction

### 5.1 诊断逻辑

Φ(M) → Peak Location → Latent Correction: z ← z + λ(z_inv − z_recon)

诊断的贡献是告诉你瓶颈在哪；校正只是例示诊断的价值。这不是"我们发明了一个更好的校正方法"——而是"诊断充分后，最简校正就是最优校正"。

### 5.2 位点依赖性

| 架构 | 洞察 |
|------|------|
| UNet (SD 1.5) | random5 ≈ top5 (ΔPSNR < 0.3 dB) — skip 连接传播校正信号 |
| MM-DiT (FLUX) | joint_only = single_only = latent_all (到 1e-12 dB) — 残差流线性 |
| Transformer (HunyuanDiT) | transition-only ≫ top5 (+5.65 vs +2.50 dB) — 选层关键 |

### 5.3 统计等价性

- P2P vs Ours: TOST ≤ 0.2 dB (p1<0.001, p2=0.033) — 等价
- 100-image 独立评估：ΔPSNR +3.30 dB (d=1.34)
- Cut A LPIPS/SSIM 三指标均改善（排除模糊化伪影）

### 5.4 编辑中的内容锚定

- λ 悬崖曲线：L 形前沿，不存在 sweet spot
- 编辑 LPIPS −85% (121 pairs)

**负结果**：Feature-level injection (−0.27 dB), DCSC 闭环控制 (无增益), Plan B error-edit separation (证伪)。全部支持"诊断→极简"叙事。

---

## 6. Discussion

**概念必要性**：Architecture Fingerprint 和 layer-wise drift profile 的关系，类似于 NMR spectrum 和原子坐标列表的关系——前者是从后者提取的、具有鉴别力的结构化表示。它不替代 profile，而是在 profile 之上增加了一个架构级的抽象层，使跨架构比较、拓扑映射和因果诊断成为可能。

**证据强度**：本文区分了 Observation、Correlation、Intervention 三层证据。Skip conflict 因果链和剂量曲线有因果干预支持；功能子空间错位和跨模态边界效应分别处于 Correlation 和 Observation 层级。任何结论的措辞不超过其证据层级。

**泛化性**：Fingerprint 已在覆盖 UNet、single-stream Transformer、dual-stream MM-DiT 三类架构的六个模型上验证。如果在更多架构上成立，Architecture Fingerprint 可以发展成一种新的架构表示——与 activation、attention、feature 并列的、专门编码架构身份的表示形式。

**局限**：Transformer-only 架构的机制分析限于观测层级；跨架构矩阵 n=2 per class；反演协议仅验证 DDIM/Euler；CFG scale、VAE 维度、文本编码器等 confound 未完全解耦。

---

## 7. 实验状态

### 已完成

C1 分量级不变性 (7 模型边界谱系) · C2 v2 跨架构矩阵 (6 架构 15 pair) · C3 结晶曲线 (eps+flow) · 随机 init · PixArt-Σ · MMDiT boundary · Skip 因果链 (SD1.5+SDXL) · 最近质心盲测 · TOST/BH/FDR · 度量审计

### 待补齐

HunyuanDiT 组件拆分 (pos_embed 调试) · MI shuffle 基线 · PERMANOVA (数据已齐, 分析待做) · 写作

---

## 附录计划

A. v2 度量审计 (peak_count artifact 发现与修复) · B. Bootstrap 噪声底分布 · C. 编辑 benchmark 逐类分布 · D. 预注册哈希承诺 · E. PixArt learned_sigma 验证
