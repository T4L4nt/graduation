# Architecture Fingerprint of Feature Drift in Diffusion Inversion

> ICLR 2027 投稿大纲。v2.1 — 论证驱动，信息密度收束。
> 三条主线：概念必要性（融入 Intro）→ 证据层级（贯穿 Findings）→ 泛化意义（Discussion）。

---

## Abstract

扩散模型反演（DDIM inversion）在重建图像时引入的特征漂移，通常被视为需要抑制的不稳定因素。本文提出，这种漂移不是随机副作用，而是一种由网络架构决定的、可测量且可复现的结构属性。我们将这一属性定义为 **Architecture Fingerprint**，并证明它可以作为描述、比较和诊断扩散模型架构行为的统一分析对象。

在 SD 1.5、SDXL、HunyuanDiT、FLUX、SD 3.5 与 PixArt-Σ 六个架构上的系统实验表明，该指纹在同一架构内对继续训练、LoRA 蒸馏、文本编码器替换和全量微调均保持稳定——峰位在层粒度上不变，度量分量变异（D_mag ≤ 0.023）远小于跨形态类距离（≈0.2–0.3）；DiT-S/2 的训练追踪显示，指纹在训练早期即快速形成并锁定。随机初始化实验揭示了指纹的生成机制：网络拓扑决定了可能形成的指纹空间，而训练负责选择并锁定其中的具体实例。

跨架构比较揭示了一个不被任何教科书标签预测的形态二分：六种架构在统一的多步协议下干净裂为**内层局域型**（漂移在内部瓶颈层爆发；SD 1.5、SDXL、HunyuanDiT）与**末端累积型**（漂移随深度单调爬升、峰恒落末层；FLUX、PixArt-Σ、SD 3.5）。该二分在整条反演轨迹上稳定，且指纹的分辨力有明确极限：形态类内（斜坡类）三特征空间统计不可区分（低于同模型 bootstrap 噪声底），类间则分辨到层。机制上，两类形态对应反演误差的两种组织方式——瓶颈处的局域化冲突（skip 因果链可干预）与随深度的误差累积。由此出发，Architecture Fingerprint 可直接驱动架构级诊断：同一 skip 连接在不同架构中可能承担不同甚至相反的功能角色，而一种一次性的架构级校正即可达到与逐图像优化方法等价的内容保持效果。

这些结果表明，特征漂移并非扩散反演中的随机副作用，而是一种稳定的架构属性；Architecture Fingerprint 为扩散模型提供了统一的架构级分析、比较与校正视角。

---

## Figure Storyline

| Figure | 一句话 | 承载的论证 |
|--------|--------|-----------|
| Fig.1 | 为什么 profile 不够，为什么需要 Fingerprint | 概念必要性 |
| Fig.2 | Architecture Fingerprint 的定义与边界谱系 | C1 分量级不变性 |
| Fig.3 | DiT-S/2 指纹收敛/结晶曲线（eps+flow） | C3 目标不变性 + "拓扑×训练"生成机制 |
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

一个具体的例子：在 P-multi 协议下，H-DiT（cross-attention DiT）与 SDXL（UNet）——两个教科书标签完全不同的架构——的峰位仅差 0.071（0.500 vs 0.429），是全 15 对跨架构比较中峰位最接近的对之一；而三个形态类成员（FLUX/PixArt-Σ/SD3.5）的漂移剖面全部为单调递增斜坡（Spearman 秩相关 ≥ +0.987）。逐层 profile 无法直接揭示这种结构组织层面的亲缘关系与形态分类。

### 1.3 Architecture Fingerprint

本文提出，漂移的逐层组织模式不是随机波动，而是一种由网络架构决定的、可测量且可复现的结构属性。我们将这一属性定义为 **Architecture Fingerprint** Φ(M)——架构 M 在固定反演协议下，逐层漂移的归一化组织模式。

Φ(M) 不是理论（不解释为什么漂移发生），不是假说（不预测如何传播），而是一个**测量框架**——它为后续的跨架构比较、拓扑映射、机制分析和诊断校正提供共同基础。它和 profile 的关系类似于 NMR 谱和分子中原子坐标的关系：同一个分子在不同溶剂和温度下产生不同的 NMR 谱，但谱的峰位和分裂模式仍能识别分子结构；同理，Φ(M) 在不同的 checkpoint、训练目标和采样器下变化，但其组织模式仍能识别架构身份。

### 1.4 本文的核心主张

1. **分量级不变性**：Φ(M) 的拓扑分量（峰位）在同一架构内对 checkpoint 更替、LoRA、微调和编码器替换保持严格不变（内层峰类中为层粒度不变；末端累积类中退化为末层 argmax 稳定）；度量分量呈条件依赖的弱变异（D_mag ≤ 0.023），远小于跨形态类距离（≈0.2–0.3）。
2. **生成机制**：网络拓扑定义可形成的指纹空间，训练选择并锁定具体实例（"拓扑×训练"）。
3. **漂移形态的架构级二分**：六种架构在统一协议下裂为内层局域型与末端累积型两类形态，该二分不被教科书标签（UNet/DiT/MM-DiT 分类）预测；指纹的分辨力以形态类为界。
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

**Structural Distance（三特征空间）:** D_s(A,B) = sqrt(D_pp² + D_mag²)
- D_pp = |peak_pos_A − peak_pos_B|（峰位差，canonical 层序）
- D_mag = L2(concentration, spread)（量级分量差）
- 峰位/浓度/展宽的完整定义见 DEFINITIONS v3.6

P-multi 噪声底（同模型 bootstrap, B=1000）：median=0.0014, p95=0.0042。

**术语约定（Terminology）**——本文提出以下术语，均以学界标准概念为基础：

| 术语 | 定义 | 依据的标准概念 |
|------|------|---------------|
| 峰位 peak position | 归一化漂移剖面的 argmax 所在层的相对位置（argmax/L） | 序统计 argmax |
| 浓度 concentration | 归一化剖面中漂移最高的 20% 层占总漂移的比例 | 经济学 concentration ratio（top-k share） |
| 展宽 spread | 归一化剖面的 Gini 系数（0=均匀，1=单层集中） | Gini 系数（标准不平等度量） |
| 拓扑分量 | 指纹中由峰位携带的部分 | —（本项目术语，见 DEFINITIONS Definition 2） |
| 度量分量 | 指纹中由浓度+展宽携带的部分 | —（本项目术语） |
| 谱系 lineage | 同一架构家族、共享训练血统的模型集合（如 SD1.5 及其微调/蒸馏变体） | model provenance / fine-tuning lineage |
| 形态类 morphology class | 按归一化漂移剖面的单调性分类：内层局域型（非单调，峰在内部）vs 末端累积型（单调递增斜坡，Spearman(层序,剖面) ≥ 0.95） | 剖面单调性（本项目分类标准，定义于 §3.3） |
| 末端 right-censored 峰 | 峰位于被观测的最后一层，峰后回落不可观测的测量状态 | 生存分析中的 right-censoring |
| 轨迹偏差 trajectory deviation | 反演轨迹与重建轨迹在匹配时间步上的特征差异 | 数值分析中的误差累积（error accumulation）相关概念 |
| 噪声底 noise floor | 同模型 bootstrap 重采样所得 D_mag 分布的分位数（median/p95） | 信号处理 noise floor |

### 3.2 Finding 1: Component-Level Invariance

**Φ(M) 呈现分量级不变性：峰位在 checkpoint 谱系内严格不变，度量分量条件依赖。**

**证据**（历史值，P-tT 单步协议；P-multi 多步协议重测中，见 §3.6 协议审计）：

| 模型变体 | D_s vs SD 1.5 | D_pp | 类型 |
|----------|-------------|------|------|
| SD 1.4 (继续训练) | 0.011 | 0.000 | Observation |
| LCM LoRA (蒸馏) | 0.021 | 0.000 | Observation |
| RandText (编码器替换) | 0.034 | 0.000 | Intervention |
| RV (全量微调) | 0.043 | 0.000 | Observation |

- v4 高斯扰动剂量曲线（Intervention）：ε ≤ 1e-4 D_s < 噪声底，D_pp 全程 0.000
- ΔW 测量：真实 checkpoint ΔW (0.14–0.21) 超出高斯稳定区三个数量级——L2 权重量级不预测指纹距离
- **P-multi 重测结果**：四变体 D_pp 全 0，D_mag ∈ [0.005, 0.023]——C1 核心证据在最严协议下成立
- **内层峰类第二谱系（SDXL-Turbo, P-multi@104）**：蒸馏 checkpoint 与 SDXL base 的 D_pp=0.000、D_mag=0.0022（噪声底 p95 0.0042 之内，统计不可区分）——内层峰类 n=2 谱系证据完备

**边界谱系**（历史 P-tT 值，P-multi 重测后更新）：
```
SD1.4(0.011) < LCM(0.021) < RandText(0.034) < RV(0.043) ‖ inter-arch(0.092) ‖ DiT-FLUX(0.618) ‖ random(0.699)
```

### 3.3 Finding 2: 漂移形态的架构级二分，及其可分辨性极限

**不同形态类用不同坐标读取指纹：内层峰架构以峰位为信息坐标，斜坡架构以形状参数为信息坐标。**

**(1) 形态二分（Observation, P-multi@104）**：

| 形态类 | 成员 | 特征 |
|--------|------|------|
| **内层局域型** | SD1.5 (0.684), SDXL (0.429), H-DiT (0.500) | 漂移在内部瓶颈层局域化，峰位稳定且携带实例身份 |
| **末端累积型（斜坡）** | FLUX (0.983), PixArt-Σ (0.964), SD3.5 (0.958) | 漂移随深度单调爬升，Spearman(层序,剖面) ≥ +0.987，argmax 恒落末层 |

- 二分**不被教科书标签预测**：斜坡类混合了 MM-DiT（FLUX/SD3.5）与 cross-attention DiT（PixArt）；内层峰类混合了 UNet（SD1.5/SDXL）与 cross-attention DiT（H-DiT）
- 斜坡架构的 **argmax 退化**：单调斜坡的峰位恒等于最后一层，pp 不含信息——D_pp 在斜坡对之间只反映 1/L 量化差。斜坡类的信息坐标是形状参数（尾部质量：FLUX/PixArt 45–48% vs SD3.5 63%；凸度）
- 末端 right-censored 峰† 标注：斜坡类全部是删失峰（hook 边界截断，峰后回落不可观测）。SD3.5 在 P-t0 下曾是"已验证近末层峰"（block_22 + 47% 回落），P-multi 下移为 block_23 末层（104/104, t=138.4）——该证据的协议适用范围需显式声明

**(2) 形态全程稳定（F4, per-σ 分解）**：FLUX 末层集中对每个噪声水平成立（σ ∈ {1.0, 0.663, 0.002} 上 single_37 均为峰）——非 t≈0 端点伪影，是全程一致的架构属性。

**(3) 可分辨性极限（F1, 噪声底 + PixArt-α 谱系测试）**：斜坡类内，峰位坐标退化（D_pp 恒为 1/L 量化差），**形状参数携带实例信息但可分辨性不保证**——同谱系对可至 0.096（PixArt-α vs Σ：conc 0.552/0.495，sp 0.455/0.377），跨谱系对可低至噪声底之下（FLUX–PixArt-Σ D_mag=0.0011 < bootstrap 噪声底 median 0.0014/p95 0.0042）。指纹分辨**形态类**；类内分辨需形状参数且不保证。内层峰类内则分辨到层（0.429/0.500/0.684 互异，C1 已证其谱系稳定）。

**(4) 协议轴（P-tT/P-t0/P-multi 双协议对照表）**：指纹依时间聚合方式呈现不同侧面——高噪端单步（P-tT）凸显跨模态边界信号（FLUX s2，全 σ 一致的弱信号 0.13–0.17，Observation 级局部次峰）；全轨迹均值（P-multi）凸显全局形态。FLUX 在 P-tT 下峰于边界、P-multi 下峰于末端——同一架构在不同协议下的呈现差异（protocol-dependent presentation），非矛盾。

**机制解读（与 §4 互证）**：两类形态对应反演误差的两种组织方式——**局域化冲突**（瓶颈处爆发；§4.2 skip 因果链为其机制）vs **逐深度累积**（无局域化结构，误差随网络深度逐层累积（error accumulation across depth））。

**负结果（进负结果清单）**：
1. "D_mag 编码设计谱系"普遍主张——MMDiT 谱系对 FLUX–SD3.5 D_mag=0.306 破裂，UNet 谱系对 SD1.5–SDXL 0.024 存活；n=1 per lineage，普遍主张死亡
2. "谱系内变异远小于跨架构距离"比较级原句——被 RandText（同架构变体 0.0232）vs FLUX–PixArt（跨架构 0.0011）字面证伪；重写为"谱系内变异（≤0.023）远小于**形态类间**距离（≈0.2–0.3）"

**适用范围的主动声明（Limitations）**：指纹 = 架构身份，分三层——内层峰类谱系内识别实例（C1 强主张：峰位层粒度不变）；斜坡类谱系内峰位退化为末层 argmax 稳定（C1 弱主张），形状参数为实例级属性（PixArt-α vs Σ D_mag=0.096）；跨架构识别形态类（Finding 2）；斜坡类内实例分辨不保证（F1，同谱系对可分辨、跨谱系对可至噪声底之下）。

**C1 适用范围声明**："峰位在层粒度上不变"适用于内层峰类（强）；斜坡类中退化为"末层 argmax 稳定"（弱）。

### 3.4 Finding 3: Objective Invariance

**峰位与浓度对训练目标不变，绝对量级与预峰形状范式依赖。**

- DiT-S/2 eps vs flow 对照（Intervention）：峰位同归 block 11，浓度差 0.011
- 指纹收敛曲线（crystallization，Observation）：eps 在 10k 步锁定峰位，flow 在 30k 步锁定。最终稳态一致，收敛速度范式依赖
- **P-multi 审计（v3.6）**：eps 10k 即锁 block 11（pp=0.9167）；flow 从 block 9(10k)→block 7(20k)→block 11(30k 起稳定)——峰位范式不变性在 P-multi 下成立；eps 20k/30k/40k checkpoint 已删（scope limitation，P-t0 值保留为历史）
- 采样器 swap（Intervention）：确定性 ODE（DPM++, Euler, Euler a）保持峰位；随机 DDIM (η=1) 移位
- **随机初始化 P-multi 重跑（5 种子）**：全部种子 pp=0.0263（输入端）vs 训练后 0.684（decoder），D_pp=0.658——"拓扑定义空间、训练选择实例"的干预证据过 P-multi 门槛

### 3.5 Peak Stability Diagnostics (v3 审计)

六架构峰位稳定性（104 图, bootstrap B=10⁴）：

| 架构 | 峰 | margin | bootstrap argmax | sign agreement rate（符号一致率） | 峰型 |
|------|-----|--------|------------------|------------------|------|
| SD1.5 | U2.R0 (0.684) | 22.3% | 100% | — | 内层局域峰 |
| H-DiT | blocks.20 (0.500) | 19.3% | 100% | — | 内层局域峰 |
| FLUX | s2 (0.368) | 0.4% | 99.8% | 66.3% | 内层局域峰（s2/s3 竞争） |
| SD3.5 | block_22 (0.917) | 25.4% | 100% | 104/104 vs b23 | 近末层已验证峰 |
| SDXL | U2.R2 (0.964) | 6.9% | 100% | 96.2% | 末端 right-censored 峰† |
| PixArt-Σ | T27 (0.964) | 30.8% | 100% | 100% | 末端 right-censored 峰† |

†末层删失（right-censored）：漂移向输出端集中，但 hook 范围在末层截断，峰后回落不可观测。不是伪影（2×2 协议×图像集证明），是测量边界的结构事实。

### 3.6 Protocol Audit (P-multi / P-t0 / P-tT)

**发现**：论文历史数据混用了三种漂移测量协议——

| 协议 | K | 参考 | 合规性 |
|------|---|------|--------|
| P-multi（多步均值, Definition 1 合规） | 相对规则 {前3,中3,后3}，逐调度器展开 | DDPM-forward 外部参考 | ✅ canonical |
| P-t0（低噪端单步） | {t≈0 一步} | 反演末步 | 退化近似（K=1），信噪比低 |
| P-tT（高噪端单步） | {t≈T 一步} | 反演首步 | 退化近似（K=1），Band 1 历史值 |

估计量性质：P-t0 单步混入轨迹偏差（trajectory deviation）的通用端点分量（所有架构在输出端都存在轨迹偏差），稀释架构特异信号——量化证据：三带类间分离度 P-t0 下 1.2× vs P-multi 下 ~3×（占位，重算后落定）。

**协议选择准则：跟随预先写下的 Definition 1 与估计量性质，不跟随结果。历史协议数据全部保留，写进稳健性附录。**

**P-multi 重测状态**（104 图, coco_val100）：

| 架构 | 状态 |
|------|------|
| SD1.5 | ✅ pp=0.684, conc=0.583, sp=0.588 |
| SDXL | 运行中 |
| H-DiT | 待重跑（OOM 后重启） |
| FLUX | 待写脚本 |
| SD3.5 | 待写脚本 |
| PixArt-Σ | 待写脚本 |
| Band 1 四变体 | 运行中 |

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
| ρ(drift, Cut A Δdrift) = −0.395 | peak layer = peak response | Correlation |

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

- **P-multi 修订**：FLUX 的全局峰在 P-multi 下位于 single_37（末端，删失†），P-tT 下的 joint→single 边界峰（s2）在 P-multi 下为全 σ 一致的局部弱信号（0.13–0.17），Observation 级局部次峰
- 双模态 spike（joint_18: image drift 1.55× + text drift 3.0×）保留为边界行为证据，措辞降级："边界存在局部双模态 spike，与跨模态交互边界作为特征稳定器的假说一致；**全局漂移组织为末端累积型**。因果验证留待后续工作。"
- 协议依赖声明：FLUX 峰位在 P-tT（边界 s2）与 P-multi（末端 single_37）之间移动——同一架构在两个协议坐标上的两副面孔，双协议对照表见附录

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

- **P-multi@104 六架构全量重测**（SD1.5/SDXL/H-DiT/FLUX/SD3.5/PixArt-Σ，全部 104 图 0 失败）
- **Band 1 四变体 P-multi**：D_pp 全 0，D_mag ∈ [0.005, 0.023]——C1 核心证据（primary evidence）在最严协议下成立
- **诊断核查组 F1–F5（diagnostic verification suite）**：P-multi 噪声底 bootstrap、防污染核查（原始量级差 37×）、剖面形态检验（Spearman）、per-σ 时间步分解、SD3.5 峰位移复核
- **形态二分定稿**：内层局域型×3（SD1.5/SDXL/H-DiT）vs 末端累积型×3（FLUX/PixArt/SD3.5）
- C3 指纹收敛曲线 (eps+flow) · 随机 init · Skip 因果链 (SD1.5+SDXL) · TOST/BH/FDR · 度量审计 v1→v2 · 层序 canonical 化 · 协议审计（三协议 + 诊断核查）
- 跨协议形态稳定性表（3/5 稳定，SDXL 翻转已注释，FLUX P-t0 作废）

### 进行中 / 待补齐

| 项 | 状态 |
|----|------|
| 谱系测试（两形态类） | ✅ 斜坡类：PixArt-α（D_mag=0.096 可分辨）；内层峰类：SDXL-Turbo（D_mag=0.0022 不可区分） |
| SDXL 谱系测试（内层峰类第二谱系） | 排期 |
| Finding 2 英文版成稿 | 骨架已入大纲，动笔前等 PixArt-α |
| Fig.2 重画（形态二分 + 可分辨性极限） | 待 Finding 2 定稿 |
| HunyuanDiT 组件拆分 · MI shuffle · PERMANOVA | 排期 |
| VAE-swap | 降级为 rebuttal 备用 |
| 随机初始化重跑（P-multi 5 种子） | 排期 |

---

## 附录计划

A. 度量审计 v1→v2 (peak_count artifact 发现与修复) · A2. 层序审计 (字典序 bug, UNet topo key, canonical hash) · A3. 协议审计 (P-multi/P-t0/P-tT 历史登记表 + 协议选择准则) · A4. 峰型三分类与 right-censoring 标注 (双场景矩阵敏感性) · B. Bootstrap 噪声底分布 · C. 编辑 benchmark 逐类分布 · D. 预注册哈希承诺 · E. PixArt learned_sigma 验证 · F. P-t0 协议稳健性对照 (带内排序跨协议一致 / 类间分离度协议依赖披露)
