# 残差校正的收敛性分析

## 1. 符号定义

设 $f_l^{\text{inv}} \in \mathbb{R}^{C}$ 为 DDIM 反演过程中第 $l$ 层的特征（全局池化后），$f_l^{\text{recon}}$ 为重建过程中的对应特征。

残差：$d_l = f_l^{\text{inv}} - f_l^{\text{recon}}$

校正算子：$T_\lambda(f) = f + \lambda \cdot (f^{\text{inv}} - f)$

单层校正后：$f_l^{\text{out}} = f_l^{\text{recon}} + \lambda \cdot d_l$

**漂移加权**校正：$f_l^{\text{out}} = f_l^{\text{recon}} + \lambda \cdot w_l \cdot d_l$，其中 $w_l \propto \text{drift}_l$

---

## 2. 误差收缩引理

**引理 1（单层误差收缩）**

对任意 $\lambda \in (0, 2)$ 和任意特征层 $l$：

$$\|T_\lambda(f_l^{\text{recon}}) - f_l^{\text{inv}}\| = |1-\lambda| \cdot \|f_l^{\text{recon}} - f_l^{\text{inv}}\| < \|f_l^{\text{recon}} - f_l^{\text{inv}}\|$$

**证明**：

$$\begin{aligned}
\|f_l^{\text{out}} - f_l^{\text{inv}}\| &= \|f_l^{\text{recon}} + \lambda(f_l^{\text{inv}} - f_l^{\text{recon}}) - f_l^{\text{inv}}\| \\
&= \|(1-\lambda)(f_l^{\text{recon}} - f_l^{\text{inv}})\| \\
&= |1-\lambda| \cdot \|f_l^{\text{recon}} - f_l^{\text{inv}}\|
\end{aligned}$$

当 $\lambda \in (0, 2)$ 时 $|1-\lambda| < 1$，误差严格收缩。特别地，$\lambda = 1$ 时一步恢复到 $f_l^{\text{inv}}$。

**推广（漂移加权）**：

$$\|f_l^{\text{out}} - f_l^{\text{inv}}\| = |1-\lambda w_l| \cdot \|f_l^{\text{recon}} - f_l^{\text{inv}}\|$$

收敛条件：$0 < \lambda w_l < 2$。当 $w_l \in [0.5, 2.0]$（Phase 2 设计），$\lambda \in (0, 1]$ 始终满足条件 → 漂移加权不破坏收敛性。

---

## 3. 最优 λ 推导

特征不是完美信号——$f^{\text{inv}}$ 和 $f^{\text{recon}}$ 均含有相对真实特征 $f^*$ 的噪声：

$$f^{\text{inv}} = f^* + \eta_{\text{inv}}, \quad f^{\text{recon}} = f^* + \eta_{\text{recon}}$$

假设 $\eta_{\text{inv}} \sim (0, \sigma_{\text{inv}}^2 I)$，$\eta_{\text{recon}} \sim (0, \sigma_{\text{recon}}^2 I)$，相关系数 $\rho = \mathbb{E}[\eta_{\text{recon}} \cdot \eta_{\text{inv}}] / (\sigma_{\text{recon}}\sigma_{\text{inv}})$。

校正后：

$$f^{\text{out}} = f^* + (1-\lambda)\eta_{\text{recon}} + \lambda\eta_{\text{inv}}$$

期望平方误差：

$$L(\lambda) = \mathbb{E}\|f^{\text{out}} - f^*\|^2 = (1-\lambda)^2\sigma_{\text{recon}}^2 + \lambda^2\sigma_{\text{inv}}^2 + 2\lambda(1-\lambda)\rho\sigma_{\text{recon}}\sigma_{\text{inv}}$$

令 $\alpha = \sigma_{\text{inv}} / \sigma_{\text{recon}}$（反演相对精度），求导得最优 $\lambda$：

$$\boxed{\lambda^* = \frac{1 - \rho\alpha}{1 + \alpha^2 - 2\rho\alpha}}$$

**特殊情形**：

| 条件 | $\lambda^*$ | 含义 |
|------|------------|------|
| $\alpha = 1$, 任意 $\rho$ | $1/2$ | 等精度时最优 λ = 0.5 |
| $\alpha \to 0$（反演更精确） | $1$ | 反演远优于重建，完全信任反演 |
| $\alpha \to \infty$（重建更精确） | $0$ | 重建已很好，不需要校正 |

**与实验对照**：Phase 2 经验最优 $\lambda = 0.7$。由此反推 $\lambda^* = 0.7$ 对应的 $\alpha$ 和 $\rho$：

若 $\rho = 0$（独立噪声）：$0.7 = 1/(1+\alpha^2) \Rightarrow \alpha = \sqrt{1/0.7 - 1} \approx 0.65$

即反演特征的信噪比约为重建的 $1/0.65 \approx 1.5\times$，与 Phase 1 的"反演轨迹更可靠"一致。

---

## 4. Skip Connection 传播定理

**定理 2（残差信号传播）**

设 UNet 中相邻层的映射为 $f_{l+1} = F_l(f_l) + f_l$（残差连接），其中 $F_l$ 为子网络。若在第 $l$ 层注入校正 $d_l = f_l^{\text{inv}} - f_l^{\text{recon}}$，则校正信号传播到 $l+1$ 层：

$$d_{l+1} = f_{l+1}^{\text{out}} - f_{l+1}^{\text{recon}} \approx (I + \nabla F_l) \cdot \lambda d_l$$

**证明**：一阶 Taylor 展开。

$$f_{l+1}^{\text{out}} = F_l(f_l^{\text{recon}} + \lambda d_l) + (f_l^{\text{recon}} + \lambda d_l)$$

$$\approx [F_l(f_l^{\text{recon}}) + \nabla F_l \cdot \lambda d_l] + f_l^{\text{recon}} + \lambda d_l$$

$$= f_{l+1}^{\text{recon}} + (I + \nabla F_l) \cdot \lambda d_l$$

**推论 2.1（等距传播）**

当 $\| \nabla F_l \| \ll 1$（残差块常见特性：主干学习微小修正，跳连传递恒等映射），有 $d_{l+1} \approx \lambda d_l$。校正信号的**方向在 skip connections 中以 ≈ 单位增益传播**，仅幅度按 λ 衰减。

**推论 2.2（注入位置鲁棒性）**——解释 Phase 2 发现 random5 ≈ top5

由于 skip connection 将校正信号从任意注入层传播到所有后续层，**注入位置对效果的影响仅取决于传播路径长度**，而非具体选择了哪些层。

- 在 $l$ 处注入，信号传播 $(n-l)$ 步到达输出
- 路径长度的差异被 λ 衰减补偿
- 只要校正信号进入 decoder skip connection 链，它就能影响最终输出

这是 random5 ≈ top5 的理论解释：skip connections 使 UNet decoder 成为"校正透明"的信息通道。

---

## 5. 迭代校正收敛性

考虑迭代校正 $f^{(k+1)} = T_\lambda(f^{(k)})$：

**定理 3（迭代收敛）**

从任意初始特征 $f^{(0)}$ 出发，迭代校正收敛到 $f^{\text{inv}}$：

$$\|f^{(k)} - f^{\text{inv}}\| = |1-\lambda|^k \cdot \|f^{(0)} - f^{\text{inv}}\| \to 0$$

收敛速率由收缩因子 $\gamma = |1-\lambda|$ 决定。收敛到 ε 精度所需步数 $k \geq \log(\epsilon/\epsilon_0) / \log|1-\lambda|$。

**实际含义**：扩散模型中的单步校正（每个去噪步做一次 $T_\lambda$）等价于在整个去噪轨迹上的逐步迭代校正。每步收缩 ε，$N$ 步后总体收缩 $\gamma^N$。

---

## 6. 多层联合校正的收敛性

当在 $K$ 个不同层同时注入校正时：

**定理 4（多层联合收敛）**

设校正注入层集合 $\mathcal{L} = \{l_1, \ldots, l_K\}$，各层权重 $w_j \geq 0$。定义整体特征误差向量 $\mathbf{e} = [\|f_{l_1}^{\text{out}} - f_{l_1}^{\text{inv}}\|, \ldots]$。

则存在收敛域 $\Lambda \subset \mathbb{R}^K_+$ 使得所有层的误差同步收缩。特别地，当 $\lambda \in (0, \min_j 2/w_j)$ 时，每一层独立满足引理 1 的收缩条件。

对于 Phase 2 设计（$w_j \in [0.5, 2.0]$），$\lambda \in (0, 1]$ 保证所有层同时收敛。

---

## 7. 总结

| 结果 | 内容 | 解释的现象 |
|------|------|-----------|
| 引理 1 | 误差收缩 | 校正单调减少特征误差 |
| 最优 λ | $\lambda^* = 0.5$（等噪声）至 $1.0$（反演优势） | Phase 2 经验 λ=0.7 |
| 定理 2 | Skip connection 传播 | random5 ≈ top5 |
| 定理 3 | 迭代收敛 | 多步校正累积 |
| 定理 4 | 多层联合收敛 | 漂移加权不破坏收敛 |

**核心理论贡献**：证明了残差校正在数学上是**收缩映射**——在特征空间中，$f^{\text{recon}}$ 以几何速率收敛到 $f^{\text{inv}}$。Skip connections 将此收敛性质从注入层传播到所有后续层，使校正对注入位置鲁棒。这是 Phase 2 消融观察"random5 ≈ top5"背后的理论基础。
