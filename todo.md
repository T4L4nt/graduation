# 毕业设计 TODO

## 第一阶段：基线验证（2026.5）✅ 已完成

- [x] 搭建最简实验环境（diffusers + lpips + arcface）
- [x] 选取测试图（升级为 COCO val2017 15张 + 人脸 2张）
- [x] 实现 DDIM Inversion → Reconstruction 最简脚本
- [x] 跑 4/10/20/50/100 步反演-重建，记录 PSNR / SSIM / LPIPS
- [x] 绘制步数-指标衰减曲线，复现内容漂移现象
- [x] 记录 UNet 38 层特征 L2 距离，定位漂移分布
- [x] 输出漂移热力图
- [x] 撰写诊断报告

## 第二阶段：内容稳定性增强（2026.6）✅ 已完成

- [x] 设计残差校正模块（零训练，f_out = f_recon + λ*(f_inv - f_recon)）
- [x] 确定校正注入层（top-5 up_blocks ResNet，消融验证）
- [x] DDIM / EDICT 基线对比
- [x] 新增 NTI 基线（Null-Text Inversion）
- [x] LPIPS / SSIM / ArcFace / DISTS 全指标评估
- [x] 推理耗时增量 < 5%
- [x] 提取共享模块 phase2_common.py（消除 ~300行/文件重复）
- [x] 并排对比可视化网格
- [x] 撰写实验报告（待更新 PDF）

## 第三阶段：风格解耦与控制（2026.6–2027.1）🔄 进行中

### 已完成
- [x] CLIP ViT-L/14 集成（openai/clip-vit-large-patch14）
- [x] 正交投影数学验证（5 图全 PASS，cos(v_style, v_content) ≈ 0）
- [x] face1 完整 pipeline 验证（PSNR 36.02, ArcFace 0.949）

### 核心缺口
- [ ] v_style 直接注入生成过程（Cross-attn / AdaIN / Feature offset）
- [ ] 正交钉扎约束（确保编辑时不漂移）
- [ ] 风格参考图支持（从图提取风格向量，替代预设标签）
- [ ] 三维控制接口的"方向"维度（风格空间球面插值）
- [ ] 在艺术风格数据集上验证（水墨、赛博朋克、油画）
- [ ] 定量评估：LPIPS / DISTS / CLIP 方向相似度
- [ ] 用户主观排序实验
- [ ] 撰写风格解耦实验报告

## 第四阶段：系统集成与对比（2027.2–2027.4）

- [ ] 整合内容保持 + 风格解耦模块为统一流程
- [ ] 人像/建筑/艺术三类场景 4-50 步全区间压力测试
- [ ] 与 Prompt-to-Prompt / ControlNet / IP-Adapter / TurboEdit 横向对比
- [ ] SDXL 跨模型泛化验证
- [ ] 撰写综合实验报告

## 第五阶段：论文与答辩（2027.4–2027.6）

- [ ] 撰写硕士论文初稿
- [ ] 论文修改与定稿
- [ ] 准备答辩 PPT 与演示系统
- [ ] 答辩（2027.6）
