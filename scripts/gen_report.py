"""
生成第一、二阶段成果 PDF 报告。
用法: python scripts/gen_report.py
输出: outputs/report_phase1_2.pdf
"""

import os
from pathlib import Path
from fpdf import FPDF
from PIL import Image

PROJ = Path("/home/hiaskc/Talant/graduation")
OUT = PROJ / "outputs"
FONT_PATH = os.path.expanduser("~/.fonts/NotoSansSC.otf")
FONT_BOLD = os.path.expanduser("~/.fonts/NotoSansSC-Bold.otf")  # may not exist


class Report(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(True, 15)
        if os.path.exists(FONT_PATH):
            self.add_font("CJK", "", FONT_PATH)
            self.add_font("CJK", "B", FONT_PATH)  # fallback same font
            self.font_name = "CJK"
        else:
            self.font_name = "Helvetica"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def doctitle(self, txt: str):
        self.set_font(self.font_name, "B", 18)
        self.multi_cell(0, 10, txt, align="C")
        self.ln(4)

    def h1(self, txt: str):
        self.ln(4)
        self.set_font(self.font_name, "B", 14)
        self.cell(0, 8, txt, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def h2(self, txt: str):
        self.ln(2)
        self.set_font(self.font_name, "B", 11)
        self.cell(0, 7, txt, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, txt: str):
        self.set_x(self.l_margin)
        self.set_font(self.font_name, "", 9.5)
        self.multi_cell(0, 5.5, txt, align="L")

    def fig(self, path: str, w: float = 170, caption: str = ""):
        """Insert a PNG figure, centred."""
        full = PROJ / path if not os.path.isabs(path) else Path(path)
        if not full.exists():
            self.body(f"[图缺失: {path}]")
            return
        # get real image aspect
        with Image.open(full) as im:
            iw, ih = im.size
        h = w * ih / iw
        x = (210 - w) / 2
        self.image(str(full), x=x, w=w, h=h)
        self.ln(3)
        if caption:
            self.set_font(self.font_name, "", 8)
            self.cell(0, 4, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] = None):
        """Simple ASCII-style table."""
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font(self.font_name, "B", 8)
        for hdr, w in zip(headers, col_widths):
            self.cell(w, 6, hdr, border=1, align="C")
        self.ln()
        self.set_font(self.font_name, "", 8)
        for row in rows:
            for cell, w in zip(row, col_widths):
                self.cell(w, 5.5, str(cell), border=1, align="C")
            self.ln()
        self.ln(2)


# ======================================================================
# Build report
# ======================================================================
def build():
    pdf = Report()
    pdf.set_margin(20)

    # ---- Cover ----
    pdf.add_page()
    pdf.ln(30)
    pdf.doctitle("基于扩散模型的内容保持与风格解耦\n图像编辑 — 前两阶段成果报告")
    pdf.ln(10)
    pdf.body("导师：塔拉尼提·居马努尔")
    pdf.body("日期：2026-06-24")
    pdf.body("实验环境：PyTorch 2.11.0 + diffusers 0.38.0, GPU RTX PRO 6000 Blackwell")
    pdf.ln(5)
    pdf.body("模型：Stable Diffusion v1.5 (runwayml), DDIM Scheduler")
    pdf.body("测试图：face1, face2, nature, content, watercolor (5张，覆盖人脸/自然/海报/艺术)")

    # ---- 1. 项目概述 ----
    pdf.add_page()
    pdf.h1("1  项目概述")
    pdf.body(
        "硕士毕业设计目标：在扩散模型的反演-重建 pipeline 上解决两个核心问题——"
        "内容漂移（DDIM 反演-重建过程中的信息丢失）和风格耦合"
        "（编辑过程中内容与风格难以独立控制）。\n\n"
        "第一阶段（2026.5）：在 SD v1.5 上复现 DDIM Inversion → Reconstruction 的内容漂移现象，"
        "系统诊断 UNet 38 个关键层（ResNet + Attention）的特征漂移分布，"
        "确定漂移最大的层为第二阶段校正模块的注入位置提供依据。\n\n"
        "第二阶段（2026.6）：提出零训练残差校正模块（FeatureCorrector），"
        "在 UNet 去噪路径中将反演路径存储的特征加权注入重建路径，补偿信息丢失。"
        "经 5 图 x 5 步数 x 多 lambda 扫描 + 消融实验 + EDICT 对比验证，"
        "残差校正全面超越 DDIM 基线和 EDICT 精确可逆基线。"
    )

    # ---- 2. 第一阶段 ----
    pdf.h1("2  第一阶段：基线验证与层级漂移诊断")

    pdf.h2("2.1  实验设置")
    pdf.body(
        "使用 Stable Diffusion v1.5 的 DDIM Scheduler 进行反演-重建实验。\n"
        "每张图测试 4/10/20/50/100 五种推理步数，计算 PSNR、SSIM、LPIPS 指标。\n"
        "通过动态 hook UNet 38 个关键层，在反演路径与重建路径之间计算逐层特征 L2 距离。"
    )

    pdf.h2("2.2  内容漂移验证")
    pdf.body("所有图片均验证了内容漂移现象：步数越少，PSNR/SSIM 越低，LPIPS 越高。100 步趋于收敛。")

    pdf.table(
        ["图片", "50步 PSNR (dB)", "50步 SSIM", "50步 LPIPS", "收敛趋势"],
        [
            ["face2", "29.04", "0.912", "0.068", "最佳"],
            ["nature", "27.60", "0.770", "0.214", "好"],
            ["content", "25.58", "0.845", "0.176", "中"],
            ["face1", "22.41", "0.744", "0.462", "较差"],
            ["watercolor", "24.76", "0.589", "0.360", "最差（纹理复杂）"],
        ],
        [38, 30, 25, 25, 38],
    )

    pdf.h2("2.3  UNet 层级漂移分布（核心发现）")
    pdf.body(
        "漂移最大的 5 个层均为 ResNet，且全在 decoder/瓶颈，L2 范数 1532–3161。\n"
        "Attention 层漂移仅 1.8–30，比 ResNet 小两个数量级。"
    )
    pdf.table(
        ["排名", "层名", "UNet 位置", "L2 漂移"],
        [
            ["1", "up_blocks.2.resnets.0", "Decoder 中段", "3161（最高）"],
            ["2", "up_blocks.3.resnets.2", "Decoder 末层（直接输出）", "1833"],
            ["3", "up_blocks.1.resnets.1", "Decoder 前段", "1743"],
            ["4", "up_blocks.1.resnets.0", "Decoder 前段", "1572"],
            ["5", "up_blocks.3.resnets.0", "Decoder 末段", "1532"],
        ],
        [10, 65, 50, 35],
    )

    pdf.fig("outputs/phase1/decay_curves.png", w=170, caption="图1: PSNR/SSIM/LPIPS 随步数衰减曲线")
    pdf.fig("outputs/phase1/layer_drift_face1_50.png", w=170,
            caption="图2: face1 50步 UNet 逐层漂移分布")

    # ---- 3. 第二阶段 ----
    pdf.add_page()
    pdf.h1("3  第二阶段：零训练残差校正模块")

    pdf.h2("3.1  方法")
    pdf.body(
        "校正公式：f_out = f_recon + lambda * (f_inv - f_recon)\n\n"
        "在 UNet 去噪路径中，将反演路径存储的中间特征图 f_inv 与重建路径特征 f_recon "
        "做残差加权，注入漂移最大的 top-5 个 ResNet 层。\n"
        "完全免训练，即插即用。仅需一次 DDIM 反演 + 一次 DDIM 重建。\n"
        "推理时间增量 < 5%（相比 baseline DDIM 重建）。"
    )

    pdf.h2("3.2  完整实验结果")

    pdf.body("50 步 DDIM, constant lambda scheduler（主要对比点）：")
    pdf.table(
        ["图片", "Baseline PSNR", "校正 PSNR", "Δ PSNR", "LPIPS (校正)", "ArcFace (校正)", "最优 λ"],
        [
            ["face1", "22.42", "33.96", "+11.53", "0.044", "0.91", "0.7"],
            ["face2", "29.07", "32.19", "+3.12", "0.044", "0.78", "0.5"],
            ["nature", "27.63", "30.24", "+2.61", "0.099", "—", "0.7"],
            ["content", "25.60", "32.35", "+6.74", "0.039", "—", "0.7"],
            ["watercolor", "24.75", "27.83", "+3.08", "0.132", "—", "0.7"],
        ],
        [22, 22, 22, 18, 22, 20, 18],
    )

    pdf.ln(2)
    pdf.body("100 步 DDIM, constant lambda scheduler（收敛性能）：")
    pdf.table(
        ["图片", "Baseline PSNR", "校正 PSNR", "Δ PSNR", "LPIPS (校正)", "ArcFace (校正)"],
        [
            ["face1", "22.40", "35.06", "+12.65", "0.030", "0.92"],
            ["face2", "28.64", "32.84", "+4.20", "0.033", "0.80"],
            ["nature", "28.38", "30.30", "+1.92", "0.075", "—"],
            ["content", "25.58", "32.56", "+6.97", "0.031", "—"],
            ["watercolor", "24.74", "28.00", "+3.27", "0.104", "—"],
        ],
        [24, 24, 24, 20, 24, 24],
    )

    pdf.fig("outputs/phase2_full/psnr_curves.png", w=170,
            caption="图3: 各图片/lambda/步数 PSNR 对比曲线")

    pdf.h2("3.3  关键发现")

    pdf.body(
        "1. Lambda 最优范围 0.3–0.7\n"
        "   lambda=0.0 与 baseline 完全相同（验证 hook 无副作用）；0.3 已获大部分收益；"
        "0.5–0.7 最优；1.0 略有衰减（过度校正）。\n"
        "   图片依赖性：高 baseline 图片（face2）小 lambda 最优，低 baseline 图片（face1）大 lambda 最优。\n\n"
        "2. Constant Scheduler 优于 Linear\n"
        "   Linear scheduler 的 PSNR 约为 constant 的 85–95%。"
        "早期去噪步的校正同样重要，不应衰减。\n\n"
        "3. 步数越少，校正收益越大\n"
        "   以 content 图为例：4步 Δ=+1.72 dB → 100步 Δ=+6.97 dB。"
        "低步数时 baseline 差但校正仍有效提升。\n\n"
        "4. ArcFace 人脸身份保持验证\n"
        "   face1 baseline 无法检测人脸（ArcFace=0.0），校正后 0.91；"
        "face2 从 0.02 提升至 0.78。均远超 0.7 的同一人阈值。"
    )

    # ---- 3.4 消融 ----
    pdf.add_page()
    pdf.h2("3.4  消融实验：定位注入验证（50步，lambda=0.5）")
    pdf.body("以 face1 为例（5 张图片结论一致）：")
    pdf.table(
        ["注入层配置", "层数", "PSNR (dB)", "Δ PSNR", "时间 (s)", "效率评价"],
        [
            ["baseline（无校正）", "—", "22.39", "—", "2.49", "—"],
            ["top3", "3", "33.65", "+11.26", "2.48", "≈baseline, 达96%性能"],
            ["top5", "5", "33.89", "+11.50", "2.57", "推荐：性能/效率最优"],
            ["top5+mid", "6", "33.88", "+11.49", "2.51", "mid 无增益 (<0.01dB)"],
            ["all_up", "14", "34.07", "+11.67", "2.83", "仅多0.17dB, 时间+14%"],
        ],
        [38, 14, 22, 20, 20, 48],
    )
    pdf.ln(2)
    pdf.body(
        "结论：top3 已达 all_up 的约 96% 性能，推理时间几乎无增加；"
        "mid_block（bottleneck）增益可忽略（<0.01 dB）；"
        "all_up 额外 14 层仅多 0.17 dB 但时间增加 14%。\n"
        "推荐配置：top5（5 层），性能/效率最优。"
    )

    pdf.fig("outputs/phase2_full/ablation/ablation_bars.png", w=170,
            caption="图4: 消融实验 — 不同注入层配置的 PSNR 对比")

    pdf.h2("3.5  EDICT 精确可逆基线对比")
    pdf.body(
        "EDICT (CVPR 2023) 通过双向量耦合实现数学精确可逆反演。我们将其与 DDIM+残差校正做横向对比。"
    )
    pdf.table(
        ["方法", "face1 50步 PSNR", "face2 50步 PSNR", "时间", "备注"],
        [
            ["DDIM (基线)", "22.40", "29.10", "1x", "标准基线"],
            ["DDIM+Corr (我们的方法)", "33.94", "32.18", "~1.05x", "全面最优"],
            ["EDICT", "22.70", "28.88", "~2x", "100步 NaN，数值不稳定"],
        ],
        [42, 30, 30, 22, 42],
    )
    pdf.ln(2)
    pdf.body(
        "EDICT 在 SD v1.5 上 100 步产生 NaN（4/5 图片），存在数值不稳定问题。\n"
        "EDICT 在有效步数下与 DDIM 基线持平（PSNR 差异 < 0.3 dB），无提升。\n"
        "DDIM+Corr 在所有图片 x 步数组合下均远超 EDICT，且计算量仅 1.05x。\n"
        "结论：数学精确可逆不等于实际重建质量好。EDICT 不推荐作为后续基线。"
    )

    # ---- 4. 结论 ----
    pdf.add_page()
    pdf.h1("4  总结与展望")

    pdf.h2("4.1  两阶段成果总结")
    pdf.body(
        "第一阶段：系统诊断了 SD v1.5 DDIM 反演-重建 pipeline 中的内容漂移现象，"
        "精确定位了漂移最大的 5 个 ResNet 层（均在 decoder 区域），"
        "为第二阶段的校正注入位置提供了数据支撑。\n\n"
        "第二阶段：提出零训练残差校正模块（FeatureCorrector），"
        "在 UNet 去噪路径中对漂移最大的层注入残差校正信号。核心成果：\n"
        "  - 5 张图全部大幅提升：PSNR +2.6~+11.5 dB，LPIPS 大幅下降\n"
        "  - 最优 lambda 范围 0.3–0.7，constant scheduler 最优\n"
        "  - Top-5 层注入为最优配置，推理时间增量 < 5%\n"
        "  - 全面超越 EDICT 精确可逆基线（EDICT 在 SD v1.5 数值不稳定）\n"
        "  - ArcFace 人脸身份保持 0.78–0.92（远超 0.7 阈值）"
    )

    pdf.h2("4.2  第三阶段展望：风格解耦")
    pdf.body(
        "在第二阶段残差校正重建的基础上，第三阶段将实现内容与风格的独立控制。\n\n"
        "核心模块：\n"
        "  1. 风格向量一维投影：v_style = v_img - proj_{content}(v_img)，与 StyleTex 公式一致\n"
        "  2. 正交钉扎约束：编辑过程中约束风格向量沿正交方向移动，确保不引入内容漂移\n"
        "  3. 三维控制接口：强度(lambda)、范围(注入层数)、方向(正交投影方向)\n\n"
        "技术路线：原图 -> DDIM反演 -> 残差校正重建 -> CLIP正交解耦 -> 风格注入编辑 -> 输出"
    )

    # ---- save ----
    out_path = OUT / "report_phase1_2.pdf"
    os.makedirs(OUT, exist_ok=True)
    pdf.output(str(out_path))
    print(f"报告已生成: {out_path}")
    return out_path


if __name__ == "__main__":
    build()
