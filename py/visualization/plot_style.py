"""论文绘图样式与导出模板。

统一字号、线宽、字体和导出规范；其余可视化模板都先调用
``set_paper_style`` 再画图，保证各模块输出图表风格一致。

标签默认使用英文字体；国赛论文需要中文标签时传 ``chinese=True``，
函数会自动从回退列表中挑选当前系统可用的中文字体。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 英文默认字体按顺序回退；DejaVu Sans 随 matplotlib 发行，必定可用。
ENGLISH_FONTS = ["DejaVu Sans", "Arial", "Helvetica"]

# 覆盖 macOS 与 Windows 的常见中文字体，实际使用时只保留系统已安装的。
CHINESE_FONTS = [
    "PingFang SC",       # macOS
    "Songti SC",         # macOS
    "Hiragino Sans GB",  # macOS
    "Microsoft YaHei",   # Windows
    "SimHei",            # Windows
    "STHeiti",           # macOS 旧版本
]

# 统一论文风格：字号、线宽、去多余边框、300 dpi 导出。
PAPER_STYLE = {
    "figure.figsize": (6.4, 4.0),
    "figure.dpi": 100.0,
    "savefig.dpi": 300.0,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 11.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "lines.markersize": 5.0,
    # 部分字体不含 Unicode 负号，交给系统连字符更稳妥。
    "axes.unicode_minus": False,
}


def _installed_fonts(candidates: list[str]) -> list[str]:
    """过滤出当前系统实际安装的字体，避免逐个触发缺字体警告。"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [name for name in candidates if name in installed]


def set_paper_style(chinese: bool = False) -> None:
    """应用统一的论文绘图风格。

    Args:
        chinese: 是否启用中文字体（国赛论文用；默认英文标签）。

    Raises:
        ValueError: 请求中文字体但系统未安装任何候选中文字体时抛出。
    """
    plt.rcParams.update(PAPER_STYLE)
    fonts = _installed_fonts(ENGLISH_FONTS)
    if chinese:
        chinese_fonts = _installed_fonts(CHINESE_FONTS)
        if not chinese_fonts:
            raise ValueError(
                "未找到可用中文字体，请安装 PingFang SC / Microsoft YaHei 等，"
                "或继续使用默认英文字体"
            )
        # 中文字体放前面，缺失的英文字符再由英文字体兜底。
        fonts = chinese_fonts + fonts
    plt.rcParams["font.sans-serif"] = fonts


def new_axes(figsize: tuple[float, float] | None = None):
    """创建应用当前风格的空坐标系。

    Args:
        figsize: 可选的 ``(宽, 高)`` 英寸尺寸。

    Returns:
        ``(fig, ax)``，绘图函数内部和直接画图都可以使用。
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig, name: str, *, pdf: bool = False) -> Path:
    """按论文规范保存图片：PNG 300 dpi，可选同时导出 PDF 矢量图。

    Args:
        fig: 已完成的 Figure 对象。
        name: 文件名（不含扩展名），保存到当前目录。
        pdf: 是否额外导出同名 PDF（论文排版推荐矢量图）。

    Returns:
        PNG 文件路径。
    """
    path = Path(f"{name}.png")
    fig.savefig(path)
    if pdf:
        fig.savefig(path.with_suffix(".pdf"))
    return path


def main() -> None:
    # ====== 比赛时主要替换下面这部分 ======
    import numpy as np

    set_paper_style(chinese=True)  # 英文论文保持默认 False

    fig, ax = new_axes()
    x = np.linspace(-2.0, 2.0, 100)
    # 曲线同时包含中文与负值，用于验证字体与负号设置。
    ax.plot(x, -x**2, label="测试曲线")
    ax.set_xlabel("自变量 x")
    ax.set_ylabel("因变量 y")
    ax.set_title("中文字体与负号显示测试")
    ax.legend()

    path = save_figure(fig, "plot_style_out", pdf=True)
    print("已保存 ->", path)

    # 本例的简单验收：替换题目后可删除或改写。
    assert plt.rcParams["savefig.dpi"] == 300.0
    assert plt.rcParams["axes.unicode_minus"] is False
    assert any("PingFang" in font or "SC" in font
               for font in plt.rcParams["font.sans-serif"])
    assert path.exists() and path.with_suffix(".pdf").exists()
    plt.close(fig)


if __name__ == "__main__":
    main()
