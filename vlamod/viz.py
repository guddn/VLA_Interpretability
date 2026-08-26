"""그림. matplotlib 기본값은 논문에 그대로 못 쓰므로 최소한의 규칙만 강제합니다.

원칙
----
- 색은 **계열(entity)** 에 고정. 시리즈가 빠져도 색이 재배정되지 않게 딕셔너리로 관리.
- 이중 y축 금지. 스케일이 다른 두 지표는 그림을 나눕니다.
- 3계열까지만 색으로 구분. 그 이상은 facet.
- 시리즈가 2개 이상이면 legend + 직접 라벨을 함께 답니다 (색만으로 식별 금지).
- 색맹 판별 검증 완료(OKLab CVD ΔE 9.2, 정상시야 24.0). 색을 바꾸려면 다시 검증하세요.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 검증된 3색 (blue / orange / aqua). 이 순서를 지키세요.
SERIES = {
    "R_raw": "#2a78d6",
    "R_norm": "#eb6834",
    "R_vnorm": "#1baf7a",
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def plot_layerwise(per_layer: dict[str, list], baseline: float | None, out: str,
                   title: str = "Language attention ratio by layer"):
    """x=layer, y=ratio. 3계열 라인."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160, facecolor=SURFACE)
    _style(ax)
    for name, color in SERIES.items():
        if name not in per_layer:
            continue
        y = per_layer[name]
        x = list(range(len(y)))
        ax.plot(x, y, color=color, linewidth=2.0, label=name)
        # 직접 라벨 (색만으로 식별하지 않도록)
        ax.annotate(name, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    if baseline is not None:
        ax.axhline(baseline, color=INK2, linewidth=1.2, linestyle="--")
        ax.annotate(f"uniform baseline {baseline:.3f}", (0, baseline), xytext=(4, 4),
                    textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel("layer")
    ax.set_ylabel("language ratio  R = L / (L + V)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_ratio_vs_causal(x, y, out: str, xlabel: str = "R_norm (attention)",
                         ylabel: str = "causal_ratio (KL-based)", rho: float | None = None):
    """H2 산점도. 한 계열이므로 legend 없음."""
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=160, facecolor=SURFACE)
    _style(ax)
    ax.scatter(x, y, s=26, color=SERIES["R_norm"], edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    t = "attention ratio vs causal contribution ratio"
    if rho is not None:
        t += f"   (Spearman ρ = {rho:.3f})"
    ax.set_title(t, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_conditions(labels, values, out: str, ylabel: str, title: str,
                    color: str = "#2a78d6"):
    """조건별 막대. 한 계열 → legend 없이 값 직접 라벨."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160, facecolor=SURFACE)
    _style(ax)
    bars = ax.bar(labels, values, color=color, width=0.58)
    # 막대 사이 2px 간격 효과 + 값 직접 라벨
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
