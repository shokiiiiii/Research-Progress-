# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Hiragino Sans"
mpl.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 入力
# ============================================================

T_CSV = Path(
    "/Volumes/一ノ瀬/タッカー分解/出力/slot_colocation_tucker/tucker_T_144_time_factors.csv"
)

# 出力先を「タッカー分解2」に変更
OUT_DIR = Path(
    "/Volumes/一ノ瀬/タッカー分解/出力/タッカー分解2/"
    "visualization/time_factor_bar"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. データ読み込み
# ============================================================

Tf = pd.read_csv(T_CSV)

if "label" not in Tf.columns:
    first_col = Tf.columns[0]
    Tf = Tf.rename(columns={first_col: "label"})

factor_cols = [
    c for c in Tf.columns
    if c.startswith("T_comp")
]

if not factor_cols:
    raise ValueError("T_comp 系の列が見つかりません。")

for c in factor_cols:
    Tf[c] = pd.to_numeric(
        Tf[c],
        errors="coerce"
    )

# ============================================================
# 3. 時間因子の棒グラフ
# ============================================================

for c in factor_cols:
    fig, ax = plt.subplots(
        figsize=(14, 4),
        dpi=180
    )

    x = np.arange(len(Tf))
    y = Tf[c].to_numpy(float)

    ax.bar(x, y)

    ax.set_title(
        f"Tucker時間因子: {c}",
        fontname="Hiragino Sans"
    )

    ax.set_xlabel(
        "10分時間スロット",
        fontname="Hiragino Sans"
    )

    ax.set_ylabel(
        "因子値",
        fontname="Hiragino Sans"
    )

    # 1時間ごとにラベル表示
    tick_idx = np.arange(0, 144, 6)
    tick_labels = Tf["label"].iloc[tick_idx].tolist()

    ax.set_xticks(tick_idx)

    ax.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    out_path = (
        OUT_DIR
        / f"bar_tucker_time_{c}.png"
    )

    fig.savefig(
        out_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[OK] {out_path}")

print(
    "完了：Tucker時間因子の棒グラフを出力しました。"
)
