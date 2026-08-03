# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import CRS
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatter
import matplotlib as mpl

# ============================================================
# 0. 日本語フォント設定
# ============================================================

mpl.rcParams["font.family"] = "Hiragino Sans"
mpl.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 入力
# ============================================================

# Tucker分解で出力された建物因子
B_CSV = Path(
    "/Volumes/一ノ瀬/タッカー分解/出力/slot_colocation_tucker/tucker_B_building_factors.csv"
)

# 建物座標CSV
BUILDINGS_CSV = Path(
    "/Users/tsg/Desktop/Ichinose_work/05_共滞在データのエンリッチメント/"
    "05_data/02_広島県土地利用データ/Saijo500m_with_counts.csv"
)

# 出力先
OUT_DIR = Path(
    "/Volumes/一ノ瀬/タッカー分解/出力/タッカー分解2/"
    "visualization/building_factor_map"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 表示スケール設定
# ============================================================

SCALE = "log"          # "log" or "linear"
ROBUST = True
PCT_LOW = 1
PCT_HIGH = 99

# ============================================================
# 3. 読み込み
# ============================================================

Bf = pd.read_csv(B_CSV)
Saijo = pd.read_csv(BUILDINGS_CSV)

print("B因子ファイル列:")
print(Bf.columns.tolist())

print("建物CSV列:")
print(Saijo.columns.tolist())

# ============================================================
# 4. カラム確認
# ============================================================

# Tucker出力では index が unnamed で保存される場合があるため対応
if "label" not in Bf.columns:
    first_col = Bf.columns[0]
    Bf = Bf.rename(columns={first_col: "label"})

for need_col in ["建物名", "緯度", "経度"]:
    if need_col not in Saijo.columns:
        raise KeyError(f"建物CSVに '{need_col}' 列がありません。")

# ============================================================
# 5. 建物名でマージ
# ============================================================

Bf["label"] = Bf["label"].astype(str).str.strip()
Saijo["建物名"] = Saijo["建物名"].astype(str).str.strip()

merged = Bf.merge(
    Saijo,
    left_on="label",
    right_on="建物名",
    how="inner"
)

print(f"一致した建物数: {len(merged):,} / {len(Bf):,}")

if len(merged) == 0:
    raise RuntimeError(
        "建物名が一致しません。"
        "B因子のindexと建物CSVの建物名を確認してください。"
    )

# ============================================================
# 6. Tucker建物因子列の抽出
# ============================================================

factor_cols = [
    c for c in Bf.columns
    if c.startswith("B_comp")
]

if not factor_cols:
    factor_cols = [
        c for c in Bf.columns
        if c != "label" and np.issubdtype(Bf[c].dtype, np.number)
    ]

if not factor_cols:
    raise ValueError(
        "B因子CSVに B_comp 系の数値列が見つかりません。"
    )

print("可視化する建物因子列:")
print(factor_cols)

for c in factor_cols:
    merged[c] = pd.to_numeric(
        merged[c],
        errors="coerce"
    )

# ============================================================
# 7. GeoDataFrame化
# ============================================================

gdf = gpd.GeoDataFrame(
    merged,
    geometry=[
        Point(xy)
        for xy in zip(
            merged["経度"],
            merged["緯度"]
        )
    ],
    crs=CRS.from_epsg(4326)
).to_crs(epsg=3857)

# ============================================================
# 8. 描画範囲
# ============================================================

xmin, ymin, xmax, ymax = gdf.total_bounds

pad_x = (xmax - xmin) * 0.05
pad_y = (ymax - ymin) * 0.05

extent = [
    xmin - pad_x,
    xmax + pad_x,
    ymin - pad_y,
    ymax + pad_y,
]

# ============================================================
# 9. 正規化器
# ============================================================

def make_norm(
    values,
    scale="log",
    robust=True,
    pct_low=1,
    pct_high=99
):
    v = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(dtype=float)

    if scale == "log":
        pos = v[v > 0]

        if pos.size == 0:
            return None, None, None

        if robust:
            vmin = np.nanpercentile(
                pos,
                pct_low
            )
            vmax = np.nanpercentile(
                pos,
                pct_high
            )
        else:
            vmin = np.nanmin(pos)
            vmax = np.nanmax(pos)

        med = np.nanmedian(pos)

        eps = max(
            1e-12,
            (
                med
                if np.isfinite(med)
                else 1.0
            ) * 1e-6
        )

        v_plot = np.where(
            v > 0,
            v,
            eps
        )

        vmin = max(vmin, eps)

        if (
            not np.isfinite(vmax)
            or vmax <= vmin
        ):
            vmax = vmin * 10.0

        norm = LogNorm(
            vmin=vmin,
            vmax=vmax
        )

        return norm, v_plot, (vmin, vmax)

    else:
        if robust:
            vmin = np.nanpercentile(
                v,
                pct_low
            )
            vmax = np.nanpercentile(
                v,
                pct_high
            )
        else:
            vmin = np.nanmin(v)
            vmax = np.nanmax(v)

        if (
            not np.isfinite(vmin)
            or not np.isfinite(vmax)
        ):
            return None, None, None

        if vmax <= vmin:
            vmax = vmin + 1e-12

        norm = Normalize(
            vmin=vmin,
            vmax=vmax
        )

        return norm, v, (vmin, vmax)

# ============================================================
# 10. 建物因子散布図
# ============================================================

def plot_scatter_factor(
    gdf,
    value_col,
    out_path,
    scale="log"
):
    norm, vals, lims = make_norm(
        gdf[value_col].values,
        scale=scale,
        robust=ROBUST,
        pct_low=PCT_LOW,
        pct_high=PCT_HIGH,
    )

    used_scale = scale

    if norm is None or vals is None:
        if scale == "log":
            norm, vals, lims = make_norm(
                gdf[value_col].values,
                scale="linear",
                robust=ROBUST,
                pct_low=PCT_LOW,
                pct_high=PCT_HIGH,
            )

            used_scale = "linear"

        if norm is None or vals is None:
            print(
                f"[skip] {value_col}: "
                "有効な値がありません"
            )
            return

    fig, ax = plt.subplots(
        figsize=(8, 8),
        dpi=180
    )

    x = gdf.geometry.x.values
    y = gdf.geometry.y.values

    sc = ax.scatter(
        x,
        y,
        c=vals,
        norm=norm,
        cmap="plasma",
        s=45,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.2,
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik
    )

    ax.set_xlim(
        extent[0],
        extent[1]
    )

    ax.set_ylim(
        extent[2],
        extent[3]
    )

    ax.set_xticks([])
    ax.set_yticks([])

    title_scale = (
        "log10"
        if used_scale == "log"
        else "linear"
    )

    ax.set_title(
        f"Tucker建物因子スコア分布: "
        f"{value_col} ({title_scale})",
        fontname="Hiragino Sans",
    )

    cb = fig.colorbar(
        sc,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    if used_scale == "log":
        cb.formatter = LogFormatter(10)
        cb.set_label(
            "建物因子値（対数スケール）",
            fontname="Hiragino Sans"
        )
    else:
        cb.set_label(
            "建物因子値",
            fontname="Hiragino Sans"
        )

    cb.update_normal(sc)

    if (
        lims is not None
        and np.all(np.isfinite(lims))
    ):
        vmin, vmax = lims

        ax.text(
            0.01,
            0.02,
            f"range≈[{vmin:.3g}, {vmax:.3g}]",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="0.7",
                alpha=0.8,
            ),
        )

    fig.tight_layout()
    fig.savefig(
        out_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[OK] {out_path}")

# ============================================================
# 11. 出力
# ============================================================

for c in factor_cols:
    col_vals = merged[c].to_numpy()

    if (
        np.nan_to_num(
            col_vals,
            nan=0.0
        ).sum()
        == 0
    ):
        print(
            f"[skip] {c}: 全て0/NaN"
        )
        continue

    out_path = (
        OUT_DIR
        / f"scatter_tucker_building_{c}_{SCALE}.png"
    )

    plot_scatter_factor(
        gdf,
        c,
        out_path,
        scale=SCALE
    )

print(
    "完了：Tucker建物因子散布図を出力しました。"
)
