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

U_CSV = Path("/Volumes/一ノ瀬/タッカー分解/出力/slot_colocation_tucker/tucker_U_user_factors_with_attributes.csv"
)

OUT_DIR = Path(
    "/Volumes/一ノ瀬/タッカー分解/出力/タッカー分解2/"
    "visualization/user_home_hexmap"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 可視化パラメータ
# ============================================================

SCALE = "log"        # "log" or "linear"
ROBUST = True
P_LOW = 1
P_HIGH = 99
POINT_SIZE = 10

JPN_LON_RANGE = (122, 154)
JPN_LAT_RANGE = (20, 46)

# ============================================================
# 3. 正規化器
# ============================================================

def make_norm(
    vals,
    scale="log",
    robust=True,
    p_low=1,
    p_high=99,
    eps=1e-12
):
    vals = np.asarray(vals, dtype=float)
    v = vals[np.isfinite(vals)]

    if v.size == 0:
        return None

    if scale == "log":
        pos = v[v > 0]

        if pos.size == 0:
            return None

        vmin = (
            np.nanpercentile(pos, p_low)
            if robust
            else np.nanmin(pos)
        )

        vmax = (
            np.nanpercentile(pos, p_high)
            if robust
            else np.nanmax(pos)
        )

        vmin = max(vmin, eps)

        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 10.0

        return LogNorm(
            vmin=vmin,
            vmax=vmax
        )

    else:
        vmin = (
            np.nanpercentile(v, p_low)
            if robust
            else np.nanmin(v)
        )

        vmax = (
            np.nanpercentile(v, p_high)
            if robust
            else np.nanmax(v)
        )

        if (
            not np.isfinite(vmin)
            or not np.isfinite(vmax)
            or vmax <= vmin
        ):
            return None

        return Normalize(
            vmin=vmin,
            vmax=vmax
        )

# ============================================================
# 4. データ読み込み
# ============================================================

Uf = pd.read_csv(U_CSV)

print("U因子ファイル列:")
print(Uf.columns.tolist())

# index列対応
if "label" not in Uf.columns:
    first_col = Uf.columns[0]
    Uf = Uf.rename(columns={first_col: "label"})

required_cols = [
    "Home_Latitude",
    "Home_Longitude"
]

for c in required_cols:
    if c not in Uf.columns:
        raise KeyError(
            f"{c} がありません。"
            "tucker_U_user_factors_with_attributes.csv "
            "を使ってください。"
        )

# Tuckerユーザー因子列
factor_cols = [
    c for c in Uf.columns
    if c.startswith("U_comp")
]

if not factor_cols:
    raise ValueError(
        "U_comp 系の列が見つかりません。"
    )

for c in factor_cols:
    Uf[c] = pd.to_numeric(
        Uf[c],
        errors="coerce"
    )

Uf["Home_Latitude"] = pd.to_numeric(
    Uf["Home_Latitude"],
    errors="coerce"
)

Uf["Home_Longitude"] = pd.to_numeric(
    Uf["Home_Longitude"],
    errors="coerce"
)

# 欠損除外
merged = Uf.dropna(
    subset=[
        "Home_Latitude",
        "Home_Longitude"
    ]
).copy()

# 日本域フィルタ
merged = merged[
    merged["Home_Longitude"].between(
        *JPN_LON_RANGE
    )
    & merged["Home_Latitude"].between(
        *JPN_LAT_RANGE
    )
].copy()

print(
    f"ホーム座標ありユーザー数: "
    f"{len(merged):,} / {len(Uf):,}"
)

if len(merged) == 0:
    raise RuntimeError(
        "日本域に該当するホーム座標がありません。"
    )

# ============================================================
# 5. GeoDataFrame化
# ============================================================

gdf = gpd.GeoDataFrame(
    merged,
    geometry=[
        Point(xy)
        for xy in zip(
            merged["Home_Longitude"],
            merged["Home_Latitude"],
        )
    ],
    crs=CRS.from_epsg(4326),
).to_crs(epsg=3857)

# ============================================================
# 6. gridsize自動調整
# ============================================================

def auto_gridsize(n):
    return int(
        np.clip(
            np.sqrt(max(n, 1)) * 1.2,
            40,
            120
        )
    )

# ============================================================
# 7. ユーザー因子hexmap
# ============================================================

def plot_user_hexmap(
    gdf,
    value_col,
    out_path,
    scale=SCALE
):
    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()

    w = pd.to_numeric(
        gdf[value_col],
        errors="coerce"
    ).to_numpy(float)

    if np.nan_to_num(
        w,
        nan=0.0
    ).sum() == 0:
        print(
            f"[skip] {value_col}: 全て0/NaN"
        )
        return

    gs = auto_gridsize(len(gdf))

    fig, ax = plt.subplots(
        figsize=(10, 10),
        dpi=180
    )

    hb = ax.hexbin(
        x,
        y,
        C=w,
        reduce_C_function=np.sum,
        gridsize=gs,
        mincnt=1,
        cmap="plasma",
    )

    vals = hb.get_array()

    norm = make_norm(
        vals,
        scale=scale,
        robust=ROBUST,
        p_low=P_LOW,
        p_high=P_HIGH,
    )

    used_mode = "weighted"
    used_scale = scale

    if norm is None:
        norm = make_norm(
            vals,
            scale="linear",
            robust=ROBUST,
            p_low=P_LOW,
            p_high=P_HIGH,
        )
        used_scale = "linear"

    if (
        norm is None
        or vals.size == 0
        or np.nanmax(vals) <= 0
    ):
        hb = ax.hexbin(
            x,
            y,
            gridsize=gs,
            mincnt=1,
            cmap="plasma",
        )

        vals = hb.get_array()

        norm = make_norm(
            vals,
            scale=scale,
            robust=ROBUST,
            p_low=P_LOW,
            p_high=P_HIGH,
        )

        used_mode = "count"
        used_scale = scale

        if norm is None:
            norm = make_norm(
                vals,
                scale="linear",
                robust=ROBUST,
                p_low=P_LOW,
                p_high=P_HIGH,
            )
            used_scale = "linear"

    if norm is None:
        print(
            f"[skip] {value_col}: "
            "hexbinでも有効な値が得られません"
        )
        plt.close(fig)
        return

    hb.set_norm(norm)

    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik
    )

    xmin, ymin, xmax, ymax = gdf.total_bounds

    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05

    ax.set_xlim(
        xmin - pad_x,
        xmax + pad_x
    )

    ax.set_ylim(
        ymin - pad_y,
        ymax + pad_y
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        c="white",
        alpha=0.25,
        edgecolors="none",
    )

    tscale = (
        "log10"
        if used_scale == "log"
        else "linear"
    )

    ax.set_title(
        f"Tuckerユーザー因子ホーム分布: "
        f"{value_col} "
        f"({tscale}, {used_mode}, gridsize={gs})",
        fontname="Hiragino Sans",
    )

    cb = fig.colorbar(
        hb,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    if used_scale == "log":
        cb.formatter = LogFormatter(10)

        cb.set_label(
            f"{value_col} intensity"
            f"（対数スケール） | "
            f"mode={used_mode}",
            fontname="Hiragino Sans",
        )

        cb.update_normal(hb)

    else:
        cb.set_label(
            f"{value_col} intensity | "
            f"mode={used_mode}",
            fontname="Hiragino Sans",
        )

    fig.tight_layout()

    fig.savefig(
        out_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[OK] {out_path.name}  "
        f"(gridsize={gs}, "
        f"mode={used_mode}, "
        f"scale={used_scale})"
    )

# ============================================================
# 8. 出力
# ============================================================

for c in factor_cols:
    out_path = (
        OUT_DIR
        / f"hex_tucker_user_home_{c}_{SCALE}.png"
    )

    plot_user_hexmap(
        gdf,
        c,
        out_path,
        scale=SCALE
    )

print(
    "完了：Tuckerユーザー因子ホーム位置"
    "hexヒートマップを出力しました。"
)
