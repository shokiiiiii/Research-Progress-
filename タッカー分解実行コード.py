# ============================================================
# スロット型共滞在を用いた非負値Tucker分解
#
# 共滞在の定義:
#   同じ日・同じ建物・同じ10分スロットに、
#   8分以上滞在した異なるユーザーが2人以上いる場合、
#   そのユーザー全員を共滞在参加者とする。
#
# Tensor:
#   TimeSlot144 × User × Building
#
# X[t, u, b]:
#   ユーザーuが時間スロットt・建物bで
#   スロット型共滞在に参加した日数
# ============================================================

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import non_negative_tucker


# ============================================================
# 0. 入出力設定
# ============================================================

INPUT_PARQUET = Path(
    "/Volumes/一ノ瀬/タッカー分解/ファイル/"
    "東広島滞在データwith_building.parquet"
)

INPUT_NAMES_CSV = Path(
    "/Users/tsg/Desktop/Ichinose_work/"
    "05_共滞在データのエンリッチメント/"
    "05_data/02_広島県土地利用データ/"
    "Saijo500m_with_counts.csv"
)

OUT_DIR = Path(
    "/Volumes/一ノ瀬/タッカー分解/出力/"
    "slot_colocation_tucker"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. カラム・分析条件
# ============================================================

USER_COL = "User_Id"
START_COL = "Time_start"
END_COL = "Time_end"
BUILDING_COL = "building_name"
TRIP_MODE_COL = "TripMode"

SLOT_MINUTES = 10
MIN_OVERLAP_MINUTES = 8
MIN_USERS_PER_GROUP = 2

SLOT_FREQ = f"{SLOT_MINUTES}min"
SLOT_WIDTH = pd.Timedelta(minutes=SLOT_MINUTES)
MIN_OVERLAP = pd.Timedelta(minutes=MIN_OVERLAP_MINUTES)


# 軸順：時間 × ユーザー × 建物
TUCKER_RANK = (
    6,  # 時間成分数
    8,  # ユーザー成分数
    6,  # 建物成分数
)

N_ITER_MAX = 300
TOL = 1e-6
RANDOM_STATE = 0

tl.set_backend("numpy")


# ============================================================
# 2. 補助関数
# ============================================================

def try_read_csv(path: Path) -> pd.DataFrame:
    """文字コードを切り替えながらCSVを読み込む。"""

    encodings = [
        "utf-8-sig",
        "cp932",
        "shift_jis",
        "utf-8",
    ]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def normalize_name(series: pd.Series) -> pd.Series:
    """建物名の空白・表記揺れを軽く補正する。"""

    return (
        series.astype("string")
        .fillna("")
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def make_time_labels() -> list[str]:
    """00:00～23:50の144ラベルを作成する。"""

    return [
        f"{hour:02d}:{minute:02d}"
        for hour in range(24)
        for minute in range(0, 60, SLOT_MINUTES)
    ]


def normalize_columns_l1(
    matrix: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """各因子列の合計を1に正規化する。"""

    matrix = np.asarray(matrix, dtype=np.float64)
    column_sum = matrix.sum(axis=0, keepdims=True)

    return matrix / np.maximum(column_sum, eps)


def top_k(
    labels: list,
    weights: np.ndarray,
    k: int = 10,
) -> list[list]:
    """因子値の大きい要素を上位k件取得する。"""

    weights = np.asarray(weights, dtype=float).ravel()
    indices = np.argsort(-weights)[:k]

    return [
        [str(labels[index]), float(weights[index])]
        for index in indices
    ]


def check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """必要なカラムが存在するか確認する。"""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"必要なカラムがありません: {missing}\n"
            f"実際のカラム: {df.columns.tolist()}"
        )


# ============================================================
# 3. 対象建物名を読み込む
# ============================================================

names_df = try_read_csv(INPUT_NAMES_CSV)

name_candidates = [
    column
    for column in names_df.columns
    if any(
        keyword in column.lower()
        for keyword in [
            "建物",
            "名称",
            "施設",
            "name",
        ]
    )
]

if not name_candidates:
    raise KeyError(
        "対象建物CSVに建物名列が見つかりません。"
    )

NAMES_COL = name_candidates[0]

names_df["_building_key"] = normalize_name(
    names_df[NAMES_COL]
)

valid_building_names = set(
    names_df.loc[
        names_df["_building_key"].ne(""),
        "_building_key",
    ].unique()
)

print("=" * 70)
print("対象建物")
print("=" * 70)
print(f"建物名列: {NAMES_COL}")
print(f"対象建物名数: {len(valid_building_names):,}")


# ============================================================
# 4. 滞在データを読み込む
# ============================================================

df = pd.read_parquet(INPUT_PARQUET)

required_columns = [
    USER_COL,
    START_COL,
    END_COL,
    BUILDING_COL,
]

check_required_columns(df, required_columns)

print("\n" + "=" * 70)
print("入力データ")
print("=" * 70)
print(f"読み込み行数: {len(df):,}")
print(f"カラム数: {df.shape[1]:,}")


# ============================================================
# 5. 必要カラムを保持
# ============================================================

attribute_columns = [
    "gender",
    "age_group",
    "device_os",
    "device_model",
    "device_os_version",
    "device_carrier",
    "Home_Latitude",
    "Home_Longitude",
    "Office_Latitude",
    "Office_Longitude",
]

optional_columns = [
    "Trip_Id",
    TRIP_MODE_COL,
    "Latitude_start",
    "Latitude_end",
    "Longitude_start",
    "Longitude_end",
    "Latitude_center",
    "Longitude_center",
    "building_id",
    "source_file",
] + attribute_columns

use_columns = required_columns + [
    column
    for column in optional_columns
    if column in df.columns
]

use_columns = list(dict.fromkeys(use_columns))

df = df[use_columns].copy()


# ============================================================
# 6. activity滞在のみ抽出・クリーニング
# ============================================================

df[START_COL] = pd.to_datetime(
    df[START_COL],
    errors="coerce",
)

df[END_COL] = pd.to_datetime(
    df[END_COL],
    errors="coerce",
)

df[BUILDING_COL] = normalize_name(
    df[BUILDING_COL]
)

df = df.dropna(
    subset=[
        USER_COL,
        START_COL,
        END_COL,
    ]
).copy()

df = df[
    df[END_COL].gt(df[START_COL])
    & df[BUILDING_COL].ne("")
].copy()


# activityだけを残す
if TRIP_MODE_COL in df.columns:

    before = len(df)

    df = df[
        df[TRIP_MODE_COL]
        .astype("string")
        .str.lower()
        .eq("activity")
    ].copy()

    print("\nactivity抽出")
    print(f"抽出前: {before:,}")
    print(f"抽出後: {len(df):,}")


# 対象建物だけを残す
before = len(df)

df = df[
    df[BUILDING_COL].isin(valid_building_names)
].copy()

print("\n対象建物フィルタ")
print(f"フィルタ前: {before:,}")
print(f"フィルタ後: {len(df):,}")


# 完全に同じ滞在レコードの重複を除去
dedup_columns = [
    USER_COL,
    BUILDING_COL,
    START_COL,
    END_COL,
]

before = len(df)

df = df.drop_duplicates(
    subset=dedup_columns
).copy()

print("\n滞在レコードの重複除去")
print(f"除去前: {before:,}")
print(f"除去後: {len(df):,}")
print(f"ユーザー数: {df[USER_COL].nunique():,}")
print(f"建物数: {df[BUILDING_COL].nunique():,}")


if df.empty:
    raise ValueError(
        "条件を満たすactivity滞在がありません。"
    )


# ============================================================
# 7. 滞在を10分スロット候補へ展開
# ============================================================

df["_first_slot"] = df[START_COL].dt.floor(
    SLOT_FREQ
)

# 終了時刻ちょうどの次スロットを含めない
df["_last_slot"] = (
    df[END_COL] - pd.Timedelta("1ns")
).dt.floor(SLOT_FREQ)

df["_n_slots"] = (
    (
        df["_last_slot"] - df["_first_slot"]
    ) / SLOT_WIDTH
).astype(np.int64) + 1

df = df[
    df["_n_slots"].gt(0)
].copy()

print("\n" + "=" * 70)
print("10分スロットへの展開")
print("=" * 70)
print(f"展開前滞在数: {len(df):,}")
print(f"展開後の想定行数: {df['_n_slots'].sum():,}")
print(f"最大スロット数: {df['_n_slots'].max():,}")


df["_slot_offset"] = df["_n_slots"].apply(
    lambda n: np.arange(
        n,
        dtype=np.int32,
    )
)

slot_df = df.explode(
    "_slot_offset",
    ignore_index=True,
)

slot_df["_slot_offset"] = (
    slot_df["_slot_offset"]
    .astype(np.int32)
)

slot_df["slot_start"] = (
    slot_df["_first_slot"]
    + slot_df["_slot_offset"] * SLOT_WIDTH
)

slot_df["slot_end"] = (
    slot_df["slot_start"] + SLOT_WIDTH
)

print(f"実際の展開後行数: {len(slot_df):,}")


# ============================================================
# 8. 各スロットとの重なり時間を計算
# ============================================================

overlap_start = pd.concat(
    [
        slot_df[START_COL],
        slot_df["slot_start"],
    ],
    axis=1,
).max(axis=1)

overlap_end = pd.concat(
    [
        slot_df[END_COL],
        slot_df["slot_end"],
    ],
    axis=1,
).min(axis=1)

slot_df["overlap"] = (
    overlap_end - overlap_start
)

slot_df["overlap_minutes"] = (
    slot_df["overlap"].dt.total_seconds() / 60
)

slot_df = slot_df[
    slot_df["overlap"].ge(MIN_OVERLAP)
].copy()

print("\n8分以上重なるスロット")
print(f"採用行数: {len(slot_df):,}")
print(f"ユーザー数: {slot_df[USER_COL].nunique():,}")
print(f"建物数: {slot_df[BUILDING_COL].nunique():,}")


if slot_df.empty:
    raise ValueError(
        "8分以上重なる10分スロットがありません。"
    )


# ============================================================
# 9. 日付・時間スロットを作成
# ============================================================

# 共滞在判定には日付を残す
slot_df["date"] = (
    slot_df["slot_start"].dt.date
)

slot_df["time_slot_id"] = (
    slot_df["slot_start"].dt.hour * 6
    + slot_df["slot_start"].dt.minute // 10
).astype(np.int16)

slot_df["time_slot_label"] = (
    slot_df["slot_start"].dt.strftime("%H:%M")
)


# ============================================================
# 10. 同一ユーザーの重複を除去
# ============================================================

# 同じ日・建物・スロットに同一ユーザーの複数レコードが
# 存在しても1人として数える
slot_user_columns = [
    "date",
    BUILDING_COL,
    "time_slot_id",
    USER_COL,
]

before = len(slot_df)

slot_df = slot_df.drop_duplicates(
    subset=slot_user_columns
).copy()

print("\n日・建物・スロット・ユーザー単位の重複除去")
print(f"除去前: {before:,}")
print(f"除去後: {len(slot_df):,}")


# ============================================================
# 11. スロット型共滞在を判定
# ============================================================

group_columns = [
    "date",
    BUILDING_COL,
    "time_slot_id",
]

slot_df["colocation_group_size"] = (
    slot_df
    .groupby(group_columns)[USER_COL]
    .transform("nunique")
)

colocation_df = slot_df[
    slot_df["colocation_group_size"]
    .ge(MIN_USERS_PER_GROUP)
].copy()


print("\n" + "=" * 70)
print("スロット型共滞在の判定結果")
print("=" * 70)

print(f"共滞在参加レコード数: {len(colocation_df):,}")
print(
    "共滞在参加ユーザー数: "
    f"{colocation_df[USER_COL].nunique():,}"
)
print(
    "共滞在発生建物数: "
    f"{colocation_df[BUILDING_COL].nunique():,}"
)

if colocation_df.empty:
    raise ValueError(
        "スロット型共滞在が検出されませんでした。"
    )


# 共滞在グループ数
group_summary = (
    colocation_df[
        group_columns
        + ["colocation_group_size"]
    ]
    .drop_duplicates(
        subset=group_columns
    )
    .copy()
)

print(
    "共滞在グループ数: "
    f"{len(group_summary):,}"
)

print(
    "共滞在グループ平均人数: "
    f"{group_summary['colocation_group_size'].mean():.3f}"
)

print(
    "共滞在グループ中央値: "
    f"{group_summary['colocation_group_size'].median():.3f}"
)

print(
    "最大共滞在人数: "
    f"{group_summary['colocation_group_size'].max():,}"
)


# ============================================================
# 12. 日付を除き、144スロットへ集約
# ============================================================

time_labels = make_time_labels()

user_labels = sorted(
    colocation_df[USER_COL]
    .astype(str)
    .unique()
    .tolist()
)

building_labels = sorted(
    colocation_df[BUILDING_COL]
    .astype(str)
    .unique()
    .tolist()
)

user_to_index = {
    user: index
    for index, user in enumerate(user_labels)
}

building_to_index = {
    building: index
    for index, building in enumerate(building_labels)
}

colocation_df["_user_index"] = (
    colocation_df[USER_COL]
    .astype(str)
    .map(user_to_index)
    .astype(np.int32)
)

colocation_df["_building_index"] = (
    colocation_df[BUILDING_COL]
    .astype(str)
    .map(building_to_index)
    .astype(np.int32)
)


T = 144
U = len(user_labels)
B = len(building_labels)

print("\n" + "=" * 70)
print("最終テンソル次元")
print("=" * 70)
print(f"時間スロット数: {T:,}")
print(f"ユーザー数: {U:,}")
print(f"建物数: {B:,}")
print(f"X.shape = ({T:,}, {U:,}, {B:,})")

print(
    "float32推定メモリ: "
    f"{T * U * B * 4 / 1e9:.3f} GB"
)


# ============================================================
# 13. 共滞在テンソルを作成
# ============================================================

X = np.zeros(
    shape=(T, U, B),
    dtype=np.float32,
)

t_indices = (
    colocation_df["time_slot_id"]
    .to_numpy(dtype=np.int64)
)

u_indices = (
    colocation_df["_user_index"]
    .to_numpy(dtype=np.int64)
)

b_indices = (
    colocation_df["_building_index"]
    .to_numpy(dtype=np.int64)
)


# 1行は、
# ある日・建物・時間スロットにおける
# 1ユーザーの共滞在参加を表す
#
# 日付を除いて加算するため、
# X[t,u,b]は共滞在に参加した日数となる
np.add.at(
    X,
    (
        t_indices,
        u_indices,
        b_indices,
    ),
    1.0,
)


nnz = np.count_nonzero(X)

print("\n共滞在テンソル")
print(f"形状: {X.shape}")
print(f"非ゼロ要素数: {nnz:,}")
print(f"共滞在参加ユーザー日数の合計: {X.sum():,.0f}")
print(f"1要素の最大値: {X.max():,.0f}")
print(f"密度: {nnz / X.size:.10f}")


# ============================================================
# 14. 分解ランクを確認
# ============================================================

if TUCKER_RANK[0] > T:
    raise ValueError(
        "時間成分数が時間スロット数を超えています。"
    )

if TUCKER_RANK[1] > U:
    raise ValueError(
        "ユーザー成分数がユーザー数を超えています。"
    )

if TUCKER_RANK[2] > B:
    raise ValueError(
        "建物成分数が建物数を超えています。"
    )


# ============================================================
# 15. 非負値Tucker分解
# ============================================================

print("\n" + "=" * 70)
print("非負値Tucker分解を開始")
print("=" * 70)

result = non_negative_tucker(
    X,
    rank=TUCKER_RANK,
    n_iter_max=N_ITER_MAX,
    tol=TOL,
    init="random",
    random_state=RANDOM_STATE,
    verbose=True,
)


# TensorLyのバージョン差に対応
errors = None

if hasattr(result, "core") and hasattr(result, "factors"):

    core = result.core
    factors = result.factors

elif isinstance(result, tuple):

    if len(result) == 3:
        core, factors, errors = result

    elif len(result) == 2:
        core, factors = result

    else:
        raise ValueError(
            "non_negative_tuckerの戻り値が想定外です。"
        )

else:
    raise TypeError(
        "non_negative_tuckerの戻り値形式を認識できません。"
    )


time_factor, user_factor, building_factor = factors

core = np.asarray(
    core,
    dtype=np.float32,
)

time_factor = np.asarray(
    time_factor,
    dtype=np.float32,
)

user_factor = np.asarray(
    user_factor,
    dtype=np.float32,
)

building_factor = np.asarray(
    building_factor,
    dtype=np.float32,
)


print("\nTucker分解完了")
print(f"core.shape: {core.shape}")
print(f"time_factor.shape: {time_factor.shape}")
print(f"user_factor.shape: {user_factor.shape}")
print(f"building_factor.shape: {building_factor.shape}")

if errors is not None and len(errors) > 0:
    print(f"反復回数: {len(errors):,}")
    print(f"最終誤差: {float(errors[-1]):.8f}")


# ============================================================
# 16. 因子行列をDataFrame化
# ============================================================

time_rank, user_rank, building_rank = TUCKER_RANK

time_columns = [
    f"T_comp{i + 1}"
    for i in range(time_rank)
]

user_columns = [
    f"U_comp{i + 1}"
    for i in range(user_rank)
]

building_columns = [
    f"B_comp{i + 1}"
    for i in range(building_rank)
]


time_df = pd.DataFrame(
    time_factor,
    index=time_labels,
    columns=time_columns,
)
time_df.index.name = "time_slot"


user_df = pd.DataFrame(
    user_factor,
    index=user_labels,
    columns=user_columns,
)
user_df.index.name = USER_COL


building_df = pd.DataFrame(
    building_factor,
    index=building_labels,
    columns=building_columns,
)
building_df.index.name = BUILDING_COL


# ============================================================
# 17. ユーザー属性を付与
# ============================================================

available_attributes = [
    column
    for column in attribute_columns
    if column in colocation_df.columns
]

if available_attributes:

    user_attributes = (
        colocation_df[
            [USER_COL] + available_attributes
        ]
        .copy()
    )

    user_attributes[USER_COL] = (
        user_attributes[USER_COL].astype(str)
    )

    user_attributes = (
        user_attributes
        .groupby(USER_COL, as_index=False)
        .first()
    )

    user_with_attributes_df = (
        user_df
        .reset_index()
        .merge(
            user_attributes,
            on=USER_COL,
            how="left",
        )
    )

else:
    user_with_attributes_df = (
        user_df.reset_index()
    )


# ============================================================
# 18. コアテンソルをlong形式へ変換
# ============================================================

core_rows = []

for t_index in range(core.shape[0]):
    for u_index in range(core.shape[1]):
        for b_index in range(core.shape[2]):

            core_rows.append(
                {
                    "T_component":
                        f"T_comp{t_index + 1}",

                    "U_component":
                        f"U_comp{u_index + 1}",

                    "B_component":
                        f"B_comp{b_index + 1}",

                    "core_value":
                        float(
                            core[
                                t_index,
                                u_index,
                                b_index,
                            ]
                        ),
                }
            )

core_df = pd.DataFrame(core_rows)

core_df = (
    core_df
    .sort_values(
        "core_value",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nコアテンソル上位20組")
print(
    core_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 19. 解釈用サマリーを作成
# ============================================================

time_l1 = normalize_columns_l1(time_factor)
user_l1 = normalize_columns_l1(user_factor)
building_l1 = normalize_columns_l1(building_factor)

summary_rows = []

for _, row in core_df.head(20).iterrows():

    t_component = row["T_component"]
    u_component = row["U_component"]
    b_component = row["B_component"]

    t_index = time_columns.index(t_component)
    u_index = user_columns.index(u_component)
    b_index = building_columns.index(b_component)

    summary_rows.append(
        {
            "T_component": t_component,
            "U_component": u_component,
            "B_component": b_component,
            "core_value": float(row["core_value"]),

            "top_time_slots": top_k(
                time_labels,
                time_l1[:, t_index],
                k=10,
            ),

            "top_users": top_k(
                user_labels,
                user_l1[:, u_index],
                k=10,
            ),

            "top_buildings": top_k(
                building_labels,
                building_l1[:, b_index],
                k=10,
            ),
        }
    )


# ============================================================
# 20. 建物・時間別の共滞在統計
# ============================================================

# 日・建物・スロット単位のグループ人数
daily_group_df = (
    colocation_df
    .groupby(
        [
            "date",
            BUILDING_COL,
            "time_slot_id",
            "time_slot_label",
        ],
        as_index=False,
    )
    .agg(
        colocation_users=(
            USER_COL,
            "nunique",
        )
    )
)


# 建物・時間スロットごとの統計
building_time_stats = (
    daily_group_df
    .groupby(
        [
            BUILDING_COL,
            "time_slot_id",
            "time_slot_label",
        ],
        as_index=False,
    )
    .agg(
        colocation_days=(
            "date",
            "nunique",
        ),
        total_user_days=(
            "colocation_users",
            "sum",
        ),
        mean_group_size=(
            "colocation_users",
            "mean",
        ),
        median_group_size=(
            "colocation_users",
            "median",
        ),
        max_group_size=(
            "colocation_users",
            "max",
        ),
    )
)


# 建物単位の統計
building_stats = (
    daily_group_df
    .groupby(
        BUILDING_COL,
        as_index=False,
    )
    .agg(
        colocation_groups=(
            "date",
            "size",
        ),
        colocation_days=(
            "date",
            "nunique",
        ),
        total_user_days=(
            "colocation_users",
            "sum",
        ),
        mean_group_size=(
            "colocation_users",
            "mean",
        ),
        median_group_size=(
            "colocation_users",
            "median",
        ),
        max_group_size=(
            "colocation_users",
            "max",
        ),
    )
)


# ============================================================
# 21. 保存
# ============================================================

time_path = (
    OUT_DIR / "tucker_T_144_time_factors.csv"
)

user_path = (
    OUT_DIR / "tucker_U_user_factors.csv"
)

user_attributes_path = (
    OUT_DIR
    / "tucker_U_user_factors_with_attributes.csv"
)

building_path = (
    OUT_DIR / "tucker_B_building_factors.csv"
)

core_path = (
    OUT_DIR / "tucker_core_tensor.csv"
)

summary_path = (
    OUT_DIR / "tucker_component_summary.json"
)

metadata_path = (
    OUT_DIR / "tucker_metadata.json"
)

colocation_records_path = (
    OUT_DIR / "slot_colocation_records.parquet"
)

group_summary_path = (
    OUT_DIR / "slot_colocation_groups.csv"
)

building_time_stats_path = (
    OUT_DIR / "building_time_colocation_stats.csv"
)

building_stats_path = (
    OUT_DIR / "building_colocation_stats.csv"
)

time_labels_path = (
    OUT_DIR / "time_slot_144_labels.csv"
)


time_df.to_csv(
    time_path,
    encoding="utf-8-sig",
)

user_df.to_csv(
    user_path,
    encoding="utf-8-sig",
)

user_with_attributes_df.to_csv(
    user_attributes_path,
    index=False,
    encoding="utf-8-sig",
)

building_df.to_csv(
    building_path,
    encoding="utf-8-sig",
)

core_df.to_csv(
    core_path,
    index=False,
    encoding="utf-8-sig",
)

colocation_df.to_parquet(
    colocation_records_path,
    index=False,
)

group_summary.to_csv(
    group_summary_path,
    index=False,
    encoding="utf-8-sig",
)

building_time_stats.to_csv(
    building_time_stats_path,
    index=False,
    encoding="utf-8-sig",
)

building_stats.to_csv(
    building_stats_path,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(
    {
        "time_slot_id": np.arange(144),
        "time_slot_label": time_labels,
    }
).to_csv(
    time_labels_path,
    index=False,
    encoding="utf-8-sig",
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary_rows,
        file,
        ensure_ascii=False,
        indent=2,
    )


metadata = {
    "created_at": datetime.now().isoformat(),
    "input_file": str(INPUT_PARQUET),
    "tensor_axis_order": [
        "time",
        "user",
        "building",
    ],
    "tensor_shape": [
        int(T),
        int(U),
        int(B),
    ],
    "tucker_rank": list(TUCKER_RANK),
    "slot_minutes": SLOT_MINUTES,
    "minimum_overlap_minutes":
        MIN_OVERLAP_MINUTES,
    "minimum_users_per_group":
        MIN_USERS_PER_GROUP,
    "number_of_colocation_records":
        int(len(colocation_df)),
    "number_of_colocation_groups":
        int(len(group_summary)),
    "number_of_users": int(U),
    "number_of_buildings": int(B),
    "number_of_time_slots": int(T),
    "number_of_nonzero_elements": int(nnz),
    "tensor_sum": float(X.sum()),
    "tensor_max": float(X.max()),
}

if errors is not None and len(errors) > 0:
    metadata["iterations"] = int(len(errors))
    metadata["last_error"] = float(errors[-1])

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=2,
    )


print("\n" + "=" * 70)
print("保存完了")
print("=" * 70)
print(f"時間因子: {time_path}")
print(f"ユーザー因子: {user_path}")
print(f"属性付きユーザー因子: {user_attributes_path}")
print(f"建物因子: {building_path}")
print(f"コアテンソル: {core_path}")
print(f"成分サマリー: {summary_path}")
print(f"共滞在参加レコード: {colocation_records_path}")
print(f"共滞在グループ: {group_summary_path}")
print(f"建物・時間別統計: {building_time_stats_path}")
print(f"建物別統計: {building_stats_path}")
print(f"メタデータ: {metadata_path}")
