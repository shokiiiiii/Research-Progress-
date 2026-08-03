from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ============================================================
# 0. 設定
# ============================================================

INPUT_PARQUET = Path(
    "/Volumes/一ノ瀬/タッカー分解/ファイル/"
    "東広島滞在データwith_building.parquet"
)

INPUT_NAMES_CSV = Path(
    "/Users/tsg/Desktop/Ichinose_work/"
    "05_共滞在データのエンリッチメント/"
    "05_data/02_広島県土地利用データ/"
    "Saijo500m_with_counts.csv"
)

OUT_DIR = Path(
    "/Volumes/一ノ瀬/タッカー分解/出力/"
    "共滞在判定_新旧比較"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# 列名
# ------------------------------------------------------------

USER_COL = "User_Id"
TRIP_COL = "Trip_Id"
MODE_COL = "TripMode"
START_COL = "Time_start"
END_COL = "Time_end"
BUILDING_COL = "building_name"
BUILDING_ID_COL = "building_id"


# ------------------------------------------------------------
# 旧方法
# ------------------------------------------------------------

# 異なるユーザーの滞在が5分以上重なれば共滞在
OLD_MIN_COSTAY = pd.Timedelta("5min")

TIME_ORDER = [
    "早朝(4-6)",
    "朝ピーク(7-9)",
    "昼間(10-15)",
    "夕方(16-17)",
    "夕ピーク(18-20)",
    "夜間(21-3)",
]


# ------------------------------------------------------------
# 新方法
# ------------------------------------------------------------

# 1日を10分スロットに分割
SLOT_WIDTH = pd.Timedelta("10min")

# 10分スロットとの重なりが8分以上の場合のみ採用
NEW_MIN_SLOT_OVERLAP = pd.Timedelta("8min")


# ------------------------------------------------------------
# 出力設定
# ------------------------------------------------------------

TOP_N = 20

plt.rcParams["font.family"] = [
    "Hiragino Sans",
    "Yu Gothic",
    "Meiryo",
    "DejaVu Sans",
]


# ============================================================
# 1. ユーティリティ
# ============================================================

def try_read_csv(path: Path) -> pd.DataFrame:
    """
    日本語CSVを複数の文字コードで読み込む。
    """

    errors = []

    for encoding in [
        "utf-8-sig",
        "cp932",
        "shift_jis",
        "utf-8",
    ]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )

        except Exception as exc:
            errors.append(
                (encoding, str(exc))
            )

    error_message = "\n".join(
        f"{encoding}: {message}"
        for encoding, message in errors
    )

    raise RuntimeError(
        "CSVを読み込めませんでした。\n"
        + error_message
    )


def normalize_name(
    series: pd.Series,
) -> pd.Series:
    """
    建物名を正規化する。

    ・全角空白を半角空白に変換
    ・前後空白を削除
    ・連続する空白を1個に統一
    """

    return (
        series
        .astype("string")
        .fillna("")
        .str.replace(
            "\u3000",
            " ",
            regex=False,
        )
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )


def hour_to_time_bucket(
    hour: int,
) -> str:
    """
    時刻を旧方法の6時間帯に分類する。
    """

    if 4 <= hour <= 6:
        return "早朝(4-6)"

    if 7 <= hour <= 9:
        return "朝ピーク(7-9)"

    if 10 <= hour <= 15:
        return "昼間(10-15)"

    if 16 <= hour <= 17:
        return "夕方(16-17)"

    if 18 <= hour <= 20:
        return "夕ピーク(18-20)"

    return "夜間(21-3)"


def safe_change_rate(
    old_value: float,
    new_value: float,
) -> float:
    """
    旧方法を基準とした増減率を計算する。
    """

    if old_value == 0:
        return np.nan

    return (
        (new_value - old_value)
        / old_value
        * 100
    )


def prepare_for_parquet(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    PyArrowで安全にParquet保存できる型へ変換する。

    主な処理
    --------
    1. category型をstring型へ変換
    2. datetime64[ns]をマイクロ秒単位へ丸める

    中点時刻の計算で500nsなどの端数が発生すると、
    timestamp[us]への変換時にArrowInvalidが発生するため、
    事前にマイクロ秒へ丸める。
    """

    output = dataframe.copy()

    category_cols = output.select_dtypes(
        include=["category"]
    ).columns

    for col in category_cols:
        output[col] = (
            output[col]
            .astype("string")
        )

    datetime_cols = output.select_dtypes(
        include=["datetime64[ns]"]
    ).columns

    for col in datetime_cols:
        output[col] = (
            output[col]
            .dt.round("us")
        )

    return output


# ============================================================
# 2. 対象建物CSVの読み込み
# ============================================================

names_df = try_read_csv(
    INPUT_NAMES_CSV
)

name_col_candidates = [
    col
    for col in names_df.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "建物",
            "名称",
            "施設",
            "name",
        ]
    )
]

if not name_col_candidates:
    raise KeyError(
        "対象建物CSVに建物名列が見つかりません。\n"
        f"CSV列: {names_df.columns.tolist()}"
    )

names_col = name_col_candidates[0]

names_df["_name_key"] = normalize_name(
    names_df[names_col]
)

valid_name_keys = set(
    names_df.loc[
        names_df["_name_key"].ne(""),
        "_name_key",
    ]
    .drop_duplicates()
)

print(
    "========== 対象建物CSV =========="
)

print(
    f"使用する建物名列: {names_col}"
)

print(
    f"対象建物名数: {len(valid_name_keys):,}"
)


# ============================================================
# 3. Parquet読み込み
# ============================================================

df = pd.read_parquet(
    INPUT_PARQUET
)

required_cols = [
    USER_COL,
    TRIP_COL,
    START_COL,
    END_COL,
    BUILDING_COL,
]

missing_cols = [
    col
    for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise KeyError(
        f"Parquetに必要な列がありません: {missing_cols}\n"
        f"Parquet列: {df.columns.tolist()}"
    )

use_cols = [
    USER_COL,
    TRIP_COL,
    MODE_COL,
    START_COL,
    END_COL,
    BUILDING_ID_COL,
    BUILDING_COL,
    "source_file",
]

use_cols = [
    col
    for col in use_cols
    if col in df.columns
]

df = df[
    use_cols
].copy()

print(
    "\n========== Parquet読み込み =========="
)

print(
    f"読み込み行数: {len(df):,}"
)

print(
    f"使用列: {df.columns.tolist()}"
)


# ============================================================
# 4. 共通前処理
# ============================================================

df[START_COL] = pd.to_datetime(
    df[START_COL],
    errors="coerce",
)

df[END_COL] = pd.to_datetime(
    df[END_COL],
    errors="coerce",
)

before_invalid_filter = len(df)

df = df.dropna(
    subset=[
        USER_COL,
        TRIP_COL,
        START_COL,
        END_COL,
        BUILDING_COL,
    ]
).copy()

df = df[
    df[END_COL] > df[START_COL]
].copy()

print(
    "\n========== 時刻・欠損処理 =========="
)

print(
    f"処理前: {before_invalid_filter:,}"
)

print(
    f"処理後: {len(df):,}"
)


# ------------------------------------------------------------
# activityのみ抽出
# ------------------------------------------------------------

if MODE_COL in df.columns:

    before_activity = len(df)

    df = df[
        df[MODE_COL].eq("activity")
    ].copy()

    print(
        "\n========== TripMode抽出 =========="
    )

    print(
        f"activity抽出前: {before_activity:,}"
    )

    print(
        f"activity抽出後: {len(df):,}"
    )


# ------------------------------------------------------------
# 対象建物CSVとの照合
# ------------------------------------------------------------

df["_name_key"] = normalize_name(
    df[BUILDING_COL]
)

before_building_filter = len(df)

df = df[
    df["_name_key"].isin(
        valid_name_keys
    )
].copy()

df["B_building"] = df["_name_key"]

print(
    "\n========== 対象建物抽出 =========="
)

print(
    f"建物抽出前: {before_building_filter:,}"
)

print(
    f"建物抽出後: {len(df):,}"
)

print(
    f"対象ユーザー数: "
    f"{df[USER_COL].nunique():,}"
)

print(
    f"対象建物数: "
    f"{df['B_building'].nunique():,}"
)


# ------------------------------------------------------------
# 滞在時間
# ------------------------------------------------------------

df["duration_min"] = (
    df[END_COL]
    - df[START_COL]
).dt.total_seconds() / 60


# ------------------------------------------------------------
# 滞在識別ID
# ------------------------------------------------------------

df["stay_id"] = (
    df[USER_COL].astype(str)
    + "||"
    + df[TRIP_COL].astype(str)
    + "||"
    + df[START_COL].astype(str)
    + "||"
    + df[END_COL].astype(str)
    + "||"
    + df["B_building"].astype(str)
)

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["stay_id"]
).copy()

print(
    "\n========== 完全重複除去 =========="
)

print(
    f"除去前: {before_duplicates:,}"
)

print(
    f"除去後: {len(df):,}"
)


# ============================================================
# 5. 旧方法：6時間帯への割当
# ============================================================

old_base = df.copy()

old_base["mid_time"] = (
    old_base[START_COL]
    + (
        old_base[END_COL]
        - old_base[START_COL]
    ) / 2
)

# 共滞在を別日間で判定しないための日付
old_base["old_date"] = (
    old_base["mid_time"]
    .dt.normalize()
)

old_base["old_time_category"] = pd.Categorical(
    old_base["mid_time"]
    .dt.hour
    .map(hour_to_time_bucket),
    categories=TIME_ORDER,
    ordered=True,
)

if old_base[
    "old_time_category"
].isna().any():

    raise ValueError(
        "旧方法の時間帯分類に欠損が発生しました。"
    )

print(
    "\n========== 旧方法：6時間帯割当 =========="
)

print(
    old_base[
        "old_time_category"
    ].value_counts(
        sort=False
    )
)


# ============================================================
# 6. 旧方法：共滞在判定関数
# ============================================================

def find_old_costay_stays(
    group: pd.DataFrame,
    min_overlap: pd.Timedelta,
) -> tuple[set[str], list[dict]]:
    """
    同一建物・同一日・同一時間帯内で、
    異なるユーザーの実滞在時間が5分以上重なるか判定する。

    戻り値
    -------
    participating_stays:
        共滞在に参加したstay_idの集合

    pair_rows:
        共滞在ペアの詳細
    """

    group = (
        group
        .sort_values(
            [
                START_COL,
                END_COL,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(group) < 2:
        return set(), []

    if group[
        USER_COL
    ].nunique() < 2:

        return set(), []

    starts = (
        group[START_COL]
        .astype("int64")
        .to_numpy()
    )

    ends = (
        group[END_COL]
        .astype("int64")
        .to_numpy()
    )

    users = (
        group[USER_COL]
        .astype(str)
        .to_numpy()
    )

    stay_ids = (
        group["stay_id"]
        .astype(str)
        .to_numpy()
    )

    trip_ids = (
        group[TRIP_COL]
        .astype(str)
        .to_numpy()
    )

    threshold_ns = int(
        min_overlap.value
    )

    participating_stays = set()
    pair_rows = []

    n = len(group)

    for i in range(
        n - 1
    ):

        # 5分以上重なるために許容される
        # 相手滞在の最も遅い開始時刻
        latest_start_ns = (
            ends[i]
            - threshold_ns
        )

        # 開始時刻順なので候補範囲を限定
        upper = np.searchsorted(
            starts,
            latest_start_ns,
            side="right",
        )

        for j in range(
            i + 1,
            upper,
        ):

            # 同一ユーザー同士は除外
            if users[i] == users[j]:
                continue

            overlap_start_ns = max(
                starts[i],
                starts[j],
            )

            overlap_end_ns = min(
                ends[i],
                ends[j],
            )

            overlap_ns = (
                overlap_end_ns
                - overlap_start_ns
            )

            if overlap_ns < threshold_ns:
                continue

            participating_stays.add(
                stay_ids[i]
            )

            participating_stays.add(
                stay_ids[j]
            )

            pair_rows.append({
                "stay_id_a": stay_ids[i],
                "stay_id_b": stay_ids[j],
                "user_a": users[i],
                "user_b": users[j],
                "trip_a": trip_ids[i],
                "trip_b": trip_ids[j],
                "overlap_start": pd.to_datetime(
                    overlap_start_ns
                ),
                "overlap_end": pd.to_datetime(
                    overlap_end_ns
                ),
                "overlap_min": (
                    overlap_ns
                    / 60_000_000_000
                ),
            })

    return (
        participating_stays,
        pair_rows,
    )


# ============================================================
# 7. 旧方法：全グループの共滞在判定
# ============================================================

old_costay_ids = set()
old_pair_parts = []

old_grouped = old_base.groupby(
    [
        "old_date",
        "B_building",
        "old_time_category",
    ],
    observed=True,
    sort=False,
)

for (
    old_date,
    building_name,
    time_category,
), group in tqdm(
    old_grouped,
    total=old_grouped.ngroups,
    desc="旧方法の共滞在判定",
):

    participating, pair_rows = (
        find_old_costay_stays(
            group=group,
            min_overlap=OLD_MIN_COSTAY,
        )
    )

    old_costay_ids.update(
        participating
    )

    if pair_rows:

        pair_df = pd.DataFrame(
            pair_rows
        )

        pair_df["date"] = (
            old_date
        )

        pair_df["building_name"] = (
            building_name
        )

        pair_df["time_category"] = (
            str(time_category)
        )

        old_pair_parts.append(
            pair_df
        )


if old_pair_parts:

    old_pairs = pd.concat(
        old_pair_parts,
        ignore_index=True,
    )

else:

    old_pairs = pd.DataFrame(
        columns=[
            "stay_id_a",
            "stay_id_b",
            "user_a",
            "user_b",
            "trip_a",
            "trip_b",
            "overlap_start",
            "overlap_end",
            "overlap_min",
            "date",
            "building_name",
            "time_category",
        ]
    )


old_base["old_costay"] = (
    old_base["stay_id"]
    .isin(old_costay_ids)
)

old_costay_stays = old_base[
    old_base["old_costay"]
].copy()

print(
    "\n========== 旧方法：共滞在判定結果 =========="
)

print(
    f"共滞在ペア数: "
    f"{len(old_pairs):,}"
)

print(
    f"共滞在参加滞在数: "
    f"{len(old_costay_stays):,}"
)

print(
    f"共滞在参加ユーザー数: "
    f"{old_costay_stays[USER_COL].nunique():,}"
)

print(
    f"共滞在発生建物数: "
    f"{old_costay_stays['B_building'].nunique():,}"
)


# ============================================================
# 8. 新方法：10分スロットへの展開
# ============================================================

new_base = df.copy()

new_base["_slot_start_min"] = (
    new_base[START_COL]
    .dt.floor("10min")
)

new_base["_slot_start_max"] = (
    new_base[END_COL]
    - pd.Timedelta("1ns")
).dt.floor("10min")

new_base["_n_candidate_slots"] = (
    (
        new_base["_slot_start_max"]
        - new_base["_slot_start_min"]
    )
    / SLOT_WIDTH
).astype(int) + 1

new_base = new_base[
    new_base["_n_candidate_slots"] > 0
].copy()

new_base["_slot_offset"] = (
    new_base[
        "_n_candidate_slots"
    ]
    .apply(
        lambda n: np.arange(
            n,
            dtype=np.int32,
        )
    )
)

slot_df = new_base.explode(
    "_slot_offset",
    ignore_index=True,
)

slot_df["_slot_offset"] = (
    slot_df["_slot_offset"]
    .astype(np.int32)
)

slot_df["slot_start"] = (
    slot_df["_slot_start_min"]
    + slot_df["_slot_offset"]
    * SLOT_WIDTH
)

slot_df["slot_end"] = (
    slot_df["slot_start"]
    + SLOT_WIDTH
)

print(
    "\n========== 新方法：スロット展開 =========="
)

print(
    f"展開前滞在数: "
    f"{len(new_base):,}"
)

print(
    f"展開後行数: "
    f"{len(slot_df):,}"
)


# ============================================================
# 9. 新方法：各スロットとの重なり時間
# ============================================================

slot_df["overlap_start"] = slot_df[
    [
        START_COL,
        "slot_start",
    ]
].max(
    axis=1
)

slot_df["overlap_end"] = slot_df[
    [
        END_COL,
        "slot_end",
    ]
].min(
    axis=1
)

slot_df["slot_overlap"] = (
    slot_df["overlap_end"]
    - slot_df["overlap_start"]
)

slot_df["slot_overlap_min"] = (
    slot_df["slot_overlap"]
    .dt.total_seconds()
    / 60
)

valid_slot_df = slot_df[
    slot_df["slot_overlap"]
    >= NEW_MIN_SLOT_OVERLAP
].copy()

valid_slot_df["slot_date"] = (
    valid_slot_df["slot_start"]
    .dt.normalize()
)

valid_slot_df["time_slot_id"] = (
    valid_slot_df["slot_start"].dt.hour
    * 6
    + valid_slot_df[
        "slot_start"
    ].dt.minute
    // 10
).astype(
    np.int16
)

valid_slot_df["time_slot_label"] = (
    valid_slot_df["slot_start"]
    .dt.strftime("%H:%M")
)

# 同一滞在・同一スロットの重複を除去
valid_slot_df = (
    valid_slot_df
    .drop_duplicates(
        subset=[
            "stay_id",
            "B_building",
            "slot_start",
        ]
    )
    .copy()
)

print(
    "\n========== 新方法：8分以上のスロット =========="
)

print(
    f"有効スロット行数: "
    f"{len(valid_slot_df):,}"
)

print(
    f"有効スロットを持つ滞在数: "
    f"{valid_slot_df['stay_id'].nunique():,}"
)


# ============================================================
# 10. 新方法：共滞在判定
# ============================================================

# 同じ建物・同じ日時スロットに存在する
# ユニークユーザー数を計算
valid_slot_df[
    "users_in_building_slot"
] = (
    valid_slot_df
    .groupby(
        [
            "B_building",
            "slot_start",
        ],
        observed=True,
    )[USER_COL]
    .transform("nunique")
)

# ユーザー数が2人以上なら共滞在スロット
new_costay_slots = valid_slot_df[
    valid_slot_df[
        "users_in_building_slot"
    ] >= 2
].copy()

new_costay_ids = set(
    new_costay_slots[
        "stay_id"
    ].unique()
)

new_stay_summary = (
    new_costay_slots
    .groupby(
        "stay_id",
        observed=True,
    )
    .agg(
        new_costay_slot_count=(
            "slot_start",
            "nunique",
        ),
        first_costay_slot=(
            "slot_start",
            "min",
        ),
        last_costay_slot=(
            "slot_start",
            "max",
        ),
        new_costay_slot_labels=(
            "time_slot_label",
            lambda values: ",".join(
                sorted(
                    set(values)
                )
            ),
        ),
    )
    .reset_index()
)

new_costay_stays = (
    df[
        df["stay_id"].isin(
            new_costay_ids
        )
    ]
    .merge(
        new_stay_summary,
        on="stay_id",
        how="left",
    )
)

print(
    "\n========== 新方法：共滞在判定結果 =========="
)

print(
    f"共滞在スロット行数: "
    f"{len(new_costay_slots):,}"
)

print(
    f"共滞在参加滞在数: "
    f"{len(new_costay_stays):,}"
)

print(
    f"共滞在参加ユーザー数: "
    f"{new_costay_stays[USER_COL].nunique():,}"
)

print(
    f"共滞在発生建物数: "
    f"{new_costay_stays['B_building'].nunique():,}"
)


# ============================================================
# 11. 元の滞在単位で新旧結果を結合
# ============================================================

comparison = df[
    [
        "stay_id",
        USER_COL,
        TRIP_COL,
        START_COL,
        END_COL,
        "B_building",
        "duration_min",
    ]
].copy()

old_detail = old_base[
    [
        "stay_id",
        "mid_time",
        "old_date",
        "old_time_category",
        "old_costay",
    ]
].copy()

comparison = comparison.merge(
    old_detail,
    on="stay_id",
    how="left",
)

comparison = comparison.merge(
    new_stay_summary,
    on="stay_id",
    how="left",
)

comparison["old_costay"] = (
    comparison["old_costay"]
    .fillna(False)
    .astype(bool)
)

comparison["new_costay"] = (
    comparison["stay_id"]
    .isin(new_costay_ids)
)

comparison["new_costay_slot_count"] = (
    comparison[
        "new_costay_slot_count"
    ]
    .fillna(0)
    .astype(int)
)

comparison["new_costay_slot_labels"] = (
    comparison[
        "new_costay_slot_labels"
    ]
    .fillna("")
)


# ============================================================
# 12. 新旧間の移行区分
# ============================================================

conditions = [
    (
        comparison["old_costay"]
        & comparison["new_costay"]
    ),
    (
        comparison["old_costay"]
        & ~comparison["new_costay"]
    ),
    (
        ~comparison["old_costay"]
        & comparison["new_costay"]
    ),
]

choices = [
    "両方法で共滞在",
    "旧方法のみ共滞在",
    "新方法のみ共滞在",
]

comparison["transition"] = np.select(
    conditions,
    choices,
    default="両方法で非共滞在",
)

transition_order = [
    "両方法で共滞在",
    "旧方法のみ共滞在",
    "新方法のみ共滞在",
    "両方法で非共滞在",
]

comparison["transition"] = pd.Categorical(
    comparison["transition"],
    categories=transition_order,
    ordered=True,
)

transition_summary = (
    comparison
    .groupby(
        "transition",
        observed=False,
    )
    .agg(
        stay_count=(
            "stay_id",
            "nunique",
        ),
        unique_users=(
            USER_COL,
            "nunique",
        ),
        unique_buildings=(
            "B_building",
            "nunique",
        ),
        mean_duration_min=(
            "duration_min",
            "mean",
        ),
        median_duration_min=(
            "duration_min",
            "median",
        ),
    )
    .reset_index()
)

transition_summary["share_pct"] = (
    transition_summary["stay_count"]
    / comparison["stay_id"].nunique()
    * 100
)


# ============================================================
# 13. 全体比較
# ============================================================

old_stay_count = int(
    comparison[
        "old_costay"
    ].sum()
)

new_stay_count = int(
    comparison[
        "new_costay"
    ].sum()
)

old_user_count = (
    comparison.loc[
        comparison["old_costay"],
        USER_COL,
    ]
    .nunique()
)

new_user_count = (
    comparison.loc[
        comparison["new_costay"],
        USER_COL,
    ]
    .nunique()
)

old_building_count = (
    comparison.loc[
        comparison["old_costay"],
        "B_building",
    ]
    .nunique()
)

new_building_count = (
    comparison.loc[
        comparison["new_costay"],
        "B_building",
    ]
    .nunique()
)

old_pair_count = len(
    old_pairs
)

# 新方法は建物・スロットごとの組合せ数
new_pair_count_by_slot = (
    new_costay_slots
    .groupby(
        [
            "B_building",
            "slot_start",
        ],
        observed=True,
    )[USER_COL]
    .nunique()
)

new_pair_count = int(
    (
        new_pair_count_by_slot
        * (
            new_pair_count_by_slot
            - 1
        )
        / 2
    ).sum()
)

overall_summary = pd.DataFrame([
    {
        "indicator": "共滞在参加滞在数",
        "old_method": old_stay_count,
        "new_method": new_stay_count,
        "difference": (
            new_stay_count
            - old_stay_count
        ),
        "change_rate_pct": safe_change_rate(
            old_stay_count,
            new_stay_count,
        ),
    },
    {
        "indicator": "共滞在参加ユーザー数",
        "old_method": old_user_count,
        "new_method": new_user_count,
        "difference": (
            new_user_count
            - old_user_count
        ),
        "change_rate_pct": safe_change_rate(
            old_user_count,
            new_user_count,
        ),
    },
    {
        "indicator": "共滞在発生建物数",
        "old_method": old_building_count,
        "new_method": new_building_count,
        "difference": (
            new_building_count
            - old_building_count
        ),
        "change_rate_pct": safe_change_rate(
            old_building_count,
            new_building_count,
        ),
    },
    {
        "indicator": "共滞在関係数",
        "old_method": old_pair_count,
        "new_method": new_pair_count,
        "difference": (
            new_pair_count
            - old_pair_count
        ),
        "change_rate_pct": safe_change_rate(
            old_pair_count,
            new_pair_count,
        ),
    },
])


# ============================================================
# 14. 建物別集計
# ============================================================

old_building_summary = (
    comparison.loc[
        comparison["old_costay"]
    ]
    .groupby(
        "B_building",
        observed=True,
    )
    .agg(
        old_costay_stays=(
            "stay_id",
            "nunique",
        ),
        old_costay_users=(
            USER_COL,
            "nunique",
        ),
        old_mean_duration_min=(
            "duration_min",
            "mean",
        ),
    )
    .reset_index()
)

new_building_summary = (
    comparison.loc[
        comparison["new_costay"]
    ]
    .groupby(
        "B_building",
        observed=True,
    )
    .agg(
        new_costay_stays=(
            "stay_id",
            "nunique",
        ),
        new_costay_users=(
            USER_COL,
            "nunique",
        ),
        new_mean_duration_min=(
            "duration_min",
            "mean",
        ),
        new_total_costay_slots=(
            "new_costay_slot_count",
            "sum",
        ),
    )
    .reset_index()
)

building_comparison = (
    old_building_summary
    .merge(
        new_building_summary,
        on="B_building",
        how="outer",
    )
)

count_cols = [
    "old_costay_stays",
    "new_costay_stays",
    "old_costay_users",
    "new_costay_users",
    "new_total_costay_slots",
]

for col in count_cols:

    building_comparison[col] = (
        building_comparison[col]
        .fillna(0)
        .astype(int)
    )

building_comparison["stay_difference"] = (
    building_comparison[
        "new_costay_stays"
    ]
    - building_comparison[
        "old_costay_stays"
    ]
)

building_comparison["stay_change_rate_pct"] = np.where(
    building_comparison[
        "old_costay_stays"
    ] > 0,
    (
        building_comparison[
            "stay_difference"
        ]
        / building_comparison[
            "old_costay_stays"
        ]
        * 100
    ),
    np.nan,
)

building_comparison["user_difference"] = (
    building_comparison[
        "new_costay_users"
    ]
    - building_comparison[
        "old_costay_users"
    ]
)

# 新旧の大きい方を上位建物選択に使用
building_comparison["ranking_count"] = (
    building_comparison[
        [
            "old_costay_stays",
            "new_costay_stays",
        ]
    ].max(
        axis=1
    )
)

building_comparison = (
    building_comparison
    .sort_values(
        [
            "ranking_count",
            "old_costay_stays",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

top_buildings = (
    building_comparison
    .head(TOP_N)
    .copy()
)


# ============================================================
# 15. 建物別の移行区分
# ============================================================

building_transition = (
    comparison[
        comparison["transition"]
        != "両方法で非共滞在"
    ]
    .groupby(
        [
            "B_building",
            "transition",
        ],
        observed=False,
    )
    .agg(
        stay_count=(
            "stay_id",
            "nunique",
        )
    )
    .reset_index()
)

building_transition_wide = (
    building_transition
    .pivot(
        index="B_building",
        columns="transition",
        values="stay_count",
    )
    .fillna(0)
    .reset_index()
)

building_transition_wide.columns.name = None

for col in transition_order:

    if col not in building_transition_wide.columns:
        building_transition_wide[col] = 0

    building_transition_wide[col] = (
        building_transition_wide[col]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# 16. 滞在時間別集計
# ============================================================

duration_bins = [
    0,
    5,
    8,
    10,
    15,
    20,
    30,
    60,
    120,
    np.inf,
]

duration_labels = [
    "0-4分",
    "5-7分",
    "8-9分",
    "10-14分",
    "15-19分",
    "20-29分",
    "30-59分",
    "60-119分",
    "120分以上",
]

comparison["duration_group"] = pd.cut(
    comparison["duration_min"],
    bins=duration_bins,
    labels=duration_labels,
    right=False,
)

duration_summary = (
    comparison
    .groupby(
        "duration_group",
        observed=False,
    )
    .agg(
        total_stays=(
            "stay_id",
            "nunique",
        ),
        old_costay_stays=(
            "old_costay",
            "sum",
        ),
        new_costay_stays=(
            "new_costay",
            "sum",
        ),
    )
    .reset_index()
)

duration_summary["difference"] = (
    duration_summary[
        "new_costay_stays"
    ]
    - duration_summary[
        "old_costay_stays"
    ]
)

duration_summary["change_rate_pct"] = np.where(
    duration_summary[
        "old_costay_stays"
    ] > 0,
    (
        duration_summary[
            "difference"
        ]
        / duration_summary[
            "old_costay_stays"
        ]
        * 100
    ),
    np.nan,
)


# ============================================================
# 17. CSV・Parquet保存
# ============================================================

comparison_output_cols = [
    "stay_id",
    USER_COL,
    TRIP_COL,
    START_COL,
    END_COL,
    "B_building",
    "duration_min",
    "mid_time",
    "old_date",
    "old_time_category",
    "old_costay",
    "new_costay",
    "new_costay_slot_count",
    "first_costay_slot",
    "last_costay_slot",
    "new_costay_slot_labels",
    "transition",
    "duration_group",
]


# ------------------------------------------------------------
# 滞在単位比較データ
# ------------------------------------------------------------

comparison_output = comparison[
    comparison_output_cols
].copy()

comparison_output_parquet = prepare_for_parquet(
    comparison_output
)

comparison_output_parquet.to_parquet(
    OUT_DIR
    / "01_stay_level_costay_comparison.parquet",
    index=False,
    engine="pyarrow",
    compression="snappy",
    coerce_timestamps="us",
    allow_truncated_timestamps=True,
)

comparison_output.to_csv(
    OUT_DIR
    / "01_stay_level_costay_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


# ------------------------------------------------------------
# 全体集計
# ------------------------------------------------------------

overall_summary.to_csv(
    OUT_DIR
    / "02_overall_costay_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


# ------------------------------------------------------------
# 新旧移行状況
# ------------------------------------------------------------

transition_summary.to_csv(
    OUT_DIR
    / "03_transition_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ------------------------------------------------------------
# 全建物比較
# ------------------------------------------------------------

building_comparison.to_csv(
    OUT_DIR
    / "04_building_costay_comparison_all.csv",
    index=False,
    encoding="utf-8-sig",
)


# ------------------------------------------------------------
# 上位20建物
# ------------------------------------------------------------

top_buildings.to_csv(
    OUT_DIR
    / "05_building_costay_comparison_top20.csv",
    index=False,
    encoding="utf-8-sig",
)


# ------------------------------------------------------------
# 建物別の移行区分
# ------------------------------------------------------------

building_transition_wide.to_csv(
    OUT_DIR
    / "06_building_transition_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ------------------------------------------------------------
# 滞在時間別比較
# ------------------------------------------------------------

duration_summary.to_csv(
    OUT_DIR
    / "07_duration_costay_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


# ------------------------------------------------------------
# 旧方法の共滞在ペア
# ------------------------------------------------------------

old_pairs_output = prepare_for_parquet(
    old_pairs
)

old_pairs_output.to_parquet(
    OUT_DIR
    / "08_old_method_costay_pairs.parquet",
    index=False,
    engine="pyarrow",
    compression="snappy",
    coerce_timestamps="us",
    allow_truncated_timestamps=True,
)


# ------------------------------------------------------------
# 新方法の共滞在スロット
# ------------------------------------------------------------

new_slot_output_cols = [
    "stay_id",
    USER_COL,
    TRIP_COL,
    "B_building",
    START_COL,
    END_COL,
    "slot_start",
    "slot_end",
    "slot_overlap_min",
    "users_in_building_slot",
]

new_slot_output = new_costay_slots[
    new_slot_output_cols
].copy()

new_slot_output = prepare_for_parquet(
    new_slot_output
)

new_slot_output.to_parquet(
    OUT_DIR
    / "09_new_method_costay_slots.parquet",
    index=False,
    engine="pyarrow",
    compression="snappy",
    coerce_timestamps="us",
    allow_truncated_timestamps=True,
)

print(
    "\nCSV・Parquetの保存が完了しました。"
)


# ============================================================
# 18. グラフ1：全体比較
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

methods = [
    "旧方法",
    "新方法",
]

values = [
    old_stay_count,
    new_stay_count,
]

ax.bar(
    methods,
    values,
)

ax.set_ylabel(
    "共滞在に参加した元の滞在数"
)

ax.set_title(
    "新旧方法による共滞在参加滞在数"
)

for index, value in enumerate(
    values
):

    ax.text(
        index,
        value,
        f"{value:,}",
        ha="center",
        va="bottom",
    )

fig.tight_layout()

fig.savefig(
    OUT_DIR
    / "10_overall_costay_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 19. グラフ2：上位建物の新旧比較
# ============================================================

plot_df = (
    top_buildings
    .sort_values(
        "ranking_count",
        ascending=True,
    )
)

y = np.arange(
    len(plot_df)
)

bar_height = 0.4

fig, ax = plt.subplots(
    figsize=(
        12,
        max(
            6,
            len(plot_df) * 0.45,
        ),
    )
)

ax.barh(
    y - bar_height / 2,
    plot_df[
        "old_costay_stays"
    ],
    height=bar_height,
    label="旧方法",
)

ax.barh(
    y + bar_height / 2,
    plot_df[
        "new_costay_stays"
    ],
    height=bar_height,
    label="新方法",
)

ax.set_yticks(
    y
)

ax.set_yticklabels(
    plot_df[
        "B_building"
    ]
)

ax.set_xlabel(
    "共滞在に参加した元の滞在数"
)

ax.set_ylabel(
    "建物名"
)

ax.set_title(
    f"上位{TOP_N}建物における共滞在数の比較"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT_DIR
    / "11_top_building_costay_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 20. グラフ3：上位建物の増減
# ============================================================

difference_plot = (
    top_buildings
    .sort_values(
        "stay_difference",
        ascending=True,
    )
)

fig, ax = plt.subplots(
    figsize=(
        11,
        max(
            6,
            len(difference_plot)
            * 0.45,
        ),
    )
)

ax.barh(
    difference_plot[
        "B_building"
    ],
    difference_plot[
        "stay_difference"
    ],
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "増減数（新方法 − 旧方法）"
)

ax.set_ylabel(
    "建物名"
)

ax.set_title(
    f"上位{TOP_N}建物の共滞在参加滞在数の増減"
)

fig.tight_layout()

fig.savefig(
    OUT_DIR
    / "12_top_building_costay_difference.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 21. グラフ4：滞在時間別比較
# ============================================================

duration_plot = (
    duration_summary
    .copy()
)

x = np.arange(
    len(duration_plot)
)

bar_width = 0.4

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.bar(
    x - bar_width / 2,
    duration_plot[
        "old_costay_stays"
    ],
    width=bar_width,
    label="旧方法",
)

ax.bar(
    x + bar_width / 2,
    duration_plot[
        "new_costay_stays"
    ],
    width=bar_width,
    label="新方法",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    duration_plot[
        "duration_group"
    ].astype(str),
    rotation=45,
    ha="right",
)

ax.set_xlabel(
    "元の滞在時間"
)

ax.set_ylabel(
    "共滞在参加滞在数"
)

ax.set_title(
    "滞在時間別の共滞在判定結果"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT_DIR
    / "13_duration_costay_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# 22. 結果表示
# ============================================================

print("\n")
print("=" * 70)
print("全体比較")
print("=" * 70)

print(
    overall_summary.to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("新旧間の移行")
print("=" * 70)

print(
    transition_summary.to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print(
    f"上位{TOP_N}建物"
)
print("=" * 70)

print(
    top_buildings[
        [
            "B_building",
            "old_costay_stays",
            "new_costay_stays",
            "stay_difference",
            "stay_change_rate_pct",
            "old_costay_users",
            "new_costay_users",
        ]
    ].to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("保存完了")
print("=" * 70)

print(
    f"保存先: {OUT_DIR}"
)
