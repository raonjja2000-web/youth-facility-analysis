from pathlib import Path

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="지역별 청소년 시설 분석",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

    html, body, .stApp, .stApp * {
        font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(180deg, #EEF5FF 0%, #FFF8F1 100%);
        color: #24324A;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    .stDeployButton {
        display: none !important;
    }

    .block-container {
        max-width: 1240px;
        padding: 1.6rem 1.2rem 2rem 1.2rem !important;
    }

    div[data-testid="stVerticalBlock"] { gap: 0.7rem; }

    /* ---------- 상단 배너 ---------- */
    .report-head {
        position: relative;
        background: linear-gradient(120deg, #5B8DEF 0%, #7BA7FF 55%, #8FD3C3 100%);
        border-radius: 24px;
        padding: 26px 30px;
        margin: 0 0 14px 0;
        box-shadow: 0 10px 26px rgba(91, 141, 239, 0.28);
    }
    .report-head h1 {
        margin: 0 0 6px 0;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.2;
        transform: none;
    }
    .report-head p {
        margin: 0;
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .top-source-note {
        position: static;
        display: inline-block;
        background: rgba(255, 255, 255, 0.24);
        border-radius: 999px;
        padding: 4px 12px;
        margin-bottom: 10px;
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* ---------- 요약 숫자 ---------- */
    .summary-line {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        border: 0;
        margin-bottom: 14px;
    }
    .summary-item {
        background: #ffffff;
        border-radius: 18px;
        padding: 14px 18px;
        color: #6B7A90;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: left;
        box-shadow: 0 4px 14px rgba(36, 50, 74, 0.06);
    }
    .summary-item b {
        display: block;
        margin: 4px 0 0 0;
        color: #24324A;
        font-size: 1.5rem;
        font-weight: 800;
    }

    /* ---------- 입력 ---------- */
    .input-label {
        color: #24324A;
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .readonly-value {
        background: #ffffff;
        border: 1.5px solid #E3EAF5;
        border-radius: 12px;
        min-height: 42px;
        padding: 9px 14px;
        color: #24324A;
        font-size: 1rem;
        font-weight: 700;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1.5px solid #E3EAF5 !important;
        border-radius: 12px !important;
        color: #24324A !important;
    }
    div[data-baseweb="select"] span { color: #24324A !important; }

    /* ---------- 결과 카드 ---------- */
    .right-result-title {
        font-size: 1rem;
        font-weight: 800;
        margin: 6px 0 6px 0;
    }
    .right-result-title.good { color: #1C8C6A; }
    .right-result-title.mid  { color: #3B6FD6; }
    .right-result-title.bad  { color: #E0563F; }

    .right-result-card {
        border-radius: 20px;
        padding: 14px 18px;
        margin-bottom: 4px;
        box-shadow: 0 6px 18px rgba(36, 50, 74, 0.07);
        border-width: 1.5px;
        border-style: solid;
        border-left-width: 1.5px;
    }
    .right-result-card.good { background: #E6F7F1; border-color: #B8E6D6; }
    .right-result-card.mid  { background: #E8F0FE; border-color: #BFD3FA; }
    .right-result-card.bad  { background: #FFECE8; border-color: #FFC9BF; }

    .right-main {
        color: #24324A;
        font-size: 0.95rem;
        font-weight: 800;
    }
    .right-warning {
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .right-result-card.good .right-warning { color: #1C8C6A; }
    .right-result-card.mid  .right-warning { color: #3B6FD6; }
    .right-result-card.bad  .right-warning { color: #E0563F; }

    /* ---------- 기타 ---------- */
    .trend-panel-title {
        color: #24324A;
        font-size: 1.02rem;
        font-weight: 800;
        margin: 4px 0 8px 0;
    }
    .column-divider {
        height: 0;
        border-left: 0;
    }
    .source-note {
        color: #6B7A90;
        font-size: 0.76rem;
        text-align: right;
        margin-top: 6px;
    }
    .small-note {
        color: #6B7A90;
        font-size: 0.8rem;
    }
    [data-testid="stImage"] img { border-radius: 16px; }

    @media (max-width: 900px) {
        .summary-line { grid-template-columns: 1fr; }
        .report-head h1 { font-size: 1.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- [2번 블록] 색상 교체 ----------------------
TYPE_COLORS = {
    "시설 충분": "#2BB38A",
    "평균 수준": "#5B8DEF",
    "시설 부족": "#FF7A66",
}

FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/Library/Fonts/AppleGothic.ttf"),
]

for font_path in FONT_CANDIDATES:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
        break
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

matplotlib.rcParams["axes.unicode_minus"] = False


REGIONS = [
    "강원",
    "경기",
    "경남",
    "경북",
    "광주",
    "대구",
    "대전",
    "부산",
    "서울",
    "세종",
    "울산",
    "인천",
    "전남",
    "전북",
    "제주",
    "충남",
    "충북",
]

FACILITY_COUNTS = [77, 186, 77, 58, 19, 22, 15, 27, 76, 5, 15, 23, 63, 57, 48, 47, 49]
YOUTH_POPULATION = [20, 230, 50, 35, 25, 35, 22, 42, 130, 7, 18, 45, 26, 28, 11, 33, 25]

TYPE_COLORS = {
    "시설 충분": "#167358",
    "평균 수준": "#2C3840",
    "시설 부족": "#F2AE72",
}

FACILITY_TREND_FILE = Path("facility_trend.csv")
FACILITY_TYPE_FILE = Path("facility_type_counts.csv")
SIGUNGU_POPULATION_FILE = Path("youth_population_sigungu_trend.csv")
SIGUNGU_FACILITY_FILE = Path("facility_sigungu_counts.csv")
FACILITY_TYPE_COLUMNS = [
    "청소년수련관",
    "청소년문화의집",
    "청소년수련원",
    "청소년야영장",
    "유스호스텔",
    "청소년특화시설",
]


@st.cache_data
def build_dataset():
    df = pd.DataFrame(
        {
            "지역": REGIONS,
            "시설수": FACILITY_COUNTS,
            "청소년인구(만명)": YOUTH_POPULATION,
        }
    )
    df["인구10만명당시설수"] = df["시설수"] / df["청소년인구(만명)"] * 10
    return df


def train_model(df):
    features = ["시설수", "청소년인구(만명)", "인구10만명당시설수"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df = df.copy()
    df["군집"] = model.fit_predict(scaled)

    cluster_order = df.groupby("군집")["인구10만명당시설수"].mean().sort_values()
    labels_map = {
        cluster_order.index[0]: "시설 부족",
        cluster_order.index[1]: "평균 수준",
        cluster_order.index[2]: "시설 충분",
    }
    df["지역유형"] = df["군집"].map(labels_map)
    return df, model, scaler, labels_map


def show_facility_chart(df):
    fig, ax = plt.subplots(figsize=(8.8, 3.15))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    bars = ax.bar(df["지역"], df["시설수"], color="#2563eb", edgecolor="#d1d5db", linewidth=0.8)
    ax.set_title("전국 시·도별 청소년 문화·체육 시설 현황", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("지역(시·도)")
    ax.set_ylabel("시설 수(개)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2, int(height), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    st.pyplot(fig)


def show_trend_chart():
    years = ["2021", "2022", "2023", "2024", "2025"]
    ratio = [1.58, 1.61, 1.63, 1.64, 1.65]

    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.plot(years, ratio, marker="o", color="#167358", linestyle="-", linewidth=2.2, markersize=7)
    ax.set_title("연도별 인구 10만 명당 청소년 시설 수 추이", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("연도")
    ax.set_ylabel("시설 수(개)")
    ax.set_ylim(1.50, 1.70)
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    for i, value in enumerate(ratio):
        ax.annotate(f"{value}", (years[i], ratio[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")

    fig.tight_layout()
    st.pyplot(fig)


def load_facility_trend():
    if not FACILITY_TREND_FILE.exists():
        return None

    trend_df = pd.read_csv(FACILITY_TREND_FILE)
    required_columns = {"지역", "연도", "시설수"}
    if not required_columns.issubset(trend_df.columns):
        return None

    return trend_df


def load_facility_type_counts():
    if not FACILITY_TYPE_FILE.exists():
        return None

    type_df = pd.read_csv(FACILITY_TYPE_FILE)
    required_columns = {"지역", *FACILITY_TYPE_COLUMNS}
    if not required_columns.issubset(type_df.columns):
        return None

    return type_df


def load_sigungu_population_trend():
    if not SIGUNGU_POPULATION_FILE.exists():
        return None

    pop_df = pd.read_csv(SIGUNGU_POPULATION_FILE)
    required_columns = {"시도", "시군구", "연도", "청소년인구"}
    if not required_columns.issubset(pop_df.columns):
        return None

    return pop_df


def load_sigungu_facility_counts():
    if not SIGUNGU_FACILITY_FILE.exists():
        return None

    facility_df = pd.read_csv(SIGUNGU_FACILITY_FILE)
    required_columns = {"시도", "시군구", "총계", *FACILITY_TYPE_COLUMNS}
    if not required_columns.issubset(facility_df.columns):
        return None

    return facility_df


def show_facility_type_chart(region, type_df):
    row = type_df[type_df["지역"] == region]
    if row.empty:
        st.markdown(
            '<div class="small-note">선택한 지역의 시설 종류 데이터가 아직 없습니다.</div>',
            unsafe_allow_html=True,
        )
        return

    values = row.iloc[0][FACILITY_TYPE_COLUMNS].astype(int)
    x_limit = max(100, int(type_df[FACILITY_TYPE_COLUMNS].max().max() * 1.12))

    fig, ax = plt.subplots(figsize=(3.6, 2.15))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    short_labels = ["수련관", "문화의집", "수련원", "야영장", "유스호스텔", "특화시설"]
    bars = ax.barh(
        short_labels,
        values,
        color=["#167358", "#2C3840", "#F2AE72", "#8C8C8C", "#732F3B", "#254036"],
        edgecolor="#ffffff",
        linewidth=0.8,
    )
    ax.set_title(f"{region} 시설 종류별 개수", fontsize=8, fontweight="bold", pad=8, y=1.01)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(0, x_limit)
    ax.tick_params(axis="x", labelsize=6, pad=2)
    ax.tick_params(axis="y", labelsize=6, pad=2)
    ax.grid(axis="x", linestyle="-", alpha=0.14)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c6c6c6")
    ax.spines["bottom"].set_color("#c6c6c6")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            min(width + 1.0, x_limit - 2.0),
            bar.get_y() + bar.get_height() / 2,
            int(width),
            ha="left",
            va="center",
            fontsize=6.5,
            color="#0D0D0D",
        )

    fig.subplots_adjust(left=0.22, right=0.95, top=0.82, bottom=0.16)
    st.pyplot(fig, use_container_width=True)


def show_sigungu_facility_type_chart(sido, sigungu, sigungu_facility_df):
    row = sigungu_facility_df[
        (sigungu_facility_df["시도"] == sido) & (sigungu_facility_df["시군구"] == sigungu)
    ]
    if row.empty:
        st.markdown(
            '<div class="small-note">선택한 시군구의 시설 종류 데이터가 없습니다.</div>',
            unsafe_allow_html=True,
        )
        return

    values = row.iloc[0][FACILITY_TYPE_COLUMNS].astype(int)
    x_limit = max(10, int(sigungu_facility_df[FACILITY_TYPE_COLUMNS].max().max() * 1.2))

    fig, ax = plt.subplots(figsize=(3.6, 2.15))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    short_labels = ["수련관", "문화의집", "수련원", "야영장", "유스호스텔", "특화시설"]
    bars = ax.barh(
        short_labels,
        values,
        color=["#167358", "#2C3840", "#F2AE72", "#8C8C8C", "#732F3B", "#254036"],
        edgecolor="#ffffff",
        linewidth=0.8,
    )
    ax.set_title(f"{sigungu} 시설 종류별 개수", fontsize=8, fontweight="bold", pad=8, y=1.01)
    ax.set_xlim(0, x_limit)
    ax.tick_params(axis="x", labelsize=6, pad=2)
    ax.tick_params(axis="y", labelsize=6, pad=2)
    ax.grid(axis="x", linestyle="-", alpha=0.14)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c6c6c6")
    ax.spines["bottom"].set_color("#c6c6c6")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            min(width + 0.2, x_limit - 0.3),
            bar.get_y() + bar.get_height() / 2,
            int(width),
            ha="left",
            va="center",
            fontsize=6.5,
            color="#0D0D0D",
        )

    fig.subplots_adjust(left=0.22, right=0.95, top=0.82, bottom=0.16)
    st.pyplot(fig, use_container_width=True)


def show_region_facility_trend(region, trend_df, large=False):
    region_df = trend_df[trend_df["지역"] == region].sort_values("연도")
    if region_df.empty:
        st.markdown(
            '<div class="small-note">선택한 지역의 연도별 데이터가 아직 없습니다.</div>',
            unsafe_allow_html=True,
        )
        return

    max_facilities = trend_df["시설수"].max()
    y_limit = max(200, int(max_facilities * 1.12))

    fig_size = (3.6, 2.25) if large else (2.7, 1.1)
    title_size = 8 if large else 7.2
    label_size = 6.5 if large else 7
    tick_size = 6.5 if large else 7
    marker_size = 3.5 if large else 4
    line_width = 1.7 if large else 1.8

    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.plot(
        region_df["연도"].astype(str),
        region_df["시설수"],
        marker="o",
        color="#167358",
        linewidth=line_width,
        markersize=marker_size,
    )
    ax.fill_between(
        region_df["연도"].astype(str),
        region_df["시설수"],
        color="#167358",
        alpha=0.08,
    )
    ax.set_title(f"{region} 연도별 시설 수 변화", fontsize=title_size, fontweight="bold", pad=8 if large else 5, y=1.01)
    ax.set_ylabel("시설 수", fontsize=label_size, rotation=0, labelpad=24)
    ax.yaxis.set_label_coords(-0.075, 1.02)
    ax.set_ylim(0, y_limit)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(axis="y", linestyle="-", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c6c6c6")
    ax.spines["bottom"].set_color("#c6c6c6")

    for year, value in zip(region_df["연도"].astype(str), region_df["시설수"]):
        ax.text(year, value, f"{int(value)}", ha="center", va="bottom", fontsize=6 if large else 5.8, color="#0D0D0D")

    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.15)
    st.pyplot(fig, use_container_width=True)


def show_sigungu_ratio_trend(sido, sigungu, facility_count, sigungu_pop_df, sigungu_facility_df):
    region_df = sigungu_pop_df[
        (sigungu_pop_df["시도"] == sido) & (sigungu_pop_df["시군구"] == sigungu)
    ].sort_values("연도")
    region_df = region_df[region_df["청소년인구"] > 0]
    if region_df.empty:
        st.markdown(
            '<div class="small-note">선택한 시군구의 연도별 청소년 인구 데이터가 없습니다.</div>',
            unsafe_allow_html=True,
        )
        return

    merged = sigungu_pop_df.merge(
        sigungu_facility_df[["시도", "시군구", "총계"]],
        on=["시도", "시군구"],
        how="inner",
    )
    merged = merged[merged["청소년인구"] > 0]
    merged["인구10만명당시설수"] = merged["총계"] / merged["청소년인구"] * 100000
    max_ratio = merged["인구10만명당시설수"].replace([float("inf"), -float("inf")], pd.NA).dropna().max()
    y_limit = max(100, int(max_ratio * 1.12)) if pd.notna(max_ratio) else 100

    region_df = region_df.copy()
    region_df["인구10만명당시설수"] = facility_count / region_df["청소년인구"] * 100000

    fig, ax = plt.subplots(figsize=(3.6, 2.25))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.plot(
        region_df["연도"].astype(str),
        region_df["인구10만명당시설수"],
        marker="o",
        color="#167358",
        linewidth=1.7,
        markersize=3.5,
    )
    ax.fill_between(
        region_df["연도"].astype(str),
        region_df["인구10만명당시설수"],
        color="#167358",
        alpha=0.08,
    )
    ax.set_title(f"{sigungu} 인구 10만 명당 시설 수 변화", fontsize=8, fontweight="bold", pad=8, y=1.01)
    ax.set_ylabel("10만 명당", fontsize=6.5, rotation=0, labelpad=24)
    ax.yaxis.set_label_coords(-0.075, 1.02)
    ax.set_ylim(0, y_limit)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.grid(axis="y", linestyle="-", alpha=0.14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c6c6c6")
    ax.spines["bottom"].set_color("#c6c6c6")

    for year, value in zip(region_df["연도"].astype(str), region_df["인구10만명당시설수"]):
        ax.text(year, value, f"{value:.1f}", ha="center", va="bottom", fontsize=6, color="#0D0D0D")

    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.15)
    st.pyplot(fig, use_container_width=True)


def show_cluster_chart(df):
    fig, ax = plt.subplots(figsize=(6.85, 4.25))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    bars = ax.bar(
        df["지역"],
        df["시설수"],
        color=[TYPE_COLORS[value] for value in df["지역유형"]],
        edgecolor="#ffffff",
        linewidth=0.8,
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("시설 수", fontsize=9, rotation=0, labelpad=24)
    ax.yaxis.set_label_coords(-0.055, 1.02)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=6.5, pad=3, rotation=0)
    ax.grid(axis="y", linestyle="-", alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c6c6c6")
    ax.spines["bottom"].set_color("#c6c6c6")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2, int(height), ha="center", va="bottom", fontsize=6.5)

    legend_elements = [
        Patch(facecolor=TYPE_COLORS["시설 충분"], label="시설 충분"),
        Patch(facecolor=TYPE_COLORS["평균 수준"], label="평균 수준"),
        Patch(facecolor=TYPE_COLORS["시설 부족"], label="시설 부족"),
    ]
    ax.legend(handles=legend_elements, title="", fontsize=6.5, frameon=False, loc="upper right")
    ax.margins(x=0.02)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.11)
    st.pyplot(fig, use_container_width=True)


df_base = build_dataset()
df_ml, kmeans, scaler, labels_map = train_model(df_base)
facility_trend_df = load_facility_trend()
facility_type_df = load_facility_type_counts()
sigungu_pop_df = load_sigungu_population_trend()
sigungu_facility_df = load_sigungu_facility_counts()

st.markdown(
    f"""
    <div class="report-head">
        <div class="top-source-note">자료 기준 - 2024년말</div>
        <h1>지역별 청소년 시설 분석</h1>
        <p>시·도별 시설 수와 청소년 인구를 기준으로 시설 수준을 분류하고, 선택한 지역의 연도별 변화를 확인합니다.</p>
    </div>
    <div class="summary-line">
        <div class="summary-item">
            전체 시설 <b>{df_ml['시설수'].sum()}개</b>
        </div>
        <div class="summary-item">
            분석 지역 <b>{len(df_ml)}개 시·도</b>
        </div>
        <div class="summary-item">
            인구 대비 평균 <b>{df_ml['인구10만명당시설수'].mean():.2f}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

graph_col, divider_col, right_col = st.columns([2.0, 0.035, 1.05], gap="small")

with graph_col:
    input_region, input_sigungu, input_facility, input_youth = st.columns(
        [0.88, 0.88, 0.6, 0.74],
        gap="small",
    )

with input_region:
    st.markdown('<div class="input-label">지역</div>', unsafe_allow_html=True)
    selected_region = st.selectbox("지역", REGIONS, index=REGIONS.index("경기"), label_visibility="collapsed")

default_row = df_ml[df_ml["지역"] == selected_region].iloc[0]

sigungu_options = []
if sigungu_pop_df is not None:
    sigungu_options = ["전체"] + sorted(sigungu_pop_df[sigungu_pop_df["시도"] == selected_region]["시군구"].unique())

selected_sigungu = None
if sigungu_options:
    with input_sigungu:
        st.markdown('<div class="input-label">시군구</div>', unsafe_allow_html=True)
        selected_sigungu_value = st.selectbox("시군구", sigungu_options, label_visibility="collapsed")
        selected_sigungu = None if selected_sigungu_value == "전체" else selected_sigungu_value

with input_facility:
    st.markdown('<div class="input-label">시설 수</div>', unsafe_allow_html=True)
    if selected_sigungu and sigungu_facility_df is not None:
        facility_row = sigungu_facility_df[
            (sigungu_facility_df["시도"] == selected_region)
            & (sigungu_facility_df["시군구"] == selected_sigungu)
        ]
        facility_count = int(facility_row.iloc[0]["총계"]) if not facility_row.empty else 0
    else:
        facility_count = int(default_row["시설수"])
    st.markdown(f'<div class="readonly-value">{facility_count}</div>', unsafe_allow_html=True)

with input_youth:
    st.markdown('<div class="input-label">청소년 인구(만 명)</div>', unsafe_allow_html=True)
    if selected_sigungu and sigungu_pop_df is not None:
        youth_row = sigungu_pop_df[
            (sigungu_pop_df["시도"] == selected_region)
            & (sigungu_pop_df["시군구"] == selected_sigungu)
            & (sigungu_pop_df["연도"] == 2024)
        ]
        youth_count = float(youth_row.iloc[0]["청소년인구"]) / 10000 if not youth_row.empty else 0.0
        youth_display = f"{youth_count:.1f}"
    else:
        youth_count = float(default_row["청소년인구(만명)"])
        youth_display = f"{int(youth_count)}"
    st.markdown(f'<div class="readonly-value">{youth_display}</div>', unsafe_allow_html=True)

ratio_value = facility_count / youth_count * 10 if youth_count else 0
input_data = pd.DataFrame(
    {
        "시설수": [facility_count],
        "청소년인구(만명)": [youth_count],
        "인구10만명당시설수": [ratio_value],
    }
)
input_scaled = scaler.transform(input_data)
input_cluster = kmeans.predict(input_scaled)[0]
result_type = labels_map[input_cluster]

if result_type == "시설 부족":
    result_class = "bad"
    message = "인구 변화 관찰 필요"
elif result_type == "평균 수준":
    result_class = "mid"
    message = "인구 변화 관찰 필요"
else:
    result_class = "good"
    message = "현재 수준 유지 가능"

selected_label = selected_sigungu if selected_sigungu else selected_region

with graph_col:
    show_cluster_chart(df_ml)
    st.markdown(
        f"""
        <div class="right-result-title {result_class}">{selected_label} {result_type}</div>
        <div class="right-result-card {result_class}">
            <div class="right-main">시설 {facility_count}개 * 10만 명당 {ratio_value:.2f}</div>
            <div class="right-warning">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with divider_col:
    st.markdown('<div class="column-divider"></div>', unsafe_allow_html=True)

with right_col:
    if selected_sigungu and sigungu_pop_df is not None and sigungu_facility_df is not None:
        st.markdown('<div class="trend-panel-title">지역 요약 그래프</div>', unsafe_allow_html=True)
        show_sigungu_ratio_trend(selected_region, selected_sigungu, facility_count, sigungu_pop_df, sigungu_facility_df)
    elif facility_trend_df is not None:
        st.markdown('<div class="trend-panel-title">지역 요약 그래프</div>', unsafe_allow_html=True)
        show_region_facility_trend(selected_region, facility_trend_df, large=True)
    else:
        st.markdown(
            '<div class="small-note">facility_trend.csv 파일을 추가하면 연도별 시설 수 그래프가 표시됩니다.</div>',
            unsafe_allow_html=True,
        )

    if selected_sigungu and sigungu_facility_df is not None:
        st.markdown('<div class="trend-panel-title">시설 종류 구성</div>', unsafe_allow_html=True)
        show_sigungu_facility_type_chart(selected_region, selected_sigungu, sigungu_facility_df)
    elif facility_type_df is not None:
        st.markdown('<div class="trend-panel-title">시설 종류 구성</div>', unsafe_allow_html=True)
        show_facility_type_chart(selected_region, facility_type_df)
    else:
        st.markdown(
            '<div class="small-note">facility_type_counts.csv 파일을 추가하면 시설 종류별 개수가 표시됩니다.</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="source-note">자료 기준 : 2024년말</div>', unsafe_allow_html=True)
