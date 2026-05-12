"""
=============================================================================
SG LABOUR MARKET INTELLIGENCE DASHBOARD  v2
=============================================================================
Stack  : Streamlit · DuckDB · SQL · Pandas · NumPy · Plotly
Data   : SGJobData_clean.parquet  (run data_cleaning.py first)

Run:
    streamlit run sg_labour_dashboard.py
=============================================================================
"""
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# Path resolves relative to this script's location (dashboard/)
# so we go up one level to project root, then into data/clean/
DATA_PATH = Path(__file__).parent.parent / "data" / "clean" / "SGJobData_clean.parquet"

# Fallback: if parquet not found at expected path, check same folder as script
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent / "SGJobData_clean.parquet"

if not DATA_PATH.exists():
    import streamlit as st
    st.error(
        f"❌ Cannot find SGJobData_clean.parquet.\n\n"
        f"Expected at: `{DATA_PATH}`\n\n"
        f"Run the cleaning pipeline first:\n"
        f"```\npython pipeline/data_cleaning.py\n```"
    )
    st.stop()

st.set_page_config(
    page_title="SG Labour Market Intelligence",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = dict(
    red="#C8102E", navy="#002B5C", gold="#F5A623", teal="#16A085",
    purple="#7D3C98", sky="#2980B9", orange="#E67E22", aqua="#1ABC9C",
    grey="#6C757D",
)
SEQ  = [C["red"], C["navy"], C["gold"], C["teal"], C["purple"], C["sky"], C["orange"], C["aqua"]]
CONT = ["#002B5C", "#6B3FA0", "#C8102E", "#F5A623"]

st.markdown(f"""
<style>
.main .block-container {{ max-width:1440px; padding-top:1.2rem; }}
.kpi {{ background:#fff; border-radius:12px; padding:16px 18px;
        box-shadow:0 2px 8px rgba(0,0,0,.07); border-top:4px solid {C["red"]}; margin-bottom:4px; }}
.kpi-val {{ font-size:1.85rem; font-weight:800; line-height:1.1; }}
.kpi-lbl {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.07em;
            color:{C["grey"]}; margin-top:5px; }}
.sec {{ font-weight:700; font-size:.95rem; color:{C["navy"]};
        padding-bottom:5px; border-bottom:2px solid {C["red"]}; margin:14px 0 8px 0; }}
.ibox {{ border-radius:8px; padding:11px 15px; margin:6px 0;
         font-size:.85rem; line-height:1.45; }}
.ibox-blue  {{ background:#EBF5FB; border-left:3px solid {C["sky"]}; }}
.ibox-gold  {{ background:#FEF9E7; border-left:3px solid {C["gold"]}; }}
.ibox-red   {{ background:#FDEDEC; border-left:3px solid {C["red"]}; }}
.ibox-teal  {{ background:#E8F8F5; border-left:3px solid {C["teal"]}; }}
.ibox-grey  {{ background:#F4F6F6; border-left:3px solid {C["grey"]}; }}
#MainMenu, footer {{ visibility:hidden; }}
/* Force sidebar toggle button to always be visible */
[data-testid="collapsedControl"] {{
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    background-color: {C["red"]} !important;
    border-radius: 0 6px 6px 0 !important;
}}
[data-testid="collapsedControl"] svg {{
    fill: white !important;
}}
</style>
""", unsafe_allow_html=True)


def kpi(col, val, lbl, accent=None):
    accent = accent or C["red"]
    col.markdown(
        f'<div class="kpi" style="border-top-color:{accent}">'
        f'<div class="kpi-val" style="color:{accent}">{val}</div>'
        f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

def ibox(text, style="blue"):
    st.markdown(f'<div class="ibox ibox-{style}">{text}</div>', unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)

def sql_list(vals):
    safe = [str(v).replace("'", "''") for v in vals]
    return "(" + ", ".join(f"'{v}'" for v in safe) + ")"

def fmt_k(n):
    if n >= 1_000_000: return f"{n/1e6:.1f}M"
    if n >= 1_000:     return f"{n/1e3:.0f}K"
    return f"{int(n):,}"


# ── Engine: load parquet → DuckDB ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳  Loading clean dataset …")
def get_engine(path: str):
    df = pd.read_parquet(path)
    con = duckdb.connect()
    con.register("jobs", df)
    return con, df

try:
    con, df_raw = get_engine(str(DATA_PATH))
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.info(f"Looking for: `{DATA_PATH}`")
    st.stop()


# ── Sidebar Filters ──────────────────────────────────────────────────────────
d_min = df_raw["posting_date"].min().date()
d_max = df_raw["posting_date"].max().date()
level_order = ["Fresh/entry level","Non-executive","Junior Executive","Executive",
               "Senior Executive","Professional","Manager","Middle Management","Senior Management"]
all_levels = [l for l in level_order if l in df_raw["position_level"].unique()]
all_status = sorted(df_raw["job_status"].dropna().unique())
all_emp    = sorted(df_raw["employmentTypes"].dropna().unique())
all_cat    = sorted(df_raw["category"].unique())
s_lo = int(df_raw["salary"].quantile(0.01))
s_hi = int(df_raw["salary"].quantile(0.99))

with st.sidebar:
    st.markdown("## 🇸🇬 SG Labour Intel")
    st.markdown("*MCF Job Postings · Oct 2022 – May 2024*")
    st.divider()
    d_range    = st.date_input("📅 Date Range", (d_min, d_max),
                               min_value=d_min, max_value=d_max)
    sel_status = st.multiselect("📌 Job Status", all_status, default=all_status)
    sel_levels = st.multiselect("🏷 Position Level", all_levels, default=all_levels)
    sel_emp    = st.multiselect("💼 Employment Type", all_emp,
                                default=["Permanent","Full Time","Contract"])
    sel_cat    = st.multiselect("🏭 Industry Category", all_cat,
                                placeholder="All categories (default)")
    sal_rng    = st.slider("💰 Monthly Salary (S$)", s_lo, s_hi, (s_lo, s_hi), step=200)
    st.divider()
    st.caption(f"Dataset: **{len(df_raw):,}** records")
    st.caption("**HFI** Hiring Friction Index · **ACR** App Conversion Rate · **EBI** Exp Barrier Index")


# ── WHERE clause ──────────────────────────────────────────────────────────────
clauses = ["salary_valid = true", f"salary BETWEEN {sal_rng[0]} AND {sal_rng[1]}"]
if len(d_range) == 2:
    clauses.append(f"posting_date >= '{d_range[0]}' AND posting_date <= '{d_range[1]}'")
if sel_status:  clauses.append(f"job_status IN {sql_list(sel_status)}")
if sel_levels:  clauses.append(f"position_level IN {sql_list(sel_levels)}")
if sel_emp:     clauses.append(f"employmentTypes IN {sql_list(sel_emp)}")
if sel_cat:     clauses.append(f"category IN {sql_list(sel_cat)}")
WHERE = " AND ".join(clauses)

df = con.execute(f"SELECT * FROM jobs WHERE {WHERE}").df()
N  = len(df)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "## 🇸🇬 Singapore Labour Market Intelligence Dashboard\n"
    "*Transforming MCF job-posting metadata into leading indicators of national economic health and workforce productivity.*"
)
st.caption(f"Showing **{N:,}** salary-valid records · Use sidebar to filter by date, sector, level or salary band")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  National Overview",
    "🔬  Bottlenecks & Friction",
    "📈  Time Trends",
    "🔎  Data Quality & EDA",
])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — NATIONAL OVERVIEW                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown("### 📊 National Labour Market Pulse")
    st.caption("**For:** MOM Policy Analysts · Enterprise Singapore · National Planners  |  Macro demand, salary benchmarks, sector health")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    open_n   = int((df["job_status"]=="Open").sum())
    med_sal  = df["salary"].median()
    tot_vac  = int(df["vacancies"].sum())
    rng_rate = df["has_salary_range"].mean()*100
    avg_apps = df["applications"].mean()

    kpi(k1, fmt_k(N),            "Total Postings",            C["navy"])
    kpi(k2, fmt_k(open_n),       "Open / Active Roles",       C["red"])
    kpi(k3, f"S${med_sal:,.0f}", "Median Monthly Salary",     C["teal"])
    kpi(k4, fmt_k(tot_vac),      "Total Vacancies",           C["gold"])
    kpi(k5, f"{rng_rate:.1f}%",  "Salary Range Disclosure",   C["purple"])
    kpi(k6, f"{avg_apps:.1f}",   "Avg Applications / Post",   C["sky"])
    st.markdown("")

    c1, c2 = st.columns([6,4])
    with c1:
        sec("Top 20 Industry Categories — Volume & Median Salary")
        cat_df = con.execute(f"""
            SELECT category, COUNT(*) AS postings, SUM(vacancies) AS total_vac,
                   MEDIAN(salary) AS med_salary, AVG(hfi) AS avg_hfi
            FROM jobs WHERE {WHERE}
            GROUP BY category ORDER BY postings DESC LIMIT 20
        """).df().sort_values("postings")
        fig = px.bar(cat_df, x="postings", y="category", orientation="h",
                     color="med_salary", color_continuous_scale=CONT,
                     labels={"postings":"Job Postings","category":"","med_salary":"Median Salary (S$)"},
                     hover_data={"total_vac":True,"avg_hfi":":.3f","med_salary":":,.0f"},
                     template="plotly_white")
        fig.update_coloraxes(colorbar_title="Median<br>Salary (S$)")
        fig.update_layout(height=520, margin=dict(l=0,r=10,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ibox("💡 <b>Policy signal:</b> Dark-blue = high-volume / lower-wage sectors — prime PSG automation candidates where technology reduces labour dependency without cutting output.", "blue")

    with c2:
        sec("Position Level Split")
        lv_df = con.execute(f"""
            SELECT position_level AS lvl, COUNT(*) AS n
            FROM jobs WHERE {WHERE} AND position_level IS NOT NULL
            GROUP BY lvl ORDER BY n DESC
        """).df()
        fig = px.pie(lv_df, values="n", names="lvl", hole=0.44, color_discrete_sequence=SEQ)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          hovertemplate="%{label}<br>%{value:,} posts (%{percent})")
        fig.update_layout(height=290, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        sec("Employment Type Mix")
        emp_df = con.execute(f"""
            SELECT employmentTypes AS etype, COUNT(*) AS n
            FROM jobs WHERE {WHERE} AND employmentTypes IS NOT NULL
            GROUP BY etype ORDER BY n DESC LIMIT 7
        """).df()
        fig2 = px.bar(emp_df, x="n", y="etype", orientation="h",
                      color_discrete_sequence=[C["navy"]], template="plotly_white",
                      labels={"n":"Postings","etype":""})
        fig2.update_layout(height=260, margin=dict(l=0,r=10,t=0,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns([4,6])
    with c3:
        sec("Monthly Salary Distribution (S$) — valid records only")
        sal_df = con.execute(f"""
            SELECT salary FROM jobs
            WHERE {WHERE} AND salary BETWEEN 500 AND {int(df['salary'].quantile(0.98))}
        """).df()
        med_v = sal_df["salary"].median()
        p25_v = sal_df["salary"].quantile(0.25)
        p75_v = sal_df["salary"].quantile(0.75)
        fig = px.histogram(sal_df, x="salary", nbins=65, color_discrete_sequence=[C["red"]],
                           labels={"salary":"Monthly Salary (S$)"}, template="plotly_white")
        for xv, lbl, col in [(p25_v,f"P25 S${p25_v:,.0f}",C["sky"]),
                             (med_v,f"Median S${med_v:,.0f}",C["navy"]),
                             (p75_v,f"P75 S${p75_v:,.0f}",C["sky"])]:
            fig.add_vline(x=xv, line_dash="dot" if xv!=med_v else "dash",
                          line_color=col, annotation_text=lbl,
                          annotation_font_color=col, annotation_font_size=10)
        fig.update_traces(hovertemplate="S$%{x:,.0f}<br>Count: %{y:,}")
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        ibox(f"📌 <b>EDA:</b> Right-skewed — IQR S${p25_v:,.0f}–S${p75_v:,.0f}. All charts use "
             f"<b>median</b>, not mean, to avoid distortion from senior-tier outliers. "
             f"Policy focus: workers below S$2,900 (P25) for SkillsFuture subsidies.", "blue")

    with c4:
        sec("Top 25 Job Titles — Volume & Salary Colour")
        ttl_df = con.execute(f"""
            SELECT title, COUNT(*) AS postings, MEDIAN(salary) AS med_salary,
                   SUM(vacancies) AS total_vac
            FROM jobs WHERE {WHERE}
            GROUP BY title ORDER BY postings DESC LIMIT 25
        """).df().sort_values("postings")
        fig = px.bar(ttl_df, x="postings", y="title", orientation="h",
                     color="med_salary", color_continuous_scale=CONT,
                     labels={"postings":"Postings","title":"","med_salary":"Median Salary"},
                     hover_data={"total_vac":True,"med_salary":":,.0f"}, template="plotly_white")
        fig.update_coloraxes(colorbar_title="Median<br>Salary (S$)")
        fig.update_layout(height=520, margin=dict(l=0,r=10,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    sec("Salary Benchmark by Position Level — Median with Interquartile Range")
    sal_lv = con.execute(f"""
        SELECT position_level AS lvl, MEDIAN(salary) AS med,
               QUANTILE_CONT(salary,0.25) AS p25, QUANTILE_CONT(salary,0.75) AS p75,
               COUNT(*) AS n
        FROM jobs WHERE {WHERE} AND position_level IS NOT NULL
        GROUP BY lvl ORDER BY med
    """).df()
    fig = go.Figure(go.Bar(
        x=sal_lv["lvl"], y=sal_lv["med"], name="Median", marker_color=C["red"],
        error_y=dict(type="data", symmetric=False,
                     array=(sal_lv["p75"]-sal_lv["med"]).tolist(),
                     arrayminus=(sal_lv["med"]-sal_lv["p25"]).tolist(),
                     color=C["navy"], thickness=2.5, width=7),
        customdata=sal_lv[["n","p25","p75"]].values,
        hovertemplate="%{x}<br>Median: S$%{y:,.0f}<br>P25: S$%{customdata[1]:,.0f}  P75: S$%{customdata[2]:,.0f}<br>n=%{customdata[0]:,}<extra></extra>",
    ))
    fig.update_layout(height=340, template="plotly_white", yaxis_title="Monthly Salary (S$)",
                      margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    ibox("💡 Error bars = IQR (P25–P75). Wide IQR at Manager/Senior levels = intense compensation competition. HR teams benchmark against P75 to retain top performers.", "gold")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — BOTTLENECKS & FRICTION                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown("### 🔬 Labor Bottlenecks, Hiring Friction & Workforce Analytics")
    st.caption("**For:** Business Leaders · HR Professionals · Workforce Development Bodies  |  Mismatch diagnosis, automation targeting, over-specification detection")

    b1,b2,b3,b4,b5 = st.columns(5)
    avg_hfi   = float(df["hfi"].mean())
    avg_acr   = float(df["acr"].mean(skipna=True))*100
    avg_ebi   = float(df["ebi"].mean(skipna=True))
    repost_rt = float((df["repost_count"]>0).mean())*100
    acr_range = float(df.loc[df["has_salary_range"]==1,"acr"].mean(skipna=True))*100
    acr_point = float(df.loc[df["has_salary_range"]==0,"acr"].mean(skipna=True))*100
    rng_lift  = acr_range - acr_point

    kpi(b1, f"{avg_hfi:.3f}",    "Avg Hiring Friction Index",       C["red"])
    kpi(b2, f"{avg_acr:.1f}%",   "Avg App Conversion Rate",         C["navy"])
    kpi(b3, f"{avg_ebi:.1f}",    "Avg Exp Barrier Index",           C["gold"])
    kpi(b4, f"{repost_rt:.1f}%", "Repost Rate",                     C["purple"])
    kpi(b5, f"+{rng_lift:.1f}%", "ACR Lift: Range vs Point Salary", C["teal"])
    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        sec("Hiring Friction Index (HFI) — by Industry (≥50 postings)")
        hfi_df = con.execute(f"""
            SELECT category, AVG(hfi) AS avg_hfi, COUNT(*) AS n,
                   SUM(vacancies) AS total_vac, AVG(repost_count) AS avg_repost,
                   MEDIAN(salary) AS med_salary
            FROM jobs WHERE {WHERE}
            GROUP BY category HAVING n >= 50
            ORDER BY avg_hfi DESC LIMIT 20
        """).df()
        fig = px.bar(hfi_df, x="avg_hfi", y="category", orientation="h",
                     color="avg_hfi", color_continuous_scale=["#27AE60","#F5A623","#C8102E"],
                     labels={"avg_hfi":"Avg HFI (0–1)","category":""},
                     hover_data={"n":True,"total_vac":True,"avg_repost":":.2f","med_salary":":,.0f"},
                     template="plotly_white")
        fig.add_vline(x=avg_hfi, line_dash="dash", line_color=C["navy"],
                      annotation_text=f"Market avg: {avg_hfi:.3f}", annotation_font_color=C["navy"])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=480, margin=dict(l=0,r=10,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ibox("🚨 <b>Policy:</b> Sectors above market-average HFI = prime PSG funding candidates. Persistent friction signals roles where automation delivers highest productivity return.", "red")

    with c2:
        sec("Application Conversion Rate (ACR) — by Industry (worst first)")
        acr_df = con.execute(f"""
            SELECT category, AVG(acr) AS avg_acr, COUNT(*) AS n,
                   MEDIAN(min_exp_years) AS med_exp, MEDIAN(salary) AS med_salary
            FROM jobs WHERE {WHERE} AND acr IS NOT NULL
            GROUP BY category HAVING n >= 50
            ORDER BY avg_acr ASC LIMIT 20
        """).df()
        fig = px.bar(acr_df, x="avg_acr", y="category", orientation="h",
                     color="avg_acr", color_continuous_scale=["#C8102E","#F5A623","#27AE60"],
                     labels={"avg_acr":"Avg ACR","category":""},
                     hover_data={"n":True,"med_exp":True,"med_salary":":,.0f"},
                     template="plotly_white")
        fig.update_xaxes(tickformat=".0%")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=480, margin=dict(l=0,r=10,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ibox("💡 <b>HR:</b> Low ACR + adequate views = posting quality or pay band problem — not a talent shortage. Fix JD language and salary range first before escalating to PSG.", "blue")

    sec("Experience Barrier Analysis — Years Required vs. Salary Offered  (bubble = vacancy volume)")
    ebi_df = con.execute(f"""
        SELECT category, AVG(min_exp_years) AS avg_exp, MEDIAN(salary) AS med_salary,
               SUM(vacancies) AS total_vac, AVG(ebi) AS avg_ebi, COUNT(*) AS n
        FROM jobs WHERE {WHERE} AND salary > 500
        GROUP BY category HAVING n >= 30
        ORDER BY avg_ebi DESC
    """).df()
    med_exp_v = ebi_df["avg_exp"].median()
    med_sal_v = ebi_df["med_salary"].median()
    fig = px.scatter(ebi_df, x="avg_exp", y="med_salary",
                     size="total_vac", color="avg_ebi", text="category",
                     color_continuous_scale=["#27AE60","#F5A623","#C8102E"],
                     labels={"avg_exp":"Avg Min. Years Experience Required",
                             "med_salary":"Median Monthly Salary (S$)","avg_ebi":"EBI"},
                     template="plotly_white", hover_name="category",
                     hover_data={"total_vac":True,"avg_ebi":":.1f","n":True})
    fig.update_traces(textposition="top center", textfont_size=8,
                      marker=dict(opacity=0.8, line=dict(width=0.5, color="white")))
    fig.add_hline(y=med_sal_v, line_dash="dot", line_color="grey", opacity=0.4)
    fig.add_vline(x=med_exp_v, line_dash="dot", line_color="grey", opacity=0.4)
    for x, y, txt, col in [
        (ebi_df["avg_exp"].max()*0.88, ebi_df["med_salary"].max()*0.96, "High exp · High pay<br><i>(Premium talent)</i>",    C["teal"]),
        (ebi_df["avg_exp"].max()*0.88, ebi_df["med_salary"].min()*1.08, "⚠️ High exp · Low pay<br><i>(OVER-SPECIFIED)</i>",  C["red"]),
        (ebi_df["avg_exp"].min(),      ebi_df["med_salary"].max()*0.96, "Low exp · High pay<br><i>(Emerging roles)</i>",    C["navy"]),
        (ebi_df["avg_exp"].min(),      ebi_df["med_salary"].min()*1.08, "Low exp · Low pay<br><i>(Entry volume)</i>",       C["grey"]),
    ]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(size=9, color=col))
    fig.update_coloraxes(colorbar_title="EBI")
    fig.update_layout(height=520, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    ibox("⚠️ <b>EDA:</b> Experience capped at 30 yrs after finding 39 records requesting 31–88 years (data errors). Bottom-right quadrant = over-specification trap: firms blocking their own talent pipeline. Bubble = total vacancies → larger = higher urgency.", "gold")

    c3, c4 = st.columns([5,5])
    with c3:
        sec("Top 20 Companies — Unfilled Vacancy Volume & Friction Colour")
        comp_df = con.execute(f"""
            SELECT company, COUNT(*) AS postings, SUM(vacancies) AS total_vac,
                   AVG(hfi) AS avg_hfi, AVG(acr) AS avg_acr, MEDIAN(salary) AS med_salary
            FROM jobs WHERE {WHERE} AND company IS NOT NULL
            GROUP BY company HAVING postings >= 15
            ORDER BY total_vac DESC LIMIT 20
        """).df().sort_values("total_vac")
        fig = px.bar(comp_df, x="total_vac", y="company", orientation="h",
                     color="avg_hfi", color_continuous_scale=["#27AE60","#F5A623","#C8102E"],
                     labels={"total_vac":"Total Vacancies","company":"","avg_hfi":"Avg HFI"},
                     hover_data={"postings":True,"avg_acr":":.1%","med_salary":":,.0f"},
                     template="plotly_white")
        fig.update_coloraxes(colorbar_title="Avg HFI")
        fig.update_layout(height=480, margin=dict(l=0,r=10,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sec("Pay Range vs Point Estimate — ACR Impact by Sector")
        disc_cat = con.execute(f"""
            SELECT category,
                   AVG(CASE WHEN has_salary_range=1 THEN acr END) AS acr_range,
                   AVG(CASE WHEN has_salary_range=0 THEN acr END) AS acr_point,
                   COUNT(*) AS n
            FROM jobs WHERE {WHERE} AND acr IS NOT NULL
            GROUP BY category HAVING n >= 100
            ORDER BY (AVG(CASE WHEN has_salary_range=1 THEN acr END)
                    - AVG(CASE WHEN has_salary_range=0 THEN acr END)) DESC LIMIT 15
        """).df().dropna(subset=["acr_range","acr_point"])
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Salary Range (min≠max)", x=disc_cat["category"],
                             y=disc_cat["acr_range"], marker_color=C["teal"],
                             hovertemplate="%{x}<br>ACR (range): %{y:.1%}"))
        fig.add_trace(go.Bar(name="Point Estimate (min=max)", x=disc_cat["category"],
                             y=disc_cat["acr_point"], marker_color=C["red"],
                             hovertemplate="%{x}<br>ACR (point): %{y:.1%}"))
        fig.update_yaxes(tickformat=".0%", title="Avg App Conversion Rate")
        fig.update_layout(barmode="group", height=380, template="plotly_white",
                          legend=dict(x=0.01,y=0.99),
                          margin=dict(l=0,r=0,t=0,b=0), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        ibox(f"💡 <b>EDA finding:</b> Salary range postings achieve <b>+{rng_lift:.1f}%</b> higher ACR than point estimates. "
             "The 1.4% point-estimate postings (where min=max) are a quick-win improvement target for HR teams.", "teal")

    sec("Hiring Friction Pattern by Salary Band — Repost Rate & Avg HFI")
    band_df = con.execute(f"""
        SELECT salary_band,
               AVG(CASE WHEN repost_count>0 THEN 1.0 ELSE 0.0 END) AS repost_rate,
               AVG(hfi) AS avg_hfi, COUNT(*) AS n
        FROM jobs WHERE {WHERE} AND salary_band IS NOT NULL
        GROUP BY salary_band ORDER BY salary_band
    """).df()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=band_df["salary_band"], y=band_df["repost_rate"],
                         name="Repost Rate", marker_color=C["red"], opacity=0.8,
                         hovertemplate="%{x}<br>Repost: %{y:.1%}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=band_df["salary_band"], y=band_df["avg_hfi"],
                             name="Avg HFI", line=dict(color=C["navy"],width=2.5),
                             mode="lines+markers",
                             hovertemplate="%{x}<br>Avg HFI: %{y:.3f}"), secondary_y=True)
    fig.update_yaxes(title_text="Repost Rate", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Avg HFI", secondary_y=True)
    fig.update_layout(height=350, template="plotly_white", hovermode="x unified",
                      legend=dict(x=0.01,y=0.98), margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    ibox("⚠️ <b>EDA:</b> Friction peaks at S$2–5K bands — mass-market segment. Raising offer bands modestly at S$2–3.5K tier cuts repost cycles more cost-effectively than large senior salary jumps.", "gold")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — TIME TRENDS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown("### 📈 Time Trends, Wage Pressure & Leading Indicators")
    st.caption("**For:** All user groups — cyclical vs structural demand, leading signals before official stats")

    sec("Monthly Job Posting Volume & Total Vacancies — Dual Axis")
    vol_df = con.execute(f"""
        SELECT year_month, COUNT(*) AS postings, SUM(vacancies) AS vacancies,
               MEDIAN(salary) AS med_salary, AVG(hfi) AS avg_hfi
        FROM jobs WHERE {WHERE}
        GROUP BY year_month ORDER BY year_month
    """).df()
    vol_df["year_month"] = pd.to_datetime(vol_df["year_month"])
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=vol_df["year_month"], y=vol_df["postings"],
                         name="Job Postings", marker_color=C["navy"], opacity=0.72,
                         hovertemplate="%{x|%b %Y}<br>Postings: %{y:,}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=vol_df["year_month"], y=vol_df["vacancies"],
                             name="Total Vacancies", line=dict(color=C["red"],width=2.5),
                             mode="lines+markers",
                             hovertemplate="%{x|%b %Y}<br>Vacancies: %{y:,}"), secondary_y=True)
    fig.update_yaxes(title_text="Job Postings", secondary_y=False)
    fig.update_yaxes(title_text="Total Vacancies", secondary_y=True)
    fig.update_layout(height=380, template="plotly_white", hovermode="x unified",
                      legend=dict(x=0.01,y=0.98), margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    ibox("💡 <b>EDA:</b> Oct 2022–Mar 2023 is a platform ramp-up (172 → 70K posts/month). "
         "<b>Do not interpret as demand growth.</b> Vacancy-to-posting gap is the real leading indicator.", "blue")

    c1, c2 = st.columns(2)
    with c1:
        sec("Median Salary Trend — Key Position Levels")
        sal_trend = con.execute(f"""
            SELECT year_month, position_level, MEDIAN(salary) AS med_salary, COUNT(*) AS n
            FROM jobs WHERE {WHERE}
              AND position_level IN ('Fresh/entry level','Executive','Professional','Manager','Senior Management')
            GROUP BY year_month, position_level ORDER BY year_month
        """).df()
        sal_trend["year_month"] = pd.to_datetime(sal_trend["year_month"])
        fig = px.line(sal_trend, x="year_month", y="med_salary", color="position_level",
                      labels={"med_salary":"Median Monthly Salary (S$)","year_month":"","position_level":"Level"},
                      color_discrete_sequence=SEQ, markers=True, template="plotly_white")
        fig.update_layout(height=370, hovermode="x unified",
                          legend=dict(x=0.01,y=0.98), margin=dict(l=0,r=0,t=0,b=0))
        fig.update_traces(hovertemplate="%{y:,.0f}")
        st.plotly_chart(fig, use_container_width=True)
        ibox("💡 Gradient widening = healthy productivity-linked wage growth. Compression (lines converging) = retention risk — signal for talent pressure ahead.", "blue")

    with c2:
        sec("Structural vs Cyclical: Top Growing & Declining Sectors")
        gro_df = con.execute(f"""
            WITH h1 AS (SELECT category, COUNT(*) AS early FROM jobs
                        WHERE {WHERE} AND year_month < '2023-07-01' GROUP BY category),
            h2 AS (SELECT category, COUNT(*) AS late FROM jobs
                   WHERE {WHERE} AND year_month >= '2023-07-01' GROUP BY category)
            SELECT h1.category, h1.early, h2.late,
                   ROUND(((h2.late-h1.early)/CAST(NULLIF(h1.early,0) AS DOUBLE))*100,1) AS pct_change
            FROM h1 JOIN h2 ON h1.category=h2.category
            WHERE h1.early >= 200 AND h2.late >= 200
            ORDER BY pct_change
        """).df().dropna()
        gro_chart = pd.concat([gro_df.tail(6).iloc[::-1], gro_df.head(6)]).reset_index(drop=True)
        gro_chart["colour"] = gro_chart["pct_change"].apply(lambda x: C["teal"] if x>0 else C["red"])
        fig = go.Figure(go.Bar(x=gro_chart["pct_change"], y=gro_chart["category"],
                               orientation="h", marker_color=gro_chart["colour"],
                               customdata=gro_chart[["early","late"]].values,
                               hovertemplate="%{y}<br>Change: %{x:.1f}%<br>"
                                             "H1 2023: %{customdata[0]:,} | H2 2023+: %{customdata[1]:,}<extra></extra>"))
        fig.add_vline(x=0, line_color=C["grey"], line_width=1)
        fig.update_layout(height=370, template="plotly_white",
                          xaxis_title="% Change", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ibox("💡 Sustained growth (green) = structural demand → permanent SkillsFuture pipelines. Declines (red) = automation displacement → retraining priority.", "gold")

    c3, c4 = st.columns(2)
    with c3:
        sec("Hiring Friction Index Trend — Monthly Market Average")
        hfi_t = con.execute(f"""
            SELECT year_month, AVG(hfi) AS avg_hfi,
                   AVG(CASE WHEN repost_count>0 THEN 1.0 ELSE 0.0 END) AS repost_rate
            FROM jobs WHERE {WHERE}
            GROUP BY year_month ORDER BY year_month
        """).df()
        hfi_t["year_month"] = pd.to_datetime(hfi_t["year_month"])
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=hfi_t["year_month"], y=hfi_t["avg_hfi"],
                                 name="Avg HFI", line=dict(color=C["red"],width=2.5),
                                 mode="lines+markers",
                                 hovertemplate="%{x|%b %Y}<br>HFI: %{y:.3f}"), secondary_y=False)
        fig.add_trace(go.Bar(x=hfi_t["year_month"], y=hfi_t["repost_rate"],
                             name="Repost Rate", marker_color=C["gold"], opacity=0.55,
                             hovertemplate="%{x|%b %Y}<br>Repost: %{y:.1%}"), secondary_y=True)
        fig.update_yaxes(title_text="Avg HFI (0–1)", secondary_y=False)
        fig.update_yaxes(title_text="Repost Rate", tickformat=".0%", secondary_y=True)
        fig.update_layout(height=350, template="plotly_white", hovermode="x unified",
                          legend=dict(x=0.01,y=0.98), margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ibox("💡 Rising HFI = leading indicator of upcoming wage pressure — visible 1–2 quarters before official MOM employment statistics.", "blue")

    with c4:
        sec("Repost Timing Distribution — Days from Original to Re-listing")
        rep_days = con.execute(f"""
            SELECT days_to_repost AS d, COUNT(*) AS n
            FROM jobs WHERE {WHERE} AND repost_count>0 AND days_to_repost>0
            GROUP BY d ORDER BY d
        """).df()
        mean_days = (rep_days["d"]*rep_days["n"]).sum()/rep_days["n"].sum() if len(rep_days) else 67
        fig = px.histogram(rep_days, x="d", y="n", nbins=40,
                           color_discrete_sequence=[C["purple"]],
                           labels={"d":"Days from First to Re-listing","n":"Count"},
                           template="plotly_white")
        fig.add_vline(x=mean_days, line_dash="dash", line_color=C["navy"],
                      annotation_text=f"Mean: {mean_days:.0f} days",
                      annotation_font_color=C["navy"])
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ibox(f"💡 <b>EDA:</b> Mean {mean_days:.0f}-day repost gap = employers wait ~2 full 30-day listing cycles before retrying. "
             "Actual vacancy duration is 2–3× the listing period — friction is chronically underestimated.", "gold")

    sec("Sector Activity Heatmap — Quarterly Posting Intensity (row-normalised)")
    heat_df = con.execute(f"""
        SELECT quarter, category, COUNT(*) AS n
        FROM jobs WHERE {WHERE}
        GROUP BY quarter, category ORDER BY quarter
    """).df()
    top_cats_h = heat_df.groupby("category")["n"].sum().nlargest(20).index.tolist()
    pivot = (heat_df[heat_df["category"].isin(top_cats_h)]
             .pivot_table(index="category", columns="quarter", values="n",
                          aggfunc="sum", fill_value=0))
    pivot_norm = pivot.div(pivot.max(axis=1), axis=0)
    fig = px.imshow(pivot_norm, color_continuous_scale=["#EBF5FB","#002B5C"],
                    aspect="auto",
                    labels={"color":"Relative Intensity","x":"Quarter","y":""},
                    template="plotly_white")
    fig.update_traces(hovertemplate="%{y}<br>%{x}<br>Intensity: %{z:.2f}<extra></extra>")
    fig.update_layout(height=520, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    ibox("💡 Row-normalised: each sector's own hiring rhythm. Uniformly dark = structural demand → permanent SkillsFuture pipelines. Single dark quarter = cyclical → short-cycle training. 2022Q4 is artificially light (platform ramp-up).", "teal")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — DATA QUALITY & EDA                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown("### 🔎 Data Quality Audit & EDA Findings")
    st.caption("Every cleaning decision documented; anomalies that shaped dashboard design choices.")

    sec("Cleaning Pipeline Summary")
    dq1,dq2,dq3,dq4 = st.columns(4)
    kpi(dq1, "1,048,585", "Raw Rows Ingested",         C["navy"])
    kpi(dq2, "3,998",     "Rows Removed (0.38%)",      C["red"])
    kpi(dq3, "1,044,587", "Clean Rows (99.6%)",        C["teal"])
    kpi(dq4, "37",        "Output Columns (was 22)",   C["gold"])
    st.markdown("")

    st.dataframe(pd.DataFrame({
        "Issue Found": [
            "3,988 completely null / ghost rows",
            "10 RANDOM_JOB_* synthetic test records",
            "occupationId — 100% null",
            "status_id — always = 0 (source bug)",
            "salary_type — always = 'Monthly' (constant)",
            "191 records: salary > S$100,000 / month",
            "7,448 records: salary S$1–499 / month",
            "32 records: experience > 30 years (max = 88!)",
            "78 records: 999 vacancies (sentinel value)",
            "1 record: applications (626) > views (192)",
            "Company names: 1 case variant",
        ],
        "Action Taken": [
            "Dropped entirely",
            "Dropped entirely",
            "Column dropped",
            "Column dropped",
            "Column dropped",
            "salary_valid = False · retained",
            "salary_valid = False · retained",
            "Capped at 30 years · retained",
            "Flagged vacancies_is_sentinel · capped at 99",
            "ACR capped at 1.0 · retained",
            "Standardised to UPPER + strip",
        ],
        "Rows Affected": ["3,988","10","—","—","—","191","7,448","32","78","1","1"],
        "Business Rationale": [
            "All meaningful fields null; salary/dates = 0. Zero analytical signal.",
            "Job IDs stamped 2025-11-15 (future date). Salaries S$1.5M–S$12.7M. Injected test data.",
            "No values in entire dataset.",
            "Source system bug — status_jobStatus carries the correct label.",
            "Zero variance constant — no analytical value.",
            "Likely annual salary entered on monthly form (e.g. S$120K/yr shown as S$120,000/mo).",
            "S$9–15 matches SG hourly rates. Hourly values keyed into monthly salary field.",
            "88 years is physically impossible. Capping preserves record for non-salary analytics.",
            "Agencies use 999 as 'unlimited/many'. Not a real vacancy count.",
            "Views likely undercounted (external channels). Record otherwise valid.",
            "Prevents future duplication risk.",
        ],
    }), use_container_width=True, hide_index=True)

    st.markdown("")
    sec("8 Key EDA Findings That Shaped Dashboard Design")

    eda_findings = [
        (C["red"],    "#1  Salary Distribution Is Right-Skewed — Median ≠ Mean",
         "Median S$3,800 vs mean S$4,769 — a 26% gap driven by senior tech, legal and finance outliers. "
         "<b>Design impact:</b> All charts use MEDIAN not mean. Histogram shows P25/Median/P75 explicitly to make skew visible. "
         "Policy framing targets workers below S$2,900 (Q1) for SkillsFuture subsidies."),

        (C["navy"],   "#2  63% of Postings Have Zero Applications (Zero-Inflation)",
         "Median applications = 0; mean = 2.14. Real phenomenon — high-friction roles attract no response. "
         "<b>Design impact:</b> ACR uses views > 0 filter and NaN-excludes zeros from averages. HFI weights "
         "'low applications (<3)' at 35%. Zero-inflation bar chart shown in this tab."),

        (C["gold"],   "#3  Listing Duration Is Platform-Enforced, Not Time-To-Fill",
         "72% of postings expire in exactly 30 days; 14% in 14d; 8% in 7d. This is the MCF platform's "
         "standard listing period — NOT actual vacancy fill time. "
         "<b>Design impact:</b> Column renamed 'listing_duration_days'; explicitly NOT used as time-to-fill. "
         "Instead, 'days_to_repost' (mean 67 days = ~2 failed cycles) is the real friction clock."),

        (C["teal"],   "#4  Salary Range Disclosure Is Near-Universal (98.6%) — Not a Problem",
         "Unlike many job markets, 98.6% of valid MCF records show a proper salary RANGE (min ≠ max). "
         "Only 1.4% give a point estimate (min = max). Range postings achieve 62% higher ACR (5.8% vs 3.6%). "
         "<b>Design impact:</b> v1 dashboard's 'salary_disclosed' KPI was misleading (99.6% disclosed = meaningless). "
         "Replaced with 'has_salary_range' flag targeting the 1.4% point-estimate gap."),

        (C["purple"], "#5  Platform Ramp-Up Phase (Oct 2022–Mar 2023) Must Not Be Misread",
         "Posts grew 172 → 70,000/month over 6 months — a 400× increase driven by platform adoption, "
         "not labour demand. 2022Q4 volumes are artificially suppressed. "
         "<b>Design impact:</b> Heatmap uses row-normalisation. Time-trend chart carries explicit ramp-up annotation. "
         "H1 vs H2 sector comparison excludes 2022Q4 from baseline."),

        (C["sky"],    "#6  Social Services: Highest Views (54/post) but Below-Median Pay",
         "Social Services averages the highest views per posting (54.3) of all 40 sectors, yet offers only "
         "S$3,675/month median (below market S$3,800). High public interest — low conversion. "
         "<b>Design impact:</b> This sector's EBI is elevated by the low salary denominator, making it appear "
         "in the over-specification quadrant. ACR analysis flags it as a salary-mismatch case, not talent shortage."),

        (C["orange"], "#7  999-Vacancy Records Are Agency Placeholders, Not Real Counts",
         "78 records have exactly 999 vacancies; no records have 100–998. Agencies use 999 as an 'unlimited' "
         "sentinel for bulk-hiring mandates. "
         "<b>Design impact:</b> Vacancies capped at 99 for all aggregations. "
         "Sentinel flag added. If included uncapped, these 78 records would inflate vacancy totals by ~77,000."),

        (C["red"],    "#8  Repost Timing Reveals True Friction Duration Is 2–3× the Listing Period",
         "Mean 67 days from original posting to re-listing confirms employers exhaust 2 full 30-day listing "
         "cycles before retrying. This means single-repost records represent 60–90 days of unfilled demand. "
         "<b>Design impact:</b> Repost timing histogram added to Time Trends tab. HFI repost weight set at 40% — "
         "highest component — because a repost represents a much larger time cost than a simple flag suggests."),
    ]

    for colour, title, body in eda_findings:
        st.markdown(
            f'<div class="ibox" style="background:#fff;border-left:5px solid {colour};'
            f'box-shadow:0 1px 5px rgba(0,0,0,.06);margin-bottom:10px">'
            f'<div style="font-weight:800;color:{colour};margin-bottom:4px">{title}</div>'
            f'<div style="color:#2C3E50">{body}</div></div>',
            unsafe_allow_html=True)

    st.markdown("")
    sec("EDA Visualisations — Raw Distributions Before Cleaning")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Salary distribution — full raw range (log x-axis)**")
        raw_sal = con.execute("SELECT average_salary FROM jobs WHERE average_salary > 0").df()
        fig = px.histogram(raw_sal, x="average_salary", nbins=80, log_x=True,
                           color_discrete_sequence=[C["red"]],
                           labels={"average_salary":"Monthly Salary — Log Scale (S$)"},
                           template="plotly_white")
        fig.add_vline(x=500,     line_dash="dash", line_color=C["gold"],
                      annotation_text="< S$500 flag", annotation_font_size=9)
        fig.add_vline(x=100_000, line_dash="dash", line_color=C["red"],
                      annotation_text="> S$100K flag", annotation_font_size=9)
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("**Application count — zero-inflation (63% of posts have 0 applications)**")
        apps_df = con.execute("""
            SELECT LEAST(applications,50) AS app_bin, COUNT(*) AS n
            FROM jobs GROUP BY app_bin ORDER BY app_bin
        """).df()
        fig = px.bar(apps_df, x="app_bin", y="n", color_discrete_sequence=[C["navy"]],
                     labels={"app_bin":"Applications per Post (capped at 50)","n":"Count"},
                     template="plotly_white")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        fig.update_traces(hovertemplate="Apps=%{x}<br>Count=%{y:,}")
        st.plotly_chart(fig, use_container_width=True)

    ch3, ch4 = st.columns(2)
    with ch3:
        st.markdown("**Listing duration — platform-standardised spikes (not time-to-fill)**")
        dur_df = con.execute(
            "SELECT listing_duration_days AS d, COUNT(*) AS n FROM jobs GROUP BY d ORDER BY d"
        ).df()
        fig = px.bar(dur_df, x="d", y="n", color_discrete_sequence=[C["teal"]],
                     labels={"d":"Listing Duration (days)","n":"Count"},
                     template="plotly_white")
        for xv in [7,14,21,30]:
            fig.add_vline(x=xv, line_dash="dot", line_color=C["red"], opacity=0.6,
                          annotation_text=f"{xv}d", annotation_font_size=9)
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ibox("7/14/21/30-day spikes are platform-enforced — NOT vacancy fill time. Renamed to listing_duration_days.", "grey")

    with ch4:
        st.markdown("**Days to re-post (reposted records only) — the real friction clock**")
        rep_d = con.execute(
            "SELECT days_to_repost AS d, COUNT(*) AS n FROM jobs WHERE repost_count>0 AND days_to_repost>0 GROUP BY d ORDER BY d"
        ).df()
        mean_d = (rep_d["d"]*rep_d["n"]).sum()/rep_d["n"].sum() if len(rep_d) else 67
        fig = px.histogram(rep_d, x="d", y="n", nbins=40, color_discrete_sequence=[C["purple"]],
                           labels={"d":"Days from First to Re-listing","n":"Count"},
                           template="plotly_white")
        fig.add_vline(x=mean_d, line_dash="dash", line_color=C["navy"],
                      annotation_text=f"Mean: {mean_d:.0f} days")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ibox(f"Mean {mean_d:.0f} days ≈ 2 failed 30-day listing cycles. Actual vacancy duration is 2–3× the platform listing period.", "grey")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f'<div style="text-align:center;color:{C["grey"]};font-size:.78rem;line-height:1.7">'
    "<b>🇸🇬 SG Labour Market Intelligence Dashboard  v2</b><br>"
    "Stack: <b>Streamlit · DuckDB · SQL · Pandas · NumPy · Plotly</b><br>"
    "Data: MyCareersFuture (MCF) · Oct 2022–May 2024 · 1,044,587 clean records<br>"
    "<i>Leading-indicator framework for national productivity analytics — MOM &amp; Enterprise Singapore</i>"
    "</div>", unsafe_allow_html=True)
