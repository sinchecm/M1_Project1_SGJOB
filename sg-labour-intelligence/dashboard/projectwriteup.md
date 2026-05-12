# SG Labour Market Intelligence Dashboard
## Project Writeup

---

## 1. Business Case

### Business Scenario
Singapore's Ministry of Manpower (MOM) and agencies like Enterprise Singapore require
near-real-time signals of labour market health to direct productivity grants, design
upskilling programmes, and pre-empt wage pressure — yet official statistics lag by one or
more quarters. MyCareersFuture (MCF) job-posting metadata, refreshed continuously across
40 industry sectors, provides a uniquely granular micro-level signal that traditional
employment surveys cannot match.

### Objective
Transform 1,044,587 raw MCF job postings (Oct 2022 – May 2024) into actionable leading
indicators across three decision problems:

- **Where is hiring hardest?** Identify occupations and sectors with persistent friction
  (repeated repostings, negligible applicant response) to direct Productivity Solutions
  Grant (PSG) automation funding to the highest-return targets.
- **Where is talent supply mismatched with demand?** Surface over-specified roles — where
  employers demand excessive experience for below-market pay — creating their own pipeline
  shortage without realising it.
- **What is the trajectory?** Detect wage pressure and structural demand shifts 1–2 quarters
  before they appear in official GDP or unemployment figures, enabling proactive policy
  rather than reactive response.

### Target Users and Value

| User Group | How the Dashboard Helps |
|---|---|
| **MOM Policy Analysts / Enterprise Singapore** | Macro sector health, HFI ranking by industry, salary disclosure rates, leading indicator trends — supports PSG grant allocation and SkillsFuture pipeline decisions |
| **Business Leaders / Strategic Planners** | Company-level vacancy leaderboard coloured by friction score — identifies whether hiring difficulty is a compensation problem or a structural role-design problem |
| **HR & Recruitment Professionals** | Application Conversion Rate (ACR) by sector — distinguishes "talent shortage" from "poor JD quality or pay band problem"; discloses where a free transparency change lifts ACR by 62% |
| **Workforce Development Bodies** | Sector growth analysis (H1 vs H2) and quarterly activity heatmap — identifies structural demand (invest in permanent pipelines) vs cyclical spikes (use short-cycle training) |

---

## 2. Data Handling & Process

### Tools Used

| Layer | Technology | Purpose |
|---|---|---|
| CSV ingest | **DuckDB** `read_csv_auto` | Columnar reader — handles 1M+ rows in <5s |
| Wrangling | **Python · Pandas · NumPy** | Complex transforms, JSON parsing, statistical flags |
| Analytics | **DuckDB SQL** | In-memory aggregations — registered DataFrame as VIEW |
| Output | **Apache Parquet (pyarrow)** | 4× smaller than CSV, 10× faster to load, typed columns |
| Dashboard | **Streamlit 1.57** | Web UI, sidebar filters, tab layout |
| Charts | **Plotly** | Interactive, hover-enabled, dual-axis charts |
| Deployment | **GitHub → Streamlit Cloud** | Free hosting, auto-redeploy on push |

### Loading the CSV

DuckDB's `read_csv_auto` was chosen over `pd.read_csv` for the initial 273 MB / 1,048,585-row
source file. It reads columnar data and projects only the 19 required columns at parse time,
reducing RAM by ~40% versus a full pandas ingest. The resulting DataFrame was then registered
as a DuckDB VIEW for all downstream SQL aggregations, keeping peak memory under 300 MB on
Streamlit Cloud's free tier (1 GB limit).

```python
con.execute(f"CREATE VIEW jobs AS SELECT * FROM read_parquet('{path}')")
```

This approach means each query reads only the columns it needs — a histogram of salary
never loads the `company` or `categories` columns.

### Key Cleaning Steps

**1. Structural nulls removed (3,988 rows)**
All 12 meaningful fields were simultaneously null — a source-system export artefact. Zero
analytical value; dropped entirely.

**2. Synthetic test records removed (10 rows)**
Job IDs stamped `RANDOM_JOB_20251115*` (a future date) with salaries up to S$12.7M.
Injected test data; dropped entirely.

**3. Zero-information columns dropped**
`occupationId` (100% null), `status_id` (always 0 — source bug), `salary_type` (always
"Monthly" — zero variance constant). Three columns eliminated with no information loss.

**4. Salary outliers flagged (7,639 records), not dropped**
- Upper: 191 records with salary > S$100,000/month. Inspection revealed "Clinic Assistant
  at S$5,000,000/month" — annual salary or decimal-point error keyed into a monthly field.
  Flagged `salary_valid = False`; retained for non-salary analytics (volume, friction, etc.).
- Lower: 7,448 records with salary S$1–499/month. Values of S$9, S$12, S$15 exactly match
  Singapore's hourly wage bands (S$9–15/hr × 176 hrs/month ≈ S$1,584–2,640). Hourly rates
  keyed into a monthly salary field. Flagged `salary_valid = False`; retained.

**5. Experience capped at 30 years (32 records)**
Maximum raw value was 88 years; other extremes included 87, 76, 63, 50. Physically
impossible — data-entry errors (likely typos: 8 → 80). Capped at 30 to preserve the record
for non-experience analytics without polluting the Experience Barrier Index.

**6. Vacancy sentinel flagged (78 records)**
Exactly 999 vacancies — no records had 100–998. Agencies use 999 as an "unlimited" placeholder
for bulk hiring mandates. Flagged `vacancies_is_sentinel = 1`; capped at 99 for all volume
aggregations. Without this, 78 records would have inflated vacancy totals by ~77,000.

**7. Dates parsed; listing duration renamed**
`metadata_originalPostingDate`, `metadata_expiryDate`, and `metadata_newPostingDate` parsed
to `datetime64`. A key discovery: `listing_duration_days` spikes at exactly 7, 14, 21, and
30 days (72% expire at 30 days) — these are MCF's platform-enforced posting windows, **not
actual vacancy fill times**. Column explicitly renamed to prevent misinterpretation.

**8. Categories parsed from JSON arrays**
The `categories` column is a JSON array (e.g. `[{"category":"IT"},{"category":"Engineering"}]`).
Average 1.69 categories per posting. Primary category extracted as first element; all
categories stored as a pipe-delimited string for full coverage. 40 unique sectors.

**9. Company names standardised**
Strip whitespace + convert to UPPER. Only 1 case variant found — minimal issue — but prevents
future duplication risk across 53,150 unique companies.

### Feature Engineering

| Feature | Formula / Logic | Purpose |
|---|---|---|
| `salary` | `average_salary.clip(0, p99)` | Outlier-capped salary for all visualisations |
| `salary_band` | 7-tier cut: `<$2K`, `$2–3.5K` … `>$15K` | Segmentation in friction analysis |
| `salary_range_width` | `salary_max − salary_min` | Pay transparency proxy |
| `has_salary_range` | `1` if width > 0, else `0` | Range postings get +62% higher ACR |
| `listing_duration_days` | `expiry_date − posting_date` (clipped 0–180) | Platform window; NOT fill time |
| `days_to_repost` | `new_posting_date − posting_date` | True friction signal: mean = 67 days |
| `year_month` / `quarter` | Truncated from `posting_date` | Time-series groupers |
| `HFI` | `0.40×(repost>0) + 0.35×(apps<3) + 0.25×(vac>5)` | Hiring Friction Index [0–1] |
| `ACR` | `applications / views`, clipped [0,1]; NaN if views=0 | Application Conversion Rate |
| `EBI` | `min_exp / log(1 + salary) × 1000` | Experience Barrier Index |

### EDA Highlights: Key Patterns That Shaped the Dashboard

**Finding 1 — Salary is right-skewed, not normal.**
Median S$3,800 vs mean S$4,769 — a 26% gap from senior tech/finance outliers. All charts
use `MEDIAN`, not mean. Salary histogram displays P25/Median/P75 markers explicitly.

**Finding 2 — 63% of postings have zero applications.**
Not a data error — genuine zero-inflation. High-friction roles attract no response at all.
ACR computation filters `views > 0`; HFI weights "fewer than 3 applications" at 35%.

**Finding 3 — Listing duration is platform-enforced, not time-to-fill.**
72% expire in exactly 30 days; 8% in 7 days; 14% in 14 days — MCF's standard listing windows.
This would have been a misleading "vacancy duration" metric. Column renamed `listing_duration_days`.
Mean `days_to_repost` (67 days) was adopted as the real friction clock instead — implying
employers exhaust 2 full listing cycles before retrying.

**Finding 4 — Salary disclosure is near-universal (98.6%).**
Unlike most job markets, 98.6% of valid MCF records post a proper salary range (min ≠ max).
The v1 dashboard's "salary disclosure rate" KPI was effectively constant at 99.6% — meaningless.
Replaced with `has_salary_range` (1.4% point-estimate gap), which has genuine ACR lift impact.

**Finding 5 — Platform ramp-up phase (Oct 2022–Mar 2023) must not be misread as demand growth.**
Posts grew from 172/month to 70,000/month over 6 months — platform adoption, not labour surge.
The quarterly heatmap uses row-normalisation so 2022Q4 does not appear as a demand trough.

**Finding 6 — Social Services paradox.**
Highest average views per posting (54.3) across all 40 sectors, yet below-median salary
(S$3,675 vs market median S$3,800). High public interest, low conversion — a structural
pay-expectation mismatch visible in the EBI scatter chart.

**Finding 7 — 999 vacancies is an agency sentinel, not a real count.**
No records had 100–998 vacancies; 78 records had exactly 999. Uncapped, these would inflate
total vacancy counts by ~77,000 (17% of total). Capped at 99 for all aggregations.

**Finding 8 — Repost timing reveals true vacancy duration is 2–3× the listing period.**
Mean 67-day repost gap = employers waited ~2 full 30-day cycles before retrying. A single
repost therefore represents 60–90 days of unfilled demand. HFI weights the repost component
at 40% — highest of the three components — because it represents a disproportionately large
time cost.

---

## 3. Dashboard / App

### Type of Solution

**Streamlit web application** — interactive, filter-driven, browser-based. No installation
required for end users; deployed to Streamlit Community Cloud (free tier) with a public URL
that auto-updates on every `git push`. Built with DuckDB for in-memory SQL, Plotly for
interactive charts, and a DuckDB VIEW architecture that keeps RAM under 300 MB on the 1 GB
free-tier limit.

---

### Tab 1 — National Overview
**Target users:** MOM Policy Analysts · Enterprise Singapore · National Strategic Planners

| Component | Chart type | Business purpose |
|---|---|---|
| 6 KPI cards | Metric tiles | Instant pulse: total postings, open roles, median salary, vacancies, salary range disclosure %, avg applications per post |
| Top 20 industries | Horizontal bar (colour = median salary) | Dual encoding: bar length = volume, colour = wage level — dark-blue/high-volume sectors are PSG automation candidates |
| Position level split | Donut chart | Shows demand distribution across 9 seniority tiers — tracks shift from junior to senior demand |
| Employment type mix | Horizontal bar | Monitors permanent vs contract vs gig balance as economic signal |
| Salary distribution | Histogram with P25/Median/P75 lines | Right-skew made explicit; bottom quartile (below S$2,900) identified for SkillsFuture targeting |
| Top 25 job titles | Horizontal bar (colour = salary) | Most-posted roles with wage context — recruiters and policy planners most-used view |
| Salary by position level | Bar chart with IQR error bars | Benchmarks median salary; error bars show P25–P75 range — HR teams use P75 as retention target |

**Design choices:** Median (not mean) used throughout to resist outlier distortion. Colour-continuous salary encoding on industry bars lets analysts read salary and volume simultaneously without switching charts. IQR error bars replace box plots for clarity with non-technical stakeholders.

---

### Tab 2 — Labor Bottlenecks & Friction
**Target users:** Business Leaders · HR Professionals · Workforce Development Bodies

| Component | Chart type | Business purpose |
|---|---|---|
| HFI by industry | Horizontal bar (green→red scale) with market-avg line | Sectors above dashed average = prime PSG candidates; colour intensity signals urgency |
| ACR by industry (worst first) | Horizontal bar (red→green) | Low ACR + adequate views = JD quality or pay problem; shows HR where to fix the posting, not expand the budget |
| Experience Barrier scatter | Bubble chart (x=avg exp, y=salary, size=vacancies, colour=EBI) | 4-quadrant annotation surfaces over-specification trap (bottom-right: high exp, low pay) — firms creating their own pipeline shortage |
| Company leaderboard | Horizontal bar (colour = HFI) | Volume + friction combined: red = urgent; identifies firms for direct MOM/PSG outreach |
| Salary range vs point ACR | Grouped bar | Quantifies the 62% ACR lift from posting a salary range — zero-cost HR intervention |
| Repost rate by salary band | Dual-axis (bar + line) | Repost rate peaks at S$2–5K bands — confirms friction is compensation mismatch, not senior talent scarcity |

**Design choices:** HFI bar chart is sorted descending with a market-average vline — the most actionable charts show the worst cases first. The experience scatter uses quadrant annotations to tell the story without requiring users to interpret raw axes. Bubble size encodes vacancy volume so urgency is immediately visible.

---

### Tab 3 — Time Trends & Wage Pressure
**Target users:** All groups — strategic planning, macroeconomic monitoring

| Component | Chart type | Business purpose |
|---|---|---|
| Monthly postings + vacancies | Dual-axis bar+line | Divergence between lines = unmet demand accumulating — leads official GDP stats by 1–2 quarters |
| Salary trend by position level | Multi-line chart | Gradient widening = healthy productivity growth; compression = retention risk and talent pressure signal |
| Sector growth H1→H2 2023 | Horizontal diverging bar | Green = structural demand (permanent upskilling), red = declining (automation displacement / retraining priority) |
| HFI trend + repost rate | Dual-axis line+bar | Rising HFI line is the leading indicator — visible before wage/unemployment stats move |
| Repost timing histogram | Histogram with mean line | Mean 67-day gap = employers wait 2 cycles; visualises true vacancy duration vs platform listing period |
| Sector activity heatmap | Row-normalised imshow | Each row = sector's own seasonal rhythm; uniform dark = structural demand; single dark quarter = cyclical spike |

**Design choices:** Row-normalisation on the heatmap is a critical design decision — without it, large sectors (IT, Engineering) dominate with dark colours and small sectors are invisible. Row-normalisation reveals each sector's own hiring cycle independent of its absolute size. The dual-axis charts pair complementary metrics (postings + vacancies; HFI + repost rate) to reduce cognitive load without losing context.

---

### Tab 4 — Data Quality & EDA
**Target users:** Analysts, data stewards, project reviewers, academic assessors

Documents every cleaning decision in a structured table (issue, action, rows affected,
rationale) and presents 8 EDA findings as annotated cards explaining how each discovery
shaped a specific dashboard design choice. Four raw-distribution charts show the data
before cleaning: salary on a log x-axis (reveals the extreme outliers), application
zero-inflation, listing duration standardised spikes, and repost timing distribution.

**Purpose:** Builds stakeholder trust by making the methodology transparent and auditable.
Every KPI number in the other three tabs can be traced back to a documented cleaning
decision shown here.

---

### Interactivity

All four tabs respond to a single **sidebar filter panel** with six controls:
- Date range picker (Oct 2022 – May 2024, any sub-period)
- Job status multiselect (Open / Closed / Re-open)
- Position level multiselect (9 tiers from Fresh/entry to Senior Management)
- Employment type multiselect (Permanent, Full Time, Contract, Part Time, Temporary, Freelance)
- Industry category multiselect (40 sectors)
- Monthly salary slider (S$1,100 – S$16,500; p1–p99 range)

Filters generate a DuckDB `WHERE` clause applied to all charts simultaneously. First load
~15 seconds (connecting to parquet); subsequent filter changes are near-instant as DuckDB
queries the in-memory VIEW.

All Plotly charts include **hover tooltips** showing exact values, counts, and secondary
metrics. Dual-axis charts use `hovermode="x unified"` so both axes display on hover.
Bubble charts show the category name, EBI score, vacancy total, and record count.

---

### Colour Scheme

| Colour | Hex | Usage |
|---|---|---|
| Singapore Red | `#C8102E` | Primary metrics, alert-level HFI, salary histogram |
| Deep Navy | `#002B5C` | Structure, bar fills, low-end of continuous scales |
| Amber / Gold | `#F5A623` | Warning-level friction, caution insights |
| Teal | `#16A085` | Positive / healthy signals, salary range disclosed |
| Purple | `#7D3C98` | Secondary segmentation, position level |
| Graduated scale | Navy → Purple → Red → Gold | Continuous salary and HFI encoding |

**Insight boxes** use `rgba()` semi-transparent tints (18% opacity) with a 3px left border
in the matching hue, ensuring visibility on both Streamlit's light and dark themes.
`var(--secondary-background-color)` and `var(--text-color)` CSS variables are used for KPI
cards and text so they auto-invert in dark mode without explicit media queries.

---

### How Each View Supports the Business Objectives

| Business Objective | Primary Tab | Key Chart |
|---|---|---|
| Diagnose labour bottlenecks | Tab 2 | HFI ranking + company leaderboard |
| Improve matching efficiency | Tab 2 | ACR by industry + salary range ACR lift |
| Direct targeted interventions (PSG) | Tab 1 + 2 | Top industry volume/salary + HFI above market avg |
| Detect wage & talent pressure | Tab 3 | HFI trend + salary trend + vacancy divergence |
| Distinguish structural vs cyclical demand | Tab 3 | Sector growth bar + quarterly heatmap |
| Benchmark salary for HR teams | Tab 1 | Salary by level (IQR) + top titles |
| Build stakeholder trust in the numbers | Tab 4 | Cleaning audit table + EDA findings cards |

---

*Stack: Streamlit 1.57 · DuckDB 1.5 · Python 3.13 · Pandas · NumPy · Plotly · PyArrow · GitHub · Streamlit Community Cloud*
*Data: MyCareersFuture (MCF) · Oct 2022 – May 2024 · 1,044,587 clean records*
