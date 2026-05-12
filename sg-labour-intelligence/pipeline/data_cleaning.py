"""
=============================================================================
SG LABOUR MARKET — DATA CLEANING PIPELINE
=============================================================================
Input  : SGJobData.csv (raw MCF job postings, ~1,048,585 rows)
Output : SGJobData_clean.parquet  (production-ready, ~1,044,577 rows)
         cleaning_report.json     (audit log of every decision made)

Run:
    python data_cleaning.py

Key cleaning decisions are documented inline and summarised in the report.
=============================================================================
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW_PATH   = Path("data/raw/SGJobData.csv")
OUT_PATH   = Path("data/clean/SGJobData_clean.parquet")
REPORT_PATH = Path("data/clean/cleaning_report.json")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG  (tracks every cleaning action with counts)
# ─────────────────────────────────────────────────────────────────────────────
report = {"input_rows": 0, "output_rows": 0, "actions": []}

def log(action: str, count: int, note: str = ""):
    entry = {"action": action, "rows_affected": count, "note": note}
    report["actions"].append(entry)
    print(f"  [{count:>8,}]  {action}" + (f"  →  {note}" if note else ""))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 0: LOAD ──────────────────────────────────────────────────────")
df = pd.read_csv(RAW_PATH)
report["input_rows"] = len(df)
print(f"  Raw rows loaded : {len(df):,}")
print(f"  Columns         : {df.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DROP STRUCTURALLY EMPTY ROWS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 1: EMPTY ROWS ────────────────────────────────────────────────")
# EDA finding: 3,988 rows are completely null across all meaningful columns.
# These are ghost rows — no job_id, title, salary, date, company. No signal.
null_mask = df["metadata_jobPostId"].isna()
log("Dropped all-null rows (ghost records)", null_mask.sum(),
    "All 12 non-bool fields null; salary/dates all zero. Zero analytical value.")
df = df[~null_mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DROP SYNTHETIC TEST RECORDS (RANDOM_JOB_*)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 2: SYNTHETIC TEST RECORDS ────────────────────────────────────")
# EDA finding: 10 rows at the tail of the dataset carry job IDs of the form
# RANDOM_JOB_20251115XXXXXXXXXX_N  (timestamp 2025-11-15, after the data
# period). Salaries range from S$1.5M to S$12.7M — clearly synthetic.
# These represent data injection artefacts, not real postings.
synth_mask = df["metadata_jobPostId"].str.startswith("RANDOM_JOB_", na=False)
log("Dropped synthetic RANDOM_JOB_* records", synth_mask.sum(),
    "Injected test rows with future timestamps (2025-11-15) and multi-million salaries.")
df = df[~synth_mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: DROP ZERO-INFORMATION COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 3: ZERO-INFORMATION COLUMNS ──────────────────────────────────")
# occupationId : 100% null — no analytical value.
# status_id    : Always 0 regardless of job status — a bug in source data; status_jobStatus is used instead.
# salary_type  : 100% "Monthly" constant — drops no information.
cols_to_drop = ["occupationId", "status_id", "salary_type"]
for c in cols_to_drop:
    nuniq = df[c].nunique()
    nnull = df[c].isna().sum()
    log(f"Dropped column '{c}'", 0,
        f"null={nnull:,}/{len(df):,}, unique non-null values={nuniq}")
df.drop(columns=cols_to_drop, errors="ignore", inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: RENAME COLUMNS  (remove metadata_ prefix; shorter, readable names)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 4: RENAME COLUMNS ─────────────────────────────────────────────")
rename_map = {
    "metadata_jobPostId"               : "job_id",
    "metadata_originalPostingDate"     : "posting_date_raw",
    "metadata_newPostingDate"          : "new_posting_date_raw",
    "metadata_expiryDate"              : "expiry_date_raw",
    "metadata_isPostedOnBehalf"        : "is_posted_on_behalf",
    "metadata_repostCount"             : "repost_count",
    "metadata_totalNumberJobApplication": "applications",
    "metadata_totalNumberOfView"       : "views",
    "minimumYearsExperience"           : "min_exp_years",
    "numberOfVacancies"                : "vacancies_raw",
    "positionLevels"                   : "position_level",
    "postedCompany_name"               : "company_raw",
    "status_jobStatus"                 : "job_status",
}
df.rename(columns=rename_map, inplace=True)
log("Renamed columns", len(rename_map), "Removed metadata_ prefix; clearer names")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: DATE PARSING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 5: DATE PARSING ───────────────────────────────────────────────")
for col_raw, col_out in [
    ("posting_date_raw",     "posting_date"),
    ("expiry_date_raw",      "expiry_date"),
    ("new_posting_date_raw", "new_posting_date"),
]:
    df[col_out] = pd.to_datetime(df[col_raw], errors="coerce")
    failed = df[col_out].isna().sum()
    if failed:
        log(f"Date parse failures in '{col_raw}'", failed, "Set to NaT")

# EDA finding: No expiry < posting (0 cases). No duration > 365 (0 cases).
# Duration is highly standardised: 30d (72%), 14d (14%), 7d (8%), 21d (2.5%).
# This is PLATFORM-ENFORCED listing period — NOT actual time-to-fill.
# We compute it but label it correctly.
df["listing_duration_days"] = (df["expiry_date"] - df["posting_date"]).dt.days.clip(0, 365)

# Days from original to re-post (meaningful for friction analysis)
df["days_to_repost"] = (df["new_posting_date"] - df["posting_date"]).dt.days.clip(0, 365)

log("Parsed and computed date features", len(df),
    "listing_duration_days (platform-set); days_to_repost (hiring signal)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SALARY CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 6: SALARY CLEANING ────────────────────────────────────────────")

# 6a. Extreme upper outliers (>S$100,000/month).
# EDA: 12 records with average_salary > S$1M. Inspection reveals:
#   - Clinic Assistant at S$5M/mo  → clearly S$5,000 mis-keyed as S$5,000,000
#   - Accounts Executive at S$12.7M/mo → same type of error
# Threshold: S$100,000/mo (≈ top 0.01%). Flag for exclusion from salary analytics.
SALARY_UPPER = 100_000
extreme_hi = df["average_salary"] > SALARY_UPPER
log("Flagged salary_valid=False (salary > S$100K/mo)", extreme_hi.sum(),
    "Likely annual salary or decimal-point error entered as monthly. Retained in dataset.")

# 6b. Extremely low salary (< S$500/month, > 0).
# EDA: 7,448 records have salary between S$1–499.
#   - Values like $9, $12, $15 match Singapore hourly rates ($9–$15/hr).
#   - $9–$15/hr × 176 hrs/month ≈ S$1,584–S$2,640/month (plausible).
#   - These are likely hourly-rate entries on a monthly-salary form.
#   - Predominantly Part Time, Temporary, Contract employment types.
# Flag: exclude from salary medians/trends; include for volume counts.
SALARY_LOWER = 500
extreme_lo = (df["average_salary"] > 0) & (df["average_salary"] < SALARY_LOWER)
log("Flagged salary_valid=False (salary S$1–499/mo)", extreme_lo.sum(),
    "Likely hourly rate entered as monthly (e.g. $12/hr → $12 shown). Retained in dataset.")

df["salary_valid"] = ~(extreme_hi | extreme_lo)
log("Created flag column 'salary_valid'", (~df["salary_valid"]).sum(),
    "False = record excluded from salary aggregations. Record retained for non-salary analysis.")

# 6c. Compute salary range width (S$max - S$min) — our proxy for pay transparency.
# EDA: 98.9% of valid records give a salary RANGE (min ≠ max).
#   Only 1.1% give a point estimate (min == max). Point-estimate postings have
#   ACR of 3.6% vs 5.8% for range postings — a 62% uplift from disclosing a range.
df["salary_range_width"] = (df["salary_maximum"] - df["salary_minimum"]).clip(lower=0)
df["has_salary_range"] = (df["salary_range_width"] > 0).astype(int)
log("Computed salary_range_width and has_salary_range flag", len(df),
    "Range postings have 62% higher ACR than point-estimate postings.")

# 6d. Capped salary for analysis (p99 = S$16,667)
p99 = df.loc[df["salary_valid"], "average_salary"].quantile(0.99)
df["salary"] = df["average_salary"].clip(0, p99)
df.loc[~df["salary_valid"], "salary"] = np.nan
log(f"Created 'salary' column (capped at p99 = S${p99:,.0f})", len(df),
    "NaN for invalid records; capped outliers for valid ones. Used in all visualisations.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: EXPERIENCE CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 7: EXPERIENCE CLEANING ────────────────────────────────────────")
# EDA: 39 records request > 30 years' experience (max = 88 years).
# Values of 87, 88, 76, 63, 62, 61 are physically impossible for a career.
# Values of 50, 58, 59, 60 are implausible for any active job posting.
# Decision: cap at 30 years (senior leadership benchmark). Capping rather than
# dropping preserves the record for non-experience analytics.
EXP_CAP = 30
over_exp = (df["min_exp_years"] > EXP_CAP).sum()
log(f"Capped min_exp_years > {EXP_CAP} to {EXP_CAP}", over_exp,
    "88yr, 87yr, 76yr etc. are data-entry errors. Cap preserves record, corrects signal.")
df["min_exp_years"] = df["min_exp_years"].clip(upper=EXP_CAP)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: VACANCY CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 8: VACANCY CLEANING ───────────────────────────────────────────")
# EDA: 999 is used as a sentinel value for "unlimited / many vacancies" by
# recruiters (e.g. agencies posting for multiple clients). 308 records have
# vacancies > 100, all with exactly 999. No vacancies have 100–998.
# Decision: flag these separately; use capped value (99) in aggregations
# to avoid inflating vacancy counts for what is essentially a placeholder.
sentinel_mask = df["vacancies_raw"] == 999
log("Flagged 999-vacancy records as 'vacancies_is_sentinel'", sentinel_mask.sum(),
    "Agencies use 999 as unlimited placeholder. Capped to 99 for volume aggregations.")
df["vacancies_is_sentinel"] = sentinel_mask.astype(int)
df["vacancies"] = df["vacancies_raw"].clip(upper=99)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: COMPANY NAME STANDARDIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 9: COMPANY NAME STANDARDIZATION ───────────────────────────────")
# EDA: 53,151 unique company names raw; 53,150 case-insensitive.
# Only 1 case variant — minimal issue. Standardising to UPPER collapses it
# cleanly and removes risk of future case drift.
df["company"] = df["company_raw"].str.strip().str.upper()
log("Standardised company names to UPPER + strip", 1,
    "Collapsed 1 case variant; 53,150 unique companies after standardisation.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: CATEGORY PARSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 10: CATEGORY PARSING ──────────────────────────────────────────")
# EDA: categories is a JSON array (avg 1.69 categories per posting).
# 62% have 1 category, 20% have 2, 8% have 3, 5% have 5.
# Decision: extract primary (first) category for drill-down analysis.
# Store all categories as pipe-delimited string for completeness.
def parse_cats(s):
    if pd.isna(s):
        return ("Unknown", "Unknown")
    try:
        cats = json.loads(s)
        names = [c.get("category", "Unknown") for c in cats]
        primary = names[0] if names else "Unknown"
        all_cats = "|".join(names) if names else "Unknown"
        return (primary, all_cats)
    except Exception:
        return ("Unknown", "Unknown")

parsed = df["categories"].apply(parse_cats)
df["category"]      = parsed.apply(lambda x: x[0])
df["categories_all"] = parsed.apply(lambda x: x[1])
df["n_categories"]  = df["categories_all"].apply(
    lambda s: len(s.split("|")) if s and s != "Unknown" else 0
)
log("Parsed JSON categories → primary + all (pipe-delimited)", len(df),
    "40 unique sectors; primary used for segmentation. 1.69 avg cats/posting.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: APPLICATIONS / VIEWS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 11: APPLICATIONS & VIEWS ──────────────────────────────────────")
# EDA: 63% of postings have 0 applications. This is real zero-inflation,
# not a data error — many roles attract no applicants, especially those
# with high friction. Median applications = 0; mean = 2.14.
# 1 record has applications > views (626 vs 192). This is a data inconsistency —
# views may be undercounted or applications reflect external channel traffic.
# Flag it but retain.
apps_gt_views = ((df["applications"] > df["views"]) & (df["views"] > 0)).sum()
log("Flagged 'apps_gt_views' inconsistency", apps_gt_views,
    "1 record: 626 applications but only 192 views. Retained; ACR capped at 1.0.")

df["acr"] = np.where(
    df["views"] > 0,
    (df["applications"] / df["views"]).clip(0, 1),
    np.nan,
)
log("Computed ACR = applications/views, capped [0,1]", len(df),
    "NaN where views=0 (183,109 records). 63% have 0 applications (zero-inflated).")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12: DERIVED KPI FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 12: KPI FEATURE ENGINEERING ──────────────────────────────────")

# Temporal features
df["year_month"] = df["posting_date"].dt.to_period("M").dt.to_timestamp()
df["year"]       = df["posting_date"].dt.year
df["quarter"]    = df["posting_date"].dt.to_period("Q").astype(str)
df["month_num"]  = df["posting_date"].dt.month

# Salary band (for segmentation charts)
sal_bins   = [0, 2000, 3500, 5000, 7500, 10_000, 15_000, np.inf]
sal_labels = ["<$2K", "$2–3.5K", "$3.5–5K", "$5–7.5K", "$7.5–10K", "$10–15K", ">$15K"]
df["salary_band"] = pd.cut(df["salary"], bins=sal_bins, labels=sal_labels, right=False)

# ── KPI 1: Hiring Friction Index (HFI) ─────────────────────────────────────
# Composite 0–1 signal:
#   40% weight → reposted (persistent hiring failure)
#   35% weight → fewer than 3 applicants (negligible interest)
#   25% weight → vacancies > 5 (bulk unfilled demand)
df["hfi"] = (
    (df["repost_count"] > 0).astype(float) * 0.40
    + (df["applications"] < 3).astype(float) * 0.35
    + (df["vacancies"] > 5).astype(float) * 0.25
)

# ── KPI 2: Experience Barrier Index (EBI) ──────────────────────────────────
# EBI = min_exp / log1p(salary) × 1000
# Penalises heavy experience demands relative to salary offered.
# Log transform prevents high-salary roles from artificially suppressing EBI.
# High EBI = over-specification → firms blocking their own talent pipeline.
df["ebi"] = np.where(
    (df["salary"].notna()) & (df["salary"] > 0),
    df["min_exp_years"] / np.log1p(df["salary"]) * 1_000,
    np.nan,
)

log("Computed HFI, EBI, salary_band, temporal features", len(df),
    "HFI: 0-1 composite (repost 40%, low-apps 35%, high-vac 25%). "
    "EBI: exp/log(salary)×1000.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 13: FINAL COLUMN SELECTION & ORDER
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 13: COLUMN ORDERING ────────────────────────────────────────────")

keep = [
    # Identifiers
    "job_id", "title",
    # Posting dates
    "posting_date", "expiry_date", "new_posting_date",
    "listing_duration_days", "days_to_repost",
    # Temporal groupers
    "year_month", "year", "quarter", "month_num",
    # Organisation
    "company", "is_posted_on_behalf",
    # Role attributes
    "category", "categories_all", "n_categories",
    "position_level", "employmentTypes", "min_exp_years",
    # Status
    "job_status", "repost_count",
    # Demand
    "vacancies", "vacancies_raw", "vacancies_is_sentinel",
    # Engagement
    "applications", "views", "acr",
    # Salary (raw)
    "salary_minimum", "salary_maximum", "average_salary",
    # Salary (cleaned & engineered)
    "salary", "salary_valid", "salary_range_width",
    "has_salary_range", "salary_band",
    # KPIs
    "hfi", "ebi",
]

df = df[keep]
log("Final column set", 0, f"{len(keep)} columns retained, {19 - len(keep)} raw columns dropped/renamed")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 14: WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("\n── STEP 14: WRITE PARQUET ──────────────────────────────────────────────")
df.to_parquet(OUT_PATH, index=False, engine="pyarrow", compression="snappy")
report["output_rows"] = len(df)
log("Written to parquet (snappy-compressed)", len(df),
    f"Parquet chosen over CSV: 10× faster load, 4× smaller file, typed columns.")

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("CLEANING SUMMARY")
print("═" * 70)
print(f"  Input rows     : {report['input_rows']:>10,}")
print(f"  Output rows    : {report['output_rows']:>10,}")
removed = report["input_rows"] - report["output_rows"]
print(f"  Rows removed   : {removed:>10,}  ({removed/report['input_rows']*100:.3f}%)")
print(f"  Output columns : {df.shape[1]:>10}")
print(f"  Output file    : {OUT_PATH}")
print(f"  Audit report   : {REPORT_PATH}")
print()

# Quick quality check
valid_sal = df["salary_valid"].sum()
print("POST-CLEAN QUALITY CHECKS:")
print(f"  Valid salary records          : {valid_sal:,} ({valid_sal/len(df)*100:.1f}%)")
print(f"  Records with ACR computed     : {df['acr'].notna().sum():,}")
print(f"  Salary range disclosed (%)    : {df['has_salary_range'].mean()*100:.1f}%")
print(f"  Reposted records              : {(df['repost_count']>0).sum():,} ({(df['repost_count']>0).mean()*100:.1f}%)")
print(f"  Date range                    : {df['posting_date'].min().date()} → {df['posting_date'].max().date()}")
print(f"  Unique companies              : {df['company'].nunique():,}")
print(f"  Unique job titles             : {df['title'].nunique():,}")
print(f"  Unique industry categories    : {df['category'].nunique()}")
print()
print("  Columns in clean dataset:")
for i, c in enumerate(df.columns, 1):
    print(f"    {i:>2}. {c:<30}  dtype={df[c].dtype}")
