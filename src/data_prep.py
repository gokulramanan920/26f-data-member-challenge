"""Single source of truth for cleaning survey.csv.

Imported by notebooks/01_eda.ipynb (Pass B) and, later, the modeling
notebook and dashboard, so all three always agree on what "clean" means.
Every decision here is backed by a specific chart in notebooks/01_eda.ipynb
(Pass A) — see that notebook for the reasoning, not just the result.
"""

import re

import numpy as np
import pandas as pd

from constants import (
    CLEAN_PATH,
    FILL_UNKNOWN_COLS,
    RAW_PATH,
    TOP_N_COUNTRIES,
    TOP_N_DATABASES,
    TOP_N_DEVTYPES,
    TOP_N_LANGUAGES,
)


def age_to_midpoint(bucket):
    """Map an Age bucket string to a numeric midpoint; NaN if it has no range."""
    if pd.isna(bucket):
        return np.nan
    m = re.match(r"(\d+)-(\d+)", bucket)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo + hi) / 2
    m = re.match(r"(\d+) years or older", bucket)
    if m:
        return float(m.group(1)) + 5.0  # open-ended top bucket, rough proxy
    return np.nan  # "Prefer not to say"


def orgsize_to_midpoint(bucket):
    """Map an OrgSize bucket string to a numeric employee-count midpoint."""
    if pd.isna(bucket) or bucket == "Unknown":
        return np.nan
    s = bucket.replace(",", "")
    if s.startswith("Just me"):
        return 1.0
    if s.startswith("Less than 20"):
        return 10.0
    m = re.match(r"(\d+) or more", s)
    if m:
        return float(m.group(1)) * 1.5  # open-ended top bucket, rough proxy
    m = re.match(r"(\d+) to (\d+)", s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    return np.nan  # e.g. "I don't know"


def expand_multiselect(series, prefix, top_n, count_col):
    """Turn a ';'-delimited multi-select column into top-N binary flags.

    Returns a DataFrame (same index as `series`) with one 0/1 column per
    top-N value, an f"{prefix}_other" catch-all, and a count column —
    naive one-hot doesn't work here since a single cell can hold many values.
    """
    tokens_per_row = series.fillna("").apply(
        lambda s: [tok for tok in s.split(";") if tok]
    )
    all_tokens = pd.Series([tok for toks in tokens_per_row for tok in toks])
    top_values = all_tokens.value_counts().nlargest(top_n).index.tolist()

    flags = pd.DataFrame(index=series.index)
    for value in top_values:
        col_name = f"{prefix}_{value.lower().replace(' ', '_').replace('/', '_')}"
        flags[col_name] = tokens_per_row.apply(lambda toks: value in toks).astype(int)
    flags[f"{prefix}_other"] = tokens_per_row.apply(
        lambda toks: any(tok not in top_values for tok in toks)
    ).astype(int)
    flags[count_col] = tokens_per_row.apply(len)
    return flags


def group_top_n(series, top_n):
    """Fold every value outside the top-N most frequent into "Other".

    Shared by Country and DevType — both are single-select categoricals with
    a long tail of rare values that would otherwise blow up one-hot cardinality.
    """
    top_values = series.value_counts().nlargest(top_n).index
    return np.where(series.isin(top_values), series, "Other")


def cramers_v(x, y):
    """Association strength (0-1) between two categorical series.

    Uncorrected version (no bias adjustment for small samples/many
    categories) — fine for comparing relative association strength across
    this dataset's columns, not meant as a precise statistic.
    """
    from scipy.stats import chi2_contingency

    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    if n == 0 or min(r, k) < 2:
        return np.nan
    return float(np.sqrt((chi2 / n) / (min(r, k) - 1)))


def load_and_clean(path=RAW_PATH):
    df = pd.read_csv(path)

    # pandas already reads the literal "NA" strings as NaN by default, but
    # coerce explicitly so WorkExp/YearsCode are numeric dtype regardless of
    # what other stray tokens might be in there.
    df["WorkExp"] = pd.to_numeric(df["WorkExp"], errors="coerce")
    df["YearsCode"] = pd.to_numeric(df["YearsCode"], errors="coerce")

    # Can't train or evaluate against a missing target. This is the only
    # row-level filter here — salary outlier handling (floor/cap) is
    # split-aware and lives in notebooks/02_modeling.ipynb instead (see
    # constants.py for why).
    df = df.dropna(subset=["annual_salary_usd"]).copy()

    # Age/OrgSize are ordinal buckets, not free categories — keep both the
    # original string (for tree models / grouped bar charts) and a numeric
    # midpoint (for correlation and linear models).
    df["age_mid"] = df["Age"].map(age_to_midpoint)
    df["orgsize_mid"] = df["OrgSize"].map(orgsize_to_midpoint)

    for col in FILL_UNKNOWN_COLS:
        df[col] = df[col].fillna("Unknown")

    lang_flags = expand_multiselect(
        df["LanguageHaveWorkedWith"], "lang", TOP_N_LANGUAGES, "num_languages"
    )
    db_flags = expand_multiselect(
        df["DatabaseHaveWorkedWith"], "db", TOP_N_DATABASES, "num_databases"
    )
    df = df.join(lang_flags).join(db_flags)

    # Long-tail cardinality in Country (130 uniques) and DevType (32 uniques):
    # keep the top-N (sized per-column by coverage, see constants.py), fold
    # the rest to "Other".
    df["country_grouped"] = group_top_n(df["Country"], TOP_N_COUNTRIES)
    df["devtype_grouped"] = group_top_n(df["DevType"], TOP_N_DEVTYPES)

    # Currency is near-redundant with Country (see the Country/Currency
    # Cramér's V check in Pass A) and the target is already USD-normalized,
    # so it adds cardinality without much independent signal.
    df = df.drop(columns=["Currency"])

    # Descriptive/EDA convenience column only — the actual modeling target is
    # computed from the winsorized salary, split-aware, in the modeling notebook.
    df["log_salary"] = np.log1p(df["annual_salary_usd"])

    return df.reset_index(drop=True)


if __name__ == "__main__":
    cleaned = load_and_clean()
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"wrote {CLEAN_PATH}: {cleaned.shape}")
