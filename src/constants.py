"""Cleaning parameters for data_prep.py, kept separate so the thresholds are
easy to find and justify on their own, apart from the cleaning logic itself.
"""

RAW_PATH = "data/survey.csv"
CLEAN_PATH = "data/survey_clean.csv"

# Salary outlier handling (floor/cap) is NOT here. It depends on which split
# (train/test/val) a row falls into — the bounds are fit on train only inside
# notebooks/02_modeling.ipynb, so they can't live in this shared, split-agnostic
# cleaning step without leaking. See that notebook for the winsorize+flag logic.

# How many distinct values to keep as their own one-hot/flag column before
# folding the long tail into "Other" — sized per column by row/token coverage
# (checked directly against data/survey.csv), not a single arbitrary number
# applied everywhere:
#   Country:  130 uniques, top-15 only covered 71.7% of rows (too much lost to
#             "Other" for a likely strong cost-of-living proxy) -> bumped to 20 (77.1%)
#   DevType:  32 uniques, a flatter distribution -> top-10 already covers 86.2%
#   Language/Database (multi-select): top-15 covers ~99% of rows having at
#             least one top-N pick, and the tail entries are cheap (a handful
#             of columns) and plausibly still informative -> left at 15
TOP_N_COUNTRIES = 20
TOP_N_DEVTYPES = 10
TOP_N_LANGUAGES = 15
TOP_N_DATABASES = 15

# Columns where "NA" is a real answer state (respondent skipped/doesn't
# apply), not something to drop rows over — Pass A's missingness chart
# shows these are all >2.5% null, too much to throw away.
FILL_UNKNOWN_COLS = ["EdLevel", "OrgSize", "ICorPM", "RemoteWork", "Industry"]
