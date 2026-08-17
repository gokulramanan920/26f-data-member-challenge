# Technical Challenge

## The challenge

Your challenge is to predict developer salaries from a real survey dataset
(about 5,000 responses), included here as `data/survey.csv`. Build a model that
predicts `annual_salary_usd`.

This survey is published as-is, so expect to spend time determining what data you actually have before starting.
Look at it column by column, decide what to do about what you find, and be ready to say why you did it that way.

There's no particular set of fixes/steps we're looking for. We just want to understand what you found, how you
went about it, and why.

**AI Usage Info**

AI tools are allowed and expected. You'll walk through your submission live in
your interview and we'll ask detailed questions about your specific choices, so
understand every line you submit, including anything AI wrote. A simple
approach you can fully explain beats an impressive one you can't.

## Optional additions 

Pick one if you want to go further:

- a more rigorous ML pipeline with proper validation and feature engineering,
- a frontend/dashboard to display the data and your predictions, or
- an additional API connection that enriches the dataset with outside data.

**Additions are a bonus, not a requirement. A clean base submission is a
complete submission.** We would much rather see a straightforward model you
understand completely than an ambitious one you're guessing about during the interview. 
Every minute spent on an addition is a minute not spent being able to explain your core work.
These additions should be built on an already solid core submission.

---

## Getting started

This is a template repository, please fork it to create your own repository and go:

```bash
git clone <your-new-repo-url>
```

```bash
pip install -r requirements.txt
```


## Submitting

**Commit and push to your assignment repo before your interview time**
Your last push before your interview is what we review. There's no separate
submission step.

Your repo should contain:

1. **Your code.** However you'd normally organize it — a notebook, a script, a
   small project. It doesn't need to be production-grade.
2. **Your predictions or your metrics** — enough that we can see how well it
   worked, in whatever form makes sense for what you built.

Commit as you go rather than in one push at the end. We don't grade commit
history, but it protects you from losing work.

## Practical notes

- **Language and tools are up to you.** Most people use Python with pandas and
  scikit-learn, and `requirements.txt` covers that.
- **There is no target score.** We are not ranking submissions by accuracy. A
  model with modest error and clear reasoning scores better than a strong one
  you can't account for. This is a genuinely hard prediction problem and the
  numbers are meant to look modest.
- **Don't over-engineer.** 2 to 3 hours is the intended scope. If you find
  yourself at hour six, stop and write up what you have.
- **Bring your submission to the interview**, on your own machine, ready to open
  and run. Have it up before we start.
- **Questions?** Ask Michael Maaseide (maaseide.m@northeastern.edu) or Samuel Baldwin (baldwin.sam@northeastern.edu). Asking a
  clarifying question is not a mark against you.

## How we'll evaluate it

We're interested in your reasoning far more than your results. During the
interview we'll ask about:

- **What you noticed in the data** and how you decided what to do about it.
  Where you made a judgement call, we'll ask why you made it that way.
- **Your modelling choices** — why that model, how you know it works, what you
  compared it against.
- **Your code** — we'll pick a few specific lines and ask what they do and why
  they're there. This applies equally to code you wrote and code an AI wrote for
  you.
- **What you'd do differently** with more time, and what you think the
  weaknesses are.

---

## The data

`data/survey.csv` — 5,000 responses, one row per respondent. These are
professional developers who answered the compensation question in the **Stack
Overflow Annual Developer Survey 2025**, published by Stack Exchange under the
[Open Database License](https://opendatacommons.org/licenses/odbl/1-0/).

---

## My Analysis Process and Methodology

**Data cleaning & feature engineering** (`src/data_prep.py`, `notebooks/01_eda.ipynb`)
- Cleaned `survey.csv` column-by-column based on what the EDA actually showed: coerced numeric-as-text fields, mapped ordinal buckets (`Age`/`OrgSize`) to numeric midpoints, expanded multi-select language/database fields into top-N binary flags, grouped high-cardinality `Country`/`DevType` into top-N + "Other" (sized by actual coverage, not a round number), and dropped `Currency` as redundant with `Country` (Cramér's V = 0.905).
- Target: `log1p(annual_salary_usd)`, with outlier bounds computed only from the training split (5th/95th percentile) and applied as a winsorize + explicit flag rather than a blind drop, so nothing about the split leaks into a step meant to be shared/reusable.
- Started with **ElasticNet** first on purpose: a linear model forces every design decision (encoding, target definition, split protocol) to be fully explainable before adding a second, less transparent model on top of it.

**ElasticNet — feature set & results** (`notebooks/02_modeling.ipynb`, Section 3)
- Base feature set: 6 numeric + 7 categorical (one-hot) + 34 binary flags → 112 model features.
- Final feature set adds 9 engineered interaction/self-interaction terms, found via 3 rounds of test-then-prune (e.g. `log_workexp`/`log_yearscode` for a Mincer-curve-shaped experience effect) → 121 model features.
- Full metrics for both, plus the ablation isolating the interactions' real contribution, in `data/model_metrics.txt` — see line 50 (best ElasticNet, val set) and lines 71–75 (ablation table).
- Full coefficient list: `data/elasticnet_coefs.txt`.

**CatBoost + SHAP** (`notebooks/02_modeling.ipynb`, Section 4)
- Same split/target/winsorization as ElasticNet, but a leaner feature set suited to a tree model: full-cardinality `Country`/`DevType` via native `cat_features` (no one-hot, no top-N grouping needed), only the log-transformed seniority terms (raw and log forms are redundant for trees — they're invariant to monotonic transforms), and just 1 of the 9 ElasticNet interaction terms (the one a tree can't easily reconstruct from its own splits).
- Beat ElasticNet on every validation metric — see line 92 (CatBoost, val set) and lines 113–117 (side-by-side final comparison) in `data/model_metrics.txt`.
- SHAP corroborates rather than overturns the ElasticNet story (same top features, same direction of effect) — summary/dependence plots at the end of Section 4.

**Future steps, time permitting**
- Compare the boosting model (CatBoost) against a bagging model (Random Forest) on the same split/features, to see whether the gain is boosting-specific or just "any tree model beats linear here."
- A small dashboard (React or Panel): predicted-vs-actual scatter, filterable/grouped by category (`Country`, `DevType`, etc.) and by metric, to see where the model is strong or weak visually rather than only from a metrics table.
- Try a GAM (Generalized Additive Model) for CatBoost-level non-linearity with ElasticNet-level interpretability — a natural next step given the log-transform findings.
