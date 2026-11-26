"""P1-P3: non-parental adult mentors, family income, and child flourishing (NSCH 2024).

Children aged 6-17. "Mentor" = parent says yes to: is there at least one other adult in the
child's school, neighborhood or community who knows the child well and who the child can
rely on for advice or guidance (k9q96).

Survey weights (fwc). Family income (% of poverty line) is multiply imputed (fpl_i1-fpl_i6);
every model is run on each of the 6 versions and combined with Rubin's rules.
Standard errors: heteroskedasticity-robust with weights. This approximates the survey design
but ignores strata, which usually makes SEs slightly conservative.

Run: python3 -I scripts/01_nsch_mentors.py   ->  output/nsch/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nsch"
OUT = ROOT / "output" / "nsch"
OUT.mkdir(parents=True, exist_ok=True)

COLS = ["k9q96", "sc_age_years", "sc_sex", "sc_race_r", "sc_hispanic_r", "higrade", "family_r",
        "fwc", "k6q71_r", "k7q84_r", "k7q85_r", "k7q82_r", "k7q83_r", "hopeful",
        "k7q30", "k7q31", "k7q32", "k7q37", "ace1"] + [f"ace{i}" for i in range(3, 11)] \
       + [f"fpl_i{i}" for i in range(1, 7)]
d = pd.read_stata(RAW / "nsch_2024e_topical.dta", columns=COLS, convert_categoricals=False)
d = d[(d.sc_age_years >= 6) & d.k9q96.isin([1, 2])].copy()

# ---- Variables ----
yes = lambda s: (s == 1).astype(float).where(s.isin([1, 2]))
usually = lambda s: s.isin([1, 2]).astype(float).where(s.isin([1, 2, 3, 4]))
d["mentor"] = yes(d.k9q96)
# Flourishing (ages 6-17, Data Resource Center definition): curiosity, finishes tasks, stays
# calm, each "always" or "usually".
flour = pd.concat([usually(d[c]) for c in ["k6q71_r", "k7q84_r", "k7q85_r"]], axis=1)
d["flourishing"] = flour.min(axis=1).where(flour.notna().all(axis=1))
eng = pd.concat([usually(d[c]) for c in ["k7q82_r", "k7q83_r"]], axis=1)
d["school_engaged"] = eng.min(axis=1).where(eng.notna().all(axis=1))
d["hopeful_most"] = d.hopeful.isin([1, 2]).astype(float).where(d.hopeful.isin([1, 2, 3, 4]))
acts = pd.concat([yes(d[c]) for c in ["k7q30", "k7q31", "k7q32"]], axis=1)
d["any_activity"] = acts.max(axis=1).where(acts.notna().any(axis=1))
d["volunteer"] = yes(d.k7q37)
aces = pd.concat([yes(d[f"ace{i}"]) for i in range(3, 11)], axis=1)
d["ace_count"] = aces.sum(axis=1, min_count=6)
d["ace_2plus"] = (d.ace_count >= 2).astype(float).where(d.ace_count.notna())
d["hardship"] = d.ace1.isin([3, 4]).astype(float).where(d.ace1.isin([1, 2, 3, 4]))
d["female"] = (d.sc_sex == 2).astype(float)
d["raceeth"] = np.select(
    [d.sc_hispanic_r == 1, d.sc_race_r == 1, d.sc_race_r == 2, d.sc_race_r == 4],
    ["Hispanic", "White", "Black", "Asian"], "Other/multiracial")
d["parent_ed"] = d.higrade.map({1: "Less than HS", 2: "HS", 3: "More than HS"})
d["single_parent"] = d.family_r.isin([5, 6]).astype(float).where(d.family_r.notna())
d["age"] = d.sc_age_years

INC_LABELS = ["<100% FPL", "100-199%", "200-399%", "400%+"]


def income_group(fpl):
    return pd.cut(fpl, [0, 99.99, 199.99, 399.99, 1e9], labels=INC_LABELS)


def design(df, terms, imp):
    """Build the X matrix for one imputation."""
    df = df.copy()
    df["inc"] = income_group(df[f"fpl_i{imp}"])
    df["low_income"] = (df[f"fpl_i{imp}"] < 200).astype(float)
    parts = []
    for t in terms:
        if t == "inc":
            parts.append(pd.get_dummies(df.inc, prefix="inc", dtype=float).drop(columns="inc_400%+"))
        elif t in ("raceeth", "parent_ed"):
            dm = pd.get_dummies(df[t], prefix=t, dtype=float)
            parts.append(dm.drop(columns=dm.columns[0]))
        elif t == "age":
            parts.append(pd.get_dummies(df.age, prefix="age", dtype=float).iloc[:, 1:])
        elif ":" in t:
            a, b = t.split(":")
            parts.append((df[a] * df[b]).rename(t))
        else:
            parts.append(df[t])
    return sm.add_constant(pd.concat(parts, axis=1))


def fit_mi(df, y, terms):
    """Weighted linear probability model on each income imputation; Rubin's rules."""
    est, var = [], []
    for imp in range(1, 7):
        X = design(df, terms, imp)
        ok = df[y].notna() & X.notna().all(axis=1) & df.fwc.gt(0)
        r = sm.WLS(df.loc[ok, y], X[ok], weights=df.loc[ok, "fwc"]).fit(cov_type="HC1")
        est.append(r.params); var.append(r.bse ** 2)
    est, var = pd.concat(est, axis=1), pd.concat(var, axis=1)
    q = est.mean(axis=1)
    u = var.mean(axis=1)
    b = est.var(axis=1, ddof=1)
    se = np.sqrt(u + (1 + 1 / 6) * b)
    return pd.DataFrame({"coef": q, "se": se, "ci_low": q - 1.96 * se, "ci_high": q + 1.96 * se,
                         "n": int(ok.sum())})


def wmean(df, y, by):
    out = {}
    for imp in range(1, 7):
        g = income_group(df[f"fpl_i{imp}"])
        ok = df[y].notna()
        out[imp] = df[ok].assign(g=g[ok]).groupby("g", observed=True).apply(
            lambda x: np.average(x[y], weights=x.fwc), include_groups=False)
    return pd.concat(out, axis=1).mean(axis=1)


DEMOG = ["age", "female", "raceeth"]
CONTROLS = DEMOG + ["inc", "parent_ed", "single_parent", "ace_count"]
results = []

# ---- P1: access gap ----
prev = pd.DataFrame({y: wmean(d, y, "inc") for y in ["mentor", "any_activity", "volunteer"]})
prev.index.name = "income_group"
prev.round(4).to_csv(OUT / "prevalence_by_income.csv")
for label, terms in [("P1 mentor ~ income + demographics", ["inc"] + DEMOG),
                     ("P1 mentor ~ income + demographics + activities", ["inc", "any_activity", "volunteer"] + DEMOG),
                     ("P1 mentor ~ all controls + activities", CONTROLS + ["any_activity", "volunteer"])]:
    r = fit_mi(d, "mentor", terms)
    for k in [i for i in r.index if i.startswith("inc_") or i in ("any_activity", "volunteer")]:
        results.append({"model": label, "outcome": "mentor", "term": k, **r.loc[k].to_dict()})

# ---- P2 & P3: mentor and outcomes, overall and by income / adversity ----
for y in ["flourishing", "school_engaged", "hopeful_most"]:
    r = fit_mi(d, y, ["mentor"] + CONTROLS)
    results.append({"model": "P2 outcome ~ mentor + controls", "outcome": y, "term": "mentor",
                    **r.loc["mentor"].to_dict()})
    # low_income main effect is already spanned by the income-group dummies in CONTROLS
    r = fit_mi(d, y, ["mentor", "mentor:low_income"] + CONTROLS)
    for k in ["mentor", "mentor:low_income"]:
        results.append({"model": "P3 + mentor x low income (<200% FPL)", "outcome": y, "term": k,
                        **r.loc[k].to_dict()})
    r = fit_mi(d, y, ["mentor", "ace_2plus", "mentor:ace_2plus"] + CONTROLS)
    for k in ["mentor", "mentor:ace_2plus"]:
        results.append({"model": "P3 + mentor x 2+ adverse experiences", "outcome": y, "term": k,
                        **r.loc[k].to_dict()})

res = pd.DataFrame(results)
res.to_csv(OUT / "models.csv", index=False, float_format="%.4f")

# Mentor difference within each income group (for the figure)
sub = []
for y in ["flourishing", "school_engaged", "hopeful_most"]:
    for gi, g in enumerate(INC_LABELS):
        est, var, n = [], [], 0
        for imp in range(1, 7):
            grp = income_group(d[f"fpl_i{imp}"]) == g
            s = d[grp]
            X = design(s, ["mentor", "age", "female", "raceeth", "parent_ed", "single_parent",
                           "ace_count"], imp)
            ok = s[y].notna() & X.notna().all(axis=1)
            rr = sm.WLS(s.loc[ok, y], X[ok], weights=s.loc[ok, "fwc"]).fit(cov_type="HC1")
            est.append(rr.params["mentor"]); var.append(rr.bse["mentor"] ** 2); n = ok.sum()
        q, u, b = np.mean(est), np.mean(var), np.var(est, ddof=1)
        se = np.sqrt(u + (1 + 1 / 6) * b)
        sub.append({"outcome": y, "income_group": g, "mentor_diff": q, "ci_low": q - 1.96 * se,
                    "ci_high": q + 1.96 * se, "n": n})
sub = pd.DataFrame(sub)
sub.to_csv(OUT / "mentor_difference_by_income.csv", index=False, float_format="%.4f")

# ---- Figure ----
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"flourishing": "#2a78d6", "school_engaged": "#eb6834", "hopeful_most": "#1baf7a"}
NAMES = {"flourishing": "Flourishing", "school_engaged": "Engaged in school",
         "hopeful_most": "Stays hopeful"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), dpi=100, gridspec_kw={"width_ratios": [1, 1.5]})
fig.patch.set_facecolor(SURFACE)
ax = axes[0]; ax.set_facecolor(SURFACE)
x = np.arange(len(INC_LABELS))
ax.bar(x, prev.loc[INC_LABELS, "mentor"] * 100, color="#2a78d6", width=0.6)
for xi, v in zip(x, prev.loc[INC_LABELS, "mentor"] * 100):
    ax.text(xi, v + 0.6, f"{v:.0f}%", ha="center", color=TEXT, fontsize=10.5)
ax.set_xticks(x, INC_LABELS); ax.set_ylim(0, 105)
ax.set_title("Has a non-parental adult to rely on", color=TEXT, loc="left", fontsize=12)
ax.set_xlabel("Family income (% of federal poverty line)", color=TEXT2)
ax2 = axes[1]; ax2.set_facecolor(SURFACE)
for k, (y, c) in enumerate(COLORS.items()):
    s = sub[sub.outcome == y]
    xx = x + (k - 1) * 0.22
    ax2.errorbar(xx, s.mentor_diff * 100, yerr=[(s.mentor_diff - s.ci_low) * 100,
                 (s.ci_high - s.mentor_diff) * 100], fmt="o", ms=7, color=c, ecolor=c,
                 elinewidth=2, label=NAMES[y])
ax2.axhline(0, color=TEXT2, lw=0.8)
ax2.set_xticks(x, INC_LABELS)
ax2.set_title("Difference when child has such an adult (percentage points, 95% CI)",
              color=TEXT, loc="left", fontsize=12)
ax2.set_xlabel("Family income (% of federal poverty line)", color=TEXT2)
ax2.set_ylim(top=31)
ax2.legend(frameon=False, labelcolor=TEXT, loc="upper left", ncol=3)
for a in axes:
    a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    for s_ in ["top", "right"]:
        a.spines[s_].set_visible(False)
    a.tick_params(colors=TEXT2)
fig.text(0.02, 0.95, "Non-parental adults: fewer for low-income children, and they go with more flourishing",
         fontsize=14, weight="bold", color=TEXT)
fig.text(0.02, 0.012, "NSCH 2024, children 6-17, parent report, weighted. Right panel adjusts for age, sex, "
         "race/ethnicity, parent education, single parent, adverse experiences. Associations, not "
         "causal effects. Exploratory - verify before external use.", fontsize=8.5, color=TEXT2)
fig.subplots_adjust(left=0.05, right=0.98, top=0.84, bottom=0.17, wspace=0.18)
fig.savefig(OUT / "mentors_by_income.png", dpi=150, facecolor=SURFACE)

pd.set_option("display.width", 160)
print("Prevalence by income:\n", (prev * 100).round(1).to_string())
print("\nModels:\n", res[["model", "outcome", "term", "coef", "ci_low", "ci_high", "n"]]
      .round(3).to_string(index=False))
print("\nMentor difference by income:\n", sub.round(3).to_string(index=False))
