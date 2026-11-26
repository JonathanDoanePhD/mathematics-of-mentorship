"""P6: the self-efficacy pathway in PISA 2022.

A. United States: supportive teacher relationships (RELATST) -> sense of belonging (BELONG)
   -> math self-efficacy (MATHEFF) -> math achievement (PV1-10MATH), with indirect effects.
B. United States: cross-class exposure. For disadvantaged students (bottom national quartile
   of ESCS), how do belonging, self-efficacy and math relate to the school's average ESCS?
   Tests a "big fish, little pond" counter-effect. Also: does teacher support matter more
   for disadvantaged students?
C. All PISA countries: the same pathway estimated country by country (final weights only),
   to see how general it is.

Variance: plausible values combined with Rubin's rules; sampling variance from the 80 BRR
replicate weights with Fay's factor 0.5 (variance = sum((theta_r - theta)^2) / 20).
All index variables standardized within the analysis sample (student-weighted).
Associations from one time point, not causal effects.

Run: python3 -I scripts/04_pisa_selfefficacy.py -> output/pisa/
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pisa"
OUT = ROOT / "output" / "pisa"
OUT.mkdir(parents=True, exist_ok=True)

PV = [f"PV{i}MATH" for i in range(1, 11)]
REP = [f"W_FSTURWT{i}" for i in range(1, 81)]
IDX = ["ESCS", "BELONG", "RELATST", "TEACHSUP", "MATHEFF", "ANXMAT", "FAMSUP"]
COLS = ["CNT", "CNTSCHID", "CNTSTUID", "ST004D01T", "W_FSTUWT"] + IDX + PV + REP

sav = sorted(glob.glob(str(RAW / "*.sav")) + glob.glob(str(RAW / "*.SAV")))[0]
meta_cols = pyreadstat.read_sav(sav, metadataonly=True)[1].column_names
missing = [c for c in COLS if c not in meta_cols]
if missing:
    raise SystemExit(f"Variables not found in {sav}: {missing}")
cache = RAW / "pisa_subset.pkl"
if cache.exists():
    df = pd.read_pickle(cache)
else:
    df, _ = pyreadstat.read_sav(sav, usecols=COLS)
    df.to_pickle(cache)
df["female"] = (df.ST004D01T == 1).astype(float)


def wstd(s, w):
    m = np.average(s, weights=w)
    return (s - m) / np.sqrt(np.average((s - m) ** 2, weights=w))


def wls_coef(y, X, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def estimate(d, fn):
    """fn(d, weight_column, pv_column) -> dict of statistics. Returns point, SE combining
    BRR sampling variance (Fay 0.5) and PV imputation variance (Rubin)."""
    per_pv, samp_var = [], []
    for pv in PV:
        full = fn(d, "W_FSTUWT", pv)
        keys = list(full)
        reps = np.array([[fn(d, r, pv)[k] for k in keys] for r in REP])
        th = np.array([full[k] for k in keys])
        per_pv.append(th)
        samp_var.append(((reps - th) ** 2).sum(axis=0) / (80 * 0.25))
    per_pv, samp_var = np.array(per_pv), np.array(samp_var)
    point = per_pv.mean(axis=0)
    var = samp_var.mean(axis=0) + (1 + 1 / 10) * per_pv.var(axis=0, ddof=1)
    return pd.DataFrame({"stat": keys, "estimate": point, "se": np.sqrt(var),
                         "ci_low": point - 1.96 * np.sqrt(var), "ci_high": point + 1.96 * np.sqrt(var)})


# ---------------- A. US pathway ----------------
us = df[df.CNT == "USA"].dropna(subset=["ESCS", "BELONG", "RELATST", "MATHEFF", "W_FSTUWT"]).copy()
for c in ["ESCS", "BELONG", "RELATST", "TEACHSUP", "MATHEFF", "ANXMAT", "FAMSUP"]:
    ok = us[c].notna()
    us.loc[ok, c + "_z"] = wstd(us.loc[ok, c], us.loc[ok, "W_FSTUWT"])
math_sd = np.sqrt(np.average((us.PV1MATH - np.average(us.PV1MATH, weights=us.W_FSTUWT)) ** 2, weights=us.W_FSTUWT))


def path_model(d, w, pv):
    wt = d[w].values
    one = np.ones(len(d))
    ctrl = [d.ESCS_z.values, d.female.values]
    rel, bel, eff = d.RELATST_z.values, d.BELONG_z.values, d.MATHEFF_z.values
    y = (d[pv].values - np.average(d[pv], weights=d.W_FSTUWT)) / math_sd  # math in SD units
    a = wls_coef(bel, np.column_stack([one, rel, *ctrl]), wt)                 # belong ~ rel
    b = wls_coef(eff, np.column_stack([one, bel, rel, *ctrl]), wt)            # eff ~ belong + rel
    c = wls_coef(y, np.column_stack([one, eff, bel, rel, *ctrl]), wt)         # math ~ eff + belong + rel
    tot = wls_coef(y, np.column_stack([one, rel, *ctrl]), wt)                 # total effect of rel
    return {
        "a: teacher relationships -> belonging": a[1],
        "b1: belonging -> self-efficacy": b[1],
        "b2: teacher relationships -> self-efficacy (direct)": b[2],
        "c1: self-efficacy -> math": c[1],
        "c2: belonging -> math (direct)": c[2],
        "c3: teacher relationships -> math (direct)": c[3],
        "total: teacher relationships -> math": tot[1],
        "indirect: rel -> belong -> efficacy -> math": a[1] * b[1] * c[1],
        "indirect: rel -> efficacy -> math": b[2] * c[1],
        "indirect: rel -> belong -> math": a[1] * c[2],
    }


A = estimate(us, path_model)
A["n_students"] = len(us)
A.to_csv(OUT / "us_pathway.csv", index=False, float_format="%.4f")

# Also: is belonging tied to self-efficacy beyond family support and math anxiety?
def eff_model(d, w, pv):
    dd = d.dropna(subset=["FAMSUP_z", "ANXMAT_z"])
    wt = dd[w].values
    X = np.column_stack([np.ones(len(dd)), dd.BELONG_z, dd.RELATST_z, dd.FAMSUP_z, dd.ANXMAT_z, dd.ESCS_z, dd.female])
    b = wls_coef(dd.MATHEFF_z.values, X, wt)
    return {"belonging": b[1], "teacher relationships": b[2], "family support": b[3], "math anxiety": b[4], "ESCS": b[5]}


A2 = estimate(us, eff_model)
A2.to_csv(OUT / "us_selfefficacy_correlates.csv", index=False, float_format="%.4f")

# ---------------- B. Cross-class exposure (US) ----------------
q25 = np.quantile(us.ESCS, 0.25)  # unweighted approx is fine for a cut; report it
us["disadv"] = (us.ESCS <= q25).astype(float)
sch = us.groupby("CNTSCHID").apply(lambda g: pd.Series({"sch_escs": np.average(g.ESCS, weights=g.W_FSTUWT),
                                                         "sch_n": len(g)}), include_groups=False)
us = us.merge(sch, left_on="CNTSCHID", right_index=True)
# leave-one-out school mean (so a student's own ESCS doesn't drive their school's average)
tot_w = us.groupby("CNTSCHID").W_FSTUWT.transform("sum")
tot_wx = (us.ESCS * us.W_FSTUWT).groupby(us.CNTSCHID).transform("sum")
us["sch_escs_loo"] = (tot_wx - us.ESCS * us.W_FSTUWT) / (tot_w - us.W_FSTUWT)
okl = us.sch_escs_loo.notna()  # schools with a single sampled student have no leave-one-out mean
us.loc[okl, "sch_escs_z"] = wstd(us.loc[okl, "sch_escs_loo"], us.loc[okl, "W_FSTUWT"])
dis = us[(us.disadv == 1) & (us.sch_n >= 5) & us.sch_escs_z.notna()].copy()


def exposure_model(d, w, pv):
    wt = d[w].values
    X = np.column_stack([np.ones(len(d)), d.sch_escs_z, d.ESCS_z, d.female])
    y = (d[pv].values - np.average(us[pv], weights=us.W_FSTUWT)) / math_sd
    out = {}
    for name, yy in [("math (SD)", y), ("belonging", d.BELONG_z.values), ("self-efficacy", d.MATHEFF_z.values)]:
        out[f"school SES mix -> {name}"] = wls_coef(yy, X, wt)[1]
    # Self-efficacy holding achievement constant: the 'big fish, little pond' test
    Xa = np.column_stack([X, y])
    out["school SES mix -> self-efficacy, same math score"] = wls_coef(d.MATHEFF_z.values, Xa, wt)[1]
    out["school SES mix -> belonging, same math score"] = wls_coef(d.BELONG_z.values, Xa, wt)[1]
    return out


B = estimate(dis, exposure_model)
B["n_students"] = len(dis)
B.to_csv(OUT / "us_exposure_disadvantaged.csv", index=False, float_format="%.4f")


def moderation(d, w, pv):
    wt = d[w].values
    X = np.column_stack([np.ones(len(d)), d.RELATST_z, d.disadv, d.RELATST_z * d.disadv, d.ESCS_z, d.female])
    out = {}
    for name, yy in [("belonging", d.BELONG_z.values), ("self-efficacy", d.MATHEFF_z.values)]:
        b = wls_coef(yy, X, wt)
        out[f"teacher relationships -> {name} (others)"] = b[1]
        out[f"extra for disadvantaged -> {name}"] = b[3]
    return out


Bm = estimate(us, moderation)
Bm.to_csv(OUT / "us_teacher_support_by_disadvantage.csv", index=False, float_format="%.4f")

# ---------------- C. International ----------------
rows = []
for cnt, g in df.groupby("CNT"):
    g = g.dropna(subset=["ESCS", "BELONG", "RELATST", "MATHEFF", "W_FSTUWT"])
    if len(g) < 1000:
        continue
    g = g.copy()
    for c in ["ESCS", "BELONG", "RELATST", "MATHEFF"]:
        g[c + "_z"] = wstd(g[c], g.W_FSTUWT)
    msd = np.sqrt(np.average((g.PV1MATH - np.average(g.PV1MATH, weights=g.W_FSTUWT)) ** 2, weights=g.W_FSTUWT))
    ests = []
    for pv in PV:
        saved, math_sd = math_sd, msd
        ests.append(path_model(g, "W_FSTUWT", pv))
        math_sd = saved
    mean = pd.DataFrame(ests).mean()
    rows.append({"country": cnt, "n": len(g), **mean.to_dict()})
C = pd.DataFrame(rows)
C.to_csv(OUT / "international_pathway.csv", index=False, float_format="%.4f")

# ---------------- Figure: path diagram numbers + international spread ----------------
SURFACE, TEXT, TEXT2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
paths = [("a: teacher relationships -> belonging", "Teacher relationships → belonging"),
         ("b1: belonging -> self-efficacy", "Belonging → self-efficacy"),
         ("c1: self-efficacy -> math", "Self-efficacy → math"),
         ("b2: teacher relationships -> self-efficacy (direct)", "Teacher relationships → self-efficacy"),
         ("c2: belonging -> math (direct)", "Belonging → math (direct)")]
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=100)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for i, (k, lab) in enumerate(paths):
    vals = C[k].dropna()
    y = len(paths) - 1 - i
    ax.scatter(vals, np.full(len(vals), y) + np.random.default_rng(i).uniform(-0.18, 0.18, len(vals)),
               s=14, color=C1, alpha=0.35, lw=0)
    r = A.set_index("stat").loc[k]
    ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt="D", ms=8,
                color=C2, ecolor=C2, elinewidth=2.5)
ax.axvline(0, color=TEXT2, lw=0.8)
ax.set_yticks(range(len(paths)), [p[1] for p in paths][::-1], color=TEXT)
ax.set_xlabel("Standardized path coefficient (SD change per SD)", color=TEXT2)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ["top", "right", "left"]:
    ax.spines[s_].set_visible(False)
ax.tick_params(colors=TEXT2)
ax.scatter([], [], s=30, color=C1, alpha=0.6, label=f"Each PISA country/economy ({len(C)})")
ax.errorbar([], [], fmt="D", color=C2, label="United States (95% CI)")
ax.legend(loc="lower right", frameon=False, labelcolor=TEXT)
n_all = int(((C[paths[0][0]] > 0) & (C[paths[1][0]] > 0) & (C[paths[3][0]] > 0)).sum())
fig.text(0.02, 0.95, f"Supportive teachers go with belonging and self-efficacy in {n_all} of {len(C)} countries studied",
         fontsize=13.5, weight="bold", color=TEXT)
fig.text(0.02, 0.91, "PISA 2022, 15-year-olds. Each path adjusts for family socioeconomic status and gender.",
         color=TEXT2)
fig.text(0.02, 0.012, "Source: OECD PISA 2022. Associations at one time point, not causal effects. "
         "Exploratory - verify before external use.", fontsize=8.5, color=TEXT2)
fig.subplots_adjust(left=0.3, right=0.97, top=0.86, bottom=0.12)
fig.savefig(OUT / "pathway_international.png", dpi=150, facecolor=SURFACE)

pd.set_option("display.width", 170)
print(f"US students: {len(us)}; disadvantaged (ESCS <= {q25:.2f}) in schools with 5+ sampled: {len(dis)}")
print(A.round(3).to_string(index=False))
print(A2.round(3).to_string(index=False))
print(B.round(3).to_string(index=False))
print(Bm.round(3).to_string(index=False))
share_pos = {k: (C[k] > 0).mean() for k, _ in paths}
print(f"Countries: {len(C)}; share with positive path:", {k.split(':')[0]: round(v, 2) for k, v in share_pos.items()})
print(C[[k for k, _ in paths]].describe().round(3).to_string())
