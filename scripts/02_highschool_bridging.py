"""P4: Do high schools where lower-income students befriend higher-income peers graduate more
economically disadvantaged students, beyond exposure and school poverty?

Unit: U.S. public high school. Cross-class measures (Social Capital Atlas, parental-SES
versions, students born 1990-2000):
  exposure_parent_ses_hs  how many higher-SES schoolmates a lower-SES student has (x2; 1 = even mix)
  bias_parent_ses_hs      friending bias: 1 - (high-SES friend share / high-SES exposure);
                          higher = fewer cross-class friendships than exposure would allow
  ec_parent_ses_hs        economic connectedness (combines both)
Outcome: 2018-19 four-year adjusted cohort graduation rate for economically disadvantaged
students (EDFacts via Urban Institute; reported as ranges, midpoint used).
Controls: % free/reduced-price lunch and log enrollment (NCES CCD 2018-19), state fixed
effects. Weighted by disadvantaged cohort size; standard errors clustered by district.

Run: python3 -I scripts/02_highschool_bridging.py   ->  output/highschool/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "highschool"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Social Capital Atlas (public schools have 12-digit NCES IDs) ----
sca = pd.read_csv(RAW / "social_capital_high_school.csv", dtype={"high_school": str, "county": str})
sca = sca[sca.high_school.str.len() == 12].rename(columns={"high_school": "ncessch"})

# ---- Graduation rates ----
g = pd.read_csv(RAW / "grad_rates_2018.csv", dtype={"ncessch": str, "leaid": str})
g["ncessch"] = g.ncessch.str.zfill(12)
sub = ["race", "lep", "homeless", "disability", "econ_disadvantaged", "foster_care"]
others = lambda keep: np.logical_and.reduce([g[c] == 99 for c in sub if c != keep])
allst = g[np.logical_and.reduce([g[c] == 99 for c in sub])]
ed = g[(g.econ_disadvantaged == 1) & others("econ_disadvantaged")]
grad = (allst[["ncessch", "leaid", "fips", "school_name", "cohort_num", "grad_rate_midpt"]]
        .rename(columns={"cohort_num": "cohort_all", "grad_rate_midpt": "grad_all"})
        .merge(ed[["ncessch", "cohort_num", "grad_rate_midpt", "grad_rate_low", "grad_rate_high"]]
               .rename(columns={"cohort_num": "cohort_ed", "grad_rate_midpt": "grad_ed",
                                "grad_rate_low": "grad_ed_low", "grad_rate_high": "grad_ed_high"}),
               on="ncessch", how="inner"))
grad = grad[(grad.grad_ed >= 0) & (grad.grad_all >= 0)]  # negative codes = missing/suppressed

# ---- NCES CCD: free/reduced lunch and enrollment ----
lunch = pd.read_csv(RAW / "ccd" / "ccd_sch_033_1819_l_1a_091019.csv", dtype=str, encoding="cp1252",
                    usecols=["NCESSCH", "DATA_GROUP", "LUNCH_PROGRAM", "STUDENT_COUNT"])
frl = (lunch[(lunch.DATA_GROUP == "Free and Reduced-price Lunch Table")
             & lunch.LUNCH_PROGRAM.isin(["Free lunch qualified", "Reduced-price lunch qualified"])]
       .assign(n=lambda x: pd.to_numeric(x.STUDENT_COUNT, errors="coerce"))
       .groupby("NCESSCH").n.sum(min_count=1).rename("frl_count"))
mem = pd.read_csv(RAW / "ccd" / "ccd_SCH_052_1819_l_1a_091019.csv", dtype=str, encoding="cp1252",
                  usecols=["NCESSCH", "TOTAL_INDICATOR", "STUDENT_COUNT"])
enr = (mem[mem.TOTAL_INDICATOR == "Education Unit Total"]
       .assign(enrollment=lambda x: pd.to_numeric(x.STUDENT_COUNT, errors="coerce"))
       .set_index("NCESSCH").enrollment)
ccd = pd.concat([frl, enr], axis=1).reset_index().rename(columns={"index": "ncessch",
                                                                  "NCESSCH": "ncessch"})
ccd["frl_share"] = (ccd.frl_count / ccd.enrollment).where(ccd.enrollment > 0).clip(upper=1)

d = (sca.merge(grad, on="ncessch", how="inner").merge(ccd, on="ncessch", how="left"))
CC = ["exposure_parent_ses_hs", "bias_parent_ses_hs", "ec_parent_ses_hs"]
d = d.dropna(subset=CC + ["frl_share", "enrollment", "grad_ed", "cohort_ed"])
d = d[(d.enrollment >= 100) & (d.cohort_ed >= 20)].copy()
d["log_enrollment"] = np.log(d.enrollment)
d["state"] = d.fips.astype(int)
match_info = {"sca_public_schools": len(sca), "analysis_schools": len(d)}


def wz(s, w):
    m = np.average(s, weights=w)
    return (s - m) / np.sqrt(np.average((s - m) ** 2, weights=w))


for c in CC + ["frl_share", "log_enrollment"]:
    d[c + "_z"] = wz(d[c], d.cohort_ed)


def fit(y, xs, w="cohort_ed", fe=True, data=None):
    data = d if data is None else data
    X = data[xs].copy()
    if fe:
        X = pd.concat([X, pd.get_dummies(data.state, prefix="st", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    return sm.WLS(data[y], X, weights=data[w]).fit(cov_type="cluster",
                                                    cov_kwds={"groups": data.leaid})


rows = []
specs = [
    ("A. exposure + friending bias", ["exposure_parent_ses_hs_z", "bias_parent_ses_hs_z"]),
    ("B. + school poverty & size", ["exposure_parent_ses_hs_z", "bias_parent_ses_hs_z",
                                    "frl_share_z", "log_enrollment_z"]),
    ("C. economic connectedness + poverty & size", ["ec_parent_ses_hs_z", "frl_share_z",
                                                    "log_enrollment_z"]),
]
for y, w, yl in [("grad_ed", "cohort_ed", "Grad rate, economically disadvantaged"),
                 ("grad_all", "cohort_all", "Grad rate, all students")]:
    for name, xs in specs:
        for fe in (False, True):
            r = fit(y, xs, w=w, fe=fe)
            for x in xs:
                rows.append({"outcome": yl, "model": name, "state_fe": fe, "term": x.replace("_z", ""),
                             "coef_per_sd": r.params[x], "ci_low": r.conf_int().loc[x, 0],
                             "ci_high": r.conf_int().loc[x, 1], "n_schools": int(r.nobs),
                             "r2": r.rsquared})
res = pd.DataFrame(rows)
res.to_csv(OUT / "models.csv", index=False, float_format="%.4f")

# Does friending bias matter more in mixed schools (where there are cross-class peers to befriend)?
d["mixed"] = (d.exposure_parent_ses_hs >= d.exposure_parent_ses_hs.median()).astype(float)
d["bias_x_mixed"] = d.bias_parent_ses_hs_z * d.mixed
r_int = fit("grad_ed", ["exposure_parent_ses_hs_z", "bias_parent_ses_hs_z", "bias_x_mixed",
                        "frl_share_z", "log_enrollment_z"])
inter = {k: (r_int.params[k], *r_int.conf_int().loc[k]) for k in ["bias_parent_ses_hs_z", "bias_x_mixed"]}

# Descriptives
desc = d[CC + ["frl_share", "enrollment", "grad_ed", "grad_all", "cohort_ed"]].describe().T
desc.to_csv(OUT / "descriptives.csv", float_format="%.3f")
corr = d[CC + ["frl_share", "grad_ed"]].corr().round(2)
corr.to_csv(OUT / "correlations.csv")

# ---- Figure: binned friending bias vs disadvantaged grad rate, residualized ----
SURFACE, TEXT, TEXT2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
ctrl = ["exposure_parent_ses_hs_z", "frl_share_z", "log_enrollment_z"]
Xc = sm.add_constant(pd.concat([d[ctrl], pd.get_dummies(d.state, prefix="st", drop_first=True, dtype=float)], axis=1))
ry = d.grad_ed - sm.WLS(d.grad_ed, Xc, weights=d.cohort_ed).fit().fittedvalues
rx = d.bias_parent_ses_hs - sm.WLS(d.bias_parent_ses_hs, Xc, weights=d.cohort_ed).fit().fittedvalues
bins = pd.qcut(rx, 20, labels=False)
bx = rx.groupby(bins).mean() + np.average(d.bias_parent_ses_hs, weights=d.cohort_ed)
by = pd.Series(ry).groupby(bins).apply(lambda s: np.average(s, weights=d.cohort_ed[s.index])) \
    + np.average(d.grad_ed, weights=d.cohort_ed)
fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.scatter(bx, by, s=60, color=C1, zorder=3)
b_slope = sm.WLS(ry, sm.add_constant(rx), weights=d.cohort_ed).fit().params.iloc[1]
xx = np.linspace(bx.min(), bx.max(), 20)
ax.plot(xx, by.mean() + b_slope * (xx - bx.mean()), color=TEXT, lw=2)
ax.set_xlabel("Friending bias (higher = fewer cross-class friendships than exposure allows)", color=TEXT2)
ax.set_ylabel("Graduation rate, economically disadvantaged students (%)", color=TEXT2)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
ax.tick_params(colors=TEXT2)
fig.text(0.02, 0.955, "Where lower-income students befriend higher-income peers, more of them graduate",
         fontsize=13.5, weight="bold", color=TEXT)
fig.text(0.02, 0.918, f"{len(d):,} public high schools in 20 equal-size groups, after adjusting for exposure, "
         "school poverty, size and state.", color=TEXT2)
fig.text(0.02, 0.012, "Sources: Social Capital Atlas; EDFacts via Urban Institute; NCES CCD 2018-19. "
         "Association, not a causal effect. Exploratory - verify before external use.",
         fontsize=8.5, color=TEXT2)
fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12)
fig.savefig(OUT / "friending_bias_graduation.png", dpi=150, facecolor=SURFACE)

pd.set_option("display.width", 160)
print(match_info)
print(corr.to_string())
print(res[res.state_fe | (res.model.str.startswith("A"))][["outcome", "model", "state_fe", "term",
      "coef_per_sd", "ci_low", "ci_high", "n_schools", "r2"]].round(3).to_string(index=False))
print("\nInteraction (state FE):", {k: tuple(round(v, 3) for v in t) for k, t in inter.items()})
