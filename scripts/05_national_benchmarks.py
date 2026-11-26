"""National benchmarks for the Project 03 survey items (same answer definitions).

NSCH 2024 (parent report), teens 12-17: SA1 mentor, EN1-EN5 engagement/persistence
(always/usually), HO1 hopeful (all/most of the time). By family income (< 200% vs 200%+ FPL,
averaged over the 6 income imputations).
YRBS 2023 (student report), high school: BE1 feel close to people at school (agree/strongly
agree), HO2 persistent sadness/hopelessness. By household hardship (basic needs not reliably met).

CIs use a Kish effective sample size (weights only), ignoring clustering and strata, so true
design-based intervals are somewhat wider.

Run: python3 -I scripts/05_national_benchmarks.py -> output/benchmarks/national_benchmarks.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)


def wshare(y, w):
    ok = y.notna() & w.notna()
    y, w = y[ok], w[ok]
    p = np.average(y, weights=w)
    neff = w.sum() ** 2 / (w ** 2).sum()
    se = np.sqrt(p * (1 - p) / neff)
    return p, p - 1.96 * se, p + 1.96 * se, int(ok.sum())


rows = []
# ---- NSCH ----
cols = ["k9q96", "sc_age_years", "fwc", "k7q82_r", "k7q83_r", "k7q84_r", "k7q85_r", "k6q71_r", "hopeful"] + \
       [f"fpl_i{i}" for i in range(1, 7)]
n = pd.read_stata(RAW / "nsch" / "nsch_2024e_topical.dta", columns=cols, convert_categoricals=False)
n = n[n.sc_age_years.between(12, 17)]
always_usually = lambda s: s.isin([1, 2]).astype(float).where(s.isin([1, 2, 3, 4]))
items = {
    "SA1": ("Has a non-parental adult to rely on for advice", (n.k9q96 == 1).astype(float).where(n.k9q96.isin([1, 2]))),
    "EN1": ("Cares about doing well in school (always/usually)", always_usually(n.k7q82_r)),
    "EN2": ("Does all required homework (always/usually)", always_usually(n.k7q83_r)),
    "EN3": ("Works to finish tasks started (always/usually)", always_usually(n.k7q84_r)),
    "EN4": ("Stays calm when challenged (always/usually)", always_usually(n.k7q85_r)),
    "EN5": ("Shows interest and curiosity (always/usually)", always_usually(n.k6q71_r)),
    "HO1": ("Stays hopeful facing problems (all/most of the time)", n.hopeful.isin([1, 2]).astype(float).where(n.hopeful.isin([1, 2, 3, 4]))),
}
for item, (measure, y) in items.items():
    p = wshare(y, n.fwc)
    rows.append([item, measure, "All teens 12-17", *p])
    for lab, sel in [("Family income < 200% of poverty line", lambda f: f < 200),
                     ("Family income 200%+ of poverty line", lambda f: f >= 200)]:
        est = [wshare(y[sel(n[f"fpl_i{i}"])], n.fwc[sel(n[f"fpl_i{i}"])]) for i in range(1, 7)]
        rows.append([item, measure, lab, *np.mean([e[:3] for e in est], axis=0), int(np.mean([e[3] for e in est]))])
nsch_rows = len(rows)

# ---- YRBS ----
layout = {m.group(2).upper(): (int(m.group(1)) - 1, int(m.group(1)) - 1 + int(m.group(3)))
          for m in re.finditer(r"@(\d+)\s+(\w+)\s+\$?(\d+)\.", (RAW / "yrbs" / "yrbs_2023_input.sas").read_text())}
need = ["QN103", "QN26", "Q99", "WEIGHT"]
y = pd.read_fwf(RAW / "yrbs" / "yrbs_2023.dat", colspecs=[layout[v] for v in need], names=need)
y = y[y.WEIGHT > 0]
qn = lambda s: (s == 1).astype(float).where(s.isin([1, 2]))
hard = y.Q99.isin([1, 2, 3])
for item, measure, v in [("BE1", "Feels close to people at school (agree/strongly agree)", qn(y.QN103)),
                         ("HO2", "Felt sad or hopeless almost every day for 2+ weeks (past year)", qn(y.QN26))]:
    rows.append([item, measure, "All high school students", *wshare(v, y.WEIGHT)])
    rows.append([item, measure, "Basic needs not reliably met", *wshare(v[hard], y.WEIGHT[hard])])
    rows.append([item, measure, "Basic needs reliably met", *wshare(v[y.Q99.isin([4, 5])], y.WEIGHT[y.Q99.isin([4, 5])])])

out = pd.DataFrame(rows, columns=["item_id", "measure", "group", "share", "low", "high", "n"])
out["source"] = ["NSCH 2024" if i < nsch_rows else "YRBS 2023" for i in range(len(out))]
out["note"] = np.where(out.source == "NSCH 2024", "Parent report, ages 12-17; weighted",
                       "Student report, grades 9-12; weighted")
out.to_csv(OUT / "national_benchmarks.csv", index=False, float_format="%.4f")
print(out.round(3).to_string(index=False))
