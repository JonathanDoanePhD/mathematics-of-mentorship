"""P5: School belonging, grades and hopelessness, by household hardship (YRBS 2023, national).

Belonging   QN103: strongly agree/agree "I feel close to people at my school"
Grades      QN87:  mostly A's or B's
Sad/hopeless QN26: felt sad or hopeless almost every day for 2+ weeks in the past year
Hardship    Q99:   an adult in the household tried hard to make sure basic needs were met
                   "never", "rarely" or "sometimes" (vs. most of the time / always)
Controls: age, sex, grade, race/ethnicity. Weighted (WEIGHT); SEs clustered by PSU.
CDC's derived QN variables are coded 1 = yes, 2 = no.

Run: python3 -I scripts/03_yrbs_belonging.py   ->  output/yrbs/
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "yrbs"
OUT = ROOT / "output" / "yrbs"
OUT.mkdir(parents=True, exist_ok=True)

# Column layout from CDC's SAS input program ("@start NAME width.")
layout = {}
for m in re.finditer(r"@(\d+)\s+(\w+)\s+\$?(\d+)\.", (RAW / "yrbs_2023_input.sas").read_text()):
    layout[m.group(2).upper()] = (int(m.group(1)) - 1, int(m.group(1)) - 1 + int(m.group(3)))
NEED = ["Q1", "Q2", "Q3", "Q99", "QN26", "QN87", "QN103", "RACEETH", "WEIGHT", "STRATUM", "PSU"]
d = pd.read_fwf(RAW / "yrbs_2023.dat", colspecs=[layout[v] for v in NEED], names=NEED, dtype=str)
for c in NEED:
    d[c] = pd.to_numeric(d[c], errors="coerce")

qn = lambda s: (s == 1).astype(float).where(s.isin([1, 2]))
d["belong"] = qn(d.QN103)
d["grades_ab"] = qn(d.QN87)
d["sad_hopeless"] = qn(d.QN26)
d["hardship"] = d.Q99.isin([1, 2, 3]).astype(float).where(d.Q99.isin([1, 2, 3, 4, 5]))
d["belong_x_hardship"] = d.belong * d.hardship
d = d[d.WEIGHT > 0]


def dummies(s, prefix):
    return pd.get_dummies(s, prefix=prefix, dtype=float, dummy_na=False).iloc[:, 1:]


def fit(y, xs):
    s = d.dropna(subset=[y, "Q1", "Q2", "Q3", "RACEETH", "PSU"] + xs)
    X = sm.add_constant(pd.concat([s[xs], dummies(s.Q1, "age"), dummies(s.Q2, "sex"),
                                   dummies(s.Q3, "grade"), dummies(s.RACEETH, "re")], axis=1))
    return sm.WLS(s[y], X, weights=s.WEIGHT).fit(cov_type="cluster", cov_kwds={"groups": s.PSU})


rows = []
for y in ["grades_ab", "sad_hopeless"]:
    for name, xs in [("belonging + controls", ["belong", "hardship"]),
                     ("+ belonging x hardship", ["belong", "hardship", "belong_x_hardship"])]:
        r = fit(y, xs)
        for x in xs:
            rows.append({"outcome": y, "model": name, "term": x, "coef": r.params[x],
                         "ci_low": r.conf_int().loc[x, 0], "ci_high": r.conf_int().loc[x, 1],
                         "n": int(r.nobs)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "models.csv", index=False, float_format="%.4f")

# Weighted prevalence of belonging by hardship
s = d.dropna(subset=["belong", "hardship"])
prev = s.groupby("hardship").apply(lambda x: np.average(x.belong, weights=x.WEIGHT),
                                   include_groups=False).rename({0.0: "no hardship", 1.0: "hardship"})
share_hard = np.average(d.dropna(subset=["hardship"]).hardship, weights=d.dropna(subset=["hardship"]).WEIGHT)
pd.DataFrame({"belong_share": prev}).to_csv(OUT / "belonging_by_hardship.csv", float_format="%.4f")

print(f"Students: {len(d):,}; answered belonging: {d.belong.notna().sum():,}; "
      f"hardship share (weighted): {share_hard:.1%}")
print("Belonging by hardship:", prev.round(3).to_dict())
print(res.round(3).to_string(index=False))
