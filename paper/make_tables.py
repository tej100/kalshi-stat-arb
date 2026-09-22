"""Write the appendix robustness table from artifacts/sensitivity.csv.

Run after analysis/sensitivity.py:  python paper/make_tables.py
Writes paper/tables/sensitivity.tex, which main.tex \\input's.
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = pd.read_csv(os.path.join(ROOT, "artifacts", "sensitivity.csv"))
GROUPS = {"baseline": "Headline", "cleaning": "Cleaning", "smile": "Smile",
          "sizing": "Sizing", "costs": "Costs", "strategy": "Exit rule",
          "timing": "Execution", "marking": "Marking"}


def f(x):
    return f"{x:.2f}".replace("-", "$-$")


lines = [r"\begin{tabular}{llrrrcr}",
         r"& Variant & 2022 & 2023 & 2024 & Pooled [95\% CI] & MAE \\", r"\hline"]
last = None
for _, r in R.iterrows():
    g = GROUPS.get(r.group, r.group) if r.group != last else ""
    if r.group != last and last is not None:
        lines.append(r"\hline")
    last = r.group
    variant = "as in the paper" if r.variant == "headline" else r.variant
    name = variant.replace("%", r"\%").replace("$", r"\$")
    name = name.replace("beta = ", r"$\beta$ = ").replace("lot 1 contracts", "lot of 1 contract") \
               .replace("lot 4 contracts", "lot of 4 contracts").replace("lot 16 contracts", "lot of 16 contracts")
    lines.append(f"{g} & {name} & {f(r.s2022)} & {f(r.s2023)} & {f(r.s2024)} & "
                 f"{f(r.pooled)} [{f(r.lo)},\\,{f(r.hi)}] & \\${r.mae:.2f} \\\\")
lines.append(r"\end{tabular}")
os.makedirs(os.path.join(ROOT, "paper", "tables"), exist_ok=True)
with open(os.path.join(ROOT, "paper", "tables", "sensitivity.tex"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote paper/tables/sensitivity.tex")
