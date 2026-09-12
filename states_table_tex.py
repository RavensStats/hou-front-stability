"""Turn STATES_TABLE.md (written by papertable.py from the snapshots themselves) into the paper's Table 1,
tex/../scratchpad/p1_states.tex.  One row per state, both amplification labels, and the T-free quantities.
    python states_table_tex.py
"""
import os, re

ROOT = r"C:\Users\mtgra\Desktop\Milennium Prize" + "\\"
SCR = (r"C:\Users\mtgra\AppData\Local\Temp\claude\c--Users-mtgra-Desktop-Milennium-Prize"
       r"\c425528c-ff75-49b5-ad48-a25823509b82\scratchpad" + "\\")

RUN = {"axiphys_513_512_e513a5": ("Euler", "$513\\times512$, fixed"),
       "axiphys_513_512_nsz": ("NS first stage", "$513\\times512$, fixed"),
       "nsz2": ("continuation", "$513\\times1024$, follower"),
       "nsz5": ("continuation", "$513\\times1536$, follower"),
       "nsz6": ("continuation", "$513\\times2048$, follower")}

rows = []
for line in open(ROOT + "STATES_TABLE.md", encoding="utf-8"):
    if not line.startswith("| ") or line.startswith("| file") or line.startswith("|---"):
        continue
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    f, t, tmt, aa0, lab, rz, om, q, qat, core, rur, ruz = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]
    key = next((k for k in RUN if f.startswith(k)), None)
    rows.append(dict(f=f, key=key, t=float(t), tmt=float(tmt), lab=float(lab), om=float(om),
                     q=float(q), core=float(core), rur=float(rur), ruz=float(ruz)))
rows.sort(key=lambda r: (r["key"] != "axiphys_513_512_e513a5", r["t"], r["f"]))

out = ["\\begin{table}[tbp]", "\\centering\\small", "\\begin{tabular}{llcccccccc}", "\\hline",
       "state & run & $t$ & $T - t$ & $\\max|\\omega|$ & label & $\\sup |u||x'|$ & $|u|r$ core & $|r u^r|$ & $|r u^z|$ \\\\",
       " & & & & ratio & $\\max|u_1|/12000$ & & & core & core \\\\", "\\hline"]
for r in rows:
    excl = r["q"] > 100
    name = ("Euler x28" if r["key"] == "axiphys_513_512_e513a5" else
            "NS x144" if r["key"] == "axiphys_513_512_nsz" else f"x{r['lab']:.0f}")
    if excl:
        name += "$^\\dagger$"
    past = r["tmt"] < 0
    out.append(f"{name} & {RUN[r['key']][0]} & {r['t']:.7f} & " +
               (f"$-${abs(r['tmt']):.2e}" if past else f"{r['tmt']:.2e}") +
               f" & {r['om']:.1f} & {r['lab']:.1f} & {r['q']:.2f} & {r['core']:.1f} & {r['rur']:.1f} & {r['ruz']:.1f} \\\\")
out += ["\\hline", "\\end{tabular}",
        "\\caption{The states used in this paper, with every quantity recomputed from the archived snapshot. "
        "`x28' and `x144' are amplifications of the full vorticity maximum, which are Hou's labels; the `label' "
        "column is $\\max|u_1|/12000$ of Section~\\ref{sec:gates}, which is how the states are named "
        "in Sections~\\ref{sec:consts} to \\ref{sec:theorems}. The other row names are the "
        "labels $\\max|u_1|/12000$ of Section~\\ref{sec:gates}. The last four columns are independent of the fitted "
        "singular time: the global supremum of $|u|\\,|x'|$, and $|u|\\,r$, $|r u^r|$, $|r u^z|$ in the core box "
        "$r < 3R$, $|z - Z| < 3Z$ of \\eqref{eq:lambda}. $^\\dagger$The snapshot excluded in "
        "Section~\\ref{sec:l3}, on a degraded map; its twin at the same instant is the row below it. "
        "The $T - t$ column uses the fitted $T = 0.0022865$ for every row; on the Euler convention of "
        "Section~\\ref{sec:euler} ($T_E = 0.002278$) the Euler state has $T_E - t = \\sn{7.0}{-5}$. "
        "The last row lies past the fitted singular time.}",
        "\\label{tab:states}", "\\end{table}", ""]
open(SCR + "p1_states.tex", "w", encoding="utf-8").write("\n".join(out))
print(f"p1_states.tex written: {len(rows)} states")
