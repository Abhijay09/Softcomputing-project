"""Render the report formulas as images (matplotlib mathtext, no LaTeX install needed)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
os.makedirs("docs/eq", exist_ok=True)

EQ = {
 "output":   r"$(c_1,\rho_1),\ (c_2,\rho_2),\ \ldots,\ (c_n,\rho_n)$",
 "rule":     r"$\mathrm{IF}\ x_1\ \mathrm{is}\ {\sim}c_1\ \mathrm{AND}\ x_2\ \mathrm{is}\ {\sim}c_2\ \mathrm{AND}\ \ldots\ \mathrm{THEN}\ \mathrm{class}=k$",
 "gauss":    r"$\mu(x)=e^{-\,(x-c)^2\,/\,2\sigma^{2}}$",
 "firing":   r"$f=\left(\mu_1\cdot\mu_2\cdots\mu_{177}\right)^{1/177}$",
 "classmem": r"$\mu_k=\max\ (f\ \mathrm{of\ the\ rules\ of\ class}\ k)$",
 "fcm_c":    r"$c=\dfrac{\sum u^{2}\,x}{\sum u^{2}}$",
 "centroid": r"$\rho=\dfrac{\sum y\cdot\mu(y)}{\sum \mu(y)}$",
}

for name, tex in EQ.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=18)
    fig.savefig(f"docs/eq/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
print(len(EQ), "formulas")
