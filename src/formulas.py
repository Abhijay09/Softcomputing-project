"""Render the report formulas as images (matplotlib mathtext, no LaTeX install needed)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
os.makedirs("docs/eq", exist_ok=True)

EQ = {
 "frame":    r"$I:\{0,\ldots,H-1\}\times\{0,\ldots,W-1\}\ \rightarrow\ \{0,\ldots,255\}^{3}$",
 "dataset":  r"$D=\{(g_i,\,y_i)\}_{i=1}^{N},\qquad g_i\in[0,1]^{28\times 28},\quad y_i\in A,\quad |A|=47$",
 "output":   r"$\left((c_1,\rho_1),\ (c_2,\rho_2),\ \ldots,\ (c_n,\rho_n)\right),\qquad c_k\in A,\quad \rho_k\in[0,100]$",
 "features": r"$x=\varphi(g)\in[0,1]^{177}\qquad(49\ \mathrm{ink\ zones}\ +\ 128\ \mathrm{stroke\ directions})$",
 "rule":     r"$R_{k,r}:\ \ \mathrm{IF}\ x_1\ \mathrm{is}\ {\sim}c_1\ \mathrm{AND}\ x_2\ \mathrm{is}\ {\sim}c_2\ \mathrm{AND}\ \ldots\ \mathrm{AND}\ x_{177}\ \mathrm{is}\ {\sim}c_{177}\ \ \mathrm{THEN}\ \mathrm{class}=k$",
 "gauss":    r"$\mu_j(x_j)=\exp\!\left(-\dfrac{(x_j-c_j)^2}{2\sigma_j^{2}}\right)$",
 "firing":   r"$f_{k,r}(x)=\left(\prod_{j=1}^{177}\mu_j(x_j)\right)^{1/177}=\exp\!\left(-\dfrac{1}{177}\sum_{j=1}^{177}\dfrac{(x_j-c_j)^2}{2\sigma_j^{2}}\right)$",
 "classmem": r"$\mu_k(x)=\max_{r}\ f_{k,r}(x),\qquad \hat{c}=\mathrm{arg\,max}_{k}\ \mu_k(x)$",
 "strength": r"$s=\mu_{\hat{c}}(x),\qquad \gamma=\mu_{\hat{c}}(x)-\mu_{\mathrm{second}}(x)$",
 "fcm_c":    r"$c_k=\dfrac{\sum_{i}u_{ik}^{2}\,x_i}{\sum_{i}u_{ik}^{2}}$",
 "fcm_u":    r"$u_{ik}=\dfrac{1}{\sum_{l}\left(\dfrac{d_{ik}}{d_{il}}\right)^{2}},\qquad \sum_{k}u_{ik}=1$",
 "centroid": r"$\rho=\dfrac{\int y\,\mu_{agg}(y)\,dy}{\int \mu_{agg}(y)\,dy}$",
 "accuracy": r"$\mathrm{Acc}=\dfrac{1}{N}\sum_{i=1}^{N}\left[\ \hat{c}(x_i)=y_i\ \right]$",
}

for name, tex in EQ.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=20)
    fig.savefig(f"docs/eq/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
print(len(EQ), "formulas")
