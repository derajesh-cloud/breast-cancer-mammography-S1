# Re-draws Fig. 5 from results/C2_imbalance.csv with exactly the plotting code in run_experiments.py
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
H=os.path.dirname(os.path.abspath(__file__)); s2=pd.read_csv(f"{H}/../results/C2_imbalance.csv", keep_default_na=False)
prev=260/11183; cols=["#1f77b4","#2ca02c","#ff7f0e"]
order_s=["None","Class weighting","Random oversampling","SMOTE","Borderline-SMOTE","ADASYN"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
for j, lname in enumerate(["Logistic regression","Random forest","XGBoost"]):
    sub = s2[s2.learner == lname].set_index("strategy").loc[order_s]; xx = np.arange(len(order_s)) + (j - 1) * 0.27
    axs[0].bar(xx, sub.sensitivity, width=0.27, label=lname, color=cols[j]); axs[1].bar(xx, sub.pr_auc, width=0.27, label=lname, color=cols[j])
for ax, yl in zip(axs, ["Sensitivity at p ≥ 0.5", "PR-AUC (average precision)"]):
    ax.set_xticks(np.arange(len(order_s))); ax.set_xticklabels(order_s, rotation=35, ha="right", fontsize=7); ax.set_ylabel(yl)
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=7, frameon=False)
axs[1].axhline(prev, color="grey", ls=":", lw=0.8); fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{H}/../figs/fig_imbalance.png"); print("ok")
