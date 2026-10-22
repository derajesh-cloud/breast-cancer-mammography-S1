"""
Reproducible experiments for:
"Interpretable machine learning for breast cancer diagnosis from structured mammography
 report descriptors and image-derived features: a three-cohort public-data study"

All data are public and loaded from local copies (see README / paper, Data Availability):
  C1  Mammographic Mass (Elter et al. 2007; UCI id 161; KEEL complete-case copy, n=830)
  C2  Mammography microcalcification features (Woods et al. 1993; n=11,183, 260 positive)
  C3  Wisconsin Diagnostic Breast Cancer, WDBC (Street et al. 1993; bundled in scikit-learn, n=569)

Run:  python run_experiments.py   (CPU only; ~30-60 min on 2 cores)
Outputs: ../results/*.json, ../results/*.csv, ../figs/*.png
"""
import json, time, warnings, os, sys, platform
import numpy as np, pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, roc_curve, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
QUICK = os.environ.get("QUICK") == "1"   # smoke-test mode only
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figs")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})

# ----------------------------------------------------------------------------- data
def load_c1():
    d = pd.read_csv(os.path.join(DATA, "mammographic.dat"), header=None,
                    names=["birads", "age", "shape", "margin", "density", "severity"])
    n0 = len(d)
    d.loc[d.birads == 55, "birads"] = 5          # documented data-entry error in UCI source
    excl0 = int((d.birads == 0).sum()); excl6 = int((d.birads == 6).sum())
    d = d[(d.birads >= 1) & (d.birads <= 5)].reset_index(drop=True)  # 0 = incomplete, 6 = known cancer
    info = dict(n_source=n0, recoded_55_to_5=1, excluded_birads0=excl0, excluded_birads6=excl6,
                n_analysis=len(d), n_malignant=int(d.severity.sum()), n_benign=int((1 - d.severity).sum()),
                age_mean=float(d.age.mean()), age_sd=float(d.age.std()))
    return d, info

def load_c2():
    w = pd.read_csv(os.path.join(DATA, "woods_mammography.csv"), header=None)
    X = w.iloc[:, :6].values.astype(float)
    y = (w.iloc[:, 6].astype(str).str.strip("'") == "1").astype(int).values
    return X, y, [f"f{i+1}" for i in range(6)]

def load_c3():
    b = load_breast_cancer()
    return b.data, (1 - b.target).astype(int), list(b.feature_names)   # 1 = malignant

# ----------------------------------------------------------------------------- stats
def delong_roc_variance(y, p):
    """Fast DeLong (Sun & Xu 2014). Returns auc, var."""
    return _delong(y, p[None, :])

def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def _delong(y, preds):
    order = np.argsort(-y, kind="mergesort"); y = y[order]; preds = preds[:, order]
    m = int(y.sum()); n = len(y) - m; k = preds.shape[0]
    tx = np.array([_compute_midrank(p[:m]) for p in preds])
    ty = np.array([_compute_midrank(p[m:]) for p in preds])
    tz = np.array([_compute_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n

def delong_test(y, p1, p2):
    aucs, cov = _delong(np.asarray(y), np.vstack([p1, p2]))
    l = np.array([1, -1]); z = (aucs[0] - aucs[1]) / np.sqrt(l @ cov @ l)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))

def auc_ci(y, p):
    aucs, cov = _delong(np.asarray(y), np.asarray(p)[None, :])
    se = np.sqrt(cov[0, 0]); a = aucs[0]
    return float(a), float(max(0, a - 1.96 * se)), float(min(1, a + 1.96 * se))

def boot_ci(fn, y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def calib_slope_intercept(y, p):
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y)
    slope = float(m.coef_[0, 0])
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model), via Newton steps
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp))); g = np.sum(y - q); h = np.sum(q * (1 - q)); a += g / h
    return slope, float(a)

def threshold_metrics(y, p, t=0.5):
    yp = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan; npv = tn / (tn + fn) if tn + fn else np.nan
    return dict(sensitivity=float(sens), specificity=float(spec), ppv=float(ppv), npv=float(npv),
                accuracy=float((tp + tn) / len(y)), f1=float(f1_score(y, yp, zero_division=0)),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))

def sens_at_spec(y, p, spec=0.95):
    fpr, tpr, _ = roc_curve(y, p); ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0

def net_benefit(y, p, ts):
    n = len(y); out = []
    for t in ts:
        yp = p >= t; tp = np.sum(yp & (y == 1)); fp = np.sum(yp & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)

# ----------------------------------------------------------------------------- models
def model_zoo(pre):
    """Returns {name: (pipeline, param_grid)}; preprocessing `pre` is fitted inside each fold."""
    xgb = XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        random_state=SEED, n_jobs=1, verbosity=0)
    stack = StackingClassifier(
        estimators=[("lr", LogisticRegression(C=1.0, max_iter=2000)),
                    ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=1)),
                    ("xgb", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
                                          n_jobs=1, verbosity=0))],
        final_estimator=LogisticRegression(max_iter=2000), cv=5, stack_method="predict_proba", n_jobs=1)
    return {
        "Logistic regression": (Pipeline([("pre", clone(pre)), ("clf", LogisticRegression(max_iter=2000))]),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "SVM (RBF)": (Pipeline([("pre", clone(pre)), ("clf", SVC(probability=True, random_state=SEED))]),
                      {"clf__C": [0.1, 1, 10]}),
        "Random forest": (Pipeline([("pre", clone(pre)), ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1))]),
                          {"clf__min_samples_leaf": [1, 5, 10], "clf__max_features": ["sqrt", 0.5]}),
        "XGBoost": (Pipeline([("pre", clone(pre)), ("clf", xgb)]),
                    {"clf__max_depth": [2, 4], "clf__learning_rate": [0.05, 0.1]}),
        "MLP": (Pipeline([("pre", clone(pre)), ("clf", MLPClassifier(max_iter=2000, early_stopping=True, random_state=SEED))]),
                {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4, 1e-2]}),
        "Stacked ensemble (LR+RF+XGB)": (Pipeline([("pre", clone(pre)), ("clf", stack)]), None),
    }

def nested_cv(X, y, zoo, n_splits=10, n_repeats=5, inner=3, tag=""):
    if QUICK: n_splits, n_repeats = 3, 1
    """Outer repeated stratified K-fold; inner GridSearchCV (AUC). Returns OOF probabilities
    with shape (n_repeats, n_samples) per model, plus per-repeat AUCs and timing."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    out = {}
    for name, (pipe, grid) in zoo.items():
        t0 = time.time(); oof = np.full((n_repeats, len(y)), np.nan); chosen = []
        for k, (tr, te) in enumerate(rskf.split(X, y)):
            r = k // n_splits
            if grid:
                gs = GridSearchCV(pipe, grid, scoring="roc_auc", n_jobs=2,
                                  cv=StratifiedKFold(inner, shuffle=True, random_state=SEED + k))
                gs.fit(X[tr], y[tr]); est = gs.best_estimator_; chosen.append(gs.best_params_)
            else:
                est = clone(pipe).fit(X[tr], y[tr])
            oof[r, te] = est.predict_proba(X[te])[:, 1]
        rep_auc = [roc_auc_score(y, oof[r]) for r in range(n_repeats)]
        out[name] = dict(oof=oof, rep_auc=rep_auc, secs=time.time() - t0, params=chosen)
        print(f"  [{tag}] {name:30s} AUC {np.mean(rep_auc):.3f} ± {np.std(rep_auc):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return out

def summarise(y, res, ref="Logistic regression", t=0.5):
    rows = []
    pref = res[ref]["oof"].mean(0)
    for name, r in res.items():
        p = r["oof"].mean(0)                                        # average OOF prob across repeats
        a, lo, hi = auc_ci(y, p)
        pr = average_precision_score(y, p); prlo, prhi = boot_ci(average_precision_score, y, p)
        tm = threshold_metrics(y, p, t); slope, cil = calib_slope_intercept(y, p)
        _, _, pval = delong_test(y, p, pref) if name != ref else (0, 0, np.nan)
        rows.append(dict(model=name, auc_mean_repeats=float(np.mean(r["rep_auc"])),
                         auc_sd_repeats=float(np.std(r["rep_auc"], ddof=1)),
                         auc=a, auc_lo=lo, auc_hi=hi, pr_auc=float(pr), pr_lo=prlo, pr_hi=prhi,
                         brier=float(brier_score_loss(y, p)), calib_slope=slope, calib_intercept=cil,
                         sens_at_95spec=sens_at_spec(y, p, 0.95), delong_p_vs_LR=pval,
                         train_eval_secs=r["secs"], **tm))
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- main
def main():
    T0 = time.time(); results = {"env": dict(python=sys.version.split()[0], platform=platform.platform(),
               numpy=np.__version__, pandas=pd.__version__, sklearn=__import__("sklearn").__version__,
               xgboost=__import__("xgboost").__version__, imblearn=__import__("imblearn").__version__,
               shap=shap.__version__, seed=SEED)}

    # ======================= Cohort 1: structured mammography report (BI-RADS) ==============
    d1, info1 = load_c1(); results["C1_info"] = info1; print("C1", info1)
    y1 = d1.severity.values.astype(int)
    pre_desc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [0, 1]),   # shape, margin
                                  ("num", StandardScaler(), [2, 3])])                        # density, age
    pre_all = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [0, 1]),
                                 ("num", StandardScaler(), [2, 3, 4])])                      # + BI-RADS
    X1d = d1[["shape", "margin", "density", "age"]].values.astype(float)
    X1a = d1[["shape", "margin", "density", "age", "birads"]].values.astype(float)
    print("C1 descriptors + age"); r1d = nested_cv(X1d, y1, model_zoo(pre_desc), tag="C1-desc")
    print("C1 descriptors + age + BI-RADS"); r1a = nested_cv(X1a, y1, model_zoo(pre_all), tag="C1-all")
    s1d = summarise(y1, r1d); s1a = summarise(y1, r1a)
    # radiologist baseline: raw BI-RADS category as ordinal score (no training)
    a, lo, hi = auc_ci(y1, d1.birads.values.astype(float))
    base = dict(auc=a, auc_lo=lo, auc_hi=hi, **threshold_metrics(y1, (d1.birads.values >= 5).astype(float), 0.5),
                note="BI-RADS >=5 called malignant; BI-RADS 4 called benign at this cut")
    base4 = threshold_metrics(y1, (d1.birads.values >= 4).astype(float), 0.5)
    results["C1_birads_baseline"] = dict(cut5=base, cut4=base4)
    best = "Stacked ensemble (LR+RF+XGB)"
    pa = r1a[best]["oof"].mean(0); pd_ = r1d[best]["oof"].mean(0)
    results["C1_delong_all_vs_desc"] = delong_test(y1, pa, pd_)
    results["C1_delong_all_vs_biradsraw"] = delong_test(y1, pa, d1.birads.values.astype(float))
    results["C1_delong_desc_vs_biradsraw"] = delong_test(y1, pd_, d1.birads.values.astype(float))
    s1d.to_csv(os.path.join(RES, "C1_descriptors_age.csv"), index=False)
    s1a.to_csv(os.path.join(RES, "C1_descriptors_age_birads.csv"), index=False)
    results["C1_desc"] = s1d.to_dict("records"); results["C1_all"] = s1a.to_dict("records")

    # ======================= Cohort 3: WDBC image-derived cytology morphometry =================
    X3, y3, f3 = load_c3(); results["C3_info"] = dict(n=len(y3), n_malignant=int(y3.sum()), n_benign=int((1-y3).sum()), n_features=X3.shape[1])
    pre3 = StandardScaler(); print("C3 WDBC")
    r3 = nested_cv(X3, y3, model_zoo(pre3), tag="C3")
    s3 = summarise(y3, r3); s3.to_csv(os.path.join(RES, "C3_wdbc.csv"), index=False); results["C3"] = s3.to_dict("records")

    # ======================= Cohort 2: mammogram microcalcification features + imbalance =========
    X2, y2, f2 = load_c2(); results["C2_info"] = dict(n=len(y2), n_pos=int(y2.sum()), n_neg=int((1-y2).sum()),
                                                      prevalence=float(y2.mean()))
    print("C2 imbalance study")
    learners = {"Logistic regression": lambda cw: LogisticRegression(max_iter=2000, class_weight=cw),
                "Random forest": lambda cw: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight=cw, random_state=SEED, n_jobs=2),
                "XGBoost": lambda cw: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                                    colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=2, verbosity=0,
                                                    scale_pos_weight=(y2 == 0).sum() / (y2 == 1).sum() if cw else 1)}
    samplers = {"None": None, "Class weighting": "cw", "Random oversampling": RandomOverSampler(random_state=SEED),
                "SMOTE": SMOTE(random_state=SEED), "Borderline-SMOTE": BorderlineSMOTE(random_state=SEED),
                "ADASYN": ADASYN(random_state=SEED)}
    R2 = 1 if QUICK else 3
    rskf2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=R2, random_state=SEED)
    rows2 = []; oof2 = {}
    for lname, mk in learners.items():
        for sname, smp in samplers.items():
            t0 = time.time(); oof = np.full((R2, len(y2)), np.nan)
            for k, (tr, te) in enumerate(rskf2.split(X2, y2)):
                cw = "balanced" if smp == "cw" else None
                steps = [("sc", StandardScaler())]
                if smp not in (None, "cw"): steps.append(("smp", clone(smp)))
                steps.append(("clf", mk(cw)))
                m = ImbPipeline(steps).fit(X2[tr], y2[tr]); oof[k // 5, te] = m.predict_proba(X2[te])[:, 1]
            p = oof.mean(0); a, lo, hi = auc_ci(y2, p); tm = threshold_metrics(y2, p, 0.5)
            rows2.append(dict(learner=lname, strategy=sname, auc=a, auc_lo=lo, auc_hi=hi,
                              pr_auc=float(average_precision_score(y2, p)),
                              sens_at_95spec=sens_at_spec(y2, p, 0.95), brier=float(brier_score_loss(y2, p)),
                              secs=time.time() - t0, **tm))
            oof2[(lname, sname)] = p
            print(f"  [C2] {lname:20s} {sname:20s} AUC {a:.3f} PR {rows2[-1]['pr_auc']:.3f} sens@.5 {tm['sensitivity']:.3f} spec {tm['specificity']:.3f}", flush=True)
    s2 = pd.DataFrame(rows2); s2.to_csv(os.path.join(RES, "C2_imbalance.csv"), index=False); results["C2"] = s2.to_dict("records")
    # leakage demonstration: SMOTE applied BEFORE splitting (incorrect) vs inside folds (correct)
    Xs, ys = SMOTE(random_state=SEED).fit_resample(StandardScaler().fit_transform(X2), y2)
    leak = []
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs, ys)):
        m = learners["XGBoost"](None).fit(Xs[tr], ys[tr]); pp = m.predict_proba(Xs[te])[:, 1]
        leak.append(dict(auc=roc_auc_score(ys[te], pp), pr=average_precision_score(ys[te], pp),
                         **threshold_metrics(ys[te], pp)))
    results["C2_leakage_SMOTE_before_split"] = pd.DataFrame(leak).mean().to_dict()
    results["C2_n_after_global_smote"] = int(len(ys))

    # ======================= Explainability (SHAP) =================================
    shap_out = {}
    for tag, X, y, names, prep in [("C1", X1a, y1, None, pre_all), ("C3", X3, y3, f3, pre3)]:
        pp = clone(prep).fit(X); Xt = pp.transform(X); Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
        if tag == "C1":
            names = ([f"shape={s}" for s in ["round", "oval", "lobular", "irregular"]] +
                     [f"margin={m}" for m in ["circumscribed", "microlobulated", "obscured", "ill-defined", "spiculated"]] +
                     ["density", "age", "BI-RADS"])
        m = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=SEED, n_jobs=2, verbosity=0).fit(Xt, y)
        sv = shap.TreeExplainer(m).shap_values(Xt); imp = np.abs(sv).mean(0)
        order = np.argsort(-imp)
        shap_out[tag] = [(names[i], float(imp[i])) for i in order]
        top = order[:10][::-1]
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        ax.barh([names[i] for i in top], imp[top], color="#3b6ea5")
        ax.set_xlabel("mean |SHAP value| (log-odds)")
        ax.set_title({"C1": "Cohort 1: report descriptors", "C3": "Cohort 3: WDBC morphometry"}[tag], fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_shap_{tag}.png")); plt.close(fig)
    results["shap"] = shap_out

    # ======================= Inference latency (deployment check) ==================
    lat = {}
    for tag, X, y, prep in [("C1", X1a, y1, pre_all), ("C3", X3, y3, pre3)]:
        est = clone(model_zoo(prep)[best][0]).fit(X, y)
        ts = []
        for i in range(200):
            x = X[i % len(X)][None, :]; t0 = time.perf_counter(); est.predict_proba(x); ts.append(time.perf_counter() - t0)
        lat[tag] = dict(median_ms=float(np.median(ts) * 1e3), p95_ms=float(np.percentile(ts, 95) * 1e3))
    results["latency_single_case"] = lat

    # ======================= Figures ===============================================
    cols = {"Logistic regression": "#1f77b4", "SVM (RBF)": "#9467bd", "Random forest": "#2ca02c",
            "XGBoost": "#ff7f0e", "MLP": "#8c564b", "Stacked ensemble (LR+RF+XGB)": "#d62728"}
    # ROC panels
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, (title, y, res, extra) in zip(axs, [("Cohort 1: descriptors + age", y1, r1d, None),
                                               ("Cohort 1: + BI-RADS assessment", y1, r1a, d1.birads.values),
                                               ("Cohort 3: WDBC", y3, r3, None)]):
        for name, r in res.items():
            p = r["oof"].mean(0); f, t, _ = roc_curve(y, p)
            ax.plot(f, t, lw=1.1, color=cols[name], label=f"{name.split(' (')[0]} {roc_auc_score(y, p):.3f}")
        if extra is not None:
            f, t, _ = roc_curve(y, extra); ax.plot(f, t, "k--", lw=1, label=f"BI-RADS alone {roc_auc_score(y, extra):.3f}")
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8); ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.legend(fontsize=5.6, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_roc.png")); plt.close(fig)
    # calibration
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.6))
    for ax, (title, y, res) in zip(axs, [("Cohort 1 (+ BI-RADS)", y1, r1a), ("Cohort 3: WDBC", y3, r3)]):
        for name in ["Logistic regression", "XGBoost", best]:
            p = res[name]["oof"].mean(0); fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(mp, fr, "o-", ms=2.5, lw=1, color=cols[name], label=name.split(" (")[0])
        ax.plot([0, 1], [0, 1], ":", color="grey"); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
        ax.set_title(title, fontsize=8.5); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_calibration.png")); plt.close(fig)
    # decision curve C1
    ts = np.linspace(0.02, 0.60, 59); prev = y1.mean()
    nb_all = prev - (1 - prev) * ts / (1 - ts)
    nb_ens = net_benefit(y1, pa, ts); nb_desc = net_benefit(y1, pd_, ts)
    nb_b4 = net_benefit(y1, (d1.birads.values >= 4).astype(float), ts)
    nb_b5 = net_benefit(y1, (d1.birads.values >= 5).astype(float), ts)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(ts, nb_ens, color="#d62728", lw=1.3, label="Ensemble: descriptors+age+BI-RADS")
    ax.plot(ts, nb_desc, color="#1f77b4", lw=1.3, label="Ensemble: descriptors+age")
    ax.plot(ts, nb_b4, color="k", lw=1, ls="--", label="Biopsy if BI-RADS ≥4")
    ax.plot(ts, nb_b5, color="grey", lw=1, ls="-.", label="Biopsy if BI-RADS 5")
    ax.plot(ts, nb_all, color="#999", lw=0.8, ls=":", label="Biopsy all"); ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-0.05, prev + 0.03); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.legend(fontsize=5.8, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_dca.png")); plt.close(fig)
    results["C1_dca"] = {f"{t:.2f}": dict(ens_all=float(a), ens_desc=float(b), birads4=float(c), birads5=float(e), treat_all=float(f_))
                         for t, a, b, c, e, f_ in zip(ts, nb_ens, nb_desc, nb_b4, nb_b5, nb_all)}
    # imbalance figure
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    order_s = list(samplers.keys())
    for j, lname in enumerate(learners):
        sub = s2[s2.learner == lname].set_index("strategy").loc[order_s]
        xx = np.arange(len(order_s)) + (j - 1) * 0.27
        axs[0].bar(xx, sub.sensitivity, width=0.27, label=lname, color=list(cols.values())[[0, 2, 3][j]])
        axs[1].bar(xx, sub.pr_auc, width=0.27, label=lname, color=list(cols.values())[[0, 2, 3][j]])
    for ax, yl in zip(axs, ["Sensitivity at p ≥ 0.5", "PR-AUC (average precision)"]):
        ax.set_xticks(np.arange(len(order_s))); ax.set_xticklabels(order_s, rotation=35, ha="right", fontsize=7); ax.set_ylabel(yl)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=7, frameon=False)
    axs[1].axhline(y2.mean(), color="grey", ls=":", lw=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIG, "fig_imbalance.png")); plt.close(fig)

    results["runtime_minutes"] = (time.time() - T0) / 60
    with open(os.path.join(RES, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done in %.1f min" % results["runtime_minutes"])

if __name__ == "__main__":
    main()
