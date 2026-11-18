"""
Semi-synthetic evaluation of context-aware expected-price models for TBML screening.

Everything here is SIMULATED. Legitimate trade flows are generated from a known price
process (commodity, route, volume, season, noise). Over-/under-invoicing is then injected
into a random subset of test flows. Methods are compared on how well they rank / flag the
injected flows. Results show detection capability under the simulation assumptions only.

Requires: numpy, pandas, scikit-learn, matplotlib.
Run:  python tbml_sim.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

ALPHA = 0.02            # alert budget: share of legitimate flows allowed to be flagged
PREVALENCE = 0.01       # share of test flows with injected distortion
MAGNITUDES = [0.0, 0.10, 0.25, 0.50, 1.00]   # 0.0 = correctly priced laundering (blind spot)
N_COMM, N_ORIG, N_DEST = 30, 15, 15
N_TRAIN, N_CAL, N_TEST = 30000, 10000, 20000
N_SEEDS = 10


# ----------------------------------------------------------------------------- data
def make_world(rng):
    heterogeneous = rng.random(N_COMM) < 0.5
    return dict(
        mu=rng.normal(np.log(8), 1.0, N_COMM),
        sigma=np.where(heterogeneous, rng.uniform(0.25, 0.45, N_COMM), rng.uniform(0.06, 0.12, N_COMM)),
        heterogeneous=heterogeneous,
        gamma=rng.normal(-0.05, 0.02, N_COMM),          # bulk discount slope
        small_premium=rng.uniform(0.05, 0.25, N_COMM),  # premium for small lots (non-linear)
        freight=rng.uniform(0.02, 0.35, N_COMM),        # freight sensitivity
        season=rng.uniform(0.0, 0.12, N_COMM),
        phase=rng.uniform(0, 2 * np.pi, N_COMM),
        theta=rng.normal(0, 0.08, N_ORIG),
        dist=rng.uniform(0.1, 1.0, (N_ORIG, N_DEST)),
        weights=rng.dirichlet(np.ones(N_COMM) * 0.8),
    )


def make_flows(rng, w, n):
    c = rng.choice(N_COMM, n, p=w["weights"])
    o, d = rng.integers(0, N_ORIG, n), rng.integers(0, N_DEST, n)
    month = rng.integers(1, 13, n)
    lv = np.clip(rng.normal(9, 1.5, n), 4, 14)
    dist = w["dist"][o, d]
    logp = (w["mu"][c]
            + w["gamma"][c] * (lv - 9)
            + w["small_premium"][c] * np.maximum(0, 7 - lv)
            + w["freight"][c] * dist
            + w["season"][c] * np.sin(2 * np.pi * month / 12 + w["phase"][c])
            + w["theta"][o]
            + rng.normal(0, w["sigma"][c]))
    return pd.DataFrame(dict(comm=c, orig=o, dest=d, month=month, lv=lv,
                             dist_obs=dist + rng.normal(0, 0.05, n), logp=logp,
                             label=0, het=w["heterogeneous"][c]))


def inject(rng, df, prevalence, magnitudes):
    """Over-/under-invoice a random subset; magnitude may be a list (drawn at random) or a float."""
    df = df.copy()
    idx = rng.random(len(df)) < prevalence
    m = rng.choice(magnitudes, idx.sum()) if np.ndim(magnitudes) else np.full(idx.sum(), magnitudes)
    sign = rng.choice([-1, 1], idx.sum())
    df.loc[idx, "logp"] += sign * np.log1p(m)
    df.loc[idx, "label"] = 1
    return df


# ----------------------------------------------------------------------------- methods
def robust_scale(resid, comm):
    """Per-commodity robust spread (MAD) of residuals on legitimate-dominated calibration data."""
    g = pd.Series(resid).groupby(comm)
    s = g.apply(lambda r: 1.4826 * np.median(np.abs(r - np.median(r))))
    n = g.size()
    pooled = s[n >= 30].median()                       # fall back to the pooled scale for rare commodities
    s = s.where(n >= 30, pooled).reindex(range(N_COMM)).fillna(pooled)
    return np.maximum(s.values, 1e-3)


def run(seed, drop_dist=False, train_contam=0.01):
    rng = np.random.default_rng(seed)
    w = make_world(rng)
    train = inject(rng, make_flows(rng, w, N_TRAIN), train_contam, [0.1, 0.25, 0.5, 1.0])  # unlabeled contamination
    cal = inject(rng, make_flows(rng, w, N_CAL), train_contam, [0.1, 0.25, 0.5, 1.0])
    test_base = make_flows(rng, w, N_TEST)

    feats_num = ["lv", "month"] + ([] if drop_dist else ["dist_obs"])
    feats_cat = ["comm", "orig", "dest"]

    # B1: conventional price filter around the commodity-wide average (no context)
    med = train.groupby("comm").logp.median()
    mean_price = train.assign(p=np.exp(train.logp)).groupby("comm").p.mean()
    def b1_score(df):                      # fixed-band logic expressed as a ranking score
        return np.abs(df.logp.values - med.reindex(df.comm).values)
    mad = robust_scale(train.logp.values - med.reindex(train.comm).values, train.comm.values)
    def b1z_score(df):                     # per-commodity robust z-score
        return b1_score(df) / mad[df.comm.values]

    # B2: log-linear regression on context (gravity-style interpretable benchmark)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[feats_cat + ["month"]])
    def lin_X(df):
        X = np.hstack([enc.transform(df[feats_cat + ["month"]]), df[[c for c in feats_num if c != "month"]].values])
        return X
    lin = Ridge(alpha=1.0).fit(lin_X(train), train.logp)

    # M: gradient-boosted trees on the same context
    cols = feats_cat + feats_num
    cat_mask = [c in feats_cat for c in cols]
    gbt = HistGradientBoostingRegressor(categorical_features=cat_mask, max_iter=300, learning_rate=0.08,
                                        max_leaf_nodes=31, random_state=seed).fit(train[cols], train.logp)

    # U: Isolation Forest on commodity-relative price + volume + freight proxy (no price target)
    def if_X(df):
        X = np.c_[df.logp.values - med.reindex(df.comm).values, df.lv.values]
        return X if drop_dist else np.c_[X, df.dist_obs.values]
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(if_X(train))

    # residual scales per commodity, from the calibration set
    s_lin = robust_scale(cal.logp.values - lin.predict(lin_X(cal)), cal.comm.values)
    s_gbt = robust_scale(cal.logp.values - gbt.predict(cal[cols]), cal.comm.values)

    scorers = {
        "B1 commodity band": b1_score,
        "B1z commodity z-score": b1z_score,
        "B2 log-linear context": lambda d: np.abs(d.logp.values - lin.predict(lin_X(d))) / s_lin[d.comm.values],
        "M gradient boosting": lambda d: np.abs(d.logp.values - gbt.predict(d[cols])) / s_gbt[d.comm.values],
        "U isolation forest": lambda d: -iso.score_samples(if_X(d)),
    }
    thr = {k: np.quantile(f(cal), 1 - ALPHA) for k, f in scorers.items()}   # conformal-style calibration

    rows = []
    for m in MAGNITUDES:
        test = inject(np.random.default_rng(seed * 1000 + int(m * 100)), test_base, PREVALENCE, m)
        y = test.label.values
        for k, f in scorers.items():
            s = f(test)
            flag = s > thr[k]
            rows.append(dict(seed=seed, magnitude=m, method=k,
                             ap=average_precision_score(y, s),
                             recall=flag[y == 1].mean(), fpr=flag[y == 0].mean(),
                             precision=y[flag].mean() if flag.any() else np.nan,
                             recall_het=flag[(y == 1) & test.het.values].mean(),
                             recall_std=flag[(y == 1) & ~test.het.values].mean()))
        # the conventional fixed +-50 % rule around the commodity mean price
        p, ref = np.exp(test.logp.values), mean_price.reindex(test.comm).values
        flag = np.abs(p / ref - 1) > 0.5
        rows.append(dict(seed=seed, magnitude=m, method="R50 fixed +-50% rule", ap=np.nan,
                         recall=flag[y == 1].mean(), fpr=flag[y == 0].mean(),
                         precision=y[flag].mean() if flag.any() else np.nan,
                         recall_het=flag[(y == 1) & test.het.values].mean(),
                         recall_std=flag[(y == 1) & ~test.het.values].mean()))
    return rows


def summarise(rows):
    df = pd.DataFrame(rows)
    g = df.groupby(["magnitude", "method"]).agg(["mean", "std"])
    return df, g


if __name__ == "__main__":
    out = {}
    for name, kw in {"main": {}, "no_freight_proxy": {"drop_dist": True}, "contam5": {"train_contam": 0.05}}.items():
        rows = [r for s in range(N_SEEDS) for r in run(s, **kw)]
        df, _ = summarise(rows)
        df.to_csv(f"results_{name}.csv", index=False)
        out[name] = df.groupby(["magnitude", "method"])[["ap", "recall", "fpr", "precision", "recall_het", "recall_std"]].mean().round(3).reset_index().to_dict("records")
        print(name, "done")
    json.dump(out, open("results_summary.json", "w"))
