#!/usr/bin/env python3
"""
FICD benchmark v0.3
===================

Formula-Invariant Chemical Diversification, experimental benchmark.

Core change from v0.2
---------------------
v0.2 still allowed the diversification stage to trade too much ranking quality
for chemical distance.

v0.3 therefore makes quality robustness a HARD constraint.

Stage 1 — Cross-family scoring fingerprint
------------------------------------------
Four scalarization families are used:
  - arithmetic
  - geometric
  - harmonic
  - Chebyshev

Each family is evaluated at four stability/property preference settings.

For each material:
  C   = geometric rank consensus
  V   = scoring-formula sensitivity
  SII = C * exp(-V)

A candidate is "multi-family eligible" if it appears in the Top-K list of
at least two distinct scalarization families. This means it is not supported
only by one scoring geometry.

Family support:
  F_i in {0,1,2,3,4}

Quality score:
  Q_i = sqrt(SII_i * F_i/4)

Stage 2 — Regret-preserving diversification
-------------------------------------------
1. Start from the Top-K candidates by Q inside the eligible pool.
2. Measure the initial shortlist's WORST scoring-rule regret R0.
3. Search one-for-one swaps with other eligible candidates.
4. A swap is allowed only if:
       worst_rule_regret(new shortlist) <= R0
5. Among allowed swaps, accept the one that gives the largest increase in
   mean nearest-neighbor chemical distance d_NN.
6. Repeat until no admissible swap improves d_NN.

Therefore diversity is NOT allowed to worsen the cross-formula quality
robustness of the initial formula-consistent shortlist.

The algorithm contains no fitted quality/diversity mixing coefficient.

The script also runs family-support thresholds 1,2,3,4 as a sensitivity
analysis so the default 2-family eligibility is not evaluated in isolation.
"""

from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
FAMILIES = ("arith", "geom", "harm", "cheb")


def parse_formula(formula):
    comp = {}
    for el, num in FORMULA_RE.findall(str(formula).replace(" ", "")):
        comp[el] = comp.get(el, 0.0) + (float(num) if num else 1.0)
    return comp


def minmax(x, higher_is_better=True):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    z = np.ones_like(x) if hi <= lo else (x - lo) / (hi - lo)
    return z if higher_is_better else 1.0 - z


def prepare_dataset(name):
    if name == "MP":
        df = pd.read_csv(DATA / "mp_transition_candidates.csv")
        if "is_metal" in df.columns:
            df = df[(df["is_metal"].astype(bool) == False) & (df["band_gap"] > 0)].copy()
        else:
            df = df[df["band_gap"] > 0].copy()
    elif name == "WBM":
        df = pd.read_csv(DATA / "wbm_transition_candidates.csv")
        if "is_nonmetal_proxy" in df.columns:
            df = df[
                (df["is_nonmetal_proxy"].astype(bool) == True)
                & (df["band_gap"] > 0)
            ].copy()
        else:
            df = df[df["band_gap"] > 0].copy()
    else:
        raise ValueError(name)

    df = df.reset_index(drop=True)
    df["S"] = minmax(df["energy_above_hull"].to_numpy(float), False)
    df["G"] = minmax(df["band_gap"].to_numpy(float), True)
    return df


def cation_fraction_matrix(formulas):
    comps = [parse_formula(x) for x in formulas]
    elems = sorted({e for c in comps for e in c if e != "O"})
    X = np.zeros((len(comps), len(elems)), dtype=float)
    for i, c in enumerate(comps):
        total = sum(v for e, v in c.items() if e != "O")
        if total > 0:
            for j, e in enumerate(elems):
                X[i, j] = c.get(e, 0.0) / total
    return X, elems


def chemical_distance_matrix(df):
    X, elems = cation_fraction_matrix(df["formula"])
    # Oxygen-excluded cation-fraction total-variation distance, in [0,1].
    D = 0.5 * np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
    return X, D, elems


def build_scoring_rules(df):
    S = np.clip(df["S"].to_numpy(float), 1e-6, 1.0)
    G = np.clip(df["G"].to_numpy(float), 1e-6, 1.0)
    rules = {}

    for ws in (0.4, 0.5, 0.6, 0.7):
        wg = 1.0 - ws

        rules[f"arith_ws{ws:.1f}"] = ws * S + wg * G
        rules[f"geom_ws{ws:.1f}"] = (S ** ws) * (G ** wg)
        rules[f"harm_ws{ws:.1f}"] = 1.0 / (ws / S + wg / G)

        loss = np.maximum(ws * (1.0 - S), wg * (1.0 - G))
        rules[f"cheb_ws{ws:.1f}"] = 1.0 - loss

    return rules


def scoring_fingerprint(rules):
    names = list(rules.keys())
    n = len(next(iter(rules.values())))
    R = np.zeros((n, len(names)), dtype=float)

    for j, name in enumerate(names):
        ranks = rankdata(-np.asarray(rules[name]), method="average")
        if n == 1:
            R[:, j] = 1.0
        else:
            R[:, j] = np.maximum(1e-6, 1.0 - (ranks - 1.0) / n)

    logR = np.log(R)
    C = np.exp(logR.mean(axis=1))
    V = logR.std(axis=1)
    SII = C * np.exp(-V)
    return names, R, C, V, SII


def family_support_statistics(rules, rule_names, K):
    n = len(next(iter(rules.values())))
    family_hits = {f: np.zeros(n, dtype=bool) for f in FAMILIES}
    rule_support = np.zeros(n, dtype=int)

    for rn in rule_names:
        idx = np.argsort(-np.asarray(rules[rn]))[: min(K, n)]
        rule_support[idx] += 1
        family = rn.split("_")[0]
        family_hits[family][idx] = True

    family_support = np.zeros(n, dtype=int)
    for f in FAMILIES:
        family_support += family_hits[f].astype(int)

    return family_support, rule_support, family_hits


def family_reservoir(SII, family_support, min_family_support):
    mask = family_support >= min_family_support
    reservoir = np.where(mask)[0]
    family_fraction = family_support / float(len(FAMILIES))
    Q = np.sqrt(
        np.clip(SII, 1e-12, 1.0)
        * np.clip(family_fraction, 1e-12, 1.0)
    )
    return reservoir, family_fraction, Q


def worst_rule_regret(idx, R):
    idx = np.asarray(idx, dtype=int)
    k = len(idx)
    regrets = []
    for j in range(R.shape[1]):
        best = np.sort(R[:, j])[-k:].mean()
        got = R[idx, j].mean()
        regrets.append(best - got)
    return float(np.max(regrets)), np.asarray(regrets)


def dnn(idx, D):
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 2:
        return np.nan
    M = D[np.ix_(idx, idx)].copy()
    np.fill_diagonal(M, np.inf)
    return float(M.min(axis=1).mean())


def mean_pairwise(idx, D):
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 2:
        return np.nan
    M = D[np.ix_(idx, idx)]
    return float(M[np.triu_indices(len(idx), 1)].mean())


def unique_profiles(idx, D):
    reps = []
    for i in idx:
        if not any(D[i, j] < 1e-12 for j in reps):
            reps.append(i)
    return len(reps)


def quality_seed(reservoir, Q, SII, family_support, K):
    return sorted(
        map(int, reservoir),
        key=lambda i: (Q[i], SII[i], family_support[i], -i),
        reverse=True,
    )[: min(K, len(reservoir))]


def regret_preserving_diversification(
    reservoir, seed, R, D, Q, SII, family_support, max_iterations=100
):
    """
    Best-improvement 1-swap local search.

    Hard quality constraint:
       worst_rule_regret(candidate) <= worst_rule_regret(seed)

    Objective:
       maximize dNN.
    Tie breakers:
       more unique profiles -> higher pairwise diversity -> higher mean Q.
    """
    selected = list(map(int, seed))
    regret_budget, _ = worst_rule_regret(selected, R)

    def objective(a):
        return (
            dnn(a, D),
            unique_profiles(a, D),
            mean_pairwise(a, D),
            float(np.mean(Q[a])),
        )

    current_obj = objective(selected)
    history = [
        {
            "iteration": 0,
            "removed_index": "",
            "added_index": "",
            "dNN": current_obj[0],
            "unique_profiles": current_obj[1],
            "mean_pairwise_distance": current_obj[2],
            "mean_Q": current_obj[3],
            "worst_rule_regret": worst_rule_regret(selected, R)[0],
            "regret_budget": regret_budget,
        }
    ]

    reservoir = list(map(int, reservoir))

    for iteration in range(1, max_iterations + 1):
        outside = [i for i in reservoir if i not in selected]
        best = None

        for pos, rem in enumerate(selected):
            for add in outside:
                cand = selected.copy()
                cand[pos] = int(add)

                reg, _ = worst_rule_regret(cand, R)
                if reg > regret_budget + 1e-12:
                    continue

                obj = objective(cand)
                # Require strict dNN improvement. The rest are tie breakers only.
                if obj[0] <= current_obj[0] + 1e-12:
                    continue

                key = (obj[0], obj[1], obj[2], obj[3], -reg)
                if best is None or key > best[0]:
                    best = (key, cand, int(rem), int(add), reg, obj)

        if best is None:
            break

        _, selected, rem, add, reg, current_obj = best
        history.append(
            {
                "iteration": iteration,
                "removed_index": rem,
                "added_index": add,
                "dNN": current_obj[0],
                "unique_profiles": current_obj[1],
                "mean_pairwise_distance": current_obj[2],
                "mean_Q": current_obj[3],
                "worst_rule_regret": reg,
                "regret_budget": regret_budget,
            }
        )

    return selected, pd.DataFrame(history), regret_budget


def effective_separation(i, selected, D):
    if not selected:
        return 1.0
    ds = np.maximum(D[i, selected], 1e-8)
    return float(np.clip(len(ds) / np.sum(1.0 / ds), 0.0, 1.0))


def harmonic_couple(a, b):
    return 0.0 if a <= 0 or b <= 0 else 2.0 * a * b / (a + b)


def ficd_v01_select(SII, D, K):
    selected = []
    for _ in range(min(K, len(SII))):
        best = None
        for i in range(len(SII)):
            if i in selected:
                continue
            deff = effective_separation(i, selected, D)
            phi = harmonic_couple(float(SII[i]), deff)
            key = (phi, SII[i], -i)
            if best is None or key > best[0]:
                best = (key, i)
        selected.append(best[1])
    return selected


def ficd_v02_select(rules, rule_names, SII, family_support, D, K):
    """
    Reconstruct v0.2 style:
    reservoir = union of Top-K across all rules;
    Q = sqrt(SII * rule_support_fraction);
    harmonic admission.
    """
    n = len(SII)
    rule_support = np.zeros(n, dtype=int)
    for rn in rule_names:
        idx = np.argsort(-np.asarray(rules[rn]))[: min(K, n)]
        rule_support[idx] += 1

    reservoir = np.where(rule_support > 0)[0]
    rule_fraction = rule_support / float(len(rule_names))
    Q = np.sqrt(
        np.clip(SII, 1e-12, 1.0)
        * np.clip(rule_fraction, 1e-12, 1.0)
    )

    selected = []
    for _ in range(min(K, len(reservoir))):
        best = None
        for i in reservoir:
            if i in selected:
                continue
            deff = effective_separation(int(i), selected, D)
            phi = harmonic_couple(float(Q[i]), deff)
            key = (phi, Q[i], SII[i], -int(i))
            if best is None or key > best[0]:
                best = (key, int(i))
        selected.append(best[1])
    return selected


def fps_select(D, seed_quality, K):
    seed = int(np.argmax(seed_quality))
    selected = [seed]
    mind = D[:, seed].copy()
    mind[seed] = -1.0

    while len(selected) < min(K, len(D)):
        i = int(np.argmax(mind))
        selected.append(i)
        mind = np.minimum(mind, D[:, i])
        mind[selected] = -1.0
    return selected


def mmr_select(quality, D, K, lam=0.7):
    q = minmax(quality, True)
    selected = [int(np.argmax(q))]
    while len(selected) < min(K, len(q)):
        novelty = np.min(D[:, selected], axis=1)
        score = lam * q + (1.0 - lam) * novelty
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


def old_memory_select(df, D, K):
    S = df["S"].to_numpy(float)
    G = df["G"].to_numpy(float)
    selected = []

    while len(selected) < min(K, len(df)):
        novelty = np.ones(len(df)) if not selected else np.min(D[:, selected], axis=1)
        score = 0.5 * S + 0.3 * G + 0.2 * novelty
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


def topk(v, K):
    return np.argsort(-np.asarray(v))[: min(K, len(v))].tolist()


def shortlist_metrics(
    df, idx, D, R, C, V, SII,
    Q=None, family_support=None
):
    idx = np.asarray(idx, dtype=int)
    sub = df.iloc[idx]
    reg, all_reg = worst_rule_regret(idx, R)

    out = {
        "N": len(idx),
        "mean_band_gap_eV": float(sub["band_gap"].mean()),
        "mean_Ehull_eV_atom": float(sub["energy_above_hull"].mean()),
        "mean_pairwise_cation_distance": mean_pairwise(idx, D),
        "mean_nearest_neighbor_cation_distance": dnn(idx, D),
        "unique_cation_profiles": int(unique_profiles(idx, D)),
        "mean_consensus_C": float(C[idx].mean()),
        "mean_formula_sensitivity_V": float(V[idx].mean()),
        "mean_SII": float(SII[idx].mean()),
        "avg_rule_regret": float(all_reg.mean()),
        "worst_rule_regret": reg,
    }

    if Q is not None:
        out["mean_cross_family_Q"] = float(Q[idx].mean())
    if family_support is not None:
        out["mean_family_support"] = float(family_support[idx].mean())
        out["min_family_support"] = int(family_support[idx].min())
    return out


def run_support_sensitivity(
    name, df, rules, rule_names, R, C, V, SII, D,
    family_support, K
):
    rows = []

    for threshold in (1, 2, 3, 4):
        reservoir, famfrac, Q = family_reservoir(
            SII, family_support, threshold
        )
        if len(reservoir) < K:
            rows.append(
                {
                    "dataset": name,
                    "min_family_support": threshold,
                    "reservoir_size": len(reservoir),
                    "feasible": False,
                }
            )
            continue

        seed = quality_seed(
            reservoir, Q, SII, family_support, K
        )
        final, hist, budget = regret_preserving_diversification(
            reservoir, seed, R, D, Q, SII, family_support
        )

        row = {
            "dataset": name,
            "min_family_support": threshold,
            "reservoir_size": len(reservoir),
            "feasible": True,
            "seed_worst_rule_regret": budget,
            "number_of_accepted_swaps": max(0, len(hist) - 1),
            **shortlist_metrics(
                df, final, D, R, C, V, SII,
                Q=Q, family_support=family_support
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_dataset(name, K, min_family_support=2):
    df = prepare_dataset(name)
    X, D, elems = chemical_distance_matrix(df)
    rules = build_scoring_rules(df)
    rule_names, R, C, V, SII = scoring_fingerprint(rules)

    family_support, rule_support, family_hits = family_support_statistics(
        rules, rule_names, K
    )

    reservoir, family_fraction, Q = family_reservoir(
        SII, family_support, min_family_support
    )

    if len(reservoir) < K:
        raise RuntimeError(
            f"{name}: family-consistency reservoir has only "
            f"{len(reservoir)} candidates for K={K}."
        )

    seed = quality_seed(
        reservoir, Q, SII, family_support, K
    )

    ficd03, swap_history, regret_budget = regret_preserving_diversification(
        reservoir, seed, R, D, Q, SII, family_support
    )

    arith06 = rules["arith_ws0.6"]
    geom06 = rules["geom_ws0.6"]
    cheb06 = rules["cheb_ws0.6"]

    ficd01 = ficd_v01_select(SII, D, K)
    ficd02 = ficd_v02_select(
        rules, rule_names, SII, family_support, D, K
    )

    methods = {
        "Band-gap TopK": topk(df["G"].to_numpy(float), K),
        "Arithmetic TopK": topk(arith06, K),
        "Geometric TopK": topk(geom06, K),
        "Chebyshev TopK": topk(cheb06, K),
        "FPS": fps_select(D, arith06, K),
        "MMR": mmr_select(arith06, D, K, lam=0.7),
        "Old sequential memory": old_memory_select(df, D, K),
        "FICD v0.1": ficd01,
        "FICD v0.2": ficd02,
        "Cross-family Q seed": seed,
        "FICD v0.3": ficd03,
    }

    # Candidate diagnostics.
    fp = df[
        ["material_id", "formula", "band_gap",
         "energy_above_hull", "S", "G"]
    ].copy()

    fp["consensus_C"] = C
    fp["formula_sensitivity_V"] = V
    fp["SII"] = SII
    fp["family_support_count"] = family_support
    fp["rule_support_count"] = rule_support
    fp["family_support_fraction"] = family_fraction
    fp["cross_family_Q"] = Q
    fp["eligible_v03"] = family_support >= min_family_support

    for f in FAMILIES:
        fp[f"supported_by_{f}_family"] = family_hits[f]

    for j, rn in enumerate(rule_names):
        fp[f"rankscore__{rn}"] = R[:, j]

    fp.to_csv(
        RESULTS / f"{name.lower()}_scoring_fingerprint_v03.csv",
        index=False
    )

    fp[fp["eligible_v03"]].sort_values(
        ["cross_family_Q", "SII", "family_support_count"],
        ascending=False,
    ).to_csv(
        RESULTS / f"{name.lower()}_cross_family_reservoir_v03.csv",
        index=False,
    )

    # Final shortlist.
    final = df.iloc[ficd03].copy().reset_index(drop=True)
    final.insert(0, "FICD_v03_rank", np.arange(1, len(final) + 1))
    final["consensus_C"] = C[ficd03]
    final["formula_sensitivity_V"] = V[ficd03]
    final["SII"] = SII[ficd03]
    final["family_support_count"] = family_support[ficd03]
    final["rule_support_count"] = rule_support[ficd03]
    final["cross_family_Q"] = Q[ficd03]
    final.to_csv(
        RESULTS / f"{name.lower()}_FICD_v03_top{K}.csv",
        index=False
    )

    swap_history.to_csv(
        RESULTS / f"{name.lower()}_FICD_v03_swap_history.csv",
        index=False
    )

    # Benchmark.
    rows = []
    for method, idx in methods.items():
        rows.append(
            {
                "dataset": name,
                "method": method,
                **shortlist_metrics(
                    df, idx, D, R, C, V, SII,
                    Q=Q, family_support=family_support
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        RESULTS / f"{name.lower()}_benchmark_summary_v03.csv",
        index=False
    )

    # Sensitivity to minimum family support.
    sens = run_support_sensitivity(
        name, df, rules, rule_names, R, C, V, SII, D,
        family_support, K
    )
    sens.to_csv(
        RESULTS / f"{name.lower()}_family_support_sensitivity.csv",
        index=False
    )

    # Overlaps with v0.3.
    ref = set(ficd03)
    overlaps = []
    for method, idx in methods.items():
        s = set(idx)
        overlaps.append(
            {
                "dataset": name,
                "method": method,
                "overlap_with_FICD_v03":
                    len(s & ref) / min(len(s), len(ref)),
            }
        )
    pd.DataFrame(overlaps).to_csv(
        RESULTS / f"{name.lower()}_overlap_with_FICD_v03.csv",
        index=False
    )

    # Plots.
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    ax.scatter(
        summary["mean_nearest_neighbor_cation_distance"],
        summary["mean_band_gap_eV"],
        s=52,
    )
    for _, r in summary.iterrows():
        ax.annotate(
            r["method"],
            (
                r["mean_nearest_neighbor_cation_distance"],
                r["mean_band_gap_eV"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.3,
        )
    ax.set_xlabel("Mean nearest-neighbor cation distance")
    ax.set_ylabel("Mean band gap (eV)")
    ax.set_title(f"{name}: FICD v0.3 quality–diversity benchmark")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{name.lower()}_v03_quality_diversity.png",
        dpi=220,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    ax.scatter(
        summary["worst_rule_regret"],
        summary["mean_nearest_neighbor_cation_distance"],
        s=52,
    )
    for _, r in summary.iterrows():
        ax.annotate(
            r["method"],
            (
                r["worst_rule_regret"],
                r["mean_nearest_neighbor_cation_distance"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.3,
        )
    ax.set_xlabel("Worst scoring-rule regret (lower is better)")
    ax.set_ylabel("Mean nearest-neighbor cation distance")
    ax.set_title(f"{name}: robustness–diversity benchmark")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{name.lower()}_v03_regret_diversity.png",
        dpi=220,
    )
    plt.close(fig)

    meta = {
        "dataset": name,
        "admissible_candidates": len(df),
        "K": K,
        "scoring_rule_count": len(rule_names),
        "scoring_families": list(FAMILIES),
        "default_min_family_support": min_family_support,
        "default_reservoir_size": int(len(reservoir)),
        "default_regret_budget": float(regret_budget),
        "number_of_accepted_swaps": int(max(0, len(swap_history) - 1)),
        "v03_quality_rule":
            "worst-rule regret may not exceed the Cross-family Q seed regret",
        "v03_diversity_objective":
            "increase mean nearest-neighbor cation distance by 1-swap local search",
        "chemical_distance":
            "oxygen-excluded cation-fraction total-variation distance",
        "cation_elements": elems,
    }

    return summary, sens, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k", type=int, default=25,
        help="Shortlist size (default: 25)"
    )
    parser.add_argument(
        "--min-family-support", type=int, default=2,
        choices=(1, 2, 3, 4),
        help="Default v0.3 eligibility threshold (default: 2)"
    )
    args = parser.parse_args()

    all_summary = []
    all_sens = []
    metadata = {}

    for name in ("MP", "WBM"):
        summary, sens, meta = analyze_dataset(
            name, args.k, args.min_family_support
        )
        all_summary.append(summary)
        all_sens.append(sens)
        metadata[name] = meta

    combined = pd.concat(all_summary, ignore_index=True)
    combined.to_csv(
        RESULTS / "benchmark_all_v03.csv",
        index=False
    )

    sensitivity = pd.concat(all_sens, ignore_index=True)
    sensitivity.to_csv(
        RESULTS / "family_support_sensitivity_all.csv",
        index=False
    )

    with open(
        RESULTS / "run_metadata_v03.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    cols = [
        "method",
        "mean_band_gap_eV",
        "mean_Ehull_eV_atom",
        "mean_nearest_neighbor_cation_distance",
        "unique_cation_profiles",
        "mean_SII",
        "mean_family_support",
        "worst_rule_regret",
    ]

    lines = [
        "FICD benchmark v0.3 completed successfully.",
        f"Shortlist size K = {args.k}",
        f"Default minimum family support = {args.min_family_support}/4",
        "",
        "Core v0.3 rule:",
        "cross-family quality seed -> fixed worst-rule-regret budget -> "
        "diversity-improving swaps only",
        "",
        "Please send back the entire results folder.",
        "Most important file: results/benchmark_all_v03.csv",
        "",
    ]

    for name in ("MP", "WBM"):
        x = combined[combined["dataset"] == name]
        lines.append(f"=== {name} ===")
        lines.append(x[cols].to_string(index=False))
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (RESULTS / "RUN_COMPLETE_v03.txt").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
