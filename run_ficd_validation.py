#!/usr/bin/env python3
"""
FICD validation package v0.4
============================

IMPORTANT
---------
The FICD v0.3 core algorithm is frozen in this package.
v0.4 adds validation only; it does not tune the main method.

This package performs four jobs:

A) Fix the family-support sensitivity bookkeeping bug from v0.3.
B) Re-run the frozen v0.3 method with official ElMD composition distance.
C) Perform leave-one-scoring-family-out (LOFO) external formula validation.
D) Perform MP structural diversity validation using crystal system / space group,
   neither of which is used in candidate selection.

Core v0.3
---------
1) 16 scoring rules = 4 scalarization families x 4 preference settings.
2) Candidate eligibility requires Top-K support from >=2 scoring families.
3) Q = sqrt(SII * family_support_fraction).
4) The Q Top-K shortlist is the quality seed.
5) Freeze the seed shortlist's worst-rule regret as a hard quality budget.
6) Permit only 1-for-1 swaps that improve nearest-neighbor chemical diversity
   without exceeding the frozen regret budget.

No fitted quality-diversity mixing coefficient is introduced.
"""

from pathlib import Path
import argparse
import json
import math
import re
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = ROOT / "cache"
for p in (RESULTS, FIGURES, CACHE):
    p.mkdir(exist_ok=True)

FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
FAMILIES = ("arith", "geom", "harm", "cheb")
WEIGHTS = (0.4, 0.5, 0.6, 0.7)


def parse_formula(formula):
    comp = {}
    s = str(formula).replace(" ", "")
    for el, num in FORMULA_RE.findall(s):
        comp[el] = comp.get(el, 0.0) + (float(num) if num else 1.0)
    return comp


def clean_formula(formula):
    return str(formula).replace(" ", "")


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


def tv_distance_matrix(df):
    X, elems = cation_fraction_matrix(df["formula"])
    D = 0.5 * np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
    return X, D, elems


def compute_elmd_distance_matrix(df, dataset_name):
    """
    Official ElMD validation.

    Preferred backend:
      1. ElM2D (fast matrix implementation), if installed.
      2. ElMD reference package.

    No surrogate is silently substituted. If neither official implementation is
    available, the script raises a clear ImportError.
    """
    formulas = [clean_formula(x) for x in df["formula"].tolist()]
    cache_npy = CACHE / f"{dataset_name.lower()}_ElMD_mod_petti.npy"
    cache_csv = CACHE / f"{dataset_name.lower()}_ElMD_mod_petti.csv"
    info_json = CACHE / f"{dataset_name.lower()}_ElMD_backend.json"

    if cache_npy.exists():
        D = np.load(cache_npy)
        if D.shape == (len(formulas), len(formulas)):
            return D, {"backend": "cache", "metric": "mod_petti", "cache": str(cache_npy.name)}

    # Backend 1: ElM2D
    try:
        from ElM2D import ElM2D
        mapper = ElM2D(metric="mod_petti")
        mapper.fit(pd.Series(formulas))
        D = np.asarray(mapper.dm, dtype=float)
        if D.shape != (len(formulas), len(formulas)):
            raise RuntimeError("ElM2D returned unexpected distance-matrix shape.")
        backend = {"backend": "ElM2D", "metric": "mod_petti"}
    except Exception as elm2d_error:
        # Backend 2: reference ElMD
        try:
            from ElMD import ElMD
        except Exception as elmd_import_error:
            raise ImportError(
                "Official ElMD implementation is not available.\n"
                "Install one of:\n"
                "  python -m pip install ElMD --no-deps\n"
                "or\n"
                "  python -m pip install ElM2D\n"
                f"ElM2D error: {elm2d_error}\n"
                f"ElMD import error: {elmd_import_error}"
            )

        n = len(formulas)
        D = np.zeros((n, n), dtype=float)

        # Try the fast modified-Pettifor path first. If unsupported, fall back
        # to the default reference metric. Both are official ElMD behavior.
        objects = []
        backend_metric = "fast(mod_petti)"
        try:
            objects = [ElMD(f, metric="fast") for f in formulas]
            _ = objects[0].elmd(formulas[0]) if objects else 0.0
        except Exception:
            objects = [ElMD(f) for f in formulas]
            backend_metric = "default(mod_petti)"

        t0 = time.time()
        for i in range(n):
            if i % 25 == 0:
                print(f"[{dataset_name}] ElMD row {i+1}/{n}")
            for j in range(i + 1, n):
                d = float(objects[i].elmd(formulas[j]))
                D[i, j] = d
                D[j, i] = d
        backend = {
            "backend": "ElMD",
            "metric": backend_metric,
            "elapsed_seconds": round(time.time() - t0, 3),
        }

    np.save(cache_npy, D)
    pd.DataFrame(D).to_csv(cache_csv, index=False)
    info_json.write_text(json.dumps(backend, indent=2), encoding="utf-8")
    return D, backend


def build_scoring_rules(df, families=FAMILIES):
    S = np.clip(df["S"].to_numpy(float), 1e-6, 1.0)
    G = np.clip(df["G"].to_numpy(float), 1e-6, 1.0)
    rules = {}

    for ws in WEIGHTS:
        wg = 1.0 - ws
        if "arith" in families:
            rules[f"arith_ws{ws:.1f}"] = ws * S + wg * G
        if "geom" in families:
            rules[f"geom_ws{ws:.1f}"] = (S ** ws) * (G ** wg)
        if "harm" in families:
            rules[f"harm_ws{ws:.1f}"] = 1.0 / (ws / S + wg / G)
        if "cheb" in families:
            loss = np.maximum(ws * (1.0 - S), wg * (1.0 - G))
            rules[f"cheb_ws{ws:.1f}"] = 1.0 - loss

    return rules


def rank_desirability_matrix(rules):
    names = list(rules.keys())
    n = len(next(iter(rules.values())))
    R = np.zeros((n, len(names)), dtype=float)

    for j, name in enumerate(names):
        ranks = rankdata(-np.asarray(rules[name]), method="average")
        if n == 1:
            R[:, j] = 1.0
        else:
            R[:, j] = np.maximum(1e-6, 1.0 - (ranks - 1.0) / n)
    return names, R


def scoring_fingerprint(rules):
    names, R = rank_desirability_matrix(rules)
    logR = np.log(R)
    C = np.exp(logR.mean(axis=1))
    V = logR.std(axis=1)
    SII = C * np.exp(-V)
    return names, R, C, V, SII


def family_support_statistics(rules, rule_names, K, families):
    n = len(next(iter(rules.values())))
    family_hits = {f: np.zeros(n, dtype=bool) for f in families}
    rule_support = np.zeros(n, dtype=int)

    for rn in rule_names:
        idx = np.argsort(-np.asarray(rules[rn]))[: min(K, n)]
        rule_support[idx] += 1
        family = rn.split("_")[0]
        family_hits[family][idx] = True

    family_support = np.zeros(n, dtype=int)
    for f in families:
        family_support += family_hits[f].astype(int)

    return family_support, rule_support, family_hits


def family_reservoir(SII, family_support, min_family_support, family_count):
    reservoir = np.where(family_support >= min_family_support)[0]
    family_fraction = family_support / float(family_count)
    Q = np.sqrt(
        np.clip(SII, 1e-12, 1.0)
        * np.clip(family_fraction, 1e-12, 1.0)
    )
    return reservoir, family_fraction, Q


def regret_vector(idx, R):
    idx = np.asarray(idx, dtype=int)
    k = len(idx)
    regrets = []
    for j in range(R.shape[1]):
        best = np.sort(R[:, j])[-k:].mean()
        got = R[idx, j].mean()
        regrets.append(best - got)
    return np.asarray(regrets, dtype=float)


def worst_rule_regret(idx, R):
    rv = regret_vector(idx, R)
    return float(rv.max()), rv


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


def unique_profiles(idx, D, tol=1e-12):
    reps = []
    for i in idx:
        if not any(D[i, j] <= tol for j in reps):
            reps.append(i)
    return len(reps)


def quality_seed(reservoir, Q, SII, family_support, K):
    return sorted(
        map(int, reservoir),
        key=lambda i: (Q[i], SII[i], family_support[i], -i),
        reverse=True,
    )[: min(K, len(reservoir))]


def regret_preserving_diversification(
    reservoir, seed, R_quality, D_objective, Q,
    max_iterations=100
):
    """
    Frozen v0.3 swap logic.

    Quality hard constraint:
        worst_regret(candidate) <= worst_regret(seed)
    Diversity objective:
        strictly improve mean nearest-neighbor distance.
    """
    selected = list(map(int, seed))
    regret_budget, _ = worst_rule_regret(selected, R_quality)

    def objective(a):
        return (
            dnn(a, D_objective),
            unique_profiles(a, D_objective),
            mean_pairwise(a, D_objective),
            float(np.mean(Q[a])),
        )

    current_obj = objective(selected)
    history = [{
        "iteration": 0,
        "removed_index": "",
        "added_index": "",
        "dNN_objective": current_obj[0],
        "unique_profiles_objective": current_obj[1],
        "mean_pairwise_objective": current_obj[2],
        "mean_Q": current_obj[3],
        "worst_rule_regret": worst_rule_regret(selected, R_quality)[0],
        "regret_budget": regret_budget,
    }]

    reservoir = list(map(int, reservoir))

    for iteration in range(1, max_iterations + 1):
        outside = [i for i in reservoir if i not in selected]
        best = None

        for pos, rem in enumerate(selected):
            for add in outside:
                cand = selected.copy()
                cand[pos] = int(add)

                reg, _ = worst_rule_regret(cand, R_quality)
                if reg > regret_budget + 1e-12:
                    continue

                obj = objective(cand)
                if obj[0] <= current_obj[0] + 1e-12:
                    continue

                key = (obj[0], obj[1], obj[2], obj[3], -reg)
                if best is None or key > best[0]:
                    best = (key, cand, int(rem), int(add), reg, obj)

        if best is None:
            break

        _, selected, rem, add, reg, current_obj = best
        history.append({
            "iteration": iteration,
            "removed_index": rem,
            "added_index": add,
            "dNN_objective": current_obj[0],
            "unique_profiles_objective": current_obj[1],
            "mean_pairwise_objective": current_obj[2],
            "mean_Q": current_obj[3],
            "worst_rule_regret": reg,
            "regret_budget": regret_budget,
        })

    return selected, pd.DataFrame(history), regret_budget


def topk(v, K):
    return np.argsort(-np.asarray(v))[: min(K, len(v))].tolist()


def mmr_select(quality, D, K, lam=0.7):
    q = minmax(quality, True)
    selected = [int(np.argmax(q))]
    while len(selected) < min(K, len(q)):
        novelty = np.min(D[:, selected], axis=1)
        score = lam * q + (1.0 - lam) * novelty
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


def structural_entropy(values):
    s = pd.Series(values).dropna().astype(str)
    if len(s) == 0:
        return np.nan, np.nan, 0
    p = s.value_counts(normalize=True).to_numpy(float)
    H = float(-(p * np.log(p)).sum())
    return H, float(np.exp(H)), int(s.nunique())


def base_metrics(df, idx, R, C, V, SII):
    idx = np.asarray(idx, dtype=int)
    sub = df.iloc[idx]
    reg, rv = worst_rule_regret(idx, R)
    return {
        "N": len(idx),
        "mean_band_gap_eV": float(sub["band_gap"].mean()),
        "mean_Ehull_eV_atom": float(sub["energy_above_hull"].mean()),
        "mean_consensus_C": float(C[idx].mean()),
        "mean_formula_sensitivity_V": float(V[idx].mean()),
        "mean_SII": float(SII[idx].mean()),
        "avg_rule_regret": float(rv.mean()),
        "worst_rule_regret": reg,
    }


def run_frozen_v03(df, D, K=25, min_family_support=2, families=FAMILIES):
    rules = build_scoring_rules(df, families=families)
    names, R, C, V, SII = scoring_fingerprint(rules)
    family_support, rule_support, family_hits = family_support_statistics(
        rules, names, K, families
    )
    reservoir, family_fraction, Q = family_reservoir(
        SII, family_support, min_family_support, len(families)
    )

    if len(reservoir) < K:
        return None

    seed = quality_seed(reservoir, Q, SII, family_support, K)
    final, history, budget = regret_preserving_diversification(
        reservoir, seed, R, D, Q
    )

    return {
        "rules": rules,
        "rule_names": names,
        "R": R,
        "C": C,
        "V": V,
        "SII": SII,
        "family_support": family_support,
        "rule_support": rule_support,
        "family_hits": family_hits,
        "reservoir": reservoir,
        "family_fraction": family_fraction,
        "Q": Q,
        "seed": seed,
        "final": final,
        "history": history,
        "regret_budget": budget,
    }


def run_fixed_sensitivity(name, df, D, K):
    """
    FIXED bookkeeping:
    eligibility_threshold_families is never overwritten by shortlist metrics.
    """
    full_rules = build_scoring_rules(df)
    names, R, C, V, SII = scoring_fingerprint(full_rules)
    famsup, _, _ = family_support_statistics(full_rules, names, K, FAMILIES)

    rows = []
    for threshold in (1, 2, 3, 4):
        reservoir, famfrac, Q = family_reservoir(
            SII, famsup, threshold, len(FAMILIES)
        )
        row = {
            "dataset": name,
            "eligibility_threshold_families": threshold,
            "reservoir_size": int(len(reservoir)),
            "feasible": bool(len(reservoir) >= K),
        }

        if len(reservoir) >= K:
            seed = quality_seed(reservoir, Q, SII, famsup, K)
            final, history, budget = regret_preserving_diversification(
                reservoir, seed, R, D, Q
            )
            row.update(base_metrics(df, final, R, C, V, SII))
            row.update({
                "selected_min_family_support": int(famsup[final].min()),
                "selected_mean_family_support": float(famsup[final].mean()),
                "seed_worst_rule_regret": float(budget),
                "accepted_swaps": int(max(0, len(history) - 1)),
                "dNN_TV": float(dnn(final, D)),
            })
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(
        RESULTS / f"{name.lower()}_family_support_sensitivity_FIXED.csv",
        index=False,
    )
    return out


def distance_validation(name, df, tv_run, D_tv, D_elmd, K):
    """
    Re-run the frozen v0.3 using ElMD as the chemical-distance objective.
    Quality seed and scoring rules are unchanged.
    """
    elmd_run = run_frozen_v03(
        df, D_elmd, K=K, min_family_support=2, families=FAMILIES
    )
    if elmd_run is None:
        raise RuntimeError(f"{name}: ElMD v0.3 run infeasible.")

    methods = {
        "Cross-family Q seed": tv_run["seed"],
        "FICD v0.3 (TV objective)": tv_run["final"],
        "FICD v0.3 (ElMD objective)": elmd_run["final"],
    }

    rows = []
    for method, idx in methods.items():
        m = base_metrics(
            df, idx, tv_run["R"], tv_run["C"], tv_run["V"], tv_run["SII"]
        )
        m.update({
            "dataset": name,
            "method": method,
            "dNN_TV": dnn(idx, D_tv),
            "mean_pairwise_TV": mean_pairwise(idx, D_tv),
            "dNN_ElMD": dnn(idx, D_elmd),
            "mean_pairwise_ElMD": mean_pairwise(idx, D_elmd),
            "unique_profiles_TV": unique_profiles(idx, D_tv),
            "overlap_with_TV_FICD": len(
                set(idx) & set(tv_run["final"])
            ) / float(K),
        })
        rows.append(m)

    val = pd.DataFrame(rows)
    val.to_csv(
        RESULTS / f"{name.lower()}_distance_metric_validation.csv",
        index=False,
    )

    elmd_run["history"].to_csv(
        RESULTS / f"{name.lower()}_FICD_v03_ElMD_swap_history.csv",
        index=False,
    )

    final = df.iloc[elmd_run["final"]].copy().reset_index(drop=True)
    final.insert(0, "FICD_v03_ElMD_rank", np.arange(1, len(final) + 1))
    final["SII"] = elmd_run["SII"][elmd_run["final"]]
    final["family_support_count"] = elmd_run["family_support"][elmd_run["final"]]
    final["cross_family_Q"] = elmd_run["Q"][elmd_run["final"]]
    final.to_csv(
        RESULTS / f"{name.lower()}_FICD_v03_ElMD_top{K}.csv",
        index=False,
    )

    return val, elmd_run


def lofo_validation(name, df, D_tv, K):
    """
    Leave one scoring FAMILY entirely out of construction.

    Training:
      3 families only, with >=2-family eligibility and regret hard constraint.

    Evaluation:
      all 4 weight settings from the held-out family, never used during
      shortlist construction.
    """
    full_rules = build_scoring_rules(df)
    full_names, full_R = rank_desirability_matrix(full_rules)

    rows = []
    shortlist_rows = []

    for heldout in FAMILIES:
        train_families = tuple(f for f in FAMILIES if f != heldout)
        train = run_frozen_v03(
            df, D_tv, K=K, min_family_support=2, families=train_families
        )

        if train is None:
            rows.append({
                "dataset": name,
                "heldout_family": heldout,
                "method": "LOFO FICD",
                "feasible": False,
            })
            continue

        held_rules = {
            rn: vals for rn, vals in full_rules.items()
            if rn.startswith(heldout + "_")
        }
        held_names, held_R = rank_desirability_matrix(held_rules)

        for method, idx in (
            ("LOFO quality seed", train["seed"]),
            ("LOFO FICD", train["final"]),
        ):
            held_reg = regret_vector(idx, held_R)
            train_reg = regret_vector(idx, train["R"])
            rows.append({
                "dataset": name,
                "heldout_family": heldout,
                "method": method,
                "feasible": True,
                "training_families": ",".join(train_families),
                "training_reservoir_size": int(len(train["reservoir"])),
                "accepted_swaps": int(max(0, len(train["history"]) - 1))
                    if method == "LOFO FICD" else 0,
                "mean_band_gap_eV": float(df.iloc[idx]["band_gap"].mean()),
                "mean_Ehull_eV_atom": float(
                    df.iloc[idx]["energy_above_hull"].mean()
                ),
                "dNN_TV": float(dnn(idx, D_tv)),
                "training_avg_regret": float(train_reg.mean()),
                "training_worst_regret": float(train_reg.max()),
                "heldout_avg_regret": float(held_reg.mean()),
                "heldout_worst_regret": float(held_reg.max()),
            })

            for rank_pos, ii in enumerate(idx, 1):
                shortlist_rows.append({
                    "dataset": name,
                    "heldout_family": heldout,
                    "method": method,
                    "rank": rank_pos,
                    "material_id": df.iloc[ii]["material_id"],
                    "formula": df.iloc[ii]["formula"],
                })

    out = pd.DataFrame(rows)
    out.to_csv(
        RESULTS / f"{name.lower()}_LOFO_external_formula_validation.csv",
        index=False,
    )
    pd.DataFrame(shortlist_rows).to_csv(
        RESULTS / f"{name.lower()}_LOFO_shortlists.csv",
        index=False,
    )
    return out


def mp_structural_validation(df, method_indices):
    """
    External structural validation.

    Crystal system and space group are NOT used by FICD selection.
    """
    required = ["crystal_system", "Space Group Number", "Space Group Symbol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"MP structural columns missing: {missing}")

    rows = []
    crystal_dist = []
    sg_dist = []

    for method, idx in method_indices.items():
        sub = df.iloc[idx].copy()

        Hc, effc, uc = structural_entropy(sub["crystal_system"])
        Hs, effs, us = structural_entropy(sub["Space Group Number"])

        rows.append({
            "dataset": "MP",
            "method": method,
            "N": len(sub),
            "unique_crystal_systems": uc,
            "crystal_system_entropy": Hc,
            "effective_crystal_systems_expH": effc,
            "unique_space_groups": us,
            "space_group_entropy": Hs,
            "effective_space_groups_expH": effs,
        })

        for key, count in sub["crystal_system"].fillna("Missing").value_counts().items():
            crystal_dist.append({
                "method": method,
                "crystal_system": key,
                "count": int(count),
                "fraction": float(count / len(sub)),
            })

        for key, count in sub["Space Group Number"].fillna(-1).value_counts().items():
            sg_dist.append({
                "method": method,
                "space_group_number": key,
                "count": int(count),
                "fraction": float(count / len(sub)),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        RESULTS / "mp_external_structural_validation.csv",
        index=False,
    )
    pd.DataFrame(crystal_dist).to_csv(
        RESULTS / "mp_crystal_system_distributions.csv",
        index=False,
    )
    pd.DataFrame(sg_dist).to_csv(
        RESULTS / "mp_space_group_distributions.csv",
        index=False,
    )
    return summary


def make_plots(distance_all, lofo_all, structural):
    # 1. ElMD distance validation
    for dataset in distance_all["dataset"].unique():
        x = distance_all[distance_all["dataset"] == dataset].copy()
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.scatter(x["dNN_ElMD"], x["mean_band_gap_eV"], s=60)
        for _, r in x.iterrows():
            ax.annotate(
                r["method"],
                (r["dNN_ElMD"], r["mean_band_gap_eV"]),
                xytext=(5, 4), textcoords="offset points", fontsize=8,
            )
        ax.set_xlabel("Mean nearest-neighbor ElMD")
        ax.set_ylabel("Mean band gap (eV)")
        ax.set_title(f"{dataset}: ElMD validation of frozen FICD v0.3")
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"{dataset.lower()}_ElMD_validation.png", dpi=220
        )
        plt.close(fig)

    # 2. LOFO held-out regret
    for dataset in lofo_all["dataset"].unique():
        x = lofo_all[
            (lofo_all["dataset"] == dataset)
            & (lofo_all["method"] == "LOFO FICD")
            & (lofo_all["feasible"] == True)
        ].copy()
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.bar(x["heldout_family"], x["heldout_worst_regret"])
        ax.set_xlabel("Held-out scoring family")
        ax.set_ylabel("Held-out worst-rule regret")
        ax.set_title(f"{dataset}: leave-one-family-out validation")
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"{dataset.lower()}_LOFO_heldout_regret.png", dpi=220
        )
        plt.close(fig)

    # 3. MP structural coverage
    if structural is not None and len(structural):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        x = np.arange(len(structural))
        width = 0.38
        ax.bar(
            x - width / 2,
            structural["unique_crystal_systems"],
            width,
            label="Crystal systems",
        )
        ax.bar(
            x + width / 2,
            structural["unique_space_groups"],
            width,
            label="Space groups",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(structural["method"], rotation=25, ha="right")
        ax.set_ylabel("Unique structural categories")
        ax.set_title("MP external structural diversity validation")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "mp_external_structural_validation.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument(
        "--skip-elmd",
        action="store_true",
        help="Developer/smoke-test mode only. Full validation should NOT use this.",
    )
    args = parser.parse_args()
    K = args.k

    all_distance = []
    all_lofo = []
    all_sens = []
    meta = {}
    mp_structural = None

    dataset_runs = {}

    for name in ("MP", "WBM"):
        print(f"\n=== {name}: loading and frozen v0.3 ===")
        df = prepare_dataset(name)
        X, D_tv, elems = tv_distance_matrix(df)

        tv_run = run_frozen_v03(
            df, D_tv, K=K, min_family_support=2, families=FAMILIES
        )
        if tv_run is None:
            raise RuntimeError(f"{name}: frozen v0.3 TV run infeasible.")

        sens = run_fixed_sensitivity(name, df, D_tv, K)
        all_sens.append(sens)

        lofo = lofo_validation(name, df, D_tv, K)
        all_lofo.append(lofo)

        dataset_runs[name] = {
            "df": df,
            "D_tv": D_tv,
            "tv_run": tv_run,
        }

        meta[name] = {
            "admissible_candidates": len(df),
            "K": K,
            "TV_reservoir_size": int(len(tv_run["reservoir"])),
            "TV_regret_budget": float(tv_run["regret_budget"]),
            "TV_accepted_swaps": int(max(0, len(tv_run["history"]) - 1)),
            "cation_elements": elems,
        }

    if not args.skip_elmd:
        for name in ("MP", "WBM"):
            print(f"\n=== {name}: official ElMD validation ===")
            df = dataset_runs[name]["df"]
            D_tv = dataset_runs[name]["D_tv"]
            tv_run = dataset_runs[name]["tv_run"]

            D_elmd, backend = compute_elmd_distance_matrix(df, name)
            distance_val, elmd_run = distance_validation(
                name, df, tv_run, D_tv, D_elmd, K
            )
            all_distance.append(distance_val)
            dataset_runs[name]["D_elmd"] = D_elmd
            dataset_runs[name]["elmd_run"] = elmd_run
            meta[name]["ElMD_backend"] = backend
            meta[name]["ElMD_accepted_swaps"] = int(
                max(0, len(elmd_run["history"]) - 1)
            )

    # MP structural validation
    mp = dataset_runs["MP"]
    mp_df = mp["df"]
    mp_rules = mp["tv_run"]["rules"]
    arith06 = mp_rules["arith_ws0.6"]
    geom06 = mp_rules["geom_ws0.6"]

    structural_methods = {
        "Arithmetic TopK": topk(arith06, K),
        "Geometric TopK": topk(geom06, K),
        "MMR": mmr_select(arith06, mp["D_tv"], K, lam=0.7),
        "Cross-family Q seed": mp["tv_run"]["seed"],
        "FICD v0.3 (TV)": mp["tv_run"]["final"],
    }
    if not args.skip_elmd:
        structural_methods["FICD v0.3 (ElMD)"] = mp["elmd_run"]["final"]

    mp_structural = mp_structural_validation(mp_df, structural_methods)

    sens_all = pd.concat(all_sens, ignore_index=True)
    sens_all.to_csv(
        RESULTS / "family_support_sensitivity_FIXED_all.csv", index=False
    )

    lofo_all = pd.concat(all_lofo, ignore_index=True)
    lofo_all.to_csv(
        RESULTS / "LOFO_external_formula_validation_all.csv", index=False
    )

    if all_distance:
        distance_all = pd.concat(all_distance, ignore_index=True)
        distance_all.to_csv(
            RESULTS / "distance_metric_validation_all.csv", index=False
        )
    else:
        distance_all = pd.DataFrame()

    # Combined validation verdict table (descriptive, not pass/fail tuning).
    verdict_rows = []
    for name in ("MP", "WBM"):
        tv = dataset_runs[name]["tv_run"]
        df = dataset_runs[name]["df"]
        seed = tv["seed"]
        final = tv["final"]
        verdict_rows.append({
            "dataset": name,
            "comparison": "TV FICD vs quality seed",
            "seed_band_gap": float(df.iloc[seed]["band_gap"].mean()),
            "FICD_band_gap": float(df.iloc[final]["band_gap"].mean()),
            "seed_worst_regret": worst_rule_regret(seed, tv["R"])[0],
            "FICD_worst_regret": worst_rule_regret(final, tv["R"])[0],
            "seed_dNN_TV": dnn(seed, dataset_runs[name]["D_tv"]),
            "FICD_dNN_TV": dnn(final, dataset_runs[name]["D_tv"]),
        })
    pd.DataFrame(verdict_rows).to_csv(
        RESULTS / "frozen_v03_core_reproduction.csv", index=False
    )

    with open(RESULTS / "run_metadata_v04.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if len(distance_all):
        make_plots(distance_all, lofo_all, mp_structural)
    else:
        # Still produce LOFO/structure plots in smoke-test mode.
        make_plots(
            pd.DataFrame(columns=[
                "dataset", "dNN_ElMD", "mean_band_gap_eV", "method"
            ]),
            lofo_all,
            mp_structural,
        )

    # Human-readable summary.
    lines = [
        "FICD validation v0.4 completed successfully.",
        f"K = {K}",
        "",
        "Core FICD v0.3 was frozen; v0.4 adds validation only.",
        "",
        "Completed:",
        "1) FIXED family-support sensitivity bookkeeping",
        "2) Leave-one-scoring-family-out validation",
        "3) MP external structural validation",
    ]
    if args.skip_elmd:
        lines.append("4) ElMD validation: SKIPPED (smoke-test mode only)")
    else:
        lines.append("4) Official ElMD distance validation: COMPLETED")

    lines += [
        "",
        "Please send back the ENTIRE results folder.",
        "Also send cache/ if possible, especially the ElMD matrices.",
        "",
        "Most important files:",
        "- results/distance_metric_validation_all.csv",
        "- results/LOFO_external_formula_validation_all.csv",
        "- results/mp_external_structural_validation.csv",
        "- results/family_support_sensitivity_FIXED_all.csv",
        "- results/frozen_v03_core_reproduction.csv",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (RESULTS / "RUN_COMPLETE_v04.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
