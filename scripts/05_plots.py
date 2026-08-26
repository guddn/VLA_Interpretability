"""02/03/04 결과를 그림으로.

사용:
    python scripts/05_plots.py --suite spatial
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod import viz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="spatial")
    args = ap.parse_args()
    made = []

    pl_path = f"outputs/perlayer_{args.suite}.npz"
    csv_path = f"outputs/analysis_{args.suite}.csv"
    cf_path = f"outputs/counterfactual_{args.suite}.csv"

    if os.path.exists(pl_path):
        z = np.load(pl_path)
        per_layer = {k: z[k].mean(axis=0).tolist() for k in ("R_raw", "R_norm", "R_vnorm")}
        base = None
        if os.path.exists(csv_path):
            base = float(pd.read_csv(csv_path)["uniform_baseline"].mean())
        made.append(viz.plot_layerwise(per_layer, base, f"outputs/fig_layerwise_{args.suite}.png"))

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if {"KL_lang_knockout", "KL_vis_knockout"} <= set(df.columns):
            cr = df["KL_lang_knockout"] / (df["KL_lang_knockout"] + df["KL_vis_knockout"])
            m = np.isfinite(cr) & np.isfinite(df["R_norm"])
            rho = None
            if m.sum() > 8:
                from scipy import stats
                rho = float(stats.spearmanr(df["R_norm"][m], cr[m])[0])
            made.append(viz.plot_ratio_vs_causal(
                df["R_norm"][m], cr[m], f"outputs/fig_h2_{args.suite}.png", rho=rho))

    if os.path.exists(cf_path):
        cf = pd.read_csv(cf_path)
        order = [c for c in ["valid", "paraphrase", "swapped", "empty"]
                 if c in set(cf["condition"])]
        g = cf.groupby("condition")["R_norm"].mean().reindex(order)
        made.append(viz.plot_conditions(
            list(g.index), list(g.values), f"outputs/fig_cf_ratio_{args.suite}.png",
            ylabel="R_norm", title="Language attention ratio by instruction condition (H1)"))
        g2 = cf.groupby("condition")["KL_vs_valid"].mean().reindex(order)
        made.append(viz.plot_conditions(
            list(g2.index), list(g2.values), f"outputs/fig_cf_kl_{args.suite}.png",
            ylabel="KL vs valid", title="Action distribution shift by instruction condition (H1)",
            color="#eb6834"))

    if not made:
        print("그릴 데이터가 없습니다. 02/04 를 먼저 돌리세요.")
        return 1
    for m in made:
        print("저장:", m)
    return 0


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    sys.exit(main())
