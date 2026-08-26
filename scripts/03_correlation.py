"""H2 검정: attention 비율이 인과 기여를 예측하는가?

02_run_analysis.py 가 만든 CSV 하나만 있으면 됩니다.
결과가 어느 쪽이든 발표 슬라이드가 됩니다:
  - 상관 낮음 → IVAR 류 attention 지표는 인과 기여의 proxy 로 부적합 (2605.00321 비판 정량 확인)
  - 상관 높음 → attention 지표를 정당화 (기존 연구를 방어해 주는 결과)

사용:
    python scripts/03_correlation.py --csv outputs/analysis_spatial.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def report(x: np.ndarray, y: np.ndarray, xname: str, yname: str) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 8:
        return {"x": xname, "y": yname, "n": len(x), "note": "표본 부족"}
    rho, p_rho = stats.spearmanr(x, y)
    r, p_r = stats.pearsonr(x, y)
    return {
        "x": xname, "y": yname, "n": int(len(x)),
        "spearman_rho": round(float(rho), 4), "p_spearman": float(f"{p_rho:.3g}"),
        "pearson_r": round(float(r), 4), "p_pearson": float(f"{p_r:.3g}"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/analysis_spatial.csv")
    ap.add_argument("--out", default="outputs/correlation.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"R_raw", "R_norm", "R_vnorm", "KL_lang_knockout", "KL_vis_knockout"}
    missing = need - set(df.columns)
    if missing:
        print(f"필요한 컬럼이 없습니다: {missing}. --no-intervene 없이 02 를 다시 돌리세요.")
        return 1

    df = df.copy()
    df["causal_ratio"] = df["KL_lang_knockout"] / (
        df["KL_lang_knockout"] + df["KL_vis_knockout"]
    )

    results = []
    for xname in ["R_raw", "R_norm", "R_vnorm"]:
        results.append(report(df[xname].to_numpy(), df["causal_ratio"].to_numpy(),
                              xname, "causal_ratio"))
        results.append(report(df[xname].to_numpy(), df["KL_lang_knockout"].to_numpy(),
                              xname, "KL_lang_knockout"))

    out = pd.DataFrame(results)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))

    print("\n--- 해석 가이드 ---")
    print("|rho| < 0.3  → attention 비율은 인과 기여를 거의 예측 못 함 (H2 지지)")
    print("|rho| > 0.6  → attention 비율이 쓸 만한 proxy (H2 기각)")
    print("그 사이       → 조건부. 층/head 별로 나눠서 다시 보세요.")

    # 대조군 sanity check
    if "lang_vs_control" in df.columns:
        v = df["lang_vs_control"].replace([np.inf, -np.inf], np.nan).dropna()
        t, p = stats.ttest_1samp(v, 1.0) if len(v) > 2 else (np.nan, np.nan)
        print(f"\n[대조군 검정] lang_vs_control 평균={v.mean():.3f}, "
              f"H0(=1) 대비 t={t:.3f}, p={p:.3g}")
        print("  p 가 크면: 언어 토큰 차단이 같은 개수의 랜덤 visual 토큰 차단과 구별 안 됨")
        print("  → '언어를 안 쓴다'의 가장 강한 형태의 증거입니다.")
    return 0


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    sys.exit(main())
