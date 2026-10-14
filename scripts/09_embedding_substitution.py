"""09: 토큰 임베딩 치환 [L2] — knockout(직접 효과) vs mean 치환(총 효과) 를 **같은 프레임**에서.

왜 필요한가
-----------
02 의 knockout 은 Q 행(action 을 예측하는 자리) → 대상 토큰 attention 만 막습니다.
  - L knockout: L → 템플릿 286–289 → Q 우회 경로가 열려 있음 (4개 위치)
  - V knockout: V → 템플릿 257–265 / **L 266–285** / 템플릿 286–289 → Q 우회 (33개 위치)
즉 02 의 causal_ratio 는 **직접 효과끼리의 비교**이고, 우회 기회가 V 쪽에 훨씬 많습니다.
이 스크립트는 0층 입력에서 그 위치의 **내용**을 평균값으로 바꿔 총 효과를 재고,
같은 프레임의 knockout 과 행 단위로 비교합니다.

기준값 (docs/05_interventions.md [L2])
-------------------------------------
  mean          L: suite 전체 지시문의 L 토큰 임베딩 빈도 가중 평균 (벡터 1개)
                V: 수집한 모든 프레임의 0층 V 입력(= projector 출력)을 **패치 위치별** 평균
  mean_rescaled 방향은 mean, 크기는 치환되는 원래 임베딩들의 평균 L2 크기
  (zero / Gaussian noise 는 분포 밖이라 쓰지 않음. 실제 토큰 치환은 0층에서 04·06 과 동일해 생략)

조건 (한 프레임당 forward 12회)
-------------------------------
  base · knockout L/V/ctrl · mean L/V/ctrl · mean_rescaled L/V · 구간 L_loc/L_shared(spatial 전용)
  ctrl 은 knockout 대조군과 **같은** 무작위 |L| 개 visual 위치 (seed --control-seed)

통과 기준 (gates)
-----------------
  G1 정렬   매 프레임: 0층 입력의 V 가 아닌 모든 위치 == embed_tokens(텍스트 id)  → 실패 시 중단
  G2 항등   앞 --gate-frames 개: L·V 를 원래 0층 입력으로 '치환' → KL ≈ 0      → 실패 시 중단
  G3 바닥값 앞 --gate-frames 개: base 를 한 번 더 → KL (같은 GPU 재현성)        → 기록만
  G4 대조군 실행 끝: KL_lang_sub > KL_control_sub 인 행 비율                      → 요약에 출력

출력: outputs/09_embedding_substitution/
  embed_sub_<suite>.csv     한 행 = 한 프레임
  baselines_<suite>.pt      L_mean, V_mean, 사용한 L 토큰 id, 프레임 수
  gates_<suite>.txt         G1–G4 결과

사용:
    python scripts/09_embedding_substitution.py --gpu 4 --suite spatial --tasks 0 1 2 3 4 --episodes 3
!! 정책 궤적은 GPU 종류에 따라 다릅니다. 02 와 같은 GPU 로 돌리면 (task, ep, t) 프레임이 02 와 같아집니다.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod.device import apply_env, apply_overrides, report_gpu  # noqa: E402
from vlamod import capture as cap  # noqa: E402
from vlamod import env_libero as EL  # noqa: E402
from vlamod import intervene as IV  # noqa: E402
from vlamod import metrics as M  # noqa: E402
from vlamod import paths as OP  # noqa: E402
from vlamod import pipeline as P  # noqa: E402
from vlamod import substitute as SB  # noqa: E402
from vlamod import token_index as TI  # noqa: E402
from vlamod.model_loader import load_openvla  # noqa: E402


def kl(c0, c) -> float:
    return float(M.action_kl(c0.action_logits, c.action_logits))


def n_changed(c0, c) -> int:
    return int((M.decode_actions(c0.action_logits) != M.decode_actions(c.action_logits)).sum())


def prepare(vla, image, instruction, vspan, cfg):
    prompt, span = TI.build_prompt(instruction)
    inputs = TI.prepare_inputs(vla, prompt, image)   # 29871 포함 — 정책과 같은 입력
    spans = TI.build_spans(
        vla, prompt, span, inputs["input_ids"][0], vspan,
        n_action_tokens=7,
        sink_positions=cfg["tokens"]["sink_positions"],
        instruction_only=cfg["tokens"]["instruction_only"],
    )
    return prompt, span, inputs, spans


def segment_positions(vla, prompt, instr_span, ids, spans, vspan):
    """spatial 지시문의 L 을 (위치구 L_loc, 나머지 L_shared) 로 나눔. 형식이 다르면 (None, None)."""
    loc = SB.spatial_location_span(prompt[instr_span[0]:instr_span[1]])
    if loc is None:
        return None, None
    a, b = instr_span[0] + loc[0], instr_span[0] + loc[1]
    offsets = TI.token_char_offsets(vla.tokenizer, prompt, ids)
    n_vis = vspan[1] - vspan[0]
    pos_char = {}
    for p in spans.language:
        s, e = offsets[p if p < vspan[0] else p - n_vis]
        while s < e and prompt[s].isspace():      # build_spans 와 같은 공백 제거 규칙
            s += 1
        while e > s and prompt[e - 1].isspace():
            e -= 1
        pos_char[p] = (s, e)
    loc_pos = SB.overlapping_positions(pos_char, (a, b))
    shared = [p for p in spans.language if p not in set(loc_pos)]
    return loc_pos, shared


def episode_success(task, ep_done):
    for obj in (task.env, getattr(task.env, "env", None)):
        fn = getattr(obj, "check_success", None) or getattr(obj, "_check_success", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:  # noqa: BLE001
                pass
    return bool(ep_done)


def cluster_bootstrap_median(df, col, n_boot=2000, seed=0):
    """에피소드 단위 cluster bootstrap 으로 행 중앙값의 95% CI."""
    d = df[["task_id", "episode", col]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return float("nan"), (float("nan"), float("nan"))
    groups = [g[col].to_numpy() for _, g in d.groupby(["task_id", "episode"])]
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        meds.append(np.median(np.concatenate([groups[i] for i in pick])))
    return float(np.median(d[col])), (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--stride", type=int, default=10, help="몇 스텝마다 프레임을 수집할지")
    ap.add_argument("--gate-frames", type=int, default=3, help="G2·G3 를 검사할 앞쪽 프레임 수")
    ap.add_argument("--control-seed", type=int, default=0)
    ap.add_argument("--no-rescaled", action="store_true", help="mean_rescaled 조건 생략 (forward 2회 절약)")
    ap.add_argument("--no-segments", action="store_true", help="L_loc / L_shared 구간 치환 생략")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--gpu", type=int, default=None, metavar="N")
    ap.add_argument("--headroom-gb", type=float, default=None, metavar="G")
    ap.add_argument("--allow-cpu-offload", action="store_true")
    ap.add_argument("--egl-device", type=int, default=None, metavar="E")
    ap.add_argument("--gpus", default=None, metavar="0,1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--tag", default=None, help="출력 하위 폴더/파일명 접미사")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    apply_env(cfg, args)          # ← transformers/libero import 전에 반드시 먼저
    mcfg = cfg["model"]
    apply_overrides(mcfg, args)
    torch.manual_seed(cfg["run"]["seed"])
    vla = load_openvla(path=mcfg["path"], device=mcfg["device"], dtype=mcfg["dtype"],
                       attn_implementation=mcfg["attn_implementation"],
                       gpus=mcfg.get("gpus"),
                       headroom_gb=mcfg.get("headroom_gb", 1.5),
                       cpu_offload=mcfg.get("cpu_offload", False))
    report_gpu(mcfg["device"], mcfg.get("gpus"))
    tok_cfg = dict(sink_positions=cfg["tokens"]["sink_positions"],
                   instruction_only=cfg["tokens"]["instruction_only"])

    from PIL import Image
    rng = np.random.default_rng(0)
    ia = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    ib = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    p0, _ = TI.build_prompt("dummy instruction")
    vspan = TI.probe_visual_span(
        vla,
        vla.processor(p0, ia).to(vla.device, dtype=vla.dtype),
        vla.processor(p0, ib).to(vla.device, dtype=vla.dtype),
    )
    v0, v1 = vspan
    print(f"[visual span] {vspan}")

    stem = f"{args.suite}{'_' + args.tag if args.tag else ''}"
    out_dir = OP.out_dir(OP.EMBED_SUB, args.tag)

    # ---------------- 1) 롤아웃: 프레임 수집 ----------------
    print("\n[1/3] 롤아웃 — 프레임 수집 (02 와 같은 정책·같은 규칙)")
    instrs: dict[int, str] = {}
    frames: dict[tuple[int, int, int], Image.Image] = {}
    success: dict[tuple[int, int], bool] = {}
    for tid in args.tasks:
        task = EL.make_task(args.suite, tid)
        instrs[tid] = task.instruction
        for ep in range(args.episodes):
            EL.reset_to(task, ep)
            obs = EL.step_noop(task, 10)
            done, n_got = False, 0
            for t in range(args.max_steps):
                if t % args.stride == 0:
                    frames[(tid, ep, t)] = EL.obs_to_image(obs)
                    n_got += 1
                action = P.policy_action(vla, EL.obs_to_image(obs), task.instruction, mcfg["unnorm_key"])
                obs, _, done, _ = task.env.step(action.tolist())
                if done:
                    break
            success[(tid, ep)] = episode_success(task, done)
            print(f"  task {tid} ep {ep}: 프레임 {n_got}개, 성공={success[(tid, ep)]}")
        task.env.close()
    sr = np.mean(list(success.values()))
    print(f"  성공률 {sum(success.values())}/{len(success)} = {sr:.1%}   총 프레임 {len(frames)}")

    # ---------------- 2) 기준값 ----------------
    print("\n[2/3] 기준값 계산")
    try:
        suite_instrs = EL.suite_instructions(args.suite)
        src = f"suite 전체 {len(suite_instrs)}개 지시문"
    except Exception as e:  # noqa: BLE001
        suite_instrs = [instrs[t] for t in args.tasks]
        src = f"!! suite 목록을 못 읽어 실행한 task {len(suite_instrs)}개만 사용 ({e})"
    L_mean, L_ids = SB.language_mean(vla, suite_instrs, vspan, **tok_cfg)
    E = vla.language_model.get_input_embeddings().weight
    L_norm_avg = float(E[torch.tensor(L_ids, device=E.device)].float().norm(dim=-1).mean())
    print(f"  L mean: {src}, 토큰 {len(L_ids)}개  ‖mean‖={L_mean.norm():.4f}  "
          f"원래 평균 ‖e‖={L_norm_avg:.4f}  (비 {L_mean.norm() / L_norm_avg:.3f})")

    V_sum = torch.zeros(v1 - v0, E.shape[1], dtype=torch.float64)
    V_norm_sum = 0.0
    for (tid, ep, t), img in sorted(frames.items()):
        prompt, _ = TI.build_prompt(instrs[tid])
        H = SB.record_layer0(vla, TI.prepare_inputs(vla, prompt, img))
        V_sum += H[v0:v1].double()
        V_norm_sum += float(H[v0:v1].norm(dim=-1).mean())
    V_mean = (V_sum / len(frames)).float()
    V_norm_avg = V_norm_sum / len(frames)
    print(f"  V mean: 프레임 {len(frames)}개의 위치별 평균  평균 ‖mean_p‖={V_mean.norm(dim=-1).mean():.4f}  "
          f"원래 평균 ‖v‖={V_norm_avg:.4f}  (비 {V_mean.norm(dim=-1).mean() / V_norm_avg:.3f})")
    torch.save({"L_mean": L_mean, "V_mean": V_mean, "L_token_ids": L_ids, "L_source": src,
                "visual_span": vspan, "n_frames": len(frames)},
               os.path.join(out_dir, f"baselines_{stem}.pt"))

    # ---------------- 3) 분석 ----------------
    print("\n[3/3] 분석 — 프레임마다 knockout 3 + 치환 최대 7 조건")
    rows, gate_log = [], []
    for fi, ((tid, ep, t), img) in enumerate(sorted(frames.items())):
        prompt, instr_span, inputs, spans = prepare(vla, img, instrs[tid], vspan, cfg)
        L, V = spans.language, spans.visual
        base_ids = cap.generate_action_tokens(vla, inputs, n_action=7)

        rec = SB.Layer0Recorder()
        c0 = cap.teacher_forced_capture(vla, inputs, base_ids, spans.query, attn_mask_fn=rec.as_fn())
        H = rec.hidden
        # G1 — 정렬
        full_ids = torch.cat([inputs["input_ids"][0].cpu(), base_ids.cpu()])
        with torch.no_grad():
            text_emb = E[full_ids.to(E.device)].float().cpu()
        SB.check_alignment(H, text_emb, vspan)
        r0 = M.compute_ratios(c0.attn, c0.vnorm, V, L, spans.sink)

        def run(hook):
            return cap.teacher_forced_capture(vla, inputs, base_ids, spans.query, attn_mask_fn=hook.as_fn())

        # G2·G3 — 항등·바닥값
        if fi < args.gate_frames:
            c_rep = cap.teacher_forced_capture(vla, inputs, base_ids, spans.query)
            kl_rep = kl(c0, c_rep)
            pos = L + V
            c_id = run(SB.EmbeddingSubstitution(pos, H[pos]))
            kl_id = kl(c0, c_id)
            tol = max(1e-6, 2 * kl_rep)
            gate_log.append(f"frame (task {tid}, ep {ep}, t {t}): G3 base 재실행 KL={kl_rep:.3g}  "
                            f"G2 항등치환 KL={kl_id:.3g}  (허용 {tol:.3g})")
            print("  [gate] " + gate_log[-1])
            if kl_id > tol:
                raise RuntimeError("[G2] 원래 임베딩으로 '치환' 했는데 출력이 바뀌었습니다. hook 구현을 확인하세요.")

        # knockout (02 와 동일)
        ko_c = IV.knockout_random_control(spans, seed=args.control_seed)
        c_kl_ = run(IV.knockout_language(spans))
        c_kv = run(IV.knockout_vision(spans))
        c_kc = run(ko_c)
        C = list(ko_c.key_idx)

        # mean 치환
        L_vec = L_mean.unsqueeze(0).expand(len(L), -1)
        V_vec = V_mean[[p - v0 for p in V]]
        C_vec = V_mean[[p - v0 for p in C]]
        c_sl = run(SB.EmbeddingSubstitution(L, L_vec))
        c_sv = run(SB.EmbeddingSubstitution(V, V_vec))
        c_sc = run(SB.EmbeddingSubstitution(C, C_vec))
        r_sl = M.compute_ratios(c_sl.attn, c_sl.vnorm, V, L, spans.sink)
        r_sv = M.compute_ratios(c_sv.attn, c_sv.vnorm, V, L, spans.sink)

        KL = {
            "KL_lang_knockout": kl(c0, c_kl_), "KL_vis_knockout": kl(c0, c_kv),
            "KL_control_knockout": kl(c0, c_kc),
            "KL_lang_sub": kl(c0, c_sl), "KL_vis_sub": kl(c0, c_sv), "KL_control_sub": kl(c0, c_sc),
        }
        row = {
            "suite": args.suite, "task_id": tid, "episode": ep, "t": t,
            "ep_success": success[(tid, ep)], "instruction": instrs[tid],
            "n_L": len(L), "n_V": len(V), "n_ctrl": len(C),
            "R_raw": float(r0.r_raw.mean()), "R_norm": float(r0.r_norm.mean()),
            "R_vnorm": float(r0.r_vnorm.mean()), "uniform_baseline": r0.uniform_baseline,
            **KL,
            "causal_ratio_ko": M.causal_ratio(KL["KL_lang_knockout"], KL["KL_vis_knockout"]),
            "causal_ratio_sub": M.causal_ratio(KL["KL_lang_sub"], KL["KL_vis_sub"]),
            "ratio_L_sub_ko": KL["KL_lang_sub"] / KL["KL_lang_knockout"] if KL["KL_lang_knockout"] > 0 else np.nan,
            "ratio_V_sub_ko": KL["KL_vis_sub"] / KL["KL_vis_knockout"] if KL["KL_vis_knockout"] > 0 else np.nan,
            "argmax_changed_lang_ko": n_changed(c0, c_kl_), "argmax_changed_vis_ko": n_changed(c0, c_kv),
            "argmax_changed_lang_sub": n_changed(c0, c_sl), "argmax_changed_vis_sub": n_changed(c0, c_sv),
            "argmax_changed_control_sub": n_changed(c0, c_sc),
            # 내용을 바꿔도 attention 이 그 '자리'로 계속 가는가 (H2 보조)
            "R_raw_under_Lsub": float(r_sl.r_raw.mean()), "R_raw_under_Vsub": float(r_sv.r_raw.mean()),
            "norm_L_orig": float(H[L].norm(dim=-1).mean()), "norm_V_orig": float(H[V].norm(dim=-1).mean()),
            "norm_L_mean": float(L_mean.norm()), "norm_V_mean": float(V_mean.norm(dim=-1).mean()),
        }

        if not args.no_rescaled:
            L_r = SB.rescale_to_norm(L_vec, row["norm_L_orig"])
            V_r = SB.rescale_to_norm(V_vec, row["norm_V_orig"])
            c_rl = run(SB.EmbeddingSubstitution(L, L_r))
            c_rv = run(SB.EmbeddingSubstitution(V, V_r))
            row["KL_lang_sub_rescaled"] = kl(c0, c_rl)
            row["KL_vis_sub_rescaled"] = kl(c0, c_rv)
            row["causal_ratio_sub_rescaled"] = M.causal_ratio(row["KL_lang_sub_rescaled"],
                                                              row["KL_vis_sub_rescaled"])

        if not args.no_segments:
            loc, shared = segment_positions(vla, prompt, instr_span, inputs["input_ids"][0].tolist(), spans, vspan)
            if loc and shared:
                c_lo = run(SB.EmbeddingSubstitution(loc, L_mean.unsqueeze(0).expand(len(loc), -1)))
                c_sh = run(SB.EmbeddingSubstitution(shared, L_mean.unsqueeze(0).expand(len(shared), -1)))
                row.update({"n_L_loc": len(loc), "n_L_shared": len(shared),
                            "KL_Lloc_sub": kl(c0, c_lo), "KL_Lshared_sub": kl(c0, c_sh)})

        rows.append(row)
        print(f"  task {tid} ep {ep} t {t:3d} | KO L={KL['KL_lang_knockout']:.2f} V={KL['KL_vis_knockout']:.2f} "
              f"C={KL['KL_control_knockout']:.2f} | SUB L={KL['KL_lang_sub']:.2f} V={KL['KL_vis_sub']:.2f} "
              f"C={KL['KL_control_sub']:.2f}"
              + (f" | loc={row['KL_Lloc_sub']:.2f} shared={row['KL_Lshared_sub']:.2f}" if "KL_Lloc_sub" in row else ""))

    df = pd.DataFrame(rows)
    out = os.path.join(out_dir, f"embed_sub_{stem}.csv")
    df.to_csv(out, index=False)
    print(f"\n저장: {out}  ({len(df)} rows)")

    # ---------------- 요약 ----------------
    g4_l = float((df["KL_lang_sub"] > df["KL_control_sub"]).mean())
    g4_v = float((df["KL_vis_sub"] > df["KL_control_sub"]).mean())
    gate_log.append("G1 정렬: 전 프레임 통과 (실패했다면 여기까지 오지 않음)")
    gate_log.append(f"G4 대조군: KL_lang_sub > KL_control_sub 인 행 {g4_l:.1%}, "
                    f"KL_vis_sub > KL_control_sub 인 행 {g4_v:.1%}")
    with open(os.path.join(out_dir, f"gates_{stem}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(gate_log) + "\n")
    print("\n--- 통과 기준 ---")
    for g in gate_log:
        print("  " + g)
    if g4_l < 0.5:
        print("  !! G4 미달: L 치환 효과가 같은 개수의 무작위 이미지 토큰 치환과 구별되지 않습니다.")

    cols = ["KL_lang_knockout", "KL_vis_knockout", "KL_control_knockout",
            "KL_lang_sub", "KL_vis_sub", "KL_control_sub",
            "KL_lang_sub_rescaled", "KL_vis_sub_rescaled", "KL_Lloc_sub", "KL_Lshared_sub",
            "causal_ratio_ko", "causal_ratio_sub", "causal_ratio_sub_rescaled",
            "R_raw", "R_raw_under_Lsub", "R_raw_under_Vsub"]
    print("\n--- 평균 ---")
    print(df[[c for c in cols if c in df.columns]].mean().round(4).to_string())

    print("\n--- 해석 규칙 1: 치환(총 효과) / knockout(직접 효과), 행 중앙값 [에피소드 bootstrap 95% CI] ---")
    for c in ["ratio_L_sub_ko", "ratio_V_sub_ko"]:
        med, (lo, hi) = cluster_bootstrap_median(df, c)
        print(f"  {c}: {med:.3f}  [{lo:.3f}, {hi:.3f}]")
    print("  <1: knockout 효과 일부가 재정규화로 인한 교란 / >1: 우회 경로 기여. V 쪽이 L 보다 크면 §직접·총 효과 비대칭")

    print("\n--- 해석 규칙 2: causal_ratio 일치도 ---")
    d = df[["task_id", "episode", "causal_ratio_ko", "causal_ratio_sub"]].dropna()
    diff = d["causal_ratio_sub"] - d["causal_ratio_ko"]
    df["_diff_cr"] = df["causal_ratio_sub"] - df["causal_ratio_ko"]
    med, (lo, hi) = cluster_bootstrap_median(df, "_diff_cr")
    within = [g["causal_ratio_ko"].corr(g["causal_ratio_sub"], method="spearman")
              for _, g in d.groupby(["task_id", "episode"]) if len(g) >= 5]
    within = [w for w in within if np.isfinite(w)]
    print(f"  평균 차이(sub−ko) {diff.mean():+.4f}, 중앙값 {med:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    if within:
        print(f"  에피소드 내 Spearman ρ 중앙값 {np.median(within):.3f}  (양수 {sum(w > 0 for w in within)}/{len(within)})")

    if "KL_Lloc_sub" in df.columns:
        print("\n--- 해석 규칙 3: 구간 치환 (spatial) ---")
        print(f"  위치구 L_loc: 평균 {df['n_L_loc'].mean():.1f}토큰, KL {df['KL_Lloc_sub'].mean():.3f}")
        print(f"  나머지 L_shared: 평균 {df['n_L_shared'].mean():.1f}토큰, KL {df['KL_Lshared_sub'].mean():.3f}")
        print(f"  KL_Lloc > KL_Lshared 인 행 {(df['KL_Lloc_sub'] > df['KL_Lshared_sub']).mean():.1%}"
              "  (토큰 수가 달라 크기 효과가 섞임)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
