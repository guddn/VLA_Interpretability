"""이미지 쪽 개입: 지시문은 그대로 두고 **이미지만** 바꾸면 행동이 얼마나 바뀌는가?

04_counterfactual.py 의 거울상입니다.
  04: 같은 장면, 지시문만 바꿈  →  "지시문 내용에 얼마나 의존하나"
  06: 같은 지시문, 이미지만 바꿈 →  "이미지 내용에 얼마나 의존하나"
두 스크립트는 같은 지표·같은 규약(KL_vs_valid, teacher-forcing prefix 공유)을 쓰므로
04 의 swapped KL 과 06 의 other_scene KL 을 같은 단위로 나란히 놓을 수 있습니다.

조건 (지시문은 모두 원래 task 의 지시문)
-----------------------------------------
  valid         : 원래 이미지 (기준)
  other_scene   : 같은 에피소드 번호·같은 시점 t 의 **다른 task 장면**
                  (04 swapped 의 이미지판. 물체 배치와 로봇 자세가 함께 바뀜)
  hflip         : 좌우 반전 — 왼쪽/오른쪽 관계를 뒤집음 (학습 때 본 적 없는 시점일 수 있음)
  patch_shuffle : block×block 타일을 섞음 — 색 분포는 그대로, 공간 구조만 파괴
  noise         : 균등 난수 이미지 — 이미지 정보 없음 (04 empty 의 이미지판, 분포 밖)

측정
----
  KL_vs_valid     : valid 대비 행동 분포 변화 (nats, 7 DoF 평균)
  n_identical_dof : valid 와 argmax bin 이 같은 DoF 수 (7 이면 이미지를 바꿔도 같은 행동)
  R_raw / R_norm / R_vnorm : 이미지를 바꾸면 attention 비율이 움직이는가

진행 방식 (2단계)
-----------------
  1) 롤아웃 패스: 각 (task, episode) 를 정책으로 굴리며 --steps 시점의 이미지를 저장
     (분석은 하지 않으므로 빠름. 정책 궤적은 02/04 와 같은 규칙으로 만들어짐)
  2) 분석 패스: 저장한 이미지로 조건별 캡처. 다른 task 의 같은 시점 이미지를 바로 꺼내 쓸 수 있음

!! 정책 궤적은 GPU 종류에 따라 달라집니다(bf16 반올림). 04 와 **같은 GPU** 로 돌리면
   두 스크립트의 장면이 (task, episode, t) 단위로 같아져서 장면별로 짝지어 비교할 수 있습니다.

사용:
    python scripts/06_image_ablation.py --gpu 4 --suite spatial --tasks 0 1 2 3 4 --episodes 2
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
from vlamod import metrics as M  # noqa: E402
from vlamod import pipeline as P  # noqa: E402
from vlamod import preprocess as PP  # noqa: E402
from vlamod import token_index as TI  # noqa: E402
from vlamod.model_loader import load_openvla  # noqa: E402

ALL_CONDITIONS = ["valid", "other_scene", "hflip", "patch_shuffle", "noise"]


@torch.no_grad()
def one_image(vla, image, instruction, vspan, cfg, force_ids=None):
    """이미지 하나에 대해 (capture, ratios, action_ids) 반환.

    04_counterfactual.one_condition 과 같은 절차입니다 (지시문은 항상 비어 있지 않음).
    force_ids: valid 의 action token 을 넘기면 그 토큰으로 teacher-forcing (prefix 공유).
    """
    prompt, span = TI.build_prompt(instruction)
    inputs = TI.prepare_inputs(vla, prompt, image)   # 29871 포함 — 정책과 같은 입력
    spans = TI.build_spans(
        vla, prompt, span, inputs["input_ids"][0], vspan,
        n_action_tokens=7,
        sink_positions=cfg["tokens"]["sink_positions"],
        instruction_only=cfg["tokens"]["instruction_only"],
    )
    ids = force_ids if force_ids is not None else cap.generate_action_tokens(vla, inputs, n_action=7)
    c = cap.teacher_forced_capture(vla, inputs, ids, spans.query)
    r = M.compute_ratios(c.attn, c.vnorm, spans.visual, spans.language, spans.sink)
    return c, r, ids


def scene_seed(tid: int, ep: int, t: int) -> int:
    """장면마다 다르지만 재실행하면 같은 난수 seed (patch_shuffle / noise 용)."""
    return tid * 10_000 + ep * 1_000 + t


def make_condition_image(name, valid_img, other_img, block, seed):
    if name == "valid":
        return valid_img
    if name == "other_scene":
        return other_img                      # 없으면 None → 호출한 쪽에서 건너뜀
    if name == "hflip":
        return PP.hflip(valid_img)
    if name == "patch_shuffle":
        return PP.patch_shuffle(valid_img, block=block, seed=seed)
    if name == "noise":
        return PP.noise_image(valid_img.size, seed=seed)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 25, 40])
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS,
                    help="돌릴 조건. valid 는 항상 포함됩니다")
    ap.add_argument("--block", type=int, default=32, help="patch_shuffle 타일 크기 (이미지 256 의 약수)")
    ap.add_argument("--no-save-examples", action="store_true",
                    help="조건별 예시 PNG(episode 0, 첫 시점) 저장을 끕니다")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--gpu", type=int, default=None, metavar="N")
    ap.add_argument("--headroom-gb", type=float, default=None, metavar="G")
    ap.add_argument("--allow-cpu-offload", action="store_true")
    ap.add_argument("--egl-device", type=int, default=None, metavar="E")
    ap.add_argument("--gpus", default=None, metavar="0,1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--prefix-mode", choices=["shared", "own"], default="shared",
                    help="shared(기본)=valid 의 action token 을 전 조건 공유 → 02·04 와 같은 규약")
    ap.add_argument("--tag", default=None, help="출력 파일명 접미사")
    args = ap.parse_args()

    conditions = ["valid"] + [c for c in args.conditions if c != "valid"]
    if "other_scene" in conditions and len(args.tasks) < 2:
        print("!! task 가 1개라 other_scene 을 만들 수 없습니다. 이 조건은 건너뜁니다.")
        conditions.remove("other_scene")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    apply_env(cfg, args)          # ← transformers/libero import 전에 반드시 먼저
    mcfg = cfg["model"]
    apply_overrides(mcfg, args)
    vla = load_openvla(path=mcfg["path"], device=mcfg["device"], dtype=mcfg["dtype"],
                       attn_implementation=mcfg["attn_implementation"],
                       gpus=mcfg.get("gpus"),
                       headroom_gb=mcfg.get("headroom_gb", 1.5),
                       cpu_offload=mcfg.get("cpu_offload", False))
    report_gpu(mcfg["device"], mcfg.get("gpus"))

    from PIL import Image
    rng = np.random.default_rng(0)
    ia = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    ib = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    p0, _ = TI.build_prompt("dummy")
    vspan = TI.probe_visual_span(
        vla,
        vla.processor(p0, ia).to(vla.device, dtype=vla.dtype),
        vla.processor(p0, ib).to(vla.device, dtype=vla.dtype),
    )

    # ---------------- 1) 롤아웃 패스: 시점별 이미지 저장 ----------------
    print("\n[1/2] 롤아웃 — 시점별 이미지 수집")
    instrs: dict[int, str] = {}
    frames: dict[tuple[int, int, int], Image.Image] = {}
    max_t = max(args.steps)
    for tid in args.tasks:
        task = EL.make_task(args.suite, tid)
        instrs[tid] = task.instruction
        for ep in range(args.episodes):
            EL.reset_to(task, ep)
            obs = EL.step_noop(task, 10)
            got = []
            for t in range(max_t + 1):
                if t in args.steps:
                    frames[(tid, ep, t)] = EL.obs_to_image(obs)
                    got.append(t)
                action = P.policy_action(vla, EL.obs_to_image(obs), task.instruction, mcfg["unnorm_key"])
                obs, _, done, _ = task.env.step(action.tolist())
                if done:
                    break
            print(f"  task {tid} ep {ep}: 수집 시점 {got}")
        task.env.close()

    # other_scene 짝: args.tasks 안에서 다음 task (순환)
    order = list(args.tasks)
    partner = {tid: order[(i + 1) % len(order)] for i, tid in enumerate(order)}
    print("\nother_scene 짝 (이 task 의 지시문 + 짝 task 의 장면):")
    for tid in order:
        print(f"  [{tid}] ← 장면 [{partner[tid]}]")

    ex_dir = "outputs/image_ablation_examples"
    if not args.no_save_examples:
        os.makedirs(ex_dir, exist_ok=True)

    # ---------------- 2) 분석 패스 ----------------
    print("\n[2/2] 분석 — 조건별 캡처")
    rows = []
    for (tid, ep, t), valid_img in sorted(frames.items()):
        other_img = frames.get((partner[tid], ep, t))
        seed = scene_seed(tid, ep, t)
        base, base_ids = None, None
        for name in conditions:
            img = make_condition_image(name, valid_img, other_img, args.block, seed)
            if img is None:
                print(f"  (task {tid} ep {ep} t {t}) {name}: 짝 장면이 없어 건너뜀")
                continue
            force = base_ids if (args.prefix_mode == "shared" and name != "valid") else None
            c, r, ids = one_image(vla, img, instrs[tid], vspan, cfg, force_ids=force)
            if name == "valid":
                base, base_ids = c, ids
            kl = float(M.action_kl(base.action_logits, c.action_logits))
            same = int((M.decode_actions(base.action_logits)
                        == M.decode_actions(c.action_logits)).sum())
            rows.append({
                "suite": args.suite, "task_id": tid, "episode": ep, "t": t,
                "condition": name,
                "instruction": instrs[tid],
                "scene_task": partner[tid] if name == "other_scene" else tid,
                "R_raw": float(r.r_raw.mean()),
                "R_norm": float(r.r_norm.mean()),
                "R_vnorm": float(r.r_vnorm.mean()),
                "uniform_baseline": r.uniform_baseline,
                "mass_sink": float(r.mass_sink.mean()),
                "KL_vs_valid": kl,
                "n_identical_dof": same,
                "prefix_mode": args.prefix_mode,
            })
            if not args.no_save_examples and ep == 0 and t == min(args.steps):
                img.save(f"{ex_dir}/task{tid}_ep{ep}_t{t}_{name}.png")
        print(f"  task {tid} ep {ep} t {t:3d}: "
              + "  ".join(f"{rw['condition']}={rw['KL_vs_valid']:.2f}"
                          for rw in rows if (rw['task_id'], rw['episode'], rw['t']) == (tid, ep, t)))

    df = pd.DataFrame(rows)
    stem = f"{args.suite}{'_' + args.tag if args.tag else ''}"
    out = f"outputs/image_ablation_{stem}.csv"
    df.to_csv(out, index=False)
    print(f"\n저장: {out}  ({len(df)} rows)")
    if not args.no_save_examples:
        print(f"예시 이미지: {ex_dir}/  ← 조건별로 이미지가 의도대로 바뀌었는지 눈으로 확인하세요")
    print(f"teacher-forcing prefix 규약: {args.prefix_mode}"
          + ("  (02·04 와 동일 — KL 비교 가능)" if args.prefix_mode == "shared" else ""))

    print("\n--- 조건별 요약 ---")
    print(df.groupby("condition")[["R_raw", "R_norm", "R_vnorm", "KL_vs_valid",
                                   "n_identical_dof"]].mean().reindex(conditions).round(4).to_string())

    # 04 결과가 있으면 지시문 개입과 나란히 보여줌
    cf_path = f"outputs/counterfactual_{stem}.csv"
    if os.path.exists(cf_path):
        cf = pd.read_csv(cf_path)
        g = cf.groupby("condition")["KL_vs_valid"].mean()
        h = df.groupby("condition")["KL_vs_valid"].mean()
        print("\n--- 지시문 개입(04) vs 이미지 개입(06): 평균 KL_vs_valid ---")
        for a, b in [("swapped", "other_scene"), ("empty", "noise"), ("paraphrase", "hflip")]:
            if a in g and b in h:
                print(f"  지시문 {a:<10s} {g[a]:7.3f}   |   이미지 {b:<13s} {h[b]:7.3f}")
        print("  (paraphrase↔hflip 은 '의미 유지 vs 공간 반전'이라 엄밀한 짝은 아님)")
        print("  !! 04 와 06 을 다른 GPU 로 돌렸다면 장면이 달라 평균끼리만 비교 가능합니다.")
    return 0


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    sys.exit(main())
