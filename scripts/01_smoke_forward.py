"""Phase 0~1 스모크 테스트. **가장 먼저 이걸 통과시키세요.**

확인하는 것:
  1) eager attention 으로 로드되어 attentions 가 실제로 나오는가
  2) visual 토큰이 몇 개이고 어디에 삽입되는가 (실측)
  3) 언어 구간 L 이 instruction 단어들과 정확히 대응하는가 (눈으로 확인)
  4) 생성된 action token 이 action bin id 범위 안에 있는가
  5) 3종 비율 지표가 계산되는가

사용:
    python scripts/01_smoke_forward.py --config configs/default.yaml
    python scripts/01_smoke_forward.py --libero            # 실제 LIBERO 관측 사용
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod.device import apply_env, apply_overrides, report_gpu  # noqa: E402
from vlamod import capture as cap  # noqa: E402
from vlamod import metrics as M  # noqa: E402
from vlamod import token_index as TI  # noqa: E402
from vlamod.model_loader import action_bin_token_ids, load_openvla  # noqa: E402


def synth_image(seed: int, size: int = 256) -> Image.Image:
    """LIBERO 없이도 구조 검증이 가능하도록 하는 합성 이미지.
    (내용 분석용이 아니라 '두 이미지가 다르다'만 필요할 때 씀)"""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--instruction", default="pick up the black bowl and place it on the plate")
    ap.add_argument("--libero", action="store_true", help="LIBERO 관측으로 테스트")
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--hf-home", default=None,
                    help="HuggingFace 캐시 루트. config 의 env.hf_home 을 덮어씀")
    ap.add_argument("--gpu", type=int, default=None, metavar="N",
                    help="사용할 GPU 번호 (예: --gpu 3)")
    ap.add_argument("--gpus", default=None, metavar="0,1",
                    help="여러 GPU 에 모델을 분할 로드 (예: --gpus 0,1). "
                         "한 장에 안 들어갈 때 씁니다. 수치는 단일 GPU 와 동일합니다.")
    ap.add_argument("--device", default=None,
                    help='--gpu 대신 문자열로 지정. "cuda:3" / "3" / "cpu"')
    ap.add_argument("--model", default=None, help="체크포인트 경로/HF repo. config 값을 덮어씀")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    apply_env(cfg, args)          # ← transformers/libero import 전에 반드시 먼저
    mcfg = cfg["model"]
    apply_overrides(mcfg, args)

    print("=" * 70)
    print("[1] 모델 로딩")
    vla = load_openvla(
        path=mcfg["path"],
        device=mcfg["device"],
        dtype=mcfg["dtype"],
        attn_implementation=mcfg["attn_implementation"],
        gpus=mcfg.get("gpus"),
    )
    report_gpu(mcfg["device"], mcfg.get("gpus"))
    print(f"  path      : {vla.path}")
    print(f"  layers    : {vla.n_layers}, heads: {vla.n_heads}")
    print(f"  vocab_size: {vla.tokenizer.vocab_size}")

    # ---- 이미지 준비 --------------------------------------------------
    instruction = args.instruction
    if args.libero:
        from vlamod import env_libero as EL

        print("\n[1b] LIBERO 관측 취득")
        task = EL.make_task(args.suite, args.task_id)
        instruction = task.instruction
        EL.reset_to(task, 0)
        obs = EL.step_noop(task, 10)
        img_a = EL.obs_to_image(obs)
        obs = EL.step_noop(task, 20)
        img_b = EL.obs_to_image(obs)
        print(f"  instruction: {instruction!r}")
        img_a.save("outputs/smoke_libero_view.png")
        print("  → outputs/smoke_libero_view.png 를 열어서 **상하가 맞는지** 확인하세요.")
    else:
        img_a, img_b = synth_image(0), synth_image(1)

    # ---- 프롬프트 -----------------------------------------------------
    prompt, instr_span = TI.build_prompt(instruction)
    print("\n[2] 프롬프트")
    print(f"  {prompt!r}")
    print(f"  instruction 문자 구간: {instr_span} → {prompt[instr_span[0]:instr_span[1]]!r}")

    proc = vla.processor
    inputs_a = proc(prompt, img_a).to(vla.device, dtype=vla.dtype)
    inputs_b = proc(prompt, img_b).to(vla.device, dtype=vla.dtype)

    # ---- visual span 실측 ---------------------------------------------
    print("\n[3] visual 토큰 구간 실측 (이미지만 바꿔서 embedding 차이 확인)")
    v_start, v_end = TI.probe_visual_span(vla, inputs_a, inputs_b)
    print(f"  visual span = [{v_start}, {v_end})  →  n_visual = {v_end - v_start}")
    if v_start != 1:
        print(f"  !! 주의: BOS 직후(1)가 아니라 {v_start} 에서 시작합니다. 구조 재확인 필요.")
    if (v_end - v_start) != 256:
        print(f"  !! 주의: visual 토큰이 256개가 아닙니다 ({v_end - v_start}). 문서 가정과 다릅니다.")

    # ---- 구간 나누기 ---------------------------------------------------
    print("\n[4] 토큰 구간 분해 — **아래 출력을 눈으로 검증하세요**")
    spans = TI.build_spans(
        vla,
        prompt=prompt,
        instr_char_span=instr_span,
        input_ids=inputs_a["input_ids"][0],
        visual_span=(v_start, v_end),
        n_action_tokens=7,
        sink_positions=cfg["tokens"]["sink_positions"],
        instruction_only=cfg["tokens"]["instruction_only"],
    )
    print(TI.pretty_print_spans(spans))
    print("\n  기대: L 태그가 instruction 의 내용어에만 붙어 있어야 합니다.")
    print("        'In', ':', 'What', 'action' 등에 L 이 붙어 있으면 인덱싱이 틀린 것입니다.")

    # ---- action token 생성 ---------------------------------------------
    print("\n[5] action token 생성 + bin 범위 검증")
    action_ids = cap.generate_action_tokens(vla, inputs_a, n_action=7)
    bins = action_bin_token_ids(vla.tokenizer)
    lo, hi = int(bins.min()), int(bins.max())
    print(f"  action token ids: {action_ids.tolist()}")
    print(f"  기대 bin id 범위: [{lo}, {hi}]")
    ok = bool(((action_ids >= lo) & (action_ids <= hi)).all())
    print(f"  범위 검증: {'통과' if ok else '실패 !! action_bin_token_ids 의 오프셋을 고치세요'}")

    # ---- 캡처 -----------------------------------------------------------
    print("\n[6] teacher-forced 캡처")
    c = cap.teacher_forced_capture(vla, inputs_a, action_ids, spans.action)
    print(f"  attn  : {tuple(c.attn.shape)}   (L, H, n_action, T)")
    print(f"  vnorm : {tuple(c.vnorm.shape)}")
    print(f"  logits: {tuple(c.action_logits.shape)}")
    print(f"  seq_len: {c.seq_len}  (= n_total_prompt {spans.n_total_prompt} + 7)")

    row_sum = c.attn.sum(dim=-1)
    print(f"  attention 행 합 (1.0 이어야 함): min={row_sum.min():.4f} max={row_sum.max():.4f}")

    # ---- 지표 -----------------------------------------------------------
    print("\n[7] 비율 지표")
    r = M.compute_ratios(c.attn, c.vnorm, spans.visual, spans.language, spans.sink)
    raw, norm, vn = r.mean_over_heads()
    print(f"  |L|={len(spans.language)}  |V|={len(spans.visual)}")
    print(f"  균등분포 baseline R_raw = {r.uniform_baseline:.4f}  ← 이보다 낮으면 언어를 '덜' 보는 것")
    print(f"  R_raw   전체평균 = {raw.mean():.4f}")
    print(f"  R_norm  전체평균 = {norm.mean():.4f}")
    print(f"  R_vnorm 전체평균 = {vn.mean():.4f}")
    print(f"  sink 질량 평균   = {r.mass_sink.mean():.4f}   ← 크면 attention sink 가 지배 중")
    print(f"  기타 질량 평균   = {r.mass_other.mean():.4f}")

    torch.save(
        {"spans": spans.__dict__, "ratios": {k: v for k, v in r.__dict__.items()}},
        "outputs/smoke_capture.pt",
    )
    print("\n저장: outputs/smoke_capture.pt")
    print("=" * 70)
    print("전부 통과했으면 scripts/02_attention_ratio.py 로 넘어가세요.")
    return 0 if ok else 1


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    sys.exit(main())
