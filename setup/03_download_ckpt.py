"""Phase 0-c: OpenVLA 체크포인트 사전 다운로드.

7B bf16 ≈ 15GB. 실험 중에 받다가 끊기면 짜증나므로 미리 받아둡니다.

저장 위치
---------
기본값은 HuggingFace 캐시(`~/.cache/huggingface`)입니다. 홈 파티션이 작은
랩 서버에서는 터지므로 `--cache-dir` 로 큰 디스크를 지정하세요.

**기본값은 `configs/default.yaml` 의 `env.hf_home` 입니다.**
따라서 보통은 인자 없이 이것만 하면 됩니다:

    python setup/03_download_ckpt.py --suite spatial

경로를 바꾸려면 `configs/default.yaml` 의 `env.hf_home` 한 줄만 고치세요.
다운로드와 모델 로드가 **같은 값**을 읽으므로 .bashrc 를 만질 필요가 없습니다.
일회성으로 덮어쓰려면 `--cache-dir` 를 주면 됩니다.

실제 모델 파일은 `<hf_home>/hub/models--openvla--...` 아래에 들어갑니다.

`--local-dir` 을 쓰면 캐시 구조 대신 평범한 폴더로 받습니다. 그 경우
분석 스크립트에 `--model <그 폴더 경로>` 로 직접 경로를 넘겨야 합니다.
"""

from __future__ import annotations

import argparse
import os
import sys

# 2026-08 기준 HF Hub 상의 OpenVLA 체크포인트.
# !! 반드시 huggingface.co/openvla 에서 실제 존재를 확인하세요.
#    repo 이름은 바뀔 수 있습니다.
CKPTS = {
    "base": "openvla/openvla-7b",
    "spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "object": "openvla/openvla-7b-finetuned-libero-object",
    "goal": "openvla/openvla-7b-finetuned-libero-goal",
    "long": "openvla/openvla-7b-finetuned-libero-10",
}


def _free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="spatial", choices=sorted(CKPTS))
    ap.add_argument("--all", action="store_true", help="전부 다운로드 (~75GB)")
    ap.add_argument(
        "--config",
        default="configs/default.yaml",
        help="여기서 env.hf_home 을 읽습니다. --cache-dir 이 없으면 이 값을 씁니다.",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="HF 캐시 루트. config 의 env.hf_home 을 덮어씀",
    )
    ap.add_argument(
        "--local-dir",
        default=None,
        help="캐시 구조 대신 평범한 폴더로 받기. 이 경우 --model 에 경로를 직접 넘겨야 합니다.",
    )
    args = ap.parse_args()

    # --- 캐시 경로 설정 : huggingface_hub import 전에 해야 적용됩니다 ---
    # 우선순위: --cache-dir > config 의 env.hf_home > 기존 HF_HOME > 시스템 기본
    cache_dir = args.cache_dir
    if cache_dir is None and os.path.exists(args.config):
        try:
            import yaml
            cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
            cache_dir = (cfg.get("env") or {}).get("hf_home")
            if cache_dir:
                print(f"[config] env.hf_home = {cache_dir}   ({args.config})")
        except Exception as e:  # noqa: BLE001
            print(f"  (config 읽기 실패, 무시: {e!r})")

    if cache_dir:
        root = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(root, exist_ok=True)
        os.environ["HF_HOME"] = root
        os.environ["HF_HUB_CACHE"] = os.path.join(root, "hub")
        os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_cache = os.environ.get("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    target_for_space = os.path.abspath(os.path.expanduser(args.local_dir or hub_cache))
    os.makedirs(target_for_space, exist_ok=True)

    from huggingface_hub import snapshot_download

    targets = list(CKPTS.values()) if args.all else [CKPTS[args.suite]]
    need_gb = 15.0 * len(targets)
    free = _free_gb(target_for_space)

    print("=" * 66)
    print(f"저장 위치 : {target_for_space}")
    print(f"여유 공간 : {free:.1f}GB   /   예상 필요: 약 {need_gb:.0f}GB")
    print(f"받을 대상 : {', '.join(targets)}")
    print("=" * 66)

    if free < need_gb * 1.1:
        print("!! 공간이 부족할 수 있습니다. --cache-dir 로 더 큰 디스크를 지정하거나")
        print("!! --all 대신 --suite 하나만 받으세요.")
        if free < need_gb:
            return 1

    for repo in targets:
        print(f"\n→ 다운로드: {repo}")
        try:
            kwargs = {"repo_id": repo}
            if args.local_dir:
                kwargs["local_dir"] = os.path.join(
                    os.path.abspath(os.path.expanduser(args.local_dir)), repo.split("/")[-1]
                )
            path = snapshot_download(**kwargs)
            print(f"  저장 위치: {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  실패: {exc!r}")
            print("  repo 이름이 바뀌었을 수 있습니다. huggingface.co/openvla 확인 요망.")
            return 1

    print("\n" + "=" * 66)
    if cache_dir:
        print(f"캐시 루트: {cache_dir}")
        print("분석 스크립트도 같은 config 의 env.hf_home 을 읽으므로 추가 설정은 필요 없습니다.")
        print("(다른 경로를 쓰려면 configs/*.yaml 의 env.hf_home 만 바꾸세요)")
    if args.local_dir:
        print("!! --local-dir 로 받았으므로 분석 스크립트에 경로를 직접 넘기세요:")
        print(f"     python scripts/01_smoke_forward.py --gpu 0 --model {kwargs['local_dir']}")
    print("다음: python scripts/01_smoke_forward.py --gpu 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
