"""Phase 0-c: OpenVLA 체크포인트 사전 다운로드.

7B bf16 ≈ 15GB. 실험 중에 받다가 끊기면 짜증나므로 미리 받아둡니다.

사용:
    python setup/03_download_ckpt.py --suite spatial
    python setup/03_download_ckpt.py --suite base      # 사전학습 원본
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="spatial", choices=sorted(CKPTS))
    ap.add_argument("--all", action="store_true", help="전부 다운로드 (~75GB)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    targets = list(CKPTS.values()) if args.all else [CKPTS[args.suite]]
    for repo in targets:
        print(f"→ 다운로드: {repo}")
        try:
            path = snapshot_download(repo_id=repo)
            print(f"  저장 위치: {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  실패: {exc!r}")
            print("  repo 이름이 바뀌었을 수 있습니다. huggingface.co/openvla 확인 요망.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
