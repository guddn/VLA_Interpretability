"""출력 폴더 규약 — 스크립트 번호별 하위 폴더.

    outputs/
      01_smoke_forward/      smoke_libero_view.png, smoke_capture.pt
      02_run_analysis/       analysis_<suite>.csv, perlayer_<suite>.npz
        <tag>/               --tag 를 준 실행은 하위 폴더로 (파일명에도 tag 유지)
      03_correlation/        correlation.csv
      04_counterfactual/     counterfactual_<suite>.csv
      05_plots/              fig_*.png
      06_image_ablation/     image_ablation_<suite>.csv, examples/
      07_dump_model_input/   input_*.png
      setup_04_verify/       verify_render.png
      setup_05_diag_render/  diag_render.png

모든 경로는 **현재 작업 디렉터리 기준 상대경로**입니다 (기존과 동일하게 저장소 루트에서 실행).
"""

from __future__ import annotations

import os

OUTPUTS = "outputs"

SMOKE = "01_smoke_forward"
ANALYSIS = "02_run_analysis"
CORRELATION = "03_correlation"
COUNTERFACTUAL = "04_counterfactual"
PLOTS = "05_plots"
IMAGE_ABLATION = "06_image_ablation"
MODEL_INPUT = "07_dump_model_input"
SETUP_VERIFY = "setup_04_verify"
SETUP_DIAG = "setup_05_diag_render"


def out_dir(script: str, tag: str | None = None, *sub: str, make: bool = True) -> str:
    """outputs/<script>[/<tag>][/<sub>...] — make=True 면 폴더를 만듭니다."""
    parts = [OUTPUTS, script]
    if tag:
        parts.append(tag)
    parts.extend(sub)
    path = os.path.join(*parts)
    if make:
        os.makedirs(path, exist_ok=True)
    return path


def out_path(script: str, filename: str, tag: str | None = None, make: bool = True) -> str:
    """outputs/<script>[/<tag>]/<filename>"""
    return os.path.join(out_dir(script, tag, make=make), filename)


def find_input(script: str, filename: str, tag: str | None = None) -> str:
    """다른 스크립트의 출력을 읽을 때 쓰는 경로.

    새 위치(outputs/<script>[/<tag>]/<filename>)를 먼저 보고, 없으면 폴더 정리 전의
    평평한 위치(outputs/<filename>)를 봅니다. 옛 위치를 쓰게 되면 경고를 찍습니다.
    둘 다 없으면 새 위치를 그대로 돌려줍니다 (호출 쪽의 exists 검사가 처리).
    """
    new = out_path(script, filename, tag, make=False)
    if os.path.exists(new):
        return new
    legacy = os.path.join(OUTPUTS, filename)
    if os.path.exists(legacy):
        print(f"!! {new} 가 없어 옛 위치 {legacy} 를 읽습니다. "
              f"폴더 정리 전 파일일 수 있으니 확인하세요.")
        return legacy
    return new
