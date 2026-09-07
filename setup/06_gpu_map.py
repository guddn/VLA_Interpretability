"""GPU 번호가 어느 물리 카드를 가리키는지 **확정**합니다.

    python setup/06_gpu_map.py

네 가지를 나란히 놓습니다:

  [1] nvidia-smi / nvtop     — PCI 버스 순서 (사람이 보는 번호). **기준**
  [2] torch (현재 환경)      — 우리 스크립트가 실제로 보게 될 번호. **판정 대상**
  [3] torch 강제 FASTEST_FIRST — 설정을 안 걸었을 때 어떻게 어긋나는지 (대조군)
  [4] torch 강제 PCI_BUS_ID    — 설정을 걸면 어떻게 되는지 (대조군)

판정은 **[1] 과 [2] 를 비교**해서 냅니다.
[3] 과 [4] 는 항상 다릅니다 — 그건 정상이고, 문제의 존재를 보여줄 뿐입니다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

SNIPPET = (
    "import torch,json;"
    "print(json.dumps([[i,torch.cuda.get_device_name(i),"
    "round(torch.cuda.get_device_properties(i).total_memory/1e9,1)]"
    " for i in range(torch.cuda.device_count())]))"
)


def torch_list(order: str | None | object = "INHERIT") -> list[list]:
    """별도 프로세스에서 torch 를 띄웁니다.

    order="INHERIT" → 현재 환경 그대로
    order=None      → CUDA_DEVICE_ORDER 를 지우고 (= 시스템 기본 FASTEST_FIRST)
    order="..."     → 그 값으로 강제

    CUDA_DEVICE_ORDER 는 CUDA 런타임 초기화 때 한 번만 읽히므로
    한 프로세스 안에서 두 번 바꿔 볼 수 없습니다. 그래서 서브프로세스를 씁니다.
    """
    env = dict(os.environ)
    if order != "INHERIT":
        if order:
            env["CUDA_DEVICE_ORDER"] = str(order)
        else:
            env.pop("CUDA_DEVICE_ORDER", None)
    r = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True,
                       text=True, env=env, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-800:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def smi_list() -> list[list]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,pci.bus_id",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            out.append([int(parts[0]), parts[1], parts[2], parts[3]])
    return out


def _short(name: str) -> str:
    """'NVIDIA GeForce RTX 3090' → 'RTX 3090' (비교/표시용)"""
    return name.replace("NVIDIA ", "").replace("GeForce ", "").strip()


def _show(rows, label_fmt) -> list[str]:
    names = []
    for row in rows:
        i, name = row[0], row[1]
        names.append(_short(name))
        print("    " + label_fmt(i, row))
    return names


def main() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    order_now = os.environ.get("CUDA_DEVICE_ORDER")
    print("=" * 74)
    print(f"CUDA_VISIBLE_DEVICES = {cvd if cvd else '(설정 안 됨)'}")
    print(f"CUDA_DEVICE_ORDER    = {order_now or '(설정 안 됨 → FASTEST_FIRST)'}")
    print("=" * 74)

    print("\n[1] nvidia-smi — PCI 순서. nvtop 과 같은 번호. ★기준")
    try:
        smi = smi_list()
        n_smi = _show(smi, lambda i, r: f"GPU {i}: {_short(r[1]):18s} {r[2]:>10s}  {r[3]}")
    except Exception as e:  # noqa: BLE001
        smi, n_smi = [], []
        print(f"    조회 실패: {e}")

    print("\n[2] torch — 현재 환경 그대로. ★우리 스크립트가 보게 될 번호")
    try:
        cur = torch_list("INHERIT")
        n_cur = _show(cur, lambda i, r: f"cuda:{i}: {_short(r[1]):18s} {r[2]:>6.1f} GB")
    except Exception as e:  # noqa: BLE001
        cur, n_cur = [], []
        print(f"    조회 실패: {e}")

    print("\n[3] (참고) torch 강제 FASTEST_FIRST — 설정을 안 걸면 이렇게 됩니다")
    try:
        fast = torch_list(None)
        n_fast = _show(fast, lambda i, r: f"cuda:{i}: {_short(r[1]):18s} {r[2]:>6.1f} GB")
    except Exception as e:  # noqa: BLE001
        fast, n_fast = [], []
        print(f"    조회 실패: {e}")

    print("\n[4] (참고) torch 강제 PCI_BUS_ID")
    try:
        pci = torch_list("PCI_BUS_ID")
        n_pci = _show(pci, lambda i, r: f"cuda:{i}: {_short(r[1]):18s} {r[2]:>6.1f} GB")
    except Exception as e:  # noqa: BLE001
        pci, n_pci = [], []
        print(f"    조회 실패: {e}")

    # ================================================================ 판정
    print("\n" + "=" * 74)
    print("판정 — [1] nvidia-smi 와 [2] 현재 환경을 비교합니다")
    print("=" * 74)

    if not (n_smi and n_cur):
        print("  판정 불가 (조회 실패한 항목이 있습니다)")
        return 1

    if len(n_smi) != len(n_cur):
        print(f"★ 보이는 GPU 수가 다릅니다: nvidia-smi {len(n_smi)}장, torch {len(n_cur)}장")
        print("  CUDA_VISIBLE_DEVICES 로 일부만 노출된 상태입니다.")
    elif n_smi == n_cur:
        print("✓ 일치합니다. **nvtop 에서 본 번호를 그대로 --gpu 에 쓰시면 됩니다.**")
        print(f"  (현재 CUDA_DEVICE_ORDER = {order_now or 'FASTEST_FIRST'})")
        if not order_now:
            print("  !! 다만 설정 없이 우연히 맞은 것일 수 있습니다. "
                  "configs 의 env.cuda_device_order 를 그대로 두세요.")
    else:
        print("★ 어긋나 있습니다. nvtop 을 보고 --gpu 를 고르면 다른 카드를 잡습니다.")
        print("\n  대응 (택 1):")
        print("    A) configs/default.yaml 의 env.cuda_device_order: \"PCI_BUS_ID\"")
        print("       (apply_env() 가 CUDA 초기화 전에 걸어 줍니다 — 이게 기본값입니다)")
        print("    B) export CUDA_DEVICE_ORDER=PCI_BUS_ID")
        print("\n  현재 매핑 (nvtop 번호 → 지금 torch 번호):")
        used = set()
        for i, nm in enumerate(n_smi):
            for j, nm2 in enumerate(n_cur):
                if nm2 == nm and j not in used:
                    used.add(j)
                    print(f"    nvtop {i} ({nm:12s})  →  cuda:{j}"
                          + ("" if i == j else "   ← 다름"))
                    break

    # 참고 정보: [3] vs [4] 는 항상 다릅니다 (문제의 존재 증명일 뿐)
    if n_fast and n_pci and n_fast != n_pci:
        print("\n(참고) 이 서버는 카드 모델이 섞여 있어 FASTEST_FIRST 와 PCI_BUS_ID 가 다릅니다.")
        print("       즉 CUDA_DEVICE_ORDER 를 안 걸면 반드시 어긋납니다.")
    if n_pci and n_smi and n_pci == n_smi:
        print("(참고) PCI_BUS_ID 로 강제하면 nvidia-smi 와 정확히 일치합니다. ✓")

    if cvd:
        print(f"\n★ CUDA_VISIBLE_DEVICES={cvd} 때문에 번호가 다시 매겨집니다:")
        for k, real in enumerate([x.strip() for x in cvd.split(",") if x.strip()]):
            print(f"    cuda:{k}  =  물리 GPU {real}")

    # ------------------------------------------------------ 추천 GPU
    if n_smi == n_cur:
        big = [(i, nm) for i, nm in enumerate(n_cur)
               if any(k in nm for k in ("A6000", "A100", "H100", "A40"))]
        if big:
            ids = ", ".join(f"--gpu {i} ({nm})" for i, nm in big)
            print(f"\n추천: 단독으로 충분한 카드 → {ids}")
            print("  nvtop 에서 비어 있는지 확인하고 고르세요. 분할(--gpus)은 불필요합니다.")

    print("\n※ EGL(렌더링) 번호는 또 다른 체계입니다. 규격상 순서 보장이 없습니다.")
    print("  확정하려면 05_diag_render.py 를 돌리는 **동안** 다른 터미널에서")
    print("  `nvidia-smi` 로 어느 카드에 python 프로세스가 붙었는지 보세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
