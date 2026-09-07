"""GPU 번호가 어느 물리 카드를 가리키는지 **확정**합니다.

    python setup/06_gpu_map.py

세 가지 번호 체계를 나란히 놓고 비교합니다:

  1) nvidia-smi / nvtop      — PCI 버스 순서 (사람이 보는 번호)
  2) torch 기본              — CUDA_DEVICE_ORDER=FASTEST_FIRST (재정렬됨!)
  3) torch + PCI_BUS_ID      — nvidia-smi 와 일치해야 함

2와 3이 다르면 **재정렬이 실제로 일어나고 있다**는 증거입니다.
그 경우 nvtop 에서 "0번이 비었네" 하고 `--gpu 0` 을 주면 엉뚱한 카드를 잡습니다.

CUDA_VISIBLE_DEVICES 가 걸려 있으면 그것도 함께 알려줍니다
(이게 있으면 번호가 그 목록 안에서 0부터 다시 매겨집니다).
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


def torch_list(order: str | None) -> list[list]:
    """별도 프로세스에서 torch 를 띄웁니다. CUDA_DEVICE_ORDER 는 런타임 초기화
    시점에 한 번만 읽히므로, 같은 프로세스에서 두 번 바꿔 볼 수 없습니다."""
    env = dict(os.environ)
    if order:
        env["CUDA_DEVICE_ORDER"] = order
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


def main() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("=" * 74)
    print(f"CUDA_VISIBLE_DEVICES = {cvd if cvd else '(설정 안 됨)'}")
    print(f"CUDA_DEVICE_ORDER    = {os.environ.get('CUDA_DEVICE_ORDER', '(설정 안 됨 → FASTEST_FIRST)')}")
    print("=" * 74)

    print("\n[1] nvidia-smi (PCI 순서 — nvtop 과 같은 번호)")
    try:
        smi = smi_list()
        for i, name, mem, bus in smi:
            print(f"    GPU {i}: {name:28s} {mem:>10s}  {bus}")
    except Exception as e:  # noqa: BLE001
        smi = []
        print(f"    조회 실패: {e}")

    print("\n[2] torch 기본 (CUDA_DEVICE_ORDER 미설정 → FASTEST_FIRST)")
    try:
        fast = torch_list(None)
        for i, name, mem in fast:
            print(f"    cuda:{i}: {name:28s} {mem:>6.1f} GB")
    except Exception as e:  # noqa: BLE001
        fast = []
        print(f"    조회 실패: {e}")

    print("\n[3] torch + CUDA_DEVICE_ORDER=PCI_BUS_ID")
    try:
        pci = torch_list("PCI_BUS_ID")
        for i, name, mem in pci:
            print(f"    cuda:{i}: {name:28s} {mem:>6.1f} GB")
    except Exception as e:  # noqa: BLE001
        pci = []
        print(f"    조회 실패: {e}")

    # ----------------------------------------------------------------- 판정
    print("\n" + "=" * 74)
    if fast and pci:
        names_fast = [n for _, n, _ in fast]
        names_pci = [n for _, n, _ in pci]
        if names_fast != names_pci:
            print("★ 재정렬이 일어나고 있습니다.")
            print("  torch 기본 번호 ≠ nvidia-smi 번호. nvtop 을 보고 --gpu 를 고르면 틀립니다.")
            print("\n  대응 방법 (택 1):")
            print("    A) configs/default.yaml 의 env.cuda_device_order: \"PCI_BUS_ID\" (이미 기본값)")
            print("    B) export CUDA_DEVICE_ORDER=PCI_BUS_ID")
            print("\n  매핑 (PCI 번호 → FASTEST_FIRST 번호):")
            used = set()
            for i, n, _ in pci:
                for j, n2, _ in fast:
                    if n2 == n and j not in used:
                        used.add(j)
                        mark = "" if i == j else "   ← 다름"
                        print(f"    nvtop {i}  =  torch기본 {j}   ({n}){mark}")
                        break
        else:
            print("정렬이 일치합니다. nvtop 번호를 그대로 --gpu 에 쓰셔도 됩니다.")

    if smi and pci and len(smi) != len(pci):
        print(f"\n★ 보이는 GPU 수가 다릅니다: nvidia-smi {len(smi)}장, torch {len(pci)}장")
        print("  CUDA_VISIBLE_DEVICES 로 일부만 노출된 상태입니다.")

    if cvd:
        print(f"\n★ CUDA_VISIBLE_DEVICES={cvd} 때문에 번호가 다시 매겨집니다:")
        for k, real in enumerate([x.strip() for x in cvd.split(",") if x.strip()]):
            print(f"    cuda:{k}  =  물리 GPU {real}")
        print("  MUJOCO_EGL_DEVICE_ID 는 이 재매핑을 따르지 않을 수 있으니 주의하세요.")

    print("\n※ EGL(렌더링) 디바이스 번호는 또 다른 체계입니다. 규격상 순서 보장이 없습니다.")
    print("  확정하려면 05_diag_render.py 를 돌리는 **동안** 다른 터미널에서")
    print("  `nvidia-smi` 를 실행해 어느 카드에 python 프로세스가 붙었는지 보세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
