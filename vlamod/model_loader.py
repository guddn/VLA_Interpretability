"""OpenVLA 로딩 + 백본 접근 유틸.

가장 중요한 것: **attn_implementation="eager"**.
flash_attention_2 나 sdpa 로 로드하면 output_attentions=True 가 조용히 무시되고
attentions 가 None 으로 나옵니다. 에러도 안 납니다. 이걸로 하루 날리기 쉽습니다.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch


@dataclasses.dataclass
class LoadedVLA:
    model: Any
    processor: Any
    device: torch.device
    dtype: torch.dtype
    path: str

    # ---- 백본 접근자 -------------------------------------------------
    # OpenVLA(Prismatic) 의 내부 속성 이름은 버전에 따라 다릅니다.
    # 하드코딩 대신 탐색합니다.

    @property
    def language_model(self):
        return find_language_model(self.model)

    @property
    def decoder_layers(self):
        """LlamaDecoderLayer 리스트."""
        lm = self.language_model
        for attr in ("model", "transformer"):
            inner = getattr(lm, attr, None)
            if inner is not None and hasattr(inner, "layers"):
                return inner.layers
        if hasattr(lm, "layers"):
            return lm.layers
        raise AttributeError("decoder layers 를 찾지 못했습니다. 모델 구조를 직접 확인하세요.")

    @property
    def n_layers(self) -> int:
        return len(self.decoder_layers)

    @property
    def n_heads(self) -> int:
        return int(self.language_model.config.num_attention_heads)

    @property
    def tokenizer(self):
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            raise AttributeError("processor.tokenizer 가 없습니다.")
        return tok


def find_language_model(model):
    """Prismatic 래퍼 안의 LM(보통 LlamaForCausalLM)을 찾아 반환."""
    for attr in ("language_model", "llm_backbone", "lm", "text_model"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        # llm_backbone 은 한 겹 더 감싸져 있을 수 있음
        inner = getattr(obj, "llm", None)
        if inner is not None:
            return inner
        if hasattr(obj, "config") and hasattr(obj.config, "num_attention_heads"):
            return obj
    # 마지막 수단: 이름으로 훑기
    for name, module in model.named_modules():
        if module.__class__.__name__.endswith("ForCausalLM"):
            return module
    raise AttributeError(
        "language model 을 찾지 못했습니다. "
        "`for n, _ in model.named_children(): print(n)` 로 직접 확인하세요."
    )


# OpenVLA-7B(≈7.5B) bf16 가중치의 대략적인 크기 (GB)
WEIGHTS_GB = 15.0


def _max_memory_map(gpus: list[int], headroom_gb: float = 1.5,
                    cpu_offload: bool = False) -> tuple[dict, float]:
    """각 GPU 의 **현재 여유**에서 headroom 을 뺀 값을 상한으로 씁니다.

    고정값(예: "10GiB")을 쓰면 남의 작업이 늘었을 때 OOM 이 납니다.

    headroom_gb : CUDA 컨텍스트 + 활성값 + KV 캐시용 마진.
                  메모리가 빠듯하면 0.8 까지 줄일 수 있지만 OOM 위험이 커집니다.
    cpu_offload : True 면 GPU 에 다 못 올린 층을 CPU 로 넘깁니다.
                  **매우 느려집니다**(층마다 PCIe 전송). 구조 검증용으로만 쓰세요.
    """
    mm, total = {}, 0.0
    for i in gpus:
        free, _ = torch.cuda.mem_get_info(i)
        usable = max(free / 1e9 - headroom_gb, 0.3)
        mm[i] = f"{usable:.1f}GiB"
        total += usable
    # 기본은 CPU 오프로드 금지 — 조용히 100배 느려지는 것을 막습니다
    mm["cpu"] = "48GiB" if cpu_offload else "0GiB"
    return mm, total


def load_openvla(
    path: str = "openvla/openvla-7b-finetuned-libero-spatial",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    gpus: list[int] | None = None,
    headroom_gb: float = 1.5,
    cpu_offload: bool = False,
) -> LoadedVLA:
    """gpus 가 2개 이상이면 층을 나눠 올립니다(model sharding).

    수치는 단일 GPU 와 **동일**합니다. 같은 연산을 배치만 나눠 하는 것이라
    attention 값도 action logit 도 바뀌지 않습니다. (양자화와 다른 점)
    대신 층 경계마다 GPU 간 전송이 생겨 느려집니다.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if attn_implementation != "eager":
        raise ValueError(
            f"attn_implementation={attn_implementation!r} 로는 attention 을 못 뽑습니다. "
            "'eager' 를 쓰세요. (계획서 Phase 0 실무 함정)"
        )

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    common = dict(
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if gpus and len(gpus) >= 2:
        mm, total = _max_memory_map(gpus, headroom_gb, cpu_offload)
        print(f"[shard ] GPU {gpus} 에 분할 로드 (headroom {headroom_gb}GB/장). 상한: "
              + ", ".join(f"{k}={v}" for k, v in mm.items() if k != "cpu"))
        print(f"[shard ] 사용 가능 합계 {total:.1f}GB  /  가중치 약 {WEIGHTS_GB:.0f}GB 필요")
        if total < WEIGHTS_GB:
            msg = (
                f"사용 가능 합계 {total:.1f}GB 가 가중치 {WEIGHTS_GB:.0f}GB 보다 작습니다.\n"
                f"  선택지:\n"
                f"   1) --headroom-gb 0.8  로 마진을 줄인다 (OOM 위험 증가)\n"
                f"   2) --allow-cpu-offload 로 일부 층을 CPU 로 넘긴다 (매우 느림, 구조 검증용)\n"
                f"   3) 더 빈 GPU 를 기다린다 (24GB 한 장이면 --gpu N 으로 충분)\n"
                f"   4) --device cpu 로 CPU 에서만 돌린다 (Stage 2 구조 검증에는 충분)"
            )
            if not cpu_offload:
                raise RuntimeError(msg)
            print("[shard ] !! " + msg.replace("\n", "\n[shard ] !! "))
        elif total < WEIGHTS_GB + 1.5:
            print("[shard ] !! 여유가 매우 빠듯합니다. OOM 이 나면 --allow-cpu-offload 를 붙이세요.")
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                path, device_map="auto", max_memory=mm, **common
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"멀티 GPU 분할 로드에 실패했습니다: {e}\n"
                "  이 모델이 accelerate 의 device_map 을 지원하지 않을 수 있습니다.\n"
                "  (trust_remote_code 모델은 _no_split_modules 가 없으면 실패합니다)\n"
                "  → 여유 20GB 이상인 단일 GPU 를 찾아 --gpu N 으로 쓰세요."
            ) from e
        # 입력은 첫 번째 장치로 보냅니다 (embedding 이 거기 있음)
        primary = getattr(model, "hf_device_map", None)
        first = f"cuda:{gpus[0]}"
        if isinstance(primary, dict) and primary:
            v = next(iter(primary.values()))
            if isinstance(v, int):
                first = f"cuda:{v}"
            elif isinstance(v, str) and v.startswith("cuda"):
                first = v
        dev = torch.device(first)
    else:
        if device != "cpu":
            free, _ = torch.cuda.mem_get_info(int(device.split(":")[1]))
            if free / 1e9 < WEIGHTS_GB + 1.0:
                raise RuntimeError(
                    f"{device} 의 여유가 {free/1e9:.1f}GB 인데 가중치만 {WEIGHTS_GB:.0f}GB 필요합니다.\n"
                    f"  → 더 빈 GPU 를 쓰거나, 두 장을 묶으세요:  --gpus a,b"
                )
        model = AutoModelForVision2Seq.from_pretrained(path, **common).to(device)
        dev = torch.device(device)

    model.eval()
    return LoadedVLA(
        model=model,
        processor=processor,
        device=dev,
        dtype=torch_dtype,
        path=path,
    )


# ---------------------------------------------------------------------
# action token id 구간
# ---------------------------------------------------------------------
def action_bin_token_ids(tokenizer, n_bins: int = 256) -> torch.Tensor:
    """OpenVLA 의 action bin 에 대응하는 vocab id 를 오름차순으로 반환.

    OpenVLA ActionTokenizer 규약(요약):
        encode : token_id = tokenizer.vocab_size - discretized_bin
        decode : discretized_bin = tokenizer.vocab_size - token_id
    즉 action token 은 vocab 의 **끝쪽 256개**를 빌려 씁니다.

    !! 검증 필수: 아래 assert 가 통과해도 오프셋이 1 어긋날 수 있습니다.
       scripts/01_smoke_forward.py 가 실제 생성된 action token id 가
       이 구간 안에 들어오는지 확인해 줍니다. 반드시 돌려보세요.
    """
    v = int(tokenizer.vocab_size)
    ids = torch.arange(v - n_bins, v, dtype=torch.long)
    assert ids.numel() == n_bins
    return ids


def action_logit_slice(logits: torch.Tensor, tokenizer, n_bins: int = 256) -> torch.Tensor:
    """전체 vocab logits 에서 action bin 구간만 잘라냅니다.

    logits: [..., vocab_out]  (LM head 출력 차원은 tokenizer.vocab_size 보다 클 수 있음)
    반환  : [..., n_bins]
    """
    v = int(tokenizer.vocab_size)
    if logits.shape[-1] < v:
        raise ValueError(f"logits 차원({logits.shape[-1]}) < tokenizer.vocab_size({v})")
    return logits[..., v - n_bins : v]
