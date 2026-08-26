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


def load_openvla(
    path: str = "openvla/openvla-7b-finetuned-libero-spatial",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
) -> LoadedVLA:
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if attn_implementation != "eager":
        raise ValueError(
            f"attn_implementation={attn_implementation!r} 로는 attention 을 못 뽑습니다. "
            "'eager' 를 쓰세요. (계획서 Phase 0 실무 함정)"
        )

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        path,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    return LoadedVLA(
        model=model,
        processor=processor,
        device=torch.device(device),
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
