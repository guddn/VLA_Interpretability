"""Phase 3: 인과 개입 (attention knockout).

관찰(attention)만으로는 Embodied Interpretability(2605.00321)의
"attention 은 task-irrelevant 영역에 뜬다" 비판에 그대로 당합니다.
반드시 개입을 붙여서 **행동 분포가 실제로 바뀌는지**를 봐야 합니다.

구현 방식과 버전 의존성
-----------------------
transformers 4.40.x 의 LlamaModel 은 4D additive causal mask 를 만들어
각 decoder layer 에 `attention_mask=` 키워드로 넘깁니다.
그래서 각 layer 에 forward_pre_hook(with_kwargs=True) 를 걸고
mask 에 -inf 를 더하는 방식이 가장 안전합니다.
(softmax **이전**에 개입하므로 나머지 열들이 자동 재정규화됩니다.
 softmax 이후에 0으로 만들고 다시 normalize 하는 것과 결과가 다릅니다.
 pre-softmax 마스킹이 "그 토큰이 없었다면" 에 더 가깝습니다.)

!! transformers 를 4.41 이상으로 올리면 mask 생성 규약이 바뀝니다.
   그 경우 `attach()` 안에서 kwargs 에 'attention_mask' 가 없다고 에러가 납니다.
   에러 메시지를 보고 해당 버전의 LlamaDecoderLayer.forward 시그니처를 확인하세요.
"""

from __future__ import annotations

from typing import Sequence

import torch


def apply_block(mask: torch.Tensor, query_idx: Sequence[int], key_idx: Sequence[int]) -> torch.Tensor:
    """4D additive mask [B,1,Tq,Tk] 의 (query_idx, key_idx) 교차점을 -inf 로.

    별도 함수로 뺀 이유: GPU/모델 없이 단위 테스트하기 위해서입니다.
    (tests/test_core.py)
    """
    if mask.dim() != 4:
        raise ValueError(f"4D mask 가 필요합니다. got {mask.dim()}D")
    out = mask.clone()
    q = [i for i in query_idx if 0 <= i < out.shape[-2]]
    k = [i for i in key_idx if 0 <= i < out.shape[-1]]
    if not q or not k:
        return out
    neg = torch.finfo(out.dtype).min
    qi = torch.tensor(q, dtype=torch.long, device=out.device)
    ki = torch.tensor(k, dtype=torch.long, device=out.device)
    sub = out.index_select(-2, qi)
    sub.index_fill_(-1, ki, neg)
    out.index_copy_(-2, qi, sub)
    return out


class AttentionKnockout:
    """지정한 query 행 → key 열 경로를 pre-softmax 에서 차단."""

    def __init__(
        self,
        query_idx: Sequence[int],
        key_idx: Sequence[int],
        layers: Sequence[int] | None = None,
    ):
        """
        query_idx : 차단할 query 위치 (보통 action token 위치)
        key_idx   : 차단할 key 위치   (보통 language 또는 visual 구간)
        layers    : None 이면 전체 층. 특정 층만 주면 layer-wise knockout.
        """
        self.query_idx = list(query_idx)
        self.key_idx = list(key_idx)
        self.layers = None if layers is None else set(layers)
        self._seen_mask = False

    # -- hook 본체 --------------------------------------------------
    def _make_hook(self, layer_i: int):
        def hook(module, args, kwargs):  # noqa: ANN001
            if self.layers is not None and layer_i not in self.layers:
                return None

            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                raise RuntimeError("hidden_states 를 못 찾았습니다. layer 시그니처 확인 필요.")
            b, t, _ = hidden.shape
            dtype = hidden.dtype
            device = hidden.device

            mask = kwargs.get("attention_mask", None)
            if mask is None:
                # 인과 마스크를 직접 만든다 (eager 경로에서 None 이 오는 경우 대비)
                neg = torch.finfo(dtype).min
                causal = torch.full((t, t), neg, dtype=dtype, device=device)
                causal = torch.triu(causal, diagonal=1)
                mask = causal[None, None, :, :].expand(b, 1, t, t).clone()
            else:
                mask = mask.clone()
                if mask.dim() != 4:
                    raise RuntimeError(
                        f"4D attention mask 를 기대했는데 {mask.dim()}D 가 왔습니다. "
                        "transformers 버전이 4.40.x 인지 확인하세요."
                    )
            self._seen_mask = True

            kwargs["attention_mask"] = apply_block(mask, self.query_idx, self.key_idx)
            return (args, kwargs)

        return hook

    # -- 부착/해제 ---------------------------------------------------
    def attach(self, vla) -> list:
        handles = []
        for i, layer in enumerate(vla.decoder_layers):
            handles.append(
                layer.register_forward_pre_hook(self._make_hook(i), with_kwargs=True)
            )
        return handles

    def as_fn(self):
        """capture.teacher_forced_capture(attn_mask_fn=...) 에 넣을 형태."""
        return lambda vla: self.attach(vla)


# ---------------------------------------------------------------------
def knockout_language(spans, layers=None) -> AttentionKnockout:
    return AttentionKnockout(spans.action, spans.language, layers=layers)


def knockout_vision(spans, layers=None) -> AttentionKnockout:
    return AttentionKnockout(spans.action, spans.visual, layers=layers)


def knockout_random_control(spans, seed: int = 0, layers=None) -> AttentionKnockout:
    """대조군: 언어와 **같은 개수**의 visual 토큰을 무작위로 차단.

    이게 없으면 "언어를 지웠더니 안 변했다"가
    "토큰 몇 개 지운 건 원래 아무 영향 없다"와 구분되지 않습니다.
    반드시 함께 돌리세요.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(spans.language)
    pool = torch.tensor(spans.visual)
    pick = pool[torch.randperm(len(pool), generator=g)[:n]].tolist()
    return AttentionKnockout(spans.action, pick, layers=layers)
