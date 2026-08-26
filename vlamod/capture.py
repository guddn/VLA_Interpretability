"""Phase 2: attention / value-norm / action logit 캡처.

전략: generate 로 attention 을 뽑으려 하지 마세요.
  - generate 는 step 마다 attention tuple 이 쪼개져 나오고, KV cache 때문에
    shape 이 step 0 과 그 이후가 다릅니다. 버전에 따라 None 이 나오기도 합니다.
  - 대신 **2-pass** 로 갑니다:
      pass 1: greedy generate 로 action token 7개를 얻는다
      pass 2: [prompt + action tokens] 전체를 한 번에 teacher-forced forward
              → 모든 층/헤드의 attention 을 한 방에 얻는다
    greedy 이므로 두 pass 의 분포는 동일합니다.
"""

from __future__ import annotations

import dataclasses

import torch

from .model_loader import LoadedVLA, action_logit_slice


@dataclasses.dataclass
class Capture:
    action_ids: torch.Tensor          # [n_action]
    attn: torch.Tensor                # [L, H, n_action, T]  (float32, cpu)
    vnorm: torch.Tensor               # [L, H, T]            (float32, cpu)
    action_logits: torch.Tensor       # [n_action, n_bins]   (float32, cpu)
    seq_len: int
    n_layers: int
    n_heads: int


# ---------------------------------------------------------------------
def _register_value_hooks(vla: LoadedVLA, store: dict):
    handles = []
    layers = vla.decoder_layers
    n_heads = vla.n_heads

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            # out: [B, T, n_kv_heads * head_dim]
            b, t, d = out.shape
            head_dim = d // n_heads
            v = out.view(b, t, n_heads, head_dim)
            store[idx] = v.float().norm(dim=-1)[0].transpose(0, 1).detach().cpu()  # [H, T]
        return hook

    for i, layer in enumerate(layers):
        vproj = layer.self_attn.v_proj
        handles.append(vproj.register_forward_hook(make_hook(i)))
    return handles


# ---------------------------------------------------------------------
@torch.no_grad()
def generate_action_tokens(vla: LoadedVLA, inputs, n_action: int = 7) -> torch.Tensor:
    """greedy 로 action token n개 생성."""
    out = vla.model.generate(
        **inputs,
        max_new_tokens=n_action,
        min_new_tokens=n_action,
        do_sample=False,
        num_beams=1,
    )
    # generate 출력이 프롬프트를 포함하는지 여부가 구현마다 다릅니다.
    gen = out[0]
    prompt_len = inputs["input_ids"].shape[1]
    if gen.shape[0] > n_action:
        gen = gen[-n_action:]
    if gen.shape[0] != n_action:
        raise RuntimeError(
            f"action token {n_action}개를 기대했는데 {gen.shape[0]}개가 나왔습니다 "
            f"(prompt_len={prompt_len}). generate 출력 규약을 확인하세요."
        )
    return gen.detach().cpu()


@torch.no_grad()
def teacher_forced_capture(
    vla: LoadedVLA,
    inputs,
    action_ids: torch.Tensor,
    action_positions: list[int],
    n_bins: int = 256,
    attn_mask_fn=None,
) -> Capture:
    """[prompt + action] 전체를 한 번에 forward 하면서 attention/value 캡처.

    attn_mask_fn: 개입용. vlamod.intervene 에서 주입합니다. None 이면 무개입.
    """
    device = vla.device
    full_ids = torch.cat(
        [inputs["input_ids"][0].cpu(), action_ids.cpu()], dim=0
    ).unsqueeze(0).to(device)

    fwd_inputs = dict(inputs)
    fwd_inputs["input_ids"] = full_ids
    if "attention_mask" in fwd_inputs and fwd_inputs["attention_mask"] is not None:
        fwd_inputs["attention_mask"] = torch.ones_like(full_ids)

    vstore: dict[int, torch.Tensor] = {}
    handles = _register_value_hooks(vla, vstore)
    mask_handles = attn_mask_fn(vla) if attn_mask_fn is not None else []

    try:
        out = vla.model(
            **fwd_inputs,
            output_attentions=True,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()
        for h in mask_handles:
            h.remove()

    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError(
            "attentions 가 None 입니다. attn_implementation='eager' 로 로드했는지 확인하세요."
        )

    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    seq_len = out.attentions[0].shape[-1]

    rows = torch.tensor(action_positions, dtype=torch.long)
    if rows.max().item() >= seq_len:
        raise RuntimeError(
            f"action 위치 {rows.max().item()} 가 시퀀스 길이 {seq_len} 를 벗어납니다. "
            "token_index 의 visual 삽입 위치 가정이 틀렸을 수 있습니다."
        )

    attn = torch.stack(
        [a[0, :, rows, :].float().cpu() for a in out.attentions], dim=0
    )  # [L, H, n_action, T]

    vnorm = torch.stack([vstore[i] for i in range(n_layers)], dim=0)  # [L, H, T]

    # action logits: 위치 p 의 토큰을 예측하는 logits 는 index p-1
    pred_rows = rows - 1
    logits = out.logits[0][pred_rows].float().cpu()          # [n_action, vocab]
    action_logits = action_logit_slice(logits, vla.tokenizer, n_bins=n_bins)

    return Capture(
        action_ids=action_ids.cpu(),
        attn=attn,
        vnorm=vnorm,
        action_logits=action_logits,
        seq_len=seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
    )
