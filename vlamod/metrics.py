"""모달리티 기여 비율 지표 3종 + KL.

정의 (action query s, layer l, head h):

  M_X = Σ_{j∈X} A(s, j)                     ... raw mass
  R_raw  = M_L / (M_L + M_V)                 ... IVAR 재현. **단독 주장 금지**
  R_norm = (M_L/|L|) / (M_L/|L| + M_V/|V|)   ... 토큰 개수 효과 제거
  R_vn   = value-norm 가중 (Kobayashi+2020)  ... Σ A(s,j)·‖v_j‖ 로 대체

왜 R_raw 단독이 위험한가:
  |V| ≈ 256, |L| ≈ 5~15. attention 은 합이 1이므로, 균등 분포만 가정해도
  R_raw ≈ |L|/(|L|+|V|) ≈ 0.04 가 나옵니다. "vision 이 96% 먹는다"는
  발견이 아니라 산수입니다. 그래서 `uniform_baseline()` 을 반드시 같이 보고하세요.
"""

from __future__ import annotations

import dataclasses

import torch

EPS = 1e-12


@dataclasses.dataclass
class RatioResult:
    r_raw: torch.Tensor      # [L, H, n_action]
    r_norm: torch.Tensor     # [L, H, n_action]
    r_vnorm: torch.Tensor    # [L, H, n_action]
    mass_v: torch.Tensor     # [L, H, n_action]
    mass_l: torch.Tensor     # [L, H, n_action]
    mass_sink: torch.Tensor  # [L, H, n_action]
    mass_other: torch.Tensor # [L, H, n_action]  (template + action + 그 외)
    uniform_baseline: float  # |L| / (|L| + |V|)

    def mean_over_heads(self):
        return (
            self.r_raw.mean(dim=1),
            self.r_norm.mean(dim=1),
            self.r_vnorm.mean(dim=1),
        )


def _gather_mass(attn: torch.Tensor, idx: list[int]) -> torch.Tensor:
    """attn: [L, H, Q, T] → [L, H, Q] (idx 열의 합)."""
    if not idx:
        return torch.zeros(attn.shape[:3], dtype=attn.dtype)
    cols = torch.tensor(idx, dtype=torch.long)
    return attn.index_select(dim=-1, index=cols).sum(dim=-1)


def compute_ratios(
    attn: torch.Tensor,
    vnorm: torch.Tensor,
    visual_idx: list[int],
    language_idx: list[int],
    sink_idx: list[int] | None = None,
) -> RatioResult:
    """attn: [L, H, Q, T], vnorm: [L, H, T]"""
    sink_idx = sink_idx or []
    if attn.dim() != 4:
        raise ValueError(f"attn 은 [L,H,Q,T] 여야 합니다. got {tuple(attn.shape)}")
    if vnorm.shape[0] != attn.shape[0] or vnorm.shape[1] != attn.shape[1]:
        raise ValueError("attn 과 vnorm 의 (L,H) 가 다릅니다.")
    if vnorm.shape[-1] != attn.shape[-1]:
        raise ValueError("attn 과 vnorm 의 시퀀스 길이가 다릅니다.")

    m_v = _gather_mass(attn, visual_idx)
    m_l = _gather_mass(attn, language_idx)
    m_s = _gather_mass(attn, sink_idx)
    m_all = attn.sum(dim=-1)
    m_other = m_all - m_v - m_l - m_s

    n_v = max(len(visual_idx), 1)
    n_l = max(len(language_idx), 1)

    r_raw = m_l / (m_l + m_v + EPS)
    r_norm = (m_l / n_l) / ((m_l / n_l) + (m_v / n_v) + EPS)

    # value-norm 가중: A(s,j)·‖v_j‖ 를 더한다
    w = attn * vnorm.unsqueeze(2)            # [L,H,Q,T]
    wv = _gather_mass(w, visual_idx)
    wl = _gather_mass(w, language_idx)
    r_vnorm = wl / (wl + wv + EPS)

    return RatioResult(
        r_raw=r_raw,
        r_norm=r_norm,
        r_vnorm=r_vnorm,
        mass_v=m_v,
        mass_l=m_l,
        mass_sink=m_s,
        mass_other=m_other,
        uniform_baseline=len(language_idx) / max(len(language_idx) + len(visual_idx), 1),
    )


# ---------------------------------------------------------------------
# 인과 개입용 거리
# ---------------------------------------------------------------------
def action_kl(logits_p: torch.Tensor, logits_q: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """KL(P ‖ Q). logits_*: [n_action, n_bins] (DoF 별 256-bin 분포)

    반환: reduction='mean' → 스칼라, 'none' → [n_action]
    """
    if logits_p.shape != logits_q.shape:
        raise ValueError("두 logits 의 shape 이 다릅니다.")
    logp = torch.log_softmax(logits_p.float(), dim=-1)
    logq = torch.log_softmax(logits_q.float(), dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(dim=-1)   # [n_action]
    if reduction == "none":
        return kl
    return kl.mean()


def causal_ratio(kl_lang: float, kl_vis: float) -> float:
    """CE_L / (CE_L + CE_V). 언어 knockout 이 더 큰 변화를 만들수록 1에 가까움."""
    denom = kl_lang + kl_vis
    if denom <= 0:
        return float("nan")
    return kl_lang / denom


def decode_actions(action_logits: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """argmax bin index [n_action]. 연속 action 값 비교용 proxy."""
    return action_logits.argmax(dim=-1)
