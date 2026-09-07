"""GPU/모델 없이 돌아가는 단위 테스트. 로직 실수를 여기서 잡습니다.

    pytest -q tests/test_core.py
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlamod import metrics as M  # noqa: E402
from vlamod import token_index as TI  # noqa: E402
from vlamod.intervene import apply_block  # noqa: E402


# ---------------------------------------------------------------- metrics
def test_ratios_uniform_attention():
    """균등 attention 이면 R_raw ≈ |L|/(|L|+|V|), R_norm ≈ 0.5 여야 한다."""
    L, H, Q, T = 2, 3, 4, 20
    attn = torch.full((L, H, Q, T), 1.0 / T)
    vnorm = torch.ones(L, H, T)
    visual = list(range(1, 17))   # 16개
    language = list(range(17, 20))  # 3개
    r = M.compute_ratios(attn, vnorm, visual, language, sink_idx=[0])

    expected_raw = 3 / (3 + 16)
    assert abs(float(r.r_raw.mean()) - expected_raw) < 1e-5
    assert abs(float(r.r_norm.mean()) - 0.5) < 1e-5
    assert abs(r.uniform_baseline - expected_raw) < 1e-9
    # value-norm 이 전부 1이면 R_vnorm == R_raw
    assert abs(float(r.r_vnorm.mean()) - expected_raw) < 1e-5


def test_ratios_all_attention_on_language():
    L, H, Q, T = 1, 1, 1, 10
    attn = torch.zeros(L, H, Q, T)
    attn[..., 8] = 0.5
    attn[..., 9] = 0.5
    vnorm = torch.ones(L, H, T)
    r = M.compute_ratios(attn, vnorm, visual_idx=list(range(1, 8)), language_idx=[8, 9])
    assert float(r.r_raw.mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(r.r_norm.mean()) == pytest.approx(1.0, abs=1e-6)


def test_value_norm_changes_ratio():
    """언어 토큰의 value norm 이 크면 R_vnorm > R_raw 여야 한다."""
    L, H, Q, T = 1, 1, 1, 6
    attn = torch.full((L, H, Q, T), 1.0 / T)
    vnorm = torch.ones(L, H, T)
    vnorm[..., 4:] = 10.0            # 언어 토큰의 value 가 훨씬 큼
    r = M.compute_ratios(attn, vnorm, visual_idx=[0, 1, 2, 3], language_idx=[4, 5])
    assert float(r.r_vnorm.mean()) > float(r.r_raw.mean())


def test_mass_decomposition_sums_to_one():
    L, H, Q, T = 2, 2, 3, 12
    attn = torch.softmax(torch.randn(L, H, Q, T), dim=-1)
    vnorm = torch.rand(L, H, T) + 0.1
    r = M.compute_ratios(attn, vnorm, visual_idx=list(range(1, 8)),
                         language_idx=[8, 9], sink_idx=[0])
    total = r.mass_v + r.mass_l + r.mass_sink + r.mass_other
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_kl_self_is_zero():
    logits = torch.randn(7, 256)
    assert float(M.action_kl(logits, logits)) == pytest.approx(0.0, abs=1e-6)


def test_kl_positive_and_asymmetric():
    a = torch.randn(7, 256)
    b = torch.randn(7, 256)
    kl_ab = float(M.action_kl(a, b))
    kl_ba = float(M.action_kl(b, a))
    assert kl_ab > 0 and kl_ba > 0
    assert kl_ab != pytest.approx(kl_ba, abs=1e-6)


def test_causal_ratio():
    assert M.causal_ratio(1.0, 1.0) == pytest.approx(0.5)
    assert M.causal_ratio(0.0, 2.0) == pytest.approx(0.0)
    assert M.causal_ratio(3.0, 1.0) == pytest.approx(0.75)


# ------------------------------------------------------------- intervene
def test_apply_block_masks_only_target_cells():
    mask = torch.zeros(1, 1, 6, 6)
    out = apply_block(mask, query_idx=[4, 5], key_idx=[1, 2])
    neg = torch.finfo(out.dtype).min
    assert out[0, 0, 4, 1] == neg and out[0, 0, 5, 2] == neg
    assert out[0, 0, 3, 1] == 0.0        # 다른 query 행은 그대로
    assert out[0, 0, 4, 3] == 0.0        # 다른 key 열은 그대로
    assert mask.sum() == 0.0             # 원본 불변 (clone 확인)


def test_apply_block_out_of_range_is_ignored():
    mask = torch.zeros(1, 1, 4, 4)
    out = apply_block(mask, query_idx=[99], key_idx=[1])
    assert torch.equal(out, mask)


def test_knockout_removes_mass_after_softmax():
    """마스킹 후 softmax 하면 해당 열의 확률이 0이 되고 나머지가 재정규화되어야 한다."""
    scores = torch.randn(1, 1, 5, 5)
    masked = apply_block(torch.zeros(1, 1, 5, 5), query_idx=[4], key_idx=[0, 1])
    p = torch.softmax(scores + masked, dim=-1)
    assert p[0, 0, 4, 0] == pytest.approx(0.0, abs=1e-8)
    assert p[0, 0, 4, 1] == pytest.approx(0.0, abs=1e-8)
    assert float(p[0, 0, 4].sum()) == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------ token_index
def test_build_prompt_offsets_point_at_instruction():
    prompt, (s, e) = TI.build_prompt("Pick up the Black Bowl.")
    assert prompt[s:e] == "pick up the black bowl"
    assert prompt.startswith("In: What action should the robot take to")
    assert prompt.endswith("Out:")


def test_build_prompt_lowercases_and_strips_period():
    prompt, (s, e) = TI.build_prompt("  OPEN the drawer.  ")
    assert prompt[s:e] == "open the drawer"


class _FakeTokenizer:
    """offset_mapping 을 주는 최소 토크나이저 (fast tokenizer 흉내)."""

    vocab_size = 32000
    bos_token = "<s>"
    eos_token = "</s>"

    def __init__(self, prompt: str):
        self.prompt = prompt
        # 공백 단위로 자르고 앞에 BOS 를 붙인다
        self._offsets = [(0, 0)]
        i = 0
        for w in prompt.split(" "):
            j = prompt.find(w, i)
            self._offsets.append((j, j + len(w)))
            i = j + len(w)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        return {"offset_mapping": self._offsets}

    def decode(self, ids, skip_special_tokens=False):
        return f"<tok{ids[0]}>"


class _FakeVLA:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_build_spans_separates_language_from_template():
    instruction = "pick up the bowl"
    prompt, char_span = TI.build_prompt(instruction)
    tok = _FakeTokenizer(prompt)
    n_text = len(tok._offsets)
    ids = torch.arange(n_text)
    vla = _FakeVLA(tok)

    spans = TI.build_spans(
        vla, prompt, char_span, ids,
        visual_span=(1, 1 + 8),      # visual 8개가 index 1 에 삽입
        n_action_tokens=7,
        sink_positions=(0,),
        instruction_only=True,
    )
    assert spans.n_visual == 8
    assert spans.visual == list(range(1, 9))
    assert len(spans.language) >= 3          # pick/up/the/bowl 중 최소 3개
    assert len(spans.action) == 7
    assert spans.action[0] == spans.n_total_prompt
    # 언어와 템플릿이 겹치지 않아야 한다
    assert not (set(spans.language) & set(spans.template))
    # 언어 인덱스는 전부 visual 구간 뒤에 있어야 한다
    assert min(spans.language) > spans.visual[-1]


def test_build_spans_raises_when_language_empty():
    prompt, _ = TI.build_prompt("pick up the bowl")
    tok = _FakeTokenizer(prompt)
    ids = torch.arange(len(tok._offsets))
    with pytest.raises(RuntimeError):
        TI.build_spans(
            _FakeVLA(tok), prompt,
            instr_char_span=(10**6, 10**6 + 1),   # 존재하지 않는 구간
            input_ids=ids, visual_span=(1, 9),
        )


# ------------------------------------------------------------------ device
from types import SimpleNamespace  # noqa: E402

from vlamod.device import apply_overrides, device_index, normalize_device  # noqa: E402


@pytest.mark.parametrize(
    "given,expected",
    [
        (0, "cuda:0"), (3, "cuda:3"), ("3", "cuda:3"), ("cuda:3", "cuda:3"),
        ("cuda", "cuda:0"), ("CUDA:2", "cuda:2"), ("cpu", "cpu"), ("-1", "cpu"),
    ],
)
def test_normalize_device(given, expected):
    assert normalize_device(given) == expected


@pytest.mark.parametrize("bad", ["gpu3", "cuda:x", "", "foo"])
def test_normalize_device_rejects_garbage(bad):
    with pytest.raises(ValueError):
        normalize_device(bad)


def test_device_index():
    assert device_index("cuda:5") == 5
    assert device_index("cpu") is None


def _cfg():
    return {"path": "openvla/openvla-7b", "device": "cuda:0",
            "dtype": "bfloat16", "unnorm_key": "libero_spatial"}


def test_apply_overrides_gpu_flag(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=3, device=None, model=None, unnorm_key=None))
    assert c["device"] == "cuda:3"
    # !! EGL 번호는 CUDA 번호와 **다른 체계**이므로 3 이 그대로 들어가면 안 됩니다.
    #    EGL 을 못 열거하는 환경(CI/컨테이너)에서는 0 으로 폴백합니다.
    assert os.environ["MUJOCO_EGL_DEVICE_ID"].isdigit()
    assert c["egl_device"] == int(os.environ["MUJOCO_EGL_DEVICE_ID"])


def test_apply_overrides_device_string(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=None, device="2", model=None, unnorm_key=None))
    assert c["device"] == "cuda:2"


def test_apply_overrides_rejects_both(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit):
        apply_overrides(_cfg(), SimpleNamespace(gpu=1, device="cuda:2", model=None, unnorm_key=None))


def test_apply_overrides_model_and_unnorm(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=None, device=None,
                                       model="openvla/openvla-7b-finetuned-libero-object",
                                       unnorm_key="libero_object"))
    assert c["path"].endswith("libero-object")
    assert c["unnorm_key"] == "libero_object"


def test_apply_overrides_ignores_missing_attrs(monkeypatch):
    """01 스크립트처럼 --unnorm-key/--tag 가 없는 경우에도 죽지 않아야 한다."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=1))
    assert c["device"] == "cuda:1"


def test_apply_overrides_detects_visible_devices_conflict(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        apply_overrides(_cfg(), SimpleNamespace(gpu=3, device=None, model=None, unnorm_key=None))


def test_cpu_still_sets_egl_device(monkeypatch):
    """모델이 CPU 여도 LIBERO 렌더링은 여전히 EGL 을 씁니다.

    (예전 버전은 cpu 일 때 MUJOCO_EGL_DEVICE_ID 를 아예 안 걸었는데,
     그러면 렌더링이 EGL 기본 디바이스로 조용히 새어 나갑니다.)
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=None, device="cpu", model=None, unnorm_key=None))
    assert c["device"] == "cpu"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"].isdigit()


def test_egl_device_explicit_wins(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=3, device=None, model=None,
                                       unnorm_key=None, egl_device=1))
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
    assert c["egl_device"] == 1


def test_resolve_egl_device_rules():
    from vlamod.device import resolve_egl_device
    # 명시 지정이 최우선
    assert resolve_egl_device(8, 2)[0] == 2
    # EGL 을 못 열거하면 0
    idx, why = resolve_egl_device(8, None)
    assert isinstance(idx, int) and idx >= 0 and why


# -------------------------------------------------------------- apply_env
from vlamod.device import apply_env  # noqa: E402


def test_apply_env_sets_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    cfg = {"env": {"hf_home": str(tmp_path / "cache"), "mujoco_gl": "egl",
                   "pyopengl_platform": "egl"}}
    applied = apply_env(cfg, SimpleNamespace())
    assert os.environ["HF_HOME"] == str(tmp_path / "cache")
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "cache" / "hub")
    assert (tmp_path / "cache" / "hub").is_dir()      # 디렉토리를 실제로 만든다
    assert os.environ["MUJOCO_GL"] == "egl"
    assert applied["HF_HOME"].endswith("cache")


def test_apply_env_cli_beats_config(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    cfg = {"env": {"hf_home": str(tmp_path / "from_config")}}
    apply_env(cfg, SimpleNamespace(hf_home=str(tmp_path / "from_cli")))
    assert os.environ["HF_HOME"] == str(tmp_path / "from_cli")


def test_apply_env_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    apply_env({"env": {"hf_home": "~/mycache"}}, SimpleNamespace())
    assert os.environ["HF_HOME"] == str(tmp_path / "mycache")


def test_apply_env_noop_without_section(monkeypatch):
    """env 섹션이 없으면 HF_HOME / MUJOCO_GL 은 건드리지 않습니다.

    단 CUDA_DEVICE_ORDER 는 **설정이 없어도 PCI_BUS_ID 를 강제**합니다.
    CUDA 기본값(FASTEST_FIRST)은 카드가 섞인 서버에서 torch 번호와
    nvidia-smi 번호를 어긋나게 만들기 때문입니다.
    """
    monkeypatch.delenv("HF_HOME", raising=False)
    applied = apply_env({}, SimpleNamespace())
    assert applied == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    assert "HF_HOME" not in os.environ


def test_apply_env_cuda_order_can_be_disabled(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    applied = apply_env({"env": {"cuda_device_order": None}}, SimpleNamespace())
    assert "CUDA_DEVICE_ORDER" not in applied
    assert "CUDA_DEVICE_ORDER" not in os.environ


def test_apply_env_cuda_order_explicit(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    applied = apply_env({"env": {"cuda_device_order": "FASTEST_FIRST"}}, SimpleNamespace())
    assert applied["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
    assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_apply_env_warns_if_hf_already_imported(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", object())
    with pytest.warns(UserWarning, match="huggingface_hub"):
        apply_env({"env": {"hf_home": str(tmp_path / "c")}}, SimpleNamespace())


# ----------------------------------------------------------- multi-GPU
from vlamod.device import parse_gpus  # noqa: E402


@pytest.mark.parametrize("given,expected", [
    ("0,1", [0, 1]), ("0 1", [0, 1]), ("2", [2]), ([3, 4], [3, 4]), (None, []),
    ("0, 1, 2", [0, 1, 2]),
])
def test_parse_gpus(given, expected):
    assert parse_gpus(given) == expected


@pytest.mark.parametrize("bad", ["0,a", "x", "0,,1,b"])
def test_parse_gpus_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_gpus(bad)


def test_parse_gpus_rejects_duplicates():
    with pytest.raises(ValueError, match="중복"):
        parse_gpus("1,1")


def test_apply_overrides_multi_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=None, gpus="0,1", device=None,
                                       model=None, unnorm_key=None))
    assert c["gpus"] == [0, 1]
    assert c["device"] == "cuda:0"          # 입력은 첫 장치로


def test_apply_overrides_single_element_gpus_is_plain_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=None, gpus="3", device=None,
                                       model=None, unnorm_key=None))
    assert c["gpus"] == []                  # 1장이면 분할 안 함
    assert c["device"] == "cuda:3"


def test_apply_overrides_rejects_gpu_and_gpus(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit):
        apply_overrides(_cfg(), SimpleNamespace(gpu=0, gpus="0,1", device=None,
                                                model=None, unnorm_key=None))


# ------------------------------------------------- headroom / cpu offload
def test_apply_overrides_headroom_default(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=0, gpus=None, device=None, model=None,
                                       unnorm_key=None))
    assert c["headroom_gb"] == 1.5
    assert c["cpu_offload"] is False


def test_apply_overrides_headroom_override(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    c = _cfg()
    apply_overrides(c, SimpleNamespace(gpu=None, gpus="0,6", device=None, model=None,
                                       unnorm_key=None, headroom_gb=0.8,
                                       allow_cpu_offload=True))
    assert c["headroom_gb"] == 0.8
    assert c["cpu_offload"] is True
    assert c["gpus"] == [0, 6]
