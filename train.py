"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Runtime selection and backend feature flags
# ---------------------------------------------------------------------------

ALLOWED_DEVICE_HINTS = {"auto", "cuda", "vulkan", "cpu"}

fa3 = None
_ALLOW_FA3_ATTENTION = False
_ALLOW_SDPA_ATTENTION = True
_FA3_FALLBACK_WARNED = False
_SDPA_FALLBACK_WARNED = False


@dataclass(frozen=True)
class RuntimeConfig:
    requested_device: str
    device: torch.device
    selection_reason: str
    cuda_probe: str
    vulkan_probe: str
    fa3_probe: str
    fa3_enabled: bool
    sdpa_enabled: bool
    amp_enabled: bool
    compile_enabled: bool
    muon_enabled: bool
    tokenizer_threads: int
    cpu_threads: int
    interop_threads: int
    nice_level: int


def _parse_env_flag(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {name}={raw!r}. Expected one of: 1,0,true,false,yes,no,on,off"
    )


def _parse_env_int(name, default, min_value=1):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}. Expected an integer.") from exc
    if value < min_value:
        raise ValueError(f"Invalid {name}={raw!r}. Expected integer >= {min_value}.")
    return value


def _parse_env_float(name, default, min_value=None, max_value=None):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}. Expected a number.") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"Invalid {name}={raw!r}. Expected >= {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"Invalid {name}={raw!r}. Expected <= {max_value}.")
    return value


def _get_total_system_memory_bytes():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _resolve_ram_budget():
    total_ram_bytes = _get_total_system_memory_bytes()
    budget_source = None
    budget_bytes = None

    ram_mb_raw = os.getenv("AUTORESEARCH_RAM_MB")
    if ram_mb_raw is not None and ram_mb_raw.strip():
        ram_mb = _parse_env_int("AUTORESEARCH_RAM_MB", 0, min_value=256)
        budget_bytes = ram_mb * 1024 * 1024
        budget_source = f"AUTORESEARCH_RAM_MB={ram_mb}"
    else:
        ram_fraction_raw = os.getenv("AUTORESEARCH_RAM_FRACTION")
        if ram_fraction_raw is not None and ram_fraction_raw.strip():
            ram_fraction = _parse_env_float(
                "AUTORESEARCH_RAM_FRACTION", 0.5, min_value=0.05, max_value=0.95
            )
            if total_ram_bytes is None:
                raise RuntimeError(
                    "AUTORESEARCH_RAM_FRACTION was set but total system RAM could not be detected."
                )
            budget_bytes = int(total_ram_bytes * ram_fraction)
            budget_source = f"AUTORESEARCH_RAM_FRACTION={ram_fraction:.3f}"

    if budget_bytes is not None:
        min_budget = 256 * 1024 * 1024
        budget_bytes = max(budget_bytes, min_budget)
        if total_ram_bytes is not None:
            budget_bytes = min(budget_bytes, int(total_ram_bytes * 0.95))
    return budget_bytes, total_ram_bytes, budget_source


def _shape_from_ram_budget_bytes(ram_budget_bytes, device_type):
    ram_budget_gib = ram_budget_bytes / (1024 ** 3)
    if device_type == "vulkan":
        # Vulkan backends on iGPU are usually memory-bandwidth/allocator constrained.
        if ram_budget_gib < 2:
            return 2, 1
        if ram_budget_gib < 4:
            return 2, 2
        if ram_budget_gib < 6:
            return 2, 3
        if ram_budget_gib < 8:
            return 3, 3
        if ram_budget_gib < 12:
            return 3, 4
        return 4, 4

    if ram_budget_gib < 2:
        return 2, 2
    if ram_budget_gib < 4:
        return 2, 4
    if ram_budget_gib < 6:
        return 3, 4
    if ram_budget_gib < 8:
        return 3, 6
    if ram_budget_gib < 12:
        return 4, 8
    return 4, 12


def _auto_total_batch_size(tokens_per_fwdbwd, cap_tokens):
    # Keep at least one micro-step and make total batch divisible by per-step tokens.
    cap_tokens = max(tokens_per_fwdbwd, cap_tokens)
    return (cap_tokens // tokens_per_fwdbwd) * tokens_per_fwdbwd


def _resolve_time_budget_seconds():
    return _parse_env_int("AUTORESEARCH_TIME_BUDGET_SECONDS", TIME_BUDGET, min_value=30)


def _format_exception(exc):
    return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


def _probe_cuda():
    if not torch.cuda.is_available():
        return False, "unavailable (torch.cuda.is_available() is False)"
    try:
        torch.empty(1, device="cuda")
        capability = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        return True, f"ready ({name}, cc={capability[0]}.{capability[1]})"
    except (RuntimeError, AssertionError) as exc:
        return False, f"unavailable ({_format_exception(exc)})"


def _probe_vulkan():
    if not hasattr(torch, "is_vulkan_available"):
        return False, "unavailable (torch.is_vulkan_available missing)"
    try:
        if not torch.is_vulkan_available():
            return False, "unavailable (torch.is_vulkan_available() is False)"
    except (RuntimeError, OSError, AssertionError) as exc:
        return False, f"unavailable ({_format_exception(exc)})"

    vulkan_backend = getattr(torch.backends, "vulkan", None)
    if vulkan_backend is None:
        return False, "unavailable (torch.backends.vulkan missing)"
    if not hasattr(vulkan_backend, "is_available"):
        return False, "unavailable (torch.backends.vulkan.is_available missing)"
    try:
        if not vulkan_backend.is_available():
            return False, "unavailable (torch.backends.vulkan.is_available() is False)"
    except (RuntimeError, OSError) as exc:
        return False, f"unavailable ({_format_exception(exc)})"

    device = torch.device("vulkan")
    try:
        test_tensor = torch.empty(1, device=device)
        _ = test_tensor + 1
    except (RuntimeError, NotImplementedError) as exc:
        return False, f"unavailable ({_format_exception(exc)})"

    try:
        vocab_size = 128
        hidden = 64
        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        embedding = nn.Embedding(vocab_size, hidden, device=device)
        projection = nn.Linear(hidden, vocab_size, bias=False, device=device)
        logits = projection(embedding(tokens))
        loss = F.cross_entropy(logits.view(-1, vocab_size).float(), targets.view(-1))
        loss.backward()
    except (RuntimeError, NotImplementedError, AssertionError) as exc:
        return False, f"unavailable (training-smoke failed: {_format_exception(exc)})"

    return True, "ready (training-smoke passed)"


def _try_init_fa3():
    try:
        from kernels import get_kernel
    except ImportError as exc:
        return None, f"disabled ({_format_exception(exc)})"

    try:
        capability = torch.cuda.get_device_capability()
        repo = "varunneal/flash-attention-3" if capability == (9, 0) else "kernels-community/flash-attn3"
        return get_kernel(repo).flash_attn_interface, f"enabled ({repo})"
    except (RuntimeError, OSError, AttributeError, ImportError) as exc:
        return None, f"disabled ({_format_exception(exc)})"


def resolve_runtime():
    requested_device = os.getenv("AUTORESEARCH_DEVICE", "auto").strip().lower()
    if requested_device not in ALLOWED_DEVICE_HINTS:
        allowed = ", ".join(sorted(ALLOWED_DEVICE_HINTS))
        raise ValueError(
            f"Invalid AUTORESEARCH_DEVICE={requested_device!r}. Expected one of: {allowed}"
        )

    cuda_available, cuda_probe = _probe_cuda()
    vulkan_available, vulkan_probe = _probe_vulkan()

    if requested_device == "auto":
        if cuda_available:
            selected_device = "cuda"
            selection_reason = "auto fallback selected cuda"
        elif vulkan_available:
            selected_device = "vulkan"
            selection_reason = "auto fallback selected vulkan (cuda unavailable)"
        else:
            selected_device = "cpu"
            selection_reason = "auto fallback selected cpu (cuda and vulkan unavailable)"
    elif requested_device == "cuda":
        if not cuda_available:
            raise RuntimeError(f"CUDA requested but unavailable: {cuda_probe}")
        selected_device = "cuda"
        selection_reason = "explicit cuda request"
    elif requested_device == "vulkan":
        if not vulkan_available:
            raise RuntimeError(f"Vulkan requested but unavailable: {vulkan_probe}")
        selected_device = "vulkan"
        selection_reason = "explicit vulkan request"
    else:
        selected_device = "cpu"
        selection_reason = "explicit cpu request"

    fa3_probe = "disabled (requires CUDA)"
    fa3_impl = None
    if selected_device == "cuda":
        fa3_impl, fa3_probe = _try_init_fa3()

    compile_enabled = _parse_env_flag(
        "AUTORESEARCH_COMPILE", selected_device == "cuda" and hasattr(torch, "compile")
    )
    amp_enabled = _parse_env_flag("AUTORESEARCH_AMP", selected_device == "cuda")
    muon_enabled = _parse_env_flag("AUTORESEARCH_USE_MUON", selected_device == "cuda")
    tokenizer_threads = _parse_env_int("AUTORESEARCH_TOKENIZER_THREADS", 8, min_value=1)
    cpu_threads = _parse_env_int(
        "AUTORESEARCH_CPU_THREADS", max(1, (os.cpu_count() or 1) // 2), min_value=1
    )
    interop_threads = _parse_env_int("AUTORESEARCH_INTEROP_THREADS", 1, min_value=1)
    nice_level = _parse_env_int("AUTORESEARCH_NICE", 0, min_value=0)

    if selected_device != "cuda" and compile_enabled:
        raise RuntimeError(
            "AUTORESEARCH_COMPILE=1 is only supported when AUTORESEARCH_DEVICE resolves to cuda."
        )
    if selected_device != "cuda" and amp_enabled:
        raise RuntimeError(
            "AUTORESEARCH_AMP=1 is only supported when AUTORESEARCH_DEVICE resolves to cuda."
        )
    if selected_device == "vulkan" and muon_enabled:
        raise RuntimeError(
            "AUTORESEARCH_USE_MUON=1 is not supported on Vulkan; use AUTORESEARCH_USE_MUON=0."
        )
    if selected_device == "cpu" and muon_enabled:
        raise RuntimeError(
            "AUTORESEARCH_USE_MUON=1 is not supported on CPU; use AUTORESEARCH_USE_MUON=0."
        )

    runtime = RuntimeConfig(
        requested_device=requested_device,
        device=torch.device(selected_device),
        selection_reason=selection_reason,
        cuda_probe=cuda_probe,
        vulkan_probe=vulkan_probe,
        fa3_probe=fa3_probe,
        fa3_enabled=fa3_impl is not None,
        sdpa_enabled=selected_device in {"cuda", "cpu"},
        amp_enabled=amp_enabled,
        compile_enabled=compile_enabled,
        muon_enabled=muon_enabled,
        tokenizer_threads=tokenizer_threads,
        cpu_threads=cpu_threads,
        interop_threads=interop_threads,
        nice_level=nice_level,
    )
    return runtime, fa3_impl


def initialize_attention_runtime(runtime, fa3_impl):
    global fa3, _ALLOW_FA3_ATTENTION, _ALLOW_SDPA_ATTENTION
    global _FA3_FALLBACK_WARNED, _SDPA_FALLBACK_WARNED
    fa3 = fa3_impl
    _ALLOW_FA3_ATTENTION = runtime.fa3_enabled
    _ALLOW_SDPA_ATTENTION = runtime.sdpa_enabled
    _FA3_FALLBACK_WARNED = False
    _SDPA_FALLBACK_WARNED = False


def log_runtime(runtime):
    print("Runtime diagnostics:")
    print(f"  AUTORESEARCH_DEVICE: {runtime.requested_device}")
    print(f"  CUDA probe:   {runtime.cuda_probe}")
    print(f"  Vulkan probe: {runtime.vulkan_probe}")
    print(f"  Selected:     {runtime.device} ({runtime.selection_reason})")
    print(
        "  Features: "
        f"fa3={runtime.fa3_enabled} ({runtime.fa3_probe}), "
        f"sdpa={runtime.sdpa_enabled}, "
        f"amp={runtime.amp_enabled}, "
        f"compile={runtime.compile_enabled}, "
        f"muon={runtime.muon_enabled}"
    )
    print(
        "  Host: "
        f"torch_threads={runtime.cpu_threads}, "
        f"interop_threads={runtime.interop_threads}, "
        f"tokenizer_threads={runtime.tokenizer_threads}, "
        f"nice={runtime.nice_level}"
    )

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def _build_attention_invalid_mask(seq_len, window_size, device):
    row = torch.arange(seq_len, device=device).view(seq_len, 1)
    col = torch.arange(seq_len, device=device).view(1, seq_len)
    invalid = col > row  # causal

    left_window = None
    if isinstance(window_size, (tuple, list)) and len(window_size) > 0:
        left_window = window_size[0]
    elif isinstance(window_size, int):
        left_window = window_size

    if isinstance(left_window, int) and left_window >= 0:
        invalid = invalid | (col < (row - left_window))
    return invalid


def _attention_forward(q, k, v, window_size):
    """
    Compute causal attention with fallback chain: FA3 -> SDPA -> manual.
    q, k, v: (B, T, n_head/n_kv_head, head_dim) - already has rotary applied & normalized
    window_size: window_size for windowed attention
    """
    global _ALLOW_FA3_ATTENTION, _ALLOW_SDPA_ATTENTION
    global _FA3_FALLBACK_WARNED, _SDPA_FALLBACK_WARNED

    # Handle grouped-query attention in fallback kernels.
    if q.size(2) != k.size(2):
        if q.size(2) % k.size(2) != 0:
            raise RuntimeError(
                f"Incompatible q/k head counts for fallback attention: q={q.size(2)}, k={k.size(2)}"
            )
        repeat_factor = q.size(2) // k.size(2)
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

    # Try FA3 (CUDA only)
    if _ALLOW_FA3_ATTENTION and fa3 is not None:
        try:
            return fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        except (RuntimeError, NotImplementedError, AssertionError) as exc:
            _ALLOW_FA3_ATTENTION = False
            if not _FA3_FALLBACK_WARNED:
                print(f"WARNING: disabling FA3 attention fallback: {_format_exception(exc)}")
                _FA3_FALLBACK_WARNED = True

    # Try scaled_dot_product_attention (SDPA) - supports CUDA, CPU, and others
    if _ALLOW_SDPA_ATTENTION:
        B, T, n_head, head_dim = q.shape
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        invalid = _build_attention_invalid_mask(T, window_size, q.device)
        attn_mask = torch.zeros((T, T), dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(invalid, float("-inf"))
        try:
            y = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            return y.transpose(1, 2)
        except (RuntimeError, NotImplementedError, AssertionError) as exc:
            _ALLOW_SDPA_ATTENTION = False
            if not _SDPA_FALLBACK_WARNED:
                print(f"WARNING: disabling SDPA attention fallback: {_format_exception(exc)}")
                _SDPA_FALLBACK_WARNED = True

    # Fallback: manual causal attention (always works)
    B, T, n_head, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    q_flat = q.transpose(1, 2)
    k_flat = k.transpose(1, 2)
    v_flat = v.transpose(1, 2)

    scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale  # (B, n_head, T, T)

    invalid = _build_attention_invalid_mask(T, window_size, q.device)
    scores = scores.masked_fill(invalid, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    y = torch.matmul(attn_weights, v_flat)
    return y.transpose(1, 2)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Attention backend fallback chain: FA3 -> SDPA -> manual causal attention
        y = _attention_forward(q, k, v, window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        dtype = self.transformer.wte.weight.dtype
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        muon_enabled=True):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if muon_enabled:
            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
        else:
            print("Muon disabled for this backend; falling back to AdamW for matrix parameters.")
            param_groups.append(dict(
                kind='adamw', params=matrix_params, lr=matrix_lr,
                betas=adam_betas, eps=1e-10, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        
    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step on CUDA
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size on CUDA


def resolve_training_shape(runtime):
    total_batch_size_explicit = os.getenv("AUTORESEARCH_TOTAL_BATCH_SIZE")

    if runtime.device.type == "cuda":
        default_depth = DEPTH
        default_device_batch_size = DEVICE_BATCH_SIZE
        default_total_batch_size_cap = TOTAL_BATCH_SIZE
    elif runtime.device.type == "vulkan":
        # More conservative defaults than CPU because Vulkan iGPU allocators are sensitive.
        default_depth = min(DEPTH, 3)
        default_device_batch_size = min(DEVICE_BATCH_SIZE, 4)
        default_total_batch_size_cap = min(TOTAL_BATCH_SIZE, 2**14)
    else:
        # Conservative defaults for CPU to avoid OOM on smaller machines.
        default_depth = min(DEPTH, 4)
        default_device_batch_size = min(DEVICE_BATCH_SIZE, 16)
        default_total_batch_size_cap = min(TOTAL_BATCH_SIZE, 2**15)

    ram_budget_bytes, total_ram_bytes, budget_source = _resolve_ram_budget()
    if ram_budget_bytes is not None:
        ram_depth, ram_device_batch_size = _shape_from_ram_budget_bytes(
            ram_budget_bytes, runtime.device.type
        )
        default_depth = min(default_depth, ram_depth)
        default_device_batch_size = min(default_device_batch_size, ram_device_batch_size)

    depth = _parse_env_int("AUTORESEARCH_DEPTH", default_depth, min_value=1)
    device_batch_size = _parse_env_int(
        "AUTORESEARCH_DEVICE_BATCH_SIZE", default_device_batch_size, min_value=1
    )
    tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
    if total_batch_size_explicit is None:
        total_batch_size = _auto_total_batch_size(tokens_per_fwdbwd, default_total_batch_size_cap)
    else:
        total_batch_size = _parse_env_int(
            "AUTORESEARCH_TOTAL_BATCH_SIZE", default_total_batch_size_cap, min_value=1
        )
    if total_batch_size % tokens_per_fwdbwd != 0:
        raise ValueError(
            "AUTORESEARCH_TOTAL_BATCH_SIZE must be divisible by "
            f"AUTORESEARCH_DEVICE_BATCH_SIZE * MAX_SEQ_LEN ({tokens_per_fwdbwd}). "
            f"Got {total_batch_size}."
        )
    return depth, device_batch_size, total_batch_size, ram_budget_bytes, total_ram_bytes, budget_source

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
if hasattr(torch, "vulkan") and hasattr(torch.vulkan, "manual_seed_all"):
    torch.vulkan.manual_seed_all(42)
torch.set_float32_matmul_precision("high")

# Device and backend selection
runtime, fa3_impl = resolve_runtime()
initialize_attention_runtime(runtime, fa3_impl)
device = runtime.device
time_budget_seconds = _resolve_time_budget_seconds()

torch.set_num_threads(runtime.cpu_threads)
try:
    torch.set_num_interop_threads(runtime.interop_threads)
except RuntimeError as exc:
    print(f"Warning: could not set torch interop threads to {runtime.interop_threads}: {exc}")
if runtime.nice_level > 0:
    try:
        os.nice(runtime.nice_level)
    except OSError as exc:
        print(f"Warning: unable to increase niceness by {runtime.nice_level}: {exc}")

log_runtime(runtime)
depth, device_batch_size, total_batch_size, ram_budget_bytes, total_ram_bytes, budget_source = resolve_training_shape(runtime)
print(
    "Training shape: "
    f"depth={depth}, device_batch_size={device_batch_size}, total_batch_size={total_batch_size}"
)
if total_ram_bytes is not None:
    print(f"System RAM: {total_ram_bytes / (1024 ** 3):.1f} GiB")
if ram_budget_bytes is not None:
    print(
        "RAM budget: "
        f"{ram_budget_bytes / (1024 ** 3):.1f} GiB"
        + (f" ({budget_source})" if budget_source else "")
    )

# Setup autocast context
if runtime.amp_enabled:
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    autocast_ctx = nullcontext()

H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(depth)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
grad_accum_steps = total_batch_size // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    muon_enabled=runtime.muon_enabled,
)

if runtime.compile_enabled:
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(
    tokenizer,
    device_batch_size,
    MAX_SEQ_LEN,
    "train",
    device=device.type,
    tokenizer_threads=runtime.tokenizer_threads,
)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {time_budget_seconds}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / time_budget_seconds)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / time_budget_seconds, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(total_batch_size / dt)
    mfu = (
        100 * num_flops_per_token * total_batch_size / dt / H100_BF16_PEAK_FLOPS
        if device.type == "cuda"
        else 0.0
    )
    remaining = max(0, time_budget_seconds - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= time_budget_seconds:
        break

print()  # newline after \r training log

total_tokens = step * total_batch_size

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(
        model,
        tokenizer,
        device_batch_size,
        device=device.type,
        tokenizer_threads=runtime.tokenizer_threads,
    )

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100 * num_flops_per_token * total_batch_size * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
if device.type != "cuda":
    steady_state_mfu = 0.0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {depth}")
