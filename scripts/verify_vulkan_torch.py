#!/usr/bin/env python3
"""
Verify whether this Python environment has a usable Vulkan-backed PyTorch.

This script is stricter than checking `torch.device("vulkan")`, because many builds
expose the device type string but still miss key operators.
"""

import math
import sys

def fail(message: str) -> None:
    print(f"[verify-vulkan] FAIL: {message}")
    sys.exit(1)


def main() -> None:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        fail(f"importing torch failed: {type(e).__name__}: {e}")

    print(f"[verify-vulkan] torch: {torch.__version__}")

    try:
        device = torch.device("vulkan")
    except Exception as e:
        fail(f"torch.device('vulkan') failed: {type(e).__name__}: {e}")

    try:
        _ = torch.tensor([1.0], device=device)
    except Exception as e:
        fail(f"tensor allocation on Vulkan failed: {type(e).__name__}: {str(e).splitlines()[0]}")

    try:
        x = torch.randn(2, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)
        y = (x @ w).pow(2).mean()
        y.backward()
    except Exception as e:
        fail(f"basic matmul/backward failed on Vulkan: {type(e).__name__}: {str(e).splitlines()[0]}")

    # Try SDPA; if unavailable, verify manual attention still works.
    q = torch.randn(1, 2, 8, 16, device=device)
    k = torch.randn(1, 2, 8, 16, device=device)
    v = torch.randn(1, 2, 8, 16, device=device)

    try:
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print("[verify-vulkan] SDPA: available")
    except Exception as e:
        print(f"[verify-vulkan] SDPA unavailable ({type(e).__name__}), checking manual attention...")
        try:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            causal_mask = torch.triu(
                torch.ones((q.size(-2), q.size(-2)), dtype=torch.bool, device=device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
            _ = torch.matmul(scores.softmax(dim=-1), v)
        except Exception as manual_e:
            fail(
                "both SDPA and manual attention path failed on Vulkan: "
                f"{type(manual_e).__name__}: {str(manual_e).splitlines()[0]}"
            )

    print("[verify-vulkan] PASS: Vulkan backend is usable for autoresearch.")


if __name__ == "__main__":
    main()
