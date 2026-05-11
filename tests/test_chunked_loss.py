"""
Numerical-equivalence test for ``CCSMLoss.compute_loss_chunked``.

When ``seq_chunk_size`` covers the whole sequence (chunk_size >= T), the
chunked path must produce gradients identical (up to float reduction
order) to the single-shot ``compute_loss``. This isolates the chunking
plumbing from the cross-chunk attention approximation: with one chunk
there's nothing approximated.

The test mocks the backbone — ``evaluate_hidden_chunked`` and
``evaluate_hidden_ref_chunked`` slice a pre-computed hidden tensor — so
the comparison is pure math on the loss layer (lm_head, value_head,
masking, returns, advantages, KL).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from marlllm.config import TrainingConfig
from marlllm.loss import CCSMLoss
from marlllm.types import TokenType


class _FakeAgent:
    def __init__(self, hidden_full, hidden_ref_full, lm_head_module, value_head_module, backbone_param):
        self._hidden_full = hidden_full
        self._hidden_ref_full = hidden_ref_full
        self._lm_head_module = lm_head_module
        self._value_head = value_head_module
        self._keep_ref_model = hidden_ref_full is not None
        # compute_loss_chunked inspects backbone parameters for dtype. Provide
        # a stand-in module with one parameter of the right dtype.
        self._backbone = nn.Sequential()
        self._backbone.register_parameter("dummy", nn.Parameter(backbone_param))
        self.device = hidden_full.device

    def evaluate_hidden_chunked(self, ids, attn, chunk_size):
        B, T, H = self._hidden_full.shape
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            hidden_chunk = self._hidden_full[:, s:e, :]
            values_chunk = self._value_head(hidden_chunk.detach())
            yield s, e, hidden_chunk, values_chunk

    def evaluate_hidden_ref_chunked(self, ids, attn, chunk_size):
        B, T, H = self._hidden_full.shape
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            ref = self._hidden_ref_full[:, s:e, :] if self._hidden_ref_full is not None else None
            yield s, e, ref

    def lm_head(self, h):
        return self._lm_head_module(h)

    def lm_head_ref(self, h):
        # Same module, same weights — KL term then computes KL between
        # the policy and (numerically) itself, which is ~0 but still has
        # a non-trivial autograd graph w.r.t. h. Fine for the equivalence
        # check; we just need both paths to do the same computation.
        return self._lm_head_module(h)

    def train_mode(self):
        pass


def _build_fake_world(T: int, H: int, V: int, seed: int = 0):
    torch.manual_seed(seed)
    B = 1
    # ``hidden_full`` is the LEAF tensor whose grad we read out, so that
    # slicing produces views that share *only* the leaf (no shared
    # intermediate ops). Each chunk's backward accumulates into
    # ``hidden_full.grad`` without freeing graph nodes the next chunk
    # will traverse.
    hidden_full = torch.randn(B, T, H, requires_grad=True)
    hidden_ref_full = torch.randn(B, T, H)  # frozen reference

    lm_head_module = nn.Linear(H, V, bias=False)
    value_head_module = nn.Linear(H, 1, bias=False)

    # Build masks: alternating OBS/ACT, agent_id always = target.
    types = torch.zeros(B, T, dtype=torch.long)
    types[:, ::2] = int(TokenType.OBS)
    types[:, 1::2] = int(TokenType.ACT)
    types[:, 0] = int(TokenType.OBS)  # position 0 is OBS, never ACT
    agents = torch.zeros(B, T, dtype=torch.long)
    target_idx = 0
    input_ids = torch.randint(0, V, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)

    # ValueHead is (H -> 1); compute_loss expects values shape (B, T).
    class _ValueWrapper(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.lin = lin

        def forward(self, x):
            return self.lin(x).squeeze(-1)

    value_head = _ValueWrapper(value_head_module)

    backbone_param = torch.zeros(1, dtype=torch.float32)
    agent = _FakeAgent(hidden_full, hidden_ref_full, lm_head_module, value_head, backbone_param)
    return agent, hidden_full, hidden_ref_full, lm_head_module, value_head, input_ids, attn, types, agents, target_idx


def test_chunked_equals_single_shot_grad():
    """compute_loss_chunked with chunk_size == T must match compute_loss."""
    T, H, V = 12, 16, 32
    cfg = TrainingConfig(
        gamma=0.9, beta=0.05, alpha_perc=1.0, alpha_act=1.0, alpha_val=1.0,
        normalise_returns=True, kl_coef=0.1,
    )

    # ----- Single-shot path -----
    agent_a, hidden_a, hidden_ref_a, lm_a, vh_a, ids, attn, types, agents, target = _build_fake_world(T, H, V, seed=42)
    values_a = vh_a(hidden_a.detach())
    loss = CCSMLoss()
    loss_val, _ = loss.compute_loss(
        last_hidden=hidden_a,
        lm_head=lm_a,
        values=values_a,
        input_ids=ids,
        token_type_mask=types,
        agent_id_mask=agents,
        target_agent_idx=target,
        config=cfg,
        last_hidden_ref=hidden_ref_a,
        lm_head_ref=lm_a,
    )
    loss_val.backward()
    grad_h_single = hidden_a.grad.detach().clone()
    grad_lm_single = lm_a.weight.grad.detach().clone()
    grad_vh_single = vh_a.lin.weight.grad.detach().clone()

    # ----- Chunked path, chunk_size == T -----
    cfg_chunked = TrainingConfig(
        gamma=0.9, beta=0.05, alpha_perc=1.0, alpha_act=1.0, alpha_val=1.0,
        normalise_returns=True, kl_coef=0.1, seq_chunk_size=T,
    )
    agent_b, hidden_b, hidden_ref_b, lm_b, vh_b, ids2, attn2, types2, agents2, target2 = _build_fake_world(T, H, V, seed=42)
    loss.compute_loss_chunked(
        agent=agent_b,
        input_ids=ids2,
        attention_mask=attn2,
        token_type_mask=types2,
        agent_id_mask=agents2,
        target_agent_idx=target2,
        config=cfg_chunked,
        backward_scale=1.0,
    )
    grad_h_chunked = hidden_b.grad.detach().clone()
    grad_lm_chunked = lm_b.weight.grad.detach().clone()
    grad_vh_chunked = vh_b.lin.weight.grad.detach().clone()

    assert torch.allclose(grad_h_single, grad_h_chunked, atol=1e-5, rtol=1e-4), \
        f"hidden grads diverge: max abs diff {(grad_h_single - grad_h_chunked).abs().max().item()}"
    assert torch.allclose(grad_lm_single, grad_lm_chunked, atol=1e-5, rtol=1e-4), \
        f"lm_head grads diverge: max abs diff {(grad_lm_single - grad_lm_chunked).abs().max().item()}"
    assert torch.allclose(grad_vh_single, grad_vh_chunked, atol=1e-5, rtol=1e-4), \
        f"value_head grads diverge: max abs diff {(grad_vh_single - grad_vh_chunked).abs().max().item()}"


def test_chunked_smaller_chunks_close_to_single_shot():
    """With multiple chunks (no cross-chunk dep here since hidden is precomputed),
    the chunked path should still match the single-shot path bitwise within
    floating-point reduction order."""
    T, H, V = 16, 16, 32
    cfg = TrainingConfig(
        gamma=0.9, beta=0.05, alpha_perc=1.0, alpha_act=1.0, alpha_val=1.0,
        normalise_returns=True, kl_coef=0.0,  # disable KL to simplify
    )
    agent_a, hidden_a, _, lm_a, vh_a, ids, attn, types, agents, target = _build_fake_world(T, H, V, seed=7)
    values_a = vh_a(hidden_a.detach())
    loss = CCSMLoss()
    loss_val, _ = loss.compute_loss(
        last_hidden=hidden_a, lm_head=lm_a, values=values_a,
        input_ids=ids, token_type_mask=types, agent_id_mask=agents,
        target_agent_idx=target, config=cfg,
        last_hidden_ref=None, lm_head_ref=None,
    )
    loss_val.backward()
    g_h = hidden_a.grad.detach().clone()

    cfg_chunked = TrainingConfig(
        gamma=0.9, beta=0.05, alpha_perc=1.0, alpha_act=1.0, alpha_val=1.0,
        normalise_returns=True, kl_coef=0.0, seq_chunk_size=4,  # 4 chunks of 4 tokens
    )
    agent_b, hidden_b, _, lm_b, vh_b, ids2, attn2, types2, agents2, target2 = _build_fake_world(T, H, V, seed=7)
    # Drop reference for simplicity
    agent_b._hidden_ref_full = None
    agent_b._keep_ref_model = False
    loss.compute_loss_chunked(
        agent=agent_b, input_ids=ids2, attention_mask=attn2,
        token_type_mask=types2, agent_id_mask=agents2,
        target_agent_idx=target2, config=cfg_chunked, backward_scale=1.0,
    )
    g_h_c = hidden_b.grad.detach().clone()
    assert torch.allclose(g_h, g_h_c, atol=1e-4, rtol=1e-3), \
        f"max abs diff {(g_h - g_h_c).abs().max().item()}"
