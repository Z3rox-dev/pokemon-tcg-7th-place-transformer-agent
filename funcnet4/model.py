"""FuncNetTF: the text-first policy and value network.

A card enters as its precomputed features (functional, effect-atom, text embedding) rather than as a
learned id, which is what lets the network play cards it never saw in training. The armed blocks
(act_pointer, ptr_belief, gru_zones) are inferred from a checkpoint's own tensors when one is
loaded, so weights and configuration cannot drift apart.
"""
import math

import torch

from bytrl.model import RecurrentNet
from funcnet4.spec import (ENC_DENSE_DIM, ENC_DENSE_DIM_TF, DEC_DENSE_DIM_TF,
                           N_ROLE, N_PLANE, PLANE_PAD, NUM_WORDS, N_ZONES, ZONE_WORDS)
from funcnet.effects_spec import EFF_VOCAB_DIM, EFF_DELTA_DIM


_ID_GUARD_SEEN = set()


def _guard_ids(t, n, name):
    """Clamp embedding ids into range, logging loudly when anything was out of bounds."""
    mn = int(t.min())
    mx = int(t.max())
    if mn < 0 or mx >= n:
        key = (name, mn, mx)
        if key not in _ID_GUARD_SEEN:
            _ID_GUARD_SEEN.add(key)
            bad = t[(t < 0) | (t >= n)]
            vals = sorted(set(bad.detach().cpu().tolist()))[:10]
            print(f"[id-guard] {name} out of range {vals} (valid [0,{n})) in {bad.numel()} ids; "
                  f"clamped", flush=True)
        return t.clamp(0, n - 1)
    return t


_LEGACY_ATTN_KEYS = frozenset({
    "act_sa_norm.weight", "act_sa_norm.bias",
    "act_sa.in_proj_weight", "act_sa.in_proj_bias",
    "act_sa.out_proj.weight", "act_sa.out_proj.bias",
})
_COMPARE_ATTN_KEYS = frozenset({
    "act_compare_norm.weight", "act_compare_norm.bias",
    "act_compare_sa.in_proj_weight", "act_compare_sa.in_proj_bias",
    "act_compare_sa.out_proj.weight", "act_compare_sa.out_proj.bias",
    "act_compare_fc.weight", "act_compare_fc.bias",
})
PTR_KEY_DIM = 32
_PTR_SYNTH_SEED = 0x7F4A11
_POINTER_KEYS = frozenset({
    "ptr_key_fc.weight",
    "ptr_query_mlp.0.weight", "ptr_query_mlp.0.bias",
    "ptr_query_mlp.2.weight", "ptr_query_mlp.2.bias",
    "ptr_bias",
})
BELIEF_KEY_DIM = 32
_BELIEF_KEYS = frozenset({
    "belief_key.weight",
    "belief_query.weight", "belief_query.bias",
})
_ZONE_SYNTH_SEED = 0x60A1E5
_ZONES_KEYS = frozenset({"zone_proj.weight", "zone_proj.bias"})


def _sd_d_model(sd):
    w = sd.get("encoder.layers.0.self_attn.in_proj_weight")
    if w is None:
        raise RuntimeError("state_dict has no encoder: cannot infer d_model")
    return int(w.shape[1])


def _sd_rec_hidden(sd):
    w = sd.get("rnn.weight_hh_l0")
    return int(w.shape[1]) if w is not None else None


def _infer_complete_attention_block(sd, expected, prefixes, label):
    present = {k for k in sd if k.startswith(prefixes)}
    if not present:
        return 0
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise RuntimeError(f"checkpoint {label} incomplete: missing={missing} extra={extra}")
    return 1


def infer_act_selfattn(sd) -> int:
    """1 if the state dict carries the legacy action self-attention block."""
    return _infer_complete_attention_block(
        sd, _LEGACY_ATTN_KEYS, ("act_sa.", "act_sa_norm."), "act_sa legacy")


def infer_act_compare(sd) -> int:
    """1 if the state dict carries the action-comparison branch."""
    return _infer_complete_attention_block(
        sd, _COMPARE_ATTN_KEYS, ("act_compare_",), "act_compare v2")


def infer_act_pointer(sd) -> int:
    """1 if the state dict carries the complete pointer block."""
    return _infer_complete_attention_block(
        sd, _POINTER_KEYS, ("ptr_",), "act_pointer")


def infer_ptr_belief(sd, rec_hidden=None) -> int:
    """1 if the pointer query also reads the belief state (wider fc1)."""
    w = sd.get("ptr_query_mlp.0.weight")
    if w is None:
        return 0
    d_model = _sd_d_model(sd)
    width = int(w.shape[1])
    if width == d_model:
        return 0
    rh = _sd_rec_hidden(sd)
    if rh is None and rec_hidden:
        rh = int(rec_hidden)
    if rh is not None and width == d_model + rh:
        return 1
    raise RuntimeError(f"ptr_query_mlp.0.weight has unknown width {width} "
                       f"(expected {d_model} or d_model+rec_hidden; rnn in state_dict: "
                       f"{_sd_rec_hidden(sd) is not None}, rec_hidden hint: {rec_hidden})")


def infer_gru_zones(sd) -> int:
    """1 if the GRU input is [summary || zones]: weight_ih 2*d_model wide plus zone_proj.

    The block is complete or absent; any partial combination is a corrupt checkpoint and raises."""
    zp = {k for k in sd if k.startswith("zone_proj.")}
    if zp and zp != _ZONES_KEYS:
        raise RuntimeError(f"checkpoint gru_zones incomplete: zone_proj={sorted(zp)}")
    w = sd.get("rnn.weight_ih_l0")
    if w is None:
        if zp:
            raise RuntimeError("corrupt checkpoint: zone_proj present without rnn.*")
        return 0
    d_model = _sd_d_model(sd)
    width = int(w.shape[1])
    if width == d_model:
        if zp:
            raise RuntimeError("corrupt checkpoint: zone_proj present but rnn.weight_ih_l0 "
                               f"is narrow ({width})")
        return 0
    if width == 2 * d_model:
        if not zp:
            raise RuntimeError("corrupt checkpoint: rnn.weight_ih_l0 is 2*d_model wide "
                               "without zone_proj")
        return 1
    raise RuntimeError(f"rnn.weight_ih_l0 has unknown width {width} "
                       f"(expected {d_model} or {2 * d_model})")


def infer_action_attention(sd):
    """Constructor kwargs for an exact checkpoint architecture; rejects ambiguous checkpoints."""
    legacy, compare = infer_act_selfattn(sd), infer_act_compare(sd)
    pointer = infer_act_pointer(sd)
    if legacy and compare:
        raise RuntimeError("checkpoint contains both act_sa legacy and act_compare v2")
    if legacy and pointer:
        raise RuntimeError("checkpoint contains both act_sa legacy and act_pointer")
    return {"act_selfattn": legacy, "act_compare": compare, "act_pointer": pointer,
            "ptr_belief": infer_ptr_belief(sd), "gru_zones": infer_gru_zones(sd)}


_DECODER_INPUT_KEY = "dec_action_mlp.0.weight"
_PRE_DCX_WIDTH = 1058
_LEGACY_DCX_END = 1062
_LEGACY_DCX_UNLOCK_WIDTH = 1065
_PRE_TGT_WIDTH = 1061
_RELEASED_WIDTHS = {1051, 1054, 1058, 1062, 1065, _PRE_TGT_WIDTH, 1076}
_KNOWN_DECODER_WIDTHS = _RELEASED_WIDTHS | {DEC_DENSE_DIM_TF}
_ROLE_EMBED_KEY = "enc_role_embed.weight"
_KNOWN_ROLE_ROWS = {12, 13, 15}

_ENCODER_INPUT_KEY = "enc_card_mlp.0.weight"
_RELEASED_ENCODER_WIDTHS = {1109}
_KNOWN_ENCODER_WIDTHS = _RELEASED_ENCODER_WIDTHS | {ENC_DENSE_DIM_TF}


def migrate_decoder_input_layout(state_dict, *, log=False):
    """Return a copy with the decoder reader migrated to the DCX-free 1061-column layout."""
    w = state_dict.get(_DECODER_INPUT_KEY)
    if w is None:
        return state_dict
    if not torch.is_tensor(w) or w.ndim != 2:
        raise RuntimeError(f"{_DECODER_INPUT_KEY}: expected a rank-2 tensor, got {type(w).__name__} "
                           f"shape={getattr(w, 'shape', None)}")
    src = int(w.shape[1])
    if src not in _KNOWN_DECODER_WIDTHS:
        raise RuntimeError(f"{_DECODER_INPUT_KEY}: unsupported decoder input width {src}; "
                           f"known layouts={sorted(_KNOWN_DECODER_WIDTHS)}")
    if src == DEC_DENSE_DIM_TF:
        return state_dict
    if DEC_DENSE_DIM_TF < _PRE_TGT_WIDTH:
        raise RuntimeError(f"decoder migration expects target width >= {_PRE_TGT_WIDTH}, "
                           f"got {DEC_DENSE_DIM_TF}")

    if src == _LEGACY_DCX_UNLOCK_WIDTH:
        migrated = torch.cat([w[:, :_PRE_DCX_WIDTH], w[:, _LEGACY_DCX_END:]], dim=1)
        detail = "drop DCX[1058:1062], preserve UNLOCK[1062:1065]"
        if migrated.shape[1] < DEC_DENSE_DIM_TF:
            migrated = torch.cat(
                [migrated, migrated.new_zeros((w.shape[0], DEC_DENSE_DIM_TF - migrated.shape[1]))], dim=1)
            detail += " + zero-pad code successive"
    elif src == _LEGACY_DCX_END:
        migrated = torch.cat([w[:, :_PRE_DCX_WIDTH],
                              w.new_zeros((w.shape[0], DEC_DENSE_DIM_TF - _PRE_DCX_WIDTH))], dim=1)
        detail = "drop DCX[1058:1062], add zero UNLOCK (+ code successive)"
    else:
        migrated = torch.cat([w, w.new_zeros((w.shape[0], DEC_DENSE_DIM_TF - src))], dim=1)
        detail = "append zero columns for missing tails"
    if migrated.shape[1] != DEC_DENSE_DIM_TF:
        raise AssertionError(f"decoder migration produced {migrated.shape[1]} columns")
    out = dict(state_dict)
    out[_DECODER_INPUT_KEY] = migrated
    if log:
        print(f"[decoder-layout] {_DECODER_INPUT_KEY} {src}->{DEC_DENSE_DIM_TF}: {detail}", flush=True)
    return out


def migrate_encoder_input_layout(state_dict, *, log=False):
    """Copy with the encoder reader migrated to the current row width."""
    w = state_dict.get(_ENCODER_INPUT_KEY)
    if w is None:
        return state_dict
    if not torch.is_tensor(w) or w.ndim != 2:
        raise RuntimeError(f"{_ENCODER_INPUT_KEY}: expected a rank-2 tensor, got {type(w).__name__} "
                           f"shape={getattr(w, 'shape', None)}")
    src = int(w.shape[1])
    if src not in _KNOWN_ENCODER_WIDTHS:
        raise RuntimeError(f"{_ENCODER_INPUT_KEY}: unsupported encoder input width {src}; "
                           f"known layouts={sorted(_KNOWN_ENCODER_WIDTHS)}")
    if src == ENC_DENSE_DIM_TF:
        return state_dict
    if src > ENC_DENSE_DIM_TF:
        raise RuntimeError(f"{_ENCODER_INPUT_KEY}: refusing destructive crop {src}->{ENC_DENSE_DIM_TF}; "
                           "set FUNCNET4_LOC_FEAT to match the checkpoint before importing funcnet4")
    migrated = torch.cat([w, w.new_zeros((w.shape[0], ENC_DENSE_DIM_TF - src))], dim=1)
    if migrated.shape[1] != ENC_DENSE_DIM_TF:
        raise AssertionError(f"encoder migration produced {migrated.shape[1]} columns")
    out = dict(state_dict)
    out[_ENCODER_INPUT_KEY] = migrated
    if log:
        print(f"[encoder-layout] {_ENCODER_INPUT_KEY} {src}->{ENC_DENSE_DIM_TF}: "
              f"append zero columns (LOC-tail)", flush=True)
    return out


def migrate_role_embed_layout(state_dict, target_rows, target_dim, *, log=False):
    """Pad only released 12/13/15-row role layouts; never crop or accept invented counts."""
    w = state_dict.get(_ROLE_EMBED_KEY)
    if w is None:
        return state_dict
    if not torch.is_tensor(w) or w.ndim != 2:
        raise RuntimeError(f"{_ROLE_EMBED_KEY}: expected rank-2 tensor, got "
                           f"{type(w).__name__} shape={getattr(w, 'shape', None)}")
    src, dim = map(int, w.shape)
    target_rows, target_dim = int(target_rows), int(target_dim)
    if src not in _KNOWN_ROLE_ROWS or target_rows not in _KNOWN_ROLE_ROWS:
        raise RuntimeError(f"{_ROLE_EMBED_KEY}: unsupported role rows {src}->{target_rows}; "
                           f"known layouts={sorted(_KNOWN_ROLE_ROWS)}")
    if dim != target_dim:
        raise RuntimeError(f"{_ROLE_EMBED_KEY}: embedding width {dim} != target {target_dim}")
    if src > target_rows:
        raise RuntimeError(f"{_ROLE_EMBED_KEY}: refusing destructive role crop {src}->{target_rows}; "
                           "set FUNCNET4_DKR_FEAT to match the checkpoint before importing funcnet4")
    if src == target_rows:
        return state_dict
    padded = w.new_zeros((target_rows, target_dim))
    padded[:src] = w
    out = dict(state_dict)
    out[_ROLE_EMBED_KEY] = padded
    if log:
        print(f"[role-layout] {_ROLE_EMBED_KEY} pad {src}->{target_rows} rows with zeros", flush=True)
    return out


_VALUE_BINS_KEYS = frozenset({"value_dist_head.weight", "value_dist_head.bias", "value_bin_centers"})


def _fit_bins_shape(n_bins, *, device=None, dtype=torch.float64, tol=0.01):
    """Fit (g, beta) so that the binned value head reproduces tanh in expectation."""
    c = torch.linspace(-1.0, 1.0, n_bins, dtype=dtype)
    s = torch.cat([torch.linspace(-8, -3, 200, dtype=dtype),
                   torch.linspace(-3, 3, 1200, dtype=dtype),
                   torch.linspace(3, 8, 200, dtype=dtype)]).unique()
    tgt = torch.tanh(s)
    g = (3.0 * c).clone().requires_grad_(True)
    beta = torch.zeros(n_bins, dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([g, beta], lr=0.02)
    for _ in range(4000):
        E = (torch.softmax(s[:, None] * g[None, :] + beta[None, :], dim=1) * c[None, :]).sum(1)
        err = E - tgt
        loss = (err ** 2).mean() + 0.5 * torch.logsumexp(err.abs() * 50, dim=0) / 50
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        E = (torch.softmax(s[:, None] * g[None, :] + beta[None, :], dim=1) * c[None, :]).sum(1)
        emax = float((E - tgt).abs().max())
    if emax > tol:
        raise RuntimeError(f"value-bins init: the {n_bins}-bin head does not reproduce the scalar head "
                           f"(max error {emax:.4f} > {tol})")
    return g.detach(), beta.detach(), emax


def migrate_value_bins(state_dict, want_bins, *, log=False):
    """Deterministically synthesize or strip the binned value head."""
    has = "value_dist_head.weight" in state_dict
    if not want_bins:
        if not has:
            return state_dict
        out = {k: v for k, v in state_dict.items() if k not in _VALUE_BINS_KEYS}
        if log:
            print("[value-bins] strip the binned head, back to the scalar head", flush=True)
        return out
    if has:
        n = int(state_dict["value_dist_head.weight"].shape[0])
        if n != int(want_bins):
            raise RuntimeError(f"value-bins: checkpoint has {n} bins, {want_bins} requested")
        return state_dict

    src = "value_fc.weight" if "value_fc.weight" in state_dict else "encoder_fc.weight"
    bsrc = src.replace(".weight", ".bias")
    if src not in state_dict:
        raise RuntimeError("value-bins: no scalar head (value_fc/encoder_fc) to synthesize from")
    w = state_dict[src].detach().to(torch.float64).reshape(-1)
    b0 = float(state_dict[bsrc].detach().reshape(-1)[0]) if bsrc in state_dict else 0.0

    n_bins = int(want_bins)
    g, beta, emax = _fit_bins_shape(n_bins)
    W = g[:, None] * w[None, :]
    b = g * b0 + beta

    out = dict(state_dict)
    ref = state_dict[src]
    out["value_dist_head.weight"] = W.to(dtype=ref.dtype, device=ref.device)
    out["value_dist_head.bias"] = b.to(dtype=ref.dtype, device=ref.device)
    out["value_bin_centers"] = torch.linspace(-1.0, 1.0, n_bins).to(dtype=ref.dtype, device=ref.device)
    if log:
        print(f"[value-bins] synthesized {n_bins} bins from {src} reproducing the scalar head "
              f"(max error {emax:.5f} on V)", flush=True)
    return out


def migrate_act_pointer(state_dict, want_pointer, *, log=False):
    """Deterministically synthesize or strip the pointer block, for arming and rollback."""
    has_ptr = infer_act_pointer(state_dict)
    if not want_pointer:
        if not has_ptr:
            return state_dict
        out = {k: v for k, v in state_dict.items() if k not in _POINTER_KEYS}
        if log:
            print(f"[act-pointer] strip {len(_POINTER_KEYS)} ptr_* keys (act_pointer=0: scoring "
                  "falls back to decoder_fc)", flush=True)
        return out
    if has_ptr:
        return state_dict
    w = state_dict.get("decoder_fc.weight")
    b = state_dict.get("decoder_fc.bias")
    if w is None or b is None:
        raise RuntimeError("act_pointer=1: cannot synthesize, decoder_fc.weight/bias "
                           "missing from the input state_dict")
    if not torch.is_tensor(w) or w.ndim != 2 or int(w.shape[0]) != 1:
        raise RuntimeError(f"decoder_fc.weight: expected shape (1, d_model), got "
                           f"{tuple(getattr(w, 'shape', ()))}")
    d_model = int(w.shape[1])
    bound = 1.0 / math.sqrt(d_model)
    g = torch.Generator().manual_seed(_PTR_SYNTH_SEED)
    key_w = torch.empty(PTR_KEY_DIM, d_model, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    q1_w = torch.empty(PTR_KEY_DIM, d_model, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    q1_b = torch.empty(PTR_KEY_DIM, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    key_w = key_w.to(dtype=w.dtype, device=w.device)
    key_w[0] = w.detach().reshape(-1)
    q2_w = torch.zeros(PTR_KEY_DIM, PTR_KEY_DIM, dtype=w.dtype, device=w.device)
    q2_b = torch.zeros(PTR_KEY_DIM, dtype=w.dtype, device=w.device)
    q2_b[0] = 1.0
    out = dict(state_dict)
    out["ptr_key_fc.weight"] = key_w
    out["ptr_query_mlp.0.weight"] = q1_w.to(dtype=w.dtype, device=w.device)
    out["ptr_query_mlp.0.bias"] = q1_b.to(dtype=w.dtype, device=w.device)
    out["ptr_query_mlp.2.weight"] = q2_w
    out["ptr_query_mlp.2.bias"] = q2_b
    out["ptr_bias"] = b.detach().clone().reshape(1)
    if log:
        print(f"[act-pointer] synthesized ptr_* (seed {_PTR_SYNTH_SEED:#x}, d_model {d_model}): "
              f"key row 0 = w, ptr_bias = b, query = e0, so the pointer equals decoder_fc at arm time",
              flush=True)
    return out


def migrate_ptr_belief(state_dict, want_ptr_belief, rec_hidden=None, *, log=False):
    """Widen or crop the pointer query for arming and rolling back ptr_belief."""
    has = infer_ptr_belief(state_dict, rec_hidden=rec_hidden)
    w = state_dict.get("ptr_query_mlp.0.weight")
    if not want_ptr_belief:
        if not has:
            return state_dict
        d_model = _sd_d_model(state_dict)
        out = dict(state_dict)
        out["ptr_query_mlp.0.weight"] = w[:, :d_model].clone()
        if log:
            print(f"[ptr-belief] crop {int(w.shape[1]) - d_model} belief columns from "
                  "ptr_query_mlp.0 (ptr_belief=0: the query reads the summary only)", flush=True)
        return out
    if w is None:
        raise RuntimeError("ptr_belief=1 needs the pointer block in the state_dict: call "
                           "migrate_ptr_belief after migrate_act_pointer (act_pointer=1)")
    if has:
        return state_dict
    rh = _sd_rec_hidden(state_dict)
    if rh is None:
        rh = int(rec_hidden) if rec_hidden else 0
    if not rh:
        raise RuntimeError("ptr_belief=1: cannot infer rec_hidden (no rnn.* in state_dict) "
                           "and the caller did not pass it")
    out = dict(state_dict)
    out["ptr_query_mlp.0.weight"] = torch.cat(
        [w, w.new_zeros((int(w.shape[0]), rh))], dim=1)
    if log:
        print(f"[ptr-belief] append {rh} zero columns to ptr_query_mlp.0 (query unchanged at arm time)",
              flush=True)
    return out


def migrate_gru_zones(state_dict, want_zones, *, log=False):
    """Arm or roll back the per-zone GRU input: wider weight_ih plus the zone projection."""
    has = infer_gru_zones(state_dict)
    if not want_zones:
        if not has:
            return state_dict
        d_model = _sd_d_model(state_dict)
        out = {k: v for k, v in state_dict.items() if k not in _ZONES_KEYS}
        w = out["rnn.weight_ih_l0"]
        out["rnn.weight_ih_l0"] = w[:, :d_model].clone()
        if log:
            print("[gru-zones] strip zone_proj and crop the zone columns of rnn.weight_ih_l0 "
                  "(gru_zones=0: the core reads the global mean only)", flush=True)
        return out
    if has:
        return state_dict
    w = state_dict.get("rnn.weight_ih_l0")
    if w is None:
        return state_dict
    d_model = _sd_d_model(state_dict)
    g = torch.Generator().manual_seed(_ZONE_SYNTH_SEED)
    fan_in = N_ZONES * d_model
    bound = 1.0 / math.sqrt(fan_in)
    zp_w = torch.empty(d_model, fan_in, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    zp_b = torch.empty(d_model, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    out = dict(state_dict)
    out["rnn.weight_ih_l0"] = torch.cat([w, w.new_zeros((int(w.shape[0]), d_model))], dim=1)
    out["zone_proj.weight"] = zp_w.to(dtype=w.dtype, device=w.device)
    out["zone_proj.bias"] = zp_b.to(dtype=w.dtype, device=w.device)
    if log:
        print(f"[gru-zones] append {d_model} zero columns to rnn.weight_ih_l0 and synthesize zone_proj "
              f"(seed {_ZONE_SYNTH_SEED:#x}, {N_ZONES} zones): input unchanged at arm time", flush=True)
    return out


class FuncNetTF(RecurrentNet):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int,
                 num_layers_encoder: int, num_layers_decoder: int,
                 recurrent: str = "none", rec_hidden: int | None = None,
                 num_words: int | None = None, value_bins: int = 0, act_selfattn: int = 0,
                 act_compare: int = 0, act_pointer: int = 0, ptr_belief: int = 0,
                 gru_zones: int = 0):
        super().__init__(d_model, num_heads, d_feedforward,
                         num_layers_encoder, num_layers_decoder,
                         recurrent, rec_hidden, num_words or NUM_WORDS)
        self.value_bins = int(value_bins)
        if self.value_bins > 0:
            _vin = d_model if recurrent == "none" else (rec_hidden or d_model)
            self.value_dist_head = torch.nn.Linear(_vin, self.value_bins)
            torch.nn.init.zeros_(self.value_dist_head.weight)
            torch.nn.init.zeros_(self.value_dist_head.bias)
            self.register_buffer("value_bin_centers", torch.linspace(-1.0, 1.0, self.value_bins))
        self.act_selfattn = int(act_selfattn)
        self.act_compare = int(act_compare)
        if self.act_selfattn and self.act_compare:
            raise ValueError("act_selfattn and act_compare are mutually exclusive")
        if self.act_selfattn:
            self.act_sa_norm = torch.nn.LayerNorm(d_model)
            self.act_sa = torch.nn.MultiheadAttention(d_model, num_heads)
            torch.nn.init.zeros_(self.act_sa.out_proj.weight)
            torch.nn.init.zeros_(self.act_sa.out_proj.bias)
        if self.act_compare:
            self.act_compare_norm = torch.nn.LayerNorm(d_model)
            self.act_compare_sa = torch.nn.MultiheadAttention(d_model, num_heads)
            self.act_compare_fc = torch.nn.Linear(d_model, 1)
            torch.nn.init.zeros_(self.act_compare_fc.weight)
            torch.nn.init.zeros_(self.act_compare_fc.bias)
        self.act_pointer = int(act_pointer)
        if self.act_pointer and self.act_selfattn:
            raise ValueError("act_pointer and act_selfattn are mutually exclusive")
        self.ptr_belief = int(ptr_belief)
        if self.ptr_belief and not self.act_pointer:
            raise ValueError("ptr_belief requires act_pointer=1 (the pointer query reads the belief)")
        if self.ptr_belief and self.recurrent == "none":
            raise ValueError("ptr_belief requires a recurrent model (the belief is its state)")
        # Pointer scoring head, following the TargetUnitHead of DI-star (OpenDILab, Apache-2.0).
        if self.act_pointer:
            _q1_in = d_model + (self.rec_hidden if self.ptr_belief else 0)
            self.ptr_key_fc = torch.nn.Linear(d_model, PTR_KEY_DIM, bias=False)
            self.ptr_query_mlp = torch.nn.Sequential(
                torch.nn.Linear(_q1_in, PTR_KEY_DIM), torch.nn.ReLU(),
                torch.nn.Linear(PTR_KEY_DIM, PTR_KEY_DIM))
            torch.nn.init.zeros_(self.ptr_query_mlp[2].weight)
            torch.nn.init.zeros_(self.ptr_query_mlp[2].bias)
            with torch.no_grad():
                self.ptr_query_mlp[2].bias[0] = 1.0
                self.ptr_key_fc.weight[0].copy_(self.decoder_fc.weight[0])
                if self.ptr_belief:
                    self.ptr_query_mlp[0].weight[:, d_model:].zero_()
            self.ptr_bias = torch.nn.Parameter(self.decoder_fc.bias.detach().clone().reshape(1))
        del self.encoder_bag
        del self.decoder_bag
        self.enc_card_mlp = torch.nn.Sequential(
            torch.nn.Linear(ENC_DENSE_DIM_TF, d_model), torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model))
        self.enc_in_dim = int(ENC_DENSE_DIM_TF)
        self.dec_action_mlp = torch.nn.Sequential(
            torch.nn.Linear(DEC_DENSE_DIM_TF, d_model), torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model))
        self.enc_role_embed = torch.nn.Embedding(N_ROLE, d_model)
        self.dec_plane_embed = torch.nn.Embedding(N_PLANE, d_model)
        self.aux_eff_logits = torch.nn.Linear(d_model, EFF_VOCAB_DIM)
        self.aux_eff_deltas = torch.nn.Linear(d_model, EFF_DELTA_DIM)
        torch.nn.init.zeros_(self.aux_eff_logits.weight)
        torch.nn.init.zeros_(self.aux_eff_logits.bias)
        torch.nn.init.zeros_(self.aux_eff_deltas.weight)
        torch.nn.init.zeros_(self.aux_eff_deltas.bias)
        self.gru_zones = int(gru_zones)
        if self.gru_zones and self.recurrent != "gru":
            raise ValueError("gru_zones requires recurrent='gru'")
        if self.gru_zones and int(self.num_words) != int(NUM_WORDS):
            raise ValueError(f"gru_zones requires num_words == NUM_WORDS ({self.num_words} != "
                             f"{NUM_WORDS}): zones are defined on spec.ZONE_WORDS")
        if self.recurrent != "none":
            if self.gru_zones:
                self.rnn = torch.nn.GRU(2 * d_model, self.rec_hidden)
                zm = torch.zeros(N_ZONES, int(self.num_words))
                for _zi, _ws in enumerate(ZONE_WORDS):
                    for _wi in _ws:
                        zm[_zi, _wi] = 1.0 / len(_ws)
                self.register_buffer("zone_mix", zm, persistent=False)
                self.zone_proj = torch.nn.Linear(N_ZONES * d_model, d_model)
            torch.nn.init.zeros_(self.mem_proj.weight)
            torch.nn.init.zeros_(self.mem_proj.bias)
            torch.nn.init.zeros_(self.value_fc.weight)
            torch.nn.init.zeros_(self.value_fc.bias)
            for _n, _p in self.rnn.named_parameters():
                if _n.startswith("weight_hh"):
                    torch.nn.init.orthogonal_(_p)
                elif _n.startswith("weight_ih"):
                    torch.nn.init.xavier_uniform_(_p)
                elif _n.startswith("bias"):
                    torch.nn.init.zeros_(_p)
            if self.gru_zones:
                with torch.no_grad():
                    self.rnn.weight_ih_l0[:, d_model:].zero_()
            self.belief_key = torch.nn.Linear(d_model, BELIEF_KEY_DIM, bias=False)
            self.belief_query = torch.nn.Linear(self.rec_hidden, BELIEF_KEY_DIM)
            torch.nn.init.zeros_(self.belief_query.weight)
            torch.nn.init.zeros_(self.belief_query.bias)

    def load_state_dict(self, state_dict, strict=True, **kw):
        """Apply only explicit, documented layout migrations before the normal PyTorch loader."""
        state_dict = migrate_encoder_input_layout(state_dict, log=True)
        state_dict = migrate_decoder_input_layout(state_dict, log=True)
        tr = self.enc_role_embed.weight
        state_dict = migrate_role_embed_layout(
            state_dict, tr.shape[0], tr.shape[1], log=True)
        state_dict = migrate_act_pointer(state_dict, self.act_pointer, log=True)
        state_dict = migrate_ptr_belief(state_dict, getattr(self, "ptr_belief", 0),
                                        rec_hidden=self.rec_hidden, log=True)
        state_dict = migrate_gru_zones(state_dict, getattr(self, "gru_zones", 0), log=True)
        return super().load_state_dict(state_dict, strict=strict, **kw)

    def value_logits_from(self, encoder_out, belief):
        """(B, value_bins) logits. 'none': per-word logits meaned over words (mirrors the parent's
        mean-over-words of scalars); recurrent: logits from the belief."""
        if self.recurrent == "none":
            return self.value_dist_head(encoder_out).mean(0)
        return self.value_dist_head(belief)

    def value_from(self, encoder_out, belief):
        """Override: scalar expectation over the bin distribution when bins are on (NO tanh — the
        expectation of centers spanning [-1,1] is bounded by construction; tanh would warp calibration);
        parent behavior (tanh scalar head) when off."""
        if self.value_bins > 0:
            p = torch.softmax(self.value_logits_from(encoder_out, belief), dim=-1)
            return (p * self.value_bin_centers).sum(-1, keepdim=True)
        return super().value_from(encoder_out, belief)

    def encode(self, dense_e, role_e):
        """dense_e: (W*B, ENC_DENSE_DIM_TF); role_e: (W*B,) long ROLE id."""
        role_e = _guard_ids(role_e, N_ROLE, "ROLE")
        v = self.enc_card_mlp(dense_e) + self.enc_role_embed(role_e)
        v = v.reshape(-1, self.num_words, self.d_model).transpose(0, 1)
        encoder_out = self.encoder(v)
        return encoder_out, encoder_out.mean(0)

    def action_tokens(self, encoder_out, belief, dense_d, plane_d):
        """(A,B,d) per-action representation BEFORE decoder_fc."""
        batch = encoder_out.size(1)
        plane_d = _guard_ids(plane_d, N_PLANE, "PLANE")
        p = self.dec_action_mlp(dense_d) + self.dec_plane_embed(plane_d)
        p = p.reshape(batch, -1, self.d_model).transpose(0, 1)
        if self.recurrent == "none":
            kv = encoder_out
        else:
            mem = self.mem_proj(belief).unsqueeze(0)
            kv = torch.cat([encoder_out, mem], dim=0)
        for layer in self.decoder:
            p = layer(p, kv)
        if self.act_selfattn:
            pad = plane_d.reshape(batch, -1) == PLANE_PAD
            q = self.act_sa_norm(p)
            y, _ = self.act_sa(q, q, q, key_padding_mask=pad, need_weights=False)
            p = p + y
        return p

    def score(self, encoder_out, belief, dense_d, plane_d, summary=None):
        p = self.action_tokens(encoder_out, belief, dense_d, plane_d)
        kw = {}
        if belief is not None:
            kw["belief"] = belief
        if self.act_pointer:
            if summary is None:
                summary = encoder_out.mean(0)
            kw["summary"] = summary
        return self.score_from_tokens(p, plane_d, **kw)

    def policy_components_from_tokens(self, action_tokens, plane_d, summary=None, belief=None):
        """Return raw ``(main, compare)`` scores, both shaped (A,B,1)."""
        if self.act_pointer:
            if summary is None:
                raise RuntimeError(
                    "act_pointer=1 requires summary, the (B, d_model) tensor returned by encode()")
            q_in = summary
            if getattr(self, "ptr_belief", 0):
                if belief is None:
                    raise RuntimeError(
                        "ptr_belief=1 requires belief, the (B, rec_hidden) state of the recurrent core")
                q_in = torch.cat([summary, belief], dim=-1)
            q = self.ptr_query_mlp(q_in)
            k = self.ptr_key_fc(action_tokens)
            main = (q.unsqueeze(0) * k).sum(-1, keepdim=True) + self.ptr_bias
        else:
            main = self.decoder_fc(action_tokens)
        if belief is not None and getattr(self, "belief_query", None) is not None:
            bq = self.belief_query(belief)
            bk = self.belief_key(action_tokens)
            main = main + (bq.unsqueeze(0) * bk).sum(-1, keepdim=True)
        compare = torch.zeros_like(main)
        if self.act_compare:
            batch = action_tokens.size(1)
            expected = action_tokens.size(0) * batch
            if plane_d.numel() != expected:
                raise ValueError(f"plane_d has {plane_d.numel()} ids, expected {expected}")
            pad = plane_d.reshape(batch, -1).eq(PLANE_PAD)
            q = self.act_compare_norm(action_tokens)
            y, _ = self.act_compare_sa(q, q, q, key_padding_mask=pad, need_weights=False)
            compare = self.act_compare_fc(y)
        return main, compare

    def score_from_tokens(self, action_tokens, plane_d, summary=None, belief=None):
        """Convert decoder tokens to one bounded score per candidate action."""
        main, compare = self.policy_components_from_tokens(
            action_tokens, plane_d, summary=summary, belief=belief)
        batch = action_tokens.size(1)
        raw = main + compare
        return torch.tanh(raw.transpose(0, 1).reshape(batch, -1))

    def aux_predict(self, action_tokens, chosen_rows):
        """Train-only grounding head. action_tokens: (A,B,d); chosen_rows: (B,) long -> the chosen action's
        token per sample -> (logits (B,EFF_VOCAB_DIM), deltas (B,EFF_DELTA_DIM))."""
        at = action_tokens.transpose(0, 1)
        sel = at[torch.arange(at.size(0), device=at.device), chosen_rows]
        return self.aux_eff_logits(sel), self.aux_eff_deltas(sel)

    def recur(self, summary_seq, state):
        """Guard: on a gru_zones model, a caller passing only the summary would silently lose the zone input."""
        if getattr(self, "gru_zones", 0):
            data = (summary_seq.data if isinstance(summary_seq, torch.nn.utils.rnn.PackedSequence)
                    else summary_seq)
            if int(data.shape[-1]) != 2 * self.d_model:
                raise RuntimeError(
                    f"gru_zones=1: recur() expects the output of recur_input(encoder_out, summary) "
                    f"(width {2 * self.d_model}, got {int(data.shape[-1])})")
        return super().recur(summary_seq, state)

    def recur_input(self, encoder_out, summary):
        """Input to the recurrent core: the global summary, or summary plus per-zone means when armed."""
        if not getattr(self, "gru_zones", 0):
            return summary
        w, b, d = encoder_out.shape
        z = torch.matmul(self.zone_mix.to(dtype=encoder_out.dtype),
                         encoder_out.reshape(w, b * d))
        z = z.reshape(-1, b, d).permute(1, 0, 2).reshape(b, -1)
        return torch.cat([summary, self.zone_proj(z)], dim=-1)

    def forward(self, dense_e, role_e, dense_d, plane_d, state=None):
        encoder_out, summary = self.encode(dense_e, role_e)
        if self.recurrent == "none":
            value = self.value_from(encoder_out, None)
            if self.act_pointer:
                policy = self.score(encoder_out, None, dense_d, plane_d, summary=summary)
            else:
                policy = self.score(encoder_out, None, dense_d, plane_d)
            return value, policy, None
        batch = summary.size(0)
        if state is None:
            state = self.initial_state(batch, summary.device)
        belief_seq, new_state = self.recur(
            self.recur_input(encoder_out, summary).unsqueeze(0), state)
        belief = belief_seq.squeeze(0)
        value = self.value_from(encoder_out, belief)
        if self.act_pointer:
            policy = self.score(encoder_out, belief, dense_d, plane_d, summary=summary)
        else:
            policy = self.score(encoder_out, belief, dense_d, plane_d)
        return value, policy, new_state
