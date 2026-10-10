"""Policy+value network with an optional RECURRENT belief core (the v2 upgrade).

Design (see docs/bytrl-paper-study.md §3 and docs/bytrl-adaptation.md):
  - Encoder: EmbeddingBag(state) -> TransformerEncoder over `num_words_encoder` board "words".
  - Recurrent core (the new bit): a GRU/LSTM carries a BELIEF state across a game's decisions —
    the POMDP mechanism the paper uses (LSTM-256). The belief feeds the value head AND is appended
    as one extra key/value token the action decoder can attend to.
  - Decoder: each engine-enumerated legal action is a token; cross-attends to [board words (+ belief)]
    and is scored (pointer-network ranking — we KEEP ranking, not the paper's type->target generation).
  - value: tanh scalar in [-1,1]; policy: one tanh score per candidate action (softmax(x*temp) is
    applied OUTSIDE the net by the trainer/agent, exactly as in the legacy model).

recurrent='none' reproduces the legacy ml/model.py MyModel EXACTLY (same submodule names), so an
existing imitation/OSFP checkpoint loads as-is for the feed-forward A/B baseline, and the recurrent
variant can warm-start its shared encoder/decoder from such a checkpoint via load_state_dict(strict=False).

Forward / method contract:
  - forward(...): single STEP for B parallel independent decisions (advances each one's state by one).
    Used at inference (B=1, carry state across moves) and for lockstep R2D2-style batched training.
  - encode()/recur()/value_from()/score(): exposed so the trainer can process a whole trajectory as a
    length-T sequence (encoder/decoder batched over the T steps, GRU run once over the sequence).
"""
import torch

from .spec import encoder_size, decoder_size, num_words_encoder


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)


class RecurrentNet(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int,
                 num_layers_encoder: int, num_layers_decoder: int,
                 recurrent: str = "gru", rec_hidden: int | None = None,
                 num_words: int | None = None):
        super().__init__()
        assert recurrent in ("none", "gru", "lstm"), recurrent
        self.d_model = d_model
        self.recurrent = recurrent
        # number of encoder "words"; 24 by default, 25 when an opponent-belief word is appended
        # (oracle-belief test / archetype belief). Warm-start is word-count-agnostic.
        self.num_words = num_words if num_words else num_words_encoder
        rec_hidden = rec_hidden if (rec_hidden and recurrent != "none") else d_model
        self.rec_hidden = rec_hidden

        # --- encoder (names match legacy MyModel for checkpoint compatibility) ---
        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        enc_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)          # legacy value head (recurrent='none')

        # --- recurrent belief core (new) ---
        if recurrent == "gru":
            self.rnn = torch.nn.GRU(d_model, rec_hidden)
        elif recurrent == "lstm":
            self.rnn = torch.nn.LSTM(d_model, rec_hidden)
        else:
            self.rnn = None
        self.mem_proj = torch.nn.Linear(rec_hidden, d_model) if recurrent != "none" else None
        self.value_fc = torch.nn.Linear(rec_hidden, 1) if recurrent != "none" else None

        # --- decoder (names match legacy MyModel) ---
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_feedforward) for _ in range(num_layers_decoder)])
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    # ---------------- encoder ----------------
    def encode(self, index_e, value_e, offset_e):
        """Returns (encoder_out (W,B,d), summary (B,d)). States are independent -> batchable over B."""
        v = self.encoder_bag(index_e, offset_e, value_e)
        v = v.reshape(-1, self.num_words, self.d_model).transpose(0, 1)      # (W, B, d)
        encoder_out = self.encoder(v)                                        # (W, B, d)
        summary = encoder_out.mean(0)                                        # (B, d)
        return encoder_out, summary

    # ---------------- recurrence ----------------
    def initial_state(self, batch_size: int, device):
        if self.recurrent == "none":
            return None
        h = torch.zeros(1, batch_size, self.rec_hidden, device=device)
        if self.recurrent == "lstm":
            return (h, torch.zeros_like(h))
        return h

    def recur(self, summary_seq, state):
        """summary_seq: (T, B, d) -> belief (T, B, h), new_state. For per-trajectory training B=1."""
        return self.rnn(summary_seq, state)

    # ---------------- heads ----------------
    def value_from(self, encoder_out, belief):
        """belief: (B, h) (recurrent) or None ('none' -> legacy per-word-mean value)."""
        if self.recurrent == "none":
            return torch.tanh(self.encoder_fc(encoder_out).mean(0))          # (B, 1)
        return torch.tanh(self.value_fc(belief))                            # (B, 1)

    def score(self, encoder_out, belief, index_d, value_d, offset_d):
        """Rank the candidate actions. Returns raw tanh scores (B, A_max). belief: (B, h) or None."""
        batch = encoder_out.size(1)
        p = self.decoder_bag(index_d, offset_d, value_d)
        p = p.reshape(batch, -1, self.d_model).transpose(0, 1)              # (A, B, d)
        if self.recurrent == "none":
            kv = encoder_out
        else:
            mem = self.mem_proj(belief).unsqueeze(0)                         # (1, B, d)
            kv = torch.cat([encoder_out, mem], dim=0)                        # (W+1, B, d)
        for layer in self.decoder:
            p = layer(p, kv)
        p = self.decoder_fc(p)                                              # (A, B, 1)
        return torch.tanh(p.transpose(0, 1).view(batch, -1))               # (B, A)

    # ---------------- single-step convenience (inference / lockstep training) ----------------
    def forward(self, index_e, value_e, offset_e, index_d, value_d, offset_d, state=None):
        """One step for B parallel independent decisions; returns (value (B,1), policy (B,A), new_state).
        For 'none' new_state is None and `state` is ignored (feed-forward)."""
        encoder_out, summary = self.encode(index_e, value_e, offset_e)
        if self.recurrent == "none":
            value = self.value_from(encoder_out, None)
            policy = self.score(encoder_out, None, index_d, value_d, offset_d)
            return value, policy, None
        batch = summary.size(0)
        if state is None:
            state = self.initial_state(batch, summary.device)
        belief_seq, new_state = self.recur(summary.unsqueeze(0), state)     # (1, B, h)
        belief = belief_seq.squeeze(0)                                      # (B, h)
        value = self.value_from(encoder_out, belief)
        policy = self.score(encoder_out, belief, index_d, value_d, offset_d)
        return value, policy, new_state
