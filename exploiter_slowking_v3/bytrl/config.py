"""Master hyperparameters for the ByteRL/OSFP-inspired Pokémon-TCG agent (v2).

Values follow the paper (arXiv:2303.05197, Tab III/IV) where they are cheap & validated, and are
down-scaled where the paper's scale is unreachable on one RTX 4080. See docs/bytrl-paper-study.md §9.2
and docs/bytrl-adaptation.md. Treat this as the single source of truth for knobs; modules import from here.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256          # keep small: net is ~92% embeddings; Kaggle bundle <95 MB (fp16)
    n_heads: int = 2
    d_ff: int = 512
    enc_layers: int = 2
    dec_layers: int = 2
    recurrent: str = "lstm"     # {"none","gru","lstm"} — the #1 upgrade vs the current net.
                                # Default LSTM to MATCH THE PAPER (LSTM-256). GRU is ~equivalent and
                                # lighter; A/B the two in the OSFP phase (one flag) and keep the winner.
    rec_hidden: int = 256       # paper LSTM states = 256
    policy_temp: float = 10.0   # softmax(score * temp); matches current model


@dataclass(frozen=True)
class RLConfig:
    # --- adopt verbatim from the paper (cheap & validated) ---
    gamma: float = 1.0          # terminal-only ±1 reward, short-ish returns (Sec IV-C)
    lr: float = 7e-5            # Tab III
    entropy_coef: float = 0.01  # Tab III
    ppo_weight: float = 1.0     # Tab III
    upgo_weight: float = 1.0    # Tab III
    value_weight: float = 1.0   # Tab III
    ppo_clip_eps: float = 0.2   # eq. 7
    vtrace_c_lo: float = 0.001  # eq. 6 lower floor (anti trace-vanish)
    vtrace_c_hi: float = 1.007  # eq. 6 upper clip (>1: less bias)
    vtrace_rho_lo: float = 0.001
    vtrace_rho_hi: float = 1.007
    sample_reuse: int = 2       # Tab III ("epochs")
    value_loss: str = "l2"      # paper uses L2; ours historically Huber (switchable)
    target_kl: float = 0.05     # off-policy cap (our single-box stand-in for queue-buffer balancing)


@dataclass(frozen=True)
class OSFPConfig:
    self_play_prob: float = 0.6     # Tab IV  p
    add_threshold_xi: float = 0.55  # Tab IV  ξ : signed-mean of ±1 outcomes (= 2·winfrac − 1); 0.55 ⇒ ~77.5% wins
    max_lp_without_add: int = 6     # Tab IV  c  (current ml/osfp default is 4 — align to 6)
    pool_max: int = 6               # cap; keep a few "anchor" old checkpoints (softer than pop-oldest)
    xi_per_opponent: bool = True    # require beating EVERY pool member (paper ∀i), not a pooled aggregate
    pfsp_sampling: bool = True      # payoff-weighted opponent sampling (favor opponents we lose to)
    # --- down-scaled LP (paper: 3.2e8 samples/LP — unreachable on one box) ---
    samples_per_lp: int = 2_000_000  # ~1–4e6; redefine the LP by a sample/decision budget


@dataclass(frozen=True)
class SelfPlayConfig:
    workers: int = 14           # ~cores-2 on the 16-core box
    max_game_steps: int = 4000  # drop (don't fabricate ±1) games exceeding the cap
    prefetch: bool = True       # overlap CPU self-play with GPU train (bounded staleness)
    # opponents must span the meta, not just the mirror (see adaptation B2 #8)
    opp_mix: float = 0.4        # fraction of non-mirror games drawn from diverse opponents
    diverse_decks: list = field(default_factory=list)   # archetype deck csvs
    rule_based_roster: list = field(default_factory=list)


@dataclass(frozen=True)
class EvalConfig:
    games_per_opp: int = 40     # down-scaled from the paper's 2500/cell; Wilson CI on top
    gate_metric: str = "roster_winrate"  # real objective = beat the rule-based roster, not wr_vs_pool


@dataclass(frozen=True)
class DeployConfig:
    fp16: bool = True           # ~106 MB fp32 -> ~53 MB; bundle must stay <95 MB
    search_free: bool = True    # 1 forward pass/move; no timeout (10-min/match budget)
    exec_safe: bool = True      # main.py: NO __file__; resolve via cwd + /kaggle_simulations/agent/


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    osfp: OSFPConfig = field(default_factory=OSFPConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


DEFAULT = Config()
