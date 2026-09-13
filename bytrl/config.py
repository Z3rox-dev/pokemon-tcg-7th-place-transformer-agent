"""Master hyperparameters for the ByteRL/OSFP-inspired Pokémon-TCG agent (v2).

Values follow the paper (arXiv:2303.05197, Tab III/IV) where they are cheap and validated, and are
down-scaled where the paper's scale is unreachable on one RTX 4080. Modules import their knobs from here.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_heads: int = 2
    d_ff: int = 512
    enc_layers: int = 2
    dec_layers: int = 2
    recurrent: str = "lstm"
    rec_hidden: int = 256
    policy_temp: float = 10.0


@dataclass(frozen=True)
class RLConfig:
    gamma: float = 1.0
    lr: float = 7e-5
    entropy_coef: float = 0.01
    ppo_weight: float = 1.0
    upgo_weight: float = 1.0
    value_weight: float = 1.0
    ppo_clip_eps: float = 0.2
    vtrace_c_lo: float = 0.001
    vtrace_c_hi: float = 1.007
    vtrace_rho_lo: float = 0.001
    vtrace_rho_hi: float = 1.007
    sample_reuse: int = 2
    value_loss: str = "l2"
    target_kl: float = 0.05


@dataclass(frozen=True)
class OSFPConfig:
    self_play_prob: float = 0.6
    add_threshold_xi: float = 0.55
    max_lp_without_add: int = 6
    pool_max: int = 6
    xi_per_opponent: bool = True
    pfsp_sampling: bool = True
    samples_per_lp: int = 2_000_000


@dataclass(frozen=True)
class SelfPlayConfig:
    workers: int = 14
    max_game_steps: int = 4000
    prefetch: bool = True
    opp_mix: float = 0.4
    diverse_decks: list = field(default_factory=list)
    rule_based_roster: list = field(default_factory=list)


@dataclass(frozen=True)
class EvalConfig:
    games_per_opp: int = 40
    gate_metric: str = "roster_winrate"


@dataclass(frozen=True)
class DeployConfig:
    fp16: bool = True
    search_free: bool = True
    exec_safe: bool = True


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    osfp: OSFPConfig = field(default_factory=OSFPConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


DEFAULT = Config()
