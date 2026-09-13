"""Training-sample containers shared by self-play, losses and imitation."""
from dataclasses import dataclass

from .encoding import SparseVector


@dataclass
class Decision:
    """One recorded decision point (a single agent turn-choice)."""
    sv_enc: SparseVector
    sv_dec: SparseVector
    n_actions: int
    action: int
    mu: float
    eff: object = None
    pi_prime: object = None


Trajectory = tuple
