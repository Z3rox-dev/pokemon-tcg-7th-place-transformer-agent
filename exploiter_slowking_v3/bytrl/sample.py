"""Training-sample containers shared by self-play, losses and imitation."""
from dataclasses import dataclass

from .encoding import SparseVector


@dataclass
class Decision:
    """One recorded decision point (a single agent turn-choice)."""
    sv_enc: SparseVector          # encoder input (24 board words)
    sv_dec: SparseVector          # decoder input (one bag per candidate action)
    n_actions: int                # number of candidate actions
    action: int                   # index of the action actually taken
    mu: float                     # behaviour-policy prob of the taken action (for IS / V-Trace)
    eff: object = None            # funcnet.effects_spec.EffectLabel | None — Stage-3 aux label, train-only.
    #                               Stored as a plain object so bytrl keeps NO funcnet dependency. Default
    #                               None => every non-recorded/eval/deploy/imitation path is byte-identical.
    pi_prime: object = None       # np.float16[n_actions] | None — funcnet3 search-improved policy target
    #                               (Gumbel-MCTS). None => unsearched decision; every funcnet/funcnet2 path
    #                               ignores it (same plain-object trick as eff, no numpy dependency here).
    #                               INVARIANT when set: mu == float(pi_prime[action]) (the action was SAMPLED
    #                               from pi_prime, so V-trace's behavior prob is pi_prime's, not the net's).


# A trajectory is (decisions_in_temporal_order: list[Decision], z: float) where z in {+1, 0, -1}
# is the terminal reward from the acting player's view (win / draw / loss). gamma=1, r_t=0 elsewhere.
Trajectory = tuple  # (list[Decision], float)
