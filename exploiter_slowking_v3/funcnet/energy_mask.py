"""Deploy-time veto of WASTED energy attachments (hand-crafted policy surgery, opt-in).

The measured failure (funcnet2/energy_probe): RL eroded energy discipline to ~chance — the net often
attaches a second copy of a type the target already has (e.g. a 2nd FIRE on a Dragapult whose Phantom
Dive costs exactly {FIRE, PSYCHIC}), burning the once-per-turn attachment and often the game.

Rule (engine-exact, per-attack, multiset-aware): an ATTACH of energy type T to MY Pokemon P is WASTED
iff it helps NONE of P's attacks, where for each attack:
  * T fills a still-open COLORED slot of that attack (cost counts minus already-attached counts), or
  * the attack still has an open COLORLESS slot (any energy pays it) after matching colored needs.
Multiset counts keep legitimate doubles legal (Duskull "Mumble" {P,P}, Dusknoir {P,P,C}).

Conservative by design: non-attackers are never judged (could be retreat fuel / tech), and the veto set
is dropped entirely if it would cover ALL legal actions. Known blind spots (accepted): energy attached
as retreat fuel to an attacker, and attach-to-evolve-target timing — the A/B decides if the trade wins.
"""
from collections import Counter

from cg.api import OptionType, EnergyType

from funcnet.encoding import get_card
from funcnet.spec import CARD_TABLE
from funcnet.features import ATTACKS

COLORLESS = int(EnergyType.COLORLESS)   # 0


def _etype(card):
    cid = getattr(card, "id", None)
    c = CARD_TABLE.get(cid) if cid is not None else None
    e = getattr(c, "energyType", None) if c is not None else None
    return None if e is None else int(e)


def wasted_attach_vetoes(o, actions):
    """-> set of indices into `actions` whose single option is a WASTED energy attach (see module doc).
    Never vetoes everything (returns empty set instead)."""
    your_index = o.current.yourIndex
    vetoes = set()
    for i, act in enumerate(actions):
        if len(act) != 1:
            continue
        opt = o.select.option[act[0]]
        if opt.type != OptionType.ATTACH:
            continue
        target = get_card(o, opt.inPlayArea, opt.inPlayIndex, your_index)
        energy = get_card(o, opt.area, opt.index, your_index)
        T = _etype(energy)
        tcid = getattr(target, "id", None)
        card = CARD_TABLE.get(tcid) if tcid is not None else None
        if T is None or card is None:
            continue
        attacks = [ATTACKS.get(a) for a in (getattr(card, "attacks", None) or [])]
        attacks = [a for a in attacks if a is not None]
        if not attacks:
            continue                       # non-attacker: never judge
        attached = Counter(t for t in (_etype(e) for e in (getattr(target, "energyCards", None) or []))
                           if t is not None)
        useful = False
        for a in attacks:
            cost = [int(e) for e in a.energies]
            need_col = Counter(t for t in cost if t != COLORLESS)
            n_free = sum(1 for t in cost if t == COLORLESS)
            if attached[T] < need_col.get(T, 0):
                useful = True; break       # fills an open colored slot
            matched = sum(min(attached[c], need_col[c]) for c in need_col)
            if (sum(attached.values()) - matched) < n_free:
                useful = True; break       # an open colorless slot remains
        if not useful:
            vetoes.add(i)
    return vetoes if len(vetoes) < len(actions) else set()
