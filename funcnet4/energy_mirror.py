"""Python mirror of the engine's energy-cost check, feeding the decoder's NRG tail: how many energies
a Pokemon still needs to pay for each of its attacks, with the modifiers currently in play.

It informs, it does not forbid: a hard cap on attachments measured -6pp at deploy, so over-attaching
stays the network's choice. The mirror is deliberately partial - typed matching, Rainbow, Team
Rocket and the colorless-cost modifiers are exact; one-shot per-turn effects and copied attacks are
out of scope. Reads no environment variables: the flag lives in the caller (funcnet4/spec.py).
"""
from funcnet.spec import CARD_TABLE
from funcnet.features import ATTACKS

NIGHTTIME_MINE = 1266
ANTIQUE_ROOT_FOSSIL = 1099
COUNTER_GAIN = 1168
HOPS_CHOICE_BAND = 1171
DECIDUEYE_EX = 1022
_ENERGY_CARDTYPES = (5, 6)
_POKEMON = 0
_C = 0
_P = 5
_D = 7
_RAINBOW = 10
_PD = 11


def nm_in_play(state) -> bool:
    """True if Nighttime Mine is the stadium in play (obs.current.stadium, shared by both players)."""
    return any(c is not None and int(getattr(c, "id", -1) or -1) == NIGHTTIME_MINE
               for c in (getattr(state, "stadium", None) or []))


def cost_change_ctx(state, your_index):
    """Precompute, once per decision, the cost modifiers readable from the observation."""
    my = state.players[your_index]
    opp = state.players[1 - your_index]
    opp_act = opp.active[0] if getattr(opp, "active", None) else None
    return {
        "nm": nm_in_play(state),
        "root": (getattr(opp_act, "id", None) == ANTIQUE_ROOT_FOSSIL),
        "prize_gain": len(getattr(my, "prize", None) or []) > len(getattr(opp, "prize", None) or []),
        "opp_hand4": int(getattr(opp, "handCount", -1)) == 4,
    }


def _mon_cost_change(cd, tmon, ctx) -> int:
    """Colorless cost change on this Pokemon, summed over every active source."""
    ch = 0
    if ctx["nm"] and getattr(cd, "tera", False):
        ch += 1
    if (ctx["root"] and int(getattr(cd, "cardType", -1) or -1) == _POKEMON
            and not getattr(cd, "stage1", False) and not getattr(cd, "stage2", False)):
        ch += 1
    for t in (getattr(tmon, "tools", None) or []):
        tid = getattr(t, "id", None)
        if tid == COUNTER_GAIN and ctx["prize_gain"]:
            ch -= 1
        elif tid == HOPS_CHOICE_BAND and str(getattr(cd, "name", "")).startswith("Hop"):
            ch -= 1
    if getattr(tmon, "id", None) == DECIDUEYE_EX and ctx["opp_hand4"]:
        ch -= 99
    return ch


def provided_types(cid):
    """Base energy an energy card provides, for a hypothetical attach."""
    c = CARD_TABLE.get(cid)
    if c is None or int(getattr(c, "cardType", -1) or -1) not in _ENERGY_CARDTYPES:
        return None
    et = getattr(c, "energyType", None)
    return [int(et)] if et is not None else [_C]


def attack_missing(required, provided, cost_change=0) -> int:
    """How many energies an attack still needs, mirroring the engine's own count."""
    req = {}
    req_c = 0
    for e in (required or []):
        t = int(e)
        if t == _C:
            req_c += 1
        else:
            req[t] = req.get(t, 0) + 1
    all_n = pool = pd_n = 0
    for e in (provided or []):
        t = int(e)
        if t == _RAINBOW:
            all_n += 1
        elif t == _C:
            pool += 1
        elif t == _PD:
            pd_n += 1
        elif req.get(t, 0) > 0:
            req[t] -= 1
        else:
            pool += 1
    if cost_change > 0:
        req_c += cost_change
    elif cost_change < 0:
        pool += -cost_change
    for _ in range(pd_n):
        if req.get(_P, 0) > 0:
            req[_P] -= 1
        elif req.get(_D, 0) > 0:
            req[_D] -= 1
        else:
            pool += 1
    need = sum(req.values()) + req_c - all_n - min(pool, req_c)
    return max(0, need)


def attack_missing_for_mon(tmon, attack, ctx, extra_energy_cid=None):
    """Missing energies for one attack of this Pokemon, with the modifiers currently in play."""
    cd = CARD_TABLE.get(getattr(tmon, "id", None))
    if cd is None:
        return None
    provided = [int(e) for e in (getattr(tmon, "energies", None) or [])]
    if extra_energy_cid is not None:
        extra = provided_types(extra_energy_cid)
        if extra is None:
            return None
        provided = provided + extra
    ch = _mon_cost_change(cd, tmon, ctx)
    return attack_missing(getattr(attack, "energies", None), provided, ch)


def mon_missing_profile(tmon, ctx, extra_energy_cid=None):
    """Missing energies for every printed attack of this Pokemon."""
    cd = CARD_TABLE.get(getattr(tmon, "id", None))
    if cd is None or not getattr(cd, "attacks", None):
        return None
    prof = []
    for aid in cd.attacks:
        a = ATTACKS.get(aid)
        if a is None:
            continue
        m = attack_missing_for_mon(tmon, a, ctx, extra_energy_cid)
        if m is None:
            return None
        prof.append(m)
    return prof or None


def attach_nrg_scalars(ecard_cid, tmon, ctx):
    """The three NRG-tail scalars for attaching a given energy card to a given Pokemon."""
    before = mon_missing_profile(tmon, ctx)
    if before is None:
        return None
    after = mon_missing_profile(tmon, ctx, extra_energy_cid=ecard_cid)
    if after is None or len(after) != len(before):
        return None
    useful = 1.0 if any(a < b for a, b in zip(after, before)) else 0.0
    return (min(after) / 4.0, max(after) / 4.0, useful)
