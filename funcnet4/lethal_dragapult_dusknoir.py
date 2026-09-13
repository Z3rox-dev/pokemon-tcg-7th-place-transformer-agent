"""Deterministic lethal finder for the Dragapult/Dusknoir line: an engine-exact DFS over the
Search API that plays a winning sequence when one exists, instead of leaving it to the policy.

Deck-specific by design, and it overrides the network only when the line is proven, never on a
heuristic. Reads no environment variables (bundle constraint): flags live in the caller.
"""
from cg.api import (search_step, search_end, OptionType, SelectType, SelectContext, AreaType,
                    LogType)
try:
    from cg.api import search_step_lazy as _search_step_dfs
except ImportError:
    _search_step_dfs = search_step
from bytrl.actions import enumerate_actions
from funcnet.encoding import get_card
from funcnet.spec import CARD_TABLE
from funcnet.features import ATTACKS
from funcnet4.search import determinize

COMBO_CIDS = {132: "Dusclops", 133: "Dusknoir", 112: "Munkidori"}
MUNKIDORI_CID = 112
DARK_ENERGY_TYPE = 7

COMPOSE_PLAY_CIDS = {1097: "Night Stretcher", 1182: "Boss's Orders", 112: "Munkidori",
                     1256: "Team Rocket's Watchtower"}
COMPOSE_EVOLVE_CIDS = {121, 132, 133, 120}
SELF_KO_ABILITY_CIDS = {132, 133}
DECK_SEARCH_NAMES = ("Ultra Ball", "Crispin")

_COUNTER_CONTEXTS = (int(SelectContext.DAMAGE_COUNTER), int(SelectContext.DAMAGE_COUNTER_ANY),
                     int(SelectContext.DAMAGE))
_COUNTER_PLACE_CONTEXTS = (int(SelectContext.DAMAGE_COUNTER), int(SelectContext.DAMAGE_COUNTER_ANY))

_PREVENT_MIST, _PREVENT_ROCK, _FIGHTING = 11, 20, 6
BATTLE_CAGE = 1264
CRUSHING_HAMMER = 1120
FEZANDIPITI_CID = 140
BUDEW_CID = 235
_POKEMON_CARDTYPE = 0
FEZ_FRAGILE_HP = 70


REPELLING_VEIL_CID = 414


def _articuno_in_play(opp_ps):
    """True if the opponent has a Team Rocket's Articuno in play (Repelling Veil)."""
    for z in (opp_ps.active, opp_ps.bench):
        for c in (z or []):
            if c is not None and int(getattr(c, "id", -1) or -1) == REPELLING_VEIL_CID:
                return True
    return False


def _effect_prevented(mon):
    """True if this Pokemon carries an energy that prevents attack effects."""
    for ec in (getattr(mon, "energyCards", None) or []):
        eid = int(getattr(ec, "id", -1) or -1)
        if eid == _PREVENT_MIST:
            return True
        if eid == _PREVENT_ROCK:
            cd = CARD_TABLE.get(getattr(mon, "id", None))
            if int(getattr(cd, "energyType", -1) or -1) == _FIGHTING:
                return True
    return False


def _cname(cid):
    c = CARD_TABLE.get(cid)
    return getattr(c, "name", None) or f"cid{cid}"


def _area_name(a):
    try:
        return AreaType(int(a)).name
    except Exception:
        return str(a)


def _describe(obs, sel, action, me):
    """Human-readable description of a selection (a list of option indices)."""
    parts = []
    for oi in action:
        opt = sel.option[oi]
        t = int(opt.type)
        if t in (int(OptionType.ABILITY), int(OptionType.SKILL)):
            card = get_card(obs, opt.area, opt.index, me)
            parts.append(f"ABILITY {_cname(getattr(card, 'id', None))}")
        elif t == int(OptionType.ATTACK):
            att = ATTACKS.get(getattr(opt, "attackId", None))
            parts.append(f"ATTACK {getattr(att, 'name', None) or getattr(opt, 'attackId', '?')}")
        elif t == int(OptionType.CARD):
            pidx = getattr(opt, "playerIndex", None)
            side = pidx if pidx is not None else me
            card = get_card(obs, opt.area, opt.index, side)
            hp = getattr(card, "hp", None)
            mx = getattr(card, "maxHp", None)
            who = "mio" if side == me else "avv"
            hps = f" {hp}/{mx}" if hp is not None else ""
            parts.append(f"CARD {_cname(getattr(card, 'id', None))} [{who} {_area_name(opt.area)}{hps}]")
        elif t == int(OptionType.ATTACH):
            ecard = get_card(obs, opt.area, opt.index, me)
            tmon = get_card(obs, opt.inPlayArea, opt.inPlayIndex, me)
            parts.append(f"ATTACH {_cname(getattr(ecard, 'id', None))} -> "
                         f"{_cname(getattr(tmon, 'id', None))}")
        elif t == int(OptionType.PLAY):
            hcard = get_card(obs, AreaType.HAND, opt.index, me)
            parts.append(f"PLAY {_cname(getattr(hcard, 'id', None))}")
        elif t == int(OptionType.EVOLVE):
            evc = get_card(obs, opt.area, opt.index, me)
            tmon = get_card(obs, opt.inPlayArea, opt.inPlayIndex, me)
            parts.append(f"EVOLVE {_cname(getattr(evc, 'id', None))} su "
                         f"{_cname(getattr(tmon, 'id', None))}")
        elif t == int(OptionType.END):
            parts.append("END")
        else:
            try:
                parts.append(OptionType(t).name)
            except Exception:
                parts.append(str(t))
    return " + ".join(parts) if parts else "<vuota>"


def has_deck_search(seq):
    """True if the line searches the deck, which makes it non-deterministic and unsafe to force."""
    return any(f"PLAY {n}" in d for n in DECK_SEARCH_NAMES for d in seq)


def lethal_sig(o):
    """Deterministic signature of hand and board, used to decide when to re-run detection."""
    st = o.current
    me = st.players[st.yourIndex]
    hand = tuple(sorted(int(c.id) for c in (me.hand or []) if c is not None))
    board = tuple(sorted(int(c.id) for z in (me.active, me.bench)
                         for c in (z or []) if c is not None))
    return hand, board


def _prize_value(cid):
    """Prizes the opponent takes if this Pokemon is knocked out: mega ex 3, ex 2, otherwise 1."""
    cd = CARD_TABLE.get(cid)
    if cd is None:
        return 1
    if getattr(cd, "megaEx", False):
        return 3
    return 2 if getattr(cd, "ex", False) else 1


def lethal_plausible(o):
    """Cheap arithmetic precheck: can the prizes on the table even add up to a win?"""
    st = o.current
    me = st.yourIndex
    my = st.players[me]
    opp = st.players[1 - me]
    needed = len(my.prize or [])
    if needed <= 0:
        return True
    my_play = [int(c.id) for z in (my.active, my.bench) for c in (z or []) if c is not None]
    my_hand = [int(c.id) for c in (my.hand or []) if c is not None]
    have = my_play + my_hand
    dive = 121 in my_play or (120 in my_play and 121 in my_hand)
    pool = (13 * min(have.count(133), 2) + 5 * min(have.count(132), 2)
            + (3 if MUNKIDORI_CID in have else 0))
    tgts = []
    for c in (opp.active or []):
        if c is not None and (getattr(c, "hp", 0) or 0) > 0:
            tgts.append((int(c.hp), _prize_value(int(c.id)), True))
    for c in (opp.bench or []):
        if c is not None and (getattr(c, "hp", 0) or 0) > 0:
            tgts.append((int(c.hp), _prize_value(int(c.id)), False))
    if not tgts:
        return True
    atk = 200 if dive else 90
    att_cid = 121 if dive else next(iter(my_play), None)
    cd_att = CARD_TABLE.get(att_cid) if att_cid is not None else None
    opp_act = next((c for c in (opp.active or []) if c is not None), None)
    if cd_att is not None and cd_att.energyType is not None and opp_act is not None:
        cd_def = CARD_TABLE.get(int(opp_act.id))
        if (cd_def is not None and cd_def.weakness is not None
                and int(cd_def.weakness) == int(cd_att.energyType)):
            atk *= 2
    dive_b = 6 if dive else 0
    n = len(tgts)
    full = (1 << n) - 1
    for mask in range(1, 1 << n):
        prizes = 0
        need_pool = 0
        need_bench = 0
        for i in range(n):
            if not (mask >> i) & 1:
                continue
            hp, pv, is_act = tgts[i]
            prizes += pv
            if is_act:
                extra = hp - atk
                if extra > 0:
                    need_pool += (extra + 9) // 10
            else:
                need_bench += (hp + 9) // 10
        if need_pool + max(0, need_bench - dive_b) > pool:
            continue
        if prizes >= needed or mask == full:
            return True
    return False


def gained_material(prev, now):
    """True if new cards appeared, compared as a multiset."""
    if prev is None:
        return False
    for a, b in zip(prev, now):
        cnt = {}
        for x in a:
            cnt[x] = cnt.get(x, 0) + 1
        for x in b:
            k = cnt.get(x, 0)
            if k == 0:
                return True
            cnt[x] = k - 1
    return False


def rearm_sig(o):
    """(material, position) signature used to decide when a new lethal could have appeared."""
    st = o.current
    me = st.yourIndex
    my = st.players[me]
    opp = st.players[1 - me]
    my_act = tuple(sorted((int(c.id), len(c.energyCards or []))
                          for c in (my.active or [])
                          if c is not None and getattr(c, "id", None) is not None))
    opp_board = tuple(sorted((is_act, int(c.id), int(getattr(c, "hp", 0) or 0))
                             for is_act, z in ((1, opp.active), (0, opp.bench))
                             for c in (z or [])
                             if c is not None and getattr(c, "id", None) is not None))
    return lethal_sig(o), (my_act, opp_board)


def pv_tail(ais):
    """The remainder of a found line, consumed one decision at a time while executing it."""
    tail = list(ais[1:])
    while tail and tail[0] is None:
        tail.pop(0)
    return tail or None


def should_rearm(prev, now):
    """True if something changed between two signatures that could create a lethal line."""
    if prev is None:
        return False
    return gained_material(prev[0], now[0]) or prev[1] != now[1]


def mask_dead_counter_targets(o, actions):
    """When placing damage counters, never spend them on a target that is already dead."""
    sel = o.select
    if int(getattr(sel, "context", -1)) not in _COUNTER_CONTEXTS:
        return None
    me = o.current.yourIndex
    src = getattr(sel, "effect", None)
    src_card = CARD_TABLE.get(int(getattr(src, "id", -1) or -1)) if src is not None else None
    atk_certain = src_card is not None and not getattr(src_card, "skills", None)
    is_place = int(getattr(sel, "context", -1)) in _COUNTER_PLACE_CONTEXTS
    cage = any(c is not None and int(getattr(c, "id", -1) or -1) == BATTLE_CAGE
               for c in (getattr(o.current, "stadium", None) or []))
    keep = []
    for ai, act in enumerate(actions):
        dead = False
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.CARD):
                continue
            pidx = getattr(opt, "playerIndex", None)
            side = pidx if pidx is not None else (1 - me)
            if side == me:
                continue
            try:
                mon = get_card(o, opt.area, opt.index, side)
            except Exception:
                continue
            hp = getattr(mon, "hp", None)
            if hp is not None and int(hp) <= 0:
                dead = True
                break
            if is_place and atk_certain and _effect_prevented(mon):
                dead = True
                break
            if (is_place and cage
                    and int(getattr(opt, "area", -1) or -1) == int(AreaType.BENCH)):
                dead = True
                break
        if not dead:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


def mask_suicidal_ability(o, actions):
    """Never use an ability that knocks out my own Pokemon for nothing."""
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    mine = st.players[me]
    opp_prizes = len(st.players[1 - me].prize or [])
    my_in_play = sum(1 for z in (mine.active, mine.bench) for c in (z or []) if c is not None)
    if opp_prizes > 1 and my_in_play > 1:
        return None
    keep = []
    for ai, act in enumerate(actions):
        bad = False
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) not in (int(OptionType.ABILITY), int(OptionType.SKILL)):
                continue
            pidx = getattr(opt, "playerIndex", None)
            if pidx is not None and pidx != me:
                continue
            try:
                card = get_card(o, opt.area, opt.index, me)
            except Exception:
                continue
            if getattr(card, "id", None) in SELF_KO_ABILITY_CIDS:
                bad = True
                break
        if not bad:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


_ENERGY_CARDTYPES = (5, 6)
NIGHTTIME_MINE = 1266


def _energy_cap(cid, nm_in_play, slack=1):
    """Energy a Pokemon can usefully hold: the cost of its most expensive attack, plus slack."""
    c = CARD_TABLE.get(cid)
    if c is None or not getattr(c, "attacks", None):
        return None
    costs = []
    for aid in c.attacks:
        a = ATTACKS.get(aid)
        if a is None:
            continue
        if getattr(a, "text", None) and "energy" in a.text.lower():
            return None
        costs.append(len(getattr(a, "energies", None) or []))
    if not costs:
        return None
    need = max(costs)
    if getattr(c, "tera", False) and nm_in_play:
        need += 1
    return need + slack


def mask_excess_energy_attach(o, actions, slack=1):
    """Never attach energy to a Pokemon that already has more than its attacks can use."""
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    nm_in_play = any(c is not None and int(getattr(c, "id", -1) or -1) == NIGHTTIME_MINE
                     for c in (getattr(st, "stadium", None) or []))
    keep = []
    for ai, act in enumerate(actions):
        bad = False
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.ATTACH):
                continue
            try:
                ecard = get_card(o, opt.area, opt.index, me)
                tmon = get_card(o, opt.inPlayArea, opt.inPlayIndex, me)
            except Exception:
                continue
            ec = CARD_TABLE.get(getattr(ecard, "id", None))
            if int(getattr(ec, "cardType", -1) or -1) not in _ENERGY_CARDTYPES:
                continue
            cap = _energy_cap(getattr(tmon, "id", None), nm_in_play, slack)
            if cap is None:
                continue
            attached = len(getattr(tmon, "energyCards", None) or [])
            if attached >= cap:
                bad = True
                break
        if not bad:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


def _has_dark(mon):
    for ec in (getattr(mon, "energyCards", None) or []):
        cd = CARD_TABLE.get(getattr(ec, "id", None))
        if int(getattr(cd, "energyType", -1) or -1) == DARK_ENERGY_TYPE:
            return True
    return False


def _munki_protected(tmon, opt):
    """True if this Munkidori is in the dark-energy-only regime."""
    if getattr(tmon, "id", None) != MUNKIDORI_CID or _has_dark(tmon):
        return False
    is_active = int(getattr(opt, "inPlayArea", -1) or -1) == int(AreaType.ACTIVE)
    n_energy = len(getattr(tmon, "energyCards", None) or [])
    if is_active and n_energy == 0:
        return False
    return True


def mask_wrong_energy_to_munkidori(o, actions):
    """Send dark energy where it can be used, not to a Munkidori that cannot pay with it."""
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    dark_available = False
    for act in actions:
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.ATTACH):
                continue
            try:
                ecard = get_card(o, opt.area, opt.index, me)
                tmon = get_card(o, opt.inPlayArea, opt.inPlayIndex, me)
            except Exception:
                continue
            ec = CARD_TABLE.get(getattr(ecard, "id", None))
            if (_munki_protected(tmon, opt)
                    and int(getattr(ec, "energyType", -1) or -1) == DARK_ENERGY_TYPE):
                dark_available = True
                break
        if dark_available:
            break
    if not dark_available:
        return None
    keep = []
    for ai, act in enumerate(actions):
        bad = False
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.ATTACH):
                continue
            try:
                ecard = get_card(o, opt.area, opt.index, me)
                tmon = get_card(o, opt.inPlayArea, opt.inPlayIndex, me)
            except Exception:
                continue
            ec = CARD_TABLE.get(getattr(ecard, "id", None))
            if (_munki_protected(tmon, opt)
                    and int(getattr(ec, "cardType", -1) or -1) in _ENERGY_CARDTYPES
                    and int(getattr(ec, "energyType", -1) or -1) != DARK_ENERGY_TYPE):
                bad = True
                break
        if not bad:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


def mask_dark_energy_misassign(o, actions):
    """Do not send a dark energy to a Munkidori that cannot use it."""
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    mine = st.players[me]
    munki_active = any(c is not None and getattr(c, "id", None) == MUNKIDORI_CID
                       and not _has_dark(c) for c in (mine.active or []))
    if not munki_active:
        return None
    keep = []
    for ai, act in enumerate(actions):
        bad = False
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.ATTACH):
                continue
            try:
                ecard = get_card(o, opt.area, opt.index, me)
                tmon = get_card(o, opt.inPlayArea, opt.inPlayIndex, me)
            except Exception:
                continue
            ec = CARD_TABLE.get(getattr(ecard, "id", None))
            if (int(getattr(ec, "energyType", -1) or -1) == DARK_ENERGY_TYPE
                    and getattr(tmon, "id", None) != MUNKIDORI_CID):
                bad = True
                break
        if not bad:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


def mask_useless_hammer(o, actions):
    """Never play Crushing Hammer when the opponent has no energy to discard."""
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    opp = st.players[1 - me]
    opp_energy = sum(len(getattr(c, "energyCards", None) or [])
                     for z in (opp.active, opp.bench) for c in (z or []) if c is not None)
    if opp_energy > 0:
        return None
    keep = []
    for ai, act in enumerate(actions):
        bad = False
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.PLAY):
                continue
            try:
                hcard = get_card(o, AreaType.HAND, opt.index, me)
            except Exception:
                continue
            if getattr(hcard, "id", None) == CRUSHING_HAMMER:
                bad = True
                break
        if not bad:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


class KOTracker:
    """Remember whether one of my Pokemon died during the opponent's turn."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev, self._cur = [], []

    def observe(self, o):
        st = getattr(o, "current", None)
        if st is None:
            return
        if int(getattr(st, "turn", 0) or 0) == 0:
            self.reset()
            return
        for lg in (getattr(o, "logs", None) or []):
            self._cur.append(lg)
            if int(lg.type) == int(LogType.TURN_END):
                self._prev, self._cur = self._cur, []

    def pre_ko(self, me):
        """True if one of my Pokemon went from the field to the discard during the turn just ended."""
        for lg in self._prev:
            try:
                if (int(lg.type) == int(LogType.MOVE_CARD)
                        and getattr(lg, "playerIndex", None) == me
                        and int(lg.fromArea) in (int(AreaType.BENCH), int(AreaType.ACTIVE))
                        and int(lg.toArea) == int(AreaType.DISCARD)):
                    _cid = getattr(lg, "cardId", None)
                    if _cid is not None:
                        _ct = getattr(CARD_TABLE.get(_cid), "cardType", None)
                        if _ct is not None and int(_ct) != _POKEMON_CARDTYPE:
                            continue
                    return True
            except Exception:
                continue
        return False


def mask_dead_fezandipiti(o, actions, pre_ko):
    """Never play Fezandipiti ex when its ability cannot fire this turn."""
    if pre_ko:
        return None
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    mine = st.players[me]

    fez_actions, other_pokemon = set(), False
    for ai, act in enumerate(actions):
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.PLAY):
                continue
            try:
                hcard = get_card(o, AreaType.HAND, opt.index, me)
            except Exception:
                continue
            cid = getattr(hcard, "id", None)
            if cid == FEZANDIPITI_CID:
                fez_actions.add(ai)
            else:
                _ct = getattr(CARD_TABLE.get(cid), "cardType", None)
                if _ct is not None and int(_ct) == _POKEMON_CARDTYPE:
                    other_pokemon = True
    if not fez_actions:
        return None
    my_in_play = sum(1 for z in (mine.active, mine.bench) for c in (z or []) if c is not None)
    if my_in_play <= 1 and not other_pokemon:
        return None

    keep = [ai for ai in range(len(actions)) if ai not in fez_actions]
    if not keep or len(keep) == len(actions):
        return None
    return keep


def mask_dominated_main(o, actions, pre_ko=None, energy_slack=1):
    """Combine the main masks into one set of allowed actions."""
    parts = [("suicide", mask_suicidal_ability(o, actions)),
             ("energy", mask_excess_energy_attach(o, actions, energy_slack)),
             ("munki", mask_wrong_energy_to_munkidori(o, actions)),
             ("dark", mask_dark_energy_misassign(o, actions)),
             ("hammer", mask_useless_hammer(o, actions))]
    if pre_ko is not None:
        parts.append(("fez", mask_dead_fezandipiti(o, actions, pre_ko)))
    active = [(t, k) for t, k in parts if k is not None]
    if not active:
        return None, None
    if len(active) == 1:
        return active[0][1], active[0][0]
    inter = set(active[0][1])
    for _, k in active[1:]:
        inter &= set(k)
    if inter:
        return sorted(inter), "both"
    return active[0][1], active[0][0]


class _Abort(Exception):
    pass


class _Pass1Stop(Exception):
    """Budget of the cheap first pass exhausted; fall through to the full search."""
    pass


class LethalFinder:
    """Engine-exact DFS: does a winning line exist from here, and which action starts it?"""

    def __init__(self, my_deck, opp_deck, rng=None, budget=3000, max_depth=14,
                 cap_follow=28, cap_opp=8, allow_ability=True, allow_munkidori=True,
                 compose=False, allow_attach_any=None, rv_reorder=True):
        self.my_deck, self.opp_deck, self.rng = my_deck, opp_deck, rng
        self.compose = compose
        self.rv_reorder = bool(rv_reorder)
        self.allow_attach_any = bool(compose) if allow_attach_any is None else bool(allow_attach_any)
        if compose:
            budget = max(budget, 12000)
            max_depth = max(max_depth, 24)
        self.budget_cap, self.max_depth = budget, max_depth
        self.cap_follow, self.cap_opp = cap_follow, cap_opp
        self.allow_ability = allow_ability
        self.allow_munkidori = allow_munkidori
        self._p1cap = None
        self._rank4_excluded = False
        self._rv_active = False

    def find(self, o, minimal=False, pv=None):
        """None, or the winning line: which action to play now and the sequence that follows."""
        self.steps = 0
        self.flags = set()
        self._rank4_excluded = False
        self.me = o.current.yourIndex
        self.root_turn = int(o.current.turn)
        self._rv_active = (self.rv_reorder
                           and _articuno_in_play(o.current.players[1 - self.me]))
        try:
            ss = determinize(o, self.me, self.my_deck, self.opp_deck, rng=self.rng)
            line = None
            if pv is not None:
                line = self._dfs(ss, 0, hint=tuple(pv))
            else:
                if self.compose:
                    self._p1cap = min(4000, self.budget_cap // 3)
                    try:
                        line = self._dfs(ss, 0)
                        if line is not None:
                            self.flags.add("pass1_hit")
                    except _Pass1Stop:
                        self.flags.add("pass1_stop")
                    finally:
                        self._p1cap = None
                if line is None and not (minimal and self.compose):
                    if (self.compose and "pass1_stop" not in self.flags
                            and not self._rank4_excluded
                            and "branch_error" not in self.flags):
                        self.flags.add("pass1_exhaustive")
                    else:
                        line = self._dfs(ss, 0)
        except _Abort:
            self.flags.add("abort")
            return None
        finally:
            try:
                search_end()
            except Exception:
                pass
        if line is None:
            return None
        seq = [d for _, d in line]
        if has_deck_search(seq):
            self.flags.add("search_line_dropped")
            return None
        return dict(first_ai=line[0][0], seq=seq, ais=[ai for ai, _ in line],
                    flags=sorted(self.flags))

    def find_in_world(self, ss):
        """Solve inside a world the caller already opened, rather than opening a new one."""
        self.steps = 0
        self.flags = set()
        st = ss.observation.current
        self.me = st.yourIndex
        self.root_turn = int(st.turn)
        self._rv_active = (self.rv_reorder
                           and _articuno_in_play(st.players[1 - self.me]))
        try:
            return self._dfs(ss, 0) is not None
        except _Abort:
            return False
        except Exception:
            return False

    def _step(self, sid, select):
        self.steps += 1
        if self.steps > self.budget_cap:
            raise _Abort()
        if self._p1cap is not None and self.steps > self._p1cap:
            raise _Pass1Stop()
        return _search_step_dfs(sid, select)

    def _dfs(self, ss, depth, min_oi=0, prev_ctx=None, hint=None):
        for _ in range(24):
            st = ss.observation.current
            if st.result >= 0:
                return [] if (st.result == self.me and int(st.turn) == self.root_turn) else None
            if int(st.turn) != self.root_turn:
                return None
            sel = ss.observation.select
            if sel is None or getattr(sel, "maxCount", 0) == 0 or len(sel.option) == 0:
                ss = self._step(ss.searchId, [])
                continue
            actions = enumerate_actions(sel)
            if len(actions) == 1:
                ss = self._step(ss.searchId, actions[0])
                continue
            break
        else:
            raise _Abort()
        if depth >= self.max_depth:
            self.flags.add("depth_cap")
            return None
        obs = ss.observation
        if st.yourIndex == self.me:
            if int(sel.type) == int(SelectType.MAIN):
                cands = []
                dup_seen = set()
                for ai, act in enumerate(actions):
                    if len(act) != 1:
                        continue
                    opt = sel.option[act[0]]
                    t = int(opt.type)
                    if t in (int(OptionType.ABILITY), int(OptionType.SKILL)):
                        if not self.allow_ability:
                            continue
                        pidx = getattr(opt, "playerIndex", None)
                        if pidx is not None and pidx != self.me:
                            continue
                        card = get_card(obs, opt.area, opt.index, self.me)
                        if getattr(card, "id", None) in COMBO_CIDS:
                            if getattr(card, "id", None) == MUNKIDORI_CID:
                                if not self.allow_munkidori:
                                    continue
                                self.flags.add("munki_ability_cand")
                            cands.append((0, ai))
                    elif t == int(OptionType.ATTACH) and self.allow_ability:
                        ecard = get_card(obs, opt.area, opt.index, self.me)
                        ecid = getattr(ecard, "id", None)
                        cdata = CARD_TABLE.get(ecid)
                        tmon = get_card(obs, opt.inPlayArea, opt.inPlayIndex, self.me)
                        if (self.allow_munkidori
                                and int(getattr(cdata, "energyType", -1) or -1) == DARK_ENERGY_TYPE
                                and getattr(tmon, "id", None) == MUNKIDORI_CID):
                            key = ("A", ecid, int(opt.inPlayArea), int(opt.inPlayIndex))
                            if key in dup_seen:
                                continue
                            dup_seen.add(key)
                            self.flags.add("munki_attach_cand")
                            cands.append((1, ai))
                        elif self.allow_attach_any and self._p1cap is None:
                            key = ("A", ecid, int(opt.inPlayArea), int(opt.inPlayIndex))
                            if key in dup_seen:
                                continue
                            dup_seen.add(key)
                            self.flags.add("compose_attach_cand")
                            cands.append((4, ai))
                        elif self.allow_attach_any:
                            self._rank4_excluded = True
                    elif (t == int(OptionType.EVOLVE) and self.compose
                          and self.allow_ability):
                        evc = get_card(obs, opt.area, opt.index, self.me)
                        evcid = getattr(evc, "id", None)
                        if evcid in COMPOSE_EVOLVE_CIDS:
                            ipa = getattr(opt, "inPlayArea", None)
                            ipi = getattr(opt, "inPlayIndex", None)
                            key = ("E", evcid, -1 if ipa is None else int(ipa),
                                   -1 if ipi is None else int(ipi))
                            if key in dup_seen:
                                continue
                            dup_seen.add(key)
                            self.flags.add("compose_evolve_cand")
                            cands.append((2, ai))
                    elif (t == int(OptionType.PLAY) and self.compose
                          and self.allow_ability):
                        hcard = get_card(obs, AreaType.HAND, opt.index, self.me)
                        hcid = getattr(hcard, "id", None)
                        if hcid in COMPOSE_PLAY_CIDS:
                            if self._p1cap is not None:
                                self._rank4_excluded = True
                                continue
                            key = ("P", hcid)
                            if key in dup_seen:
                                continue
                            dup_seen.add(key)
                            self.flags.add("compose_play_cand")
                            cands.append((4, ai))
                    elif (t == int(OptionType.RETREAT) and self.compose
                          and self.allow_ability):
                        self.flags.add("compose_retreat_cand")
                        cands.append((4, ai))
                    elif t == int(OptionType.ATTACK):
                        cands.append((3, ai))
                cands.sort()
                idxs = [ai for _, ai in cands]
            else:
                idxs = list(range(len(actions)))
                ctx = int(getattr(sel, "context", -1))
                is_counter = ctx in _COUNTER_CONTEXTS
                if is_counter:
                    keep = mask_dead_counter_targets(obs, actions)
                    if keep is not None:
                        kset = set(keep)
                        idxs = [ai for ai in idxs if ai in kset]
                    if prev_ctx == ctx and min_oi:
                        canon = [ai for ai in idxs if min(actions[ai]) >= min_oi]
                        if canon:
                            idxs = canon
                    if self._rv_active and len(idxs) > 1:
                        def _rv_key(ai):
                            oi = min(actions[ai])
                            op = sel.option[oi]
                            pi = getattr(op, "playerIndex", None)
                            pi = (1 - self.me) if pi is None else pi
                            mon = get_card(obs, op.area, op.index, pi)
                            cd = CARD_TABLE.get(getattr(mon, "id", None))
                            protected = (pi != self.me and bool(getattr(cd, "basic", False)))
                            return (1 if protected else 0, int(getattr(mon, "hp", 0) or 0))
                        idxs = sorted(idxs, key=_rv_key)
                        self.flags.add("rv_reorder")
                elif len(idxs) > 1:
                    try:
                        all_prize = all(
                            int(getattr(sel.option[oi], "area", -1) or -1) == int(AreaType.PRIZE)
                            for ai in idxs for oi in actions[ai])
                    except Exception:
                        all_prize = False
                    if all_prize:
                        self.flags.add("prize_canon")
                        idxs = idxs[:1]
                if len(idxs) > self.cap_follow:
                    self.flags.add("follow_truncated")
                    idxs = idxs[:self.cap_follow]
            h = hint[0] if hint else None
            if h is not None and h in idxs:
                idxs = [h] + [x for x in idxs if x != h]
            for ai in idxs:
                try:
                    child = self._step(ss.searchId, actions[ai])
                except (_Abort, _Pass1Stop):
                    raise
                except Exception:
                    self.flags.add("branch_error")
                    continue
                nh = hint[1:] if (hint and ai == h) else None
                if int(sel.type) != int(SelectType.MAIN) and int(getattr(sel, "context", -1)) in _COUNTER_CONTEXTS:
                    line = self._dfs(child, depth + 1, min(actions[ai]),
                                     int(getattr(sel, "context", -1)), hint=nh)
                else:
                    line = self._dfs(child, depth + 1, hint=nh)
                if line is not None:
                    return [(ai, _describe(obs, sel, actions[ai], self.me))] + line
            return None
        if len(actions) > self.cap_opp:
            self.flags.add("opp_overflow")
            return None
        first = None
        for ai in range(len(actions)):
            try:
                child = self._step(ss.searchId, actions[ai])
            except (_Abort, _Pass1Stop):
                raise
            except Exception:
                self.flags.add("branch_error")
                return None
            line = self._dfs(child, depth + 1, hint=(hint[1:] if hint else None))
            if line is None:
                return None
            if first is None:
                first = [(None, "<avv: qualsiasi scelta>")] + line
        return first
