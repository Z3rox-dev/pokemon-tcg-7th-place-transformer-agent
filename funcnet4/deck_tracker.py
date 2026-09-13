"""Remaining-deck accounting and prize inference, feeding the DKR and PRZ encoder words.

Both are derived from what the log has already revealed, never from hidden state.
"""
from collections import Counter

from cg.api import LogType, AreaType, Pokemon


def unseen_counts(obs, decklist, my_index):
    """cid -> count of my cards still in deck or prizes: decklist minus every visible zone."""
    counts = Counter()
    for cid in decklist:
        counts[int(cid)] += 1
    seen_serials = set()

    def _sub(card):
        if card is None:
            return
        is_poke = isinstance(card, Pokemon)
        if is_poke or getattr(card, "playerIndex", None) == my_index:
            ser = getattr(card, "serial", None)
            if ser not in seen_serials:
                counts[card.id] -= 1
                seen_serials.add(ser)
        if is_poke:
            for c in (card.energyCards or []):
                _sub(c)
            for c in (card.tools or []):
                _sub(c)
            for c in (card.preEvolution or []):
                _sub(c)

    st = obs.current
    ps = st.players[my_index]
    for zone in (ps.hand, ps.discard, ps.bench, ps.active, st.stadium):
        for card in (zone or []):
            _sub(card)
    if getattr(st, "looking", None) is not None:
        for card in st.looking:
            _sub(card)
    if getattr(obs, "select", None) is not None:
        _sub(getattr(obs.select, "effect", None))
    return +counts


class DeckPrizeTracker:
    """Per-game sticky state for known prizes. One instance per (game, seat)."""

    def __init__(self, decklist):
        self.decklist = [int(c) for c in decklist]
        self.reset()

    def reset(self):
        self.prized = Counter()
        self.resolved = False

    def observe(self, o):
        st = getattr(o, "current", None)
        if st is None:
            return
        if int(getattr(st, "turn", 0) or 0) == 0:
            self.reset()
            return
        me = st.yourIndex
        for lg in (getattr(o, "logs", None) or []):
            try:
                if (int(lg.type) == int(LogType.MOVE_CARD)
                        and getattr(lg, "playerIndex", None) == me
                        and int(lg.fromArea) == int(AreaType.PRIZE)
                        and getattr(lg, "cardId", None) is not None):
                    cid = int(lg.cardId)
                    if self.prized.get(cid, 0) > 0:
                        self.prized[cid] -= 1
            except Exception:
                continue
        sel = getattr(o, "select", None)
        if sel is not None and getattr(sel, "deck", None) is not None:
            unseen = unseen_counts(o, self.decklist, me)
            for card in sel.deck:
                if card is not None:
                    unseen[card.id] -= 1
            self.prized = +unseen
            self.resolved = True

    def prized_ids(self):
        """Card ids (with multiplicity) of the prizes still face down, [] if not resolved yet."""
        return list(self.prized.elements())
