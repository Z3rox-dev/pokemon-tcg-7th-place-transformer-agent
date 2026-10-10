"""DKR/PRZ-TAIL (2026-07-20) — contabilita' del MAZZO RESIDUO + inferenza PREMI ("il foglietto").

Problema (memoria energy-misplay / nrg-tail): la rete deve giocare "attorno alla carta che deve
arrivare" ma il residuo (decklist − mano − scarti − board) e' ricavabile solo sottraendo QUATTRO
somme DeepSets dentro i pesi, a ogni turno — aritmetica implicita che (misurato sul caso energie)
la rete NON consolida. Qui il residuo e' calcolato ESATTO in Python e consegnato come input.

Due livelli, due flag (bracci A/B separati, memoria funcnet3-feature-stack-net-negative):
  Word A (FUNCNET4_DKR_FEAT, ROLE_DECK_REMAINING): multiset `decklist − zone visibili` =
      carte in (mazzo ∪ premi). STATELESS: pura funzione di (obs, decklist) — `unseen_counts`.
  Word B (FUNCNET4_PRZ_FEAT, ROLE_PRIZE_KNOWN): multiset PREMIATO noto. STICKY: alla prima
      ricerca dell'INTERO mazzo l'engine consegna `select.deck` = TUTTE le carte rimaste nel
      mazzo (una entry per carta fisica, NON filtrate al tipo cercato — cg/api.py:407,530), e
      quindi `premiati = unseen − select.deck` per eliminazione. Le carte NON migrano fra mazzo
      e premi a partita in corso -> la deduzione vale per sempre; l'unico aggiornamento e' il
      premio PRESO (MOVE_CARD PRIZE->HAND con cardId, rivelato al proprietario, cg/api.py:216).
      I look-top-N viaggiano su un canale diverso (state.looking) e NON risolvono i premi.

La sottrazione e' il PORT ESATTO di agents/dragapult_rulebased.set_card_counts/add_card_count
(la routine che il RB usa in torneo da sempre): dedup per `serial`, ricorsione su energie
attaccate/tool/pre-evoluzioni, filtro ownership sulle Card, stadium/looking/select.effect
contati come VISTI. Miglioria vs RB: il RB ricalcola i premi solo alla ricerca successiva
(stantio dopo un premio preso); qui il MOVE_CARD aggiorna subito.

Observe() va chiamata su OGNI decisione, INCLUSE quelle a mossa-unica che gli agent
cortocircuitano prima della rete (stessa regola del KOTracker): le ricerche forzate e i log
dei premi transitano spesso proprio li'."""
from collections import Counter

from cg.api import LogType, AreaType, Pokemon


def unseen_counts(obs, decklist, my_index):
    """Counter cid->n: `decklist − zone visibili` = multiset delle MIE carte in (mazzo ∪ premi).
    Port esatto di agents/dragapult_rulebased.set_card_counts (stessa semantica, stesso ordine di
    zone); ritorna solo conteggi >0 (il RB lascia andare in negativo e li ignora — equivalente).
    LIMITE noto (misurato, verify X14): al SETUP (turn 0) i piazzamenti face-down hanno id nascosto
    nell'obs anche per il proprietario -> restano nel residuo finche' non si girano al turno 1.
    Off-by-few solo durante il setup, poi identita' esatta |residuo| == deckCount + premi."""
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
    return +counts                                    # +Counter = drop <=0


class DeckPrizeTracker:
    """Stato sticky per-PARTITA della Word B (premi noti). UNA istanza per (partita, seat);
    reset() a inizio partita (turn 0 riazzera come guardia, pattern KOTracker)."""

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
        if int(getattr(st, "turn", 0) or 0) == 0:     # nuova partita (guardia, come KOTracker)
            self.reset()
            return
        me = st.yourIndex
        # 1) premi PRESI dal delta-log: PRIZE -> (mano), cardId rivelato al proprietario.
        #    L'avversario che prende un premio e' MOVE_CARD_REVERSE senza cardId -> mai contato.
        for lg in (getattr(o, "logs", None) or []):
            try:
                if (int(lg.type) == int(LogType.MOVE_CARD)
                        and getattr(lg, "playerIndex", None) == me
                        and int(lg.fromArea) == int(AreaType.PRIZE)
                        and getattr(lg, "cardId", None) is not None):
                    cid = int(lg.cardId)
                    if self.prized.get(cid, 0) > 0:
                        self.prized[cid] -= 1
            except Exception:                          # log senza i campi area: non e' un premio
                continue
        # 2) ricerca dell'INTERO mazzo: select.deck = tutte le carte rimaste -> premi per
        #    eliminazione. Ricalcolo a OGNI reveal (self-correcting, copre anche gli edge).
        sel = getattr(o, "select", None)
        if sel is not None and getattr(sel, "deck", None) is not None:
            unseen = unseen_counts(o, self.decklist, me)
            for card in sel.deck:
                if card is not None:
                    unseen[card.id] -= 1
            self.prized = +unseen
            self.resolved = True

    def prized_ids(self):
        """Lista di card-id (con molteplicita') dei premi ancora coperti, [] se non risolti."""
        return list(self.prized.elements())
