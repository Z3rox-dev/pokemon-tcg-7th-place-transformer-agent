"""funcnet4 TEXT-FIRST encoding — builds the SAME 24-word encoder + per-action decoder, but with NO id-bag
indices: every card is described purely by its precomputed features (func + atom + text), plus a per-word
ROLE tag (own/opp/active/hand/...) and per-action PLANE tag (option-type/SelectContext).

REUSES the SHARED, verified dense extractors from funcnet.encoding (_fill_pokemon_dense, _fill_attack_dense
/_resolved/_action_atom/_target_dense, get_card, SparseVector) verbatim — text-first only changes the
OUTER assembly (no id-bag indices; DeepSets-summed groups; role/plane tags), not how a single card's
features are computed.

Output contract (consumed by funcnet4 losses/_tensors/farm plumbing):
  encoder sv: sv.dense = NUM_WORDS rows x ENC_DENSE_DIM_TF (1109 / 1145 con la LOC-tail armata);
              sv.role = NUM_WORDS ids in [0, N_ROLE)
  decoder sv: sv.dense = A rows x DEC_DENSE_DIM_TF
              (1076 = 283 feat/atom + 768 text + 3 NRG + 4 CNT + 3 UNLOCK + 15 TGT); sv.plane = A plane ids
  sv.index/value/offset are present (from word_start) but EMPTY/unused (no id-bag).

DESIGN (see funcnet4 + funcnet/STATUS.md): tools attached to a Pokemon are DROPPED (redundant with live
HP/state); energies & retreat cost are already in the func/dyn slice; only the card GROUPS (hand/deck/
discard/stadium) are re-summed (DeepSets SUM, not mean, to preserve a key card among 60)."""
import random
from collections import Counter

import numpy as np

from cg.api import AreaType, OptionType, SelectContext

from funcnet4.spec import (
    card_count, ENC_DENSE_DIM, ENC_DENSE_DIM_TF, ENC_LOC_OFF, ENC_LOC_DIM, LOC_WORD, LOC_FEAT,
    DEC_DENSE_DIM_TF, DEC_TEXT_OFF, EMB_DIM,
    ATOM_FEAT_DIM, V_ATOM_OFF, V_TEXT_OFF, FUNC_FEAT_DIM,
    DEC_ATOM_OFF, DEC_ATOM_DIM,
    DEC_NRG_OFF, DG_MISS_MIN, DG_MISS_MAX, DG_USEFUL, NRG_FEAT,
    DEC_CNT_OFF, DH_CNT_TO_KO, DH_KILL_PICK, DH_KILL_BUDGET, DH_PRIZE, CNT_FEAT,
    DEC_UNLOCK_OFF, DJ_UNLOCK_DMG, DJ_UNLOCK_GAIN, DJ_UNLOCK_KILLS, UNLOCK_FEAT,
    DEC_TGT_OFF, TG_NRG_N, TG_NRG_T0, TG_N_TYPES, TG_IS_ACTIVE, TG_N_TOOLS, TG_APPEARED,
    TG_HP, TG_MAXHP, TGT_FEAT,
    DEC_BENCH_OFF, BN_DMG, BN_KOS, BN_MIN_HP, BENCH_FEAT,
    SET_COUNT_OFF, SET_MULT_OFF,
    ROLE_OWN_ACTIVE, ROLE_OWN_BENCH, ROLE_OPP_ACTIVE, ROLE_OPP_BENCH,
    ROLE_OWN_PLAYER, ROLE_OPP_PLAYER, ROLE_HAND, ROLE_DECK, ROLE_STADIUM, ROLE_TURN,
    ROLE_HAND_CARD, N_HAND_TOKENS,
    ROLE_DECK_REMAINING, ROLE_PRIZE_KNOWN, PZ_RESOLVED_OFF, DKR_FEAT, PRZ_FEAT,
    PLANE_PAD, PLANE_ATTACK, PLANE_MAIN_OFF, PLANE_CTX_OFF,
)
from funcnet4.deck_tracker import unseen_counts
from funcnet4.loc_tail import fill_loc_dense
from funcnet.encoding import (
    SparseVector, get_card,
    _fill_pokemon_dense, _fill_attack_dense, _fill_attack_resolved,
    _fill_action_atom, _fill_target_dense,
)
from funcnet4.energy_mirror import (attach_nrg_scalars, cost_change_ctx,
                                    attack_missing_for_mon, provided_types as _provided_types)
from funcnet.text.tables import atom_features, attack_atom_features, text_embedding
from funcnet4.tables import attack_text_embedding
from funcnet.spec import CARD_TABLE as _CARD_TABLE_CNT
from funcnet.features import ATTACKS as _ATTACKS

_MUNKIDORI_CID = 112       # UNLOCK: {D} sblocca l'Adrena-Brain (deck-specifico come il lethal module)
_DARK_CID = 7
_ADRENA_VALUE = 30.0       # 3 segnalini/turno spostati dall'ability (conservativo: 1 turno)


def _resolved_dmg(base, cd_att, cd_def):
    """Danno stampato risolto con la regola engine (deb x2 sul tipo attaccante, res -30 floor 0).
    Nessun condizionale hardcodato: la rete vede il testo e corregge."""
    if not base:
        return 0.0
    et = int(cd_att.energyType) if (cd_att is not None and cd_att.energyType is not None) else None
    d = float(base)
    if cd_def is not None and et is not None:
        if cd_def.weakness is not None and int(cd_def.weakness) == et:
            d *= 2
        if cd_def.resistance is not None and int(cd_def.resistance) == et:
            d = max(0.0, d - 30)
    return d


def _mon_best_payable(mon, ctx, cd_def, extra_cid=None):
    """max danno RISOLTO fra gli attacchi di `mon` con mancanti==0 (opzionalmente CON extra energia)."""
    cd_att = _CARD_TABLE_CNT.get(getattr(mon, "id", None))
    if cd_att is None:
        return 0.0
    best = 0.0
    for aid in (getattr(cd_att, "attacks", None) or []):
        att = _ATTACKS.get(int(aid))
        if att is None:
            continue
        miss = attack_missing_for_mon(mon, att, ctx, extra_energy_cid=extra_cid)
        if miss == 0:
            best = max(best, _resolved_dmg(int(att.damage or 0), cd_att, cd_def))
    return best


def _unlock_value(my_mons, ctx, cd_def, extra_cid=None):
    """UNLOCK: max sui MIEI mon in gioco del miglior danno pagabile (+ valore-ability {D}/Munkidori).
    extra_cid=None -> base (cosa posso gia' fare); un cid-energia -> valore SE avessi quell'energia."""
    v = 0.0
    for mon in my_mons:
        v = max(v, _mon_best_payable(mon, ctx, cd_def, extra_cid=extra_cid))
    if extra_cid == _DARK_CID:
        for mon in my_mons:
            if int(getattr(mon, "id", -1)) == _MUNKIDORI_CID:
                has_dark = any(int(getattr(e, "id", -1)) == _DARK_CID
                               for e in (getattr(mon, "energyCards", None) or []))
                if not has_dark:
                    v = max(v, _ADRENA_VALUE)
    return v


def _prize_value_cnt(cid):
    """prize_canon per la CNT-tail: premi che l'avversario cede al KO (megaEx=3, ex=2, altrimenti 1).
    Stessa economia di lethal_dragapult_dusknoir._prize_value (duplicata qui: l'encoding non puo'
    importare il modulo lethal, che tira dentro la Search API)."""
    cd = _CARD_TABLE_CNT.get(cid) if cid is not None else None
    if cd is None:
        return 1
    if getattr(cd, "megaEx", False):
        return 3
    return 2 if getattr(cd, "ex", False) else 1


# non-Pokemon word dense layout (player / turn): scalars packed into otherwise-unused func slots; the ROLE
# tag tells the MLP which kind of word it is. Arbitrary-but-FIXED & documented.
P_DECK = 0; P_DISCARD = 1; P_HAND = 2; P_BENCH = 3; P_PRIZE = 4
P_POISON = 5; P_BURN = 6; P_ASLEEP = 7; P_PARAL = 8; P_CONFUSE = 9
# dedicated slots for the player word's discard DeepSets count/mult — MUST differ from SET_COUNT_OFF(0)/
# SET_MULT_OFF(1), else _fill_set_dense clobbers P_DECK/P_DISCARD (the live deck/discard counts). slots free.
P_DISC_CNT = 10; P_DISC_MULT = 11
T_TURN = 0; T_FIRST = 1; T_SUPPORTER = 2; T_STADIUM = 3; T_ENERGY = 4; T_RETREAT = 5


def _fill_set_dense(row, card_ids, count_off=SET_COUNT_OFF, mult_off=SET_MULT_OFF):
    """DeepSets group encoding: SUM of [atom|text] over card ids (VIEW slots) + count + multiplicity. count/
    mult are written to count_off/mult_off (default the dedicated SET slots 0/1; the PLAYER word passes
    P_DISC_CNT/P_DISC_MULT so they do NOT clobber its P_DECK/P_DISCARD scalars at slots 0/1). card_ids:
    iterable of int card ids."""
    cnt = Counter(c for c in card_ids if c is not None and 0 <= c < card_count)
    n = int(sum(cnt.values()))
    row[count_off] = n / 60.0
    row[mult_off] = (sum(1 for _c, k in cnt.items() if k >= 2) / max(1, len(cnt)))
    if n:
        atom = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
        txt = np.zeros(EMB_DIM, dtype=np.float32)
        for cid, k in cnt.items():
            atom += k * atom_features(cid)
            txt += k * text_embedding(cid)
        row[V_ATOM_OFF:V_ATOM_OFF + ATOM_FEAT_DIM] = atom
        row[V_TEXT_OFF:V_TEXT_OFF + EMB_DIM] = txt


def _card_ids(cards):
    return [getattr(c, "id", None) for c in (cards or [])]


def _valid_ids(card_ids):
    """The card ids that actually contribute to a DeepSets atom/text SUM (matches _fill_set_dense's filter:
    drop None / out-of-range). The FARM compact codec captures these so the learner can rebuild the atom+text
    view by table-lookup SUM, byte-identically to the direct dense."""
    return [int(c) for c in card_ids if c is not None and 0 <= c < card_count]


# ================================================================ ENCODER (state)
def get_encoder_input(obs, your_deck, view_dropout: float = 0.0, ids_out=None,
                      dkr: bool = False, prize_tracker=None, loc=None) -> SparseVector:
    """Text-first encoder. Same 24-word order as funcnet.encoding.get_encoder_input (8 own bench, 8 opp
    bench, own/opp active, own/opp player, hand, deck, stadium, turn); each word = a dense feature row + a
    ROLE tag, no id-bag. view_dropout (TRAINING ONLY) zeros the OPPONENT static stat fingerprint. If ids_out
    is a list, it is filled with 24 per-word card-id lists (FARM capture: the learner rebuilds each word's
    atom+text view as the table-lookup SUM over its id list, byte-identical to this dense).

    DKR/PRZ-TAIL: `dkr=True` (E flag DKR armato) appende le 2 parole mazzo-residuo/premi — SOLO per
    i modelli con l'arch a 38 parole: i call-site degli expert storici NON lo passano (default False,
    layout 36 invariato). `prize_tracker` = DeckPrizeTracker del seat (None -> Word B a zero).
    NB: per la Word A `your_deck` DEVE essere la DECKLIST INTERA (vero in tutti i siti live: selfplay/
    vecselfplay/evaluate passano la lista di partenza; la search determinizzata passa un fill parziale
    ma e' OFF e non arma mai dkr).

    LOC-TAIL: `loc=True` (E flag LOC armato) riempie la coda [ENC_LOC_OFF:] sulla parola own-player
    (LOC_WORD=18) con i conti per (categoria, zona). ⚠ Differenza deliberata rispetto a `dkr=`: qui
    il call-site arma solo la SCRITTURA DEI VALORI, mai la LARGHEZZA della riga — quella segue il
    solo flag env (ENC_DENSE_DIM_TF), quindi NESSUN call-site puo' produrre una riga di larghezza
    sbagliata per il modello del processo. E' la trappola che la DKR ha pagato al deploy Kaggle
    (agents/funcnet4_agent.py senza `dkr=` -> 36 parole vs modello a 38 -> fail-soft = gioco random).
    A `loc=False` la coda resta a ZERO: costo CPU nullo dove non serve (search, expert storici)."""
    # 🔴 GUARDIA ANTI-LOBOTOMIA (2026-08-07). Con LOC_FEAT armato la LARGHEZZA della riga segue
    # il flag env, NON il call-site: un chiamante che dimentica `loc=` ottiene una riga larga
    # giusta (1145) con la coda [ENC_LOC_OFF:] a ZERO ESATTO. Nessuna eccezione, reshape riuscito,
    # forward valido -> si misura una rete con l'ingresso staccato, in silenzio. E' la firma di
    # payoff-env-flags-lobotomized-net-74pp e di box192. Con ~40 call-site in tools/ il difetto
    # non si chiude patchandoli tutti: si chiude rendendo IMPOSSIBILE dimenticarsene.
    # Percio' `loc` non ha piu' un default silenzioso: va dichiarato. I percorsi di PRODUZIONE
    # (selfplay, vecselfplay, evaluate, imitation, replay_corpus) lo passano gia'.
    if loc is None:
        if LOC_FEAT:
            raise RuntimeError(
                "get_encoder_input: FUNCNET4_LOC_FEAT e' ARMATO ma questo call-site non ha "
                "passato `loc=`. Senza, la riga esce larga 1145 con la coda LOC a ZERO e nessun "
                "errore: misureresti una rete con l'ingresso staccato. Passa `loc=LOC_FEAT` "
                "(e `prize_tracker=`, altrimenti le 9 colonne premi restano a zero), oppure "
                "`loc=False` se vuoi DAVVERO la coda spenta.")
        loc = False
    your_index = obs.current.yourIndex
    state = obs.current
    sv = SparseVector(ENC_DENSE_DIM_TF)
    sv.role = []
    # unseen_counts e' un giro su mano+scarti+panca+attivo+stadium+looking con ricorsione: si calcola
    # UNA volta e la usano sia la LOC-tail (parola 18) sia la Word A del DKR (parola 36). Hoisting
    # value-identical: stessa funzione, stessi argomenti, nessun consumo di RNG.
    _rem = None
    if (loc and LOC_FEAT) or (dkr and DKR_FEAT):
        _rem = unseen_counts(obs, list(your_deck), your_index)

    def _poke_word(poke, is_own, role):
        sv.word_start()
        wids = []
        if poke is not None:
            _fill_pokemon_dense(sv.dense[-1], poke)
            wids = [poke.id]
            if (not is_own) and view_dropout > 0.0 and random.random() < view_dropout:
                sv.dense[-1][:FUNC_FEAT_DIM] = 0.0
        sv.role.append(role)
        if ids_out is not None:
            ids_out.append(_valid_ids(wids) if poke is not None else [])

    for i in range(2):                                # 8 own bench, 8 opp bench (own seat = i==0)
        ps = state.players[i ^ your_index]
        is_own = (i == 0)
        role = ROLE_OWN_BENCH if is_own else ROLE_OPP_BENCH
        for j in range(8):
            _poke_word(ps.bench[j] if j < len(ps.bench) else None, is_own, role)

    for i in range(2):                                # own active, opp active
        ps = state.players[i ^ your_index]
        is_own = (i == 0)
        _poke_word(ps.active[0] if ps.active else None, is_own,
                   ROLE_OWN_ACTIVE if is_own else ROLE_OPP_ACTIVE)

    for i in range(2):                                # own player, opp player (scalars + DeepSets discard)
        ps = state.players[i ^ your_index]
        is_own = (i == 0)
        sv.word_start(); row = sv.dense[-1]
        row[P_DECK] = ps.deckCount / 60.0
        row[P_DISCARD] = len(ps.discard) / 60.0
        row[P_HAND] = ps.handCount / 8.0
        row[P_BENCH] = len(ps.bench) / 5.0
        row[P_PRIZE] = len(ps.prize) / 6.0
        row[P_POISON] = float(ps.poisoned); row[P_BURN] = float(ps.burned)
        row[P_ASLEEP] = float(ps.asleep); row[P_PARAL] = float(ps.paralyzed); row[P_CONFUSE] = float(ps.confused)
        _fill_set_dense(row, _card_ids(ps.discard), count_off=P_DISC_CNT, mult_off=P_DISC_MULT)
        # LOC-TAIL: solo sulla parola OWN (18). Sulla parola opp-player sarebbe un LEAK (i conti
        # sono derivati dalla MIA decklist, che l'avversario non conosce) -> resta a zero.
        if is_own and loc and LOC_FEAT:
            fill_loc_dense(row, ENC_LOC_OFF, _rem, prize_tracker, ps.discard, your_index)
        sv.role.append(ROLE_OWN_PLAYER if is_own else ROLE_OPP_PLAYER)
        if ids_out is not None:
            ids_out.append(_valid_ids(_card_ids(ps.discard)))

    sv.word_start()                                  # hand (own)
    _fill_set_dense(sv.dense[-1], _card_ids(state.players[your_index].hand))
    sv.role.append(ROLE_HAND)
    if ids_out is not None:
        ids_out.append(_valid_ids(_card_ids(state.players[your_index].hand)))

    sv.word_start()                                  # deck (own) — your_deck is a list of int ids
    _fill_set_dense(sv.dense[-1], list(your_deck))
    sv.role.append(ROLE_DECK)
    if ids_out is not None:
        ids_out.append(_valid_ids(list(your_deck)))

    sv.word_start()                                  # stadium (shared)
    _fill_set_dense(sv.dense[-1], _card_ids(state.stadium))
    sv.role.append(ROLE_STADIUM)
    if ids_out is not None:
        ids_out.append(_valid_ids(_card_ids(state.stadium)))

    sv.word_start(); row = sv.dense[-1]               # turn info
    row[T_TURN] = state.turn / 10.0
    row[T_FIRST] = float(state.firstPlayer == your_index)
    row[T_SUPPORTER] = float(state.supporterPlayed)
    row[T_STADIUM] = float(state.stadiumPlayed)
    row[T_ENERGY] = float(state.energyAttached)
    row[T_RETREAT] = float(state.retreated)
    sv.role.append(ROLE_TURN)
    if ids_out is not None:
        ids_out.append([])                            # turn word carries no atom/text view

    # ---- funcnet4 PUNTO-2: TOKEN-MANO (parole 24..24+N_HAND_TOKENS-1, DOPO le 24 funcnet3 cosi'
    # il prefisso resta il layout tf3 esatto — warm-start e loop per-parola esistenti invariati).
    # Ogni carta in mano = una parola con i SUOI atom+text (slot VIEW), ROLE_HAND_CARD; ordinate
    # per card-id (canonico, permutation-invariant); slot oltre la mano = riga zero (pattern
    # bench vuoti); overflow oltre N resta coperto dalla parola-somma ROLE_HAND (superset).
    # FARM capture: ids_out riceve [cid] per token -> il learner ricostruisce la view della
    # parola come table-lookup della singola carta, byte-identica alla dense diretta.
    hand_cids = sorted(_valid_ids(_card_ids(state.players[your_index].hand)))
    for j in range(N_HAND_TOKENS):
        sv.word_start()
        if j < len(hand_cids):
            cid = hand_cids[j]
            row = sv.dense[-1]
            row[V_ATOM_OFF:V_ATOM_OFF + ATOM_FEAT_DIM] = atom_features(cid)
            row[V_TEXT_OFF:V_TEXT_OFF + EMB_DIM] = text_embedding(cid)
            if ids_out is not None:
                ids_out.append([cid])
        else:
            if ids_out is not None:
                ids_out.append([])
        sv.role.append(ROLE_HAND_CARD)

    # ---- DKR/PRZ-TAIL (2026-07-20): parole 36 (Word A residuo) e 37 (Word B premi noti).
    # Costruite SOLO se il call-site arma dkr (modello a 38 parole) E il flag env e' acceso —
    # entrambe le condizioni, cosi' un sito armato con env spento resta al layout 36 (no-op)
    # e un env acceso non contamina gli expert storici (che non passano dkr).
    if dkr and DKR_FEAT:
        rem_ids = list(_rem.elements())          # residuo gia' calcolato in testa (hoisting, vedi sopra)
        sv.word_start()                              # Word A: residuo (mazzo ∪ premi), SET-word standard
        _fill_set_dense(sv.dense[-1], rem_ids)
        sv.role.append(ROLE_DECK_REMAINING)
        if ids_out is not None:
            ids_out.append(_valid_ids(rem_ids))
        sv.word_start(); row = sv.dense[-1]          # Word B: premi noti (sticky post-reveal), zero se PRZ off
        prz_ids = []
        if PRZ_FEAT and prize_tracker is not None and prize_tracker.resolved:
            prz_ids = prize_tracker.prized_ids()
            _fill_set_dense(row, prz_ids)
            row[PZ_RESOLVED_OFF] = 1.0               # anche a premi finiti (multiset vuoto ma RISOLTO)
        sv.role.append(ROLE_PRIZE_KNOWN)
        if ids_out is not None:
            ids_out.append(_valid_ids(prz_ids))
    return sv


# ================================================================ DECODER (actions)
_IN_PLAY_AREAS = (AreaType.ACTIVE, AreaType.BENCH)


def _in_play_mon(obs, area, index, player_index):
    """Il Pokemon IN GIOCO a (area,index) del giocatore, o None se l'area non e' attivo/panca.
    Bounds-safe come get_card: un indice fuori range non deve mai far cadere l'encoding."""
    if area not in _IN_PLAY_AREAS or index is None or player_index is None:
        return None
    try:
        mon = get_card(obs, area, int(index), int(player_index))
    except (IndexError, KeyError, TypeError):
        return None
    return mon if getattr(mon, "id", None) is not None else None


def _fill_tgt_tail(row, mon, area):
    """TGT-TAIL: lo stato VIVO del Pokemon in gioco associato a questa azione (energie attaccate per
    tipo, posizione, tool, appena-entrato). E' cio' che separa due copie della stessa carta: senza,
    'attacca il {D} al Munkidori A' e '...al Munkidori B' sono la STESSA riga (misurato: 85/85 nel
    percorso ability di Blaziken, pareggio di score esatto). Flag OFF -> chiamante non ci arriva.

    Le scale sono quelle dell'ENCODER (_fill_pokemon_dense): n/8 e conteggio-per-tipo/4, cosi' il
    reader impara la stessa aritmetica su parola-board e riga-azione.

    ACCUMULA (+=) su ogni slot, come _fill_card_views: un'azione MULTI-carta e' una somma DeepSets
    sui Pokemon che tocca. Con l'assegnazione vinceva l'ultimo bersaglio e due azioni che finivano
    sullo stesso restavano identiche (misurato: tutti i 16 gruppi residui erano di questa forma).
    Su un bersaglio singolo += e = coincidono: la riga parte da zero."""
    if mon is None:
        return
    energies = getattr(mon, "energies", None) or []
    row[DEC_TGT_OFF + TG_NRG_N] += len(energies) / 8.0
    for e in energies:
        t = int(e)
        if 0 <= t < TG_N_TYPES:
            row[DEC_TGT_OFF + TG_NRG_T0 + t] += 1.0 / 4.0
    row[DEC_TGT_OFF + TG_IS_ACTIVE] += 1.0 if area == AreaType.ACTIVE else 0.0
    row[DEC_TGT_OFF + TG_N_TOOLS] += len(getattr(mon, "tools", None) or []) / 2.0
    row[DEC_TGT_OFF + TG_APPEARED] += 1.0 if getattr(mon, "appearThisTurn", False) else 0.0
    row[DEC_TGT_OFF + TG_HP] += (getattr(mon, "hp", None) or 0) / 400.0
    row[DEC_TGT_OFF + TG_MAXHP] += (getattr(mon, "maxHp", None) or 0) / 400.0


def _fill_action_target(row, obs, area, index, player_index, is_opp=False):
    """Descrive il Pokemon in gioco bersagliato/coinvolto da un'azione: colonne DEC_TGT_* gia'
    esistenti (hp/maxHp/ex/ritirata — il ramo MIO non le scriveva affatto) + la TGT-tail.
    No-op se il flag e' spento o se l'azione non tocca un Pokemon in gioco -> riga bit-identica."""
    if not TGT_FEAT:
        return None
    mon = _in_play_mon(obs, area, index, player_index)
    if mon is None:
        return None
    _fill_target_dense(row, mon, is_opp=is_opp)
    _fill_tgt_tail(row, mon, area)
    return mon


def _fill_card_views(row, cid, cap=None):
    """ACCUMULATE the action's atom (DEC_ATOM slot) + text (DEC_TEXT slot) from a card id. Uses += (not =) so
    a MULTI-card selection (e.g. discard A+B) SUMS all chosen cards' views (DeepSets), instead of keeping only
    the LAST (the deleted id-bag used to accumulate; text-first must too). The row starts at zeros, so a
    single-card action is bit-identical to assignment. If cap is a list, the contributing cid is appended (FARM
    capture: the learner rebuilds this action's atom+text as the table SUM over cap)."""
    if cid is None or not (0 <= cid < card_count):
        return
    row[DEC_ATOM_OFF:DEC_ATOM_OFF + DEC_ATOM_DIM] += atom_features(cid)
    row[DEC_TEXT_OFF:DEC_TEXT_OFF + EMB_DIM] += text_embedding(cid)
    if cap is not None:
        cap.append(int(cid))


def get_decoder_input(obs, actions, ids_out=None, bench_vals=None) -> SparseVector:
    """Text-first decoder. Per action: a dense row (attack/target/resolved/atom + the acting card/attack
    TEXT embedding) + a PLANE tag (option-type / SelectContext). No id-bag indices. If ids_out is a list, it
    is filled with one (attack_id, [card_ids]) capture per action (FARM: the learner rebuilds each action's
    atom+text from the ATTACK tables for attack_id>=0, else the SUM over the card table for [card_ids],
    byte-identical to this dense).

    BENCH-TAIL: `bench_vals` = mappa attackId -> (danno_piazzabile, max_ko, hp_minimi), calcolata
    dal CHIAMANTE con funcnet4/bench_counters.py (che interroga il motore). NON viene calcolata qui:
    questa funzione gira anche dentro la search come valutatore di foglie, e una query al motore
    qui dentro chiuderebbe l'albero della search chiamante. None (default) = coda a zero, cioe' il
    comportamento pre-coda bit-identico — ed e' il valore giusto per i call-site dentro la search."""
    sv = SparseVector(DEC_DENSE_DIM_TF)
    sv.plane = []
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    opp = obs.current.players[1 - your_index]
    my_active = ps.active[0] if ps.active else None
    opp_active = opp.active[0] if opp.active else None
    context = obs.select.context
    # NRG-TAIL: contesto modificatori-di-costo valutato UNA volta per decisione (flag OFF = salta tutto)
    nrg_ctx = cost_change_ctx(obs.current, your_index) if NRG_FEAT else None
    # CNT-TAIL: attivo SOLO sui contesti di piazzamento segnalini (13/14); il budget rimanente e'
    # engine-exact (select.remainDamageCounter). Flag OFF = colonne a zero (bit-identico pre-tail).
    cnt_place = CNT_FEAT and int(context) in (int(SelectContext.DAMAGE_COUNTER),
                                              int(SelectContext.DAMAGE_COUNTER_ANY))
    cnt_remain = getattr(obs.select, "remainDamageCounter", None) if cnt_place else None
    # UNLOCK-TAIL: valore-sbloccato per le righe carta-energia MIE (recupero/ricerca). base e ctx
    # calcolati UNA volta per decisione; unlock(E) cache-ato per cid-energia (righe con la stessa
    # energia danno lo stesso valore). Flag OFF = niente (colonne a zero, bit-identico).
    unlock_ctx = unlock_base = None
    unlock_cache = {}
    _unlock_mons = None
    if UNLOCK_FEAT:
        unlock_ctx = nrg_ctx if nrg_ctx is not None else cost_change_ctx(obs.current, your_index)
        _unlock_mons = [m for z in (ps.active, ps.bench) for m in (z or []) if m is not None]
        _cd_def_u = _CARD_TABLE_CNT.get(getattr(opp_active, "id", None)) if opp_active is not None else None
        unlock_base = _unlock_value(_unlock_mons, unlock_ctx, _cd_def_u)
        _uohp = int(getattr(opp_active, "hp", 0) or 0)

    for _ai, action in enumerate(actions):
        sv.word_start(); row = sv.dense[-1]
        if len(action) == 0:
            sv.plane.append(PLANE_PAD)
            if ids_out is not None:
                ids_out.append((-1, []))
            continue
        plane = PLANE_PAD
        cap_attack = -1
        cap_cids = []                                # FARM: card ids accumulated into this action's atom+text
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    plane = 1
                case OptionType.YES:
                    plane = 2
                case OptionType.NO:
                    plane = 3
                case OptionType.SPECIAL_CONDITION:
                    plane = 4 + int(o.specialConditionType)
                case OptionType.NUMBER:
                    plane = 9 + min(int(o.number), 4)
                case OptionType.ATTACK:
                    plane = PLANE_ATTACK
                    _fill_attack_dense(row, o.attackId)
                    _fill_attack_resolved(row, o.attackId, my_active, opp_active)
                    _fill_action_atom(row, attack_atom_features(o.attackId))
                    row[DEC_TEXT_OFF:DEC_TEXT_OFF + EMB_DIM] = attack_text_embedding(o.attackId)
                    cap_attack = int(o.attackId)
                    if BENCH_FEAT and bench_vals:
                        # BENCH-TAIL: la fetta risolta sopra e' calcolata SOLO contro l'ATTIVO
                        # avversario. Qui arriva cio' che l'attacco fa alla PANCA — misurato dal
                        # motore, non dedotto: 0 significa "bloccato o niente panca", e non
                        # abbiamo dovuto sapere che Battle Cage & co. esistono.
                        _bv = bench_vals.get(int(o.attackId))
                        if _bv is not None:
                            row[DEC_BENCH_OFF + BN_DMG] = float(_bv[0]) / 400.0
                            row[DEC_BENCH_OFF + BN_KOS] = float(_bv[1]) / 3.0
                            row[DEC_BENCH_OFF + BN_MIN_HP] = float(_bv[2]) / 400.0
                case OptionType.PLAY:
                    plane = PLANE_MAIN_OFF + 0
                    _fill_card_views(row, getattr(ps.hand[o.index], "id", None), cap=cap_cids)
                case OptionType.ATTACH:
                    plane = PLANE_MAIN_OFF + 1
                    ecard = get_card(obs, o.area, o.index, your_index)
                    tmon = get_card(obs, o.inPlayArea, o.inPlayIndex, your_index)
                    _fill_card_views(row, getattr(ecard, "id", None), cap=cap_cids)
                    _fill_target_dense(row, tmon, is_opp=False)
                    # TGT-TAIL: qui _fill_target_dense c'era gia' (hp/ex/ritirata), mancavano le
                    # ENERGIE del bersaglio — il 53% delle collisioni misurate si separa solo con
                    # quelle (due Munkidori 70/70, uno col {D} e uno no).
                    _fill_action_target(row, obs, o.inPlayArea, o.inPlayIndex, your_index)
                    if NRG_FEAT and tmon is not None:
                        # NRG-TAIL: il conto-energie derivato (mirror engine, modificatori inclusi)
                        # sull'attach di una carta-ENERGIA; non-energia/mon ignoto -> None -> slice zero.
                        nrg = attach_nrg_scalars(getattr(ecard, "id", None), tmon, nrg_ctx)
                        if nrg is not None:
                            row[DEC_NRG_OFF + DG_MISS_MIN] = nrg[0]
                            row[DEC_NRG_OFF + DG_MISS_MAX] = nrg[1]
                            row[DEC_NRG_OFF + DG_USEFUL] = nrg[2]
                case OptionType.EVOLVE:
                    plane = PLANE_MAIN_OFF + 3
                    _fill_card_views(row, getattr(get_card(obs, o.area, o.index, your_index), "id", None), cap=cap_cids)
                    _fill_target_dense(row, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index), is_opp=False)
                    # su quale delle due copie evolvo: quella carica o quella appena giocata?
                    _fill_action_target(row, obs, o.inPlayArea, o.inPlayIndex, your_index)
                case OptionType.ABILITY:
                    plane = PLANE_MAIN_OFF + 5
                    _fill_card_views(row, getattr(get_card(obs, o.area, o.index, your_index), "id", None), cap=cap_cids)
                    # la riga ABILITY portava SOLO l'identita' della carta: due Munkidori che
                    # offrono entrambi Adrena-Brain erano la stessa riga (audit ability_sight_probe)
                    _fill_action_target(row, obs, o.area, o.index, your_index)
                case OptionType.DISCARD:
                    plane = PLANE_MAIN_OFF + 6
                    _fill_card_views(row, getattr(get_card(obs, o.area, o.index, your_index), "id", None), cap=cap_cids)
                    _fill_action_target(row, obs, o.area, o.index, your_index)
                case OptionType.RETREAT:
                    plane = PLANE_MAIN_OFF + 7
                    _fill_target_dense(row, my_active, is_opp=False)
                    if TGT_FEAT:
                        _fill_tgt_tail(row, my_active, AreaType.ACTIVE)
                case OptionType.CARD | OptionType.TOOL_CARD | OptionType.ENERGY_CARD | OptionType.ENERGY:
                    plane = PLANE_CTX_OFF + int(context)
                    is_opp = (o.playerIndex is not None and o.playerIndex != your_index)
                    holder = get_card(obs, o.area, o.index, o.playerIndex)
                    if o.type == OptionType.CARD:
                        target = holder
                    elif o.type == OptionType.TOOL_CARD:
                        target = holder.tools[o.toolIndex] if holder is not None else None
                    else:
                        target = holder.energyCards[o.energyIndex] if holder is not None else None
                    if target is not None:
                        if is_opp:
                            descr = holder if getattr(holder, "id", None) is not None else target
                            _fill_target_dense(row, descr, is_opp=True)
                            _fill_card_views(row, getattr(descr, "id", None), cap=cap_cids)
                            # TGT-TAIL anche di la': l'avversario aveva gia' hp/ex ma non le energie
                            # (due suoi Dreepy identici, uno carico, collidono lo stesso).
                            _fill_action_target(row, obs, o.area, o.index, o.playerIndex, is_opp=True)
                            if cnt_place and o.type == OptionType.CARD:
                                # CNT-TAIL: aritmetica-segnalini del bersaglio candidato (il conto
                                # a soglia che la rete non consolida dal float hp/400). Solo pick di
                                # piazzamento su pokemon AVVERSARIO in gioco; il resto resta a zero.
                                _hp = getattr(holder, "hp", None)
                                if _hp is not None and _hp > 0:
                                    _need = -(-int(_hp) // 10)               # ceil(hp/10)
                                    row[DEC_CNT_OFF + DH_CNT_TO_KO] = _need / 30.0
                                    row[DEC_CNT_OFF + DH_KILL_PICK] = 1.0 if _need == 1 else 0.0
                                    row[DEC_CNT_OFF + DH_KILL_BUDGET] = (
                                        1.0 if cnt_remain is not None and cnt_remain >= _need else 0.0)
                                    row[DEC_CNT_OFF + DH_PRIZE] = _prize_value_cnt(
                                        getattr(holder, "id", None)) / 3.0
                        else:
                            _fill_card_views(row, getattr(target, "id", None), cap=cap_cids)
                            # ---- IL BUCO PRINCIPALE (2026-08-02). Fin qui la riga di un bersaglio
                            # MIO portava SOLO atom+text della carta: nessun hp, nessuna energia,
                            # nessuno slot. E' il ramo che serve l'ability "Seething Spirit" di
                            # Blaziken ex ("a quale mio Pokemon attacco l'energia?", ctx ATTACH_FROM)
                            # e ogni SWITCH/TO_ACTIVE/DISCARD/segnalini sul mio lato: due copie della
                            # stessa carta davano righe IDENTICHE bit-per-bit (85/85 misurate) e
                            # quindi score in pareggio ESATTO -> monetina, e gradiente cieco.
                            # NB: descrive il DETENTORE (il Pokemon), non la carta scelta: per una
                            # riga ENERGY_CARD la carta e' l'energia, ma cio' che distingue due
                            # opzioni gemelle e' lo stato del mon a cui e' attaccata.
                            _fill_action_target(row, obs, o.area, o.index,
                                                o.playerIndex if o.playerIndex is not None else your_index)
                            if UNLOCK_FEAT:
                                # UNLOCK-TAIL: se questa carta e' un'energia MIA (recupero/ricerca),
                                # il valore che sblocca. "prendo {P} -> Phantom Dive 200" scritto sulla
                                # riga (il conto costo-attacco che la rete non deriva dal float). La
                                # scelta resta sua (Regola-3). Cache per cid (righe stessa energia = ==).
                                _ecid = getattr(target, "id", None)
                                if _ecid is not None and _ecid in unlock_cache:
                                    _u = unlock_cache[_ecid]
                                elif _ecid is not None:
                                    _u = (_unlock_value(_unlock_mons, unlock_ctx, _cd_def_u,
                                                        extra_cid=int(_ecid))
                                          if _provided_types(int(_ecid)) is not None else None)
                                    unlock_cache[_ecid] = _u
                                else:
                                    _u = None
                                if _u is not None:
                                    row[DEC_UNLOCK_OFF + DJ_UNLOCK_DMG] = _u / 400.0
                                    row[DEC_UNLOCK_OFF + DJ_UNLOCK_GAIN] = max(0.0, _u - unlock_base) / 400.0
                                    row[DEC_UNLOCK_OFF + DJ_UNLOCK_KILLS] = (
                                        1.0 if _uohp > 0 and _u >= _uohp else 0.0)
                case OptionType.SKILL:
                    plane = PLANE_CTX_OFF + int(context)
                    _fill_card_views(row, o.cardId, cap=cap_cids)
        sv.plane.append(plane)
        if ids_out is not None:
            ids_out.append((cap_attack, cap_cids))
    return sv
