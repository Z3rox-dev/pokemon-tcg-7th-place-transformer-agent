"""Kaggle deploy agent for a funcnet4 FuncNetTF (12 hand tokens: 36 words base, 38 with DKR/PRZ).
Greedy, one forward per move; mirrors funcnet4.evaluate.RecurrentAgent.__call__ but
self-contained (no evaluate.py training imports).

FORK di agents/funcnet3_agent.py — le UNICHE differenze deploy-rilevanti di funcnet4:
  * import da funcnet4.* (encoding TOKEN-MANO: 12 parole in piu'; DKR/PRZ ne aggiungono altre 2).
    L'encoding tf4 emette i token-mano SOLO se spec.N_HAND_TOKENS>0 (default 12, via env
    FUNCNET_N_HAND_TOKENS). Il ckpt tf4 e' allenato con 12 token -> il default combacia; per
    sicurezza settiamo FUNCNET_N_HAND_TOKENS=12 in main.py PRIMA dell'import (pinning).
  * attention legacy/v2 auto-rilevata dal ckpt senza ignorare head sconosciuti.
Tutto il resto (auto-detect d_model/ff/heads, lethal deterministico, no-overkill,
fail-soft) e' identico a funcnet3_agent — funcnet4 e' un fork 1:1 col solo namespace/encoding diverso.

Config via env: FUNCNET4_CKPT (default model.pth), FUNCNET4_ENC/DEC, FUNCNET4_LETHAL (0/1/2, def 2),
FUNCNET4_NO_OVERKILL (0/1, def 1), FUNCNET4_FEZ_MASK (0/1, def 1 — Regola 6: mai Fezandipiti ex
con Flip the Script morto; richiede NO_OVERKILL), FUNCNET4_ENERGY_STRICT (0/1, def 0 — Regola 3
a cap SECCO: slack 0 invece del +1 storico), FUNCNET4_ENERGY_TELEM (0/1, def 0 — contatori probe
Regola 3 in ENERGY_TELEM, solo per harness A/B). Fail-soft: qualunque eccezione in una mossa ->
selezione legale random (mai crash della partita).
"""
import os
import sys
import time
import random
import pathlib
import traceback

# TOKEN-MANO pinning: DEVE stare PRIMA di importare funcnet4.spec (NUM_WORDS si fissa a import-time
# da FUNCNET_N_HAND_TOKENS). Il ckpt usa 12 token; DKR può portare il totale da 36 a 38 parole.
os.environ.setdefault("FUNCNET_N_HAND_TOKENS", "12")

_paths = [os.getcwd(), "/kaggle_simulations/agent"]
try:
    _repo = str(pathlib.Path(__file__).resolve().parents[1])
    _paths += [_repo, os.path.join(_repo, "engine")]
except NameError:
    pass
for _p in _paths:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from cg.api import to_observation_class
from bytrl.actions import enumerate_actions
from funcnet4.model import FuncNetTF, infer_action_attention
from funcnet4.encoding import get_encoder_input, get_decoder_input

CKPT = os.environ.get("FUNCNET4_CKPT", "model.pth")
ENC = int(os.environ.get("FUNCNET4_ENC", "8"))
DEC = int(os.environ.get("FUNCNET4_DEC", "8"))


def _read_deck():
    for fp in ["deck.csv", "/kaggle_simulations/agent/deck.csv"]:
        if os.path.exists(fp):
            rows = [r for r in open(fp).read().split("\n") if r.strip()]
            return [int(rows[i]) for i in range(60)]
    raise FileNotFoundError("deck.csv")


def _find_ckpt():
    cands = [CKPT, os.path.join("/kaggle_simulations/agent", os.path.basename(CKPT))]
    try:
        here = pathlib.Path(__file__).resolve().parent
        cands += [str(here / CKPT), str(here / os.path.basename(CKPT)),
                  str(here.parent / os.path.basename(CKPT))]
    except NameError:
        pass
    for fp in cands:
        if fp and os.path.exists(fp):
            return fp
    return None


DECK = _read_deck()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_ck = _find_ckpt()
_sd = None
if _ck:
    _sd = torch.load(_ck, map_location="cpu")
    _sd = {k: (v.float() if v.is_floating_point() else v) for k, v in _sd.items()}

if _sd is not None:
    _dm = int(_sd["encoder.layers.0.self_attn.in_proj_weight"].shape[1])
    _ff = int(_sd["encoder.layers.0.linear1.weight"].shape[0])
    _attn_kw = infer_action_attention(_sd)
    # RECURRENT auto-detect (2026-07-26): a GRU checkpoint carries rnn.* / belief* keys; build the model
    # with recurrent=gru + its rec_hidden so they load (miss=0/unexp=0). The play loop already threads
    # `state` across the game (state=_state -> new_state), so the belief flows. Back-compat: a none ckpt
    # has no rnn.* -> recurrent=none, byte-identical to before.
    _recurrent = "gru" if any(k.startswith("rnn.") for k in _sd) else "none"
    _rec_hidden = int(_sd["rnn.weight_hh_l0"].shape[1]) if _recurrent != "none" else 256
    # VALUE-BINS auto-detect (2026-08-07): dal 7/8 il main gira con --value-bins 51 e i suoi ckpt
    # portano value_dist_head + value_bin_centers. Senza rilevarli il modello nasce a testa scalare e
    # quelle tre chiavi finiscono in `unexpected` -> l'assert qui sotto fa fallire il bundle. Stessa
    # disciplina di GRU e pointer: l'architettura si legge dal CHECKPOINT, non da una env-flag.
    _vbins = int(_sd["value_dist_head.weight"].shape[0]) if "value_dist_head.weight" in _sd else 0
else:
    _dm, _ff, _attn_kw = 256, 512, {"act_selfattn": 0, "act_compare": 0}
    _recurrent, _rec_hidden = "none", 256
    _vbins = 0

# HEADS: non inferibile dai pesi (l'in_proj della MHA ha la stessa shape con 2, 4 o 8 teste, cambia
# solo la PARTIZIONE del softmax) -> _dm//64 resta il default storico (256->4), ma un ckpt allenato
# con un rapporto diverso (es. dmodel 128 a 4 teste, il clone BC 2026-08-10) DEVE poterlo dichiarare:
# a teste sbagliate il load passa pulito (miss=0/unexp=0) e la rete calcola un'attention DIVERSA da
# quella del training -> la classe di guasto silenzioso DKR-random, non un errore.
_heads = int(os.environ.get("FUNCNET4_HEADS", str(max(1, _dm // 64))))
_model = FuncNetTF(_dm, _heads, _ff, ENC, DEC, recurrent=_recurrent, rec_hidden=_rec_hidden,
                   value_bins=_vbins, **_attn_kw).eval()
if _sd is not None:
    _miss, _unexp = _model.load_state_dict(_sd, strict=False)
    if _miss or _unexp:
        raise RuntimeError(f"checkpoint arch mismatch: missing={list(_miss)[:8]} unexpected={list(_unexp)[:8]}")
    print(f"[funcnet4_agent] loaded {_ck} d{_dm} ff{_ff} words={_model.num_words} "
          f"legacy_sa={_attn_kw['act_selfattn']} act_compare={_attn_kw['act_compare']} "
          f"gru_zones={_attn_kw.get('gru_zones', 0)} miss={len(_miss)} unexp={len(_unexp)}",
          file=sys.stderr, flush=True)
else:
    print("[funcnet4_agent] *** WARNING: checkpoint NOT FOUND -> RANDOM WEIGHTS ***",
          file=sys.stderr, flush=True)
_model.to(_device)
_state = None
# DKR/PRZ-TAIL (2026-07-22 FIX DEPLOY): l'encoding tf4 appende le 2 parole mazzo-residuo/premi (Word A/B,
# NUM_WORDS 36->38) SOLO se il call-site passa dkr=True. L'agent chiamava get_encoder_input(o, DECK) SENZA
# dkr -> encoding a 36 parole vs modello a 38 -> reshape crash su OGNI mossa -> fail-soft (gioco random).
# Bug SILENZIOSO: la validazione a 2 partite controllava solo la legalita' delle mosse, non che venissero
# dalla rete. Replica ESATTA di funcnet4.evaluate.RecurrentAgent: arma dkr SOLO se num_words del modello
# combacia (un ckpt 36-parole passato qui resta a 36) + prize_tracker per-partita alimentato su ogni obs.
from funcnet4.spec import (DKR_FEAT as _DKR_FEAT, PRZ_FEAT as _PRZ_FEAT, NUM_WORDS as _NUM_WORDS,
                           LOC_FEAT as _LOC_FEAT, ENC_DENSE_DIM_TF as _ENC_DENSE_DIM_TF)
_dkr = bool(_DKR_FEAT and getattr(_model, "num_words", None) == _NUM_WORDS)
# LOC-TAIL: stessa disciplina di _dkr ma sulla LARGHEZZA della riga encoder, non sulle parole (la
# coda vive sulla parola 18, presente in ogni layout). Il ckpt e' gia' stato migrato dal loader, e
# la migrazione RIFIUTA il crop: un bundle costruito con LOC=0 su un ckpt 1145 esplode all'import
# invece di giocare con una coda a zeri (il fail-soft silenzioso che la DKR aveva pagato qui).
_loc = bool(_LOC_FEAT and getattr(_model, "enc_in_dim", 0) == _ENC_DENSE_DIM_TF)
_prize_tracker = None
if (_dkr and _PRZ_FEAT) or _loc:
    from funcnet4.deck_tracker import DeckPrizeTracker
    _prize_tracker = DeckPrizeTracker(DECK)
print(f"[funcnet4_agent] dkr={_dkr} loc={_loc} prize_tracker={'on' if _prize_tracker is not None else 'off'} "
      f"(num_words model={getattr(_model,'num_words','?')} spec={_NUM_WORDS} "
      f"enc_in model={getattr(_model,'enc_in_dim','?')} spec={_ENC_DENSE_DIM_TF})",
      file=sys.stderr, flush=True)
# SEARCH A ROLLOUT (2026-08-09). Prima di oggi il builder scriveva FUNCNET4_ESEARCH e copiava
# energy_search_deploy.py nel bundle, ma NESSUNO istanziava la search: il modulo viaggiava come peso
# morto e i bundle "con search" giocavano greedy. Qui e' il gancio che mancava.
# Il MODO decide QUANDO si cerca (env FUNCNET4_ESEARCH_MODE): "energy" = decisioni-energia,
# "endgame" = qualunque decisione ma solo a fine partita. Il trigger vive dentro la search.
_ESEARCH_ON = 1 if os.environ.get("FUNCNET4_ESEARCH", "0") not in ("", "0") else 0
_esearch = None
if _ESEARCH_ON:
    try:
        from funcnet4.energy_search_deploy import EnergySearch
        # I tensori nascono dove vive il modello: passarlo GIA' su _device (vedi il commento in
        # EnergySearch.__init__ — CPU/CUDA misti = 51 fallback e zero ricerche, in silenzio).
        _esearch = EnergySearch(_model, DECK, _dkr, _loc)
        print(f"[funcnet4_agent] esearch ARMATA modo={_esearch.mode} cand={_esearch.cand} "
              f"mondi={_esearch.worlds} trunc={_esearch.trunc} min_z={_esearch.min_z} "
              f"device={_esearch.device}", file=sys.stderr, flush=True)
    except Exception as _ex:
        # NON silenzioso: una search che non si costruisce deve essere visibile, altrimenti il
        # bundle sembra armato e gioca greedy (esattamente il difetto che sto chiudendo).
        print(f"[funcnet4_agent] !!! esearch NON costruita: {type(_ex).__name__}: {_ex}",
              file=sys.stderr, flush=True)
        _esearch = None
        _ESEARCH_ON = 0

# LETHAL OVERRIDE deterministico (fork tf3): stato per-partita/per-match.
_LETHAL_ON = int(os.environ.get("FUNCNET4_LETHAL", "2"))       # 0=off, 1=base, 2=+compose
_NO_OVERKILL = int(os.environ.get("FUNCNET4_NO_OVERKILL", "1"))
# Regola 6 (mai Fezandipiti con Flip the Script morto): 0 = off. Serve _ko_tracker alimentato
# su OGNI decisione — obs.logs e' un delta, il KO non e' leggibile dalla sola Observation.
_FEZ_MASK = int(os.environ.get("FUNCNET4_FEZ_MASK", "1"))
# Regola 3 STRICT (A/B 2026-07-16): slack del cap energia — 0 = cap secco al costo esatto,
# 1 = storico (+1 buffer ritirata/denial). Vedi _energy_cap in funcnet4/lethal_dragapult_dusknoir.
_ENERGY_SLACK = 0 if int(os.environ.get("FUNCNET4_ENERGY_STRICT", "0")) else 1
# Telemetria Regola 3 (harness A/B, default OFF -> zero costo su Kaggle): quante volte l'argmax
# pre-mask della rete verrebbe potato dal cap live e dal cap strict (probe uso slot +1).
_ENERGY_TELEM = int(os.environ.get("FUNCNET4_ENERGY_TELEM", "0"))
# LOG DI DEPLOY su stderr (Giovanni 2026-08-02): ACCESO di default. kaggle-environments cattura
# stderr per-step e lo espone in `kaggle competitions logs`; senza, un misplay in partita reale
# non e' diagnosticabile. Costo: una riga solo quando una maschera cambia davvero la scelta.
_MASK_LOG = int(os.environ.get("FUNCNET4_MASK_LOG", "1"))
ENERGY_TELEM = {"decisions": 0, "live_active": 0, "live_pruned_argmax": 0,
                "strict_active": 0, "strict_pruned_argmax": 0}
_LETHAL_MS_BUDGET = float(os.environ.get("FUNCNET4_LETHAL_MS", "120000"))
# 2026-07-18: legacy = vecchio gate detection `premi>4 && opp>2` (pre-precheck aritmetico)
_LETHAL_GATE_LEGACY = os.environ.get("FUNCNET4_LETHAL_GATE", "") == "legacy"
_ko_tracker = None
_BENCH_FEAT = __import__('os').environ.get('FUNCNET4_BENCH_FEAT', '0') not in ('', '0')
_bench_memo = {}
_lethal = None
_lethal_turn = None
_lethal_att_turn = None
_lethal_att = 0
_lethal_sig = None
_lethal_pv = None            # O3 (2026-07-20): coda della linea trovata (killer-move exec/verify)
_lethal_ms = 0.0
LAST_SCORES = None


def _enc_tensors(sv):
    de = np.asarray(sv.dense, dtype=np.float32)
    return (torch.tensor(de, device=_device).view(-1, de.shape[-1]),
            torch.tensor(np.asarray(sv.role, dtype=np.int64), device=_device))


def _dec_tensors(sv):
    dd = np.asarray(sv.dense, dtype=np.float32)
    return (torch.tensor(dd, device=_device).view(-1, dd.shape[-1]),
            torch.tensor(np.asarray(sv.plane, dtype=np.int64), device=_device))


def _lethal_pick(o, actions):
    """OVERRIDE LETHAL deterministico (identico a tf3, import da funcnet4.lethal_dragapult_dusknoir). None = nessun
    intervento. NON solleva MAI verso il chiamante; su eccezione si spegne per il resto del match."""
    global _lethal, _lethal_turn, _lethal_att_turn, _lethal_att, _lethal_sig, \
        _lethal_pv, _lethal_ms, _LETHAL_ON
    if not _LETHAL_ON:
        return None
    t0 = time.perf_counter()
    try:
        if _lethal is None:
            from funcnet4.lethal_dragapult_dusknoir import LethalFinder
            _lethal = LethalFinder(DECK, None, compose=(_LETHAL_ON >= 2))
        st = o.current
        t_now = int(st.turn)
        if _lethal_turn == t_now:                    # EXEC-MODE: linea gia' innescata
            # O3 (2026-07-20): la linea dell'ultima find guida la ri-cerca (killer-move,
            # ~lineare se ancora valida); su divergenza il finder degrada a ricerca piena
            from funcnet4.lethal_dragapult_dusknoir import pv_tail
            f = _lethal.find(o, pv=_lethal_pv)
            if f is None and _lethal_pv is not None:
                # PV stantia (shadow 07-20: 1/467): hint divergente = budget bruciato nel
                # sottoalbero sbagliato -> retry pulito = semantica pre-O3 esatta
                f = _lethal.find(o)
            if f is None:
                _lethal_turn = None
                _lethal_pv = None
                return None
            _lethal_pv = pv_tail(f["ais"])
            return int(f["first_ai"])
        if _lethal_ms > _LETHAL_MS_BUDGET:
            return None
        from cg.api import SelectType
        if int(o.select.type) != int(SelectType.MAIN):
            return None
        # gate detection (2026-07-18): precheck ARITMETICO lethal_plausible al posto del vecchio
        # `premi>4 && opp>2` cieco (ep Kaggle 86560288: doppio Mega 3+3=6 nello stesso turno
        # invisibile -> vittoria persa). Skippa anche i mid-game aritmeticamente impossibili
        # (premi<=4 irraggiungibili) dove il vecchio gate faceva girare il DFS a vuoto.
        # FUNCNET4_LETHAL_GATE=legacy ripristina il vecchio comportamento.
        if _LETHAL_GATE_LEGACY:
            if len(st.players[st.yourIndex].prize or []) > 4:
                _opp = st.players[1 - st.yourIndex]
                if sum(1 for z in (_opp.active, _opp.bench)
                       for c in (z or []) if c is not None) > 2:
                    return None
        else:
            from funcnet4.lethal_dragapult_dusknoir import lethal_plausible
            if not lethal_plausible(o):
                return None
        if _lethal_att_turn != t_now:
            _lethal_att_turn, _lethal_att, _lethal_sig = t_now, 0, None
        from funcnet4.lethal_dragapult_dusknoir import rearm_sig, should_rearm, pv_tail
        sig = rearm_sig(o)
        if _lethal_att >= 2:
            # 2026-07-16 (blind-spot ep 86275279): i RIARMI girano in modalita' MINIMAL (solo
            # passata-1 del finder, ~1/3 del costo) -> tetto 4->6 con caso peggiore per turno
            # PIU' BASSO di prima (2 full + 4 minimal). Copre le lethal nate a meta' turno
            # (es. evolve Dusknoir -> Cursed Blast + attacco = linea corta, passata-1 la trova).
            # 2026-07-20 (ep Kaggle 87104061 t9): riarmo anche sui cambi di POSIZIONE
            # (should_rearm: gust/retreat/danni/attach sull'attivo) — gained_material era cieco
            # a Boss's+RETREAT e la detection restava spenta col lethal sul tavolo.
            if _lethal_att >= 6 or not should_rearm(_lethal_sig, sig):
                return None
            _min = True
        else:
            _min = False
        _lethal_att += 1
        _lethal_sig = sig
        f = _lethal.find(o, minimal=_min)
        # verify: riconferma su un 2° mondo, PV-guidata (O3); su None retry SENZA pv
        # (= verify pre-O3 esatto: l'hint-first puo' sprecare budget se la linea non regge
        # nel 2° mondo) -> fires mai in calo rispetto a pre-O3
        if f is None:
            return None
        if _lethal.find(o, minimal=_min, pv=f["ais"]) is None and _lethal.find(o, minimal=_min) is None:
            return None
        _lethal_turn = t_now
        _lethal_pv = pv_tail(f["ais"])
        return int(f["first_ai"])
    except Exception:
        _LETHAL_ON = 0
        return None
    finally:
        _lethal_ms += (time.perf_counter() - t0) * 1000.0


def _decision_log(o, sc, idx, actions, overridden, pre_ko):
    """UNA riga per decisione su STDERR (richiesta Giovanni 2026-08-02): serve sapere cosa ha fatto
    l'agent anche quando e' andato TUTTO BENE, non solo quando cade nel fallback random. Senza,
    una partita giocata male e una giocata bene producono lo stesso identico log (vuoto), e l'unica
    diagnosi possibile e' guardare il replay a occhio.

    Campi scelti per stare nel cap di ~1024 char/step di kaggle-environments e per rispondere alle
    domande che ci siamo posti davvero sul replay 89480887:
      t     turno                       -> "a che punto della partita"
      n     opzioni disponibili         -> distingue mossa forzata da scelta vera
      a     indice azione scelta
      p1/p2 punteggio del 1o e 2o       -> il MARGINE dice se la rete era sicura o indecisa
      ovr   1 se il lethal-override ha forzato la mossa (la rete non ha deciso)
      pk    pre_ko: se la Regola 6 e' attiva in questa decisione
    Decisioni a mossa-unica non arrivano qui (l'agent le cortocircuita prima)."""
    if not _MASK_LOG:
        return
    try:
        top = sorted((float(x) for x in sc.tolist()), reverse=True)[:2]
        p1 = top[0]
        p2 = top[1] if len(top) > 1 else float("nan")
        print(f"[fn4] t={int(getattr(o.current,'turn',-1))} n={len(actions)} a={idx} "
              f"p1={p1:.3f} p2={p2:.3f} ovr={int(bool(overridden))} pk={int(bool(pre_ko))}",
              file=sys.stderr, flush=True)
    except Exception:
        pass


def _mask_log(o, tag, before, after, n_actions, n_kept, pre_ko):
    """LOG DI DEPLOY su STDERR (richiesta Giovanni 2026-08-02, PERMANENTE).

    kaggle-environments cattura stdout+stderr PER STEP e li espone in
    `kaggle competitions logs <episode_id> <agent_index>` (campo "stderr", cap ~1024 char/step),
    quindi una riga compatta qui e' l'unico modo di sapere DOPO, da una partita reale, perche'
    l'agent ha scelto una mossa. Senza, l'unica diagnosi possibile era guardare il replay a occhio
    (ed e' cosi' che il misplay di Fezandipiti del 02/08 e' rimasto invisibile per giorni).

    Si stampa SOLO quando una maschera ha davvero cambiato la scelta, piu' il caso-spia qui sotto:
    ⚠ FEZ-UNMASKED = Fezandipiti giocabile, nessun mio KO nel turno avversario (`pre_ko` falso) e
      NESSUNA maschera che lo escluda. E' esattamente la firma del difetto trovato il 02/08: la
      Regola 6 viene calcolata ma puo' perdere per precedenza (e' ULTIMA nella lista di
      `mask_dominated_main`, e a intersezione vuota vince la prima) oppure non essere mai chiamata
      (se `mask_dead_counter_targets` scatta prima e cortocircuita). Senza questa riga il sintomo
      e' indistinguibile da un errore della rete."""
    if not _MASK_LOG:
        return
    try:
        turn = int(getattr(o.current, "turn", -1))
        if before != after:
            print(f"[fn4] mask={tag} turn={turn} idx {before}->{after} keep={n_kept}/{n_actions} "
                  f"preko={int(bool(pre_ko))}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _fez_watch(o, actions, pre_ko, chosen):
    """Caso-spia: l'agent sta per GIOCARE Fezandipiti senza che l'ability possa sparare.
    Indipendente da quale maschera abbia vinto: guarda il RISULTATO, non il percorso."""
    if not _MASK_LOG or pre_ko:
        return
    try:
        # ⚠ import LOCALI: questi nomi NON esistono nel namespace dell'agent (che importa solo
        # to_observation_class da cg.api). Referenziarli globalmente sollevava NameError, che il
        # `except` qui sotto avrebbe ingoiato -> sentinella muta per sempre, cioe' il difetto che
        # questa funzione deve proprio scoprire. Verificato prima di spedire.
        from cg.api import OptionType, SelectType, AreaType
        from funcnet.encoding import get_card
        from funcnet4.lethal_dragapult_dusknoir import FEZANDIPITI_CID
        sel = o.select
        if int(sel.type) != int(SelectType.MAIN):
            return
        me = o.current.yourIndex
        for oi in actions[chosen]:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.PLAY):
                continue
            if getattr(get_card(o, AreaType.HAND, opt.index, me), "id", None) != FEZANDIPITI_CID:
                continue
            mine = o.current.players[me]
            n_play = sum(1 for z in (mine.active, mine.bench) for c in (z or []) if c is not None)
            print(f"[fn4] !! FEZ-UNMASKED turn={int(getattr(o.current,'turn',-1))} "
                  f"in_play={n_play} preko=0 -- Regola 6 NON ha impedito il play",
                  file=sys.stderr, flush=True)
            return
    except Exception:
        pass


def agent(obs_dict: dict) -> list[int]:
    global _state, LAST_SCORES, _ko_tracker
    LAST_SCORES = None
    try:
        sel = obs_dict.get("select")
        if sel is None:
            _state = None
            global _lethal_turn, _lethal_att_turn, _lethal_att, _lethal_sig, _lethal_pv
            _lethal_turn, _lethal_att_turn, _lethal_att, _lethal_sig = None, None, 0, None
            _lethal_pv = None
            if _ko_tracker is not None:
                _ko_tracker.reset()
            if _prize_tracker is not None:      # DKR/PRZ: reset per-partita (select=None = nuovo game)
                _prize_tracker.reset()
            if _esearch is not None:            # budget, contatori e orologio sono PER PARTITA
                _esearch.reset()
            return DECK
        n = len(sel["option"])
        if n == 0 or sel["maxCount"] == 0:
            return []
        o = to_observation_class(obs_dict)
        _pre_ko = None
        if _FEZ_MASK and _NO_OVERKILL:      # la Regola 6 vive dentro il ramo no-overkill
            try:
                if _ko_tracker is None:
                    from funcnet4.lethal_dragapult_dusknoir import KOTracker
                    _ko_tracker = KOTracker()
                # PRIMA di qualsiasi uscita anticipata: le decisioni a mossa-unica sono spesso
                # le promozioni post-KO, cioe' l'unico punto in cui il log del KO transita.
                _ko_tracker.observe(o)
                _pre_ko = _ko_tracker.pre_ko(o.current.yourIndex)
            except Exception:
                _pre_ko = None
        if _prize_tracker is not None:          # DKR/PRZ: osserva PRIMA dell'early-out mossa-unica
            _prize_tracker.observe(o)           # (come evaluate.py: premi/ricerche transitano anche li')
        actions = enumerate_actions(o.select)
        if len(actions) == 1:
            return actions[0]
        e = _enc_tensors(get_encoder_input(o, DECK, dkr=_dkr, loc=_loc, prize_tracker=_prize_tracker))
        # BENCH-TAIL: al DEPLOY la coda va riempita come in training, altrimenti il modello vede
        # colonne a zero su cui e' stato addestrato con valori veri (mismatch train/serve). La query
        # usa la Search API — legittima qui: il modulo lethal la usa gia' nel bundle — e gira PRIMA
        # del lethal, quindi nessun mondo e' aperto. Fail-soft: qualunque intoppo = coda a zero.
        _bench = None
        if _BENCH_FEAT:
            try:
                from funcnet4 import bench_counters
                _bench = bench_counters.values_for_actions(o, actions, DECK, None, memo=_bench_memo)
            except Exception:
                _bench = None
        d = _dec_tensors(get_decoder_input(o, actions, bench_vals=_bench))
        with torch.inference_mode():
            _, policy, new_state = _model(e[0], e[1], d[0], d[1], state=_state)
        _state = new_state
        sc = policy[0]
        idx = int(sc.argmax())
        overridden = False
        _keep_set = None            # insieme AMMESSO dalle maschere: la search non deve uscirne
        try:
            _forced = _lethal_pick(o, actions)
            if _forced is not None:
                idx, overridden = int(_forced), True
            elif _NO_OVERKILL:
                from funcnet4.lethal_dragapult_dusknoir import mask_dead_counter_targets, mask_dominated_main
                if _ENERGY_TELEM:
                    try:                          # probe A/B: solo conteggi, mai tocca idx
                        from funcnet4.lethal_dragapult_dusknoir import mask_excess_energy_attach
                        ENERGY_TELEM["decisions"] += 1
                        _kl = mask_excess_energy_attach(o, actions, slack=_ENERGY_SLACK)
                        _ks = _kl if _ENERGY_SLACK == 0 else mask_excess_energy_attach(o, actions, slack=0)
                        if _kl is not None:
                            ENERGY_TELEM["live_active"] += 1
                            if idx not in _kl:
                                ENERGY_TELEM["live_pruned_argmax"] += 1
                        if _ks is not None:
                            ENERGY_TELEM["strict_active"] += 1
                            if idx not in _ks:
                                ENERGY_TELEM["strict_pruned_argmax"] += 1
                    except Exception:
                        pass
                _tag = None
                _keep = mask_dead_counter_targets(o, actions)
                if _keep is not None:
                    _tag = "counter"
                else:
                    _keep, _tag = mask_dominated_main(o, actions, pre_ko=_pre_ko,
                                                      energy_slack=_ENERGY_SLACK)
                if _keep is not None:
                    _before = idx
                    idx = int(max(_keep, key=lambda k: float(sc[k])))
                    _keep_set = set(int(k) for k in _keep)
                    _mask_log(o, _tag, _before, idx, len(actions), len(_keep), _pre_ko)
        except Exception:
            idx = int(sc.argmax())
        # ── SEARCH A ROLLOUT ──────────────────────────────────────────────────────────────────
        # Gira DOPO le maschere e solo se il lethal NON ha forzato: una linea letale e' gia' una
        # vittoria, cercarci sopra puo' solo perderla. NB in modo "energy" la note() andrebbe
        # chiamata anche sulle decisioni a mossa unica (uscita anticipata sopra): irrilevante in
        # modo "endgame", dove il trigger guarda solo premi e turno.
        if _esearch is not None and not overridden:
            try:
                if _esearch.fires(o, o.select):
                    # candidate SOLO dentro l'insieme ammesso: -inf sugli esclusi, cosi' ne' il
                    # greedy di riferimento ne' le candidate possono essere azioni mascherate
                    _sce = [float(sc[i]) for i in range(len(actions))]
                    if _keep_set is not None:
                        for _i in range(len(_sce)):
                            if _i not in _keep_set:
                                _sce[_i] = float("-inf")
                    _p = _esearch.pick(o, actions, _sce)
                    if _p is not None:
                        idx = int(_p)
                _esearch.note(o, o.select, actions[idx])
            except Exception:
                pass    # la search e' un di piu': qualunque intoppo e' gia' contato in stats
        LAST_SCORES = {"actions": [list(a) for a in actions],
                       "scores": [float(x) for x in sc.tolist()],
                       "maxCount": int(sel["maxCount"]), "nOptions": int(n),
                       "capped": len(actions) >= 64, "override": overridden}
        _decision_log(o, sc, idx, actions, overridden, _pre_ko)
        _fez_watch(o, actions, _pre_ko, idx)
        return actions[idx]
    except Exception:
        # DIAGNOSTICA random-play (2026-07-25): il fallback su selezione random e' l'UNICO punto in cui
        # l'agent smette di usare la rete, ed era SILENZIOSO. kaggle-environments cattura lo stderr per-step
        # nei log scaricabili (`kaggle competitions logs <episode_id> <agent_index>`, campo "stderr", limite
        # ~1024 char/step) -> emettiamo un warning compatto su UNA riga con turno, n.opzioni e la coda dello
        # stacktrace, cosi' una random-play in produzione e' visibile e diagnosticabile. Zero impatto sul gioco.
        try:
            _cur = obs_dict.get("current") or {}
            _t = _cur.get("turn", "?")
        except Exception:
            _t = "?"
        sel = obs_dict.get("select") or {}
        _nopt = len(sel.get("option", []) or [])
        _exc = traceback.format_exc(limit=3).strip().replace("\n", " | ")
        print(f"[funcnet4_agent] !!! RANDOM-PLAY FALLBACK (net path raised) turn={_t} nopt={_nopt} "
              f":: {_exc[:760]}", file=sys.stderr, flush=True)
        if not sel or not sel.get("maxCount"):
            return []
        return random.sample(range(len(sel["option"])), min(sel["maxCount"], len(sel["option"])))
