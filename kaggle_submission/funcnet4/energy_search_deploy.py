r"""SEARCH A ROLLOUT SULLE SOLE DECISIONI-ENERGIA, per il bundle Kaggle. (2026-08-08)

RICHIESTA (Giovanni): "il main con la search attiva che dirama solo le top 3 della policy stando
dentro i 590 secondi, solo per le energie".

COS'E'. Sulle decisioni in cui si sceglie DOVE mettere un'energia (attacco manuale, abilita' di
Blaziken, Crispin) si aprono K mondi determinizzati, si gioca ogni candidata in ognuno fino a
`trunc` decisioni e si legge la value head sulla foglia. Vince la candidata col miglior esito
MEDIO. Tutto il resto della partita e' greedy identico a prima.

PERCHE' QUESTA E NON LA OTS BOCCIATA (-303 punti, ref 54949083). Quella cercava LINEE di turno
intere e le giudicava con la value head su foglie di fine turno con `60 Snorlax` come parola-mazzo:
input fuori distribuzione, 0,39 di rumore per foglia contro una soglia di flip di 0,03. Qui il
giudizio arriva da rollout che vanno AVANTI nel gioco (l'esito, non un'opinione), le foglie sono
stati normali e la parola-mazzo e' sempre un mazzo VERO. E' la stessa configurazione che come
maestro offline produce etichette buone e che in A/B locale sulle energie e' l'unica search del
repo che abbia mai battuto la greedy.

🔴 TRE COSE CHE SU KAGGLE SONO PEGGIORI CHE IN LABORATORIO, e vanno sapute leggendo i punteggi:

  1. L'AVVERSARIO E' IGNOTO. In laboratorio i rollout usano la rete e il mazzo VERI dell'avversario;
     qui non esistono. Si simula l'avversario con la NOSTRA rete e, come suo mazzo, un mazzo vero
     (il nostro, via `opp_deck`). La memoria dice che alla value head non importa QUALE mazzo sia
     purche' sia un mazzo vero (AUC identica alla quarta cifra, corr 0,9989) — ma il mazzo entra
     anche nella DETERMINIZZAZIONE, cioe' nelle carte che l'avversario "pesca" nei rollout. Quelle
     saranno plausibili ma sbagliate. Nessuna misura esiste di quanto costi.

  2. POTARE A 3 CANDIDATE COSTA. Misurato sulle etichette del maestro: la candidata migliore sta
     nelle prime 3 della policy solo nel 52-62% dei casi. Con `max_cand 3` nel ~40% delle decisioni
     cercate la mossa buona non e' nemmeno nel set. E' il prezzo del budget: 3 candidate x 16 mondi
     = 48 rollout, contro gli 80 della configurazione da laboratorio.

  3. L'HARDWARE KAGGLE NON E' MAI STATO MISURATO. Il fattore rispetto a casa e' ignoto (nell'OTS era
     stimato 2-4x e non fu mai verificato: e' il primo dei due sospetti per quei -303). Per questo
     il budget qui NON e' una configurazione fissa ma un REGOLATORE che si autocalibra sul costo
     osservato (EWMA) e che, se il tempo stringe, taglia i mondi e poi smette di cercare. Sforare
     l'orologio perde la partita a tavolino: il regolatore e' la parte non negoziabile.

Env (tutti opzionali, il default e' la configurazione spedita):
  FUNCNET4_ESEARCH=1        arma la search (0 = spenta, comportamento identico a prima)
  FUNCNET4_ESEARCH_BUDGET_S budget di match in secondi (default 590)
  FUNCNET4_ESEARCH_SAFETY   frazione del budget realmente spendibile (default 0.70)
  FUNCNET4_ESEARCH_CAND     candidate per decisione (default 3)
  FUNCNET4_ESEARCH_WORLDS   mondi per candidata (default 16, il regolatore puo' ridurli)
  FUNCNET4_ESEARCH_TRUNC    profondita' del rollout (default 30)
  FUNCNET4_ESEARCH_MINZ     il cancello: sovrascrive solo se z >= questo (default 2.0)
  FUNCNET4_ESEARCH_LOG      1 = stampa una riga per decisione cercata
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from bytrl.actions import enumerate_actions
from cg.api import AreaType, OptionType, SelectContext, search_end, search_step

from .encoding import get_decoder_input, get_encoder_input
from .search import determinize
from .spec import DEC_DENSE_DIM_TF, ENC_DENSE_DIM_TF, PLANE_PAD

ATTACH_OPT = int(OptionType.ATTACH)
ABILITY_OPT = int(OptionType.ABILITY)
PLAY_OPT = int(OptionType.PLAY)
HAND_AREA = int(AreaType.HAND)
MAIN_CTX = int(SelectContext.MAIN)
CRISPIN_ID = 1198               # accelera energia DAL MAZZO: apre una sequenza da seguire

# quante decisioni-energia aspettarsi in una partita: serve al regolatore per non spendere tutto
# nei primi turni. Misurato ~37 cercate/partita sul matchup di laboratorio.
EXPECTED_DECISIONS = 37.0


def _env_i(k, d):
    try:
        return int(os.environ.get(k, str(d)))
    except Exception:
        return d


def _env_f(k, d):
    try:
        return float(os.environ.get(k, str(d)))
    except Exception:
        return d


def has_attach_option(sel) -> bool:
    """Attacco MANUALE di un'energia dalla mano."""
    try:
        for op in (sel.option or []):
            if int(getattr(op, "type", -1)) == ATTACH_OPT:
                return True
    except Exception:
        pass
    return False


def _hand_card_id(o, op):
    """cardId della carta della mano a cui punta l'opzione (gli option non portano il cardId)."""
    try:
        a = getattr(op, "area", None)
        if a is not None and int(a) != HAND_AREA:
            return None
        i = getattr(op, "index", None)
        hand = o.current.players[int(o.current.yourIndex)].hand
        if i is not None and hand and 0 <= int(i) < len(hand):
            return int(getattr(hand[int(i)], "id", -1))
    except Exception:
        pass
    return None


def opens_energy_sequence(o, sel, action):
    """L'azione scelta APRE una sequenza che piazzera' energia? -> 'ability' | 'crispin' | None.

    Il piazzamento vero avviene una o due decisioni DOPO, in contesti generici: agganciare
    l'azione scatenante e' l'unico modo affidabile (`ATTACH_TO` non scatta mai, 0 su 24 partite)."""
    try:
        op = (sel.option or [])[action[0]]
    except Exception:
        return None
    t = int(getattr(op, "type", -1))
    if t == ABILITY_OPT:
        return "ability"
    if t == PLAY_OPT and _hand_card_id(o, op) == CRISPIN_ID:
        return "crispin"
    return None


class _Item:
    __slots__ = ("sv_enc", "sv_dec", "n_actions")

    def __init__(self, e, d, n):
        self.sv_enc, self.sv_dec, self.n_actions = e, d, n


class EnergySearch:
    """Un'istanza per partita. `note()` su OGNI decisione, `pick()` solo dove serve."""

    def __init__(self, model, deck, dkr, loc, opp_deck=None):
        self.model, self.deck = model, deck
        self.dkr, self.loc = dkr, loc
        # 🔴 I TENSORI DEVONO NASCERE DOVE VIVE IL MODELLO. L'agent mette la rete su CUDA quando c'e';
        # costruirli su CPU non degrada, ESPLODE ("two devices, cuda:0 and cpu") — e l'eccezione
        # finisce nel fail-soft, cioe' 51 trigger e 51 fallback con la partita giocata greedy come
        # se niente fosse. Beccato solo dall'assert `searched>0` del builder.
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = torch.device("cpu")
        # ⚠️ mazzo dell'avversario: ignoto su Kaggle. Un mazzo VERO qualsiasi e' molto meglio di un
        # riempitivo sintetico (il `60 Snorlax` dell'OTS costo' 14 punti di AUC alla value head).
        self.opp_deck = list(opp_deck or deck)
        self.budget_s = _env_f("FUNCNET4_ESEARCH_BUDGET_S", 590.0)
        self.safety = _env_f("FUNCNET4_ESEARCH_SAFETY", 0.70)
        self.cand = max(1, _env_i("FUNCNET4_ESEARCH_CAND", 3))
        self.worlds = max(2, _env_i("FUNCNET4_ESEARCH_WORLDS", 16))
        self.trunc = max(1, _env_i("FUNCNET4_ESEARCH_TRUNC", 30))
        self.min_z = _env_f("FUNCNET4_ESEARCH_MINZ", 2.0)
        self.log = _env_i("FUNCNET4_ESEARCH_LOG", 0)
        self.max_rollout = _env_i("FUNCNET4_ESEARCH_MAXROLL", 700)
        # >0 = tronca sui TURNI DI GIOCO invece che sulle decisioni (vedi tools/mk_counter_deploy.py)
        self.trunc_turns = _env_i("FUNCNET4_ESEARCH_TRUNC_TURNS", 0)
        # modo `counter`: quanti segnalini devono restare perche' si cerchi. 1 = tutti e sei.
        self.ct_min_remain = _env_i("FUNCNET4_ESEARCH_CT_MINREMAIN", 1)
        # variante TERM (12-08): cerca i piazzamenti solo dal turno N in poi. Con trunc_turns=0
        # (rollout a fine partita) gli esiti sono 0/1 VERI e il cancello z ha materia su cui
        # decidere — misurato: a 2 turni le foglie della value head differiscono di centesimi
        # e z=2 e' irraggiungibile (z_hist max 0,33). Default 0 = comportamento invariato.
        self.ct_from_turn = _env_i("FUNCNET4_ESEARCH_CT_FROM_TURN", 0)
        # copertura minima per poter sovrascrivere: sotto, si gioca greedy
        self.min_worlds = _env_i("FUNCNET4_ESEARCH_MINW", 4)
        # 12-08 (richiesta Giovanni): telemetria per-decisione su STDERR — kaggle la cattura
        # per-step, cosi' dai log si legge COSA ha fatto la search (flip/gated/agree/tagli),
        # non solo quanto e' durata. Default 0: selfplay/laboratorio non vengono spammati.
        self.log = _env_i("FUNCNET4_ESEARCH_LOG", 0)
        # ── MODO FINE-PARTITA (2026-08-09) ────────────────────────────────────────────────
        # "energy" (default) = trigger storico sulle decisioni-energia; byte-identico a prima.
        # "endgame" = si cerca SOLO quando la partita sta finendo, poche volte e a fondo.
        self.mode = (os.environ.get("FUNCNET4_ESEARCH_MODE", "energy") or "energy").strip().lower()
        # decisioni ATTESE nella partita: il regolatore ci divide il budget. 37 e' tarato
        # sulle energie; in modo `counter` le ricerche sono meno (misurato ~8 a partita nel
        # collaudo del bundle) e usare 37 lasciava 11 s a decisione, cioe' DUE mondi.
        self.expected_dec = _env_f("FUNCNET4_ESEARCH_EXPECTED",
                                   20.0 if self.mode == "counter" else EXPECTED_DECISIONS)
        self.eg_prizes = _env_i("FUNCNET4_ESEARCH_EG_PRIZES", 2)   # premi residui di UNO dei due
        self.eg_turn = _env_i("FUNCNET4_ESEARCH_EG_TURN", 12)      # oppure da questo turno in poi
        self.eg_max = max(1, _env_i("FUNCNET4_ESEARCH_EG_MAX", 5))  # ricerche MASSIME per partita
        # GUARDIA DURA: sotto questo residuo sull'episodio la search si spegne per il resto della
        # partita. Sforare il tempo = episodio PERSO, quindi qui non si tratta.
        self.hard_floor = _env_f("FUNCNET4_ESEARCH_HARD_FLOOR_S", 15.0)
        # mondi per LOTTO: piu'e piccolo, piu' fine il controllo del tempo (e piu' overhead).
        self.batch = max(2, _env_i("FUNCNET4_ESEARCH_BATCH", 25))
        self.reset()

    def reset(self):
        self.seq = None
        self.decision_no = 0
        self.spent = 0.0                 # secondi gia' bruciati nella search, in questa partita
        self.searched = 0
        self.s_per_unit = None           # EWMA: secondi per (candidata x mondo). None = mai misurato
        self.t0_game = time.perf_counter()   # orologio dell'EPISODIO, per la guardia dura
        self.hard_stop = False               # scattata la guardia: greedy per il resto della partita
        self.worlds_hist = []                # mondi EFFETTIVI per decisione cercata (per il report)
        self.z_hist = []                     # (z, mondi) dei casi in cui best != greedy
        self.t_max = 0.0                     # secondi della decisione piu' cara
        # ⚠️ dizionario NORMALE, non defaultdict: una chiave nuova alza KeyError, e dentro il
        # fail-soft di pick() diventa un fallback silenzioso. Ogni contatore va DICHIARATO qui.
        # (successo l'11-08: `thin` non dichiarato -> 10 fallback su 24 ricerche)
        self.stats = {"fires": 0, "searched": 0, "flips": 0, "gated": 0,
                      "skipped_budget": 0, "fallback": 0, "worlds_cut": 0,
                      "eg_capped": 0, "eg_offtime": 0, "thin": 0}

    # ── trigger ──────────────────────────────────────────────────────────────────────────
    def _fires_endgame(self, o) -> bool:
        """Fine partita: premi <= soglia per UNO dei due, oppure dal turno N in poi (prima delle
        due che scatta; il conteggio premi e' piu' affidabile del turno). Tetto per partita e
        guardia dura sul tempo dell'episodio."""
        try:
            if self.hard_stop:
                return False
            if (self.budget_s - (time.perf_counter() - self.t0_game)) < self.hard_floor:
                self.hard_stop = True            # da qui in poi SEMPRE greedy: non si sfora
                self.stats["eg_offtime"] += 1
                return False
            if self.searched >= self.eg_max:
                self.stats["eg_capped"] += 1
                return False
            st = o.current
            for p in (st.players or []):
                if len(getattr(p, "prize", None) or []) <= self.eg_prizes:
                    return True
            return int(getattr(st, "turn", 0)) >= self.eg_turn
        except Exception:
            return False

    def fires(self, o, sel) -> bool:
        """Questa decisione va cercata? Aggiorna anche la finestra di sequenza."""
        if self.mode == "endgame":
            return self._fires_endgame(o)
        if self.mode == "counter":
            # SOLO i piazzamenti di segnalini danno. Il regolatore di budget resta quello:
            # se il tempo scarseggia _worlds_allowed() ritorna 0 e si gioca greedy.
            try:
                if self.ct_from_turn > 0 and int(getattr(o.current, "turn", 0) or 0) < self.ct_from_turn:
                    return False               # TERM: prima del turno N si gioca greedy
                return int(getattr(sel, "remainDamageCounter", 0) or 0) >= self.ct_min_remain
            except Exception:
                return False
        try:
            if int(getattr(sel, "type", -1)) == MAIN_CTX:
                self.seq = None                 # tornati a MAIN: la sequenza precedente e' chiusa
            if self.seq is not None:
                return True
            return has_attach_option(sel)
        except Exception:
            return False

    def note(self, o, sel, action):
        """Registra se l'azione APPENA scelta apre una sequenza. Va chiamata su OGNI decisione,
        comprese quelle a una sola azione: l'abilita' forzata apre la sequenza proprio li'."""
        try:
            s = opens_energy_sequence(o, sel, action)
            if s is not None:
                self.seq = s
        except Exception:
            pass

    # ── regolatore di budget ─────────────────────────────────────────────────────────────
    def _worlds_allowed(self, n_cand):
        """Quanti mondi ci si puo' permettere ADESSO. 0 = non cercare.

        Il budget si ripartisce sulle decisioni-energia ANCORA ATTESE, non a rate fisse: il costo
        per decisione CALA lungo la partita (piu' rollout finiscono prima del taglio), quindi
        spendere il residuo e' corretto e spendere una quota fissa no."""
        left = self.budget_s * self.safety - self.spent
        if left <= 1.0:
            return 0
        atteso = float(self.eg_max) if self.mode == "endgame" else self.expected_dec
        expected_left = max(1.0, atteso - self.searched)
        allow_s = left / expected_left
        if self.s_per_unit is None:
            return self.worlds                   # prima decisione: si misura, poi ci si calibra
        units = allow_s / max(self.s_per_unit, 1e-6)
        w = int(units // max(n_cand, 1))
        return max(0, min(self.worlds, w))

    # ── la search ────────────────────────────────────────────────────────────────────────
    def pick(self, o, actions, sc):
        """Ritorna l'indice dell'azione scelta dalla search, o None per lasciar decidere la greedy."""
        self.decision_no += 1
        self.stats["fires"] += 1
        greedy = int(np.argmax([float(sc[i]) for i in range(len(actions))]))
        _lt = _lr = 0
        if self.log:
            try:
                _lt = int(o.current.turn)
                _lr = int(getattr(o.select, "remainDamageCounter", 0) or 0)
            except Exception:
                pass
        n_cand = min(self.cand, len(actions))
        w = self._worlds_allowed(n_cand)
        if w < 2:
            self.stats["skipped_budget"] += 1
            if self.log:
                print(f"[es] t={_lt} rem={_lr} SKIP-budget", file=sys.stderr, flush=True)
            return None
        t0 = time.perf_counter()
        try:
            me = int(o.current.yourIndex)
            if len(actions) <= n_cand:
                cand = list(range(len(actions)))
            else:
                order = sorted(range(len(actions)), key=lambda i: -float(sc[i]))
                cand = order[:n_cand]
                if greedy not in cand:
                    cand[-1] = greedy
            if w < self.worlds:
                self.stats["worlds_cut"] += 1
                if self.log:
                    print(f"[es] t={_lt} rem={_lr} taglio-budget w={w}/{self.worlds}",
                          file=sys.stderr, flush=True)
            base = 104_729 * self.decision_no + 7919
            # 🔴 A LOTTI, NON IN UN COLPO SOLO. Sparare w mondi tutti insieme rende il tempo della
            # decisione IGNOTO finche' non e' finita: misurato 192 s per la PRIMA decisione di ogni
            # partita (s_per_unit ancora None -> il regolatore concede il tetto pieno). Su una CPU
            # piu' lenta della nostra quella singola decisione si mangia l'episodio, e la guardia
            # dura non puo' intervenire perche' controlla PRIMA di partire, non durante.
            # A lotti il tempo e' limitato per costruzione: si controlla l'orologio fra un lotto e
            # l'altro e si smette. E' anche il modo di autoregolarsi su hardware ignoto.
            atteso = float(self.eg_max) if self.mode == "endgame" else self.expected_dec
            resto = self.budget_s * self.safety - self.spent
            per_dec = max(1.0, resto / max(1.0, atteso - self.searched))
            agg, fatti = {}, 0
            while fatti < w:
                lotto = min(self.batch, w - fatti)
                states, owner = [], []
                for k in range(fatti, fatti + lotto):
                    # UN MONDO DIVERSO PER ROLLOUT, condiviso dalle candidate dello stesso mondo:
                    # confronto APPAIATO fra bersagli, media su mondi indipendenti.
                    ss0 = determinize(o, me, self.deck, self.opp_deck,
                                      rng=np.random.default_rng(base + 1009 * k))
                    for c in cand:
                        try:
                            states.append(search_step(ss0.searchId, actions[c]))
                            owner.append(c)
                        except Exception:
                            pass
                if states:
                    for c, r in zip(owner, self._rollouts(states, me)):
                        if r is not None:
                            agg.setdefault(c, []).append(r)
                fatti += lotto
                try:
                    search_end()          # chiude i mondi del lotto prima di aprire i prossimi
                except Exception:
                    pass
                if (time.perf_counter() - t0) >= per_dec:
                    break                 # tempo della decisione esaurito: si decide con quel che c'e'
            w = fatti                     # mondi VERAMENTE girati (per il report e per s_per_unit)
            if not agg:
                raise RuntimeError("nessun mondo")
            if greedy not in agg or len(agg) < 2:
                raise RuntimeError("copertura insufficiente")
            best = max(agg, key=lambda k: float(np.mean(agg[k])))
            self.searched += 1
            self.stats["searched"] += 1
            if best == greedy:
                if self.log:
                    print(f"[es] t={_lt} rem={_lr} w={w} agree a={greedy}",
                          file=sys.stderr, flush=True)
                return None
            # ⚠️ COPERTURA MINIMA. Con 2-3 mondi la stima e' rumore e il cancello sotto non
            # protegge (varianza campionaria nulla -> z infinito). Meglio giocare greedy.
            if min(len(agg[best]), len(agg[greedy])) < self.min_worlds:
                self.stats["thin"] += 1
                if self.log:
                    print(f"[es] t={_lt} rem={_lr} w={w} thin", file=sys.stderr, flush=True)
                return None
            # IL CANCELLO. Senza, la search sovrascrive circa meta' delle decisioni mentre solo
            # l'1-3% ha un margine reale, e PERDE (misurato -12,7pp su hetzner, -5,0 in locale).
            if self.min_z > 0:
                mb, mg = float(np.mean(agg[best])), float(np.mean(agg[greedy]))
                nb, ng = len(agg[best]), len(agg[greedy])
                vb = float(np.var(agg[best], ddof=1)) if nb > 1 else 0.0
                vg = float(np.var(agg[greedy], ddof=1)) if ng > 1 else 0.0
                se = float(np.sqrt(vb / max(nb, 1) + vg / max(ng, 1)))
                # ⚠️ PAVIMENTO: due rollout concordi danno varianza 0 e z infinito. L'errore
                # di Bernoulli non collassa mai con n piccolo, quindi si prende il maggiore.
                se_b = float(np.sqrt(mb * (1.0 - mb) / max(nb, 1) + mg * (1.0 - mg) / max(ng, 1)))
                se = max(se, se_b, 1e-6)
                z = (mb - mg) / se
                self.z_hist.append((float(z), int(w)))   # per capire se il cancello e' il collo
                if z < self.min_z:
                    self.stats["gated"] += 1
                    if self.log:
                        print(f"[es] t={_lt} rem={_lr} w={w} gated z={z:.2f} "
                              f"mb={mb:.2f} mg={mg:.2f}", file=sys.stderr, flush=True)
                    return None
                if self.log:
                    print(f"[es] t={_lt} rem={_lr} w={w} FLIP a={int(best)} da={greedy} "
                          f"z={z:.2f} mb={mb:.2f} mg={mg:.2f}", file=sys.stderr, flush=True)
            self.stats["flips"] += 1
            return int(best)
        except Exception as ex:
            # fail-soft: qualunque intoppo = greedy. Ma CONTATO, perche' una search che eccepisce
            # sempre degrada in silenzio a "nessun effetto" e sembra un esperimento riuscito.
            self.stats["fallback"] += 1
            # ⚠️ il MOTIVO va conservato: un fail-soft muto degrada in silenzio a "nessun
            # effetto" e sembra un esperimento riuscito invece che un guasto.
            self.last_err = "%s: %s" % (type(ex).__name__, ex)
            if self.log:
                print(f"[esearch] fallback {type(ex).__name__}: {ex}", flush=True)
            return None
        finally:
            try:
                search_end()
            except Exception:
                pass
            el = time.perf_counter() - t0
            self.spent += el
            self.worlds_hist.append(int(w))      # mondi EFFETTIVI dopo il taglio del budget
            self.t_max = max(self.t_max, el)
            units = max(1, n_cand * max(w, 1))
            u = el / units
            self.s_per_unit = u if self.s_per_unit is None else 0.7 * self.s_per_unit + 0.3 * u
            if self.log:
                print(f"[esearch] d{self.decision_no} cand={n_cand} mondi={w} "
                      f"{el:.1f}s speso={self.spent:.0f}/{self.budget_s * self.safety:.0f} "
                      f"s/unit={self.s_per_unit * 1000:.0f}ms", flush=True)

    # ── motore dei rollout ───────────────────────────────────────────────────────────────
    @staticmethod
    def _build(samples, device="cpu"):
        """I 4 tensori del forward, allineati al reshape del modello. Copia VERBATIM del ramo
        "dense pieno" di `funcnet4.losses._build`.

        Perche' copiato e non importato: `losses.py` tira dentro `bytrl.losses` e la matematica RL,
        cioe' 1.300 righe e due dipendenze in piu' dentro un bundle dove ogni import in piu' e' una
        possibilita' di rompersi all'avvio. Il ramo che serve al deploy sono queste venti righe, e
        non tocca le CODE: arrivano gia' dentro `sv_enc.dense` alla larghezza decisa dai flag env
        (1109 senza LOC / 1145 con). Se un giorno `_build` cambia, questo va riallineato a mano."""
        a_max = max(s.n_actions for s in samples)
        dense_e, role_e, dense_d, plane_d = [], [], [], []
        for s in samples:
            dense_e.extend(s.sv_enc.dense)
            role_e.extend(s.sv_enc.role)
            dense_d.extend(s.sv_dec.dense)
            plane_d.extend(s.sv_dec.plane)
            for _ in range(a_max - s.n_actions):
                dense_d.append(np.zeros(DEC_DENSE_DIM_TF, dtype=np.float32))
                plane_d.append(PLANE_PAD)
        enc = (torch.tensor(np.asarray(dense_e, dtype=np.float32),
                            device=device).view(-1, ENC_DENSE_DIM_TF),
               torch.tensor(np.asarray(role_e, dtype=np.int64), device=device))
        dec = (torch.tensor(np.asarray(dense_d, dtype=np.float32),
                            device=device).view(-1, DEC_DENSE_DIM_TF),
               torch.tensor(np.asarray(plane_d, dtype=np.int64), device=device))
        return enc, dec

    def _batch_policy(self, items):
        """[(obs, actions, mover)] -> [indice scelto]. Un solo forward per tutto il lotto.

        Entrambi i lati sono simulati con la NOSTRA rete: su Kaggle quella dell'avversario non
        esiste. La parola-mazzo segue il mover, cosi' nessuna foglia riceve un mazzo finto."""
        out = [0] * len(items)
        groups = {}
        for i, (_o, _a, mv) in enumerate(items):
            groups.setdefault(mv, []).append(i)
        for mv, idxs in groups.items():
            deck = self.deck if mv == self._me else self.opp_deck
            shims = []
            for i in idxs:
                oo, aa, _ = items[i]
                shims.append(_Item(get_encoder_input(oo, deck, dkr=self.dkr, loc=self.loc),
                                   get_decoder_input(oo, aa), len(aa)))
            enc, dec = self._build(shims, self.device)
            with torch.inference_mode():
                _v, pol, _s = self.model(enc[0], enc[1], dec[0], dec[1])
            for b, i in enumerate(idxs):
                out[i] = int(pol[b, :shims[b].n_actions].argmax())
        return out

    def _leaf_values(self, obs_list, me):
        """V(s) sulle foglie, riportata in [0,1] dal punto di vista di `me`.

        Due dettagli che costano un bug: il modello ritorna V per il GIOCATORE DI TURNO (va girata
        di segno sulle foglie avversarie), e si usa il SOLO percorso encoder — costruire il decoder
        su una foglia senza opzioni legali esplode con IndexError e degraderebbe tutto a greedy in
        silenzio. Una foglia gia' terminale non si stima: si legge il risultato vero."""
        out = []
        for o in obs_list:
            st = o.current
            if int(getattr(st, "result", -1)) >= 0:
                r = int(st.result)
                out.append(0.5 if r == 2 else (1.0 if r == me else 0.0))
                continue
            mine = int(st.yourIndex) == me
            e = get_encoder_input(o, self.deck if mine else self.opp_deck,
                                  dkr=self.dkr, loc=self.loc)
            de = torch.tensor(np.asarray(e.dense, np.float32),
                              device=self.device).view(-1, ENC_DENSE_DIM_TF)
            re = torch.tensor(np.asarray(e.role, np.int64), device=self.device)
            with torch.inference_mode():
                eo, summ = self.model.encode(de, re)
                # GRU-ZONES: input del core via recur_input (default = summ invariato)
                bel = (self.model.recur(
                           self.model.recur_input(eo, summ).unsqueeze(0), None)[0].squeeze(0)
                       if self.model.recurrent != "none" else None)
                v = self.model.value_from(eo, bel)
            x = float(v.reshape(-1)[0])
            out.append(0.5 * ((x if mine else -x) + 1.0))
        return out

    def _rollouts(self, states, me):
        """Avanza tutti i rollout IN PARALLELO fondendo i forward; al taglio legge la value head.

        Il troncamento non e' solo un risparmio: misurato, riproduce il verdetto del rollout
        completo a 3-5x meno costo e con errore standard PIU STRETTO (la coda della partita e'
        varianza che non riguarda la domanda)."""
        self._me = me
        live = list(range(len(states)))
        ss = list(states)
        res = [None] * len(states)
        # turno di partenza di OGNI rollout: il taglio a turni e' relativo a questo
        t0t = None
        if self.trunc_turns > 0:
            try:
                t0t = [int(x.observation.current.turn) for x in states]
            except Exception:
                t0t = None
        for d in range(self.max_rollout):
            # ⚠️ TAGLIO EQUO. Contare le DECISIONI premia le mosse senza effetto: chi risolve
            # un effetto multi-passo paga quei passi col proprio budget e vede meno partita.
            if t0t is not None and live:
                maturi = [i for i in live
                          if int(ss[i].observation.current.turn) - t0t[i] >= self.trunc_turns]
                if maturi:
                    for i, x in zip(maturi, self._leaf_values(
                            [ss[i].observation for i in maturi], me)):
                        res[i] = x
                    fermi = set(maturi)
                    live = [i for i in live if i not in fermi]
                    if not live:
                        break
            if t0t is None and d >= self.trunc and live:
                for i, x in zip(live, self._leaf_values([ss[i].observation for i in live], me)):
                    res[i] = x
                return res
            if not live:
                break
            pend, meta, nxt = [], [], {}
            for i in live:
                st = ss[i].observation.current
                if int(getattr(st, "result", -1)) >= 0:
                    r = int(st.result)
                    res[i] = 0.5 if r == 2 else (1.0 if r == me else 0.0)
                    nxt[i] = None
                    continue
                sel = ss[i].observation.select
                if sel is None or getattr(sel, "maxCount", 0) == 0 or len(sel.option) == 0:
                    nxt[i] = []
                    continue
                acts = enumerate_actions(sel)
                if not acts:
                    nxt[i] = []
                elif len(acts) == 1:
                    nxt[i] = acts[0]
                else:
                    pend.append((ss[i].observation, acts, int(st.yourIndex)))
                    meta.append((i, acts))
            if pend:
                for (i, acts), k in zip(meta, self._batch_policy(pend)):
                    nxt[i] = acts[k]
            still = []
            for i in live:
                if nxt[i] is None:
                    continue
                try:
                    ss[i] = search_step(ss[i].searchId, nxt[i])
                    still.append(i)
                except Exception:
                    res[i] = None
            live = still
        return res
