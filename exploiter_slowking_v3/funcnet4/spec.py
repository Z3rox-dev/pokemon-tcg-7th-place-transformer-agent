"""funcnet4 (text-first) dimensions. Re-exports funcnet.spec's base dims UNCHANGED (shared vocab/feature
layout) and ADDS the text-first-only layout: per-word ROLE tags, per-action PLANE tags, the SET-group
scalar slots, and the wider decoder dense (id-bag deleted, card/attack TEXT appended).

The base func/atom/text feature layout is identical to funcnet.spec — we reuse the SAME precomputed tables
and the SAME _fill_* extractors; text-first only changes how those features are READ (shared MLP, no id-bag)
and adds the role/plane tags that the deleted id-bags used to carry implicitly."""
from funcnet.spec import *                       # noqa: F401,F403  base dims (card_count, ENC_DENSE_DIM, ...)
from funcnet.spec import (                        # explicit (the names used below; also documents the deps)
    card_count, decoder_main_feature, decoder_card_offset, decoder_size, EMB_DIM, DEC_DENSE_DIM,
)

# (a) encoder per-word ROLE ids -> a small nn.Embedding(N_ROLE, d_model) added to each word vector.
# Restores the own-vs-opp + word-type signal the deleted id-bags carried (own had id indices, opp did not).
ROLE_PAD        = 0
ROLE_OWN_ACTIVE = 1
ROLE_OWN_BENCH  = 2
ROLE_OPP_ACTIVE = 3
ROLE_OPP_BENCH  = 4
ROLE_OWN_PLAYER = 5
ROLE_OPP_PLAYER = 6
ROLE_HAND       = 7
ROLE_DECK       = 8
ROLE_DISCARD    = 9   # (reserved; discard rides inside the player word today)
ROLE_STADIUM    = 10
ROLE_TURN       = 11
# funcnet4 PUNTO-2 (2026-07-12): TOKEN INDIVIDUALI per le carte in MANO. In funcnet3 la mano
# e' UNA parola (somma DeepSets, ROLE_HAND): i fatti congiuntivi ("ho A E B E il pezzo che
# manca") e le relazioni mano<->board sono processabili solo de-mixando una somma. Qui ogni
# carta in mano (fino a N_HAND_TOKENS, ordinate per card-id = rappresentazione canonica
# permutation-invariant) ha la SUA parola encoder (atom+text della carta, ROLE_HAND_CARD);
# la parola-somma ROLE_HAND resta INVARIATA (superset: copre anche l'overflow oltre N e
# conserva count/multiplicity). Slot vuoti = riga zero (stesso pattern dei bench vuoti).
ROLE_HAND_CARD  = 12
N_ROLE          = 13
import os as _os
# 2026-07-13: DEFAULT alzato a 12 (era 10) su richiesta utente — i PROSSIMI agent funcnet4 si allenano
# con 12 slot carta-in-mano. Override via env FUNCNET_N_HAND_TOKENS: serve per PINNARE un run gia' vivo
# al suo valore (mescolare conteggi-parola diversi nello stesso farm rompe il codec). NB: NUM_WORDS entra
# nel cfg_hash -> cambiarlo = arch nuovo, incompatibile con i ckpt/traiettorie del run a 34 parole.
N_HAND_TOKENS   = int(_os.environ.get("FUNCNET_N_HAND_TOKENS", "12"))
NUM_WORDS       = 24 + N_HAND_TOKENS   # 36 di default (24 base + 12 token-mano)

# (b) SET words (hand/deck/stadium/discard) reuse two func slots for scalars; the DeepSets-summed views
# (atom+text) live in the EXISTING VIEW slots [V_ATOM_OFF:] of the same dense row.
SET_COUNT_OFF = 0    # number of cards in the group / 60   (func slot 0, repurposed for SET words only)
SET_MULT_OFF  = 1    # fraction of distinct cards held in >=2 copies (consistency signal) (func slot 1)

# (c) decoder per-action PLANE ids -> a small nn.Embedding(N_PLANE, d_model) added to each action vector.
# Mirrors encoding.get_decoder_input's option layout: special tokens keep their sparse-special index (1..13);
# then ATTACK; then the 8 main-feature planes; then the per-SelectContext card planes.
PLANE_PAD       = 0   # special tokens use ids 1..13 (END=1,YES=2,NO=3,SPECIAL_CONDITION 4..8, NUMBER 9..13)
PLANE_ATTACK    = 14
PLANE_MAIN_OFF  = 15                                  # main feature_index 0..7  -> planes 15..22
PLANE_CTX_OFF   = PLANE_MAIN_OFF + decoder_main_feature   # = 23; SelectContext 0..48 -> planes 23..71
_N_CTX = (decoder_size - decoder_card_offset) // card_count   # = 1 + 8 + 48 = 57 card planes
N_PLANE = PLANE_CTX_OFF + _N_CTX + 1                  # generous upper bound (covers all context ids)

# (d) decoder per-action TEXT embedding slice appended after the existing 283-d decoder dense.
DEC_TEXT_OFF     = DEC_DENSE_DIM                      # 283

# (e) NRG-TAIL (2026-07-17): coda tf4-only di 3 scalari DERIVATI sul costo-energia effettivo, scritti SOLO
# sulle azioni ATTACH di una carta-ENERGIA (funcnet4/energy_mirror.py = mirror parziale dichiarato di
# InsufficientEnergyCount, tassa Nighttime Mine inclusa). Sostituisce-come-informazione il cap secco
# Regola-3 (mask al deploy = -6pp, memoria energy-strict-cap-deploy-ab): la rete VEDE il conto, la scelta
# resta sua. Coda DOPO il testo -> funcnet/funcnet3 intoccati (layout condiviso invariato); il prefisso
# wire df [0:DEC_ATOM_OFF] non si muove, la coda viaggia nella chiave codec "dg" (farm CODEC_VERSION 6).
# FLAG: FUNCNET4_NRG_FEAT=1 arma la SCRITTURA dei valori (default 0 = colonne a zero -> forward
# bit-identico a un ckpt pre-tail pad-ato con colonne zero). Il flag entra nel cfg_hash (truthy-only,
# farm.cfg_hash): un farm misto ON/OFF si rifiuta da solo invece di mescolare encoding diversi.
DEC_NRG_OFF  = DEC_TEXT_OFF + EMB_DIM                 # 1051 (subito dopo il testo)
DG_MISS_MIN  = 0   # min sui suoi attacchi delle energie ANCORA mancanti DOPO questo attach / 4
DG_MISS_MAX  = 1   # max (= mancanti all'attacco piu' caro) DOPO questo attach / 4; 0 = carico anche per il piu' caro
DG_USEFUL    = 2   # 1 se l'attach RIDUCE il mancante di almeno un attacco (0 = energia inutile/tipo sbagliato)
DEC_NRG_DIM  = 3
assert DG_USEFUL + 1 == DEC_NRG_DIM
NRG_FEAT = _os.environ.get("FUNCNET4_NRG_FEAT", "0") not in ("", "0")

# ---------------------------------------------------------------- CNT-TAIL (segnalini, 2026-07-18)
# Coda per-azione sui PICK di piazzamento segnalini (SelectContext DAMAGE_COUNTER=13 /
# DAMAGE_COUNTER_ANY=14, bersaglio AVVERSARIO): l'aritmetica a soglia che la rete non consolida
# dal float hp/400 (misurato ep 86605549: Munki@10 ignorato al pick 5, margine appreso +-0.004 =
# livello-rumore, oscilla fra ckpt vicini). Stessa Regola-3 della NRG-tail: la rete VEDE il conto,
# la scelta resta sua. Fonti: hp bersaglio (obs), select.remainDamageCounter (budget engine-exact),
# prize_canon (Mega=3/ex=2/1). FLAG: FUNCNET4_CNT_FEAT=1 arma la SCRITTURA (default 0 = colonne a
# zero -> forward bit-identico a un ckpt pre-tail pad-ato). Truthy-only nel cfg_hash (farm.cfg_hash).
DEC_CNT_OFF   = DEC_NRG_OFF + DEC_NRG_DIM             # 1054 (subito dopo la coda NRG)
DH_CNT_TO_KO  = 0   # ceil(hp/10)/30: segnalini mancanti al KO del bersaglio
DH_KILL_PICK  = 1   # 1 se QUESTO pick lo uccide (ceil(hp/10)==1)
DH_KILL_BUDGET = 2  # 1 se uccidibile col budget RIMANENTE (remainDamageCounter >= ceil(hp/10))
DH_PRIZE      = 3   # premi che cede al KO / 3 (prize_canon: megaEx=3, ex=2, altrimenti 1)
DEC_CNT_DIM   = 4
assert DH_PRIZE + 1 == DEC_CNT_DIM
CNT_FEAT = _os.environ.get("FUNCNET4_CNT_FEAT", "0") not in ("", "0")

# ---------------------------------------------------------------- UNLOCK-TAIL (scelta-energia, 2026-07-20)
# Coda per-azione sulle righe carta-ENERGIA mie nelle select di recupero/ricerca (Night Stretcher,
# pick post-Crispin, ecc.): "quale energia mi serve?". La riga porta COS'E' la carta (text emb),
# non A COSA SERVE -> la rete deve derivare da sola costo-attacco x danno, l'aritmetica a soglia che
# NON consolida (misurato: probe energy_choice 07-20 su 95 replay = 38/71 scelte-che-contano
# SUBOTTIME = 54%, di cui 13x Dragapult che prende {R}(+0) invece di {P}(+130 = Phantom Dive) +
# ~7x rifiuto {D}-Munkidori). La NRG-tail copre solo l'ATTACH, non il recupero -> qui e' SCOPERTO.
# VALORE (non conteggio): per l'energia E, unlock(E) = max sui MIEI mon in gioco del miglior danno
# RISOLTO fra gli attacchi che diventano pagabili CON E (mirror engine-exact attack_missing_for_mon,
# extra_energy_cid=E; deb x2 / res -30) + valore-ability ({D} sblocca Adrena-Brain di Munkidori).
# base = stesso senza E. GENERICO: niente cid-energia hardcodati (i cid del mazzo escono dall'obs).
# Stessa Regola-3: la rete VEDE il conto (unlock/gain/kills), la scelta resta sua. FLAG:
# FUNCNET4_UNLOCK_FEAT=1 arma la SCRITTURA (default 0 = colonne a zero -> forward bit-identico a un
# ckpt pre-tail pad-ato). Truthy-only nel cfg_hash (farm.cfg_hash). Wire: chiave "dj" (CODEC 10).
DEC_UNLOCK_OFF   = DEC_CNT_OFF + DEC_CNT_DIM          # 1058 (DCX rimossa; subito dopo la coda CNT)
DJ_UNLOCK_DMG    = 0   # miglior danno RISOLTO sbloccato da QUESTA energia \ 400
DJ_UNLOCK_GAIN   = 1   # (unlock - base) \ 400: il GUADAGNO vero -> la colonna "prendi me"
DJ_UNLOCK_KILLS  = 2   # 1 se l'attacco sbloccato uccide l'attivo avversario (danno risolto >= suo hp)
DEC_UNLOCK_DIM   = 3
assert DJ_UNLOCK_KILLS + 1 == DEC_UNLOCK_DIM
UNLOCK_FEAT = _os.environ.get("FUNCNET4_UNLOCK_FEAT", "0") not in ("", "0")

# ---------------------------------------------------------------- TGT-TAIL (bersaglio, 2026-08-02)
# BUCO MISURATO (tools/duplicate_action_rows_probe.py, 100 partite col ckpt vivo step 780): per
# un'opzione che punta a una carta MIA la riga-azione riceve SOLO atom+text STATICI della carta
# (encoding.py, ramo CARD, branch non-avversario) — nessun hp, nessuna energia attaccata, nessuno
# slot. Il ramo AVVERSARIO riceve in piu' _fill_target_dense: l'asimmetria e' un residuo del fix
# anti-leak, non una scelta. Conseguenza: due copie della stessa carta in gioco danno righe
# IDENTICHE BIT-PER-BIT anche con stato vivo diverso.
#   9,6% delle decisioni contiene una COLLISIONE VERA (due opzioni che il gioco distingue e la rete
#   no), con PAREGGIO DI SCORE ESATTO in 2207/2207 gruppi. Non e' "sceglie male": non puo' scegliere
#   (softmax uniforme -> monetina). E il gradiente non puo' separarle: quella decisione e'
#   non-apprendibile per costruzione, e meta' delle volte la mossa sbagliata incassa il credito.
# Caso di innesco (Giovanni, replay 89549438 step 204): l'ability "Seething Spirit" di Blaziken ex
# attacca la seconda energia buio al Munkidori che ce l'ha GIA' invece che all'altro (che
# accenderebbe una seconda Adrena-Brain). Misurato sul percorso select: 85/85 righe identiche.
# COSA DISTINGUE le opzioni collidenti (400 collisioni campionate): solo ENERGIE 53,0% | hp+energie
# 13,5% | solo hp 9,0% | posizione/tool/turno 24,5%. Da cui il contenuto della coda.
#
# La coda descrive il POKEMON IN GIOCO associato all'azione (bersaglio di ATTACH/EVOLVE, detentore
# della carta scelta in una select, sorgente di un'ABILITY, attivo che si ritira). Gli hp/maxHp/ex/
# ritirata NON stanno qui: arrivano dalle colonne DEC_TGT_* gia' esistenti (slot [0:10], dentro il
# prefisso "df" che viaggia gia' verbatim sul filo), che il flag fa scrivere anche sul ramo MIO.
# Stessa Regola-3 di NRG/CNT/UNLOCK: la rete VEDE i fatti, la scelta resta sua — niente maschere,
# niente regole per-carta ("Munkidori con {D} = ability accesa" e' derivabile da identita' + tipi).
# FLAG: FUNCNET4_TGT_FEAT=1 arma la SCRITTURA (default 0 = colonne a zero E ramo MIO invariato ->
# forward bit-identico a un ckpt pre-coda pad-ato con colonne zero). Truthy-only nel cfg_hash
# (farm.cfg_hash). Wire: chiave "dk" (CODEC 11).
DEC_TGT_OFF   = DEC_UNLOCK_OFF + DEC_UNLOCK_DIM       # 1061 (subito dopo la coda UNLOCK)
TG_NRG_N      = 0   # energie attaccate / 8            (stessa scala di D_ENERGY_COUNT dell'encoder)
TG_NRG_T0     = 1   # TG_NRG_T0 + t = conteggio energie di tipo t (0..8) / 4  (scala D_ENERGY_TYPES)
TG_N_TYPES    = 9   # quanti tipi di energia (0..8) la coda distingue: identico all'encoder
TG_IS_ACTIVE  = 10  # 1.0 se il bersaglio e' l'ATTIVO, 0.0 se in panca
TG_N_TOOLS    = 11  # tool attaccati / 2
TG_APPEARED   = 12  # 1.0 se e' entrato in gioco QUESTO turno (non puo' ancora evolvere/ritirarsi)
TG_HP         = 13  # hp correnti / 400
TG_MAXHP      = 14  # hp massimi / 400
DEC_TGT_DIM   = 15
assert TG_NRG_T0 + TG_N_TYPES == TG_IS_ACTIVE and TG_MAXHP + 1 == DEC_TGT_DIM
# TUTTI gli slot ACCUMULANO (+=), come _fill_card_views: un'azione MULTI-carta (Ultra Ball scarta 2,
# gli effetti che toccano piu' Pokemon) e' una somma DeepSets sui bersagli, non "vince l'ultimo".
# Con l'assegnazione due azioni che finivano sullo stesso ultimo bersaglio restavano IDENTICHE anche
# con i primi bersagli diversi (misurato: 16 gruppi residui, tutti cosi'). Per questo hp/maxHp sono
# QUI e non solo nelle DEC_TGT_* del prefisso: quelle usano l'assegnazione (semantica storica di
# ATTACH/EVOLVE, che ha un bersaglio solo) e non si possono cambiare senza toccare valori gia'
# allenati. Su un'azione a bersaglio singolo += e = coincidono (la riga parte da zero).
TGT_FEAT = _os.environ.get("FUNCNET4_TGT_FEAT", "0") not in ("", "0")

# ---------------------------------------------------------------- BENCH-TAIL (segnalini in panca, 2026-08-03)
# Coda per-azione sulle righe ATTACK di attacchi che piazzano segnalini sulla PANCA avversaria
# (Phantom Dive di Dragapult ex: 200 all'attivo + 6 segnalini in panca a piacere). La fetta
# "combattimento risolto" (_fill_attack_resolved) e' calcolata SOLO contro l'ATTIVO avversario:
# i 6 segnalini — cioe' 60 danni piazzabili, spesso un premio — non stanno su nessuna colonna.
#
# MISURATO (tools/phantom_bench_probe.py, ckpt vivo, 4 avversari, solo dati del motore):
#   avversario            PD giocati   bloccato   >=1 KO in panca   solo setup
#   mirror dragapult          42         2,4%         80,5%           17,1%
#   alakazam rb5th            26         3,8%         48,0%           44,0%
#   cornerstone ogerpon       17         5,9%         25,0%           68,8%
#   mega lucario              25         0,0%         24,0%           48,0%
# Il reperto non e' la media: e' la VARIANZA. Il valore dei segnalini oscilla 24%->80% secondo
# gli hp della panca che ha davanti, e la rete non ha nessun numero che la segua — non e' un
# bias costante che possa assorbire. E in 3 matchup su 4 il "nessun KO ma lo lascio in fin di
# vita" e' PIU' frequente del KO: per questo lo scalare di setup c'e'.
#
# FONTE = IL MOTORE, non uno specchio. I valori NON sono calcolati riscrivendo le regole delle
# carte: funcnet4/bench_counters.py apre un mondo, applica l'attacco e legge COSA IL MOTORE
# OFFRE. Se Battle Cage / Rabsca / Shaymin / Mist Energy bloccano, il motore semplicemente non
# offre il piazzamento e il danno piazzabile e' 0 — senza che noi si debba sapere che esistono.
# (38 mazzi meta su 132 portano almeno un bloccante: uno scalare ingenuo "+60" mentirebbe li'.)
# NB: la query NON puo' vivere dentro l'encoding — get_decoder_input gira anche DENTRO la search
# (search.py:316) e search_end() libera TUTTI gli stati in blocco. Il generatore la calcola PRIMA
# e la passa a get_decoder_input, come gia' si fa con prize_tracker/dkr (armati per call-site,
# spenti dentro la search).
# NIENTE PREMI qui, di proposito: quale bersaglio uccidere e' una scelta strategica che resta
# alla rete, al passo del piazzamento, dove la CNT-tail le da' gia' i fatti per-bersaglio.
# FLAG: FUNCNET4_BENCH_FEAT=1 arma la SCRITTURA (default 0 = colonne a zero -> forward
# bit-identico a un ckpt pre-coda pad-ato). Truthy-only nel cfg_hash. Wire: chiave "dl" (CODEC 12).
DEC_BENCH_OFF = DEC_TGT_OFF + DEC_TGT_DIM             # 1076 (subito dopo la coda TGT)
BN_DMG        = 0   # danno piazzabile in panca / 400  (0 = bloccato, o panca vuota: LA guardia)
BN_KOS        = 1   # massimo numero di KO in panca ottenibili col budget / 3
BN_MIN_HP     = 2   # hp piu' bassi a cui puoi ridurre UN bersaglio / 400  (il valore di SETUP)
DEC_BENCH_DIM = 3
assert BN_MIN_HP + 1 == DEC_BENCH_DIM
BENCH_FEAT = _os.environ.get("FUNCNET4_BENCH_FEAT", "0") not in ("", "0")

DEC_DENSE_DIM_TF = DEC_BENCH_OFF + DEC_BENCH_DIM      # 283+768+3+4+3+15+3 = 1079

# NB: il "feature-stack visibilita'" (set-ctx / tools-words / probe-dmg / atk-potential) e' stato RIMOSSO
# 2026-07-14: era NET-NEGATIVE (indue il mai-attacca e blocca il WR) e da tempo lockato a 0. Storia +
# motivo nella memoria funcnet3-feature-stack-net-negative; il codice vive ancora in git se serve rifarlo
# come arm isolato. Le righe encoder/decoder restano ENC_DENSE_DIM / DEC_DENSE_DIM_TF (larghezza invariata).

# ---------------------------------------------------------------- DKR/PRZ-TAIL (mazzo-residuo, 2026-07-20)
# Due parole ENCODER nuove (non coda decoder: sono STATO, non per-azione). Il residuo
# `decklist − mano − scarti − board` = (mazzo ∪ premi) oggi e' ricavabile solo sottraendo 4 somme
# DeepSets dentro i pesi — la stessa aritmetica implicita che la rete NON consolida (misurato sul
# caso energie, memoria energy-misplay-diagnosis -> nrg-tail). Qui il conto e' engine-exact in
# Python (funcnet4/deck_tracker.py, port di agents/dragapult_rulebased.set_card_counts):
#   Word A ROLE_DECK_REMAINING (flag DKR): multiset residuo (mazzo ∪ premi indistinti — lo stato
#          informativo VERO pre-ricerca; niente oracle). SET-word standard: count/mult + view somma.
#   Word B ROLE_PRIZE_KNOWN (flag PRZ, braccio 2 SOPRA il 1): premi noti per eliminazione dalla
#          prima ricerca dell'intero mazzo (select.deck, cg/api.py:530) + scalare PZ_RESOLVED.
#          Con PRZ off la parola esiste (layout stabile fra i 2 bracci) ma resta TUTTA zero.
# Regola-3 come sempre: la rete VEDE il conto, la scelta resta sua.
# LAYOUT: a differenza delle code NRG/CNT/UNLOCK (colonne migrabili), aggiungere PAROLE
# cambia NUM_WORDS/N_ROLE -> arch nuova. Percio' il flag governa la COSTRUZIONE: DKR off (default)
# = encoder/modello/wire BYTE-IDENTICI al run v8 vivo (nessun fork); DKR on = 38 parole, N_ROLE 15,
# cfg_hash forkato via dkr_feat/prz_feat (truthy-only, farm.cfg_hash). Warm-start: le 2 parole sono
# SET-word che il reader condiviso enc_card_mlp gia' sa leggere; nuovo solo il role-embed (pad righe
# ZERO: model.load_state_dict + osfp --init, stesso pattern del fork token-mano 12->13).
# NB deploy: le parole si armano PER-MODELLO (call-site `dkr=`): gli expert 36-parole della lega
# restano al loro layout anche in un gen armato.
ROLE_DECK_REMAINING = 13
ROLE_PRIZE_KNOWN    = 14
PZ_RESOLVED_OFF = 2   # slot func della Word B (0/1 = count/mult SET): 1.0 = premi RISOLTI
DKR_FEAT = _os.environ.get("FUNCNET4_DKR_FEAT", "0") not in ("", "0")
PRZ_FEAT = _os.environ.get("FUNCNET4_PRZ_FEAT", "0") not in ("", "0")
assert not (PRZ_FEAT and not DKR_FEAT), \
    "FUNCNET4_PRZ_FEAT=1 richiede FUNCNET4_DKR_FEAT=1 (la Word B vive sopra la Word A)"
if DKR_FEAT:
    N_ROLE = 15
    NUM_WORDS = NUM_WORDS + 2   # 38 di default (24 base + 12 token-mano + Word A + Word B)

# ---------------------------------------------------------------- ZONE (ingresso per-zona del GRU, 2026-08-13)
# ByteRL (paper 2303.05197, Fig. 2) non fa MAI una media globale prima del core LSTM: ogni zona
# (mano, board mio, board avversario, mazzo, cimiteri, eroi) e' compressa PER CONTO SUO e le zone
# si CONCATENANO. Qui la stessa forma per l'ingresso del GRU (--gru-zones, funcnet4/model.py):
# medie DENTRO la zona, mai fra zone. La fonte di verita' dell'ordine parole e' get_encoder_input
# (verify_encoding.py EXPECT_ROLE); l'ablazione belief 08-12/13 (flip 5-8%, B~=C, cresce coi turni)
# ha misurato che dalla cannuccia della media globale passa quasi solo MEMORIA — queste zone sono
# il contenuto che quella memoria deve poter distinguere.
# NB: il layout dipende dal profilo spec corrente (N_HAND_TOKENS, DKR) — un modello arma gru_zones
# solo se il suo num_words == NUM_WORDS (guardia nel costruttore, principio dell'arming per-modello
# della DKR). Gli expert a layout storico non si armano mai.
ZONE_WORDS = (
    tuple(range(0, 8)),                                   # bench MIA (8 slot)
    tuple(range(8, 16)),                                  # bench AVVERSARIA
    (16,),                                                # attivo MIO
    (17,),                                                # attivo AVVERSARIO
    (18,),                                                # player MIO (scalari + LOC-tail)
    (19,),                                                # player AVVERSARIO
    (20,),                                                # mano (somma DeepSets)
    (21,),                                                # mazzo (decklist di partenza)
    (22,),                                                # stadium
    (23,),                                                # turno/flag
    tuple(range(24, 24 + N_HAND_TOKENS)),                 # token-mano individuali
) + ((tuple(range(24 + N_HAND_TOKENS, NUM_WORDS)),) if DKR_FEAT else ())   # code DKR/PRZ
N_ZONES = len(ZONE_WORDS)
assert sum(len(_z) for _z in ZONE_WORDS) == NUM_WORDS, "ZONE_WORDS deve PARTIZIONARE le parole"

# ---------------------------------------------------------------- LOC-TAIL (dove sono le mie carte, 2026-08-07)
# BUCO MISURATO: sonda su 29 carte del mazzo -> UNA sola leggibile con R2>0,5, mediana R2 0,148,
# contro R2 0,88 di uno scalare ESPLICITO come deckCount. La DKR-tail calcola il residuo ESATTO in
# Python ma lo consegna con _fill_set_dense, cioe' una SOMMA DeepSets su tutte le carte rimaste: per
# sapere "e' rimasta una Buio?" la rete dovrebbe accorgersi che manca un addendo dentro una somma di
# 14. Il conto e' esatto, l'ACCESSO no. Caso di innesco (replay 90545479 step 116): gioca Crispin per
# pescare una Buio dal MAZZO con entrambe le Buio gia' fuori (1 in mano, 1 negli scarti) -> Crispin
# resta attivato -> attach FORZATO -> la Psichica finisce su Munkidori invece che su Dragapult ex,
# che con [F]+{P} avrebbe armato Phantom Dive (200).
#
# Qui gli stessi conti diventano SCALARI ESPLICITI, uno per (categoria, zona). Regola-3 di sempre:
# la rete VEDE il conto, la scelta resta sua.
#
# "nel MAZZO" NON e' "unseen": unseen_counts conta (mazzo UNION premi). Crispin/Ultra Ball/Nest Ball
# pescano SOLO dal mazzo, quindi mazzo = unseen - premiati_noti (DeckPrizeTracker.prized, valido solo
# se .resolved). Finche' la PRZ non ha risolto: premi = 0 e mazzo = stima ALTA (prudente, dichiarato).
# LIMITE ereditato da unseen_counts: al SETUP (turn 0) i face-down restano nel residuo (deck_tracker
# docstring) -> off-by-few solo al setup, esatto dal turno 1.
#
# LAYOUT: coda IN CODA alla riga ENCODER, dopo il testo (mai dentro il prefisso func 0..73, che porta
# pesi gia' allenati su tutte le parole Pokemon -> non sarebbe un arm neutro). Scritta su UNA sola
# parola, la own-player (LOC_WORD=18): e' stato di PARTITA, non per-carta; la parola opp-player resta
# a ZERO (sarebbe un leak). ENC_DENSE_DIM (1109) e' CONDIVISO con funcnet/funcnet2/funcnet3 e NON si
# tocca: la larghezza tf4-only e' ENC_DENSE_DIM_TF (gemella di DEC_DENSE_DIM_TF).
# FLAG: FUNCNET4_LOC_FEAT=1 arma la COSTRUZIONE (riga 1109->1145) e il call-site `loc=` arma la
# SCRITTURA dei valori. A flag OFF: riga 1109, nessuna chiave di wire, cfg_hash invariato ->
# BYTE-IDENTICO a oggi (provato da funcnet4/test_loc_tail.py gate B e da verify_noop N3/N6).
# NB: la LARGHEZZA segue il solo FLAG ENV, mai il call-site — al contrario della DKR, dove `dkr=`
# governa la costruzione per call-site ed e' esattamente da li' che nacque il bug di deploy
# (agents/funcnet4_agent.py: rete a 36 parole vs modello a 38 = fail-soft, gioco random).
LOC_WORD      = 18   # parola ROLE_OWN_PLAYER (indice fisso del prefisso 24; verify_encoding.py lo usa gia')

LC_N_TYPES    = 9    # tipi 0..8, STESSA indicizzazione di D_ENERGY_TYPES e di TG_NRG_T0 (NON "tutti
                     # i membri di EnergyType", che sono 12). Nel pool le Basic Energy sono 8 (id 1..8,
                     # tipi 1..8): lo slot t=0 (COLORLESS) e' strutturalmente SEMPRE zero in tutte e 3
                     # le zone. Costa 3 colonne morte e compra l'allineamento d'indice con le altre code.
LC_DECK_T0    = 0    # +t : energie BASE di tipo t nel MAZZO      (= unseen - premiati noti)
LC_DISC_T0    = 9    # +t : energie BASE di tipo t negli SCARTI   (zona visibile)
LC_PRZ_T0     = 18   # +t : energie BASE di tipo t nei PREMI NOTI (0 finche' PRZ non ha risolto)
LC_DECK_CLS0  = 27   # +c : carte di classe c nel MAZZO
LC_CLS_POKEMON   = 0
LC_CLS_BASIC     = 1   # SOTTOINSIEME di POKEMON (Pokemon Base): B non e' una partizione, e' voluto
LC_CLS_SUPPORTER = 2
LC_CLS_ITEM      = 3
LC_CLS_TOOL      = 4
LC_CLS_STADIUM   = 5
LC_N_CLS      = 6
LC_DECK_NRG_SP = 33  # energie SPECIALI nel mazzo (cardType SPECIAL_ENERGY=6). Non e' nel brief: senza
                     # di essa le colonne-mazzo non partizionano il mazzo (e 10 dei 134 mazzi meta non
                     # hanno NESSUNA energia base -> la coda sarebbe cieca proprio li').
LC_DISC_POKE  = 34   # Pokemon negli scarti          (Night Stretcher / Lana's Aid pescano da li')
LC_DISC_NRG   = 35   # energie negli scarti, BASE + SPECIALI -> NON e' la somma delle 9 colonne-scarti
ENC_LOC_DIM   = 36
assert LC_DECK_T0 + LC_N_TYPES == LC_DISC_T0 and LC_DISC_T0 + LC_N_TYPES == LC_PRZ_T0
assert LC_PRZ_T0 + LC_N_TYPES == LC_DECK_CLS0 and LC_DECK_CLS0 + LC_N_CLS == LC_DECK_NRG_SP
assert LC_DISC_NRG + 1 == ENC_LOC_DIM
# NORMALIZZAZIONE: TUTTE le colonne sono conteggi-di-CARTE / 60.0, come SET_COUNT_OFF (_fill_set_dense,
# encoding.py) e P_DECK/P_DISCARD. NON /4 (quella e' la scala delle energie ATTACCATE a un mon).
LOC_FEAT = _os.environ.get("FUNCNET4_LOC_FEAT", "0") not in ("", "0")
ENC_LOC_OFF      = ENC_DENSE_DIM                                        # 1109 (subito dopo il testo)
ENC_DENSE_DIM_TF = ENC_DENSE_DIM + (ENC_LOC_DIM if LOC_FEAT else 0)     # 1109 (OFF) / 1145 (ON)
# ⚠ ENC_DENSE_DIM NON si ridefinisce MAI: verify_noop N1 confronta OGNI costante UPPERCASE di
# funcnet2.spec con quella di funcnet4.spec e fallirebbe. I nomi nuovi non esistono in funcnet2 ->
# non vengono confrontati.
