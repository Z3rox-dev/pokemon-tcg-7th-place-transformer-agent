"""LETHAL deterministico DECK-SPECIFICO per DRAGAPULT/DUSKNOIR (newdorf) — fork funcnet4.

NOME ESPLICITO (decisione Giovanni 2026-07-12): funcnet4 allena exploiter e NUOVI pilot; questo
modulo (whitelist compose, combo Cursed Blast/Munkidori, fizzle) e' calibrato SOLO sul mazzo
dragapult_dusknoir. Un pilot nuovo vuole il SUO modulo lethal (o nessuno) — non riusare questo
alla cieca: COMPOSE_PLAY_CIDS/COMPOSE_EVOLVE_CIDS/SELF_KO sono newdorf-centrici.

Storia e numeri: docs/lethal-override.md (probe/A-B in tools/lethal_probe.py, che importa da
qui: 620 partite override = 211/211 lethal convertiti, 0 falsi fire, 0 linee perse; compose
validato: 28% dei lethal endgame esistono SOLO componendo la board).

LethalFinder = DFS engine-exact sopra la Search API (search_begin via funcnet4.search.determinize
+ search_step): esiste una linea che VINCE la partita ENTRO il turno corrente? Regole:
  - vittoria valida solo NEL turno (un deck-out avversario al turno dopo vincerebbe con
    qualsiasi mossa, anche END: non e' il lethal che cerchiamo);
  - ai nodi avversari (es. promozione dopo un KO a meta' turno) serve vincere per OGNI sua
    scelta (forall);
  - candidati alla MAIN: ability Cursed Blast (Dusclops/Dusknoir) e Adrena-Brain (Munkidori),
    ATTACH {D}->Munkidori (setup), attacchi, e con compose=True anche PLAY/EVOLVE/RETREAT da
    whitelist SOLO-DETERMINISTICA (zone visibili: mano/cimitero/board — regola Giovanni
    2026-07-10: NIENTE ricerche nel mazzo, la carta puo' essere nei premi coperti);
  - follow-up (target, distribuzione segnalini, promozioni): esplorati tutti.
La modalita' BLIND (opp_deck=None) e' quella di deploy (Kaggle-legale): per il lethal-nel-turno
le carte nascoste avversarie sono irrilevanti — misurato blind==oracle 211/211.

VINCOLI di questo modulo (bundle di submission Kaggle): niente torch, niente funcnet4.selfplay,
niente env/set_num_threads. Dipendenze tutte gia' nel bundle (cg, bytrl.actions,
funcnet.{spec,encoding,features}, funcnet4.search).
"""
from cg.api import (search_step, search_end, OptionType, SelectType, SelectContext, AreaType,
                    LogType)
# O1 (2026-07-12): la DFS usa la variante LAZY di search_step (json->campi materializzati al
# primo accesso, semantica identica per i campi letti — vedi cg/api.py). Misurato: ~90% del
# tempo DFS era to_dataclass eager dell'intera Observation per nodo. Fallback all'eager se
# il box ha un engine/cg non ancora aggiornato (stessi risultati, solo piu' lento).
try:
    from cg.api import search_step_lazy as _search_step_dfs
except ImportError:                                   # engine/cg vecchio: eager
    _search_step_dfs = search_step
from bytrl.actions import enumerate_actions
from funcnet.encoding import get_card
from funcnet.spec import CARD_TABLE
from funcnet.features import ATTACKS
from funcnet4.search import determinize

# ability che entrano nel calcolo lethal: Cursed Blast (auto-KO + segnalini a scelta) e
# Adrena-Brain di Munkidori (con {D} attaccata sposta fino a 3 segnalini dai miei a un pokemon
# avversario = 30 danni SENZA auto-KO). L'ATTACH della {D} a Munkidori e' un SETUP candidato.
COMBO_CIDS = {132: "Dusclops", 133: "Dusknoir", 112: "Munkidori"}
MUNKIDORI_CID = 112
DARK_ENERGY_TYPE = 7                              # EnergyType.DARKNESS

# compose: la linea puo' COMPORRE la board prima di sparare. REGOLA (Giovanni 2026-07-10):
# SOLO mosse deterministiche al 100% — zone visibili. NIENTE ricerche nel mazzo (Ultra Ball,
# Crispin): la carta cercata puo' essere nei premi coperti -> esito non garantito.
# 2026-07-11 aggiunta Team Rocket's Watchtower (richiesta Giovanni): giocarla RIMPIAZZA lo
# stadium avversario — Battle Cage (65% del meta alakazam: blocca i segnalini in panchina =
# spegne Phantom Dive spread + Cursed Blast) o Nighttime Mine (+{C} sul ns Dragapult Tera) —
# e il suo effetto ("{C} Pokémon have no Abilities") NON tocca i nostri combo-mon ({P}/{D}).
# Deterministica: play dalla mano, zona visibile; l'engine valida comunque la linea intera.
COMPOSE_PLAY_CIDS = {1097: "Night Stretcher", 1182: "Boss's Orders", 112: "Munkidori",
                     1256: "Team Rocket's Watchtower"}
COMPOSE_EVOLVE_CIDS = {121, 132, 133, 120}        # Dragapult ex, Dusclops, Dusknoir, Drakloak
                                                  # (120 aggiunto 2026-07-11: evolvere il Dreepy
                                                  # attivo abilita Dragon Headbutt nello stesso turno)
# ability con AUTO-KO (Cursed Blast): regalano un premio all'avversario — vedi
# mask_suicidal_ability (guardia anti-suicidio, 2026-07-11)
SELF_KO_ABILITY_CIDS = {132, 133}
# guardia di sicurezza: se mai la whitelist venisse riallargata a ricerche-mazzo, le linee che
# le contengono vanno comunque scartate (deterministico SEMPRE) — vedi has_deck_search().
DECK_SEARCH_NAMES = ("Ultra Ball", "Crispin")

# select di piazzamento segnalini (per la mask no-overkill)
_COUNTER_CONTEXTS = (int(SelectContext.DAMAGE_COUNTER), int(SelectContext.DAMAGE_COUNTER_ANY),
                     int(SelectContext.DAMAGE))
# SOLO piazzamento SEGNALINI (13/14): il ctx DAMAGE(15) e' assegnazione di danno DIRETTO,
# che le prevention "damage is not an effect" (Mist/Rock/Battle Cage) NON bloccano — il
# fizzle-check vale solo qui (fix 2026-07-11 sera: prima includeva 15 = falso positivo).
_COUNTER_PLACE_CONTEXTS = (int(SelectContext.DAMAGE_COUNTER), int(SelectContext.DAMAGE_COUNTER_ANY))

# energie che PREVENGONO gli effetti degli attacchi avversari sul pokemon a cui sono attaccate
# (engine: NoEffectEnemyAttack, "damage is not an effect" -> il danno DIRETTO passa, i
# segnalini da effetto-ATTACCO fizzlano a 0; verificato su engine_src CardImpl.h:10403 +
# EffectInstant.h:38-55 e sul replay live 85359728: 12 segnalini di Phantom Dive sprecati
# su Mega Lucario ex protetto, partita persa). Mist(11) = qualsiasi tipo; Rock Fighting(20)
# = solo se il bersaglio e' {F} (energyType 6). Le ability (Cursed Blast/Adrena-Brain) NON
# sono attacchi: i loro segnalini PASSANO -> mai mascherarle per questo motivo.
_PREVENT_MIST, _PREVENT_ROCK, _FIGHTING = 11, 20, 6
# Battle Cage (engine_src CardImpl.h:10375: NoDamageCounterEnemyAttackAbility targetBench,
# Both): mentre e' in campo, i segnalini da ATTACCHI *E ABILITY* avversarie NON si piazzano
# sui pokemon in PANCHINA (di entrambi); il danno diretto passa. 65% del meta alakazam.
BATTLE_CAGE = 1264
CRUSHING_HAMMER = 1120
# Fezandipiti ex — Ability "Flip the Script": pesca 3, ma SOLO "if any of your Pokemon were
# Knocked Out during your opponent's last turn". Vedi KOTracker/mask_dead_fezandipiti (Regola 6).
FEZANDIPITI_CID = 140
# Budew — ability di item-lock finche' e' ATTIVO: e' l'apertura #1 della guida pro in ENTRAMBE le
# liste primo/secondo (cfr. funcnet4/guide_masks.py:33-37).
# ⚠ NON piu' usata dalla Regola 6 dal 2026-08-02 (versione SEMPLICE: l'unica eccezione e' "non c'e'
# altro pokemon piazzabile"). Tenuta perche' e' un id di carta condiviso, non una soglia della regola.
BUDEW_CID = 235
_POKEMON_CARDTYPE = 0             # CardType.POKEMON (stesso pattern di _ENERGY_CARDTYPES)
# ⚠ ORFANA dal 2026-08-02: era la soglia del caso-3 della Regola 6-BIS (turn>=3), caso rimosso nella
# versione semplice. Lasciata come riferimento se un giorno si volesse riestendere la copertura.
FEZ_FRAGILE_HP = 70


REPELLING_VEIL_CID = 414       # Team Rocket's Articuno — ability "Repelling Veil": previene gli
                               # EFFETTI degli attacchi avversari sui PROPRI Basic Team Rocket
                               # (il DANNO diretto no; le ABILITY tipo Cursed Blast NEMMENO, non
                               # sono attacchi). Vedi _rv_reorder_counters.


def _articuno_in_play(opp_ps):
    """True se l'avversario ha un Team Rocket's Articuno in gioco (Repelling Veil). Presence-based:
    la logica che lo usa e' SOLO riordino (mai potatura) -> corretta anche se l'ability fosse
    disattivata, al massimo riordino subottimo. Vedi _rv_reorder_counters."""
    for z in (opp_ps.active, opp_ps.bench):
        for c in (z or []):
            if c is not None and int(getattr(c, "id", -1) or -1) == REPELLING_VEIL_CID:
                return True
    return False


def _effect_prevented(mon):
    """True se il pokemon ha una prevention-energy attiva contro gli EFFETTI di attacco."""
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
    """Descrizione umana di una selection (lista di indici-opzione)."""
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
    """True se la linea contiene una ricerca nel mazzo (non deterministica: MAI giocarla)."""
    return any(f"PLAY {n}" in d for n in DECK_SEARCH_NAMES for d in seq)


def lethal_sig(o):
    """Firma deterministica (mano, board propria) per il RIARMO della detection: tuple
    ordinate di card-id. Usata da agent deploy e generatori insieme a gained_material()."""
    st = o.current
    me = st.players[st.yourIndex]
    hand = tuple(sorted(int(c.id) for c in (me.hand or []) if c is not None))
    board = tuple(sorted(int(c.id) for z in (me.active, me.bench)
                         for c in (z or []) if c is not None))
    return hand, board


def _prize_value(cid):
    """Premi che l'avversario CEDE se questo pokemon viene KO (prize_canon): Mega ex=3, ex=2,
    altrimenti 1. Stessa economia usata dall'engine (verificata ep 86371563)."""
    cd = CARD_TABLE.get(cid)
    if cd is None:
        return 1
    if getattr(cd, "megaEx", False):
        return 3
    return 2 if getattr(cd, "ex", False) else 1


def lethal_plausible(o):
    """Precheck ARITMETICO obs-only del gate premi (2026-07-18, ep Kaggle 86560288: doppio Mega
    3+3=6 premi nello stesso turno — INVISIBILE al vecchio gate cieco `premi>4 && opp>2`, che
    presupponeva <=2 premi/KO, pre-prize_canon: vittoria in tasca persa al t11). SOVRASTIMA
    deliberata: ignora energia/status/protezioni/resistenza (verifica engine-exact = compito del
    DFS) -> False = vittoria aritmeticamente IMPOSSIBILE nel turno, finder skippabile senza
    perdere linee. Copre il BENCHOUT (subset = TUTTI i pokemon avversari: l'engine assegna la
    vittoria a prescindere dai premi — ep 85338458, il motivo della vecchia clausola opp<=2).

    Risorse modellate (newdorf-centrico come tutto il modulo):
      - attacco sull'ATTIVO: Phantom Dive 200 (deb x2 engine-rule; resistenza IGNORATA =
        sovrastima) se Dragapult in gioco o evolvibile ORA (Drakloak in campo + Dragapult in
        mano, come il compose); fallback 90 (Dragon Headbutt);
      - 6 segnalini BENCH-only della Dive (solo se la Dive c'e');
      - pool any-target: Cursed Blast 13/5 per ogni Dusknoir/Dusclops in gioco O in mano
        (l'evolvibilita' esatta la verifica il DFS; cap 2 copie), Adrena-Brain +3 se Munkidori.
    Costo: aritmetica su <=6 bersagli, enumerazione 2^n — microsecondi, niente engine."""
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
    tgts = []                                             # (hp, premi, is_active)
    for c in (opp.active or []):
        if c is not None and (getattr(c, "hp", 0) or 0) > 0:
            tgts.append((int(c.hp), _prize_value(int(c.id)), True))
    for c in (opp.bench or []):
        if c is not None and (getattr(c, "hp", 0) or 0) > 0:
            tgts.append((int(c.hp), _prize_value(int(c.id)), False))
    if not tgts:
        return True
    atk = 200 if dive else 90
    # debolezza: raddoppia col TIPO dell'attaccante (regola engine misurata,
    # tools/measure_damage_rule.py); attaccante = Dragapult se la Dive e' la linea
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
        need_pool = 0                                     # segnalini SOLO dal pool any-target
        need_bench = 0                                    # pagabili da dive-bench, resto dal pool
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
            continue                                      # allocazione impossibile
        if prizes >= needed or mask == full:               # premi sufficienti o BENCHOUT
            return True
    return False


def gained_material(prev, now):
    """True se `now` contiene carte NON gia' presenti in `prev` (confronto per multinsieme
    su mano e board propri). E' l'UNICO evento che la DFS compose non puo' anticipare da una
    detection precedente nello stesso turno: una pescata/ricerca riuscita (es. Fezandipiti,
    Poffin) porta materiale nuovo e il lethal puo' NASCERE a meta' turno — misurato live
    (ep Kaggle 85335402 turno 11: lethal comparso dopo la pescata, cap detection gia' speso).
    Su questo evento la detection si riarma. prev None = nessun riarmo."""
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
    """(materiale, posizione) per il RIARMO della detection (2026-07-20, ep Kaggle 87104061 t9:
    Boss's Orders porta il Fez ex 210hp in attivo + RETREAT mette Dragapult davanti = lethal
    NATO senza materiale nuovo -> gained_material cieco, detection strutturalmente spenta dal
    3° tentativo e vittoria rimandata di 2 turni). materiale = lethal_sig (riarmo su carte
    NUOVE via gained_material, invariato); posizione = SOLO i cambi che possono far nascere un
    lethal a materiale fermo: attivo MIO (id + numero energie attaccate: un attach puo' rendere
    pagabile l'attacco che chiude), board AVVERSARIA (id, hp, attivo/panchina: gust, danni,
    promozioni). Mitigazione costo deliberata: i PLAY sterili nostri (stadium, supporter senza
    effetto-board) NON riarmano; il tetto 6 tentativi/turno resta."""
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
    """Coda della linea-PV (find()['ais']) per la PROSSIMA decisione dell'EXEC-MODE: consuma
    la mossa appena giocata (ais[0]) e gli eventuali slot-avversario (None) che il gioco VERO
    risolve prima del nostro prossimo select. None = linea esaurita (ricerca normale)."""
    tail = list(ais[1:])
    while tail and tail[0] is None:
        tail.pop(0)
    return tail or None


def should_rearm(prev, now):
    """True se fra prev e now (entrambi rearm_sig) e' cambiato qualcosa che puo' far NASCERE
    un lethal: materiale nuovo (gained_material, solo AGGIUNTE) oppure QUALSIASI cambio di
    posizione (confronto secco: anche un hp avversario che scende conta). prev None = nessun
    riarmo (semantica identica a gained_material)."""
    if prev is None:
        return False
    return gained_material(prev[0], now[0]) or prev[1] != now[1]


def mask_dead_counter_targets(o, actions):
    """REGOLA 2 no-overkill (docs/lethal-override.md): nelle select di piazzamento segnalini,
    le azioni che mettono segnalini su un pokemon avversario GIA' a 0hp sono strettamente
    dominate (danno sprecato; l'engine aggiorna gli hp per-segnalino, verificato nei dump).
    Ritorna la lista di indici-azione da TENERE, oppure None quando non c'e' nulla da fare:
    select non di segnalini, nessun morto tra i bersagli, o filtro che svuoterebbe tutto
    (fallback obbligatorio: non mascherare). Il bersaglio tra i vivi resta scelta della rete."""
    sel = o.select
    if int(getattr(sel, "context", -1)) not in _COUNTER_CONTEXTS:
        return None
    me = o.current.yourIndex
    # FIZZLE-CHECK (2026-07-11): i segnalini di un effetto-ATTACCO su un bersaglio con
    # Mist/Rock-Fighting Energy fanno 0 danno -> dominati come l'overkill. Attivo SOLO se
    # la sorgente dell'effetto e' un attacco CERTO: carta-sorgente SENZA ability (es.
    # Dragapult ex, skills=[]) -> il counter-select puo' venire solo dal suo attacco.
    # Cursed Blast/Adrena-Brain (sorgenti CON ability) non vengono mai mascherate qui.
    src = getattr(sel, "effect", None)
    src_card = CARD_TABLE.get(int(getattr(src, "id", -1) or -1)) if src is not None else None
    atk_certain = src_card is not None and not getattr(src_card, "skills", None)
    # fizzle SOLO sui contesti di piazzamento-segnalini (13/14): il ctx DAMAGE(15) e' danno
    # diretto e le prevention non lo bloccano ("damage is not an effect")
    is_place = int(getattr(sel, "context", -1)) in _COUNTER_PLACE_CONTEXTS
    # Battle Cage in campo? (blocca i segnalini in PANCHINA da attacchi E ability avversari)
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
                continue                      # spostare/curare sui MIEI non e' overkill
            try:                              # difensivo: opzione non risolvibile = non toccare
                mon = get_card(o, opt.area, opt.index, side)
            except Exception:
                continue
            hp = getattr(mon, "hp", None)
            if hp is not None and int(hp) <= 0:
                dead = True
                break
            if is_place and atk_certain and _effect_prevented(mon):
                dead = True                   # fizzle Mist/Rock: 0 danno, dominato
                break
            if (is_place and cage
                    and int(getattr(opt, "area", -1) or -1) == int(AreaType.BENCH)):
                dead = True                   # fizzle Battle Cage: bench avversaria protetta
                break                         # (copre attacchi E ability -> niente atk_certain)
        if not dead:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


def mask_suicidal_ability(o, actions):
    """GUARDIA ANTI-SUICIDIO (2026-07-11, famiglia Regola-2): l'ability con AUTO-KO
    (Cursed Blast di Dusclops/Dusknoir) REGALA un premio all'avversario. Usarla e'
    SCONFITTA/REGALO ISTANTANEO e certo quando: (a) all'avversario manca 1 premio
    (il self-KO glielo consegna), oppure (b) e' il mio ULTIMO pokemon in campo
    (benchout contro di me). Aritmetica engine-esatta: premi rimanenti + conteggio
    board — zero giudizio. Blunder fotografato nei probe: doppio suicidio da
    posizione vinta -> pareggio. Ritorna gli indici da TENERE oppure None (niente
    da mascherare / maschererebbe tutto). NB: una linea lethal FORZATA non passa di
    qui (precedenza forced > mask) e comunque l'engine non puo' averla dimostrata
    vincente attraverso una mossa che fa vincere l'avversario."""
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    mine = st.players[me]
    opp_prizes = len(st.players[1 - me].prize or [])
    my_in_play = sum(1 for z in (mine.active, mine.bench) for c in (z or []) if c is not None)
    if opp_prizes > 1 and my_in_play > 1:
        return None                          # il self-KO non regala nulla di decisivo
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
            try:                              # difensivo: opzione non risolvibile = non toccare
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


_ENERGY_CARDTYPES = (5, 6)        # Basic / Special Energy (cardType)
NIGHTTIME_MINE = 1266


def _energy_cap(cid, nm_in_play, slack=1):
    """Cap REGOLA 3 per un pokemon: (costo del suo attacco piu' caro) + slack.
    slack=1 (storico) = buffer ritirata/denial (Handheld Fan, Crushing Hammer);
    slack=0 = STRICT (A/B 2026-07-16): cap secco al costo esatto — l'attach sbagliato
    satura lo slot ed e' irrecuperabile. Il flag vive nei CHIAMANTI
    (FUNCNET4_ENERGY_STRICT in agent/selfplay): questo modulo resta senza env
    (vincolo bundle Kaggle).
    None = MAI cappare: niente attacchi in tabella, oppure un attacco il cui testo
    coinvolge l'energia (scaling/condizionale: piu' energia puo' valere di piu').
    Engine-aware: +1 se il mon e' Tera e Nighttime Mine e' in campo (tassa +{C})."""
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
    """REGOLA 3 (Giovanni 2026-07-11): mai ATTACH di un'ENERGIA a un mio pokemon che ha
    gia' (costo max dei suoi attacchi) + slack energie attaccate — taglia l'accumulo infinito
    (spreco dell'attach once-per-turn). Il +1 e' il buffer per ritirata/denial (Handheld
    Fan, Crushing Hammer). Solo select MAIN, solo carte-ENERGIA (i tool non c'entrano),
    guard sugli attacchi energia-dipendenti, fallback mai-svuotare. Le mosse FORZATE dal
    lethal non passano di qui (forced > mask) e la DFS attach-generica e' inalterata."""
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
            try:                              # difensivo: opzione non risolvibile = non toccare
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
    """True se QUESTO Munkidori e' nel regime 'solo {D}' della Regola 4 (invariante
    Giovanni 2026-07-11 sera): (a) in BENCH senza {D}; (b) ATTIVO con >=1 energia e senza
    {D} (la ritirata scarta l'ALTRA energia a scelta, quindi la {D} li' e' sicura).
    ECCEZIONE: ATTIVO con 0 energie -> attach LIBERA (l'energia sacrificabile e' carburante
    di ritirata che CONSERVA la {D} in mano per la backline — la {D} attaccata all'attivo
    scoperto verrebbe bruciata dalla ritirata)."""
    if getattr(tmon, "id", None) != MUNKIDORI_CID or _has_dark(tmon):
        return False
    is_active = int(getattr(opt, "inPlayArea", -1) or -1) == int(AreaType.ACTIVE)
    n_energy = len(getattr(tmon, "energyCards", None) or [])
    if is_active and n_energy == 0:
        return False                          # eccezione carburante-ritirata
    return True


def mask_wrong_energy_to_munkidori(o, actions):
    """REGOLA 4 (Giovanni 2026-07-11, + invariante ATTIVO della sera): se ho una {D} in
    mano, l'attach verso un Munkidori 'protetto' (vedi _munki_protected) puo' essere SOLO
    la {D} — un'altra energia li' non abilita Adrena-Brain e ruba un'energia utile altrove
    ({D} copre comunque gli slot {C}): dominata. Deterministica: mano mia visibile, solo
    Munkidori usa la {D} nel newdorf. NON forza l'attach a Munkidori (il bersaglio resta
    scelta della rete); maschera solo l'accoppiata sbagliata. Fallback mai-svuotare.
    NB tipo energia da CARD_TABLE (istanza in mano ha energyType None — bug noto)."""
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    # esiste un attach {D} -> Munkidori-protetto tra le azioni? (altrimenti niente da fare)
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
                bad = True                    # energia NON-{D} su Munkidori protetto
                break
        if not bad:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


def mask_dark_energy_misassign(o, actions):
    """REGOLA 4-bis (Giovanni 2026-07-11, visto in training): se il MIO Munkidori e' in
    posizione ATTIVA e senza {D}, la {D} dalla mano — SE viene attaccata — puo' andare SOLO
    a lui. NON forza l'attach della {D} (le altre energie e azioni restano libere, inclusa
    l'energia-sacrificabile su Munkidori dell'invariante attivo); maschera solo
    {D}->non-Munkidori. Razionale: con Munkidori attivo nessun altro attacca questo turno,
    la {D} su un panchinaro e' solo filler {C} futuro mentre su Munkidori abilita
    Adrena-Brain SUBITO — e la {D} e' unica nel mazzo. Le linee lethal forzate bypassano le
    mask (forced > mask). Fallback mai-svuotare."""
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
            try:                              # difensivo: opzione non risolvibile = non toccare
                ecard = get_card(o, opt.area, opt.index, me)
                tmon = get_card(o, opt.inPlayArea, opt.inPlayIndex, me)
            except Exception:
                continue
            ec = CARD_TABLE.get(getattr(ecard, "id", None))
            if (int(getattr(ec, "energyType", -1) or -1) == DARK_ENERGY_TYPE
                    and getattr(tmon, "id", None) != MUNKIDORI_CID):
                bad = True                    # la {D} verso chiunque non sia Munkidori
                break
        if not bad:
            keep.append(ai)
    if not keep or len(keep) == len(actions):
        return None
    return keep


def mask_useless_hammer(o, actions):
    """REGOLA 5 (2026-07-11 sera): mai giocare Crushing Hammer quando l'avversario ha ZERO
    energie attaccate in campo — la moneta non ha bersagli, la carta viene bruciata per
    niente (in mano resta almeno fodder per Ultra Ball): strettamente dominato. Visibile e
    deterministico (le energie attaccate avversarie si vedono). Fallback mai-svuotare."""
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
    """Ricorda se un MIO pokemon e' morto nel turno avversario appena concluso (condizione
    di Flip the Script / Unfair Stamp). NON e' derivabile dalla sola Observation: `obs.logs`
    e' un DELTA ("events since the last selection", cg/api.py) e ne' State ne' PlayerState
    espongono un campo "KO nel turno scorso" — verificato. L'evento KO arriva nel delta del
    turno AVVERSARIO (tipicamente alla mia promozione dopo il KO) e a quel punto NON e' piu'
    nel delta quando parte la mia MAIN: senza accumulo il segnale e' semplicemente perso.
    Stessa logica del rule-based newdorf (accumula, ruota su TURN_END).

    UNA istanza PER PARTITA (reset() a inizio partita; turn 0 riazzera come guardia).
    observe() va chiamata su OGNI decisione, comprese quelle a mossa-unica che gli agent
    cortocircuitano prima della rete: quelle sono spesso proprio le promozioni post-KO, cioe'
    l'unico punto in cui il log del KO transita."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev, self._cur = [], []

    def observe(self, o):
        st = getattr(o, "current", None)
        if st is None:
            return
        if int(getattr(st, "turn", 0) or 0) == 0:      # nuova partita
            self.reset()
            return
        for lg in (getattr(o, "logs", None) or []):
            self._cur.append(lg)
            if int(lg.type) == int(LogType.TURN_END):
                self._prev, self._cur = self._cur, []

    def pre_ko(self, me):
        """True se un mio POKEMON e' passato dal campo allo scarto nel turno appena concluso.

        🔴 FIX 2026-08-02 (replay 89480887, gate tools/verify_fez_preko.py): mancava il controllo
        sul TIPO di carta. Dal campo allo scarto ci vanno anche le ENERGIE e i TOOL (un attacco
        avversario che scarta energia dal difensore, la rimozione di uno strumento...), e ognuno
        di quegli eventi veniva contato come un KO -> `pre_ko=True` -> `mask_dead_fezandipiti`
        usciva alla sua prima riga (`if pre_ko: return None`) -> la Regola 6 NON si applicava.
        Sintomo osservato: Fezandipiti ex piazzato al turno 6-7 con piu' Pokemon in campo e
        nessun KO subito, cioe' un ex da 2 PREMI regalato con l'ability morta.
        Il `cardId` era gia' nel Log dell'engine, semplicemente non veniva guardato.
        ⚠ Stesso difetto in agents/dragapult_dusknoir_rulebased.py:344 e negli altri rule-based:
        NON e' un errore di porting, e' ereditato — vanno corretti separatamente.
        Carta non risolvibile in tabella -> si CONTA come KO (conservativo: meglio permettere una
        giocata di Fez in piu' che mascherarla dopo un KO vero non riconosciuto)."""
        for lg in self._prev:
            try:
                if (int(lg.type) == int(LogType.MOVE_CARD)
                        and getattr(lg, "playerIndex", None) == me
                        and int(lg.fromArea) in (int(AreaType.BENCH), int(AreaType.ACTIVE))
                        and int(lg.toArea) == int(AreaType.DISCARD)):
                    _cid = getattr(lg, "cardId", None)
                    if _cid is not None:
                        _ct = getattr(CARD_TABLE.get(_cid), "cardType", None)
                        # ⚠ CardType.POKEMON vale 0 (falsy): confronto esplicito con None, mai `or -1`
                        if _ct is not None and int(_ct) != _POKEMON_CARDTYPE:
                            continue          # energia/strumento/supporter: NON e' un KO
                    return True
            except Exception:                          # log senza i campi area: non e' un KO
                continue
        return False


def mask_dead_fezandipiti(o, actions, pre_ko):
    """REGOLA 6 (Giovanni 2026-07-15): mai PLAY di Fezandipiti ex se "Flip the Script" NON
    puo' sparare, cioe' se nessun mio pokemon e' morto nel turno avversario appena concluso.
    Senza quella condizione l'ability e' morta e la carta e' solo un bersaglio ex da 2 PREMI
    piazzato in panchina per niente (l'ability e' usabile lo stesso turno in cui la giochi,
    quindi giocarla in reazione al KO ottiene la stessa pescata senza esporla prima).

    NON e' una mossa dominata come le Regole 3/5 — e' un compromesso: pre-piazzarla la mette
    al sicuro da Lillie's Determination x4 (rimescola la mano) e assicura il corpo. Percio' e'
    gated e A/B-abile, non "aritmetica pura". Misura che l'ha motivata (main fn4 step1940,
    n=60): 22 dei 31 play erano MORTI (71%), ai turni 1-7 — contro il 7% del rule-based.

    REGOLA 6-QUATER (Giovanni 2026-08-10). La regola e' UNA sola e vale a QUALUNQUE turno:

      non piazzare Fezandipiti ex se nel turno avversario appena concluso NON e' morto nessuno,
      SALVO che sia l'ultima difesa dal BENCH-OUT: `my_in_play <= 1` **E** nessun altro pokemon
      piazzabile. Servono ENTRAMBE.

    🔴 Perche' e' cambiata: la 6-TER si asteneva sul solo `not other_pokemon`, e quel guard non
    guarda quanti corpi hai GIA' in campo. Causa radice del replay 90511497 (turno 6, Fez piazzato
    con l'ability morta): avevamo attivo + 4 in panchina — bench-out impossibile — ma in mano
    l'unico pokemon era Fez, perche' la seconda carta pokemon era un'EVOLUZIONE (opzione EVOLVE,
    non PLAY) e il guard conta solo i PLAY. Diagnosi precedente ("finestra stantia del KOTracker")
    REFUTATA: rigiocando il replay, `pre_ko` al turno 6 valeva correttamente False e l'oracolo
    engine conferma `turnHistories[1].ko == False` — vedi tools/repro_fez_stale_window.py e
    tools/autopsy_fez_replay.py. Frequenza del difetto (tools/measure_fez_guard.py, 60 partite):
    la maschera si asteneva nel 25% delle decisioni con ability morta, e l'80% di quelle
    astensioni non aveva alcun rischio di bench-out (3, 4 o 5 corpi in campo).

    Sostituisce due generazioni di casi speciali: prima tre rami separati (eccezione Budew al
    turno 1, soglia hp<=70 da turn>=3), poi un tentativo di limitare la maschera al solo primo
    turno. Erano entrambi ripieghi per lo stesso motivo sbagliato — si sospettava la regola,
    mentre il guasto era in `KOTracker.pre_ko`, che contava le ENERGIE scartate come KO e quindi
    spegneva la maschera in silenzio nei turni centrali (e' cosi' che il main ha giocato Fez al
    turno ~6-7 nel replay 89480887). Corretto quello il 2026-08-02, i ripieghi non servono piu' e
    la copertura torna piena su tutta la partita.

    ⚠ La regola DIPENDE da `pre_ko` a ogni turno: e' una scelta consapevole, non una svista. Il
    presidio e' `tools/verify_fez_preko.py` — se quel gate torna rosso, questa maschera si spegne
    di nuovo senza dare segno. Non rimuoverlo.

    La piazzata di SETUP non passa di qui (altro select, non MAIN). Fallback mai-svuotare."""
    if pre_ko:
        return None                            # l'ability spara davvero: nessun vincolo
    sel = o.select
    if int(sel.type) != int(SelectType.MAIN):
        return None
    st = o.current
    me = st.yourIndex
    mine = st.players[me]

    # Classifica le azioni UNA VOLTA: quali giocano Fez, e se esiste un ALTRO pokemon piazzabile.
    fez_actions, other_pokemon = set(), False
    for ai, act in enumerate(actions):
        for oi in act:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.PLAY):
                continue
            try:                               # difensivo: opzione non risolvibile = non toccare
                hcard = get_card(o, AreaType.HAND, opt.index, me)
            except Exception:
                continue
            cid = getattr(hcard, "id", None)
            if cid == FEZANDIPITI_CID:
                fez_actions.add(ai)
            else:
                # ⚠ NIENTE `or -1` qui: CardType.POKEMON vale **0**, che e' falsy — l'idioma usato
                # per le energie (_ENERGY_CARDTYPES = 5,6, entrambi truthy) qui trasformerebbe ogni
                # pokemon in -1 e "altro pokemon" non verrebbe MAI rilevato.
                _ct = getattr(CARD_TABLE.get(cid), "cardType", None)
                if _ct is not None and int(_ct) == _POKEMON_CARDTYPE:
                    other_pokemon = True
    if not fez_actions:
        return None                            # nessun play di Fez in tavola: niente da mascherare
    # Eccezione BENCH-OUT: ci si astiene solo se Fez e' davvero l'ultima difesa dal restare senza
    # corpi. Servono ENTRAMBE le condizioni — pochi corpi in campo E nessun altro piazzabile.
    my_in_play = sum(1 for z in (mine.active, mine.bench) for c in (z or []) if c is not None)
    if my_in_play <= 1 and not other_pokemon:
        return None                            # ultimo corpo disponibile: meglio lui che niente

    keep = [ai for ai in range(len(actions)) if ai not in fez_actions]
    if not keep or len(keep) == len(actions):
        return None
    return keep


def mask_dominated_main(o, actions, pre_ko=None, energy_slack=1):
    """Combina le mask MAIN (anti-suicidio + Regola-3 energia + Regola-4 {D}->Munkidori +
    Regola-5 Hammer-a-vuoto + Regola-6 Fezandipiti-morto): interseca i keep; se l'intersezione
    svuoterebbe, precedenza in ordine di lista (anti-suicidio = sconfitta istantanea, prima di
    tutto). Ritorna (keep, tag) per la telemetria.

    pre_ko=None (default) = il chiamante non tiene un KOTracker -> Regola 6 SPENTA, comportamento
    identico a prima. Non e' derivabile da `o`: va passata da chi accumula i log (vedi KOTracker)."""
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
    return active[0][1], active[0][0]      # precedenza in ordine di lista (suicide prima)


class _Abort(Exception):
    pass


class _Pass1Stop(Exception):
    """Tetto della PASSATA-1 (linea minima) raggiunto: si passa al compose pieno.
    NON e' un abort: il budget complessivo resta quello, viene solo speso in due fasi."""
    pass


class LethalFinder:
    """DFS engine-exact: esiste una linea {ability combo / setup / attacco (+ follow-up)} che
    vince ENTRO il turno corrente? Ai nodi avversari richiede la vittoria per OGNI scelta
    (forall). Ogni find() apre un mondo determinizzato e chiude SEMPRE con search_end() —
    percio' find() NON va MAI chiamata mentre un'altra search (es. Gumbel) ha stati aperti:
    l'API engine libera tutto in blocco. Per quel caso c'e' find_in_world()."""

    def __init__(self, my_deck, opp_deck, rng=None, budget=3000, max_depth=14,
                 cap_follow=28, cap_opp=8, allow_ability=True, allow_munkidori=True,
                 compose=False, allow_attach_any=None, rv_reorder=True):
        self.my_deck, self.opp_deck, self.rng = my_deck, opp_deck, rng
        self.compose = compose               # True = PLAY/EVOLVE/RETREAT whitelist (v3)
        self.rv_reorder = bool(rv_reorder)   # riordino Repelling Veil (A/B: False = pre-round4)
        # 2026-07-11 ATTACH GENERICA (richiesta Giovanni): un lethal puo' richiedere l'energia
        # mancante DALLA MANO (es. la 3a con la tassa Nighttime Mine sul ns Dragapult Tera).
        # Deterministica (mano visibile, attach once-per-turn imposto dall'engine).
        # Default: segue compose; False esplicito = arm A/B senza.
        self.allow_attach_any = bool(compose) if allow_attach_any is None else bool(allow_attach_any)
        if compose:
            # floor VALIDATO (620 partite A/B): sotto 12000/24 le linee compose muoiono in
            # depth_cap/abort e il livello compose sottoperforma in silenzio
            budget = max(budget, 12000)
            max_depth = max(max_depth, 24)
        self.budget_cap, self.max_depth = budget, max_depth
        self.cap_follow, self.cap_opp = cap_follow, cap_opp
        self.allow_ability = allow_ability   # False = solo attacchi ("bastava attaccare?")
        self.allow_munkidori = allow_munkidori   # False = "Munkidori era DECISIVO?"
        self._p1cap = None                   # not-None = PASSATA-1 in corso (linea minima, no rank-4)
        self._rank4_excluded = False         # O4: la passata-1 ha SCARTATO almeno un rank-4
        self._rv_active = False              # round-4: Articuno avv in gioco (calcolato 1x/find,
                                             # NON per-nodo -> costo zero su board senza Articuno)

    def find(self, o, minimal=False, pv=None):
        """None oppure dict(first_ai, seq, ais, flags). first_ai = indice (in enumerate_actions
        dello stato LIVE) della prima mossa della linea vincente trovata. Le linee con
        ricerca-mazzo vengono scartate QUI (determinismo garantito a monte di ogni caller).
        minimal=True (2026-07-16): SOLO passata-1 (linea minima, no rank-4) — usata dai
        RIARMI della detection (~1/3 del costo pieno; il materiale appena arrivato abilita
        quasi sempre linee corte, il compose profondo e' gia' stato cercato ai tentativi 1-2).
        pv (2026-07-20, O3): `ais` di una find PRECEDENTE come ordinamento killer-move — a ogni
        nodo si prova PRIMA la mossa ricordata, quindi una linea ancora valida si riconferma in
        ~len(linea) passi invece che esponenziali (misurato: EXEC era ~40% del costo lethal in
        gen). SOLO ordinamento: albero e budget identici, nessuna linea persa; se la linea e'
        saltata la DFS degrada da sola a ricerca piena. Usata da EXEC-MODE e dal verify (2°
        mondo). Con pv la passata-1 si salta: se la linea contiene rank-4 la pass-1 non puo'
        trovarla per costruzione e brucerebbe il suo tetto a vuoto."""
        self.steps = 0
        self.flags = set()
        self._rank4_excluded = False
        self.me = o.current.yourIndex
        self.root_turn = int(o.current.turn)
        # round-4: Articuno avversario in gioco? Calcolato UNA volta qui (non per-nodo): un
        # Articuno non puo' ENTRARE in gioco durante il NOSTRO turno (l'avversario non gioca) ->
        # basta lo stato radice. False = board normale -> il blocco riordino e' un no-op secco.
        self._rv_active = (self.rv_reorder
                           and _articuno_in_play(o.current.players[1 - self.me]))
        try:
            ss = determinize(o, self.me, self.my_deck, self.opp_deck, rng=self.rng)
            line = None
            if pv is not None:
                line = self._dfs(ss, 0, hint=tuple(pv))
            else:
                if self.compose:
                    # PASSATA-1 (2026-07-16, blind-spot ep Kaggle 86275279): il DFS depth-first col
                    # compose pieno muore di budget nei sottoalberi play/attach PRIMA di arrivare alle
                    # linee corte tipo evolve Dusknoir -> Cursed Blast -> attacco+segnalini. Qui si
                    # cerca prima la LINEA MINIMA (ability/attach-Munki/evolve/attacco, rank-4 esclusi)
                    # con un tetto di 1/3 del budget; il budget e' CONDIVISO (self.steps non si
                    # azzera) -> il caso peggiore complessivo resta identico a prima.
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
                        # O4 (2026-07-20): la passata-1 e' terminata NATURALMENTE (niente tetto,
                        # niente branch rotti) senza aver MAI scartato un candidato rank-4
                        # (nessun PLAY whitelist in mano, nessuna attach-generica utile) -> la
                        # passata piena esplorerebbe un albero IDENTICO: None garantito, skip.
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
        """Solver per la ISMCTS (MCTS-Solver): DFS DENTRO un mondo GIA' aperto (SearchState),
        SENZA determinize e SENZA search_end — l'API engine libera tutti gli stati in blocco,
        chiamarlo qui distruggerebbe l'albero della search chiamante (che pulisce lei nel suo
        finally). True = il giocatore di turno in ss ha una vittoria DIMOSTRATA entro il suo
        turno corrente, nel mondo di ss."""
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
        # hint (O3): coda della linea-PV, un elemento per nodo-decisione registrato (None sui
        # nodi avversari). Solo ORDINAMENTO: si prova prima hint[0], e la coda passa al figlio
        # SOLO lungo il ramo ricordato; su divergenza il ramo prosegue senza hint.
        # avanza gli stati forzati (select vuote / a scelta unica)
        for _ in range(24):
            st = ss.observation.current
            if st.result >= 0:
                # vittoria valida SOLO se arriva NEL turno del lethal: una vittoria a turno
                # avanzato (es. deck-out avversario alla loro pescata) vincerebbe con QUALSIASI
                # linea, anche END -> non e' il lethal che stiamo cercando
                return [] if (st.result == self.me and int(st.turn) == self.root_turn) else None
            if int(st.turn) != self.root_turn:
                return None                      # turno finito senza vittoria: linea morta
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
                # alla MAIN considero SOLO i candidati in scope (rank = ordine di esplorazione:
                # ability 0, attach-setup 1, evolve 2, ATTACCO 3, play/retreat 4 — linea minima
                # preferita, i play entrano solo quando senza non si chiude)
                cands = []
                # O2 dedup (2026-07-12): due COPIE identiche in mano producono sottoalberi
                # identici (i serial non toccano le regole; la copia restante in mano e' lo
                # stesso multinsieme). Vale per ATTACH (stessa energia -> stesso bersaglio),
                # EVOLVE (stessa evoluzione -> stesso bersaglio) e PLAY (stessa carta).
                # ABILITY mai dedupata: corpi diversi sul board = stati diversi dopo self-KO.
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
                        # SETUP {D}->Munkidori (abilita Adrena-Brain, rank 1) oppure ATTACH
                        # GENERICA (2026-07-11): l'energia mancante dalla mano per pagare
                        # l'attacco che chiude (rank 4: esplorata solo se senza non si vince).
                        # NB il tipo energia sta in CARD_TABLE (l'istanza in mano ha energyType
                        # None -- bug scoperto col contatore: 0/241 match prima del fix)
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
                            # O4: passata-1, candidato rank-4 SCARTATO -> la passata piena
                            # non e' una fotocopia, niente skip
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
                                # O4: passata-1, PLAY whitelist in mano SCARTATO -> pass-2 serve
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
                        # ritirata PAGATA (l'engine la offre solo se il costo e' pagabile):
                        # porta davanti l'attaccante che chiude. Il cambio GRATIS via
                        # self-KO di Cursed Blast e' gia' coperto (promozione = follow-up).
                        # 2026-07-17 (ep Kaggle 86371054): la ritirata resta VISIBILE anche in
                        # PASSATA-1/minimal — i riarmi giravano minimal e la classe piu' banale
                        # (swap -> attacco -> ultimo premio) era invisibile. Il branching extra
                        # e' <= bench (5) e il tetto in nodi del pass-1 NON cambia. Restano
                        # esclusi dal pass-1 PLAY e attach-generica (la vera esplosione).
                        self.flags.add("compose_retreat_cand")
                        cands.append((4, ai))
                    elif t == int(OptionType.ATTACK):
                        cands.append((3, ai))
                cands.sort()
                idxs = [ai for _, ai in cands]
            else:
                # follow-up (target ability, distribuzione segnalini, ecc.): esploro tutto.
                # POTATURA distribuzione segnalini (2026-07-11, ep live 85341009 turno 11:
                # benchout triplo con split 2+7 trovato solo a 300k nodi, budget 12k in abort):
                # (a) no-overkill: mai esplorare piazzamenti su bersagli avversari gia' a 0hp
                #     (dominati ai fini del turno; il caso tutto-morto NON viene mascherato,
                #     la mask torna None e i segnalini vanno dove l'engine impone);
                # (b) ordine CANONICO: i segnalini di una stessa distribuzione commutano ->
                #     esploro solo sequenze con indice-opzione non decrescente (ogni
                #     allocazione UNA volta: C(n+k,k) invece di k^n percorsi). Se la potatura
                #     dei morti svuota il ramo canonico, fallback all'esplorazione piena.
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
                    # RIORDINO Repelling Veil (2026-07-21, ep Kaggle 87280612 t14/step182,
                    # miss diagnosticato dal probe TR: deploy `abort` su board con Articuno,
                    # oracle a 250k trova; qui un Spidops a 10hp NON protetto era la chiusura).
                    # SOLO se un Articuno avversario e' in gioco (board normali INVARIATE ->
                    # zero impatto su throughput/shadow). SOLO RIORDINO dell'ordine di
                    # esplorazione, mai potatura -> ogni multiset resta raggiungibile (l'ordine
                    # canonico enumera via soglia min_oi, non via ordine di iterazione):
                    # bersagli NON-protetti e a basso hp per primi = la KO-completion si trova
                    # PRIMA di esaurire il budget sui piazzamenti che fizzano sui Basic protetti.
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
                    # POTATURA PREMI (2026-07-17, ep Kaggle 86371563: hit-rate 3/10 mondi):
                    # la scelta di QUALI premi pescare e' fra carte NASCOSTE che il mondo
                    # determinizzato ha INVENTATO -> le opzioni sono equivalenti ai fini
                    # della linea, e una linea che DIPENDESSE dal premio pescato sarebbe
                    # comunque unsound (stessa filosofia del drop has_deck_search). UN solo
                    # ramo canonico: C(n,k) sottoalberi fotocopia diventano 1.
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
            # O3 PV-first: la mossa ricordata passa in testa (dopo mask/canon/cap: se il
            # filtro l'ha esclusa, la potatura vince — l'hint non riallarga mai l'albero)
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
        # nodo AVVERSARIO (es. promozione dopo KO a meta' turno): lethal solo se vinco comunque
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
            # PV: il nodo avversario occupa UNO slot della linea (None) — la coda va a
            # ogni figlio (nel forall la continuazione registrata e' quella del 1° figlio;
            # sugli altri l'hint puo' divergere e muore da solo, e' solo ordinamento)
            line = self._dfs(child, depth + 1, hint=(hint[1:] if hint else None))
            if line is None:
                return None
            if first is None:
                first = [(None, "<avv: qualsiasi scelta>")] + line
        return first
