"""Mirror Python del conto-energie dell'engine (InsufficientEnergyCount, engine_src/GameUtil.h:28) per la
feature NRG-TAIL del decoder: "quante energie MANCANO a questo pokemon per pagare i suoi attacchi, con i
modificatori attivi ORA". La rete oggi deve dedurre la pagabilita' dal testo (4 hop cross-modulo: parola-
stadio -> testo tassa -> costo stampato -> energie attaccate); qui il conto deterministico lo fa il codice
e alla rete resta la strategia — stesso principio del danno risolto (_fill_attack_resolved) e del lethal.

NON e' una maschera: non vieta niente (il cap secco Regola-3 al deploy era -6pp, la rete "insiste" e il
redirect cade peggio — memoria energy-strict-cap-deploy-ab). E' informazione: l'over-attach legittimo
(valuta-ritirata, attacchi scalanti, pre-carica) resta scelto dal gradiente.

COPERTURA DICHIARATA (mirror PARZIALE, verificato engine-exact da funcnet4/verify_encoding X9/X10/X11):
  - matching TIPATO esatto: tipi 1..9 scalano il requisito del proprio tipo, il resto finisce nel bucket
    colorless (stessa passata greedy dell'engine, GameUtil.h:53-69);
  - RAINBOW (EnergyType 10, C++ ::All) = jolly generico (sottratto in coda, GameUtil.h:110);
  - TEAM_ROCKET (EnergyType 11, C++ Psychic|Darkness) = prova {P}, poi {D}, poi colorless
    (stesso ordine dell'engine, GameUtil.h:89-107);
  - AttackCostChangeColorless, TUTTE le sorgenti del pool (somma unica, un branch sul segno come
    GameUtil.h:74-80):
      * Nighttime Mine (1266, stadio): +1 se il pokemon e' TERA (CardImpl.h:12528);
      * Antique Root Fossil (1099, ability dall'ATTIVO avversario): +1 se il MIO pokemon e' Basic
        (CardImpl.h:3446);
      * Counter Gain (1168, tool): -1 se i MIEI premi rimanenti > premi avversari (CardImpl.h:3770);
      * Hop's Choice Band (1171, tool): -1 se il pokemon e' un "Hop's" (CardImpl.h:3790);
      * Decidueye ex (1022, ability propria): -99 (= azzera i {C}) se l'avversario ha ESATTAMENTE
        4 carte in mano (CardImpl.h:11792).
FUORI SCOPO (misurati dal harness, non specchiati): thisTurn.attackCostChange (effetti one-shot non
leggibili dall'Observation), attackCostDown/OwnAttack, i branch fixEnergy (noEnergyIfSpecialCondition /
attackEnergyColoressOne / darkness1IfDamaged — flag non esposti dall'api Attack), ability-lock che
spegnerebbero Root Fossil/Decidueye, attacchi COPIATI da altri mon.

Stato corrente: `poke.energies` e' gia' la lista PROVIDED espansa dall'engine (le speciali multi-provide
sono gia' risolte) -> nessuna mappa carta->provide serve per lo stato attuale. Per l'attach IPOTETICO la
provide base della carta-energia e' CardData.energyType (verificato sul corpus: Legacy=RAINBOW, Grow
Grass=GRASS, Telepath=PSYCHIC, Rock Fighting=FIGHTING, le altre=COLORLESS); gli upgrade condizionali
(Prism su Basic, Neo Upper su Stage2) NON sono specchiati -> per quelle carte mancanti sovrastimato di
<=1 sull'attach (mai sottostimato).

Modulo SENZA env (vincolo bundle Kaggle, come lethal_dragapult_dusknoir): il flag FUNCNET4_NRG_FEAT
vive nel chiamante (funcnet4/spec.py -> encoding)."""
from funcnet.spec import CARD_TABLE
from funcnet.features import ATTACKS

NIGHTTIME_MINE = 1266
ANTIQUE_ROOT_FOSSIL = 1099
COUNTER_GAIN = 1168
HOPS_CHOICE_BAND = 1171
DECIDUEYE_EX = 1022
_ENERGY_CARDTYPES = (5, 6)        # Basic / Special Energy (cardType) — stesso guard della Regola-3
_POKEMON = 0                      # CardType.POKEMON
_C = 0                            # EnergyType.COLORLESS
_P = 5                            # EnergyType.PSYCHIC
_D = 7                            # EnergyType.DARKNESS
_RAINBOW = 10                     # EnergyType.RAINBOW  (C++ EnergyType::All, jolly)
_PD = 11                          # EnergyType.TEAM_ROCKET (C++ Psychic|Darkness)


def nm_in_play(state) -> bool:
    """True se Nighttime Mine e' lo stadio in campo (obs.current.stadium, condiviso)."""
    return any(c is not None and int(getattr(c, "id", -1) or -1) == NIGHTTIME_MINE
               for c in (getattr(state, "stadium", None) or []))


def cost_change_ctx(state, your_index):
    """Pre-calcola UNA volta per decisione lo stato dei modificatori di costo leggibile dall'Observation
    (obs.current). Ritorna il dict consumato da _mon_cost_change."""
    my = state.players[your_index]
    opp = state.players[1 - your_index]
    opp_act = opp.active[0] if getattr(opp, "active", None) else None
    return {
        "nm": nm_in_play(state),
        "root": (getattr(opp_act, "id", None) == ANTIQUE_ROOT_FOSSIL),      # tassa +1 sui MIEI Basic
        "prize_gain": len(getattr(my, "prize", None) or []) > len(getattr(opp, "prize", None) or []),
        "opp_hand4": int(getattr(opp, "handCount", -1)) == 4,               # Sniper's Eye Decidueye
    }


def _mon_cost_change(cd, tmon, ctx) -> int:
    """attackCostChangeColorless del mon (somma di TUTTE le sorgenti attive, come il campo continual
    dell'engine, Card.h:278). cd = CARD_TABLE[tmon.id]."""
    ch = 0
    if ctx["nm"] and getattr(cd, "tera", False):
        ch += 1                                            # Nighttime Mine: +{C} ai Tera
    if (ctx["root"] and int(getattr(cd, "cardType", -1) or -1) == _POKEMON
            and not getattr(cd, "stage1", False) and not getattr(cd, "stage2", False)):
        ch += 1                                            # Primal Root: +{C} ai Basic avversari del fossile
    for t in (getattr(tmon, "tools", None) or []):
        tid = getattr(t, "id", None)
        if tid == COUNTER_GAIN and ctx["prize_gain"]:
            ch -= 1                                        # Counter Gain: -{C} se ho piu' premi rimanenti
        elif tid == HOPS_CHOICE_BAND and str(getattr(cd, "name", "")).startswith("Hop"):
            ch -= 1                                        # Hop's Choice Band: -{C} sui "Hop's"
    if getattr(tmon, "id", None) == DECIDUEYE_EX and ctx["opp_hand4"]:
        ch -= 99                                           # Sniper's Eye: azzera i {C}
    return ch


def provided_types(cid):
    """Provide BASE della carta-energia `cid` per l'attach ipotetico: [energyType] (1 unita').
    None se cid non e' una carta energia (il chiamante scrive zeri)."""
    c = CARD_TABLE.get(cid)
    if c is None or int(getattr(c, "cardType", -1) or -1) not in _ENERGY_CARDTYPES:
        return None
    et = getattr(c, "energyType", None)
    return [int(et)] if et is not None else [_C]


def attack_missing(required, provided, cost_change=0) -> int:
    """Mirror engine-exact (scope in testa al modulo) di InsufficientEnergyCount: quante energie mancano
    per pagare `required` (lista EnergyType del costo stampato) con `provided` (lista EnergyType
    attaccate/provided). cost_change = attackCostChangeColorless del mon (somma sorgenti, un solo branch
    sul segno come GameUtil.h:74-80)."""
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
            pool += 1                    # tipata che non serve -> bucket colorless (GameUtil.h:66)
    if cost_change > 0:
        req_c += cost_change             # GameUtil.h:78 requiredTable.colorless += change
    elif cost_change < 0:
        pool += -cost_change             # GameUtil.h:76 colorlessCount -= change
    for _ in range(pd_n):                # P|D DOPO la passata tipata, prova P poi D (GameUtil.h:89-107)
        if req.get(_P, 0) > 0:
            req[_P] -= 1
        elif req.get(_D, 0) > 0:
            req[_D] -= 1
        else:
            pool += 1
    need = sum(req.values()) + req_c - all_n - min(pool, req_c)   # GameUtil.h:112
    return max(0, need)


def attack_missing_for_mon(tmon, attack, ctx, extra_energy_cid=None):
    """`mancanti` per UN attacco di `tmon` (istanza Pokemon dell'Observation) coi modificatori correnti.
    extra_energy_cid: conta ANCHE l'attach ipotetico di quella carta-energia. None se il mon non e' in
    tabella o extra_energy_cid non e' un'energia."""
    cd = CARD_TABLE.get(getattr(tmon, "id", None))
    if cd is None:
        return None
    provided = [int(e) for e in (getattr(tmon, "energies", None) or [])]
    if extra_energy_cid is not None:
        extra = provided_types(extra_energy_cid)
        if extra is None:
            return None                  # non-energia (tool via ATTACH): niente feature
        provided = provided + extra
    ch = _mon_cost_change(cd, tmon, ctx)
    return attack_missing(getattr(attack, "energies", None), provided, ch)


def mon_missing_profile(tmon, ctx, extra_energy_cid=None):
    """Lista dei `mancanti` per OGNI attacco stampato di `tmon`, coi modificatori correnti (vedi
    cost_change_ctx). None se il mon/i suoi attacchi non sono in tabella (feature a zero)."""
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
    """I 3 scalari NRG-TAIL per un'azione ATTACH di `ecard_cid` a `tmon`:
      (miss_min_after/4, miss_max_after/4, useful) — useful=1 se l'attach RIDUCE il mancante di almeno un
    attacco (tipato: un'energia del tipo sbagliato che non aiuta niente = 0, classe-Munkidori).
    None -> il chiamante lascia la slice a zero (non-energia / mon sconosciuto / niente attacchi)."""
    before = mon_missing_profile(tmon, ctx)
    if before is None:
        return None
    after = mon_missing_profile(tmon, ctx, extra_energy_cid=ecard_cid)
    if after is None or len(after) != len(before):
        return None
    useful = 1.0 if any(a < b for a, b in zip(after, before)) else 0.0
    return (min(after) / 4.0, max(after) / 4.0, useful)
