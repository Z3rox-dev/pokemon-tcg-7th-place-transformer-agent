import os, sys
for _p in (os.getcwd(), "/kaggle_simulations/agent"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["FUNCNET_N_HAND_TOKENS"] = "12"
os.environ.setdefault("FUNCNET4_CKPT", "model.pth")
os.environ.setdefault("FUNCNET4_ENC", "8")
os.environ.setdefault("FUNCNET4_DEC", "8")
os.environ.setdefault("FUNCNET4_HEADS", "4")
# Profilo del checkpoint corrente. Questi flag governano i VALORI scritti nell'encoding; DKR aggiunge
# anche due parole e porta N_ROLE 13->15. Vanno fissati prima dell'import di funcnet4.spec/agent.
for _flag in ("FUNCNET4_NRG_FEAT", "FUNCNET4_CNT_FEAT", "FUNCNET4_UNLOCK_FEAT",
              "FUNCNET4_DKR_FEAT", "FUNCNET4_PRZ_FEAT"):
    os.environ[_flag] = "1"
# TGT: pinnata dal builder (vedi testa dello script) — 0 di default, 1 solo per ckpt TGT-armati.
os.environ["FUNCNET4_TGT_FEAT"] = "1"
# BENCH: come TGT, 0 di default. Armata SOLO per ckpt allenati con la coda accesa —
# e con un occhio al budget-tempo per mossa: la query apre un mondo nel motore.
os.environ["FUNCNET4_BENCH_FEAT"] = "1"
# LOC: come TGT/BENCH, ma questo flag decide la LARGHEZZA della riga encoder (1109/1145). Un
# disallineamento non degrada in silenzio: la migrazione rifiuta il crop e il bundle esplode.
os.environ["FUNCNET4_LOC_FEAT"] = "1"
# 2026-08-03: queste due erano lasciate ai DEFAULT del codice (FEZ_MASK=1, ENERGY_STRICT=0), che per
# fortuna coincidevano col training — ma era fortuna, non progetto. Lo stesso schema (una lista di
# flag da tenere allineata a mano con un'altra) ha falsato il pannello payoff per un giorno intero,
# facendo sembrare la rete crollata all'8% quando faceva 82%. Qui si PINNANO: la politica spedita
# deve stare scritta nel bundle, non dedotta da un default che qualcuno puo' cambiare.
os.environ["FUNCNET4_FEZ_MASK"] = "1"        # Regola 6 Fezandipiti ATTIVA (come nel training)
os.environ["FUNCNET4_ENERGY_STRICT"] = "0"   # cap secco energie SPENTO (come nel training)
os.environ["FUNCNET4_ESEARCH"] = "0"
os.environ["FUNCNET4_ESEARCH_MODE"] = "energy"
os.environ.setdefault("FUNCNET4_ESEARCH_BUDGET_S", "590")  # il budget di match dichiarato
os.environ.setdefault("FUNCNET4_ESEARCH_SAFETY", "0.70")
os.environ.setdefault("FUNCNET4_ESEARCH_CAND", "3")
os.environ.setdefault("FUNCNET4_ESEARCH_WORLDS", "16")  # tetto: il regolatore puo' scendere
os.environ.setdefault("FUNCNET4_ESEARCH_TRUNC", "30")
os.environ.setdefault("FUNCNET4_ESEARCH_TRUNC_TURNS", "0")
os.environ.setdefault("FUNCNET4_ESEARCH_CT_MINREMAIN", "1")
os.environ.setdefault("FUNCNET4_ESEARCH_MINZ", "2.0")      # il cancello: senza, la search PERDE
# fine-partita: premi <=2 di UNO dei due OPPURE turno >=12, max 5 ricerche, pavimento duro 15 s
os.environ.setdefault("FUNCNET4_ESEARCH_EG_PRIZES", "2")
os.environ.setdefault("FUNCNET4_ESEARCH_EG_TURN", "12")
os.environ.setdefault("FUNCNET4_ESEARCH_EG_MAX", "5")
os.environ.setdefault("FUNCNET4_ESEARCH_HARD_FLOOR_S", "15")
os.environ.setdefault("FUNCNET4_ESEARCH_BATCH", "25")   # mondi per lotto: limita il tempo PER COSTRUZIONE
os.environ.setdefault("FUNCNET4_ESEARCH_LOG", "1")      # telemetria [es] su stderr: flip/gated/agree/tagli per decisione
os.environ.setdefault("FUNCNET4_ESEARCH_EXPECTED", "37")  # divisore del budget per-decisione
from funcnet4_agent import agent  # noqa: E402
__all__ = ["agent"]
