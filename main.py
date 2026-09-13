import os, sys
for _p in (os.getcwd(), "/kaggle_simulations/agent"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["FUNCNET_N_HAND_TOKENS"] = "12"
os.environ.setdefault("FUNCNET4_CKPT", "model.pth")
os.environ.setdefault("FUNCNET4_ENC", "8")
os.environ.setdefault("FUNCNET4_DEC", "8")
os.environ.setdefault("FUNCNET4_HEADS", "4")
for _flag in ("FUNCNET4_NRG_FEAT", "FUNCNET4_CNT_FEAT", "FUNCNET4_UNLOCK_FEAT",
              "FUNCNET4_DKR_FEAT", "FUNCNET4_PRZ_FEAT"):
    os.environ[_flag] = "1"
os.environ["FUNCNET4_TGT_FEAT"] = "1"
os.environ["FUNCNET4_BENCH_FEAT"] = "1"
os.environ["FUNCNET4_LOC_FEAT"] = "1"
os.environ["FUNCNET4_FEZ_MASK"] = "1"
os.environ["FUNCNET4_ENERGY_STRICT"] = "0"
from funcnet4_agent import agent  # noqa: E402
__all__ = ["agent"]
