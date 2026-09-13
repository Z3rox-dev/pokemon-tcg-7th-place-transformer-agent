"""funcnet4 deploy-safe per-ATTACK text-embedding lookup (twin of funcnet.text.tables.text_embedding for
cards). Reuses the SAME loader + EMB_DIM as the shared tables; gives the ~190 zero-atom attacks an identity
once the decoder's per-attack id-embedding is gone. Build the .npy with funcnet4/build_attack_text_emb.py."""
import numpy as np

from funcnet.spec import attack_count
from funcnet.text.tables import EMB_DIM, _load_npy_table

ATK_EMB_TABLE = _load_npy_table("attack_text_emb.npy", attack_count, EMB_DIM, "atk_emb")
_EMB_ZERO = np.zeros(EMB_DIM, dtype=np.float32)


def attack_text_embedding(aid: int) -> np.ndarray:
    """Bounds-checked per-ATTACK text embedding (EMB_DIM,). Zeros for unknown id (deploy-safe)."""
    if aid is None or aid < 0 or aid >= attack_count:
        return _EMB_ZERO
    return ATK_EMB_TABLE[aid]
