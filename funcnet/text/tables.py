"""Per-card and per-attack lookup tables, read once at import.

  out/card_text_emb.npy      (card_count, 768)             rebuilt by build_text_tables.py
  out/card_atom_feats.npy    (card_count, ATOM_FEAT_DIM)   effect features, cannot be rebuilt
  out/attack_atom_feats.npy  (attack_count, ATOM_FEAT_DIM) effect features, cannot be rebuilt

A missing file gives an all-zero table and one log line. Lookups never raise: an unknown or out-of-range id
returns a zero vector.
"""
import pathlib

import numpy as np

from funcnet.spec import card_count, attack_count
from funcnet.text.atom_spec import ATOM_FEAT_DIM

DEFAULT_EMB_DIM = 768

_warned_emb = set()
_load_warned = set()


def _find(fname: str):
    """Locate a data file in out/, next to the package, in the working directory or the Kaggle agent dir."""
    here = pathlib.Path(__file__).resolve()
    candidates = [
        here.parents[2] / "out" / fname,
        here.parent / fname,
        here.parents[1] / fname,
        pathlib.Path.cwd() / fname,
        pathlib.Path.cwd() / "out" / fname,
        pathlib.Path("/kaggle_simulations/agent") / fname,
    ]
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def _log_once(key, msg):
    if key not in _load_warned:
        _load_warned.add(key)
        print(msg, flush=True)


def _load_emb_table():
    path = _find("card_text_emb.npy")
    if path is None:
        _log_once("emb_missing",
                  f"[funcnet.text.tables] card_text_emb.npy not found; text embeddings = zeros "
                  f"(D={DEFAULT_EMB_DIM})")
        return np.zeros((card_count, DEFAULT_EMB_DIM), dtype=np.float32)
    try:
        arr = np.load(path).astype(np.float32, copy=False)
    except (OSError, ValueError) as e:
        _log_once("emb_bad", f"[funcnet.text.tables] failed to load {path.name} ({e}); embeddings = zeros")
        return np.zeros((card_count, DEFAULT_EMB_DIM), dtype=np.float32)
    if arr.ndim != 2:
        _log_once("emb_shape", f"[funcnet.text.tables] {path.name} has ndim={arr.ndim}; embeddings = zeros")
        return np.zeros((card_count, DEFAULT_EMB_DIM), dtype=np.float32)
    d = arr.shape[1]
    if arr.shape[0] != card_count:
        fixed = np.zeros((card_count, d), dtype=np.float32)
        m = min(card_count, arr.shape[0])
        fixed[:m] = arr[:m]
        arr = fixed
        _log_once("emb_rows", f"[funcnet.text.tables] {path.name} rows != card_count; padded/trimmed")
    _log_once("emb_ok", f"[funcnet.text.tables] loaded embeddings {arr.shape} from {path.name}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _load_npy_table(fname, n_rows, dim, key):
    """Load an (n_rows, dim) float32 table; zeros and one log line if missing, unreadable or mis-shaped."""
    path = _find(fname)
    if path is None:
        _log_once(key + "_missing", f"[funcnet.text.tables] {fname} not found; {key} = zeros")
        return np.zeros((n_rows, dim), dtype=np.float32)
    try:
        arr = np.load(path).astype(np.float32, copy=False)
    except (OSError, ValueError) as e:
        _log_once(key + "_bad", f"[funcnet.text.tables] failed to load {fname} ({e}); {key} = zeros")
        return np.zeros((n_rows, dim), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        _log_once(key + "_shape", f"[funcnet.text.tables] {fname} shape {arr.shape} != (*,{dim}); {key} = zeros")
        return np.zeros((n_rows, dim), dtype=np.float32)
    if arr.shape[0] != n_rows:
        fixed = np.zeros((n_rows, dim), dtype=np.float32)
        m = min(n_rows, arr.shape[0]); fixed[:m] = arr[:m]; arr = fixed
        _log_once(key + "_rows", f"[funcnet.text.tables] {fname} rows != {n_rows}; padded/trimmed")
    _log_once(key + "_ok", f"[funcnet.text.tables] loaded {key} {arr.shape} from {path.name}")
    return np.ascontiguousarray(arr, dtype=np.float32)


EMB_TABLE = _load_emb_table()
EMB_DIM = int(EMB_TABLE.shape[1])
ATOM_TABLE = _load_npy_table("card_atom_feats.npy", card_count, ATOM_FEAT_DIM, "atom")
ATK_ATOM_TABLE = _load_npy_table("attack_atom_feats.npy", attack_count, ATOM_FEAT_DIM, "atk_atom")
_ATOM_ZERO = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
_EMB_ZERO = np.zeros(EMB_DIM, dtype=np.float32)


def text_embedding(cid: int) -> np.ndarray:
    """Text embedding of a card, (EMB_DIM,); zeros and one log line for an unknown id."""
    if cid is None or cid < 0 or cid >= card_count:
        if cid not in _warned_emb:
            _warned_emb.add(cid)
            print(f"[funcnet.text.tables] embedding id {cid} out of range [0,{card_count}); using zeros",
                  flush=True)
        return _EMB_ZERO
    return EMB_TABLE[cid]


def atom_features(cid: int) -> np.ndarray:
    """Effect-atom features of a card, (ATOM_FEAT_DIM,); zeros for an unknown id."""
    if cid is None or cid < 0 or cid >= card_count:
        return _ATOM_ZERO
    return ATOM_TABLE[cid]


def attack_atom_features(aid: int) -> np.ndarray:
    """Effect-atom features of an attack, (ATOM_FEAT_DIM,); zeros for an unknown id."""
    if aid is None or aid < 0 or aid >= attack_count:
        return _ATOM_ZERO
    return ATK_ATOM_TABLE[aid]
