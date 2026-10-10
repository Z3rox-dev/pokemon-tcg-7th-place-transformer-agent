"""DEPLOY-SAFE lookups for the Stage-2 cached card-knowledge arrays.

Loads the committed build-time artifacts ONCE at import:
  out/card_schema.jsonl   -> SCHEMA_TABLE (card_count x SCHEMA_FEAT_DIM)  via schema_spec.schema_row
  out/card_text_emb.npy   -> EMB_TABLE    (card_count x EMB_DIM)          sentence-transformer embeddings

and exposes pure array lookups:
  schema_features(cid) -> (SCHEMA_FEAT_DIM,) float32
  text_embedding(cid)  -> (EMB_DIM,)        float32

NO LLM, NO network, NO transformer forward ever happens here — only file reads at import and array
indexing at call time. Both lookups are bounds-checked and NEVER raise on an unknown / out-of-range id:
they return a zero vector and log once (mirrors funcnet/features.py.card_features exactly), so an
unseen-set id during a real match degrades gracefully instead of crashing.

If a data file is missing at deploy, the corresponding table is all-zeros (graceful no-op) and a single
warning is logged — the net still runs, just without that view.
"""
import io
import os
import json
import pathlib

import numpy as np

from funcnet.spec import card_count, attack_count
from funcnet.text.schema_spec import SCHEMA_FEAT_DIM, schema_row
from funcnet.text.atom_spec import ATOM_FEAT_DIM

DEFAULT_EMB_DIM = 768  # all-mpnet-base-v2; only used as the zero-table width if the .npy is absent

_warned_schema = set()
_warned_emb = set()
_load_warned = set()


def _find(fname: str):
    """Locate a committed data file across dev + deploy layouts. Returns a Path or None.
    Search order: repo out/ (dev), package dir, cwd, Kaggle agent dir. No network."""
    here = pathlib.Path(__file__).resolve()
    candidates = [
        here.parents[2] / "out" / fname,         # <repo>/out/  (dev)
        here.parent / fname,                       # funcnet/text/<fname> (if bundled beside code)
        here.parents[1] / fname,                   # funcnet/<fname>
        pathlib.Path.cwd() / fname,                # cwd
        pathlib.Path.cwd() / "out" / fname,        # cwd/out
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


def _load_schema_table():
    table = np.zeros((card_count, SCHEMA_FEAT_DIM), dtype=np.float32)
    path = _find("card_schema.jsonl")
    if path is None:
        _log_once("schema_missing",
                  "[funcnet.text.tables] card_schema.jsonl not found; schema features = zeros")
        return table
    n = 0
    for ln in io.open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        cid = row.get("cardId")
        if cid is None:
            continue
        cid = int(cid)
        if 0 <= cid < card_count:
            table[cid] = schema_row(row)
            n += 1
    _log_once("schema_ok", f"[funcnet.text.tables] loaded schema for {n} cards from {path.name}")
    return table


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
    # normalize shape to (card_count, D): pad/trim rows defensively so lookups stay in-bounds
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
    """Generic deploy-safe (n_rows, dim) float32 .npy loader: zeros + log-once if missing/bad/mis-shaped."""
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


SCHEMA_TABLE = _load_schema_table()
EMB_TABLE = _load_emb_table()
EMB_DIM = int(EMB_TABLE.shape[1])
# Effect-atom functional fingerprints (the rebuilt schema): per-card (encoder + PLAY/ABILITY) and per-attack
# (decoder ATTACK token). Replace the old card-schema view in the encoder; deploy-safe lookups by id.
ATOM_TABLE = _load_npy_table("card_atom_feats.npy", card_count, ATOM_FEAT_DIM, "atom")
ATK_ATOM_TABLE = _load_npy_table("attack_atom_feats.npy", attack_count, ATOM_FEAT_DIM, "atk_atom")
_ATOM_ZERO = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)

_SCHEMA_ZERO = np.zeros(SCHEMA_FEAT_DIM, dtype=np.float32)
_EMB_ZERO = np.zeros(EMB_DIM, dtype=np.float32)


def schema_features(cid: int) -> np.ndarray:
    """Bounds-checked structured-effect feature lookup. Returns a (SCHEMA_FEAT_DIM,) float32 row; zeros +
    log-once if the id is unknown / out of range (deploy safety — never crash mid-match)."""
    if cid is None or cid < 0 or cid >= card_count:
        if cid not in _warned_schema:
            _warned_schema.add(cid)
            print(f"[funcnet.text.tables] schema id {cid} out of range [0,{card_count}); using zeros",
                  flush=True)
        return _SCHEMA_ZERO
    return SCHEMA_TABLE[cid]


def text_embedding(cid: int) -> np.ndarray:
    """Bounds-checked text-embedding lookup. Returns an (EMB_DIM,) float32 row; zeros + log-once if the
    id is unknown / out of range (deploy safety — never crash mid-match)."""
    if cid is None or cid < 0 or cid >= card_count:
        if cid not in _warned_emb:
            _warned_emb.add(cid)
            print(f"[funcnet.text.tables] embedding id {cid} out of range [0,{card_count}); using zeros",
                  flush=True)
        return _EMB_ZERO
    return EMB_TABLE[cid]


def atom_features(cid: int) -> np.ndarray:
    """Bounds-checked per-CARD effect-atom fingerprint (ATOM_FEAT_DIM,). Zeros for unknown id (deploy-safe)."""
    if cid is None or cid < 0 or cid >= card_count:
        return _ATOM_ZERO
    return ATOM_TABLE[cid]


def attack_atom_features(aid: int) -> np.ndarray:
    """Bounds-checked per-ATTACK effect-atom fingerprint (ATOM_FEAT_DIM,). Zeros for unknown id (deploy-safe)."""
    if aid is None or aid < 0 or aid >= attack_count:
        return _ATOM_ZERO
    return ATK_ATOM_TABLE[aid]
