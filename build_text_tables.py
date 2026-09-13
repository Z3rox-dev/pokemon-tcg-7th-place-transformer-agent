"""Rebuild the text-embedding tables the agent reads, from your own copy of the competition engine.

    python build_text_tables.py

Writes out/card_text_emb.npy (1268 x 768) and out/attack_text_emb.npy (1557 x 768) with the
sentence-transformer all-mpnet-base-v2. Needs the engine package in ./cg and sentence-transformers.

Every text comes from the engine. Batch sizes match the original build: with sentence-transformers 3.4.1 on
the CPU that built the submitted tables, the output is identical bit for bit to them. Other CPUs or library
versions can differ in the last bits of the floats.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from cg.api import all_attack, all_card_data  # noqa: E402
from bytrl.spec import CARD_TABLE, attack_count, card_count  # noqa: E402
from funcnet.features import ATTACKS  # noqa: E402

MODEL = "all-mpnet-base-v2"
BASIC_ENERGY = {1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning",
                5: "Psychic", 6: "Fighting", 7: "Darkness", 8: "Metal"}
TYPE_NAME = {0: "Colorless", 1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning", 5: "Psychic",
             6: "Fighting", 7: "Darkness", 8: "Metal", 9: "Dragon"}


def clean(text):
    return str(text).strip() if text else ""


def encode(model, texts, **kw):
    out = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, **kw)
    return np.asarray(out, dtype=np.float32)


def card_blob(card, attacks):
    """Name plus ability and attack text, or '' when the card carries no effect text."""
    abilities = [(clean(getattr(s, "name", "")), clean(getattr(s, "text", ""))) for s in (card.skills or [])]
    atk = []
    for aid in (card.attacks or []):
        a = attacks.get(aid)
        if a is not None:
            atk.append((clean(getattr(a, "name", "")), clean(getattr(a, "text", "")),
                        int(getattr(a, "damage", 0) or 0)))
    if not any(text for _, text in abilities) and not any(text for _, text, _ in atk):
        return ""
    parts = [clean(card.name)]
    for name, text in abilities:
        if text:
            parts.append("Ability: " + " ".join(p for p in (name, text) if p))
    for name, text, dmg in atk:
        bits = [name] + ([f"({dmg} damage)"] if dmg else []) + ([text] if text else [])
        if text or dmg:
            parts.append("Attack: " + " ".join(b for b in bits if b))
    return "\n".join(p for p in parts if p).strip()


def build_card_table(model):
    attacks = {a.attackId: a for a in all_attack()}
    table = np.zeros((card_count, 768), dtype=np.float32)

    texts, ids = [], []
    for card in sorted(all_card_data(), key=lambda c: c.cardId):
        blob = card_blob(card, attacks)
        if blob:
            texts.append(blob)
            ids.append(int(card.cardId))
    for cid, vec in zip(ids, encode(model, texts, batch_size=64)):
        if 0 <= cid < card_count:
            table[cid] = vec

    # Basic energies carry no effect text; without their own row every type would read identically.
    vecs = encode(model, [f"Basic {t} Energy. Provides one {t} Energy." for t in BASIC_ENERGY.values()])
    for cid, vec in zip(BASIC_ENERGY, vecs):
        table[cid] = vec

    # Same for Pokemon without effect text: describe them by stage, type, HP and attacks.
    texts, ids = [], []
    for cid, c in CARD_TABLE.items():
        if table[cid].any() or c.hp is None:
            continue
        stage = "Basic" if c.basic else ("Stage 1" if c.stage1 else ("Stage 2" if c.stage2 else ""))
        kind = TYPE_NAME.get(int(c.energyType) if c.energyType is not None else -1, "")
        moves = ", ".join(f"{ATTACKS[aid].name} {ATTACKS[aid].damage or 0}"
                          for aid in (c.attacks or []) if aid in ATTACKS)
        texts.append(f"{c.name}. {stage} {kind} Pokemon, {c.hp} HP. Attacks: {moves}.".replace("  ", " "))
        ids.append(cid)
    for cid, vec in zip(ids, encode(model, texts, batch_size=64)):
        table[cid] = vec
    return table


def build_attack_table(model):
    table = np.zeros((attack_count, 768), dtype=np.float32)
    texts, ids = [], []
    for a in all_attack():
        aid = int(getattr(a, "attackId", -1))
        if not 0 <= aid < attack_count:
            continue
        dmg = int(getattr(a, "damage", 0) or 0)
        bits = [clean(getattr(a, "name", ""))] + ([f"({dmg} damage)"] if dmg else [])
        bits += [clean(getattr(a, "text", ""))]
        blob = " ".join(b for b in bits if b).strip()
        if blob:
            texts.append(blob)
            ids.append(aid)
    for aid, vec in zip(ids, encode(model, texts, batch_size=64)):
        table[aid] = vec
    return table


def main():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cpu")
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    for name, build in (("card_text_emb.npy", build_card_table), ("attack_text_emb.npy", build_attack_table)):
        table = build(model)
        np.save(os.path.join(ROOT, "out", name), table)
        print(f"out/{name}: {table.shape}, {int(table.any(axis=1).sum())} non-zero rows")


if __name__ == "__main__":
    main()
