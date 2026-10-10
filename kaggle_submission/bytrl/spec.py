"""Shared encoding/model dimensions, derived from the engine card pool (ported from ml/spec.py).

Single source of truth for EmbeddingBag vocab sizes so encoding.py and model.py agree. ALL sizes are
derived dynamically from the engine (`all_card_data()`/`all_attack()`) — never hardcode card counts
(verified: 1267 unique cards → card_count=1268, 1556 attacks). See memory/engine-card-pool.md.
"""
import sys
import pathlib

try:
    import cg
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "engine"))
    import cg
from cg.api import all_card_data, all_attack, SelectContext

ALL_CARDS = all_card_data()
CARD_TABLE = {c.cardId: c for c in ALL_CARDS}

card_count = max(ALL_CARDS, key=lambda c: c.cardId).cardId + 1            # max card ID + 1 (=1268)
attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1   # max attack ID + 1 (=1557)

num_words_encoder = 24          # number of "words" the encoder packs the state into
encoder_size = 22000            # encoder EmbeddingBag vocab (exceeds real feature count)

decoder_main_feature = 8        # SelectContext.Main feature count
decoder_attack_offset = 14      # first index of attack feature
decoder_card_offset = decoder_attack_offset + attack_count
decoder_size = decoder_card_offset + (1 + decoder_main_feature + int(SelectContext.RECOVER_SPECIAL_CONDITION)) * card_count
