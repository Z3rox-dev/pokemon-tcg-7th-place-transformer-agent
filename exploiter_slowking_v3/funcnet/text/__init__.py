"""funcnet.text — Stage 2 offline card-KNOWLEDGE cache (effect semantics).

Complements funcnet/features.py (Stage 1: static numeric features — hp/type/cost/etc.) with two
extra, STATIC-per-card views of the EFFECT SEMANTICS that the numeric table cannot express:

  (1) a STRUCTURED-SCHEMA multi-hot vector distilled from each card's ability/attack TEXT by an
      offline LLM interpreter           -> funcnet.text.schema_spec / out/card_schema.jsonl
  (2) a sentence-transformer TEXT EMBEDDING of the concatenated card text (build-time only)
                                        -> funcnet.text.embed / out/card_text_emb.npy

Both are precomputed ONCE at build time and committed as data artifacts; at deploy/runtime they are
accessed by pure array lookup (NO LLM, NO network, NO transformer forward) via funcnet.text.tables.

This subpackage is purely ADDITIVE — it does not import or modify the live-training modules
(model/losses/encoding/osfp/selfplay/evaluate). Integration into the net is out of scope here.
"""
