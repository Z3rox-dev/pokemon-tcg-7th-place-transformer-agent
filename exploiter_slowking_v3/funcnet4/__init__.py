"""funcnet4 — the TEXT-FIRST project (FuncNet-TF).

A clean, isolated home for the text-first redesign: a card is described by WHAT IT DOES (precomputed
func + effect-atom + text-embedding features) read by SHARED MLPs, with small role/plane tag embeddings —
NO per-card id-embedding tables (the memorization the old funcnet/ used). The goal is generalization to
unseen decks/cards.

This package REUSES (by read-only import) the shared, verified foundation and does NOT modify the old
project:
  * the engine bindings (engine/cg)            — game simulation
  * bytrl/ (RL math, the transformer trunk base, base vocab dims)
  * funcnet.text.tables / funcnet.features      — the precomputed atom/text/func DATA tables + lookups
  * funcnet.encoding._fill_* helpers            — the verified per-card/per-action dense feature extractors

The old funcnet/ package is left exactly as it was (the working s7/s9 system), preserved for reference.
"""
