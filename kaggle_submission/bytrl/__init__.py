"""ByteRL/OSFP-inspired Pokémon-TCG agent (v2).

Self-contained package (no dependency on the legacy `ml/` package) so it can be bundled for Kaggle
on its own. Ports the validated internals (engine spec, encoding, V-Trace/UPGO/PPO loss, imitation)
and adds the new pieces (recurrent memory, faithful OSFP, archetype belief).

See ../docs/bytrl-paper-study.md and ../docs/bytrl-adaptation.md.
"""
