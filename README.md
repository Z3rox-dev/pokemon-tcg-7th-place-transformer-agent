# Pokémon TCG AI Battle — 7th place agent

The agent of submission 55565305 (7th of 6807, score 1226.6): a 10.7M-parameter transformer with a GRU
belief core that scores the legal actions the engine offers at each decision.

This repository holds only the final version used for the submission, distilled from my research
workspace. The experiments and the trial-and-error that led to it stay in that workspace.

## Run

1. Copy the competition engine package (`cg/`) into this folder.
2. `pip install torch numpy sentence-transformers`
3. `python build_text_tables.py` rebuilds the card and attack text embeddings from the engine.
4. `python self_test.py` loads `model.pth` and plays two games.

`main.py` is the Kaggle entry point.

## Not included

The competition engine, and the effect-feature tables `out/card_atom_feats.npy` and
`out/attack_atom_feats.npy`, which were extracted from card text. Without the effect features the agent
runs, but chooses a different action on 405 of 2,556 recorded decisions of the submission (15.8%).

## Licence

MIT. The pointer scoring head follows the TargetUnitHead of DI-star (OpenDILab, Apache-2.0).
