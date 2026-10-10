# Pokémon TCG AI Battle — 7th place agent

The agent of submission 55565305 (7th of 6807, score 1226.6): a 10.7M-parameter transformer with a GRU
belief core that scores the legal actions the engine offers at each decision. It plays the
Dragapult / Blaziken deck in `deck.csv`.

This repository holds only the final version used for the submission, distilled from my research
workspace. The experiments and the trial-and-error that led to it stay in that workspace.

## Run

1. Copy the competition engine package (`cg/`) of the first round into this folder (1,267 cards).
2. `pip install torch numpy`
3. `python self_test.py` loads `model.pth` and plays two games. It ends with `RESULT: agent runs.`

`main.py` is the Kaggle entry point: it defines `agent(obs)`. Everything runs on CPU.

## Play against it

    python play_match.py --opponent path/to/your/main.py --games 100

The opponent is a Python file that defines `agent(obs) -> list[int]` with the Kaggle contract: called
with `obs["select"] is None` it returns its 60-card deck, otherwise the indices of the chosen options.
Seats alternate every game. `--opponent-deck deck.csv` overrides the opponent's deck, `--log games.jsonl`
writes one line per game, and with no `--opponent` the agent plays a random opponent.

To drive the agent from your own loop instead, call it once with `select` set to `None` before every
game: that is where it resets its recurrent state and its trackers. Without that call the state of the
previous game leaks into the next one.

## Card tables

`out/` holds the four tables of the submission, indexed by the card and attack ids of the first-round
engine:

- `card_text_emb.npy`, `attack_text_emb.npy`: frozen sentence embeddings of the card and attack text.
  `python build_text_tables.py` rebuilds them from the engine (needs `pip install sentence-transformers`).
- `card_atom_feats.npy`, `attack_atom_feats.npy`: effect features extracted from card text. They
  cannot be rebuilt from this repository. Without them the agent runs, but chooses a different action on
  405 of 2,556 recorded decisions of the submission (15.8%).

The engine itself is not included.

## The submission bundle as uploaded

`kaggle_submission/` is the content of the archive uploaded to Kaggle for submission 55565305, file
for file, with two things taken out: the engine package `cg/`, and the `raw_text` field of
`out/card_schema.jsonl`, which the code never reads (the tables it loads are identical without it).
The code there is the working version, with its original comments in Italian; the root of this
repository is the same agent cleaned up. To run it, copy `cg/` into that folder and start the runner
from there:

    cd kaggle_submission
    python ../play_match.py --opponent path/to/your/main.py --games 100

## Two more agents from the same league

Two exploiters trained against this agent during the first round, each a ready bundle built by the
same builder as the Kaggle one (same code as `kaggle_submission/`, final weights and deck of the round):

- `exploiter_alakazam_nightmine_v5/`: Alakazam deck, final checkpoint at iteration 1759 (9 August).
- `exploiter_slowking_v3/`: Slowking deck, final checkpoint at iteration 2194 (13 August).

They run like the submission bundle: copy `cg/` into the folder and start `python ../play_match.py`
from there. Each bundle is self-contained and uses the first-round card pool only. Two of these agents
cannot be loaded in the same Python process, because they share module names: run one per process.

## Licence

MIT. The pointer scoring head follows the TargetUnitHead of DI-star (OpenDILab, Apache-2.0).
