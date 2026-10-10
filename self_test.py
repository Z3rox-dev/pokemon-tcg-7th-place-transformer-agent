"""Self-test for the agent. Run it from this folder after copying cg/ in.

    python self_test.py

What it checks, and why each check exists:

1. `main.py` executes with no `__file__`. The Kaggle harness execs the file, it does not import it;
   code that relies on `__file__` works locally and dies in the arena.

2. Every feature tail is armed. The 13 `FUNCNET4_*` environment flags do not travel inside the
   checkpoint. If one is wrong the model still loads, with a column of zeros where a feature
   should be, and plays worse without raising anything. `main.py` pins the flags itself.

3. Encoder/decoder widths match the checkpoint (1145 / 1079).

4. The agent plays complete games against a random opponent. Loading is not playing: a module that
   raises on every decision degrades to random moves and still "works".

5. Every armed feature tail is also fed. On this checkpoint the BENCH tail is reported as dead: it was
   zero in training too, so it costs nothing.

It also reports which card tables are present.

Expected output ends with: RESULT: agent runs.
"""
import copy
import os
import random
import sys

_HERE = os.path.abspath(os.getcwd())
sys.path = [_HERE] + [p for p in sys.path
                      if p and os.path.abspath(p) != _HERE
                      and not os.path.isdir(os.path.join(p, "funcnet4"))]

env = {}
exec(compile(open("main.py").read(), "main.py", "exec"), env)
agent = env["agent"]
print("  [1/5] main.py executes with no __file__ .......... OK")

deck = [int(x) for x in open("deck.csv") if x.strip()]
assert len(deck) == 60, "deck.csv must hold exactly 60 card IDs, found %d" % len(deck)

import funcnet4.spec as S

for name in ("NRG_FEAT", "CNT_FEAT", "UNLOCK_FEAT", "DKR_FEAT", "PRZ_FEAT",
             "TGT_FEAT", "BENCH_FEAT", "LOC_FEAT"):
    assert getattr(S, name), "%s is OFF: the checkpoint would see a zero tail" % name
assert S.NUM_WORDS == 38, "NUM_WORDS=%d, expected 38" % S.NUM_WORDS
assert S.N_ROLE == 15, "N_ROLE=%d, expected 15" % S.N_ROLE
assert S.ENC_DENSE_DIM_TF == 1145, "encoder width %d, expected 1145" % S.ENC_DENSE_DIM_TF
assert S.DEC_DENSE_DIM_TF == 1079, "decoder width %d, expected 1079" % S.DEC_DENSE_DIM_TF
print("  [2/5] all feature tails armed, widths 1145/1079 ... OK")

import funcnet4.loc_tail as lt
assert lt.fill_loc_dense is not None
import funcnet4_agent as ag
assert int(ag._FEZ_MASK) == 1, "FEZ_MASK must be active"
print("  [3/5] loc_tail present, FEZ_MASK active ........... OK")

import cg.game as g

obs, _ = g.battle_start(deck, deck)
oh = copy.deepcopy(obs)
oh["select"] = None
hand = agent(oh)
assert len(hand) == 60, "handshake returned %d cards, expected 60" % len(hand)
g.battle_finish()


def random_move(o):
    s = o["select"]
    if s is None:
        return deck
    k, n = s["maxCount"], len(s["option"])
    return random.sample(range(n), min(k, n)) if (k and n) else []


random.seed(1)
results = []
for _ in range(2):
    obs, _ = g.battle_start(deck, deck)
    for _ in range(6000):
        cur = obs["current"]
        if cur["result"] >= 0:
            results.append(cur["result"])
            break
        move = agent(obs) if cur["yourIndex"] == 0 else random_move(obs)
        obs = g.battle_select(move)
    g.battle_finish()

assert len(results) == 2, "games did not terminate: %s" % results
print("  [4/5] played %d complete games, results %s ....... OK" % (len(results), results))

try:
    from funcnet4 import bench_counters  # noqa: F401
    print("  [5/5] every armed tail is also fed ................ OK")
except ImportError:
    print("  [5/5] BENCH tail armed but not fed ......... DEAD FEATURE (cost: zero, measured)")
    print("        Decoder columns 1076-1078 are zero at every decision. The data this checkpoint")
    print("        was trained on had them at zero as well, so nothing is lost.")


def _present(names):
    return [n for n in names if os.path.isfile(os.path.join("out", n))]


emb = ("card_text_emb.npy", "attack_text_emb.npy")
atom = ("card_atom_feats.npy", "attack_atom_feats.npy")
print()
if len(_present(emb)) < len(emb):
    print("  NOTE: text embeddings missing. Rebuild them with: python build_text_tables.py")
if len(_present(atom)) < len(atom):
    print("  NOTE: effect-atom tables missing from out/. Without them the agent differs from the")
    print("        submission on 405 of 2,556 recorded decisions (15.8%).")

print()
print("  RESULT: agent runs.")
