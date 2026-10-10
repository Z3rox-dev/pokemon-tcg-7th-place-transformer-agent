"""Play this agent against another agent. Run it from this folder after copying cg/ in.

    python play_match.py --opponent path/to/their/main.py --games 100
    python play_match.py --games 20                      # against a random opponent

The opponent is a Python file that defines `agent(obs) -> list[int]`, the same contract as the Kaggle
entry point: called with `obs["select"] is None` it returns its 60-card deck, otherwise it returns the
indices of the chosen options. Its folder is added to `sys.path` and becomes the working directory
while the file is loaded, so it can find its own modules and weights. Pass `--opponent-deck` to
override the deck it returns.

Seats alternate every game, so each side is player 0 in half of the games. Both agents receive the
deck handshake before every game: this agent resets its recurrent state and its trackers there.

A move that the engine rejects, or an exception raised by an agent, forfeits the game for that side
and is counted separately in the summary.
"""
import argparse
import contextlib
import copy
import json
import os
import random
import runpy
import sys
import time

_HERE = os.path.abspath(os.getcwd())
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def read_deck(path):
    deck = [int(x) for x in open(path) if x.strip()]
    assert len(deck) == 60, "%s must hold exactly 60 card IDs, found %d" % (path, len(deck))
    return deck


def load_agent(path):
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    if folder not in sys.path:
        sys.path.append(folder)
    os.chdir(folder)
    try:
        return runpy.run_path(path)["agent"]
    finally:
        os.chdir(_HERE)


def random_agent(deck):
    def agent(o):
        s = o["select"]
        if s is None:
            return deck
        k, n = s["maxCount"], len(s["option"])
        return random.sample(range(n), min(k, n)) if (k and n) else []
    return agent


def handshake(agent, obs):
    o = copy.deepcopy(obs)
    o["select"] = None
    return agent(o)


def play_game(g, agents, decks, max_decisions, verbose=False):
    """agents and decks are indexed by seat. Returns (winner seat or 2 for a draw, decisions, fault)."""
    obs, _ = g.battle_start(decks[0], decks[1])
    with contextlib.ExitStack() as stack:
        if not verbose:  # this agent logs one line per decision
            null = stack.enter_context(open(os.devnull, "w"))
            stack.enter_context(contextlib.redirect_stdout(null))
            stack.enter_context(contextlib.redirect_stderr(null))
        stack.callback(g.battle_finish)
        for a in agents:
            handshake(a, obs)
        for n in range(max_decisions):
            cur = obs["current"]
            if cur["result"] >= 0:
                return cur["result"], n, None
            seat = cur["yourIndex"]
            try:
                move = [int(i) for i in agents[seat](obs)]
                obs = g.battle_select(move)
            except Exception as exc:  # the side that raised, or sent an illegal move, forfeits
                return 1 - seat, n, "%s: %s" % (type(exc).__name__, exc)
        return 2, max_decisions, "no result after %d decisions" % max_decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--opponent", help="Python file defining agent(obs); default: a random opponent")
    ap.add_argument("--opponent-deck", help="CSV with 60 card IDs; default: the deck the opponent returns")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-decisions", type=int, default=6000)
    ap.add_argument("--log", help="write one JSON line per game to this file")
    ap.add_argument("--verbose", action="store_true", help="keep what the agents print during a game")
    args = ap.parse_args()

    random.seed(args.seed)
    env = {}
    exec(compile(open("main.py").read(), "main.py", "exec"), env)
    ours = env["agent"]
    our_deck = read_deck("deck.csv")

    import cg.game as g

    if args.opponent:
        theirs = load_agent(args.opponent)
    else:
        theirs = random_agent(our_deck)
    if args.opponent_deck:
        their_deck = read_deck(args.opponent_deck)
    else:
        try:
            their_deck = [int(c) for c in theirs({"select": None})]
        except Exception as exc:
            sys.exit("the opponent did not return a deck (%s): pass --opponent-deck" % exc)
        assert len(their_deck) == 60, "the opponent returned %d cards, expected 60" % len(their_deck)

    wins = losses = draws = faults_ours = faults_theirs = 0
    log = open(args.log, "w") if args.log else None
    t0 = time.time()
    for i in range(args.games):
        our_seat = i % 2
        agents, decks = [theirs, theirs], [their_deck, their_deck]
        agents[our_seat], decks[our_seat] = ours, our_deck
        winner, n, fault = play_game(g, agents, decks, args.max_decisions, args.verbose)
        if winner == our_seat:
            wins += 1
            faults_theirs += fault is not None
        elif winner == 1 - our_seat:
            losses += 1
            faults_ours += fault is not None
        else:
            draws += 1
        outcome = "draw" if winner == 2 else ("win" if winner == our_seat else "loss")
        if log:
            log.write(json.dumps({"game": i, "our_seat": our_seat, "outcome": outcome,
                                  "decisions": n, "fault": fault}) + "\n")
            log.flush()
        print("game %d/%d  seat %d  %-4s  %d decisions%s   running: %d-%d-%d"
              % (i + 1, args.games, our_seat, outcome, n, "  FAULT " + fault if fault else "",
                 wins, losses, draws), file=sys.stderr, flush=True)
    if log:
        log.close()

    played = wins + losses + draws
    print()
    print("  games %d   this agent %d   opponent %d   draws %d   (%.1f%% wins, %.0f s)"
          % (played, wins, losses, draws, 100.0 * wins / max(played, 1), time.time() - t0))
    if faults_ours or faults_theirs:
        print("  forfeits by fault: this agent %d, opponent %d" % (faults_ours, faults_theirs))


if __name__ == "__main__":
    main()
