"""ENERGY-MATCH LINEAR PROBE — is the funcnet4 energy misplay a REP gap or a LEARNING gap?

Observed misplay (dragapult_dusknoir, net-piloted): the net attaches the wrong energy TYPE to its
attacker and/or discards the energy its attack needs. This probe answers the fork that decides the fix.

QUESTION: can a LINEAR read of the FROZEN net's representation, at the EXACT attach/discard decision,
tell whether a candidate energy's TYPE is one the target Pokemon's attack actually requires?

  * HIGH AUC at the ACTION TOKEN      -> the match reaches the policy's decision point -> the net CAN
                                         see it -> misplay is a LEARNING/policy failure (self-play never
                                         rewarded using it) -> fix = BC-from-RB / stronger opponents.
  * LOW at the action token but HIGH  -> the info lives in the board rep but is NOT wired to the action
    at the ENCODER target token          -> fix = graft a type-match feature onto the ATTACH/discard word.
  * LOW everywhere                    -> genuine rep gap -> fix = add explicit energy-type / cost features.

Comparing FRESH-random vs LIVE vs post-BC nets isolates what TRAINING added: if even a fresh random net
linearly encodes the match (the atom/text features determine energy type, a random projection preserves
it), the info is trivially present and the problem is purely the policy's USE of it.

LABEL (engine-exact, type-relevance): for a candidate "attach energy of type T to my Pokemon P" (or
"discard energy of type T from P"), label=1 iff T != COLORLESS(0) AND T is a requirement of one of P's
attacks. The atomic capability behind the misplay: does the rep know which TYPES the attacker needs?

REUSE: the verified random-policy collector (tools.funcnet_stage3_eval.collect_holdout_decisions) +
funcnet4 encoding + funcnet4 forward. Read-only; writes NOTHING to the live farm (no --out at all).

Run (light):  python -m funcnet4.energy_probe --games 40
Run (broad):  python -m funcnet4.energy_probe --games 30 --broad --ckpts out/funcnet_tf_gen.pth,out/funcnet_tf_realbc.pth
"""
import os
import sys
import glob
import time
import random
import argparse
import pathlib
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
torch.set_num_threads(4)  # be polite: this box is also a farm generator

from cg.api import OptionType, EnergyType, AreaType
from bytrl.actions import enumerate_actions
from funcnet.encoding import get_card
from funcnet.spec import CARD_TABLE
from funcnet.features import ATTACKS
from funcnet4.model import FuncNetTF, infer_action_attention
from funcnet4.encoding import get_encoder_input, get_decoder_input
from funcnet4.spec import ENC_DENSE_DIM_TF, DEC_DENSE_DIM_TF
from tools.funcnet_stage3_eval import collect_holdout_decisions, read_deck

COLORLESS = int(EnergyType.COLORLESS)   # 0
OWN_ACTIVE_WORD = 16                     # encoder word order: 8 own bench(0-7),8 opp(8-15),own active=16


# ------------------------------------------------------------------ engine-exact label helpers
def needed_colored_types(cid):
    """The set of NON-colorless energy types required by any attack of card `cid` (engine-exact via the
    ATTACKS table). Empty for colorless-only attackers / non-Pokemon."""
    card = CARD_TABLE.get(cid)
    if card is None:
        return set()
    types = set()
    for aid in (getattr(card, "attacks", None) or []):
        att = ATTACKS.get(aid)
        if att is None:
            continue
        for e in att.energies:
            t = int(e)
            if t != COLORLESS:
                types.add(t)
    return types


def _word_idx(area, idx):
    """Encoder word index for one of MY in-play Pokemon (active=16, bench j=0..7). None if not mine/in-play."""
    if area == AreaType.ACTIVE:
        return OWN_ACTIVE_WORD
    if area == AreaType.BENCH and 0 <= idx < 8:
        return idx
    return None


def _energy_type(card):
    """Energy TYPE provided by a card. The in-play/hand instance may not carry energyType directly (like
    the Pokemon instance doesn't carry .attacks) — resolve through the static CardData via the card id."""
    cid = getattr(card, "id", None)
    if cid is None:
        return None
    c = CARD_TABLE.get(cid)
    if c is None:
        return None
    e = getattr(c, "energyType", None)
    return None if e is None else int(e)


def sample_specs(o):
    """For one decision obs `o`, return (acts, list-of-sample-specs). Each spec targets a single-option
    ATTACH-to-own-Pokemon or ENERGY(discard)-of-own action and carries everything the label needs —
    computed ENGINE-SIDE, independent of any checkpoint:
        (action_index_a, encoder_word_idx, energy_type_T, target_cid, label, kind)."""
    your_index = o.current.yourIndex
    acts = enumerate_actions(o.select)
    specs = []
    for a, act in enumerate(acts):
        if len(act) != 1:
            continue
        opt = o.select.option[act[0]]
        kind = None
        if opt.type == OptionType.ATTACH:
            widx = _word_idx(opt.inPlayArea, opt.inPlayIndex)          # target must be MY in-play mon
            if widx is None:
                continue
            energy = get_card(o, opt.area, opt.index, your_index)      # the energy being attached
            target = get_card(o, opt.inPlayArea, opt.inPlayIndex, your_index)
            kind = "attach"
        elif opt.type in (OptionType.ENERGY, OptionType.ENERGY_CARD) \
                and getattr(opt, "energyIndex", None) is not None \
                and (opt.playerIndex is None or opt.playerIndex == your_index):
            widx = _word_idx(opt.area, opt.index)
            if widx is None:
                continue
            holder = get_card(o, opt.area, opt.index, your_index)
            energy = holder.energyCards[opt.energyIndex] if holder is not None else None
            target = holder
            kind = "discard"
        else:
            continue
        T = _energy_type(energy)
        tcid = getattr(target, "id", None)
        if T is None or tcid is None:
            continue
        needed = needed_colored_types(tcid)
        if not needed:
            continue    # target has no COLORED requirement -> attaching any type is equally (ir)relevant;
                        # skip so the probe measures TYPE-matching, not the trivial "needs anything" signal
        label = 1 if (T != COLORLESS and T in needed) else 0
        specs.append((a, widx, T, tcid, label, kind))
    return acts, specs


# ------------------------------------------------------------------ funcnet4 forward
def _enc_tensors(sv, device):
    return (torch.tensor(np.asarray(sv.dense, dtype=np.float32), device=device).view(-1, ENC_DENSE_DIM_TF),
            torch.tensor(np.asarray(sv.role, dtype=np.int64), device=device))


def _dec_tensors(sv, device):
    return (torch.tensor(np.asarray(sv.dense, dtype=np.float32), device=device).view(-1, DEC_DENSE_DIM_TF),
            torch.tensor(np.asarray(sv.plane, dtype=np.int64), device=device))


def extract_features(model, decisions, specs_per_dec, device):
    """One frozen forward per decision -> gather, for every sample spec: the ACTION token, the ENCODER
    target-Pokemon token, the raw decoder input row, the net's policy score, and the label/kind/decision-id.
    Returns dict of stacked arrays."""
    Xa, Xe, Xr, POL, Y, KIND, DEC, GRP = [], [], [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for di, ((o, deck), (acts, specs)) in enumerate(zip(decisions, specs_per_dec)):
            if not specs:
                continue
            enc = _enc_tensors(get_encoder_input(o, deck), device)
            dsv = get_decoder_input(o, acts)
            dec = _dec_tensors(dsv, device)
            eo, _ = model.encode(enc[0], enc[1])                       # (24,1,d)
            at = model.action_tokens(eo, None, dec[0], dec[1])         # (A,1,d)
            pol = model.score(eo, None, dec[0], dec[1])[0]             # (A,) tanh scores
            dense_d = np.asarray(dsv.dense, dtype=np.float32)          # (A, DEC_DIM)
            for (a, widx, T, tcid, label, kind) in specs:
                Xa.append(at[a, 0].cpu().numpy())
                Xe.append(eo[widx, 0].cpu().numpy())
                Xr.append(dense_d[a])
                POL.append(float(pol[a].item()))
                Y.append(label); KIND.append(kind); DEC.append(di)
                GRP.append(di * 64 + widx)     # (decision x target-Pokemon) group: fixed target => pure match
    return {"action": np.stack(Xa), "encoder": np.stack(Xe), "raw": np.stack(Xr),
            "pol": np.asarray(POL), "y": np.asarray(Y), "kind": np.asarray(KIND),
            "dec": np.asarray(DEC), "grp": np.asarray(GRP)}


# ------------------------------------------------------------------ linear probe + AUC
def rank_auc(scores, labels):
    """Mann-Whitney rank AUC (tie-naive; LR scores are continuous)."""
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def within_dec_auc(scores, y, dec):
    """Mean rank-AUC computed WITHIN each decision (only decisions holding both classes), then averaged.
    CONTROLS for the target/board: anything constant across a decision's options (the board token, the
    per-deck base rate) cancels in within-decision ranking, so this reflects ONLY the energy-driven
    type-match signal. The confound-free metric. Returns (mean_auc, n_decisions_with_both_classes)."""
    scores = np.asarray(scores); y = np.asarray(y); dec = np.asarray(dec)
    aucs = []
    for d in np.unique(dec):
        m = dec == d
        a = rank_auc(scores[m], y[m])
        if a == a:                      # not nan => decision had both a needed and a useless option
            aucs.append(a)
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def probe_eval(X, y, dec, grp, device, seed=0, epochs=300, l2=1e-3, split=0.7):
    """Fit a global linear (logistic) probe on a DECISION-disjoint train split, then score held-out
    decisions and measure AUC WITHIN each (decision x target-Pokemon) group (`grp`) — a fixed target, so
    only the energy varies and the score reflects the pure type-match. Return (global test AUC, within-
    group test AUC, n_test_groups). Global high + within ~0.50 == the probe reads the target/base-rate."""
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    dec = np.asarray(dec); grp = np.asarray(grp)
    uniq = np.unique(dec)
    rng = np.random.RandomState(seed); rng.shuffle(uniq)
    ntr = int(len(uniq) * split); train_dec = set(uniq[:ntr].tolist())
    trm = np.array([d in train_dec for d in dec]); tem = ~trm
    if int(trm.sum()) < 10 or int(tem.sum()) < 10:
        return float("nan"), float("nan"), 0
    mu = X[trm].mean(0); sd = X[trm].std(0) + 1e-6
    Xs = (X - mu) / sd
    Xtr = torch.tensor(Xs[trm], dtype=torch.float32, device=device)
    Ytr = torch.tensor(y[trm], dtype=torch.float32, device=device)
    Xte = torch.tensor(Xs[tem], dtype=torch.float32, device=device)
    lin = torch.nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=5e-3, weight_decay=l2)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); lossf(lin(Xtr).squeeze(1), Ytr).backward(); opt.step()
    with torch.no_grad():
        s_te = lin(Xte).squeeze(1).cpu().numpy()
    g = rank_auc(s_te, y[tem])
    wd, nd = within_dec_auc(s_te, y[tem], grp[tem])
    return g, wd, nd


# ------------------------------------------------------------------ behavioral: does the POLICY pick right?
def policy_misplay_rate(F):
    """For a FIXED target Pokemon offered BOTH a needed-type and a useless-type energy (a real choice, keyed
    by the (decision x target) group), how often is the net's top-scored attach a NEEDED-type one? Low =
    the observed misplay, quantified directly on the policy."""
    by_dec = defaultdict(list)
    for i in range(len(F["y"])):
        if F["kind"][i] == "attach":
            by_dec[int(F["grp"][i])].append((F["pol"][i], int(F["y"][i])))
    good = tot = 0
    for _d, lst in by_dec.items():
        labs = [l for _s, l in lst]
        if 1 in labs and 0 in labs:                 # genuine needed-vs-useless choice
            tot += 1
            top = max(lst, key=lambda z: z[0])
            good += int(top[1] == 1)
    return (good / tot if tot else float("nan")), tot


# ------------------------------------------------------------------ main
def build_net(ckpt, args, device):
    sd = torch.load(ckpt, map_location="cpu") if ckpt else None
    m = FuncNetTF(args.dmodel, args.heads, args.ff, args.enc_layers, args.dec_layers, recurrent="none",
                  **(infer_action_attention(sd) if sd is not None else {}))
    if ckpt:
        miss, unexp = m.load_state_dict(sd, strict=False)
        nonaux_miss = [k for k in miss if not k.startswith("aux_")]
        print(f"    [load] {os.path.basename(ckpt)} missing(non-aux)={len(nonaux_miss)} unexpected={len(unexp)}",
              flush=True)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default=str(REPO / "decks/dragapult_dusknoir.csv"),
                    help="deck to pilot while collecting attach/discard decisions (the misplay deck)")
    ap.add_argument("--broad", action="store_true",
                    help="pilot ALL meta decks round-robin instead of just --pilot (deck-agnostic rep test)")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--max-dec", type=int, default=6000, help="cap on collected decisions")
    ap.add_argument("--per-game", type=int, default=150,
                    help="cap decisions collected per game (keep low in --broad to span many decks)")
    ap.add_argument("--ckpts", default="out/funcnet_tf_gen.pth,out/funcnet_tf_realbc.pth",
                    help="comma list of checkpoints to probe (a FRESH-random net is always probed first)")
    ap.add_argument("--dmodel", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--enc-layers", type=int, default=8)
    ap.add_argument("--dec-layers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    use_cuda = (not args.cpu) and torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if use_cuda else "cpu"

    # ---- pilots + opponents
    meta = {os.path.splitext(os.path.basename(p))[0]: p for p in sorted(glob.glob(str(REPO / "decks/meta/*.csv")))}
    if args.broad:
        pilots = sorted(meta.values())
    else:
        if not os.path.exists(args.pilot):
            print(f"[fatal] pilot deck not found: {args.pilot}", flush=True); sys.exit(1)
        pilots = [args.pilot]
    opp_pool = sorted(meta.values())
    print(f"=== energy-match probe | device={device} | pilots={len(pilots)} | games={args.games} ===", flush=True)

    # ---- collect decisions (random policy, engine-exact) — ONCE, checkpoint-independent
    t0 = time.time()
    decisions = []                                # list[(o, deck_ids)]
    for gi in range(args.games):
        pilot_path = pilots[gi % len(pilots)]
        deck0 = read_deck(pilot_path)
        rng = random.Random(args.seed + gi)
        opp = rng.choice([p for p in opp_pool if p != pilot_path] or [pilot_path])
        got = collect_holdout_decisions(deck0, read_deck(opp), device,
                                        max_decisions=min(args.per_game, args.max_dec - len(decisions)),
                                        seed=args.seed + gi)
        for (o, deck0_ids, seat, d) in got:
            decisions.append((o, deck0_ids))
        if len(decisions) >= args.max_dec:
            break
    print(f"[collect] {len(decisions)} decisions ({time.time()-t0:.0f}s)", flush=True)

    # ---- engine-side sample specs (checkpoint-independent) + label stats
    specs_per_dec = [sample_specs(o) for (o, _deck) in decisions]
    all_specs = [s for (_a, specs) in specs_per_dec for s in specs]
    n_attach = sum(1 for s in all_specs if s[5] == "attach")
    n_disc = sum(1 for s in all_specs if s[5] == "discard")
    n_pos = sum(1 for s in all_specs if s[4] == 1)
    print(f"[samples] total={len(all_specs)}  attach={n_attach}  discard={n_disc}  "
          f"positives(needed-type)={n_pos} ({n_pos/max(len(all_specs),1):.1%})", flush=True)
    if len(all_specs) < 50 or n_pos < 10 or (len(all_specs) - n_pos) < 10:
        print("[warn] too few / too skewed samples for a reliable probe — raise --games or use --broad.", flush=True)

    # reveal the piloted attacker's actual cost types (resolves the atk0-vs-atk1 / Dragapult question)
    active_needed = Counter()
    for (_a, specs) in specs_per_dec:
        for (_ai, widx, _T, tcid, _l, kind) in specs:
            if kind == "attach" and widx == OWN_ACTIVE_WORD:
                card = CARD_TABLE.get(tcid)
                nm = getattr(card, "name", None) or f"cid{tcid}"
                active_needed[(nm, tuple(sorted(needed_colored_types(tcid))))] += 1
    if active_needed:
        print("[active attackers] (name, needed-colored-types) : count", flush=True)
        for (k, c) in active_needed.most_common(8):
            print(f"    {k[0]:<28} needs={k[1]}  x{c}", flush=True)

    if len(all_specs) < 50 or n_pos < 10:
        print("=== abort: insufficient labeled samples ===", flush=True); return

    # ---- probe each net: FRESH-random first, then the requested checkpoints
    ckpt_list = [("FRESH-random", None)] + [("+" + os.path.basename(c), c.strip())
                                            for c in args.ckpts.split(",") if c.strip()]
    print(f"\n{'='*92}", flush=True)
    print(f"  ENERGY-MATCH PROBE — WITHIN-DECISION rank-AUC (confound-free): can a linear read of the FROZEN", flush=True)
    print(f"  rep rank a NEEDED-type energy above a useless one, among the SAME decision's options?", flush=True)
    print(f"  (0.50 = the rep cannot tell needed from useless; ~1.0 = perfectly readable).  n={len(all_specs)}", flush=True)
    print(f"  [global] in parens = confounded global AUC, shown only to prove the confound (high global +", flush=True)
    print(f"  ~0.50 within == probe reads the deck/target base-rate, not the match).", flush=True)
    print(f"{'='*92}", flush=True)
    print(f"  {'net':<22}{'ACTION(wd)':>12}{'raw(wd)':>10}{'canary(wd)':>12}{'policy(wd)':>12}{'top-pick':>12}", flush=True)
    print("  " + "-" * 80, flush=True)
    for label, ckpt in ckpt_list:
        if ckpt and not os.path.exists(ckpt):
            print(f"  {label:<22}  (missing, skipped)", flush=True); continue
        net = build_net(ckpt, args, device)
        F = extract_features(net, decisions, specs_per_dec, device)
        ga, a_wd, nd = probe_eval(F["action"], F["y"], F["dec"], F["grp"], device, seed=args.seed)
        gr, r_wd, _ = probe_eval(F["raw"], F["y"], F["dec"], F["grp"], device, seed=args.seed)
        gc, c_wd, _ = probe_eval(F["encoder"], F["y"], F["dec"], F["grp"], device, seed=args.seed)
        pol_wd, _pnd = within_dec_auc(F["pol"], F["y"], F["grp"])
        tp, ttot = policy_misplay_rate(F)
        tp_s = f"{tp:.0%}(n={ttot})" if tp == tp else "n/a"
        print(f"  {label:<22}{a_wd:>12.3f}{r_wd:>10.3f}{c_wd:>12.3f}{pol_wd:>12.3f}{tp_s:>12}"
              f"      [global A/r/c={ga:.2f}/{gr:.2f}/{gc:.2f}, wd n={nd}]", flush=True)
        del net, F
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n  READ-OUT (within-decision AUC; 0.50 = rep can't tell needed from useless energy):", flush=True)
    print("   * canary(wd) ~0.50 confirms the confound is gone (board token is constant within a decision).", flush=True)
    print("   * ACTION(wd) HIGH  -> the net's action rep carries the target-relative match -> LEARNING/policy", flush=True)
    print("                          gap (the net CAN see it; fix = BC-from-RB / stronger opponents).", flush=True)
    print("   * ACTION(wd) ~ raw(wd) ~0.50 -> no linear match feature anywhere -> REP/grounding gap (graft one).", flush=True)
    print("   * policy(wd) ~0.50 -> the net's OWN policy does NOT rank needed energy above useless == the", flush=True)
    print("                          misplay, measured directly (independent of the probe).", flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
