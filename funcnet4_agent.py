"""Kaggle deploy agent: one forward per decision, greedy, self-contained.

Everything that shapes the input is pinned in main.py before this module is imported - the feature
flags decide the encoder and decoder row widths, and they do not travel inside the checkpoint. Get
one wrong and the weights still load, with a column of zeros where a feature belongs, and the agent
simply plays worse.

The deterministic lethal finder runs on top of the policy and overrides it only on a proven winning
line. The deploy-time rollout search was measured worse than greedy and has been removed from this
release.
"""
import os
import sys
import time
import random
import pathlib
import traceback

os.environ.setdefault("FUNCNET_N_HAND_TOKENS", "12")

_paths = [os.getcwd(), "/kaggle_simulations/agent"]
try:
    _repo = str(pathlib.Path(__file__).resolve().parents[1])
    _paths += [_repo, os.path.join(_repo, "engine")]
except NameError:
    pass
for _p in _paths:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from cg.api import to_observation_class
from bytrl.actions import enumerate_actions
from funcnet4.model import FuncNetTF, infer_action_attention
from funcnet4.encoding import get_encoder_input, get_decoder_input

CKPT = os.environ.get("FUNCNET4_CKPT", "model.pth")
ENC = int(os.environ.get("FUNCNET4_ENC", "8"))
DEC = int(os.environ.get("FUNCNET4_DEC", "8"))


def _read_deck():
    for fp in ["deck.csv", "/kaggle_simulations/agent/deck.csv"]:
        if os.path.exists(fp):
            rows = [r for r in open(fp).read().split("\n") if r.strip()]
            return [int(rows[i]) for i in range(60)]
    raise FileNotFoundError("deck.csv")


def _find_ckpt():
    cands = [CKPT, os.path.join("/kaggle_simulations/agent", os.path.basename(CKPT))]
    try:
        here = pathlib.Path(__file__).resolve().parent
        cands += [str(here / CKPT), str(here / os.path.basename(CKPT)),
                  str(here.parent / os.path.basename(CKPT))]
    except NameError:
        pass
    for fp in cands:
        if fp and os.path.exists(fp):
            return fp
    return None


DECK = _read_deck()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_ck = _find_ckpt()
_sd = None
if _ck:
    _sd = torch.load(_ck, map_location="cpu")
    _sd = {k: (v.float() if v.is_floating_point() else v) for k, v in _sd.items()}

if _sd is not None:
    _dm = int(_sd["encoder.layers.0.self_attn.in_proj_weight"].shape[1])
    _ff = int(_sd["encoder.layers.0.linear1.weight"].shape[0])
    _attn_kw = infer_action_attention(_sd)
    _recurrent = "gru" if any(k.startswith("rnn.") for k in _sd) else "none"
    _rec_hidden = int(_sd["rnn.weight_hh_l0"].shape[1]) if _recurrent != "none" else 256
    _vbins = int(_sd["value_dist_head.weight"].shape[0]) if "value_dist_head.weight" in _sd else 0
else:
    _dm, _ff, _attn_kw = 256, 512, {"act_selfattn": 0, "act_compare": 0}
    _recurrent, _rec_hidden = "none", 256
    _vbins = 0

_heads = int(os.environ.get("FUNCNET4_HEADS", str(max(1, _dm // 64))))
_model = FuncNetTF(_dm, _heads, _ff, ENC, DEC, recurrent=_recurrent, rec_hidden=_rec_hidden,
                   value_bins=_vbins, **_attn_kw).eval()
if _sd is not None:
    _miss, _unexp = _model.load_state_dict(_sd, strict=False)
    if _miss or _unexp:
        raise RuntimeError(f"checkpoint arch mismatch: missing={list(_miss)[:8]} unexpected={list(_unexp)[:8]}")
    print(f"[funcnet4_agent] loaded {_ck} d{_dm} ff{_ff} words={_model.num_words} "
          f"legacy_sa={_attn_kw['act_selfattn']} act_compare={_attn_kw['act_compare']} "
          f"gru_zones={_attn_kw.get('gru_zones', 0)} miss={len(_miss)} unexp={len(_unexp)}",
          file=sys.stderr, flush=True)
else:
    print("[funcnet4_agent] *** WARNING: checkpoint NOT FOUND -> RANDOM WEIGHTS ***",
          file=sys.stderr, flush=True)
_model.to(_device)
_state = None
from funcnet4.spec import (DKR_FEAT as _DKR_FEAT, PRZ_FEAT as _PRZ_FEAT, NUM_WORDS as _NUM_WORDS,
                           LOC_FEAT as _LOC_FEAT, ENC_DENSE_DIM_TF as _ENC_DENSE_DIM_TF)
_dkr = bool(_DKR_FEAT and getattr(_model, "num_words", None) == _NUM_WORDS)
_loc = bool(_LOC_FEAT and getattr(_model, "enc_in_dim", 0) == _ENC_DENSE_DIM_TF)
_prize_tracker = None
if (_dkr and _PRZ_FEAT) or _loc:
    from funcnet4.deck_tracker import DeckPrizeTracker
    _prize_tracker = DeckPrizeTracker(DECK)
print(f"[funcnet4_agent] dkr={_dkr} loc={_loc} prize_tracker={'on' if _prize_tracker is not None else 'off'} "
      f"(num_words model={getattr(_model,'num_words','?')} spec={_NUM_WORDS} "
      f"enc_in model={getattr(_model,'enc_in_dim','?')} spec={_ENC_DENSE_DIM_TF})",
      file=sys.stderr, flush=True)

_LETHAL_ON = int(os.environ.get("FUNCNET4_LETHAL", "2"))
_NO_OVERKILL = int(os.environ.get("FUNCNET4_NO_OVERKILL", "1"))
_FEZ_MASK = int(os.environ.get("FUNCNET4_FEZ_MASK", "1"))
_ENERGY_SLACK = 0 if int(os.environ.get("FUNCNET4_ENERGY_STRICT", "0")) else 1
_ENERGY_TELEM = int(os.environ.get("FUNCNET4_ENERGY_TELEM", "0"))
_MASK_LOG = int(os.environ.get("FUNCNET4_MASK_LOG", "1"))
ENERGY_TELEM = {"decisions": 0, "live_active": 0, "live_pruned_argmax": 0,
                "strict_active": 0, "strict_pruned_argmax": 0}
_LETHAL_MS_BUDGET = float(os.environ.get("FUNCNET4_LETHAL_MS", "120000"))
_LETHAL_GATE_LEGACY = os.environ.get("FUNCNET4_LETHAL_GATE", "") == "legacy"
_ko_tracker = None
_BENCH_FEAT = __import__('os').environ.get('FUNCNET4_BENCH_FEAT', '0') not in ('', '0')
_bench_memo = {}
_lethal = None
_lethal_turn = None
_lethal_att_turn = None
_lethal_att = 0
_lethal_sig = None
_lethal_pv = None
_lethal_ms = 0.0
LAST_SCORES = None


def _enc_tensors(sv):
    de = np.asarray(sv.dense, dtype=np.float32)
    return (torch.tensor(de, device=_device).view(-1, de.shape[-1]),
            torch.tensor(np.asarray(sv.role, dtype=np.int64), device=_device))


def _dec_tensors(sv):
    dd = np.asarray(sv.dense, dtype=np.float32)
    return (torch.tensor(dd, device=_device).view(-1, dd.shape[-1]),
            torch.tensor(np.asarray(sv.plane, dtype=np.int64), device=_device))


def _lethal_pick(o, actions):
    """Deterministic lethal override. None when no winning line is proven."""
    global _lethal, _lethal_turn, _lethal_att_turn, _lethal_att, _lethal_sig, \
        _lethal_pv, _lethal_ms, _LETHAL_ON
    if not _LETHAL_ON:
        return None
    t0 = time.perf_counter()
    try:
        if _lethal is None:
            from funcnet4.lethal_dragapult_dusknoir import LethalFinder
            _lethal = LethalFinder(DECK, None, compose=(_LETHAL_ON >= 2))
        st = o.current
        t_now = int(st.turn)
        if _lethal_turn == t_now:
            from funcnet4.lethal_dragapult_dusknoir import pv_tail
            f = _lethal.find(o, pv=_lethal_pv)
            if f is None and _lethal_pv is not None:
                f = _lethal.find(o)
            if f is None:
                _lethal_turn = None
                _lethal_pv = None
                return None
            _lethal_pv = pv_tail(f["ais"])
            return int(f["first_ai"])
        if _lethal_ms > _LETHAL_MS_BUDGET:
            return None
        from cg.api import SelectType
        if int(o.select.type) != int(SelectType.MAIN):
            return None
        if _LETHAL_GATE_LEGACY:
            if len(st.players[st.yourIndex].prize or []) > 4:
                _opp = st.players[1 - st.yourIndex]
                if sum(1 for z in (_opp.active, _opp.bench)
                       for c in (z or []) if c is not None) > 2:
                    return None
        else:
            from funcnet4.lethal_dragapult_dusknoir import lethal_plausible
            if not lethal_plausible(o):
                return None
        if _lethal_att_turn != t_now:
            _lethal_att_turn, _lethal_att, _lethal_sig = t_now, 0, None
        from funcnet4.lethal_dragapult_dusknoir import rearm_sig, should_rearm, pv_tail
        sig = rearm_sig(o)
        if _lethal_att >= 2:
            if _lethal_att >= 6 or not should_rearm(_lethal_sig, sig):
                return None
            _min = True
        else:
            _min = False
        _lethal_att += 1
        _lethal_sig = sig
        f = _lethal.find(o, minimal=_min)
        if f is None:
            return None
        if _lethal.find(o, minimal=_min, pv=f["ais"]) is None and _lethal.find(o, minimal=_min) is None:
            return None
        _lethal_turn = t_now
        _lethal_pv = pv_tail(f["ais"])
        return int(f["first_ai"])
    except Exception:
        _LETHAL_ON = 0
        return None
    finally:
        _lethal_ms += (time.perf_counter() - t0) * 1000.0


def _decision_log(o, sc, idx, actions, overridden, pre_ko):
    """One line per decision on stderr: what the agent did and why."""
    if not _MASK_LOG:
        return
    try:
        top = sorted((float(x) for x in sc.tolist()), reverse=True)[:2]
        p1 = top[0]
        p2 = top[1] if len(top) > 1 else float("nan")
        print(f"[fn4] t={int(getattr(o.current,'turn',-1))} n={len(actions)} a={idx} "
              f"p1={p1:.3f} p2={p2:.3f} ovr={int(bool(overridden))} pk={int(bool(pre_ko))}",
              file=sys.stderr, flush=True)
    except Exception:
        pass


def _mask_log(o, tag, before, after, n_actions, n_kept, pre_ko):
    """Deploy log on stderr: which masks fired on this decision."""
    if not _MASK_LOG:
        return
    try:
        turn = int(getattr(o.current, "turn", -1))
        if before != after:
            print(f"[fn4] mask={tag} turn={turn} idx {before}->{after} keep={n_kept}/{n_actions} "
                  f"preko={int(bool(pre_ko))}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _fez_watch(o, actions, pre_ko, chosen):
    """Warn when the agent is about to play Fezandipiti with its ability unable to fire."""
    if not _MASK_LOG or pre_ko:
        return
    try:
        from cg.api import OptionType, SelectType, AreaType
        from funcnet.encoding import get_card
        from funcnet4.lethal_dragapult_dusknoir import FEZANDIPITI_CID
        sel = o.select
        if int(sel.type) != int(SelectType.MAIN):
            return
        me = o.current.yourIndex
        for oi in actions[chosen]:
            opt = sel.option[oi]
            if int(opt.type) != int(OptionType.PLAY):
                continue
            if getattr(get_card(o, AreaType.HAND, opt.index, me), "id", None) != FEZANDIPITI_CID:
                continue
            mine = o.current.players[me]
            n_play = sum(1 for z in (mine.active, mine.bench) for c in (z or []) if c is not None)
            print(f"[fn4] !! FEZ-UNMASKED turn={int(getattr(o.current,'turn',-1))} "
                  f"in_play={n_play} preko=0 -- the mask did not block the play",
                  file=sys.stderr, flush=True)
            return
    except Exception:
        pass


def agent(obs_dict: dict) -> list[int]:
    global _state, LAST_SCORES, _ko_tracker
    LAST_SCORES = None
    try:
        sel = obs_dict.get("select")
        if sel is None:
            _state = None
            global _lethal_turn, _lethal_att_turn, _lethal_att, _lethal_sig, _lethal_pv
            _lethal_turn, _lethal_att_turn, _lethal_att, _lethal_sig = None, None, 0, None
            _lethal_pv = None
            if _ko_tracker is not None:
                _ko_tracker.reset()
            if _prize_tracker is not None:
                _prize_tracker.reset()
            return DECK
        n = len(sel["option"])
        if n == 0 or sel["maxCount"] == 0:
            return []
        o = to_observation_class(obs_dict)
        _pre_ko = None
        if _FEZ_MASK and _NO_OVERKILL:
            try:
                if _ko_tracker is None:
                    from funcnet4.lethal_dragapult_dusknoir import KOTracker
                    _ko_tracker = KOTracker()
                _ko_tracker.observe(o)
                _pre_ko = _ko_tracker.pre_ko(o.current.yourIndex)
            except Exception:
                _pre_ko = None
        if _prize_tracker is not None:
            _prize_tracker.observe(o)
        actions = enumerate_actions(o.select)
        if len(actions) == 1:
            return actions[0]
        e = _enc_tensors(get_encoder_input(o, DECK, dkr=_dkr, loc=_loc, prize_tracker=_prize_tracker))
        _bench = None
        if _BENCH_FEAT:
            try:
                from funcnet4 import bench_counters
                _bench = bench_counters.values_for_actions(o, actions, DECK, None, memo=_bench_memo)
            except Exception:
                _bench = None
        d = _dec_tensors(get_decoder_input(o, actions, bench_vals=_bench))
        with torch.inference_mode():
            _, policy, new_state = _model(e[0], e[1], d[0], d[1], state=_state)
        _state = new_state
        sc = policy[0]
        idx = int(sc.argmax())
        overridden = False
        _keep_set = None
        try:
            _forced = _lethal_pick(o, actions)
            if _forced is not None:
                idx, overridden = int(_forced), True
            elif _NO_OVERKILL:
                from funcnet4.lethal_dragapult_dusknoir import mask_dead_counter_targets, mask_dominated_main
                if _ENERGY_TELEM:
                    try:
                        from funcnet4.lethal_dragapult_dusknoir import mask_excess_energy_attach
                        ENERGY_TELEM["decisions"] += 1
                        _kl = mask_excess_energy_attach(o, actions, slack=_ENERGY_SLACK)
                        _ks = _kl if _ENERGY_SLACK == 0 else mask_excess_energy_attach(o, actions, slack=0)
                        if _kl is not None:
                            ENERGY_TELEM["live_active"] += 1
                            if idx not in _kl:
                                ENERGY_TELEM["live_pruned_argmax"] += 1
                        if _ks is not None:
                            ENERGY_TELEM["strict_active"] += 1
                            if idx not in _ks:
                                ENERGY_TELEM["strict_pruned_argmax"] += 1
                    except Exception:
                        pass
                _tag = None
                _keep = mask_dead_counter_targets(o, actions)
                if _keep is not None:
                    _tag = "counter"
                else:
                    _keep, _tag = mask_dominated_main(o, actions, pre_ko=_pre_ko,
                                                      energy_slack=_ENERGY_SLACK)
                if _keep is not None:
                    _before = idx
                    idx = int(max(_keep, key=lambda k: float(sc[k])))
                    _keep_set = set(int(k) for k in _keep)
                    _mask_log(o, _tag, _before, idx, len(actions), len(_keep), _pre_ko)
        except Exception:
            idx = int(sc.argmax())
        LAST_SCORES = {"actions": [list(a) for a in actions],
                       "scores": [float(x) for x in sc.tolist()],
                       "maxCount": int(sel["maxCount"]), "nOptions": int(n),
                       "capped": len(actions) >= 64, "override": overridden}
        _decision_log(o, sc, idx, actions, overridden, _pre_ko)
        _fez_watch(o, actions, _pre_ko, idx)
        return actions[idx]
    except Exception:
        try:
            _cur = obs_dict.get("current") or {}
            _t = _cur.get("turn", "?")
        except Exception:
            _t = "?"
        sel = obs_dict.get("select") or {}
        _nopt = len(sel.get("option", []) or [])
        _exc = traceback.format_exc(limit=3).strip().replace("\n", " | ")
        print(f"[funcnet4_agent] !!! RANDOM-PLAY FALLBACK (net path raised) turn={_t} nopt={_nopt} "
              f":: {_exc[:760]}", file=sys.stderr, flush=True)
        if not sel or not sel.get("maxCount"):
            return []
        return random.sample(range(len(sel["option"])), min(sel["maxCount"], len(sel["option"])))
