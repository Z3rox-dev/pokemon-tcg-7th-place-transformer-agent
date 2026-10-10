"""FuncNetTF — the TEXT-FIRST model. A card is read from its precomputed features (func+atom+text) by
SHARED MLPs (enc_card_mlp / dec_action_mlp), plus small role/plane tag embeddings; NO per-card id-bag.

Reuses the SHARED, verified trunk + heads from bytrl.model.RecurrentNet (transformer encoder, decoder
layers, value head encoder_fc, policy head decoder_fc, value_from/recur/initial_state) — only the per-card
INPUT representation changes. The two id-bags (encoder_bag/decoder_bag, 24.5M params of pure memorization)
are DELETED -> the model is lean (~9M, the trunk + heads + small readers).

Forward contract (recurrent='none', the project default):
  forward(dense_e, role_e, dense_d, plane_d, state=None) -> (value (B,1), policy (B,A), new_state)
    dense_e: (W*B, ENC_DENSE_DIM_TF) per-word feature row  role_e: (W*B,) long  per-word ROLE id
    dense_d: (A*B, DEC_DENSE_DIM_TF) per-action feature row plane_d: (A*B,) long per-action PLANE id
"""
import math

import torch

from bytrl.model import RecurrentNet
from funcnet4.spec import (ENC_DENSE_DIM, ENC_DENSE_DIM_TF, DEC_DENSE_DIM_TF,
                           N_ROLE, N_PLANE, PLANE_PAD, NUM_WORDS, N_ZONES, ZONE_WORDS)
from funcnet.effects_spec import EFF_VOCAB_DIM, EFF_DELTA_DIM


_ID_GUARD_SEEN = set()   # (name, val) gia' loggati: evita di allagare il log su un veleno ricorrente


def _guard_ids(t, n, name):
    """Clampa a [0, n) gli id destinati a una nn.Embedding, loggando FORTE i fuori-range.

    07-20: un role_e/plane_d OOB (da una traiettoria avvelenata di un generatore) fa scattare un
    device-side assert dentro il lookup Embedding -> 'CUDA illegal memory access' -> contesto CUDA
    corrotto -> learner MORTO in modo irrecuperabile (pure empty_cache nel gestore fallisce -> exit 1,
    crash-restart ~ogni 1.5h con step azzerato). Il bounds-guard esistente in losses copriva solo
    card-id e attack-id, non questi due. Stessa filosofia: clampare (neutralizza il crash) + LOG del
    valore colpevole (per risalire al gen). Su dati puliti e' un no-op (una riduzione sul batch/forward:
    trascurabile in training, B=1 in generazione). NON entra nel cfg_hash: e' difesa runtime, non encoding."""
    mn = int(t.min())
    mx = int(t.max())
    if mn < 0 or mx >= n:
        key = (name, mn, mx)
        if key not in _ID_GUARD_SEEN:
            _ID_GUARD_SEEN.add(key)
            bad = t[(t < 0) | (t >= n)]
            vals = sorted(set(bad.detach().cpu().tolist()))[:10]
            print(f"[id-guard] {name} OOB {vals} (validi [0,{n})) su {bad.numel()} id -- CLAMPATO, "
                  f"processo protetto; INVESTIGA il generatore (traiettoria avvelenata?)", flush=True)
        return t.clamp(0, n - 1)
    return t


_LEGACY_ATTN_KEYS = frozenset({
    "act_sa_norm.weight", "act_sa_norm.bias",
    "act_sa.in_proj_weight", "act_sa.in_proj_bias",
    "act_sa.out_proj.weight", "act_sa.out_proj.bias",
})
_COMPARE_ATTN_KEYS = frozenset({
    "act_compare_norm.weight", "act_compare_norm.bias",
    "act_compare_sa.in_proj_weight", "act_compare_sa.in_proj_bias",
    "act_compare_sa.out_proj.weight", "act_compare_sa.out_proj.bias",
    "act_compare_fc.weight", "act_compare_fc.bias",
})
# ACT-POINTER (scoring query·key alla DI-star TargetUnitHead, docs/act-pointer-design.md).
# key_dim 32 = compressione 8x del d_model 256 di produzione, valore del config DI-star vero
# (actor_critic_default_config.yaml: entity_embedding_dim 256, key_dim 32). NIENTE scaling /sqrt(d).
PTR_KEY_DIM = 32
# Seme FISSO della sintesi deterministica all'arm (spec §5): learner e qualunque processo che arma
# dallo stesso ckpt devono sintetizzare pesi IDENTICI (niente drift di policy fra processi).
_PTR_SYNTH_SEED = 0x7F4A11
# NAMING (review 07-24, BLOCKING): la query e' un Sequential `ptr_query_mlp` (chiavi .0/.2) e NON
# ptr_query_fc1/fc2 come in DI-star — i suffissi "fc1."/"fc2." collidono con le regole endswith
# del Net2Wider (widen.py:66-73, FFN del transformer) e con le sottostringhe di _NOISE_PAT
# ("fc1.weight"/"fc2.weight" matchano via `p in k`): un widen di un ckpt pointer allenato avrebbe
# iniettato rumore SILENZIOSO sul gate zero-init. Nessuna regola esistente cattura ptr_query_mlp.*.
_POINTER_KEYS = frozenset({
    "ptr_key_fc.weight",
    "ptr_query_mlp.0.weight", "ptr_query_mlp.0.bias",
    "ptr_query_mlp.2.weight", "ptr_query_mlp.2.bias",
    "ptr_bias",
})
# BELIEF->AZIONE (accoppiamento ricorrente v2, worktree gru-coupling). Realizza il principio
# "il core alimenta la testa d'azione" (AlphaStar: l'LSTM core alimenta il pointer) per la policy
# di PRODUZIONE (decoder_fc, act_pointer OFF): un termine q(belief)·k(token) ADDITIVO su main,
# per-azione (NON un bias costante che si cancellerebbe nel softmax). key_dim 32 come il pointer.
# Costruito SOLO se recurrent!='none' -> con recurrent='none' NIENTE modulo (byte-identico al ff).
# zero-init su belief_query -> termine ESATTAMENTE 0 a init (canale dedicato che parte neutro, la
# disciplina zero-init del progetto); il gradiente lo sveglia (cascata a 2 gate come il pointer).
BELIEF_KEY_DIM = 32
_BELIEF_KEYS = frozenset({
    "belief_key.weight",
    "belief_query.weight", "belief_query.bias",
})
# PTR-BELIEF (--ptr-belief, ByteRL-fedele, worktree gru-byterl-coupling). In ByteRL (paper
# 2303.05197, Fig. 2) la query dell'inner-product legge l'LSTM embed: qui la query del pointer
# legge [summary || belief] invece del solo summary — chiude la meta' pointer della Lacuna-A
# (audit gru-coupling: "query belief-condizionata = lavoro futuro spec §4"). NESSUNA chiave nuova:
# si ALLARGA ptr_query_mlp.0 di rec_hidden colonne che nascono ZERO (pattern LOC-tail applicato
# all'input di un modulo) -> all'arm la query e' BIT-identica e il gradiente sveglia le colonne
# belief. Rilevamento dal sd = LARGHEZZA di fc1 (d_model vs d_model+rec_hidden): niente set di
# chiavi nuovo, niente whitelist warm-start, _POINTER_KEYS resta INVARIATO.
#
# GRU-ZONES (--gru-zones, ByteRL-fedele). L'ingresso del core ricorrente non e' piu' la sola
# media globale (la "cannuccia": l'ablazione belief 08-12/13 ha misurato flip 5-8% e B~=C, quasi
# solo MEMORIA) ma [summary || zone_proj(concat medie-PER-ZONA)] (spec.ZONE_WORDS): mano, bench,
# mazzo e scarti non si mescolano piu' in una media sola PRIMA della memoria — e' la forma
# dell'input di ByteRL (zone embeddate separatamente, poi concat+fc, poi LSTM). Disciplina
# UNA-SOLA-A-ZERO: il gate e' SOLO rnn.weight_ih_l0[:, d_model:] (colonne input nuove a zero);
# zone_proj nasce VIVO (init standard / sintesi deterministica all'arm) — uno zero doppio
# (zone_proj E colonne) avrebbe gradiente ESATTAMENTE 0 su entrambi per sempre (canale morto).
_ZONE_SYNTH_SEED = 0x60A1E5
_ZONES_KEYS = frozenset({"zone_proj.weight", "zone_proj.bias"})


def _sd_d_model(sd):
    w = sd.get("encoder.layers.0.self_attn.in_proj_weight")
    if w is None:
        raise RuntimeError("state_dict senza encoder: impossibile dedurre d_model")
    return int(w.shape[1])


def _sd_rec_hidden(sd):
    w = sd.get("rnn.weight_hh_l0")
    return int(w.shape[1]) if w is not None else None


def _infer_complete_attention_block(sd, expected, prefixes, label):
    present = {k for k in sd if k.startswith(prefixes)}
    if not present:
        return 0
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise RuntimeError(f"checkpoint {label} incompleto/corrotto: missing={missing} extra={extra}")
    return 1


def infer_act_selfattn(sd) -> int:
    """1 se lo state_dict contiene il blocco di self-attention azione<->azione (--act-selfattn).
    Usata dai loader arch-aware (_infer_expert_cfg/_build_model e derivati): il flag viaggia
    nel CKPT, i generatori/payoff si auto-adattano senza flag di launcher."""
    return _infer_complete_attention_block(
        sd, _LEGACY_ATTN_KEYS, ("act_sa.", "act_sa_norm."), "act_sa legacy")


def infer_act_compare(sd) -> int:
    """1 se lo state_dict contiene il nuovo ramo di confronto fra azioni candidate.

    Il namespace e' volutamente distinto da ``act_sa``: i checkpoint dell'esperimento fallito
    restano riconoscibili e non possono essere scambiati silenziosamente per attention-v2.
    """
    return _infer_complete_attention_block(
        sd, _COMPARE_ATTN_KEYS, ("act_compare_",), "act_compare v2")


def infer_act_pointer(sd) -> int:
    """1 se lo state_dict contiene il blocco pointer completo (scoring query·key DI-star).

    Set COMPLETO di chiavi (pattern _COMPARE_ATTN_KEYS): un blocco ptr_* parziale/corrotto
    RAISA invece di attivare/saltare la sintesi in silenzio (review 07-24, trigger di §5).
    """
    return _infer_complete_attention_block(
        sd, _POINTER_KEYS, ("ptr_",), "act_pointer")


def infer_ptr_belief(sd, rec_hidden=None) -> int:
    """1 se la query del pointer legge anche il belief (fc1 larga d_model+rec_hidden).

    Nessun set di chiavi dedicato: l'arming vive nella LARGHEZZA di ptr_query_mlp.0.weight
    (pattern LOC-tail). Una larghezza diversa da {d_model, d_model+rec_hidden} = ckpt corrotto,
    raise (mai un default silenzioso con la shape sbagliata).

    ``rec_hidden`` (hint, review 08-13, finding CONFERMATO): nel warm-start ff->armato il sd ha
    fc1 GIA' allargata (prima migrazione, osfp) ma NON ha ancora rnn.* (il core nasce fresco dal
    costruttore) -> senza hint la seconda chiamata (loader di classe) non riconoscerebbe la
    larghezza e raiserebbe su un ckpt SANO (crash-loop col supervisore). L'hint viene usato SOLO
    quando rnn.* manca; con rnn.* nel sd fa fede il sd."""
    w = sd.get("ptr_query_mlp.0.weight")
    if w is None:
        return 0
    d_model = _sd_d_model(sd)
    width = int(w.shape[1])
    if width == d_model:
        return 0
    rh = _sd_rec_hidden(sd)
    if rh is None and rec_hidden:
        rh = int(rec_hidden)
    if rh is not None and width == d_model + rh:
        return 1
    raise RuntimeError(f"ptr_query_mlp.0.weight larghezza {width} non riconosciuta "
                       f"(attese {d_model} o d_model+rec_hidden; rnn nel sd: "
                       f"{_sd_rec_hidden(sd) is not None}, hint rec_hidden: {rec_hidden})")


def infer_gru_zones(sd) -> int:
    """1 se l'input del GRU e' [summary || zona] (weight_ih largo 2*d_model + zone_proj presente).

    Il blocco e' COMPLETO o assente (pattern _POINTER_KEYS): zone_proj parziale, zone_proj senza
    weight_ih largo o viceversa = ckpt corrotto, raise."""
    zp = {k for k in sd if k.startswith("zone_proj.")}
    if zp and zp != _ZONES_KEYS:
        raise RuntimeError(f"checkpoint gru_zones incompleto/corrotto: zone_proj={sorted(zp)}")
    w = sd.get("rnn.weight_ih_l0")
    if w is None:
        if zp:
            raise RuntimeError("checkpoint corrotto: zone_proj presente senza rnn.*")
        return 0
    d_model = _sd_d_model(sd)
    width = int(w.shape[1])
    if width == d_model:
        if zp:
            raise RuntimeError("checkpoint corrotto: zone_proj presente ma rnn.weight_ih_l0 "
                               f"stretto ({width})")
        return 0
    if width == 2 * d_model:
        if not zp:
            raise RuntimeError("checkpoint corrotto: rnn.weight_ih_l0 largo 2*d_model "
                               "senza zone_proj")
        return 1
    raise RuntimeError(f"rnn.weight_ih_l0 larghezza {width} non riconosciuta "
                       f"(attese {d_model} o {2 * d_model})")


def infer_action_attention(sd):
    """Constructor kwargs for an exact checkpoint architecture; rejects ambiguous checkpoints.

    Regole di coesistenza (spec act-pointer §6): pointer+compare OK (compare resta nel ckpt vivo,
    neutro — la variabile A/B e' solo il pointer); pointer+act_sa legacy = errore esplicito
    (l'esperimento fallito 07-17 non deve poter viaggiare insieme allo scoring nuovo).

    PTR-BELIEF / GRU-ZONES (review 08-13, finding CONFERMATO): il dict DEVE portare anche i due
    flag del coupling — questo e' l'unico punto di inferenza per eval_runner/evaluate/health/
    agents/imitation/search_proto/energy_probe e i tool distill_*: senza, un ckpt ARMATO caricato
    da quei consumer verrebbe DISARMATO in silenzio dal rollback implicito del loader (crop/strip
    con load strict pulito = rete lobotomizzata sul pannello/deploy, classe fail-soft
    payoff-env-flags-lobotomized-net-74pp)."""
    legacy, compare = infer_act_selfattn(sd), infer_act_compare(sd)
    pointer = infer_act_pointer(sd)
    if legacy and compare:
        raise RuntimeError("checkpoint contains both act_sa legacy and act_compare v2")
    if legacy and pointer:
        raise RuntimeError("checkpoint contains both act_sa legacy and act_pointer")
    return {"act_selfattn": legacy, "act_compare": compare, "act_pointer": pointer,
            "ptr_belief": infer_ptr_belief(sd), "gru_zones": infer_gru_zones(sd)}


_DECODER_INPUT_KEY = "dec_action_mlp.0.weight"
_PRE_DCX_WIDTH = 1058
_LEGACY_DCX_END = 1062
_LEGACY_DCX_UNLOCK_WIDTH = 1065
_PRE_TGT_WIDTH = 1061                      # layout v10 (DCX rimossa), sorgente tipica di un warm-start
# Ogni larghezza RILASCIATA resta una sorgente valida di warm-start: 1051 pre-NRG, 1054 pre-CNT,
# 1058 pre-DCX, 1062/1065 era-DCX, 1061 v10 (DCX rimossa), 1076 v11 (TGT-tail), poi la corrente.
# Una coda nuova DEVE aggiungere qui la larghezza precedente, altrimenti il ckpt vivo non si importa.
_RELEASED_WIDTHS = {1051, 1054, 1058, 1062, 1065, _PRE_TGT_WIDTH, 1076}
_KNOWN_DECODER_WIDTHS = _RELEASED_WIDTHS | {DEC_DENSE_DIM_TF}
_ROLE_EMBED_KEY = "enc_role_embed.weight"
_KNOWN_ROLE_ROWS = {12, 13, 15}

_ENCODER_INPUT_KEY = "enc_card_mlp.0.weight"
# Larghezze RILASCIATE della riga ENCODER. Fino al 2026-08-07 ne esisteva UNA sola (1109 = 74 func +
# 267 atom + 768 text), identica da funcnet a funcnet4: la LOC-tail e' la PRIMA coda encoder-side.
# Una coda encoder nuova DEVE aggiungere qui la larghezza precedente, altrimenti il ckpt vivo non si
# importa (stessa regola di _RELEASED_WIDTHS per il decoder).
_RELEASED_ENCODER_WIDTHS = {1109}
_KNOWN_ENCODER_WIDTHS = _RELEASED_ENCODER_WIDTHS | {ENC_DENSE_DIM_TF}


def migrate_decoder_input_layout(state_dict, *, log=False):
    """Return a copy with the decoder reader migrated to the DCX-free 1061-column layout.

    Historical layouts grew only by appending NRG, CNT, DCX and UNLOCK. DCX is the exception to a
    normal right-pad migration because it sits *before* UNLOCK in v9. Therefore a 1065 checkpoint
    must drop columns [1058:1062] and move [1062:1065] to [1058:1061]. A generic crop/pad would
    silently reinterpret learned DCX weights as UNLOCK weights.

    TGT-TAIL (2026-08-02): il layout cresce di nuovo IN CODA (1061 -> 1074), quindi la migrazione e'
    il normale right-pad a zero — le colonne nuove nascono spente e il forward di un ckpt importato
    e' bit-identico finche' il flag FUNCNET4_TGT_FEAT resta OFF (colonne d'ingresso a zero). La
    DCX resta l'unica eccezione storica (stava PRIMA di UNLOCK, non in coda).

    Supported source widths are the actual released layouts: 1051, 1054, 1058, 1062, 1065, 1061 and
    the new 1074. Unknown widths fail loudly. The input mapping and tensor are never mutated.
    """
    w = state_dict.get(_DECODER_INPUT_KEY)
    if w is None:
        return state_dict
    if not torch.is_tensor(w) or w.ndim != 2:
        raise RuntimeError(f"{_DECODER_INPUT_KEY}: expected a rank-2 tensor, got {type(w).__name__} "
                           f"shape={getattr(w, 'shape', None)}")
    src = int(w.shape[1])
    if src not in _KNOWN_DECODER_WIDTHS:
        raise RuntimeError(f"{_DECODER_INPUT_KEY}: unsupported decoder input width {src}; "
                           f"known layouts={sorted(_KNOWN_DECODER_WIDTHS)}")
    if src == DEC_DENSE_DIM_TF:
        return state_dict
    if DEC_DENSE_DIM_TF < _PRE_TGT_WIDTH:
        raise RuntimeError(f"decoder migration expects target width >= {_PRE_TGT_WIDTH}, "
                           f"got {DEC_DENSE_DIM_TF}")

    if src == _LEGACY_DCX_UNLOCK_WIDTH:
        # DCX sta PRIMA di UNLOCK: taglia [1058:1062] e riporta UNLOCK a [1058:1061], POI il
        # right-pad generico aggiunge le code nate dopo (TGT). Un crop/pad generico
        # reinterpreterebbe i pesi DCX appresi come UNLOCK.
        migrated = torch.cat([w[:, :_PRE_DCX_WIDTH], w[:, _LEGACY_DCX_END:]], dim=1)
        detail = "drop DCX[1058:1062], preserve UNLOCK[1062:1065]"
        if migrated.shape[1] < DEC_DENSE_DIM_TF:
            migrated = torch.cat(
                [migrated, migrated.new_zeros((w.shape[0], DEC_DENSE_DIM_TF - migrated.shape[1]))], dim=1)
            detail += " + zero-pad code successive"
    elif src == _LEGACY_DCX_END:
        migrated = torch.cat([w[:, :_PRE_DCX_WIDTH],
                              w.new_zeros((w.shape[0], DEC_DENSE_DIM_TF - _PRE_DCX_WIDTH))], dim=1)
        detail = "drop DCX[1058:1062], add zero UNLOCK (+ code successive)"
    else:
        migrated = torch.cat([w, w.new_zeros((w.shape[0], DEC_DENSE_DIM_TF - src))], dim=1)
        detail = "append zero columns for missing tails"
    if migrated.shape[1] != DEC_DENSE_DIM_TF:
        raise AssertionError(f"decoder migration produced {migrated.shape[1]} columns")
    out = dict(state_dict)
    out[_DECODER_INPUT_KEY] = migrated
    if log:
        print(f"[decoder-layout] {_DECODER_INPUT_KEY} {src}->{DEC_DENSE_DIM_TF}: {detail}", flush=True)
    return out


def migrate_encoder_input_layout(state_dict, *, log=False):
    """Copia con il reader ENCODER (enc_card_mlp.0.weight) portato alla larghezza corrente.

    La riga encoder e' [func(74)|atom(267)|text(768)] + code IN CODA, quindi la migrazione e' il
    normale RIGHT-PAD a zero: le colonne nuove nascono SPENTE e il forward del ckpt importato e'
    invariato finche' non imparano (arm neutro — la disciplina zero-init del progetto). Non c'e'
    nessuna eccezione tipo DCX: nessuna coda encoder e' mai stata inserita in mezzo.

    Il CROP e' RIFIUTATO: un modello piu' stretto del checkpoint significa FUNCNET4_LOC_FEAT
    disallineato rispetto al ckpt, non una migrazione (stessa scelta di migrate_role_embed_layout).
    Larghezze ignote falliscono RUMOROSAMENTE.

    L'input non viene mai mutato (out = dict(state_dict)); a larghezza gia' giusta e' un no-op
    ESATTO (ritorna lo STESSO oggetto), cosi' il braccio a flag OFF resta byte-identico."""
    w = state_dict.get(_ENCODER_INPUT_KEY)
    if w is None:
        return state_dict
    if not torch.is_tensor(w) or w.ndim != 2:
        raise RuntimeError(f"{_ENCODER_INPUT_KEY}: expected a rank-2 tensor, got {type(w).__name__} "
                           f"shape={getattr(w, 'shape', None)}")
    src = int(w.shape[1])
    if src not in _KNOWN_ENCODER_WIDTHS:
        raise RuntimeError(f"{_ENCODER_INPUT_KEY}: unsupported encoder input width {src}; "
                           f"known layouts={sorted(_KNOWN_ENCODER_WIDTHS)}")
    if src == ENC_DENSE_DIM_TF:
        return state_dict
    if src > ENC_DENSE_DIM_TF:
        raise RuntimeError(f"{_ENCODER_INPUT_KEY}: refusing destructive crop {src}->{ENC_DENSE_DIM_TF}; "
                           "allinea FUNCNET4_LOC_FEAT al checkpoint prima di importarlo")
    migrated = torch.cat([w, w.new_zeros((w.shape[0], ENC_DENSE_DIM_TF - src))], dim=1)
    if migrated.shape[1] != ENC_DENSE_DIM_TF:
        raise AssertionError(f"encoder migration produced {migrated.shape[1]} columns")
    out = dict(state_dict)
    out[_ENCODER_INPUT_KEY] = migrated
    if log:
        print(f"[encoder-layout] {_ENCODER_INPUT_KEY} {src}->{ENC_DENSE_DIM_TF}: "
              f"append zero columns (LOC-tail)", flush=True)
    return out


def migrate_role_embed_layout(state_dict, target_rows, target_dim, *, log=False):
    """Pad only released 12/13/15-row role layouts; never crop or accept invented counts."""
    w = state_dict.get(_ROLE_EMBED_KEY)
    if w is None:
        return state_dict
    if not torch.is_tensor(w) or w.ndim != 2:
        raise RuntimeError(f"{_ROLE_EMBED_KEY}: expected rank-2 tensor, got "
                           f"{type(w).__name__} shape={getattr(w, 'shape', None)}")
    src, dim = map(int, w.shape)
    target_rows, target_dim = int(target_rows), int(target_dim)
    if src not in _KNOWN_ROLE_ROWS or target_rows not in _KNOWN_ROLE_ROWS:
        raise RuntimeError(f"{_ROLE_EMBED_KEY}: unsupported role rows {src}->{target_rows}; "
                           f"known layouts={sorted(_KNOWN_ROLE_ROWS)}")
    if dim != target_dim:
        raise RuntimeError(f"{_ROLE_EMBED_KEY}: embedding width {dim} != target {target_dim}")
    if src > target_rows:
        raise RuntimeError(f"{_ROLE_EMBED_KEY}: refusing destructive role crop {src}->{target_rows}; "
                           "set FUNCNET4_DKR_FEAT to match the checkpoint before importing funcnet4")
    if src == target_rows:
        return state_dict
    padded = w.new_zeros((target_rows, target_dim))
    padded[:src] = w
    out = dict(state_dict)
    out[_ROLE_EMBED_KEY] = padded
    if log:
        print(f"[role-layout] {_ROLE_EMBED_KEY} PAD {src}->{target_rows} righe ZERO", flush=True)
    return out


_VALUE_BINS_KEYS = frozenset({"value_dist_head.weight", "value_dist_head.bias", "value_bin_centers"})


def _fit_bins_shape(n_bins, *, device=None, dtype=torch.float64, tol=0.01):
    """(g, beta) tali che  E(s) = sum_i c_i softmax(g_i*s + beta_i)_i  ~=  tanh(s).

    Dipendono SOLO da n_bins (non dai pesi allenati): e' pura geometria della testa.
    Il vincolo e' che i logit siano AFFINI nel belief, quindi con W = g (x) w si ha
    logit_i = g_i*s + beta_i e l'attesa e' funzione del solo s. 2*n_bins parametri liberi per
    inseguire una funzione di UNA variabile -> si adatta a ~2e-3.
    Parto dalla soluzione analitica g = 3c (l'attesa di una Boltzmann sui centri e' la funzione di
    Langevin, ~x/3 vicino a 0, e tanh(s) ~ s: k=3 aggancia la pendenza) e rifinisco.
    NON mi fido del determinismo dell'ottimizzatore fra versioni di torch: alla fine VERIFICO
    l'errore e alzo se sfora. Un init che non riproduce la testa vecchia romperebbe la baseline,
    ed e' esattamente il guasto che questa funzione esiste per evitare.
    """
    c = torch.linspace(-1.0, 1.0, n_bins, dtype=dtype)
    s = torch.cat([torch.linspace(-8, -3, 200, dtype=dtype),
                   torch.linspace(-3, 3, 1200, dtype=dtype),
                   torch.linspace(3, 8, 200, dtype=dtype)]).unique()
    tgt = torch.tanh(s)
    g = (3.0 * c).clone().requires_grad_(True)
    beta = torch.zeros(n_bins, dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([g, beta], lr=0.02)
    for _ in range(4000):
        E = (torch.softmax(s[:, None] * g[None, :] + beta[None, :], dim=1) * c[None, :]).sum(1)
        err = E - tgt
        loss = (err ** 2).mean() + 0.5 * torch.logsumexp(err.abs() * 50, dim=0) / 50
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        E = (torch.softmax(s[:, None] * g[None, :] + beta[None, :], dim=1) * c[None, :]).sum(1)
        emax = float((E - tgt).abs().max())
    if emax > tol:
        raise RuntimeError(f"value-bins init: la testa a {n_bins} bin non riproduce la scalare "
                           f"(errore max {emax:.4f} > {tol}); armare cosi' romperebbe la baseline")
    return g.detach(), beta.detach(), emax


def migrate_value_bins(state_dict, want_bins, *, log=False):
    """Sintesi/strip DETERMINISTICA della testa del valore a BIN, per l'arm e il rollback.

    PERCHE' ESISTE (misurato il 2026-08-07, costato il collasso della policy): `V` non e' solo una
    testa da allenare, e' la BASELINE del gradiente di policy. In bytrl/losses._traj_loss:
        adv = GAMMA*v_next - Vd ;  upgo_adv = Gu - Vd
    Con una testa a bin ZERO-INIT si ha V == 0, quindi `adv` diventa il RITORNO GREZZO: ogni mossa
    di una partita vinta prende +1 e ogni mossa di una persa -1, a prescindere dal suo merito. E'
    REINFORCE senza baseline. Armato cosi' il 08-07, in ~200 iterazioni il MAIN e' passato da
    assorbire 12 avversari su 50 a fare **0% contro 36 su 50**, con entropia da 0,84 a 1,45.
    ⚠ Abbassare `value_coef` NON cura: il danno passa da `Vd` nei termini di POLICY, non dal
    gradiente della loss del valore.

    La cura: nascere riproducendo la testa scalare. Con W[i,:] = g_i * w e bias_i = g_i*b0 + beta_i
    (w,b0 = la Linear scalare gia' allenata) il logit e' g_i*s + beta_i, quindi V(s) ~= tanh(s) con
    errore ~2e-3 su una scala di vantaggio ~1. La baseline resta valida dal passo 0 e la CE rifinisce
    la distribuzione invece di ricostruirla da zero.
    Vale sia per recurrent='gru' (value_fc su belief) sia per 'none' (encoder_fc mediato sulle
    parole): la media sulle parole e' lineare, quindi commuta con W = g (x) w.

    Casi (stessa disciplina di migrate_act_pointer):
      - want_bins=0, sd senza testa   -> no-op.
      - want_bins=0, sd ARMATO        -> STRIP (rollback: torna la scalare, che e' rimasta allocata).
      - want_bins=N, sd ARMATO        -> no-op: MAI ri-sintetizzare pesi allenati.
      - want_bins=N, sd senza testa   -> SINTESI di equivalenza.
    Ritorna una COPIA, mai muta l'input.
    """
    has = "value_dist_head.weight" in state_dict
    if not want_bins:
        if not has:
            return state_dict
        out = {k: v for k, v in state_dict.items() if k not in _VALUE_BINS_KEYS}
        if log:
            print("[value-bins] STRIP della testa a bin: rollback alla testa scalare", flush=True)
        return out
    if has:
        n = int(state_dict["value_dist_head.weight"].shape[0])
        if n != int(want_bins):
            raise RuntimeError(f"value-bins: il ckpt ha {n} bin ma ne sono richiesti {want_bins}")
        return state_dict

    # sorgente: la Linear scalare gia' allenata (gru -> value_fc, none -> encoder_fc)
    src = "value_fc.weight" if "value_fc.weight" in state_dict else "encoder_fc.weight"
    bsrc = src.replace(".weight", ".bias")
    if src not in state_dict:
        raise RuntimeError("value-bins: nessuna testa scalare (value_fc/encoder_fc) da cui sintetizzare")
    w = state_dict[src].detach().to(torch.float64).reshape(-1)          # (H,)
    b0 = float(state_dict[bsrc].detach().reshape(-1)[0]) if bsrc in state_dict else 0.0

    n_bins = int(want_bins)
    g, beta, emax = _fit_bins_shape(n_bins)
    W = g[:, None] * w[None, :]                                          # (N, H)
    b = g * b0 + beta                                                    # (N,)

    out = dict(state_dict)
    ref = state_dict[src]
    out["value_dist_head.weight"] = W.to(dtype=ref.dtype, device=ref.device)
    out["value_dist_head.bias"] = b.to(dtype=ref.dtype, device=ref.device)
    out["value_bin_centers"] = torch.linspace(-1.0, 1.0, n_bins).to(dtype=ref.dtype, device=ref.device)
    if log:
        print(f"[value-bins] SINTESI da {src}: {n_bins} bin che riproducono la testa scalare "
              f"(errore max {emax:.5f} su V). La baseline del gradiente di policy e' preservata.",
              flush=True)
    return out


def migrate_act_pointer(state_dict, want_pointer, *, log=False):
    """Sintesi/strip DETERMINISTICA del blocco ptr_* per l'arm e il rollback di act_pointer.

    FUNZIONE ESPORTATA, non solo un ramo di FuncNetTF.load_state_dict (review 07-24, BLOCKING):
    il warm-start di osfp FILTRA lo state_dict per shape PRIMA di chiamare il loader del modello
    e raisa su chiavi sconosciute — un ckpt ARMATO caricato col flag OFF (il rollback di
    emergenza) crasherebbe li' prima che lo strip qui dentro possa girare. osfp deve chiamare
    QUESTA funzione prima del suo filtro, nello stesso punto dove gia' chiama a mano
    migrate_decoder_input_layout e lo strip act_sa legacy. NB: osfp NON deve whitelist-are le
    ptr_* in allowed_missing (la sintesi le riempie prima del load: non risultano mai missing).

    Casi (trigger = infer_act_pointer, set completo: blocco parziale/corrotto raisa):
      - want_pointer=0, sd senza ptr_*  -> no-op.
      - want_pointer=0, sd ARMATO       -> STRIP con log (rollback: torna decoder_fc, congelato
        al momento dell'arm — rollback tardivo = policy vecchia di N step, spec §10).
      - want_pointer=1, sd ARMATO       -> no-op: MAI ri-sintetizzare pesi allenati (gate H).
      - want_pointer=1, sd senza ptr_*  -> SINTESI di equivalenza (spec §3): pointer che riproduce
        ESATTAMENTE decoder_fc all'avvio, leggendo w,b DALLO state_dict in ingresso.

    DETERMINISMO (review 07-24): tutte le estrazioni casuali avvengono su CPU fp32 con un
    torch.Generator CPU a seme fisso e ordine di estrazione FISSO (K, poi fc1.weight, poi
    fc1.bias), POI il cast a dtype/device di decoder_fc.weight — un generator CUDA renderebbe
    la sequenza device-dipendente e uniform_(generator=cpu) su tensore CUDA alza RuntimeError.
    Come le altre migrazioni: ritorna una COPIA, mai muta l'input.
    """
    has_ptr = infer_act_pointer(state_dict)          # raisa forte su blocco parziale
    if not want_pointer:
        if not has_ptr:
            return state_dict
        out = {k: v for k, v in state_dict.items() if k not in _POINTER_KEYS}
        if log:
            print(f"[act-pointer] STRIP {len(_POINTER_KEYS)} chiavi ptr_* (rollback act_pointer=0: "
                  "lo scorer torna decoder_fc, congelato al momento dell'arm)", flush=True)
        return out
    if has_ptr:
        return state_dict                            # load normale, MAI clobberare pesi allenati
    w = state_dict.get("decoder_fc.weight")
    b = state_dict.get("decoder_fc.bias")
    if w is None or b is None:
        raise RuntimeError("act_pointer=1: sintesi impossibile, decoder_fc.weight/bias "
                           "assenti dallo state_dict in ingresso")
    if not torch.is_tensor(w) or w.ndim != 2 or int(w.shape[0]) != 1:
        raise RuntimeError(f"decoder_fc.weight: attesa shape (1, d_model), trovato "
                           f"{tuple(getattr(w, 'shape', ()))}")
    d_model = int(w.shape[1])
    # init standard Linear riprodotta A MANO col generator (kaiming a=sqrt(5) -> uniform
    # ±1/sqrt(fan_in)); stesso bound per le righe casuali di K (spec §5).
    bound = 1.0 / math.sqrt(d_model)
    g = torch.Generator().manual_seed(_PTR_SYNTH_SEED)
    key_w = torch.empty(PTR_KEY_DIM, d_model, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    q1_w = torch.empty(PTR_KEY_DIM, d_model, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    q1_b = torch.empty(PTR_KEY_DIM, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    key_w = key_w.to(dtype=w.dtype, device=w.device)
    # riga 0 di K = w ESATTO (NON normalizzato): con q0=e0 il dot product riproduce w·token.
    # Le righe 1..31 divergenti fra versioni torch sarebbero comunque invisibili alla policy
    # (q0 legge solo la riga 0) finche' il training non le sovrascrive.
    key_w[0] = w.detach().reshape(-1)
    q2_w = torch.zeros(PTR_KEY_DIM, PTR_KEY_DIM, dtype=w.dtype, device=w.device)  # UNICO zero
    q2_b = torch.zeros(PTR_KEY_DIM, dtype=w.dtype, device=w.device)
    q2_b[0] = 1.0                                    # q0 = e0 -> query BIT-esatta all'avvio
    out = dict(state_dict)
    out["ptr_key_fc.weight"] = key_w
    out["ptr_query_mlp.0.weight"] = q1_w.to(dtype=w.dtype, device=w.device)
    out["ptr_query_mlp.0.bias"] = q1_b.to(dtype=w.dtype, device=w.device)
    out["ptr_query_mlp.2.weight"] = q2_w
    out["ptr_query_mlp.2.bias"] = q2_b
    # decoder_fc.bias e' shape (1,) nello state_dict reale (non uno scalare puro): ptr_bias
    # resta shape (1,) e broadcasta su (A,B,1) — copia ESATTA, niente item()/round-trip float.
    out["ptr_bias"] = b.detach().clone().reshape(1)
    if log:
        print(f"[act-pointer] SINTESI deterministica ptr_* (seed {_PTR_SYNTH_SEED:#x}, d_model "
              f"{d_model}): riga0(K)=w, ptr_bias=b, query=e0 -> pointer == decoder_fc all'avvio",
              flush=True)
    return out


def migrate_ptr_belief(state_dict, want_ptr_belief, rec_hidden=None, *, log=False):
    """Allarga/ritaglia ptr_query_mlp.0 per l'arm/rollback di --ptr-belief (query = f(summary, belief)).

    FUNZIONE ESPORTATA col doppio call-site obbligato (pattern migrate_act_pointer): loader di
    classe E osfp PRIMA del filtro-per-shape. Va chiamata DOPO migrate_act_pointer: con pointer
    appena sintetizzato fc1 e' (32, d_model) e qui si allarga. Niente whitelist warm-start
    (nessuna chiave nuova: cambia solo la larghezza di una chiave gia' in _POINTER_KEYS).

    Casi (trigger = infer_ptr_belief, larghezza sconosciuta raisa):
      - want=0, fc1 stretta o pointer assente -> no-op.
      - want=0, fc1 larga  -> CROP delle rec_hidden colonne belief (rollback documentato: la meta'
        summary resta allenata; il contributo belief va perso, come lo strip del pointer).
      - want=1, fc1 larga  -> no-op (MAI clobberare pesi allenati).
      - want=1, fc1 stretta -> APPEND di rec_hidden colonne ZERO (pattern LOC-tail): la query e'
        BIT-identica finche' il gradiente non sveglia le colonne belief. Niente RNG.
    Come le altre migrazioni: ritorna una COPIA, mai muta l'input, idempotente — ANCHE nel
    warm-start ff->armato dove il sd non ha rnn.* (l'hint rec_hidden fa riconoscere la fc1 gia'
    allargata dalla prima chiamata: seconda chiamata = no-op, mai raise su ckpt sano)."""
    has = infer_ptr_belief(state_dict, rec_hidden=rec_hidden)
    w = state_dict.get("ptr_query_mlp.0.weight")
    if not want_ptr_belief:
        if not has:
            return state_dict
        d_model = _sd_d_model(state_dict)
        out = dict(state_dict)
        out["ptr_query_mlp.0.weight"] = w[:, :d_model].clone()
        if log:
            print(f"[ptr-belief] CROP {int(w.shape[1]) - d_model} colonne belief da "
                  "ptr_query_mlp.0 (rollback ptr_belief=0: la query torna al solo summary)",
                  flush=True)
        return out
    if w is None:
        raise RuntimeError("ptr_belief=1 richiede il blocco pointer nello state_dict: chiamare "
                           "migrate_ptr_belief DOPO migrate_act_pointer (act_pointer=1)")
    if has:
        return state_dict
    rh = _sd_rec_hidden(state_dict)
    if rh is None:
        rh = int(rec_hidden) if rec_hidden else 0
    if not rh:
        raise RuntimeError("ptr_belief=1: rec_hidden non deducibile (sd senza rnn.*) "
                           "e non passato dal chiamante")
    out = dict(state_dict)
    out["ptr_query_mlp.0.weight"] = torch.cat(
        [w, w.new_zeros((int(w.shape[0]), rh))], dim=1)
    if log:
        print(f"[ptr-belief] APPEND {rh} colonne ZERO a ptr_query_mlp.0 (arm: query BIT-identica, "
              "le colonne belief si svegliano col gradiente)", flush=True)
    return out


def migrate_gru_zones(state_dict, want_zones, *, log=False):
    """Arm/rollback dell'ingresso per-zona del GRU (--gru-zones): weight_ih largo + zone_proj.

    FUNZIONE ESPORTATA col doppio call-site obbligato (pattern migrate_act_pointer). Casi
    (trigger = infer_gru_zones, incoerenze raisano):
      - want=0, non armato     -> no-op.
      - want=0, armato         -> rollback: CROP delle d_model colonne zona da weight_ih_l0 e
        STRIP di zone_proj.* (il core torna a leggere la sola media globale).
      - want=1, armato         -> no-op (MAI clobberare pesi allenati).
      - want=1, sd SENZA rnn.* -> no-op: e' l'arm ff->gru, il costruttore fornisce rnn largo e
        zone_proj freschi (coperti dalla allowance warm-start, stessa via di rnn./mem_proj.).
      - want=1, rnn stretto    -> APPEND d_model colonne ZERO a weight_ih_l0 (UNICO gate a zero,
        una-sola-a-zero) + SINTESI deterministica di zone_proj (seed fisso, CPU fp32, cast dopo —
        disciplina migrate_act_pointer): il forward e' ESATTO-identico all'arm finche' il
        gradiente non sveglia le colonne zona.
    Come le altre migrazioni: ritorna una COPIA, mai muta l'input, idempotente."""
    has = infer_gru_zones(state_dict)
    if not want_zones:
        if not has:
            return state_dict
        d_model = _sd_d_model(state_dict)
        out = {k: v for k, v in state_dict.items() if k not in _ZONES_KEYS}
        w = out["rnn.weight_ih_l0"]
        out["rnn.weight_ih_l0"] = w[:, :d_model].clone()
        if log:
            print("[gru-zones] STRIP zone_proj + CROP colonne zona da rnn.weight_ih_l0 "
                  "(rollback gru_zones=0: il core torna alla sola media globale)", flush=True)
        return out
    if has:
        return state_dict
    w = state_dict.get("rnn.weight_ih_l0")
    if w is None:
        return state_dict        # arm ff->gru: rnn+zone_proj freschi dal costruttore (allowance)
    d_model = _sd_d_model(state_dict)
    g = torch.Generator().manual_seed(_ZONE_SYNTH_SEED)
    fan_in = N_ZONES * d_model
    bound = 1.0 / math.sqrt(fan_in)
    zp_w = torch.empty(d_model, fan_in, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    zp_b = torch.empty(d_model, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    out = dict(state_dict)
    out["rnn.weight_ih_l0"] = torch.cat([w, w.new_zeros((int(w.shape[0]), d_model))], dim=1)
    out["zone_proj.weight"] = zp_w.to(dtype=w.dtype, device=w.device)
    out["zone_proj.bias"] = zp_b.to(dtype=w.dtype, device=w.device)
    if log:
        print(f"[gru-zones] APPEND {d_model} colonne ZERO a rnn.weight_ih_l0 + SINTESI zone_proj "
              f"(seed {_ZONE_SYNTH_SEED:#x}, {N_ZONES} zone): ingresso ESATTO-identico all'arm",
              flush=True)
    return out


class FuncNetTF(RecurrentNet):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int,
                 num_layers_encoder: int, num_layers_decoder: int,
                 recurrent: str = "none", rec_hidden: int | None = None,
                 num_words: int | None = None, value_bins: int = 0, act_selfattn: int = 0,
                 act_compare: int = 0, act_pointer: int = 0, ptr_belief: int = 0,
                 gru_zones: int = 0):
        # funcnet4: il default e' NUM_WORDS (36 con 12 token-mano; 38 con DKR) — tutti i
        # call-site che passano None (osfp, _build_model, eval, tools) ereditano il layout nuovo.
        super().__init__(d_model, num_heads, d_feedforward,
                         num_layers_encoder, num_layers_decoder,
                         recurrent, rec_hidden, num_words or NUM_WORDS)
        # DISTRIBUTIONAL value head (funcnet4 candidate, --value-bins, default 0 = OFF -> byte-identical
        # to funcnet2). Value as CLASSIFICATION over N uniform bins in [-1,1] (two-hot / HL-Gauss targets,
        # CE loss in losses.py) instead of scalar regression; every consumer still reads the SCALAR
        # expectation via value_from, so V-trace / search leaves / eval are unchanged. The parent's
        # encoder_fc/value_fc stay allocated (unused when bins on) -> ckpt name-compat both ways.
        self.value_bins = int(value_bins)
        if self.value_bins > 0:
            _vin = d_model if recurrent == "none" else (rec_hidden or d_model)
            self.value_dist_head = torch.nn.Linear(_vin, self.value_bins)
            # zero-init -> uniform softmax -> V=0, max entropy start (CE grads are alive at zero-init)
            torch.nn.init.zeros_(self.value_dist_head.weight)
            torch.nn.init.zeros_(self.value_dist_head.bias)
            self.register_buffer("value_bin_centers", torch.linspace(-1.0, 1.0, self.value_bins))
        # ACTION SELF-ATTENTION (--act-selfattn, default 0 = OFF -> byte-identical). Buco
        # strutturale del DecoderLayer bytrl (SOLO cross-attention azione->board): ogni azione
        # e' valutata alla cieca rispetto alle ALTERNATIVE, il confronto vive solo nel softmax
        # finale tra scalari (sintomo storico: CTX_NO_ATTACK/set_ctx, la pezza a mano bocciata
        # nel blocco 2026-07). Qui la cura strutturale: UN blocco residuale pre-norm di
        # self-attention azione<->azione DOPO lo stack cross-attn, con out_proj ZERO-INIT ->
        # warm-start bit-identico a iter 0 (stessa disciplina dec_ctx_proj / gate C4); il
        # canale di confronto entra solo via gradiente. NON tocca encoder/wire format/codec.
        # VERDETTO 2026-07-17 (doppia ablazione indipendente, 2069 dec, tools/actsa_watch.py):
        # deploy-NEUTRO — 0 flip sostanziali, tutti i flip = tie float fra azioni bit-identiche;
        # ha mostrato solo sharpening (temperatura sui gemelli); il decoder_fc condiviso d->1 era
        # una possibile spiegazione, non una causa dimostrata. V2 testa quella ipotesi con head separato.
        # DISARMATO sui launcher; il codice resta SOLO per caricare i ckpt storici 07-12..07-17.
        self.act_selfattn = int(act_selfattn)
        self.act_compare = int(act_compare)
        if self.act_selfattn and self.act_compare:
            raise ValueError("act_selfattn legacy e act_compare v2 sono esperimenti alternativi")
        if self.act_selfattn:
            self.act_sa_norm = torch.nn.LayerNorm(d_model)
            self.act_sa = torch.nn.MultiheadAttention(d_model, num_heads)
            torch.nn.init.zeros_(self.act_sa.out_proj.weight)
            torch.nn.init.zeros_(self.act_sa.out_proj.bias)
        # ATTENTION-V2: il token principale non viene modificato. L'attenzione confronta le azioni
        # candidate e il suo risultato ha una proiezione scalare DEDICATA, sommata al logit principale.
        # MultiheadAttention usa l'init PyTorch predefinita; soltanto l'ultima Linear parte a zero. Cosi'
        # il warm-start e' neutro, il compare_fc riceve gradiente al primo backward e l'MHA si sveglia
        # dal passo successivo. Nessun target oracle/auxiliary e nessuna regola specifica del gioco.
        if self.act_compare:
            self.act_compare_norm = torch.nn.LayerNorm(d_model)
            self.act_compare_sa = torch.nn.MultiheadAttention(d_model, num_heads)
            self.act_compare_fc = torch.nn.Linear(d_model, 1)
            torch.nn.init.zeros_(self.act_compare_fc.weight)
            torch.nn.init.zeros_(self.act_compare_fc.bias)
        # ACT-POINTER (--act-pointer, default 0 = OFF -> byte-identico, nessun modulo costruito).
        # Terzo tentativo sul confronto fra candidate, stavolta STRUTTURALE (AlphaStar/DI-star
        # TargetUnitHead, docs/act-pointer-design.md): act_sa e act_compare erano canali ADDITIVI
        # e la rete li ha ignorati entrambi (soluzione pigra, gradiente marginale); qui lo scoring
        # STESSO diventa comparativo — query(stato)·key(azione) al posto di decoder_fc(token).
        # Con act_pointer=1 decoder_fc resta allocato/nel ckpt per il rollback ma NON e' piu' nel
        # grafo (zero gradiente); compare resta INVARIATO (la variabile A/B e' solo il pointer).
        # Coesistenza: pointer+compare OK; pointer+act_sa legacy vietato (come legacy+compare).
        self.act_pointer = int(act_pointer)
        if self.act_pointer and self.act_selfattn:
            raise ValueError("act_pointer e act_sa legacy sono incompatibili (pointer+compare OK)")
        # PTR-BELIEF (--ptr-belief, ByteRL Fig.2: la query dell'inner-product legge l'LSTM embed):
        # la query del pointer legge [summary || belief]. Vive nella LARGHEZZA di fc1 — nessuna
        # chiave nuova, nessun modulo nuovo (vedi commento a _ZONES_KEYS/_POINTER_KEYS).
        self.ptr_belief = int(ptr_belief)
        if self.ptr_belief and not self.act_pointer:
            raise ValueError("ptr_belief richiede act_pointer=1 (e' la query del pointer che "
                             "legge il belief)")
        if self.ptr_belief and self.recurrent == "none":
            raise ValueError("ptr_belief richiede un modello ricorrente (senza belief non c'e' "
                             "query da condizionare)")
        if self.act_pointer:
            # DI-star fedele (action_arg_head.py:331-363): key = UNA Linear SENZA attivazione e
            # senza bias; query = MLP 2 layer ASIMMETRICO (ReLU solo dopo il primo); key_dim 32;
            # NIENTE scaling /sqrt(d). NAMING ptr_query_mlp (Sequential .0/.2) e NON fc1/fc2:
            # i suffissi fc1/fc2 collidono con le regole endswith e _NOISE_PAT di widen.py
            # (rumore silenzioso sul gate zero-init — review 07-24, vedi _POINTER_KEYS).
            _q1_in = d_model + (self.rec_hidden if self.ptr_belief else 0)
            self.ptr_key_fc = torch.nn.Linear(d_model, PTR_KEY_DIM, bias=False)
            self.ptr_query_mlp = torch.nn.Sequential(
                torch.nn.Linear(_q1_in, PTR_KEY_DIM), torch.nn.ReLU(),
                torch.nn.Linear(PTR_KEY_DIM, PTR_KEY_DIM))
            # REGOLA UNA-SOLA-A-ZERO (lezione act_compare): l'unico zero e' ptr_query_mlp[2].weight;
            # il bias = e0 rende la query q0 costante -> pointer == riga0(K)·token + ptr_bias.
            # Init di equivalenza anche a freddo: riga0(K)=w, ptr_bias=b di decoder_fc — cosi' la
            # costruzione fresh ha la stessa semantica della sintesi all'arm (migrate_act_pointer,
            # che comunque SOVRASCRIVE tutto il blocco al primo warm-start senza ptr_*).
            # Dinamica gradiente REALE a step 0 (misurata, review 07-24 — NON "key_fc tutto vivo"):
            # vivi fc2.weight/bias, ptr_bias e SOLO la riga 0 di K (q0 ha un'unica componente
            # non nulla); fc1 e le righe 1..31 di K hanno grad ESATTAMENTE zero e si svegliano
            # dopo 1 step SGD su fc2 (cascata a DUE gate). Gate D di test_act_pointer la asserisce.
            torch.nn.init.zeros_(self.ptr_query_mlp[2].weight)
            torch.nn.init.zeros_(self.ptr_query_mlp[2].bias)
            with torch.no_grad():
                self.ptr_query_mlp[2].bias[0] = 1.0
                self.ptr_key_fc.weight[0].copy_(self.decoder_fc.weight[0])
                if self.ptr_belief:
                    # colonne belief di fc1 a ZERO (gate stile LOC-tail): a costruzione fredda e
                    # all'arm la query e' identica alla versione solo-summary; NON e' un secondo
                    # zero del canale pointer (fc2 resta l'unico gate della cascata storica) —
                    # e' il gate del canale BELIEF dentro la query.
                    self.ptr_query_mlp[0].weight[:, d_model:].zero_()
            self.ptr_bias = torch.nn.Parameter(self.decoder_fc.bias.detach().clone().reshape(1))
        # delete the per-card id-memorization tables (the whole bet) -> lean
        del self.encoder_bag
        del self.decoder_bag
        # shared "card readers" (Cardsformer-style shared projection of the features)
        self.enc_card_mlp = torch.nn.Sequential(
            torch.nn.Linear(ENC_DENSE_DIM_TF, d_model), torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model))
        # LOC-TAIL: larghezza EFFETTIVA della riga encoder di QUESTO modello. int semplice e NON un
        # buffer, apposta: non deve entrare nello state_dict (lo confronterebbero i loader per shape).
        # I call-site la usano per armare `loc=` per-MODELLO, come `num_words` fa per `dkr=`.
        self.enc_in_dim = int(ENC_DENSE_DIM_TF)
        self.dec_action_mlp = torch.nn.Sequential(
            torch.nn.Linear(DEC_DENSE_DIM_TF, d_model), torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model))
        # tag embeddings that restore the own/opp/word-type + option-type signal the id-bags carried
        self.enc_role_embed = torch.nn.Embedding(N_ROLE, d_model)
        self.dec_plane_embed = torch.nn.Embedding(N_PLANE, d_model)
        # train-only auxiliary effect-prediction head (grounding "exam"); zero-init; dropped at deploy
        self.aux_eff_logits = torch.nn.Linear(d_model, EFF_VOCAB_DIM)
        self.aux_eff_deltas = torch.nn.Linear(d_model, EFF_DELTA_DIM)
        torch.nn.init.zeros_(self.aux_eff_logits.weight)
        torch.nn.init.zeros_(self.aux_eff_logits.bias)
        torch.nn.init.zeros_(self.aux_eff_deltas.weight)
        torch.nn.init.zeros_(self.aux_eff_deltas.bias)
        # ---- RECURRENT belief -> policy/value coupling (worktree gru-coupling) ----
        # Il core ricorrente (bytrl RecurrentNet: rnn/mem_proj/value_fc) esiste gia'; qui lo si
        # ACCOPPIA meglio a policy+attention e lo si fa partire come identita' (audit gru-coupling).
        # Costruito SOLO in modo ricorrente -> con recurrent='none' questo blocco non esiste e il
        # modello resta BIT-identico al feed-forward di produzione.
        self.gru_zones = int(gru_zones)
        if self.gru_zones and self.recurrent != "gru":
            raise ValueError("gru_zones richiede recurrent='gru' (l'ingresso per-zona e' del core; "
                             "il vec-server rifiuta comunque lstm)")
        if self.gru_zones and int(self.num_words) != int(NUM_WORDS):
            raise ValueError(f"gru_zones richiede num_words == NUM_WORDS dello spec corrente "
                             f"({self.num_words} != {NUM_WORDS}): le zone sono definite sul "
                             "layout vivo (spec.ZONE_WORDS), gli expert storici non si armano")
        if self.recurrent != "none":
            # GRU-ZONES (--gru-zones, ByteRL Fig.2): l'ingresso del core diventa
            # [summary || zone_proj(concat medie-PER-ZONA)] -> rnn RICOSTRUITO a input 2*d_model
            # PRIMA dell'igiene qui sotto (cosi' orthogonal/xavier girano sul modulo vero);
            # il gate a zero delle colonne zona viene applicato DOPO l'igiene.
            if self.gru_zones:
                self.rnn = torch.nn.GRU(2 * d_model, self.rec_hidden)
                zm = torch.zeros(N_ZONES, int(self.num_words))
                for _zi, _ws in enumerate(ZONE_WORDS):
                    for _wi in _ws:
                        zm[_zi, _wi] = 1.0 / len(_ws)
                # persistent=False: costante di LAYOUT, non deve entrare nello state_dict
                # (disciplina enc_in_dim: i loader confronterebbero le shape).
                self.register_buffer("zone_mix", zm, persistent=False)
                self.zone_proj = torch.nn.Linear(N_ZONES * d_model, d_model)
            # (B) IDENTITY-START (GTrXL b_g>0: blocco ~= identita' a init -> "prima il reattivo,
            #     poi la memoria"). mem_proj e value_fc a ZERO: a iter 0 la parola-belief e' neutra
            #     e V=0, cosi' il ramo gru parte ~= feed-forward invece che da rumore random
            #     (memoria gru-ab-experiment-deferred: "mem_proj/rnn/value_fc partono FRESH ->
            #     serve dare iter di ammortamento"). Qui l'ammortamento parte da un punto neutro.
            torch.nn.init.zeros_(self.mem_proj.weight)
            torch.nn.init.zeros_(self.mem_proj.bias)
            torch.nn.init.zeros_(self.value_fc.weight)
            torch.nn.init.zeros_(self.value_fc.bias)
            # GRU hygiene (best-practice R2D2/SB3-contrib/CleanRL): pesi RICORRENTI ortogonali (il
            # default PyTorch e' uniforme -1/sqrt(H)..), input xavier, bias 0 -> gradiente stabile
            # sul BPTT partita-intera (la ricorrenza gira sull'INTERA partita, 50-150 decisioni).
            for _n, _p in self.rnn.named_parameters():
                if _n.startswith("weight_hh"):
                    torch.nn.init.orthogonal_(_p)
                elif _n.startswith("weight_ih"):
                    torch.nn.init.xavier_uniform_(_p)
                elif _n.startswith("bias"):
                    torch.nn.init.zeros_(_p)
            if self.gru_zones:
                with torch.no_grad():
                    # UNICO gate a ZERO del canale zone (una-sola-a-zero): le colonne input nuove.
                    # zone_proj resta VIVO: uno zero doppio (zone_proj E colonne) avrebbe
                    # gradiente ESATTAMENTE 0 su entrambi per sempre (canale morto).
                    self.rnn.weight_ih_l0[:, d_model:].zero_()
            # (A) BELIEF->AZIONE affinity: canale DIRETTO e per-azione dal core al punteggio, ADDITIVO
            #     su main -> vive accanto a decoder_fc (act_pointer OFF) senza competere per lo scalare
            #     condiviso. belief_query zero-init -> termine ESATTAMENTE 0 a init; belief_key parte
            #     con gradiente 0 finche' belief_query=0 (cascata a 2 gate, come il pointer).
            self.belief_key = torch.nn.Linear(d_model, BELIEF_KEY_DIM, bias=False)
            self.belief_query = torch.nn.Linear(self.rec_hidden, BELIEF_KEY_DIM)
            torch.nn.init.zeros_(self.belief_query.weight)
            torch.nn.init.zeros_(self.belief_query.bias)

    # ---------------- decoder-layout / role warm-start ----------------
    def load_state_dict(self, state_dict, strict=True, **kw):
        """Apply only explicit, documented layout migrations before the normal PyTorch loader."""
        # LOC-TAIL (2026-08-07): PRIMA coda ENCODER-side. Un ckpt 1109 caricato su un modello armato
        # (1145) viene right-pad-ato con colonne ZERO: le colonne nuove nascono spente, il forward e'
        # invariato finche' il gradiente non le sveglia. Copre per costruzione TUTTI i loader che
        # passano dal metodo di classe (selfplay._build_model, imitation, evaluate, eval_runner,
        # agents/funcnet4_agent). osfp usa la funzione ESPORTATA prima del suo filtro-per-shape.
        state_dict = migrate_encoder_input_layout(state_dict, log=True)
        state_dict = migrate_decoder_input_layout(state_dict, log=True)
        # DKR-TAIL (2026-07-20): un ckpt pre-DKR (enc_role_embed 13 righe) caricato su un modello
        # armato (N_ROLE 15) viene PAD-ato con righe ZERO per i role nuovi (DECK_REMAINING/
        # PRIZE_KNOWN) — stesso pattern del pad token-mano di osfp (12->13, fork tf3->tf4). NB: qui
        # il forward NON e' bit-identico al vecchio (2 parole in piu' passano dal trunk): e' un
        # warm-start, non un no-op — il no-op del braccio e' il flag OFF (parole non costruite).
        # Copre anche gli EXPERT storici caricati in un processo armato: le loro chiamate encoder
        # restano a 36 parole (dkr mai passato), le righe pad non vengono mai indicizzate.
        tr = self.enc_role_embed.weight
        state_dict = migrate_role_embed_layout(
            state_dict, tr.shape[0], tr.shape[1], log=True)
        # ACT-POINTER: sintesi deterministica all'arm / strip al rollback. Prima migrazione
        # "a moduli" del loader (le due sopra sono di layout); osfp chiama la STESSA funzione
        # esportata prima del suo filtro-per-shape (vedi docstring di migrate_act_pointer).
        state_dict = migrate_act_pointer(state_dict, self.act_pointer, log=True)
        # PTR-BELIEF / GRU-ZONES (2026-08-13, worktree gru-byterl-coupling): migrazioni di
        # LARGHEZZA (pattern LOC-tail applicato a moduli). DOPO la sintesi del pointer (fc1 esiste
        # e qui si allarga/ritaglia); osfp chiama le STESSE funzioni esportate prima del suo
        # filtro-per-shape (doppio call-site obbligato, vedi docstring di migrate_act_pointer).
        state_dict = migrate_ptr_belief(state_dict, getattr(self, "ptr_belief", 0),
                                        rec_hidden=self.rec_hidden, log=True)
        state_dict = migrate_gru_zones(state_dict, getattr(self, "gru_zones", 0), log=True)
        return super().load_state_dict(state_dict, strict=strict, **kw)

    # ---------------- distributional value (only when value_bins > 0) ----------------
    def value_logits_from(self, encoder_out, belief):
        """(B, value_bins) logits. 'none': per-word logits meaned over words (mirrors the parent's
        mean-over-words of scalars); recurrent: logits from the belief."""
        if self.recurrent == "none":
            return self.value_dist_head(encoder_out).mean(0)                 # (W,B,N) -> (B,N)
        return self.value_dist_head(belief)                                  # (B,N)

    def value_from(self, encoder_out, belief):
        """Override: scalar expectation over the bin distribution when bins are on (NO tanh — the
        expectation of centers spanning [-1,1] is bounded by construction; tanh would warp calibration);
        parent behavior (tanh scalar head) when off."""
        if self.value_bins > 0:
            p = torch.softmax(self.value_logits_from(encoder_out, belief), dim=-1)
            return (p * self.value_bin_centers).sum(-1, keepdim=True)        # (B,1)
        return super().value_from(encoder_out, belief)

    # ---------------- encoder ----------------
    def encode(self, dense_e, role_e):
        """dense_e: (W*B, ENC_DENSE_DIM_TF); role_e: (W*B,) long ROLE id."""
        role_e = _guard_ids(role_e, N_ROLE, "ROLE")     # anti device-side-assert (traiettoria avvelenata)
        v = self.enc_card_mlp(dense_e) + self.enc_role_embed(role_e)     # (W*B, d)
        v = v.reshape(-1, self.num_words, self.d_model).transpose(0, 1)     # (W, B, d)
        encoder_out = self.encoder(v)
        return encoder_out, encoder_out.mean(0)

    # ---------------- decoder ----------------
    def action_tokens(self, encoder_out, belief, dense_d, plane_d):
        """(A,B,d) per-action representation BEFORE decoder_fc."""
        batch = encoder_out.size(1)
        plane_d = _guard_ids(plane_d, N_PLANE, "PLANE")  # anti device-side-assert (traiettoria avvelenata)
        p = self.dec_action_mlp(dense_d) + self.dec_plane_embed(plane_d)    # (A*B, d)
        p = p.reshape(batch, -1, self.d_model).transpose(0, 1)             # (A, B, d)
        if self.recurrent == "none":
            kv = encoder_out
        else:
            mem = self.mem_proj(belief).unsqueeze(0)
            kv = torch.cat([encoder_out, mem], dim=0)
        for layer in self.decoder:
            p = layer(p, kv)
        if self.act_selfattn:
            # blocco residuale pre-norm; key_padding_mask esclude i token PAD del batch learner
            # (un PAD non deve influenzare i token reali; i suoi score restano gia' esclusi
            # dalla loss via mask/-1e9). In generazione (B=1, nessun PAD) mask tutta-False.
            pad = plane_d.reshape(batch, -1) == PLANE_PAD          # (B, A) True = padding
            q = self.act_sa_norm(p)
            y, _ = self.act_sa(q, q, q, key_padding_mask=pad, need_weights=False)
            p = p + y
        return p

    def score(self, encoder_out, belief, dense_d, plane_d, summary=None):
        p = self.action_tokens(encoder_out, belief, dense_d, plane_d)
        # Kwarg GUARDATI (review 07-24): le sottoclassi storiche fanno override 2-arg di
        # score_from_tokens (test_act_compare._CountingNet) — un kwarg INCONDIZIONATO romperebbe la
        # regressione con TypeError. Passo `summary` SOLO col pointer e `belief` SOLO in modo
        # ricorrente; nel caso feed-forward+decoder_fc la call resta a 2 argomenti, byte-identica.
        kw = {}
        if belief is not None:
            kw["belief"] = belief
        if self.act_pointer:
            # I call-site legacy (losses._forward_traj, energy_probe) non passano summary:
            # ricalcolo BIT-identico a encode() — stessa mean(0) sullo stesso tensore. Quando il
            # chiamante lo ha gia' (forward, learner) lo passa e si evita la riduzione doppia.
            if summary is None:
                summary = encoder_out.mean(0)
            kw["summary"] = summary
        return self.score_from_tokens(p, plane_d, **kw)

    def policy_components_from_tokens(self, action_tokens, plane_d, summary=None, belief=None):
        """Return raw ``(main, compare)`` scores, both shaped (A,B,1).

        Exposed for controlled diagnostics; production and every learner path call
        :meth:`score_from_tokens`, so there is one policy implementation. Con ``act_pointer=1``
        *main* E' il pointer (query(summary)·key(token) + ptr_bias, decoder_fc fuori dal grafo)
        e ``summary`` — il (B,d) che ``encode()`` gia' restituisce — e' OBBLIGATORIO.
        """
        if self.act_pointer:
            if summary is None:
                # MAI silenzioso: un call-site che dimentica il summary produrrebbe una policy
                # sbagliata con la shape giusta — la classe di bug fail-soft (lezione DKR-random).
                raise RuntimeError(
                    "act_pointer=1 richiede summary (il (B,d_model) restituito da encode()); "
                    "chiamata senza summary = scoring silenziosamente sbagliato, vietato")
            q_in = summary
            # PTR-BELIEF (ByteRL Fig.2: la query dell'inner-product legge l'LSTM embed): la query
            # legge [summary || belief]. belief OBBLIGATORIO ad arm attivo e MAI un default
            # silenzioso (lezione DKR-random): senza raise la shape non tornerebbe comunque
            # (fc1 larga), ma il messaggio deve dire il PERCHE'.
            if getattr(self, "ptr_belief", 0):
                if belief is None:
                    raise RuntimeError(
                        "ptr_belief=1 richiede belief al momento dello scoring (il (B,rec_hidden) "
                        "del core ricorrente); chiamata senza belief = query monca, vietato")
                q_in = torch.cat([summary, belief], dim=-1)
            q = self.ptr_query_mlp(q_in)                          # (B, 32)
            k = self.ptr_key_fc(action_tokens)                    # (A, B, 32)
            # dot product DI-star: NIENTE scaling /sqrt(d); all'arm q=e0 -> la somma sulle 32 dim
            # aggiunge 31 zeri ESATTI (0*k_j = 0.0) e main == w·token + b = decoder_fc(token).
            main = (q.unsqueeze(0) * k).sum(-1, keepdim=True) + self.ptr_bias   # (A, B, 1)
        else:
            main = self.decoder_fc(action_tokens)
        # BELIEF->AZIONE affinity (ricorrente): q(belief)·k(token) ADDITIVO su main, per-azione.
        # belief e' il (B,rec_hidden) del core (None fuori dal ramo ricorrente); belief_query zero-init
        # -> a init il termine e' ESATTAMENTE 0 (main invariato) e il gradiente lo sveglia. Somma su
        # main -> ininfluente per act_pointer/act_compare (il canale e' indipendente dallo scalare).
        if belief is not None and getattr(self, "belief_query", None) is not None:
            bq = self.belief_query(belief)                        # (B, BELIEF_KEY_DIM)
            bk = self.belief_key(action_tokens)                   # (A, B, BELIEF_KEY_DIM)
            main = main + (bq.unsqueeze(0) * bk).sum(-1, keepdim=True)          # (A, B, 1)
        compare = torch.zeros_like(main)
        if self.act_compare:
            batch = action_tokens.size(1)
            expected = action_tokens.size(0) * batch
            if plane_d.numel() != expected:
                raise ValueError(f"plane_d has {plane_d.numel()} ids, expected {expected}")
            pad = plane_d.reshape(batch, -1).eq(PLANE_PAD)       # (B,A), True = key da ignorare
            q = self.act_compare_norm(action_tokens)
            y, _ = self.act_compare_sa(q, q, q, key_padding_mask=pad, need_weights=False)
            compare = self.act_compare_fc(y)
        return main, compare

    def score_from_tokens(self, action_tokens, plane_d, summary=None, belief=None):
        """Convert decoder tokens to one bounded score per candidate action.

        ``summary`` (kwarg) e' richiesto SOLO con act_pointer=1; ``belief`` (kwarg) attiva il canale
        belief->azione SOLO in modo ricorrente. Entrambi None -> firma a 2 argomenti valida per i
        call-site e le sottoclassi storiche (act_pointer=0, feed-forward)."""
        main, compare = self.policy_components_from_tokens(
            action_tokens, plane_d, summary=summary, belief=belief)
        batch = action_tokens.size(1)
        raw = main + compare
        return torch.tanh(raw.transpose(0, 1).reshape(batch, -1))

    def aux_predict(self, action_tokens, chosen_rows):
        """Train-only grounding head. action_tokens: (A,B,d); chosen_rows: (B,) long -> the chosen action's
        token per sample -> (logits (B,EFF_VOCAB_DIM), deltas (B,EFF_DELTA_DIM))."""
        at = action_tokens.transpose(0, 1)                                  # (B, A, d)
        sel = at[torch.arange(at.size(0), device=at.device), chosen_rows]   # (B, d)
        return self.aux_eff_logits(sel), self.aux_eff_deltas(sel)

    # ---------------- recurrent input ----------------
    def recur(self, summary_seq, state):
        """Override GUARDIA (review 08-13): su modello gru_zones un call-site che passa il solo
        summary (larghezza d_model) fallirebbe dentro nn.GRU con uno shape-error criptico —
        qui il messaggio dice DOVE sta il problema (la flotta watcher pre-coupling chiama
        recur(summ) direttamente; vedi docs/gru-byterl-coupling.md). Con zones OFF: pass-through.
        Gestisce anche PackedSequence (percorso packed del learner)."""
        if getattr(self, "gru_zones", 0):
            data = (summary_seq.data if isinstance(summary_seq, torch.nn.utils.rnn.PackedSequence)
                    else summary_seq)
            if int(data.shape[-1]) != 2 * self.d_model:
                raise RuntimeError(
                    f"gru_zones=1: recur() vuole l'input di recur_input(encoder_out, summary) "
                    f"(larghezza attesa {2 * self.d_model}, ricevuta {int(data.shape[-1])}); "
                    "questo call-site non e' stato portato a recur_input "
                    "(docs/gru-byterl-coupling.md, sezione Watcher)")
        return super().recur(summary_seq, state)

    def recur_input(self, encoder_out, summary):
        """Input del core ricorrente. Default: il summary (media globale, la cannuccia storica).

        Con gru_zones (ByteRL Fig.2): [summary || zone_proj(concat medie-PER-ZONA)] (B, 2d) —
        mano/bench/mazzo compressi ognuno per conto suo, mai mescolati prima della memoria.
        UNICO punto che costruisce l'input del GRU: forward, losses (_forward_traj / packed /
        rami aux) e verify_packed DEVONO passare da qui. Un call-site che chiama recur() col solo
        summary su un modello armato esplode di shape (fail-loud voluto, mai numeri plausibili)."""
        if not getattr(self, "gru_zones", 0):
            return summary
        w, b, d = encoder_out.shape
        z = torch.matmul(self.zone_mix.to(dtype=encoder_out.dtype),
                         encoder_out.reshape(w, b * d))                      # (Z, B*d)
        z = z.reshape(-1, b, d).permute(1, 0, 2).reshape(b, -1)              # (B, Z*d)
        return torch.cat([summary, self.zone_proj(z)], dim=-1)

    # ---------------- single-step forward ----------------
    def forward(self, dense_e, role_e, dense_d, plane_d, state=None):
        encoder_out, summary = self.encode(dense_e, role_e)
        # ACT-POINTER: il summary di encode() (lo stesso vettore del value head) alimenta la query
        # del pointer — passarlo evita la mean(0) doppia dentro score(). Chiamata GUARDATA: con
        # act_pointer=0 la call resta a 4 argomenti, identica a prima (back-compat sottoclassi).
        if self.recurrent == "none":
            value = self.value_from(encoder_out, None)
            if self.act_pointer:
                policy = self.score(encoder_out, None, dense_d, plane_d, summary=summary)
            else:
                policy = self.score(encoder_out, None, dense_d, plane_d)
            return value, policy, None
        batch = summary.size(0)
        if state is None:
            state = self.initial_state(batch, summary.device)
        belief_seq, new_state = self.recur(
            self.recur_input(encoder_out, summary).unsqueeze(0), state)
        belief = belief_seq.squeeze(0)
        value = self.value_from(encoder_out, belief)
        if self.act_pointer:
            # query belief-condizionata = lavoro futuro (spec §4): anche nel ramo recurrent la
            # query legge il summary, il belief resta per value/mem_proj come oggi.
            policy = self.score(encoder_out, belief, dense_d, plane_d, summary=summary)
        else:
            policy = self.score(encoder_out, belief, dense_d, plane_d)
        return value, policy, new_state
