"""funcnet — functional-feature encoding package (Phase 1).

A net that reasons about cards by FUNCTION (CardData/Attack stats) instead of IDENTITY, so it
generalizes to unseen cards. Asymmetric: OUR board = id-embedding + functional features; OPPONENT
board = functional features ONLY (id used only to look up features, never embedded). Reuses
bytrl.losses MATH unchanged; only the tensor-assembly/forward plumbing that must carry dense rows is
forked. See funcnet/PLAN.md.
"""
