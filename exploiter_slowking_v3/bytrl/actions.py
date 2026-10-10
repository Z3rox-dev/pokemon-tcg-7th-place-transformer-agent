"""Legal-action enumeration (ported from ml/search.py) so the training pipeline is self-contained.

The engine hands us a `select` with `option` (legal options) and `maxCount`; we enumerate the
maxCount-sized index combinations (capped at 64), exactly as the official sample does. We RANK these
(we do not generate type->target), so this is the candidate set the decoder scores.
"""


def enumerate_actions(select):
    """All maxCount-sized index combinations over select.option (capped at 64)."""
    actions = []
    indices = list(range(select.maxCount))
    n = len(select.option)
    for _ in range(64):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < n - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions


def match_action(actions, selection):
    """Index of `selection` (a list of option indices) within the enumerated `actions`, or None."""
    target = sorted(selection)
    for k, a in enumerate(actions):
        if a == target:
            return k
    return None
