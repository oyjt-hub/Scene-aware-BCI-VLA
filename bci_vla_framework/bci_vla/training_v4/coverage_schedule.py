"""Training-only full-row coverage, independent of labels and model methods."""
from collections.abc import Mapping
import random


def coverage_schedule(rows, seed, steps):
    """Return deterministic row indices with no repeat until each round is full.

    Each round independently shuffles all input row indices exactly once, so a
    complete dataset round covers every complete choice group already present
    in the validated dataset. A partial final round is allowed. Validation,
    calibration and test rows are rejected even when steps=0. Only the public
    supervision split is inspected: labels, goals, targets and predictions do
    not affect sampling. Build once and reuse across compared methods.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('nonempty training rows required')
    if type(seed) is not int or type(steps) is not int or steps < 0:
        raise ValueError('integer seed and nonnegative integer steps required')
    # Validate the entire input BEFORE considering the requested number of
    # steps; a zero-length schedule must not silently authorize sealed rows.
    for row in rows:
        supervision = row.get('supervision') if isinstance(row, Mapping) else None
        if not isinstance(supervision, Mapping) or supervision.get('split') != 'train':
            raise ValueError('coverage schedule accepts train only; other partitions are forbidden')
    rng = random.Random(seed)
    result = []
    while len(result) < steps:
        order = list(range(len(rows)))
        rng.shuffle(order)
        result.extend(order[:steps - len(result)])
    return result
