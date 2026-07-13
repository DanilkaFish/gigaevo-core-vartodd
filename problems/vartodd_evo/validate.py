from typing import Tuple
import re

import numpy as np
from helper import Matrix, Tensor3D, get_matrix

# Fitness shaping (units: ranks). Rank stays dominant: the low-limit reward is
# a sub-rank tiebreaker, the penalties mark wasted work without hiding rank.
NO_IMPROVEMENT_PENALTY = 1.0
OVER_LIMIT_PENALTY = 0
LOW_LIMIT_REWARD_MAX = 0
LOW_LIMIT_SCALE = 2000  # small_limit scout band of the Live Path Store


def _loaded_path_limit(text: str) -> int | None:
    m = re.search(r"loaded_path_name:.*?limit_bucket=(-1\(full\)|\d+)", text)
    if m is None:
        return None
    return -1 if m.group(1).startswith("-1") else int(m.group(1))


def _child_limit_buckets(text: str) -> int | None:
    """Effective z recall of the converged profiles: TODD keep=0 counts as 0,
    any full search (-1) dominates, otherwise the largest finite cap."""
    profiles = re.findall(
        r"pool=final:\d+/tohpe:\d+/\d+/todd:(\d+)/\d+"
        r"[^\n]*?z_buckets=min_buckets:\d+/max_buckets:\d+/limit_bucket:(-?\d+)",
        text,
    )
    if not profiles:
        return None
    limits = [0 if int(keep) == 0 else int(limit) for keep, limit in profiles]
    if any(limit < 0 for limit in limits):
        return -1
    return max(limits)


def validate(
    result: Tuple[np.ndarray, str, str]
) -> dict[str, float | str]:
    context = get_matrix()
    result, report, best_path = result
    full_report = report + best_path
    loaded_path_match = re.search(r"\bloaded_path_rank:\s*(\d+)", full_report)
    loaded_match = re.search(r"\bloaded_rank:\s*(\d+)", full_report)
    initial_match = re.search(r"\binitial_rank:\s*(\d+)", full_report)
    found_rank = int(result.shape[0])

    # Strategy descriptor for MAP-Elites: the rank at which the program started
    # searching. Ab-initio runs start at initial_rank (~1701); saved-path
    # refiners start at their branch point (varies by path+margin). Exposing it
    # as a behavior dimension keeps distinct load strategies in distinct archive
    # cells instead of collapsing them into one fitness/runtime niche.
    if loaded_match is not None:
        loaded_rank = float(loaded_match.group(1))
    elif initial_match is not None:
        loaded_rank = float(initial_match.group(1))
    else:
        loaded_rank = 1701.0

    if np.any(Tensor3D(context) != Tensor3D(Matrix.from_numpy(result))):
        return {"fitness": 1290.0,
                "is_valid": 0.0,
                "loaded_rank": loaded_rank,
                "aux info": full_report,
                }

    base_fitness = result.shape[0] + np.sum(result) / np.size(result)
    penalty = 0.0
    improved = True
    rank_to_beat_match = loaded_path_match or loaded_match
    if rank_to_beat_match is not None:
        rank_to_beat = int(rank_to_beat_match.group(1))
        if found_rank >= rank_to_beat:
            improved = False
            penalty += NO_IMPROVEMENT_PENALTY

    child_limit = _child_limit_buckets(full_report)
    parent_limit = _loaded_path_limit(full_report)
    # Spending more z recall than the loaded path without beating it is the
    # most expensive way to not improve; charge it on top.
    if not improved and child_limit is not None and parent_limit is not None:
        child_full = child_limit < 0
        parent_full = parent_limit < 0
        if (child_full and not parent_full) or (
            not child_full and not parent_full and child_limit > parent_limit
        ):
            penalty += OVER_LIMIT_PENALTY

    reward = 0.0
    if child_limit is not None and child_limit >= 0:
        reward = LOW_LIMIT_REWARD_MAX * max(0.0, 1.0 - child_limit / LOW_LIMIT_SCALE)
    return {"fitness": base_fitness + penalty - reward,
            "is_valid": 1.0,
            "loaded_rank": loaded_rank,
            "aux info": full_report,
            }
