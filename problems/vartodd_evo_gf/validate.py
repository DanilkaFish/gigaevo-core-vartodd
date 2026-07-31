from typing import Tuple
import re

import numpy as np
from helper import Matrix, Tensor3D, get_matrix

# Fitness shaping in rank units. A loaded-path child should normally beat the
# rank it reuses. Reproducing the same rank with a smaller configured TODD cap
# remains admissible, but still pays a small penalty.
NO_IMPROVEMENT_PENALTY = 7.0
REDUCED_CAP_PENALTY = 1.0


def _loaded_path_todd_limit(text: str) -> int | None:
    m = re.search(r"loaded_path_name:[^\n]*?_z(?:\d+|unknown)of(-?\d+)", text)
    if m is None:
        m = re.search(r"loaded_path_name:[^\n]*?_lim(-?\d+)", text)
    if m is None:
        return None
    return int(m.group(1))


def _child_path_todd_limit(text: str) -> int | None:
    match = re.search(
        r"this path name:[^\n]*?_z(?:\d+|unknown)of(-?\d+)(?=_|\s|$)",
        text,
    )
    if match is None:
        match = re.search(
            r"this path name:[^\n]*?_lim(-?\d+)(?=_|\s|$)",
            text,
        )
    return int(match.group(1)) if match is not None else None


def _is_smaller_todd_limit(parent: int | None, child: int | None) -> bool:
    if parent is None or child is None or child < 0:
        return False
    return parent < 0 or child < parent


def _child_todd_limit_buckets(text: str) -> int | None:
    """Return only the effective TODD cap from converged policy profiles.

    TOHPEprefix may have its own z traversal, but validation penalization and
    saved-path recall classification deliberately measure full-TODD work only.
    A disabled TODD source contributes zero; an active full TODD search (-1)
    dominates finite TODD caps.
    """
    limits: list[int] = []
    for line in text.splitlines():
        pool_match = re.search(
            r"pool=final:\d+/tohpe:\d+/\d+"
            r"(?:/tohpeprefix:\d+/\d+)?/todd:(\d+)/\d+",
            line,
        )
        if pool_match is None:
            continue
        todd_limit_match = re.search(
            r"z_buckets=[^\n]*?\btodd:\d+\.\.\d+/(-?\d+)", line
        )
        if todd_limit_match is None:
            # Compatibility with the legacy two-source profile rendering,
            # where its sole z-bucket field is TODD's field.
            todd_limit_match = re.search(
                r"z_buckets=min_buckets:\d+/max_buckets:\d+/limit_bucket:(-?\d+)",
                line,
            )
        if todd_limit_match is not None:
            limits.append(
                0
                if int(pool_match.group(1)) == 0
                else int(todd_limit_match.group(1))
            )
    if not limits:
        return None
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

    # Retention must know whether this search actually beat the rank it was
    # asked to improve.  Saved-path runs use their loaded path's final rank;
    # ab-initio runs use their initial rank.  This is intentionally separate
    # from fitness so the archive selector can enforce it before fitness.
    rank_to_beat_match = loaded_path_match or loaded_match or initial_match
    rank_improved = float(
        rank_to_beat_match is not None
        and found_rank < int(rank_to_beat_match.group(1))
    )

    if np.any(Tensor3D(context) != Tensor3D(Matrix.from_numpy(result))):
        return {"fitness": 1290.0,
                "is_valid": 0.0,
                "loaded_rank": loaded_rank,
                "rank_improved": 0.0,
                "aux info": full_report,
                }

    base_fitness = result.shape[0] + np.sum(result) / np.size(result)
    penalty = 0.0
    # Keep GF16's existing fitness shaping unchanged: only a reused/loaded
    # branch pays the no-improvement penalty.  ``rank_improved`` above is
    # nevertheless available for archive retention on both fresh and loaded
    # searches.
    fitness_rank_to_beat_match = loaded_path_match or loaded_match
    if fitness_rank_to_beat_match is not None:
        rank_to_beat = int(fitness_rank_to_beat_match.group(1))
        if found_rank >= rank_to_beat:
            parent_limit = _loaded_path_todd_limit(full_report)
            child_limit = _child_path_todd_limit(full_report)
            penalty += (
                REDUCED_CAP_PENALTY
                if found_rank == rank_to_beat
                and _is_smaller_todd_limit(parent_limit, child_limit)
                else NO_IMPROVEMENT_PENALTY
            )

    return {"fitness": base_fitness + penalty,
            "is_valid": 1.0,
            "loaded_rank": loaded_rank,
            "rank_improved": rank_improved,
            "aux info": full_report,
            }
