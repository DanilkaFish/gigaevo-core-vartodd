"""Regime-specific prompt evidence for the isolated VarTODD island pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
from typing import Any

from gigaevo.evolution.strategies.base import (
    MutationRoute,
    ParentRole,
)
from gigaevo.programs.core_types import StageIO
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.collector import (
    EvolutionaryStatisticsCollector,
)
from gigaevo.programs.stages.common import Box, StringContainer
from gigaevo.programs.stages.stage_registry import StageRegistry

_CONTEXT_BLOCK_RE = re.compile(
    r"(?ms)^## (?:Execution Signal Digest|Program Aux Excerpt)\b"
    r".*?(?=^---$|^## |\Z)"
)
_REGIME_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_AUX_SECTION_ORDER = (
    "path_summary",
    "path_policy_groups",
    "converged_policy_profiles",
    "search_stat",
)
_REFINEMENT_ROUTES = frozenset({"mid_margin", "near_end"})
_INSIGHTS_ROUTE_PROFILES = {
    "ab_initio": "ab_initio",
    "mid_margin": "path_refinement",
    "near_end": "path_refinement",
}


def _load_problem_path_store(problem_dir: Path):
    problem_dir = problem_dir.resolve()
    module_name = f"_gigaevo_vartodd_islands_store_{abs(hash(str(problem_dir)))}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    variant_path = problem_dir / "variant.py"
    path_store_path = problem_dir / "path_store.py"
    if not variant_path.exists() or not path_store_path.exists():
        raise ImportError(f"incomplete VarTODD island problem at {problem_dir}")

    variant_name = f"{module_name}_variant"
    variant_spec = importlib.util.spec_from_file_location(
        variant_name,
        variant_path,
    )
    if variant_spec is None or variant_spec.loader is None:
        raise ImportError(f"cannot load variant.py from {problem_dir}")
    variant = importlib.util.module_from_spec(variant_spec)
    sys.modules[variant_name] = variant
    variant_spec.loader.exec_module(variant)

    spec = importlib.util.spec_from_file_location(module_name, path_store_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load path_store.py from {problem_dir}")
    module = importlib.util.module_from_spec(spec)
    previous_variant = sys.modules.get("variant")
    sys.modules[module_name] = module
    sys.modules["variant"] = variant
    sys.path.insert(0, str(problem_dir))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.remove(str(problem_dir))
        if previous_variant is None:
            sys.modules.pop("variant", None)
        else:
            sys.modules["variant"] = previous_variant
    return module


class _PathStoreClient:
    def __init__(
        self,
        *,
        problem_dir: str | Path,
        root_dir: str | Path | None = None,
    ):
        self.problem_dir = Path(problem_dir)
        self.root_dir = Path(root_dir) if root_dir is not None else None

    def _path_store(self):
        module = _load_problem_path_store(self.problem_dir)
        if self.root_dir is None:
            return module.PathStore()
        return module.PathStore(root_dir=str(self.root_dir))


class VartoddIslandsRouteContextProvider(_PathStoreClient):
    """Build route assignments, filtered parent evidence, and shared cards."""

    def __init__(
        self,
        *,
        problem_dir: str | Path,
        root_dir: str | Path | None = None,
        near_end_path_top_k: int = 4,
        mid_margin_path_top_k: int = 8,
        selectable_paths_per_rank: int = 2,
        path_cards_max_chars: int = 24_000,
        parent_aux_max_chars: int = 12_000,
    ):
        super().__init__(problem_dir=problem_dir, root_dir=root_dir)
        self.near_end_path_top_k = max(0, int(near_end_path_top_k))
        self.mid_margin_path_top_k = max(0, int(mid_margin_path_top_k))
        self.selectable_paths_per_rank = max(
            0,
            int(selectable_paths_per_rank),
        )
        self.path_cards_max_chars = max(0, int(path_cards_max_chars))
        self.parent_aux_max_chars = max(0, int(parent_aux_max_chars))

    def route_is_available(self, route: MutationRoute) -> bool:
        if route.context_profile == "ab_initio":
            return True
        if route.context_profile == "path_refinement":
            if (
                self._route_top_k(route) <= 0
                or self.selectable_paths_per_rank <= 0
            ):
                return False
            return bool(
                self._path_store().has_selectable_paths(
                    route_id=route.regime_id,
                )
            )
        return True

    def _route_top_k(self, route: MutationRoute) -> int:
        if route.regime_id == "near_end":
            return self.near_end_path_top_k
        if route.regime_id == "mid_margin":
            return self.mid_margin_path_top_k
        return 0

    def build_route_guidance(self, route: MutationRoute) -> str:
        regime_id = route.regime_id
        path = (
            self.problem_dir
            / "prompts"
            / "islands"
            / f"{regime_id}.txt"
        )
        if _REGIME_ID_RE.fullmatch(regime_id) is None:
            raise ValueError(
                "unsafe mutation regime id for prompt overlay "
                f"{regime_id!r}; expected path: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"missing prompt overlay for route {regime_id!r}: {path}"
            )
        try:
            guidance = path.read_text(encoding="utf-8").strip()
        except UnicodeError as exc:
            raise UnicodeError(
                f"cannot decode prompt overlay for route {regime_id!r}: {path}"
            ) from exc
        except OSError as exc:
            raise OSError(
                f"cannot read prompt overlay for route {regime_id!r}: {path}"
            ) from exc
        if not guidance:
            raise ValueError(
                f"blank prompt overlay for route {regime_id!r}: {path}"
            )
        return guidance

    def build_insights_context(self, program: Program) -> str:
        route_id = next(
            (
                value
                for key in (
                    "mutation_regime",
                    "target_island",
                    "current_island",
                )
                if isinstance((value := program.metadata.get(key)), str)
                and value in _INSIGHTS_ROUTE_PROFILES
            ),
            None,
        )
        if route_id is None:
            return ""
        return self.build_route_guidance(
            MutationRoute(
                regime_id=route_id,
                island_id=route_id,
                context_profile=_INSIGHTS_ROUTE_PROFILES[route_id],
            )
        )

    def build_assignment(
        self,
        route: MutationRoute,
        parents: list[Program],
        parent_roles: tuple[ParentRole, ...],
    ) -> str:
        lines = [
            "## Mutation Assignment",
            f"target_regime: {route.regime_id}",
            f"destination_island: {route.island_id}",
        ]
        for index, role in enumerate(parent_roles, start=1):
            lines.append(f"parent_{index}_role: {role}")
        return "\n".join(lines)

    def filter_parent_context(
        self,
        route: MutationRoute,
        parent: Program,
        role: ParentRole,
        mutation_context: str,
    ) -> str:
        del role
        base_context = _strip_generic_execution_blocks(mutation_context)
        aux = str(parent.metadata.get("aux_info") or "")
        if route.context_profile == "ab_initio":
            evidence = _scrub_path_handles(aux)
        elif route.regime_id == "near_end":
            evidence = _near_end_aux(aux)
        elif route.context_profile == "path_refinement":
            evidence = _mid_margin_aux(aux)
        else:
            evidence = ""
        parts = [base_context.strip()]
        if evidence.strip():
            parts.append("## Route-specific execution evidence\n\n" + evidence.strip())
        return _bounded_middle(
            "\n\n".join(part for part in parts if part),
            self.parent_aux_max_chars,
        )

    def build_external_context(self, route: MutationRoute) -> str:
        if route.context_profile != "path_refinement":
            return ""
        return self._path_store().render_selectable_path_cards(
            route_id=route.regime_id,
            top_k=self._route_top_k(route),
            max_per_rank=self.selectable_paths_per_rank,
            max_chars=self.path_cards_max_chars,
        )


def _resolve_island(program: Program) -> str | None:
    current = program.metadata.get("current_island")
    if isinstance(current, str) and current:
        return current
    target = program.metadata.get("target_island")
    if isinstance(target, str) and target:
        return target
    if program.metadata.get("source") == "initial_program":
        return "ab_initio"
    return None


@StageRegistry.register(description="Island-local evolutionary statistics collector")
class IslandEvolutionaryStatisticsCollector(EvolutionaryStatisticsCollector):
    """Compute archive statistics only inside the focal program's island."""

    _EXCLUDE = EXCLUDE_STAGE_RESULTS

    async def _collect_programs(self, program: Program) -> list[Program]:
        programs = await self.storage.snapshot.get_all(
            self.storage,
            exclude=self._EXCLUDE,
        )
        focal_island = _resolve_island(program)
        if focal_island is None:
            return programs
        return [
            candidate
            for candidate in programs
            if _resolve_island(candidate) == focal_island
        ]


class PathCardEnrichmentInputs(StageIO):
    metrics: Box[dict[str, float]] | None
    non_metrics: Box[dict[str, str]] | None
    runtime: Box[dict[str, float]] | None


@StageRegistry.register(description="Attach producer evidence to a saved path card")
class PathCardEnrichmentStage(_PathStoreClient, Stage):
    InputsModel = PathCardEnrichmentInputs
    OutputModel = StringContainer
    cacheable: bool = False

    def __init__(
        self,
        *,
        problem_dir: str | Path,
        root_dir: str | Path | None = None,
        **kwargs: Any,
    ):
        Stage.__init__(self, **kwargs)
        _PathStoreClient.__init__(
            self,
            problem_dir=problem_dir,
            root_dir=root_dir,
        )

    async def compute(self, program: Program) -> StageIO:
        params: PathCardEnrichmentInputs = self.params
        aux = ""
        if params.non_metrics is not None:
            aux = (
                params.non_metrics.data.get("aux info")
                or params.non_metrics.data.get("aux_info")
                or ""
            )
        name_match = re.search(r"(?m)^this path name:\s*(\S+)\s*$", aux)
        if name_match is None:
            return StringContainer(data="")

        total_evals = _match_int(aux, r"(?m)^total_evals:\s*(\d+)")
        best_seen_times = _match_int(
            aux,
            r"(?m)^best_seen_times:\s*(\d+)",
        )
        last_improvement_match = re.search(
            r"\blast_improvement[=:](\d+/\d+)",
            aux,
        )
        timeout_match = re.search(
            r"(?m)^timeout_salvaged:\s*([01])\s*$",
            aux,
        )
        runtime = None
        if params.runtime is not None:
            runtime = params.runtime.data.get("runtime")
        metrics = params.metrics.data if params.metrics is not None else {}
        producer = {
            "metrics": dict(metrics),
            "runtime": runtime,
            "total_evals": total_evals,
            "best_seen_times": best_seen_times,
            "last_improvement": (
                last_improvement_match.group(1)
                if last_improvement_match is not None
                else None
            ),
            "timeout_salvaged": (
                timeout_match is not None and timeout_match.group(1) == "1"
            ),
        }
        name = name_match.group(1)
        store = self._path_store()
        store.update_evidence_card(name, producer)

        route_id = program.metadata.get("mutation_regime")
        if route_id not in _REFINEMENT_ROUTES:
            route_id = program.metadata.get("target_island")
        if route_id in _REFINEMENT_ROUTES:
            loaded_name_match = re.search(
                r"(?m)^loaded_path_name:\s*(\S+)\s*$",
                aux,
            )
            loaded_rank = _match_int(
                aux,
                r"(?m)^loaded_path_rank:\s*(\d+)",
            )
            child_rank = _match_int(aux, r"\bfinal_rank=(\d+)")
            try:
                is_valid = float(metrics.get("is_valid", 0.0)) == 1.0
            except (TypeError, ValueError):
                is_valid = False
            if (
                is_valid
                and loaded_name_match is not None
                and loaded_rank is not None
                and child_rank is not None
            ):
                store.record_route_result(
                    loaded_name_match.group(1),
                    route_id=str(route_id),
                    program_id=program.id,
                    loaded_rank=loaded_rank,
                    child_rank=child_rank,
                )
        return StringContainer(data=name)


def _strip_generic_execution_blocks(text: str) -> str:
    stripped = _CONTEXT_BLOCK_RE.sub("", text)
    stripped = re.sub(r"(?m)^\s*---\s*$", "", stripped)
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def _scrub_path_handles(text: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if not re.search(
            r"\b(?:loaded_path_name|this path name|path_name)\s*:",
            line,
        )
    ]
    return "\n".join(lines).strip()


def _extract_aux_section(aux: str, name: str) -> str:
    marker = f"{name}:"
    start = aux.find(marker)
    if start < 0:
        return ""
    tail = aux[start:]
    following = _AUX_SECTION_ORDER[_AUX_SECTION_ORDER.index(name) + 1 :]
    stops = [
        tail.find(f"{next_name}:")
        for next_name in following
        if f"{next_name}:" in tail
    ]
    if name == "search_stat":
        boundary = re.search(r"(?m)^total_evals:\s*", tail)
        if boundary is not None:
            stops.append(boundary.start())
    if stops:
        tail = tail[: min(stop for stop in stops if stop > 0)]
    return tail.strip()


def _mid_margin_aux(aux: str) -> str:
    sections = [
        _extract_aux_section(aux, name)
        for name in _AUX_SECTION_ORDER
    ]
    fields = _selected_aux_fields(aux)
    return "\n\n".join(part for part in (*sections, fields) if part)


def _near_end_aux(aux: str) -> str:
    groups = _extract_aux_section(aux, "path_policy_groups")
    group_lines = [
        line
        for line in groups.splitlines()
        if re.match(r"^\s*g\d+\s", line)
    ][-3:]
    profile_ids = {
        profile
        for line in group_lines
        for profile in re.findall(r"\bP\d+\b", line)
    }
    profiles = _extract_aux_section(aux, "converged_policy_profiles")
    profile_lines = [
        line
        for line in profiles.splitlines()
        if any(re.search(rf"\b{re.escape(pid)}\b", line) for pid in profile_ids)
        or line.strip() == "scores:"
    ]
    parts = [
        _extract_aux_section(aux, "path_summary"),
        (
            "path_policy_groups:\n" + "\n".join(group_lines)
            if group_lines
            else ""
        ),
        (
            "converged_policy_profiles:\n" + "\n".join(profile_lines)
            if profile_lines
            else ""
        ),
        _extract_aux_section(aux, "search_stat"),
        _selected_aux_fields(aux),
    ]
    return "\n\n".join(part for part in parts if part)


def _selected_aux_fields(aux: str) -> str:
    prefixes = (
        "total_evals:",
        "best_seen_times:",
        "loaded_path_improved:",
        "initial_rank:",
        "loaded_rank:",
        "loaded_path_rank:",
        "init_rank_thr:",
        "timeout_salvaged:",
    )
    return "\n".join(
        line for line in aux.splitlines() if line.strip().startswith(prefixes)
    )


def _bounded_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n\n... route evidence truncated ...\n\n"
    available = max(0, max_chars - len(marker))
    left = available // 2
    right = available - left
    return text[:left] + marker + text[-right:]


def _match_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match is not None else None


__all__ = [
    "IslandEvolutionaryStatisticsCollector",
    "PathCardEnrichmentInputs",
    "PathCardEnrichmentStage",
    "VartoddIslandsRouteContextProvider",
]
