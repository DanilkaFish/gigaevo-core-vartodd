from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from typing import Any, Sequence, Tuple, cast, Optional

from loguru import logger

from gigaevo.exceptions import ValidationError
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer, Box

from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.evolution.mutation.context import (
    MemoryMutationContext,
    MutationContext,
    PreformattedMutationContext,
)

from gigaevo.programs.core_types import ProgramStageResult, StageError, StageIO, StageState, VoidInput
from gigaevo.programs.stages import Stage
from gigaevo.programs.stages.insights import InsightsStage as BaseInsightsStage
from gigaevo.programs.stages.insights_lineage import LineageStage as BaseLineageStage
from gigaevo.programs.stages.python_executors import PythonCodeExecutor, ValidatorInput, CallValidatorFunction, ValidatorOutput
from gigaevo.programs.stages.python_executors.execution import CallProgramFunction
from gigaevo.programs.stages.python_executors.wrapper import ExecRunnerError, run_exec_runner
from gigaevo.programs.utils import dedent_code
from gigaevo.programs.stages.mutation_context import (
    FloatDictContainer, StringContainer,
    MetricsMutationContext, InsightsMutationContext,
    TransitionAnalysis, TransitionAnalysisList,
    EvolutionaryStatistics, InsightsOutput,
    MetricsContext, FamilyTreeMutationContext,
    MUTATION_CONTEXT_METADATA_KEY, CompositeMutationContext,
    EvolutionaryStatisticsMutationContext,
    MutationContextStage as BaseMutationContextStage,
)
from custom.metrics_formatter import BroaderMetricsFormatter


class DictInput(StageIO):
    data: Box[dict[str, float | str]]


class StrDictInput(StageIO):
    data: Box[dict[str, str]]


def _normalize_aux_info_stages(stages: Sequence[str] | str | None) -> set[str]:
    if stages is None:
        return {"insights", "lineage", "mutation"}
    if isinstance(stages, str):
        return {part.strip().lower() for part in stages.split(",") if part.strip()}
    return {str(stage).strip().lower() for stage in stages if str(stage).strip()}


def _aux_info_enabled(stages: Sequence[str] | str | None, name: str) -> bool:
    return name in _normalize_aux_info_stages(stages)


_MISSING_AUX_INFO = object()


async def _without_aux_info(programs: Sequence[Program], action):
    saved: list[tuple[Program, object]] = []
    for program in programs:
        saved.append((program, program.metadata.get("aux_info", _MISSING_AUX_INFO)))
        program.metadata.pop("aux_info", None)

    try:
        return await action()
    finally:
        for program, value in saved:
            if value is _MISSING_AUX_INFO:
                program.metadata.pop("aux_info", None)
            else:
                program.metadata["aux_info"] = value


@StageRegistry.register(description="Load live vartodd path-store context for prompts")
class LivePathStoreContextStage(Stage):
    InputsModel = VoidInput
    OutputModel = StringContainer
    cacheable: bool = False

    def __init__(
        self,
        *,
        root_dir: str = "data/path_backups",
        problem_dir: str | Path | None = None,
        top_k: int = 11,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.problem_dir = Path(problem_dir) if problem_dir is not None else None
        self.top_k = top_k

    async def compute(self, program: Program) -> StageIO:
        summary = "path store unavailable"
        try:
            if self.problem_dir is not None:
                problem_dir_str = str(self.problem_dir.resolve())
                if problem_dir_str not in sys.path:
                    sys.path.insert(0, problem_dir_str)
                module_name = f"_gigaevo_stage_path_store_{abs(hash(problem_dir_str))}"
                spec = importlib.util.spec_from_file_location(
                    module_name, self.problem_dir / "path_store.py"
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load path_store.py from {self.problem_dir}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                PathStore = module.PathStore
            else:
                from path_store import PathStore

            summary = PathStore(root_dir=self.root_dir).summarize(top_k=self.top_k)
        except Exception as exc:
            summary = f"path store unavailable: {exc}"

        content = f"## Live Path Store\n\n{summary}"
        return StringContainer(data=content)

@StageRegistry.register(description="LLM insights for a single program")
class ComputeTimeStage(Stage):
    InputsModel = VoidInput
    OutputModel = Box[dict[str, float]]
    cacheable: bool = True

    async def compute(self, program: Program) -> StageIO:

        time = {"runtime": program.stage_results['CallProgramFunction'].duration_seconds()}

        logger.debug(
            "[TimeStage] Time of {}: {:.2} s",
            program.id,
            time["runtime"],
        )

        return Box[dict[str, float]](data=time)


@StageRegistry.register(
    description="Call a validator function from a Python file on program output (+ optional context)."
)
class BroaderCallValidatorFunction(PythonCodeExecutor):
    """Loads validator file and calls function `validate(context?, program_output)`."""
    InputsModel = ValidatorInput
    OutputModel = Box[dict[str, float | str]]

    def __init__(self, *, path: Path, function_name: str = "validate", **kwargs: Any):
        super().__init__(
            function_name=function_name, python_path=[Path(path).parent], **kwargs
        )
        p = Path(path)
        if not p.exists():
            raise ValidationError(f"Validator file not found: {p}")
        try:
            self._validator_code = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ValidationError(f"Failed to read validator file: {e}") from e

    def _code_str(self, program: Program) -> str:
        return self._validator_code

    # def parse_output(self, x: Any) -> Tuple[dict[str, float], Any]:
    #     return x if isinstance(x, tuple) else (x, None)

    def _build_call(self, program: Program) -> tuple[Sequence[Any], dict[str, Any]]:
        params = cast(ValidatorInput, self.params)
        payload = params.payload.data
        if params.context is not None:
            context = params.context.data
        else:
            context = None
        return ([context, payload] if context is not None else [payload]), {}


@StageRegistry.register(
    description="CallProgramFunction with a persistent on-disk execution cache"
)
class CachedCallProgramFunction(CallProgramFunction):
    """Persistent cross-run execution cache for expensive program evaluations.

    Cache key: sha256 of function_name + program code + contents of every
    file in `cache_key_files` (e.g. helper.py — editing a seed or a key file
    invalidates its entry). On a hit the stored ProgramStageResult is
    returned verbatim, including the original started_at/finished_at, so
    ComputeTimeStage reports the true historical runtime instead of the
    near-zero cache lookup. On a miss the program runs normally and a
    COMPLETED result is written back to the cache. Failed results are never
    cached. Downstream stages (validator, metrics/aux distillation,
    insights, mutation context) always recompute fresh from the payload.

    Only programs whose metadata `source` is in `cache_metadata_sources`
    participate (default: initial programs). Mutant code never repeats, so
    caching it would only bloat the directory; pass an empty list/None to
    cache every program. `cache_dir=None` disables caching entirely
    (identical behavior to CallProgramFunction).

    Delete the cache directory (or a single entry) to force re-evaluation.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        cache_key_files: Sequence[str | Path] | None = None,
        cache_metadata_sources: Sequence[str] | None = ("initial_program",),
        soft_timeout_grace_s: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_key_files = [Path(p) for p in (cache_key_files or [])]
        self.cache_metadata_sources = set(cache_metadata_sources or ())
        self.soft_timeout_grace_s = (
            max(0.0, float(soft_timeout_grace_s))
            if soft_timeout_grace_s is not None
            else None
        )

    def _soft_deadline_epoch(self) -> float | None:
        if self.soft_timeout_grace_s is None:
            return None
        return time.time() + max(0.0, float(self.timeout) - self.soft_timeout_grace_s)

    async def compute(self, program: Program) -> ProgramStageResult | Box[Any]:
        stage_name = self.__class__.__name__
        code_str = self._code_str(program)
        args, kwargs = self._build_call(program)
        soft_deadline_epoch = self._soft_deadline_epoch()

        logger.debug(
            "[{}] calling '{}' with {} arg(s), {} kwarg(s), soft_deadline={}",
            stage_name,
            self.function_name,
            len(args),
            len(kwargs),
            f"{soft_deadline_epoch:.3f}" if soft_deadline_epoch is not None else "off",
        )

        env_updates: dict[str, Any] = {
            "GIGAEVO_PROGRAM_ID": program.id,
            "GIGAEVO_PROGRAM_ID_SHORT": program.id[:8],
        }
        if soft_deadline_epoch is not None:
            env_updates.update(
                {
                    "GIGAEVO_EXEC_SOFT_DEADLINE_EPOCH": f"{soft_deadline_epoch:.6f}",
                    "GIGAEVO_EXEC_TIMEOUT_S": str(int(self.timeout)),
                    "GIGAEVO_EXEC_SOFT_TIMEOUT_GRACE_S": str(
                        int(self.soft_timeout_grace_s or 0)
                    ),
                }
            )
        else:
            env_updates["GIGAEVO_EXEC_SOFT_DEADLINE_EPOCH"] = None

        try:
            value, stdout_bytes, stderr_text = await run_exec_runner(
                code=dedent_code(code_str),
                function_name=self.function_name,
                args=args,
                kwargs=kwargs,
                python_path=self.python_path,
                env_updates=env_updates,
                timeout=int(self.timeout),
                max_memory_mb=self.max_memory_mb,
                max_output_size=self.max_output_size,
            )

            value_parsed = self.parse_output(value)

            logger.debug(
                "[{}] {} ok | result_type={}",
                stage_name,
                program.id[:8],
                type(value_parsed).__name__,
            )

            del stdout_bytes
            del stderr_text

            return self.__class__.OutputModel(data=value_parsed)

        except ExecRunnerError as e:
            error_type = "SubprocessError"
            error_msg = str(e)

            if e.stderr and (
                "MemoryError" in e.stderr or "Cannot allocate memory" in e.stderr
            ):
                error_type = "MemoryLimitExceeded"
                error_msg = (
                    f"Process exceeded memory limit of {self.max_memory_mb} MB"
                    if self.max_memory_mb
                    else "Process ran out of memory"
                )

            logger.warning(
                "[{}] {} FAILED for {}: {}",
                stage_name,
                error_type,
                program.id[:8],
                error_msg[:200],
            )
            return ProgramStageResult.failure(
                error=StageError(
                    type=error_type,
                    message=error_msg,
                    stage=stage_name,
                    traceback=e.stderr,
                )
            )
        except Exception as e:
            logger.warning(
                "[{}] Exception for {}: {}",
                stage_name,
                program.id[:8],
                str(e)[:200],
            )
            return ProgramStageResult.failure(
                error=StageError.from_exception(e, stage=stage_name)
            )

    def _cache_path(self, program: Program) -> Path | None:
        if self.cache_dir is None:
            return None
        if (
            self.cache_metadata_sources
            and program.metadata.get("source") not in self.cache_metadata_sources
        ):
            return None
        h = hashlib.sha256()
        h.update(self.function_name.encode())
        h.update(program.code.encode())
        for key_file in self.cache_key_files:
            try:
                h.update(key_file.read_bytes())
            except OSError:
                logger.warning(
                    "[{}] cache key file {} unreadable; keyed by its absence",
                    self.stage_name,
                    key_file,
                )
                h.update(f"<missing:{key_file}>".encode())
        return self.cache_dir / f"{h.hexdigest()[:32]}.json"

    async def execute(self, program: Program) -> ProgramStageResult:
        cache_path = self._cache_path(program)
        if cache_path is not None and cache_path.exists():
            try:
                entry = json.loads(cache_path.read_text())
                result = ProgramStageResult.from_dict(entry["result"])
                if result.status == StageState.COMPLETED:
                    logger.info(
                        "[{}] execution cache HIT for {} ({}, original runtime {:.0f}s)",
                        self.stage_name,
                        program.metadata.get("strategy_name", program.short_id),
                        cache_path.name,
                        result.duration_seconds() or 0.0,
                    )
                    # Stamp the current inputs hash so the in-Redis
                    # InputHashCache skip keeps working on DAG re-runs.
                    self._current_inputs_hash = self.compute_inputs_hash()
                    return self.get_cache_handler().on_complete(
                        result, self._current_inputs_hash
                    )
            except Exception as exc:
                logger.warning(
                    "[{}] failed to load cache entry {}: {} — re-executing",
                    self.stage_name,
                    cache_path.name,
                    exc,
                )

        result = await super().execute(program)

        if cache_path is not None and result.status == StageState.COMPLETED:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                entry = {
                    "strategy_name": program.metadata.get("strategy_name"),
                    "program_id": program.id,
                    "result": result.model_dump(mode="json"),
                }
                tmp_path = cache_path.with_name(
                    f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
                )
                tmp_path.write_text(json.dumps(entry))
                os.replace(tmp_path, cache_path)
                logger.info(
                    "[{}] execution cache WRITE for {} -> {}",
                    self.stage_name,
                    program.metadata.get("strategy_name", program.short_id),
                    cache_path.name,
                )
            except Exception as exc:
                logger.warning(
                    "[{}] failed to write cache entry {}: {}",
                    self.stage_name,
                    cache_path.name,
                    exc,
                )
        return result


@StageRegistry.register(
    description="Distill numeric metrics from CallValidator"
)
class DistillMetrics(Stage):
    InputsModel = DictInput
    OutputModel = Box[dict[str, float]]
    cacheable: bool = True

    async def compute(self, program: Program) -> StageIO:
        metrics = {}
        for key, val in self.params.data.data.items():
            if isinstance(val, float):
                metrics[key] = val
        logger.debug(f"[DistillMetrics] stage completed with {len(metrics)} \n{self.params.data=}")
        return Box[dict[str, float]](data=metrics)


@StageRegistry.register(
    description="Distill aux info from CallValidator"
)
class DistillNonMetrics(Stage):
    InputsModel = DictInput
    OutputModel = Box[dict[str, str]]
    cacheable: bool = True

    async def compute(self, program: Program) -> StageIO:
        non_metrics = {}
        params: DictInput = self.params
        for key, val in params.data.data.items():
            if isinstance(val, str):
                non_metrics[key] = val
        logger.info(f"[DistillNonMetrics] stage completed with {len(non_metrics)}\n{self.params.data=}")
        return Box[dict[str, str]](data=non_metrics)


@StageRegistry.register(
    description="Ensure non metrics and set formatter"
)
class EnsureNonMetricsStage(Stage):
    InputsModel = StrDictInput
    OutputModel = Box[dict[str, str]]

    def __init__(
        self,
        *,
        aux_info_stages: Sequence[str] | str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.aux_info_stages = _normalize_aux_info_stages(aux_info_stages)

    async def compute(self, program: Program) -> StageIO:
        aux_info = self.params.data.data.get("aux info")
        if aux_info and self.aux_info_stages.intersection({"insights", "lineage"}):
            program.set_metadata("aux_info", aux_info)
        elif not self.aux_info_stages.intersection({"insights", "lineage"}):
            program.metadata.pop("aux_info", None)
        return self.params.data


@StageRegistry.register(description="Insights stage with custom aux-info routing")
class AuxControlledInsightsStage(BaseInsightsStage):
    def __init__(
        self,
        *,
        aux_info_stages: Sequence[str] | str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.aux_info_stages = _normalize_aux_info_stages(aux_info_stages)

    async def compute(self, program: Program) -> StageIO | ProgramStageResult:
        if "insights" in self.aux_info_stages:
            return await super().compute(program)

        async def action():
            return await super(AuxControlledInsightsStage, self).compute(program)

        return await _without_aux_info([program], action)


@StageRegistry.register(description="Lineage stage with custom aux-info routing")
class AuxControlledLineageStage(BaseLineageStage):
    def __init__(
        self,
        *,
        aux_info_stages: Sequence[str] | str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.aux_info_stages = _normalize_aux_info_stages(aux_info_stages)

    async def compute(self, program: Program) -> StageIO | ProgramStageResult:
        if "lineage" in self.aux_info_stages:
            return await super().compute(program)

        prep = await self.preprocess(program, self.params)
        if isinstance(prep, ProgramStageResult):
            return prep

        kwargs = dict(prep)
        if self.program_kwarg is not None:
            if self.program_kwarg in kwargs:
                raise ValueError(
                    f"{self.stage_name}: program_kwarg '{self.program_kwarg}' "
                    "collides with a preprocessed arg."
                )
            kwargs[self.program_kwarg] = program

        parents = list(kwargs.get("parents") or [])

        async def action():
            result = await self._agent_call(kwargs)
            return await self.postprocess(program, result)

        return await _without_aux_info([program, *parents], action)


class MutationContextInputs(StageIO):
    """
    Optional upstream signals the stage can consume.
      - metrics: validated floats, e.g. from EnsureMetricsStage (FloatDictContainer)
      - insights: ProgramInsights wrapped by the Insights stage output
      - lineage_ancestors: TransitionAnalysisList (from collector+lineage stages on ancestors)
      - lineage_descendants: TransitionAnalysisList (from collector+lineage stages on descendants)
      - evolutionary_statistics: EvolutionaryStatistics (from EvolutionaryStatisticsCollector)
      - formatted: preformatted string block for mutation prompt
      - memory: memory-selected idea cards
    """

    metrics: Optional[FloatDictContainer]
    non_metrics: Optional[Box[dict[str, str]]]
    insights: Optional[InsightsOutput]
    lineage_ancestors: Optional[TransitionAnalysisList]
    lineage_descendants: Optional[TransitionAnalysisList]
    evolutionary_statistics: Optional[EvolutionaryStatistics]
    formatted: Optional[StringContainer]
    memory: Optional[StringContainer]


class NonMetricsMutationContext(MutationContext):
    """Context with program metrics."""

    non_metrics: dict[str, str]

    class Config:
        arbitrary_types_allowed = True

    def format(self) -> str:
        lines = ["## Program execution aux info", ""]
        for _, val in self.non_metrics.items():
            lines.append(val)
        logger.debug(f"[NonMetricsMutationContext] {lines=}")
        return "\n".join(lines)


@StageRegistry.register(description="Mutation context stage with custom aux-info routing")
class AuxControlledMutationContextStage(BaseMutationContextStage):
    InputsModel = MutationContextInputs

    def __init__(
        self,
        *,
        aux_info_stages: Sequence[str] | str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.aux_info_stages = _normalize_aux_info_stages(aux_info_stages)

    async def compute(self, program: Program) -> StageIO:
        contexts: list[MutationContext] = []
        params: MutationContextInputs = self.params

        if params.metrics is not None:
            metrics_map = params.metrics.data
            formatter = BroaderMetricsFormatter(self.metrics_context)
            contexts.append(
                MetricsMutationContext(metrics=metrics_map, metrics_formatter=formatter)
            )

        if params.non_metrics is not None and "mutation" in self.aux_info_stages:
            context = NonMetricsMutationContext(non_metrics=params.non_metrics.data)
            aux_info = context.format()
            program.set_metadata("aux_info", aux_info)
            contexts.append(context)
        elif "mutation" not in self.aux_info_stages:
            program.metadata.pop("aux_info", None)

        if params.insights is not None:
            insights = params.insights.insights
            contexts.append(InsightsMutationContext(insights=insights))

        ancestor_lineages: list[TransitionAnalysis] = []
        if params.lineage_ancestors is not None:
            ancestor_lineages = params.lineage_ancestors.items

        descendant_lineages: list[TransitionAnalysis] = []
        if params.lineage_descendants is not None:
            descendant_lineages = params.lineage_descendants.items

        if ancestor_lineages or descendant_lineages:
            formatter = BroaderMetricsFormatter(self.metrics_context)
            contexts.append(
                FamilyTreeMutationContext(
                    ancestors=ancestor_lineages,
                    descendants=descendant_lineages,
                    metrics_formatter=formatter,
                )
            )

        if params.evolutionary_statistics is not None:
            contexts.append(
                EvolutionaryStatisticsMutationContext(
                    evolutionary_statistics=params.evolutionary_statistics,
                    metrics_context=self.metrics_context,
                )
            )
            logger.debug("[{}] Evolutionary statistic data", contexts[-1].format())

        if params.memory is not None and params.memory.data.strip():
            contexts.append(MemoryMutationContext(memory_block=params.memory.data))

        if params.formatted is not None:
            contexts.append(PreformattedMutationContext(content=params.formatted.data))

        if not contexts:
            logger.info(
                "[{}] No upstream context available for {}",
                type(self).__name__,
                program.id[:8],
            )

        context = CompositeMutationContext(contexts=contexts).format()
        program.set_metadata(self.metadata_key, context)
        return StringContainer(data=context)


@StageRegistry.register(
    description="Assemble mutation context from metrics/insights/lineage"
)
class MutationContextStage(Stage):
    """
    Builds a CompositeMutationContext from whatever inputs are available.

    Notes:
      - Non-cacheable: lineage/descendant data evolves over time.
      - Writes context into Program.metadata[MUTATION_CONTEXT_METADATA_KEY].
      - Returns the context wrapped in AnyContainer so downstream stages can consume it.
    """

    InputsModel = MutationContextInputs
    OutputModel = StringContainer
    cacheable: bool = False

    def __init__(self, *, metrics_context: MetricsContext, **kwargs):
        super().__init__(**kwargs)
        self.metrics_context = metrics_context
        self.metadata_key = MUTATION_CONTEXT_METADATA_KEY

    async def compute(self, program: Program) -> StageIO:
        contexts: list = []
        params: MutationContextInputs = self.params

        if params.metrics is not None:
            metrics_map = params.metrics.data
            formatter = BroaderMetricsFormatter(self.metrics_context)
            contexts.append(
                MetricsMutationContext(metrics=metrics_map, metrics_formatter=formatter)
            )
        if params.non_metrics is not None:
            context = NonMetricsMutationContext(non_metrics=params.non_metrics.data)
            aux_info = context.format()
            program.set_metadata("aux_info", aux_info)
            contexts.append(context)

        if params.insights is not None:
            insights = params.insights.insights
            contexts.append(InsightsMutationContext(insights=insights))

        ancestor_lineages: list[TransitionAnalysis] = []
        if params.lineage_ancestors is not None:
            ancestor_lineages = params.lineage_ancestors.items

        descendant_lineages: list[TransitionAnalysis] = []
        if params.lineage_descendants is not None:
            descendant_lineages = params.lineage_descendants.items

        if ancestor_lineages or descendant_lineages:
            formatter = BroaderMetricsFormatter(self.metrics_context)
            contexts.append(
                FamilyTreeMutationContext(
                    ancestors=ancestor_lineages,
                    descendants=descendant_lineages,
                    metrics_formatter=formatter,
                )
            )
        if params.evolutionary_statistics is not None:
            contexts.append(
                EvolutionaryStatisticsMutationContext(
                    evolutionary_statistics=params.evolutionary_statistics,
                    metrics_context=self.metrics_context,
                )
            )
            logger.info(
                "[{}] Evolutionary statistic data",
                contexts[-1].format()
            )
        if params.memory is not None and params.memory.data.strip():
            contexts.append(MemoryMutationContext(memory_block=params.memory.data))

        if params.formatted is not None:
            contexts.append(PreformattedMutationContext(content=params.formatted.data))

        if not contexts:
            logger.info(
                "[{}] No upstream context available for {}",
                type(self).__name__,
                program.id[:8],
            )

        context = CompositeMutationContext(contexts=contexts).format()
        program.set_metadata(self.metadata_key, context)
        return StringContainer(data=context)


_AUX_SECTION_HEADERS = (
    "path_summary",
    "path_policy_groups",
    "converged_policy_profiles",
    "scores",
    "search_stat",
)
_POLICY_GROUP_CONTEXT_LINES = 8


def _extract_aux_section(aux_info: str, section_name: str) -> str:
    """Return one top-level vartodd aux section, including its header."""
    headers = "|".join(
        re.escape(header)
        for header in _AUX_SECTION_HEADERS
        if header != section_name
    )
    match = re.search(
        rf"(?ms)^{re.escape(section_name)}:\n.*?(?=^(?:{headers}):\n|\Z)",
        aux_info,
    )
    return match.group(0).strip() if match else ""


def _profile_ids_from_text(text: str) -> list[str]:
    """Collect profile ids, expanding compact ranges like P2-P4."""
    ordered: list[str] = []
    seen: set[str] = set()

    def add(profile_id: str) -> None:
        if profile_id not in seen:
            seen.add(profile_id)
            ordered.append(profile_id)

    for start, end in re.findall(r"\bP(\d+)-P(\d+)\b", text):
        lo, hi = int(start), int(end)
        step = 1 if hi >= lo else -1
        for idx in range(lo, hi + step, step):
            add(f"P{idx}")
    for profile_id in re.findall(r"\bP\d+\b", text):
        add(profile_id)
    return ordered


def _late_profile_ids(path_policy_groups_section: str) -> list[str]:
    group_lines = [
        line
        for line in path_policy_groups_section.splitlines()
        if re.match(r"\s*(?:group=\d+\s+|g\d+\s+)", line)
    ]
    if not group_lines:
        return _profile_ids_from_text(path_policy_groups_section)

    ids: list[str] = []
    seen: set[str] = set()
    for line in group_lines[-_POLICY_GROUP_CONTEXT_LINES:]:
        for profile_id in _profile_ids_from_text(line):
            if profile_id not in seen:
                seen.add(profile_id)
                ids.append(profile_id)
    return ids


def _profile_line_for_id(aux_info: str, profile_id: str) -> str:
    profiles = _extract_aux_section(aux_info, "converged_policy_profiles")
    for line in profiles.splitlines():
        if re.match(rf"\s*{re.escape(profile_id)}\b", line):
            return line.strip()
    return ""


def _format_late_profiles(aux_info: str, profile_ids: Sequence[str]) -> str:
    profiles = _extract_aux_section(aux_info, "converged_policy_profiles")
    if not profiles:
        return ""
    lines = profiles.splitlines()
    keep = set(profile_ids)
    if not keep:
        selected = [
            line
            for line in lines[1:]
            if re.match(r"\s*P\d+\b", line)
        ][-_POLICY_GROUP_CONTEXT_LINES:]
    else:
        selected = [
            line
            for line in lines[1:]
            if (m := re.match(r"\s*(P\d+)\b", line)) and m.group(1) in keep
        ]
    return "\n".join([lines[0], *selected]).strip() if selected else ""


def _format_relevant_scores(aux_info: str, profile_ids: Sequence[str]) -> str:
    scores = _extract_aux_section(aux_info, "scores")
    if not scores or not profile_ids:
        return ""
    lines = scores.splitlines()
    selected = [
        line
        for line in lines[1:]
        if any(re.search(rf"\b{re.escape(profile_id)}\b", line) for profile_id in profile_ids)
    ]
    return "\n".join([lines[0], *selected]).strip() if selected else ""


def _bounded_middle(text: str, *, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = (
        f"\n... [focused aux excerpt truncated to {max_chars} chars; "
        "preserved tail evidence]\n"
    )
    if max_chars <= len(marker) + 40:
        return text[-max_chars:]
    head_chars = max(0, max_chars // 3)
    tail_chars = max_chars - head_chars - len(marker)
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def _parse_execution_digest(aux_info: str) -> str:
    """Extract key saturation signals from vartodd aux_info text.

    Returns a compact '## Execution Signal Digest' block that survives
    MutationAgent._strip_aux_sections (which only strips '## Program execution aux info').
    Returns empty string when aux_info is absent or unparseable.
    """
    if not aux_info:
        return ""

    # --- path_summary ---
    final_rank_m = re.search(r"final_rank=(\d+)", aux_info)
    total_red_m = re.search(r"total_reduction=(\d+)", aux_info)

    # --- search_stat ---
    loaded_rank_m = re.search(r"\bloaded_rank:\s*(\d+)", aux_info)
    loaded_path_rank_m = re.search(r"\bloaded_path_rank:\s*(\d+)", aux_info)
    loaded_path_name_m = re.search(r"\bloaded_path_name:\s*([^\n]+)", aux_info)
    init_rank_thr_m = re.search(r"\binit_rank_thr:\s*(\d+)", aux_info)
    loaded_imp_m = re.search(r"loaded_path_improved:\s*(\d+)", aux_info)
    loaded_no_imp_m = re.search(r"NOTE:\s*no improvement over loaded path", aux_info)
    best_seen_m = re.search(r"best_seen_times:\s*(\d+)", aux_info)
    total_evals_m = re.search(r"total_evals:\s*(\d+)", aux_info)
    q01_m = re.search(r"rank 0\.1q=([\d.]+)", aux_info)
    q09_m = re.search(r"rank 0\.9q=([\d.]+)", aux_info)

    loaded_path_name = None
    loaded_path_limit = None
    if loaded_path_name_m:
        loaded_path_line = loaded_path_name_m.group(1).strip()
        loaded_path_name = loaded_path_line.split()[0] if loaded_path_line else None
        if limit_m := re.search(r"\blimit_bucket=([^\s]+)", loaded_path_line):
            loaded_path_limit = limit_m.group(1)

    # --- last path_policy_groups line (final / hardest rank region) ---
    # Match both the legacy `group=N ranks=...` form (still emitted for loaded
    # paths saved by older code) and the new compact `gN <start>-><end> ...` band
    # form. The compact form has no `ranks=`/`z=mean:`/`note=` labels.
    group_lines = re.findall(
        r"(?m)^\s*(?:group=\d+\s+(?:G\d+\s+)?ranks=\S+|g\d+\s+\d+->\d+)\s+.*$",
        aux_info,
    )

    line1: list[str] = []
    for m, name in [
        (final_rank_m, "final_rank"),
        (total_red_m, "total_reduction"),
        (loaded_rank_m, "loaded_rank"),
        (loaded_path_rank_m, "loaded_path_rank"),
    ]:
        if m:
            line1.append(f"{name}={m.group(1)}")
    if loaded_imp_m:
        line1.append(f"loaded_path_improved={loaded_imp_m.group(1)}")
    elif loaded_no_imp_m:
        line1.append("loaded_path_improved=0")

    line1b: list[str] = []
    if loaded_path_name:
        line1b.append(f"loaded_path_name={loaded_path_name}")
    if loaded_path_limit:
        line1b.append(f"loaded_path_limit_bucket={loaded_path_limit}")
    if init_rank_thr_m:
        line1b.append(f"init_rank_thr={init_rank_thr_m.group(1)}")

    line2: list[str] = []
    for m, name in [
        (best_seen_m, "best_seen_times"),
        (total_evals_m, "total_evals"),
        (q01_m, "rank_0.1q"),
        (q09_m, "rank_0.9q"),
    ]:
        if m:
            line2.append(f"{name}={m.group(1)}")

    if not (line1 or line2):
        return ""

    sections = ["## Execution Signal Digest"]
    if line1:
        sections.append("  ".join(line1))
    if line1b:
        sections.append("load: " + "  ".join(line1b))
    if line2:
        sections.append("optimizer: " + "  ".join(line2))

    if group_lines:
        last = group_lines[-1]
        parts: list[str] = []
        if m := re.search(r"ranks=(\S+)", last):
            parts.append(m.group(1))
        elif m := re.search(r"^\s*g\d+\s+(\d+->\d+)", last):  # compact band form
            parts.append(m.group(1))
        # z: compact `z:<val>` or `z:mean(mxN)`; legacy `z=mean:M/max:X`
        if m := re.search(r"\bz:([^\s]+)", last):
            parts.append(f"z={m.group(1)}")
        elif m := re.search(r"\bz=([^\s]+)", last):
            z_value = m.group(1)
            if z_value.startswith("mean:"):
                if old_m := re.search(r"z=mean:([\d.]+)/max:(\d+)", last):
                    parts.append(f"z_mean={old_m.group(1)} z_max={old_m.group(2)}")
            elif z_value.startswith("researched:"):
                parts.append(f"z={z_value.removeprefix('researched:')}")
            else:
                parts.append(f"z={z_value}")
        # accepted-per-z: compact `apz:<val>`; legacy `az=`/`accepted_per_z=mean:`
        if m := re.search(r"\bapz:([^\s]+)", last):
            parts.append(f"apz={m.group(1)}")
        elif m := re.search(r"\baz=([^\s]+)", last):
            parts.append(f"az={m.group(1)}")
        elif m := re.search(r"z_acceptance=accepted_per_z:([^\s]+)", last):
            parts.append(f"az={m.group(1)}")
        elif m := re.search(r"accepted_per_z=mean:([\d.]+)", last):
            parts.append(f"accepted_per_z={m.group(1)}")

        if m := re.search(r"quality=[^ ]*dim:([^;]+)/avail:([^;\s]+)", last):
            parts.append(f"dim={m.group(1)}/{m.group(2)}")
        elif m := re.search(r"(?:^|\s)dim:([^\s]+)", last):  # compact
            parts.append(f"dim={m.group(1)}")
        elif m := re.search(r"\bdim=([^\s]+)", last):
            parts.append(f"dim={m.group(1)}")
        elif m := re.search(r"dim_mean:([\d.]+)/dim_max:(\d+)", last):
            parts.append(f"dim={m.group(1)}/{m.group(2)}")

        if m := re.search(r"\bsrc=([^\s]+)", last):  # compact H:../T:.. and legacy
            parts.append(f"src={m.group(1)}")
        elif m := re.search(r"source=accepted:tohpe:([^/\s]+)/todd:([^\s]+)", last):
            parts.append(f"src=H:{m.group(1)},T:{m.group(2)}")
        elif m := re.search(r"source=(tohpe:[^/\s]+/todd:[^\s]+)", last):
            parts.append(f"src={m.group(1)}")

        if m := re.search(r"\bpool:([^\s]+)", last):
            parts.append(f"pool={m.group(1)}")
        elif m := re.search(r"pool=total:([^/\s]+)", last):
            parts.append(f"pool_total={m.group(1)}")

        if m := re.search(r"\bfill=([^\s]+)", last):
            parts.append(f"fill={m.group(1)}")
        elif m := re.search(r"pool_fill=([^\s]+)", last):
            fill = m.group(1)
            fill = {"underfilled": "under", "near_capacity": "near"}.get(fill, fill)
            parts.append(f"fill={fill}")

        last_flags = ""
        if m := re.search(r"\[([^\]]+)\]", last):  # compact `[flag,flag]`
            last_flags = m.group(1)
            parts.append(f"flags={last_flags}")
        elif m := re.search(r"\bflags=([^\s]+)", last):
            last_flags = m.group(1)
            parts.append(f"flags={last_flags}")
        elif m := re.search(r"note=(\S+)", last):
            last_flags = m.group(1)
            parts.append(f"flags={last_flags}")
        if parts:
            sections.append("last_group: " + "  ".join(parts))

        profile_ids = _profile_ids_from_text(last)
        if profile_ids:
            profile_id = profile_ids[-1]
            profile_line = _profile_line_for_id(aux_info, profile_id)
            if profile_line:
                profile_parts = [profile_id]
                if m := re.search(r"pool=final:([^/\s]+)", profile_line):
                    profile_parts.append(f"pool_final={m.group(1)}")
                if m := re.search(
                    r"z_buckets=min_buckets:([^/\s]+)/max_buckets:([^/\s]+)/limit_bucket:([^\s]+)",
                    profile_line,
                ):
                    profile_parts.append(
                        "z_policy="
                        f"min:{m.group(1)}/max:{m.group(2)}/limit:{m.group(3)}"
                    )
                if m := re.search(r"actions_per_bucket:([^/\s]+)", profile_line):
                    profile_parts.append(f"actions_per_bucket={m.group(1)}")
                if len(profile_parts) > 1:
                    sections.append("last_profile: " + "  ".join(profile_parts))

        # Where reduction actually came from, per rank band, so the model can
        # locate the bottleneck source instead of guessing from one group.
        red_split = []
        for line in group_lines:
            # legacy `ranks=X->Y ... reduction=N` or compact `gN X->Y red=N`
            rr = re.search(r"ranks=(\S+)", line) or re.search(r"^\s*g\d+\s+(\d+->\d+)", line)
            red = re.search(r"\b(?:reduction|red)=(-?\d+)", line)
            if rr and red:
                red_split.append(f"{rr.group(1)}:{red.group(1)}")
        if len(red_split) > 1:
            sections.append("reduction_by_group: " + "  ".join(red_split))

        # Turn the last-group flag into the lever the system prompt recommends,
        # so the actionable reading is not buried in prose.
        frontier_hint = _frontier_lever(last, last_flags)
        if frontier_hint:
            sections.append("frontier: " + frontier_hint)

    return "\n".join(sections)


def _frontier_lever(last_group_line: str, flags: str) -> str:
    """Map the hardest group's signal to the lever to move next.

    Reads only fields already present in the last group line; returns "" when
    nothing conclusive. This does not decide the mutation; it explains what
    the observed generation bottleneck implies for the next experiment.
    """
    # Normalize compact flag abbreviations back to canonical names so this works
    # on both the compact band form and the legacy note= form.
    abbr_to_full = {
        "H_only": "tohpe_only",
        "T_dom": "todd_dominant",
        "hi_acc": "high_acceptance_low_z",
        "hard_hiZ": "hard_refinement_high_z",
        "lo_acc": "low_acceptance_per_z",
        "hi_dim": "high_dim_early_region",
    }
    flag_set = {abbr_to_full.get(f, f) for f in flags.split(",") if f}
    band_m = re.search(r"ranks=(\d+)->(\d+)", last_group_line) or re.search(
        r"^\s*g\d+\s+(\d+)->(\d+)", last_group_line
    )
    band = f"{band_m.group(1)}->{band_m.group(2)}" if band_m else "late frontier"
    if "hard_refinement_high_z" in flag_set or "low_acceptance_per_z" in flag_set:
        return (
            f"{band} is z-saturated (many buckets, few accepts): positive "
            "actions are rare, so broader z-bucket research is needed for "
            "more action discovery and candidate diversity"
        )
    if "tohpe_only" in flag_set and "todd_dominant" not in flag_set:
        return f"{band} ran TOHPE-only; enabling/raising TODD may open new actions"
    if "todd_dominant" in flag_set:
        return f"{band} is TODD-driven; keep TODD recall, tune y/reserve before cutting z"
    if "high_dim_early_region" in flag_set:
        return f"{band} still high-dim/early; frontier evidence is later groups"
    return ""


def _format_execution_aux_excerpt(aux_info: str, *, max_chars: int = 18000) -> str:
    """Return a focused raw aux excerpt for mutation prompts.

    The old prefix excerpt often hid late path groups and search_stat on long
    reports. Keep the rendered evidence sections that explain the frontier and
    path-loading result, then middle-truncate only if the focused pack is still
    too large.
    """
    aux_info = aux_info.strip()
    if not aux_info:
        return ""

    path_summary = _extract_aux_section(aux_info, "path_summary")
    path_groups = _extract_aux_section(aux_info, "path_policy_groups")
    late_profiles = _late_profile_ids(path_groups)
    sections = [
        section
        for section in (
            path_summary,
            path_groups,
            _format_late_profiles(aux_info, late_profiles),
            _format_relevant_scores(aux_info, late_profiles),
            _extract_aux_section(aux_info, "search_stat"),
        )
        if section
    ]
    if not sections:
        sections = [aux_info]
    focused = _bounded_middle("\n\n".join(sections), max_chars=max_chars)
    return "## Program Aux Excerpt\n\n" + focused


@StageRegistry.register(
    description="Extract key execution signals from aux_info for mutation context"
)
class ExecutionDigestStage(Stage):
    """Parses aux_info from program metadata and emits a compact signal digest.

    Uses '## Execution Signal Digest' header which is NOT stripped by
    MutationAgent._strip_aux_sections, giving the mutation LLM direct
    access to saturation signals (best_seen_times, loaded_path_improved,
    z budget, accepted_per_z, source distribution).

    Wired to MutationContextStage.formatted so it survives into the prompt.
    """

    InputsModel = StrDictInput
    OutputModel = StringContainer
    cacheable: bool = True

    def __init__(
        self,
        *,
        aux_info_stages: Sequence[str] | str | None = None,
        enabled: bool | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.enabled = (
            _aux_info_enabled(aux_info_stages, "mutation")
            if enabled is None
            else enabled
        )

    async def compute(self, program: Program) -> StageIO:
        if not self.enabled:
            return StringContainer(data="")

        params: StrDictInput = self.params
        aux_info = (
            params.data.data.get("aux info")
            or params.data.data.get("aux_info")
            or str(program.get_metadata("aux_info") or "")
        )
        parts = [
            part
            for part in (
                _parse_execution_digest(aux_info),
                _format_execution_aux_excerpt(aux_info),
            )
            if part
        ]
        return StringContainer(data="\n\n---\n\n".join(parts))
