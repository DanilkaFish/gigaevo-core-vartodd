import ast
from collections.abc import Mapping
from datetime import UTC, datetime
import importlib.util
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import diffpatch
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_CONTEXT_METADATA_KEY,
    MUTATION_MEMORY_METADATA_KEY,
    ArchetypeName,
)
from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import (
    MultiModelRouter,
    get_last_token_usage,
    get_selected_model,
)
from gigaevo.llm.token_tracking import llm_stage_context
from gigaevo.monitoring.emit import emit as _emit_event
from gigaevo.monitoring.events import LLMCall
from gigaevo.programs.program import Program
from gigaevo.evolution.strategies.base import MutationRoute, ParentRole
from gigaevo.evolution.strategies.route_context import (
    MutationRouteContextProvider,
)

if TYPE_CHECKING:
    from gigaevo.programs.metrics.context import MetricsContext
    from gigaevo.prompts.fetcher import PromptFetcher


class MutationChange(BaseModel):
    """Tracker-friendly description of one introduced change."""

    description: str = Field(
        description=(
            "Generalizable description of the introduced change, optionally followed "
            "by concrete specifics when they matter. Prefer `general pattern + "
            "concrete instance` over a narrow one-off description."
        )
    )
    explanation: str = Field(
        description=(
            "Explain why this change was introduced, why it helped for this "
            "program, and when possible why the same idea could transfer to future "
            "mutations."
        )
    )


class MutationStructuredOutput(BaseModel):
    """Structured output from the mutation LLM.

    Simplified schema to reduce cognitive overhead and let LLM focus on code quality.
    """

    archetype: ArchetypeName = Field(
        description="Selected evolutionary archetype (one of the 8 canonical names in ARCHETYPE_NAMES)."
    )
    justification: str = Field(
        description="2-3 sentences: which insights acted on, strategy used, expected mechanism of improvement"
    )
    insights_used: list[str] = Field(
        default_factory=list,
        description="Flat list of insight strings that were acted on (verbatim from input)",
    )
    changes: list[MutationChange] = Field(
        default_factory=list,
        description=(
            "Key introduced changes. Each item must contain a reusable or "
            "generalizable description plus an explanation of why the change was "
            "introduced."
        ),
    )
    code: str = Field(
        description=(
            "The complete mutated Python source code. "
            "Must be valid Python starting with imports or def statements. "
            "NEVER put JSON, format examples, or templates here. "
            "Use actual newlines between lines, not literal backslash-n."
        )
    )


# Re-export from canonical location for backward compatibility
MUTATION_OUTPUT_METADATA_KEY = MutationSpec.META_OUTPUT


class MutationPromptFields(BaseModel):
    """
    Example template:
        "Mutate {count} parent programs:\n{parent_blocks}"
    """

    count: int = Field(description="Number of parent programs")
    parent_blocks: str = Field(
        description="Formatted parent program blocks with code, metrics, insights"
    )


class MutationState(TypedDict):
    """State for mutation agent."""

    input: list[Program]
    mutation_mode: str
    messages: list[BaseMessage]
    llm_response: Any
    final_code: str
    mutation_label: str
    # Fields set during prompt building (optional initially)
    system_prompt: NotRequired[str]
    user_prompt: NotRequired[str]
    # Prompt tracking ID (None for fixed prompts, sha256[:16] for co-evolved prompts)
    prompt_id: NotRequired[str | None]
    explicit_regime_guidance: NotRequired[str | None]
    explicit_route: NotRequired[MutationRoute | None]
    parent_roles: NotRequired[tuple[ParentRole, ...]]
    selected_mutation_regime: NotRequired[str | None]
    # Fields set during response parsing (optional initially)
    parsed_output: NotRequired[dict[str, Any]]
    structured_output: NotRequired[MutationStructuredOutput]
    metadata: NotRequired[dict[str, Any]]
    error: NotRequired[str]


class MutationAgent(LangGraphAgent):
    """Agent for LLM-based code mutation.

    This agent handles the complete workflow of mutating programs:
    1. Build prompt from parent programs using pre-formatted mutation context
    2. Call LLM to generate structured output (archetype, justification, code)
    3. Extract and parse the structured output (handling diffs if needed)

    Attributes:
        mutation_mode: "rewrite" or "diff"
        system_prompt: System prompt
        user_prompt_template: User prompt template string
        structured_llm: LLM configured for structured output
    """

    StateSchema = MutationState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
        mutation_mode: str = "rewrite",
        # Optional: enable dynamic prompt fetching
        prompt_fetcher: "PromptFetcher | None" = None,
        task_description: str = "",
        metrics_context: "MetricsContext | None" = None,
        live_path_store_root_dir: str | Path | None = None,
        live_path_store_problem_dir: str | Path | None = None,
        live_path_store_top_k: int = 6,
        mutation_regime_guidance: list[Any] | None = None,
        mutation_regime_probability: float = 1.0,
        route_context_provider: MutationRouteContextProvider | None = None,
    ):
        """Initialize mutation agent.

        Args:
            llm: LangChain chat model or router
            mutation_mode: "rewrite" or "diff"
            system_prompt: System prompt string (static or initial value)
            user_prompt_template: User prompt template string
            prompt_fetcher: Optional fetcher for dynamic prompt co-evolution.
                When set and is_dynamic=True, system_prompt is refreshed on
                every build_prompt() call. For FixedDirPromptFetcher, the
                static system_prompt is used without re-fetching.
            task_description: Task description for prompt template formatting
                (required when prompt_fetcher.is_dynamic is True)
            metrics_context: Metrics context for prompt template formatting
                (required when prompt_fetcher.is_dynamic is True)
            live_path_store_root_dir: Optional root directory for Vartodd saved
                paths. When omitted, the problem's ``path_store.DATA_PATH``
                default is used. With a problem dir containing path_store.py,
                a fresh summary is appended to each mutation prompt.
            live_path_store_problem_dir: Optional problem directory containing
                path_store.py.
            live_path_store_top_k: Number of saved paths to show in the prompt.
            mutation_regime_guidance: Optional list of diversity guidance blocks.
                Entries may be strings or mappings with text/guidance/regime and
                probability/weight. When provided, one block is sampled for each
                mutation prompt according to entry weights.
            mutation_regime_probability: Probability of appending one sampled
                regime block when guidance is configured.
        """
        self.mutation_mode = mutation_mode
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.live_path_store_root_dir = (
            Path(live_path_store_root_dir)
            if live_path_store_root_dir is not None
            else None
        )
        self.live_path_store_problem_dir = (
            Path(live_path_store_problem_dir)
            if live_path_store_problem_dir is not None
            else None
        )
        self.live_path_store_top_k = live_path_store_top_k
        self.mutation_regime_guidance = self._parse_mutation_regime_guidance(
            mutation_regime_guidance or []
        )
        self.mutation_regime_probability = min(
            1.0, max(0.0, float(mutation_regime_probability))
        )
        self.route_context_provider = route_context_provider

        # Dynamic prompt fetching support
        self._prompt_fetcher = prompt_fetcher
        self._task_description = task_description
        if metrics_context is not None:
            from gigaevo.programs.metrics.formatter import MetricsFormatter

            self._metrics_formatter: MetricsFormatter | None = MetricsFormatter(
                metrics_context
            )
        else:
            self._metrics_formatter = None

        # Create structured output LLM
        self.structured_llm = llm.with_structured_output(MutationStructuredOutput)

        super().__init__(llm)

    _PROMPT_LOG_DIR = os.environ.get("GIGAEVO_PROMPT_LOG_DIR", "")

    def _dump_prompt_to_file(
        self, prompt_id: str | None, system: str, user: str
    ) -> None:
        """Write full system+user prompts to a log file for offline inspection."""
        log_dir = self._PROMPT_LOG_DIR
        if not log_dir:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            pid = prompt_id or "fixed"
            path = os.path.join(log_dir, f"{ts}_{pid[:12]}.txt")
            with open(path, "w") as f:
                f.write(f"=== PROMPT DUMP {ts} ===\n")
                f.write(f"prompt_id: {prompt_id}\n\n")
                f.write("=== SYSTEM PROMPT ===\n")
                f.write(system)
                f.write("\n\n=== USER PROMPT ===\n")
                f.write(user)
                f.write("\n")
        except Exception as exc:
            logger.debug(f"[MutationAgent] prompt dump failed: {exc}")

    async def arun(
        self,
        input: list[Program],
        mutation_mode: str,
        explicit_regime_guidance: str | None = None,
        explicit_route: MutationRoute | None = None,
        parent_roles: tuple[ParentRole, ...] = (),
    ) -> dict:
        """Execute mutation agent.

        Args:
            input: List of parent programs to mutate
            mutation_mode: Mutation mode

        Returns:
            Dict with 'code', 'structured_output', 'prompt_id', and other results
        """
        initial_state: MutationState = {
            "input": input,
            "mutation_mode": mutation_mode,
            "messages": [],
            "llm_response": None,
            "final_code": "",
            "mutation_label": "",
            "explicit_regime_guidance": explicit_regime_guidance,
            "explicit_route": explicit_route,
            "parent_roles": parent_roles,
        }

        final_state = await self.graph.ainvoke(initial_state)
        result = final_state.get("parsed_output", {})
        # Forward prompt_id from state into result for operator to stamp in metadata
        result["prompt_id"] = final_state.get("prompt_id")
        result["mutation_regime"] = final_state.get("selected_mutation_regime")
        return result

    async def acall_llm(self, state: MutationState) -> MutationState:
        """Call LLM with structured output.

        Uses the structured LLM to get a MutationStructuredOutput response.
        Emits exactly one LLM_CALL canonical event per invocation (success or
        failure) so mutation LLM latency joins the same observability stream
        as LineageAgent / InsightsAgent — see ``gigaevo.monitoring.events``.

        Args:
            state: State with messages field

        Returns:
            Updated state with llm_response and structured_output fields
        """
        t0 = time.monotonic()
        error_type: str | None = None
        ok = False
        structured_response: Any = None
        try:
            with llm_stage_context(self.__class__.__name__):
                structured_response = await self.structured_llm.ainvoke(
                    state["messages"]
                )
            state["llm_response"] = structured_response
            state["structured_output"] = structured_response
            if "metadata" not in state:
                state["metadata"] = {}
            model_used = get_selected_model()
            if model_used:
                state["metadata"]["model_used"] = model_used
            ok = True

            logger.debug(
                "[MutationAgent] Received structured output — archetype: {}, model: {}",
                structured_response.archetype,
                model_used or "(single model)",
            )

        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"[MutationAgent] Structured LLM call failed: {e}")
            state["error"] = str(e)
            state["llm_response"] = None
        finally:
            try:
                model = getattr(self.llm, "model_name", None) or (
                    get_selected_model() or "unknown"
                )
                usage = get_last_token_usage()
                _emit_event(
                    LLMCall(
                        stage="MutationAgent",
                        endpoint="",
                        model=str(model),
                        attempt=1,
                        ok=ok,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        tokens_in=usage.context if usage else 0,
                        tokens_out=usage.generated if usage else 0,
                        error_type=error_type,
                    )
                )
            except Exception:  # pragma: no cover — never fail the call on logging
                logger.opt(exception=True).debug(
                    "[MutationAgent] LLM_CALL emission failed"
                )

        return state

    def _refresh_prompts_from_fetcher(self, state: MutationState) -> None:
        """Refresh system and user prompts from the dynamic co-evolving fetcher.

        Stamps prompt_id in state for downstream tracking.
        Called only when prompt_fetcher.is_dynamic is True.
        """
        assert self._prompt_fetcher is not None
        assert self._metrics_formatter is not None
        fetched_sys = self._prompt_fetcher.fetch("mutation", "system")
        self.system_prompt = fetched_sys.text.format(
            task_description=self._task_description,
            metrics_description=self._metrics_formatter.format_metrics_description(),
        )
        state["prompt_id"] = fetched_sys.prompt_id
        fetched_user = self._prompt_fetcher.fetch("mutation", "user")
        if fetched_user.prompt_id is not None:
            self.user_prompt_template = fetched_user.text

    def build_prompt(self, state: MutationState) -> MutationState:
        """Build mutation prompt from parent programs.

        Uses pre-formatted mutation context from MutationContextStage that includes:
        - Metrics (formatted)
        - Insights
        - Family tree lineage

        If a dynamic prompt_fetcher is configured (is_dynamic=True), refreshes the
        system prompt from the co-evolving archive and stamps prompt_id in state.

        Args:
            state: Current state with parents field

        Returns:
            Updated state with messages field and optional prompt_id
        """
        if (
            self._prompt_fetcher is not None
            and self._prompt_fetcher.is_dynamic
            and self._metrics_formatter is not None
        ):
            self._refresh_prompts_from_fetcher(state)
        else:
            state["prompt_id"] = None

        parents = state["input"]
        user_prompt = self.build_user_prompt(parents, state=state)

        # Store prompts in state for logging
        state["system_prompt"] = self.system_prompt
        state["user_prompt"] = user_prompt

        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        state["messages"] = messages

        logger.info(
            f"[MutationAgent] Built prompt with {len(parents)} parents "
            f"(system: {len(self.system_prompt)} chars, "
            f"user: {len(user_prompt)} chars, "
            f"prompt_id={state.get('prompt_id', 'N/A')})"
        )
        # Dump full prompts to file for offline verification
        self._dump_prompt_to_file(
            state.get("prompt_id"), self.system_prompt, user_prompt
        )

        return state

    def build_user_prompt(
        self, parents: list[Program], state: MutationState | None = None
    ) -> str:
        """Build the mutation user prompt for a set of parents."""
        explicit_route = (
            state.get("explicit_route") if state is not None else None
        )
        parent_roles = (
            state.get("parent_roles", ()) if state is not None else ()
        )
        if parent_roles and len(parent_roles) != len(parents):
            raise ValueError("parent_roles must align one-to-one with parents")
        if (
            explicit_route is not None
            and self.route_context_provider is not None
            and not parent_roles
        ):
            parent_roles = tuple("target_island" for _ in parents)

        parent_blocks = self._build_parent_blocks(
            parents,
            route=explicit_route,
            parent_roles=parent_roles,
        )
        memory_block = self._build_memory_block(parents)
        if memory_block:
            parent_blocks = f"{parent_blocks}\n\n{memory_block}"
        prompt_fields = MutationPromptFields(
            count=len(parents), parent_blocks=parent_blocks
        )
        user_prompt = self.user_prompt_template.format(**prompt_fields.model_dump())

        if explicit_route is not None and self.route_context_provider is not None:
            assignment = self.route_context_provider.build_assignment(
                explicit_route,
                parents,
                parent_roles,
            )
            external = self.route_context_provider.build_external_context(
                explicit_route
            )
            guidance = explicit_route.guidance.strip()
            if state is not None:
                state["selected_mutation_regime"] = explicit_route.regime_id
            return "\n\n".join(
                part
                for part in (
                    assignment.strip(),
                    user_prompt,
                    external.strip(),
                    guidance,
                )
                if part.strip()
            )

        live_path_store = self._build_live_path_store_block()
        if live_path_store:
            user_prompt = f"{user_prompt}\n\n{live_path_store}"
        explicit = (
            state.get("explicit_regime_guidance") if state is not None else None
        )
        regime = explicit if explicit is not None else self._sample_mutation_regime()
        if regime:
            if state is not None:
                state["selected_mutation_regime"] = regime
            user_prompt = f"{user_prompt}\n\n{self._format_mutation_regime(regime)}"
        elif state is not None:
            state["selected_mutation_regime"] = None
        return user_prompt

    def _sample_mutation_regime(self) -> str | None:
        """Sample one optional diversity instruction for this mutation call."""
        if not self.mutation_regime_guidance:
            return None
        if random.random() >= self.mutation_regime_probability:
            return None
        regimes = [item[0] for item in self.mutation_regime_guidance]
        weights = [item[1] for item in self.mutation_regime_guidance]
        return random.choices(regimes, weights=weights, k=1)[0]

    @staticmethod
    def _parse_mutation_regime_guidance(items: list[Any]) -> list[tuple[str, float]]:
        parsed: list[tuple[str, float]] = []
        for item in items:
            weight = 1.0
            text: str
            if isinstance(item, Mapping) or (
                hasattr(item, "get") and not isinstance(item, str)
            ):
                raw_text = (
                    item.get("text")
                    or item.get("guidance")
                    or item.get("regime")
                    or item.get("content")
                    or ""
                )
                text = str(raw_text).strip()
                raw_weight = item.get("probability", item.get("weight", 1.0))
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    weight = 0.0
            else:
                text = str(item).strip()
            if text and weight > 0:
                parsed.append((text, weight))
        return parsed

    @staticmethod
    def _format_mutation_regime(regime: str) -> str:
        return regime.strip()

    def _build_parent_blocks(
        self,
        parents: list[Program],
        *,
        route: MutationRoute | None = None,
        parent_roles: tuple[ParentRole, ...] = (),
    ) -> str:
        """Build formatted parent blocks for the mutation prompt."""
        blocks: list[str] = []
        for i, p in enumerate(parents):
            raw_context = str(
                p.metadata.get(MUTATION_CONTEXT_METADATA_KEY) or ""
            )
            role = parent_roles[i] if parent_roles else None
            if (
                route is not None
                and role is not None
                and self.route_context_provider is not None
            ):
                formatted_context = (
                    self.route_context_provider.filter_parent_context(
                        route,
                        p,
                        role,
                        raw_context,
                    )
                )
            else:
                formatted_context = self._strip_aux_sections(raw_context)

            role_suffix = f" [role={role}]" if role is not None else ""
            block = f"""=== Parent {i + 1}{role_suffix} ===
```python
{p.code}
```

{formatted_context}
"""
            blocks.append(block)

        return "\n\n".join(blocks)

    @staticmethod
    def _strip_aux_sections(text: str) -> str:
        """Keep validator/path aux reports out of mutation prompts."""
        return re.sub(
            r"(?ms)^## Program execution aux info\b.*?(?=^---$|^## |\Z)",
            "",
            text,
        ).strip()

    def _build_live_path_store_block(self) -> str:
        """Load a fresh saved-path summary for the current mutation prompt."""
        if self.live_path_store_problem_dir is None:
            return ""

        problem_dir = self.live_path_store_problem_dir.resolve()
        path_store_py = problem_dir / "path_store.py"
        if not path_store_py.exists():
            return ""

        try:
            problem_dir_str = str(problem_dir)
            if problem_dir_str not in sys.path:
                sys.path.insert(0, problem_dir_str)

            module_name = f"_gigaevo_mutation_path_store_{abs(hash(problem_dir_str))}"
            spec = importlib.util.spec_from_file_location(module_name, path_store_py)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load path_store.py from {problem_dir}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            # A problem owns its default store location through DATA_PATH.  An
            # explicit config root remains an override for legacy experiments.
            path_store = (
                module.PathStore(root_dir=str(self.live_path_store_root_dir))
                if self.live_path_store_root_dir is not None
                else module.PathStore()
            )
            summary = path_store.summarize(top_k=self.live_path_store_top_k)
        except Exception as exc:
            logger.warning("[MutationAgent] live path store unavailable: {}", exc)
            summary = f"path store unavailable: {exc}"

        return f"## Live Path Store\n\n{summary}"

    def _build_memory_block(self, parents: list[Program]) -> str:
        """Build a single memory block from any parent metadata."""
        for parent in parents:
            memory_text = str(
                parent.metadata.get(MUTATION_MEMORY_METADATA_KEY, "")
            ).strip()
            if memory_text:
                return f"## Memory Instructions\n{memory_text}"
        return ""

    def parse_response(self, state: MutationState) -> MutationState:
        """Parse LLM structured response to extract code and metadata.

        Handles both rewrite mode (direct code from structured output) and diff mode
        (extract and apply diff from code field).

        Args:
            state: Current state with llm_response (structured output) field

        Returns:
            Updated state with parsed_output field containing final code and metadata
        """
        structured_output: MutationStructuredOutput | None = state.get(
            "structured_output"
        )
        model_used = state.get("metadata", {}).get("model_used")

        if structured_output is None:
            error_msg = state.get("error", "No structured output received")
            logger.error(f"[MutationAgent] No structured output: {error_msg}")
            state["parsed_output"] = {
                "code": "",
                "structured_output": None,
                "error": error_msg,
                "model_used": model_used,
            }
            return state

        try:
            # Get code from structured output
            code_from_llm = structured_output.code

            # Fix JSON-escaped sequences from structured output serialization.
            # LLMs sometimes produce literal \n, \t, \" in the code field when
            # they confuse JSON escaping with Python syntax.
            code_from_llm = self._fix_json_escaped_code(code_from_llm)

            if state["mutation_mode"] == "diff":
                # Apply diff to parent code
                parents = state["input"]
                if len(parents) != 1:
                    raise ValueError("Diff mode requires exactly 1 parent")

                parent_code = parents[0].code
                # The code field contains the diff in diff mode
                final_code = self._apply_diff_and_extract(parent_code, code_from_llm)
            else:
                final_code = self._extract_code_block(code_from_llm)

            # Guard: reject code that is a JSON template echoed back instead of Python
            if "def " not in final_code and final_code.lstrip().startswith("{"):
                raise ValueError(
                    "LLM returned JSON template as code instead of Python. "
                    f"Code starts with: {final_code[:80]!r}"
                )

            state["final_code"] = final_code

            # Convert structured output to dict for storage
            structured_dict = structured_output.model_dump()

            state["parsed_output"] = {
                "code": final_code,
                "structured_output": structured_dict,
                "archetype": structured_output.archetype,
                "justification": structured_output.justification,
                "insights_used": structured_output.insights_used,
                "changes": structured_output.changes,
                "model_used": model_used,
                "mutation_regime": state.get("selected_mutation_regime"),
            }

            logger.debug(
                f"[MutationAgent] Extracted code ({len(final_code)} chars) "
                f"with archetype: {structured_output.archetype}"
            )

        except Exception as e:
            logger.error(f"[MutationAgent] Failed to parse structured response: {e}")
            state["error"] = str(e)
            state["parsed_output"] = {
                "code": "",
                "structured_output": (
                    structured_output.model_dump() if structured_output else None
                ),
                "error": str(e),
                "model_used": model_used,
            }

        return state

    @staticmethod
    def _fix_json_escaped_code(code: str) -> str:
        """Fix JSON-escaped sequences in code from structured output.

        LLMs using structured output sometimes produce literal JSON escape
        sequences in the code field instead of the actual characters:
        - ``\\"`` instead of ``"`` (double-escaped quotes)
        - ``\\n`` instead of actual newlines (escaped newlines)
        - ``\\t`` instead of actual tabs (escaped tabs)

        This happens when the model confuses JSON string escaping with
        the Python code content. We only apply the fix when the original
        code fails to parse and the cleaned version parses successfully.
        """
        # Quick check: does code contain any JSON escape sequences?
        if "\\n" not in code and '\\"' not in code and "\\t" not in code:
            return code
        try:
            ast.parse(code)
            return code  # Already valid — don't touch it
        except SyntaxError:
            pass

        # Try unescaping JSON sequences
        cleaned = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        try:
            ast.parse(cleaned)
            logger.debug(
                '[MutationAgent] Fixed JSON-escaped code (\\n={}, \\t={}, \\"={})',
                code.count("\\n"),
                code.count("\\t"),
                code.count('\\"'),
            )
            return cleaned
        except SyntaxError:
            return code  # Unescaping didn't help — return original

    def _extract_code_block(self, text: str) -> str:
        """Extract outer fenced code block from LLM response.

        Treats only fences at start-of-line as valid markers to avoid
        premature closing on backticks inside code (e.g., docstrings).

        Args:
            text: LLM response text

        Returns:
            Extracted code string
        """
        # Find first opening fence at start-of-line
        open_match = re.search(r"(?m)^```(?:[a-zA-Z0-9_+\-]+)?\s*$", text)
        if not open_match:
            return text.strip()

        # Find closing fence after opener
        after_open = text[open_match.end() :]
        close_match = re.search(r"(?m)^```\s*$", after_open)
        if not close_match:
            return text.strip()

        code_block = after_open[: close_match.start()]

        # Trim single leading newline if present
        if code_block.startswith("\n"):
            code_block = code_block[1:]

        return code_block.rstrip()

    def _apply_diff_and_extract(self, original_code: str, response_text: str) -> str:
        """Extract diff from response and apply to original code.

        Args:
            original_code: Original parent code
            response_text: LLM response containing diff

        Returns:
            Patched code

        Raises:
            ValueError: If diff is empty or patch fails
        """
        diff_text = self._extract_code_block(response_text)
        if not diff_text.strip():
            raise ValueError("Empty diff returned by LLM")

        try:
            return diffpatch.apply_patch(original_code, diff_text)
        except Exception as e:
            raise ValueError(f"Failed to apply patch: {e}") from e
