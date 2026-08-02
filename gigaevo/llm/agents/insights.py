"""Insights agent for program analysis using LangGraph.

This agent analyzes programs to generate actionable insights for evolution.
ALL LLM-related logic lives here - stages are just thin wrappers.
"""

from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.evolution.strategies.route_context import InsightsRouteContextProvider
from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import OPTIMIZATION_STAGES, Program


class ProgramInsight(BaseModel):
    """Single structured insight about a program.

    Schema v2 (2026-05-23): split the legacy free-string ``insight`` into
    structured grounding fields (``anchor_quote``, ``evidence_source``,
    ``mechanism``, ``substitute``, ``evidence_refs``, ``relation_to_lineage``)
    so the suggester cannot hide a missing anchor or missing substitute
    inside one ≤35-word sentence. ``insight`` is kept as an OPTIONAL legacy
    fallback for the off-path :class:`InsightsAgent` (legacy ``InsightsStage``
    prompt still emits the free string); the renderer prefers the structured
    fields when present.
    """

    type: str = Field(description="Insight category")
    anchor_quote: str = Field(
        default="",
        description=(
            "Literal string copied verbatim from one of the evidence inputs "
            "(numeric constant, identifier, cluster label, fitness number). "
            "Must appear in the source verbatim."
        ),
    )
    evidence_source: str = Field(
        default="",
        description=(
            "Which input the anchor came from: program | metrics | intra_memory "
            "| memory_cards | ancestral_trail | evolutionary_statistics | program_aux."
        ),
    )
    mechanism: str = Field(
        default="",
        description="One-clause explanation of why the anchor moves the primary metric.",
    )
    substitute: str = Field(
        default="",
        description=(
            "Concrete target value, replacement pattern, or specific guard. "
            "NOT a direction."
        ),
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of identifiers of the exact evidence items cited "
            "(memory card IDs, cluster labels, trail depth_back values). "
            "Empty when not citing specific items."
        ),
    )
    relation_to_lineage: str = Field(
        default="",
        description=(
            "Optional one-clause note describing how this suggestion DIFFERS "
            "from prior tried strategies (required when refining a "
            "negative/regressed cluster)."
        ),
    )
    insight: str = Field(
        default="",
        description=(
            "DEPRECATED legacy free-string insight (≤35 words). Retained for "
            "the legacy InsightsAgent prompt; new prompts populate the "
            "structured fields above instead."
        ),
    )
    tag: str = Field(description="Tag for the insight")
    severity: str = Field(description="Severity of the insight")


class ProgramInsights(BaseModel):
    """Collection of program insights."""

    insights: list[ProgramInsight] = Field(
        description="List of actionable insights",
    )


class InsightsState(TypedDict):
    """Complete state for insights analysis.

    This is the LangGraph state - it flows through all nodes.
    """

    # Input
    program: Program

    # LLM interaction
    messages: list[BaseMessage]
    llm_response: AIMessage | ProgramInsights | None

    # Output
    insights: ProgramInsights | None

    # Metadata
    metadata: dict


class InsightsAgent(LangGraphAgent):
    StateSchema = InsightsState

    @staticmethod
    def _strip_task_description_from_user_template(template: str) -> str:
        """Ensure insights task_description is system-only, even with custom user prompts."""
        lines = template.splitlines()
        drop: set[int] = set()
        task_labels = {
            "task",
            "task:",
            "task description",
            "task description:",
            "problem",
            "problem:",
        }
        for idx, line in enumerate(lines):
            if "{task_description}" not in line:
                continue
            drop.add(idx)
            if idx > 0 and lines[idx - 1].strip().lower() in task_labels:
                drop.add(idx - 1)

        cleaned_lines: list[str] = []
        for idx, line in enumerate(lines):
            if idx in drop:
                continue
            cleaned_lines.append(line.replace("{task_description}", ""))
        return "\n".join(cleaned_lines).strip()
    """Agent for generating program insights.

    This agent does ALL the heavy lifting:
    - Formats metrics and errors
    - Builds prompts
    - Calls LLM
    - Parses structured output

    Stages just call agent.arun(program) and store results.
    """

    StateSchema = InsightsState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt_template: str,
        user_prompt_template: str,
        max_insights: int,
        metrics_formatter: MetricsFormatter,
        route_context_provider: InsightsRouteContextProvider | None = None,
    ):
        """Initialize insights agent.

        Args:
            llm: LangChain chat model or router
            system_prompt_template: System prompt template (with {task_description} etc)
            user_prompt_template: User prompt template (with {code}, {metrics}, etc)
            max_insights: Maximum insights to generate
            metrics_formatter: Formatter for program metrics
            route_context_provider: Optional program-specific guidance source
        """
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = self._strip_task_description_from_user_template(
            user_prompt_template
        )
        self.max_insights = max_insights
        self.metrics_formatter = metrics_formatter
        self.route_context_provider = route_context_provider
        structured_llm = llm.with_structured_output(ProgramInsights)

        super().__init__(structured_llm)

    def build_prompt(self, state: InsightsState) -> InsightsState:
        """Build insights prompt - ALL formatting logic here.

        This method does:
        - Format metrics using metrics_formatter
        - Build error section
        - Format prompts with all variables
        - Create LangChain messages
        """
        program = state["program"]

        # Format metrics (agent responsibility!)
        metrics_text = (
            self.metrics_formatter.format_metrics_block(program.metrics)
            if program.metrics
            else "No metrics available"
        )

        errors = program.format_errors(
            include_traceback=True, exclude_stages=set(OPTIMIZATION_STAGES)
        )
        error_section = (
            f"**Error Analysis**: Focus on fixing or avoiding failure modes from stages:\n{errors}"
            if errors
            else ""
        )
        aux_parts = []
        aux_info = program.get_metadata("aux_info")
        if aux_info:
            aux_parts.append(str(aux_info))
        aux_context = "\n\n".join(aux_parts) if aux_parts else "No aux context available"

        user_prompt = self.user_prompt_template.format(
            code=program.code,
            metrics=metrics_text,
            aux_context=aux_context,
            error_section=error_section,
            max_insights=self.max_insights,
        )
        route_context = (
            self.route_context_provider.build_insights_context(program)
            if self.route_context_provider is not None
            else ""
        )
        if route_context.strip():
            user_prompt = (
                f"{user_prompt}\n\n"
                "## Required Island Regime for Insight Analysis\n\n"
                f"{route_context.strip()}"
            )

        state["messages"] = [
            SystemMessage(content=self.system_prompt_template),
            HumanMessage(content=user_prompt),
        ]

        return state

    def parse_response(self, state: InsightsState) -> InsightsState:
        """Parse LLM response (already validated by LangChain structured output)."""
        llm_response = state["llm_response"]
        if not isinstance(llm_response, ProgramInsights):
            raise ValueError(f"Expected ProgramInsights, got {type(llm_response)}")
        state["insights"] = llm_response
        return state

    async def arun_with_metadata(
        self,
        program: Program,
    ) -> tuple[ProgramInsights, dict[str, Any]]:
        """Run insights analysis and return LLM-call metadata.

        Args:
            program: Program to analyze

        Returns:
            Structured insights plus metadata captured during the LLM call.
        """
        initial_state: InsightsState = {
            "program": program,
            "messages": [],
            "llm_response": None,
            "insights": None,
            "metadata": {"program_id": program.id},
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state["insights"], dict(final_state.get("metadata") or {})

    async def arun(
        self,
        program: Program,
    ) -> ProgramInsights:
        """Run insights analysis on a program."""
        insights, _metadata = await self.arun_with_metadata(program)
        return insights
