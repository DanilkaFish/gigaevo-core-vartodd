from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextvars import ContextVar
import json
import os
import random
import re
from typing import TYPE_CHECKING, Any, cast
import urllib.error
import urllib.request

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from loguru import logger

from gigaevo.llm.token_tracking import TokenTracker, TokenUsage
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


_selected_model_var: ContextVar[str | None] = ContextVar("selected_model", default=None)
_last_token_usage_var: ContextVar[TokenUsage | None] = ContextVar(
    "last_token_usage", default=None
)


def get_selected_model() -> str | None:
    """Return the last selected model name for the current async context."""
    return _selected_model_var.get()


def get_last_token_usage() -> TokenUsage | None:
    """Return token usage from the most recent LLM call in the current async context.

    Populated by ``MultiModelRouter`` and ``_StructuredOutputRouter`` after every
    invocation that yields a response with usage metadata. ``None`` if the last
    response had no usage info (e.g. stream chunk without metadata).
    """
    return _last_token_usage_var.get()


def _remember_selected_model(model_name: str) -> None:
    _selected_model_var.set(model_name)


def _remember_token_usage(response: Any) -> None:
    usage = TokenUsage.from_response(response)
    if usage is not None:
        _last_token_usage_var.set(usage)


def _create_langfuse_handler() -> CallbackHandler | None:
    """Create Langfuse handler if credentials are configured.

    In langfuse v4 the handler is a thin wrapper over the global ``Langfuse``
    client returned by ``get_client()``; flush tuning must be set on the
    client at construction time. Constructing the ``Langfuse`` client here
    installs it as the singleton the handler picks up.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    Langfuse(flush_at=1, flush_interval=1)
    handler = CallbackHandler()
    logger.info("[MultiModelRouter] Langfuse tracing enabled")
    return handler


def _with_langfuse(
    config: RunnableConfig | None,
    handler: CallbackHandler | None,
    model_name: str | None = None,
) -> RunnableConfig | None:
    """Add Langfuse handler and metadata to config."""
    if handler is None:
        return config

    cfg: dict[str, Any] = dict(config or {})
    callbacks: list[Any] = cfg.setdefault("callbacks", [])
    if handler not in callbacks:
        callbacks.append(handler)

    if model_name:
        metadata: dict[str, Any] = cfg.setdefault("metadata", {})
        metadata["selected_model"] = model_name

    return cast(RunnableConfig, cfg)


class MultiModelRouter(Runnable):
    """Probabilistic model router with token tracking and Langfuse tracing.

    Example:
        >>> router = MultiModelRouter(
        ...     [ChatOpenAI(model="gpt-4"), ChatOpenAI(model="gpt-3.5-turbo")],
        ...     [0.8, 0.2],
        ...     writer=metrics_writer,
        ...     name="mutation",  # metrics go to llm/tokens/mutation/...
        ... )
        >>> response = await router.ainvoke("Hello!")
        >>> structured = router.with_structured_output(MySchema)
    """

    def __init__(
        self,
        models: list[ChatOpenAI],
        probabilities: list[float],
        writer: LogWriter | None = None,
        name: str = "default",
        structured_output_method: str | None = None,
        structured_output_optional_tool_model_prefixes: Sequence[str] | None = None,
    ):
        if len(models) != len(probabilities):
            raise ValueError(
                f"Length mismatch: {len(models)} models, {len(probabilities)} probabilities"
            )
        if any(p <= 0 for p in probabilities):
            raise ValueError("All probabilities must be positive")

        self.models = models
        self.model_names = [m.model_name for m in models]
        self.probabilities = [p / sum(probabilities) for p in probabilities]
        self._task_model_map: dict[int, str] = {}
        self._name = name
        self._structured_output_method = structured_output_method
        self._structured_output_optional_tool_model_prefixes = tuple(
            structured_output_optional_tool_model_prefixes or ()
        )

        self._tracker = TokenTracker(
            name=name,
            writer=writer.bind(path=["llm", "tokens"]) if writer else None,
        )
        self._langfuse = _create_langfuse_handler()

        model_desc = ", ".join(
            f"{n} ({p:.0%})" for n, p in zip(self.model_names, self.probabilities)
        )
        logger.info(
            "[MultiModelRouter:{}] Initialized with {} models: {}",
            name,
            len(models),
            model_desc,
        )
        # Log base URLs for debugging server connectivity
        for m in models:
            # ChatOpenAI exposes base_url as a property (langchain 0.1+)
            base_url = getattr(m, "base_url", None)
            if base_url:
                logger.info(
                    "[MultiModelRouter:{}] Model {} at {}", name, m.model_name, base_url
                )

        self._verify_models()

    def _verify_models(self) -> None:
        """Startup probe — verify configured models exist on their servers.

        Raises ``RuntimeError`` if any base URL is unreachable. Missing model
        names on a reachable server are logged as warnings.
        """
        by_base_url: dict[str, list[ChatOpenAI]] = {}
        for model in self.models:
            base_url = getattr(model, "base_url", None) or getattr(
                model, "openai_api_base", None
            )
            if not isinstance(base_url, str):
                continue
            by_base_url.setdefault(base_url, []).append(model)

        for base_url, models in by_base_url.items():
            # Use the key actually configured on the model so the probe
            # auths the same way as real requests. Fall back to env vars.
            secret: Any = getattr(models[0], "openai_api_key", None)
            if hasattr(secret, "get_secret_value"):
                model_token = secret.get_secret_value()
            elif isinstance(secret, str):
                model_token = secret
            else:
                model_token = ""
            token = (
                model_token
                or os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENAI_API_TOKEN", "")
            )
            url = f"{base_url.rstrip('/')}/models"
            req = urllib.request.Request(
                url, method="GET", headers={"Authorization": f"Bearer {token}"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                    payload = json.loads(resp.read())
            except (urllib.error.URLError, OSError) as exc:
                raise RuntimeError(
                    f"[MultiModelRouter:{self._name}] Cannot reach LLM endpoint "
                    f"at {base_url}: {exc}"
                ) from exc

            available = {entry["id"] for entry in payload.get("data", [])}
            for model in models:
                if model.model_name in available:
                    logger.info(
                        "[MultiModelRouter:{}] Model {} verified on {}",
                        self._name,
                        model.model_name,
                        base_url,
                    )
                else:
                    logger.warning(
                        "[MultiModelRouter:{}] Model {} NOT FOUND on {}. Available: {}",
                        self._name,
                        model.model_name,
                        base_url,
                        sorted(available),
                    )

    @staticmethod
    def _current_task_id() -> int | None:
        """Return ``id(asyncio.current_task())`` or *None* outside an event loop."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return None
        return id(task) if task is not None else None

    def _select(self) -> tuple[ChatOpenAI, str]:
        """Select a model based on probabilities."""
        idx = random.choices(range(len(self.models)), weights=self.probabilities)[0]
        model, name = self.models[idx], self.model_names[idx]
        _remember_selected_model(name)
        tid = self._current_task_id()
        if tid is not None:
            self._task_model_map[tid] = name
        return model, name

    def get_last_model(self) -> str | None:
        """Return the model name selected in the most recent ``_select()`` call for the current async task."""
        tid = self._current_task_id()
        if tid is not None:
            return self._task_model_map.pop(tid, None)
        return None

    def on_mutation_outcome(
        self,
        program: Program,
        parents: list[Program],
        outcome: Any = None,
    ) -> None:
        """Callback when a mutated program completes evaluation. Override for feedback."""

    def _config(
        self, config: RunnableConfig | None, model_name: str
    ) -> RunnableConfig | None:
        return _with_langfuse(config, self._langfuse, model_name)

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> BaseMessage:
        model, name = self._select()
        response = model.invoke(input, self._config(config, name), **kwargs)
        self._tracker.track(response, name)
        _remember_token_usage(response)
        return response

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> BaseMessage:
        model, name = self._select()
        response = await model.ainvoke(input, self._config(config, name), **kwargs)
        self._tracker.track(response, name)
        _remember_token_usage(response)
        return response

    def stream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Iterator[BaseMessage]:
        model, name = self._select()
        last = None
        for chunk in model.stream(input, self._config(config, name), **kwargs):
            last = chunk
            yield chunk
        if last:
            self._tracker.track(last, name)
            _remember_token_usage(last)

    async def astream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> AsyncIterator[BaseMessage]:
        model, name = self._select()
        last = None
        async for chunk in model.astream(input, self._config(config, name), **kwargs):
            last = chunk
            yield chunk
        if last:
            self._tracker.track(last, name)
            _remember_token_usage(last)

    def with_structured_output(self, schema: Any, **kwargs) -> _StructuredOutputRouter:
        """Create a router that returns parsed Pydantic models with token tracking.

        If the router was constructed with ``structured_output_method`` (e.g.
        ``"json_schema"`` or ``"function_calling"``), that method is forwarded
        to each underlying model. Explicit ``method`` in ``**kwargs`` wins.
        Models whose names match ``structured_output_optional_tool_model_prefixes``
        receive the same schema as an optional OpenAI tool with no ``tool_choice``.
        """
        if self._structured_output_method is not None:
            kwargs.setdefault("method", self._structured_output_method)
        wrapped = [
            self._wrap_structured_model(m, name, schema, kwargs)
            for m, name in zip(self.models, self.model_names, strict=True)
        ]
        return _StructuredOutputRouter(
            wrapped,
            self.model_names,
            self.probabilities,
            self._langfuse,
            self._tracker,
            task_model_map=self._task_model_map,
        )

    def _uses_optional_tool_structured_output(self, model_name: str) -> bool:
        return any(
            model_name.startswith(prefix)
            for prefix in self._structured_output_optional_tool_model_prefixes
        )

    def _wrap_structured_model(
        self, model: ChatOpenAI, model_name: str, schema: Any, kwargs: dict[str, Any]
    ) -> Runnable:
        if self._uses_optional_tool_structured_output(model_name):
            return _OptionalToolStructuredOutput(model, schema, kwargs)
        return model.with_structured_output(schema, include_raw=True, **kwargs)


class _OptionalToolStructuredOutput(Runnable):
    """Structured-output adapter that does not force a tool call.

    Some thinking models exposed through OpenRouter reject forced tool choice, but
    still accept a normal ``tools`` list. This adapter sends the schema as an
    optional tool, then validates either a returned tool call or JSON content into
    the requested Pydantic schema. The returned mapping mirrors LangChain's
    ``include_raw=True`` shape consumed by :class:`_StructuredOutputRouter`.
    """

    def __init__(self, model: ChatOpenAI, schema: Any, kwargs: dict[str, Any]):
        self._model = model
        self._schema = schema
        self._tool = self._to_openai_tool(schema)
        self._invoke_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"include_raw", "method", "tool_choice"}
        }

    @staticmethod
    def _to_openai_tool(schema: Any) -> dict[str, Any]:
        try:
            from langchain_core.utils.function_calling import convert_to_openai_tool
        except Exception:  # pragma: no cover - version compatibility fallback
            convert_to_openai_tool = None

        if convert_to_openai_tool is not None:
            return cast(dict[str, Any], convert_to_openai_tool(schema))

        if isinstance(schema, dict):
            if schema.get("type") == "function":
                return schema
            name = str(schema.get("title") or "StructuredOutput")
            return {
                "type": "function",
                "function": {"name": name, "parameters": schema},
            }

        json_schema = schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": getattr(schema, "__name__", "StructuredOutput"),
                "description": (getattr(schema, "__doc__", "") or "").strip(),
                "parameters": json_schema,
            },
        }

    @staticmethod
    def _json_loads(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"expected JSON object or string, got {type(value).__name__}"
            )
        return cast(dict[str, Any], json.loads(value))

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def _extract_tool_payload(self, raw: Any) -> dict[str, Any] | None:
        tool_calls = getattr(raw, "tool_calls", None)
        if isinstance(tool_calls, list) and tool_calls:
            call = tool_calls[0]
            if isinstance(call, dict):
                args = call.get("args")
                if args is None:
                    args = (call.get("function") or {}).get("arguments")
                return self._json_loads(args)

            args = getattr(call, "args", None)
            if args is None:
                function = getattr(call, "function", None)
                args = getattr(function, "arguments", None)
            return self._json_loads(args)

        additional_kwargs = getattr(raw, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            raw_calls = additional_kwargs.get("tool_calls")
            if isinstance(raw_calls, list) and raw_calls:
                function = raw_calls[0].get("function") or {}
                return self._json_loads(function.get("arguments"))
        return None

    def _extract_content_payload(self, raw: Any) -> dict[str, Any]:
        text = self._strip_json_fence(self._content_text(getattr(raw, "content", "")))
        try:
            return self._json_loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return self._json_loads(text[start : end + 1])
            raise

    def _parse(self, raw: Any) -> Any:
        payload = self._extract_tool_payload(raw)
        if payload is None:
            payload = self._extract_content_payload(raw)

        if hasattr(self._schema, "model_validate"):
            return self._schema.model_validate(payload)
        if hasattr(self._schema, "parse_obj"):
            return self._schema.parse_obj(payload)
        return payload

    def _process_raw(self, raw: Any) -> dict[str, Any]:
        try:
            parsed = self._parse(raw)
            return {"raw": raw, "parsed": parsed}
        except Exception as exc:
            return {"raw": raw, "parsed": None, "parsing_error": str(exc)}

    def _call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        call_kwargs = {**self._invoke_kwargs, **kwargs}
        call_kwargs.pop("tool_choice", None)
        call_kwargs["tools"] = [self._tool]
        return call_kwargs

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        raw = self._model.invoke(input, config, **self._call_kwargs(kwargs))
        return self._process_raw(raw)

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        raw = await self._model.ainvoke(input, config, **self._call_kwargs(kwargs))
        return self._process_raw(raw)


class _StructuredOutputRouter(Runnable):
    """Router for structured output with token tracking from raw responses."""

    def __init__(
        self,
        models: list,
        model_names: list[str],
        probabilities: list[float],
        langfuse: CallbackHandler | None,
        tracker: TokenTracker,
        task_model_map: dict[int, str] | None = None,
        select_override: Callable[[], tuple[Any, str]] | None = None,
    ):
        self._models = models
        self._names = model_names
        self._probs = probabilities
        self._langfuse = langfuse
        self._tracker = tracker
        self._task_model_map = task_model_map
        self._select_override = select_override

    def _select(self) -> tuple[Any, str]:
        if self._select_override is not None:
            return self._select_override()
        idx = random.choices(range(len(self._models)), weights=self._probs)[0]
        model, name = self._models[idx], self._names[idx]
        _remember_selected_model(name)
        if self._task_model_map is not None:
            tid = MultiModelRouter._current_task_id()
            if tid is not None:
                self._task_model_map[tid] = name
        return model, name

    def _config(
        self, config: RunnableConfig | None, model_name: str
    ) -> RunnableConfig | None:
        return _with_langfuse(config, self._langfuse, model_name)

    def _process(self, response: dict, name: str) -> Any:
        raw = response.get("raw")
        if raw:
            self._tracker.track(raw, name)
            _remember_token_usage(raw)
        parsed = response.get("parsed")
        if parsed is None and raw is not None:
            content = getattr(raw, "content", "")
            excerpt = (content[:500] + "…") if len(content) > 500 else content
            parsing_error = response.get("parsing_error")
            detail = f" parsing_error={parsing_error!r}." if parsing_error else ""
            raise ValueError(
                f"[{name}] Structured output parse failed: raw response had no parsable schema. "
                f"{detail} content_excerpt={excerpt!r}"
            )
        return parsed

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        model, name = self._select()
        return self._process(
            model.invoke(input, self._config(config, name), **kwargs), name
        )

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        model, name = self._select()
        return self._process(
            await model.ainvoke(input, self._config(config, name), **kwargs), name
        )
