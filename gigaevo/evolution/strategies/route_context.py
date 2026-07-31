"""Route-specific mutation-context extension point."""

from __future__ import annotations

from typing import Protocol

from gigaevo.evolution.strategies.base import (
    MutationRoute,
    ParentRole,
)
from gigaevo.programs.program import Program


class MutationRouteContextProvider(Protocol):
    """Build route-aware prompt context without coupling it to a strategy."""

    def route_is_available(self, route: MutationRoute) -> bool:
        """Return whether the route currently has enough external context."""
        ...

    def build_route_guidance(self, route: MutationRoute) -> str | None:
        """Return problem-owned binding guidance, or None for inline guidance."""
        ...

    def build_assignment(
        self,
        route: MutationRoute,
        parents: list[Program],
        parent_roles: tuple[ParentRole, ...],
    ) -> str:
        """Describe the sampled route and the roles of its selected parents."""
        ...

    def filter_parent_context(
        self,
        route: MutationRoute,
        parent: Program,
        role: ParentRole,
        mutation_context: str,
    ) -> str:
        """Select the parent evidence relevant to this route."""
        ...

    def build_external_context(self, route: MutationRoute) -> str:
        """Build route-specific evidence that does not belong to one parent."""
        ...
