"""Tests for direction-aware ``gigaevo top`` command construction in
``tools/canonical_benchmark/``.

Why this exists: when a benchmark problem's primary metric is
``higher_is_better: false`` (e.g. ``alphaevolve/erdos_minimum_overlap``,
sentinel = 1000.0), ``gigaevo top -n 1`` without ``--minimize`` returns
the SENTINEL — silently corrupting the benchmark row. We bake the
direction lookup into the script so future maintainers don't have to
remember the flag, and we lock the behavior in with these tests.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from tools.canonical_benchmark.benchmark import build_top_cmd
from tools.canonical_benchmark.run_benchmark import primary_metric_higher_is_better


class TestBuildTopCmd:
    def test_default_max_direction_omits_minimize(self) -> None:
        cmd = build_top_cmd(gigaevo_exe="gigaevo", problem="heilbron", db=0)
        assert cmd == ["gigaevo", "-r", "heilbron@0", "-f", "json", "top", "-n", "1"]
        assert "--minimize" not in cmd

    def test_higher_is_better_true_omits_minimize(self) -> None:
        cmd = build_top_cmd(
            gigaevo_exe="gigaevo",
            problem="heilbron",
            db=0,
            higher_is_better=True,
        )
        assert "--minimize" not in cmd

    def test_higher_is_better_false_appends_minimize(self) -> None:
        cmd = build_top_cmd(
            gigaevo_exe="gigaevo",
            problem="alphaevolve/erdos_minimum_overlap",
            db=6,
            higher_is_better=False,
        )
        assert cmd[-1] == "--minimize"
        assert "-r" in cmd
        assert "alphaevolve/erdos_minimum_overlap@6" in cmd

    def test_db_is_stringified_into_prefix(self) -> None:
        cmd = build_top_cmd(gigaevo_exe="gigaevo", problem="toy", db=42)
        assert "toy@42" in cmd


class TestPrimaryMetricHigherIsBetterCustomYAML:
    """Synthetic metrics files via monkeypatching REPO_ROOT — avoids touching
    the real problems tree. Confirms the helper handles malformed files,
    missing primary specs, and explicit ``higher_is_better: false``.
    """

    @pytest.fixture
    def temp_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from tools.canonical_benchmark import run_benchmark as rb

        monkeypatch.setattr(rb, "REPO_ROOT", tmp_path)
        return tmp_path

    def _write_metrics(self, repo: Path, problem: str, body: str) -> None:
        path = repo / "problems" / problem / "metrics.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))

    def test_explicit_higher_is_better_false(self, temp_repo: Path) -> None:
        self._write_metrics(
            temp_repo,
            "min_problem",
            """
            specs:
              fitness:
                is_primary: true
                higher_is_better: false
                sentinel_value: 1000.0
            """,
        )
        assert primary_metric_higher_is_better("min_problem") is False

    def test_explicit_higher_is_better_true(self, temp_repo: Path) -> None:
        self._write_metrics(
            temp_repo,
            "max_problem",
            """
            specs:
              fitness:
                is_primary: true
                higher_is_better: true
            """,
        )
        assert primary_metric_higher_is_better("max_problem") is True

    def test_primary_spec_without_direction_defaults_to_max(
        self, temp_repo: Path
    ) -> None:
        self._write_metrics(
            temp_repo,
            "ambiguous",
            """
            specs:
              fitness:
                is_primary: true
            """,
        )
        assert primary_metric_higher_is_better("ambiguous") is True

    def test_no_primary_spec_defaults_to_max(self, temp_repo: Path) -> None:
        self._write_metrics(
            temp_repo,
            "no_primary",
            """
            specs:
              fitness:
                is_primary: false
            """,
        )
        assert primary_metric_higher_is_better("no_primary") is True

    def test_malformed_yaml_defaults_to_max(self, temp_repo: Path) -> None:
        path = temp_repo / "problems" / "malformed" / "metrics.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is: not: valid: yaml: at all: [unclosed")
        assert primary_metric_higher_is_better("malformed") is True

    def test_empty_yaml_defaults_to_max(self, temp_repo: Path) -> None:
        self._write_metrics(temp_repo, "empty", "")
        assert primary_metric_higher_is_better("empty") is True
