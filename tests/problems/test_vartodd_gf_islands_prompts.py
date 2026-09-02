from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf_islands"


def _all_prompt_text() -> str:
    files = [
        PROBLEM_DIR / "task_description.txt",
        *(PROBLEM_DIR / "prompts").glob("*/*.txt"),
    ]
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def test_mutation_prompt_defines_parent_roles_and_path_authority() -> None:
    mutation = (
        PROBLEM_DIR / "prompts" / "mutation" / "system.txt"
    ).read_text(encoding="utf-8")
    compact = " ".join(mutation.split())

    assert "`target_island` parent" in compact
    assert "primary implementation base" in compact
    assert "bootstrap_donor" in compact
    assert "mixed_donor" in compact
    assert "without changing the assigned destination" in compact
    assert "External inventories or handles" in compact
    assert "only" in compact.lower()
    assert "use only currently supplied entries" in compact
    assert "island-local" in compact
    assert "percentile" in compact
    assert "raw execution evidence" in compact


def test_task_description_keeps_current_api_and_route_card_semantics() -> None:
    task = (PROBLEM_DIR / "task_description.txt").read_text(encoding="utf-8")

    assert 'map_par(mapping, group="<semantic-name>")' in task
    assert "select_parameter_groups" in task
    assert "set_up_new_init" in task
    assert "ExplorationScore" in task
    assert "FinalizationScore" in task
    assert "ActionSelection" in task
    assert "ZBucketSearch" in task
    assert "ab_initio receives no path inventory" in task
    assert "four ab-initio roots" in task
    assert "four mid-margin family" in task
    assert "producer_start_rank" in task
    assert "path_uses" in task
    assert "family_uses" in task
    assert "producer_search" in task
    assert "Pymoo" in task


def test_task_description_distinguishes_raw_reduction_center() -> None:
    task = (PROBLEM_DIR / "task_description.txt").read_text(encoding="utf-8")
    compact = " ".join(task.split())

    assert "`C_RED` is not normalized" in compact
    assert "exact rank-reduction count" in compact
    assert "All other centers are normalized to `[0, 1]`" in compact
    assert "Score weights use the same scale" in compact


def test_each_island_owns_a_nonblank_mutation_overlay() -> None:
    overlays = PROBLEM_DIR / "prompts" / "islands"
    expected = {
        "ab_initio.txt": ('path_name="init"', "standalone TOHPE"),
        "mid_margin.txt": ("Selectable Shared Paths", "INITIAL_RANK"),
        "near_end.txt": ("Selectable Shared Paths", "path_uses"),
    }

    assert {path.name for path in overlays.glob("*.txt")} == set(expected)
    for filename, required in expected.items():
        text = (overlays / filename).read_text(encoding="utf-8").strip()
        compact = " ".join(text.split())
        assert text
        for phrase in required:
            assert phrase in compact


def test_near_end_overlay_explains_plateaus_and_todd_breadth() -> None:
    text = (
        PROBLEM_DIR / "prompts" / "islands" / "near_end.txt"
    ).read_text(encoding="utf-8")

    assert "plateau" in text.lower()
    assert "max_buckets" in text
    assert "limit_bucket" in text
    assert "action starvation" in text.lower()
    assert "family_uses" in text
    assert "successful family refinement" in text.lower()
    assert "additional" in text.lower()
    assert "universally" not in text.lower()


def test_near_end_overlay_explains_inclusive_frontier_rank_window() -> None:
    text = (
        PROBLEM_DIR / "prompts" / "islands" / "near_end.txt"
    ).read_text(encoding="utf-8")

    assert "frontier_rank + 100" in text
    assert "inclusive" in text.lower()


def test_mid_margin_overlay_uses_global_initial_rank_for_margin() -> None:
    text = (
        PROBLEM_DIR / "prompts" / "islands" / "mid_margin.txt"
    ).read_text(encoding="utf-8")
    compact = " ".join(text.split())

    assert "INITIAL_RANK - selected_path_rank" in text
    assert "producer_start_rank" in text
    assert "must not replace `INITIAL_RANK`" in text
    assert "one eighth to one third" in compact
    assert "uses=.../6" in text
    assert "family_uses=.../8" in text


def test_island_prompts_omit_retired_or_unavailable_controls() -> None:
    text = _all_prompt_text().lower()

    assert "tohpeprefix" not in text
    assert "tohpe-prefix" not in text
    assert "bucket temperature" not in text
    assert "random fraction" not in text
    assert "nevergrad" not in text
    assert "pyswarms" not in text
    assert "scipy.optimize" not in text
