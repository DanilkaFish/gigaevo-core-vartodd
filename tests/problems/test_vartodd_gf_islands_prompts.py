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

    assert "target_island parent" in mutation
    assert "primary parent" in mutation
    assert "bootstrap_donor" in mutation
    assert "mixed_donor" in mutation
    assert "do not change the destination regime" in mutation
    assert "Selectable Shared Paths" in mutation
    assert "only" in mutation.lower()
    assert "historical parent handles" in mutation
    assert "island-local archive-percentile" in mutation
    assert "raw execution evidence" in mutation


def test_task_description_keeps_current_api_and_unified_card_semantics() -> None:
    task = (PROBLEM_DIR / "task_description.txt").read_text(encoding="utf-8")

    assert 'map_par(mapping, group="<semantic-name>")' in task
    assert "select_parameter_groups" in task
    assert "set_up_new_init" in task
    assert "ExplorationScore" in task
    assert "FinalizationScore" in task
    assert "ActionSelection" in task
    assert "ZBucketSearch" in task
    assert "ab_initio receives no path inventory" in task
    assert "same selectable inventory" in task
    assert "producer_search" in task
    assert "Pymoo" in task


def test_each_island_owns_a_nonblank_mutation_overlay() -> None:
    overlays = PROBLEM_DIR / "prompts" / "islands"
    expected = {
        "ab_initio.txt": ('path_name="init"', "standalone TOHPE"),
        "mid_margin.txt": ("Selectable Shared Paths", "margin 30..100"),
        "near_end.txt": ("Selectable Shared Paths", "margin 5..30"),
    }

    assert {path.name for path in overlays.glob("*.txt")} == set(expected)
    for filename, required in expected.items():
        text = (overlays / filename).read_text(encoding="utf-8").strip()
        assert text
        for phrase in required:
            assert phrase in text


def test_near_end_overlay_explains_plateaus_and_todd_breadth() -> None:
    text = (
        PROBLEM_DIR / "prompts" / "islands" / "near_end.txt"
    ).read_text(encoding="utf-8")

    assert "plateau" in text.lower()
    assert "max_buckets" in text
    assert "limit_bucket" in text
    assert "action starvation" in text.lower()
    assert "universally" not in text.lower()


def test_island_prompts_omit_retired_or_unavailable_controls() -> None:
    text = _all_prompt_text().lower()

    assert "tohpeprefix" not in text
    assert "tohpe-prefix" not in text
    assert "bucket temperature" not in text
    assert "random fraction" not in text
    assert "nevergrad" not in text
    assert "pyswarms" not in text
    assert "scipy.optimize" not in text
