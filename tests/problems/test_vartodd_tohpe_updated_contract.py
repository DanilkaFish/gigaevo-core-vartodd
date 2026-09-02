from pathlib import Path
import re
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf"


def test_runtime_and_z_research_contract() -> None:
    metrics = (PROBLEM_DIR / "metrics_template.yaml").read_text(encoding="utf-8")
    base_algorithm = (
        REPO_ROOT / "config" / "algorithm" / "vartodd_diverse_gf16.yaml"
    ).read_text(encoding="utf-8")
    updated_algorithm = (
        REPO_ROOT
        / "config"
        / "algorithm"
        / "vartodd_diverse_gf16_tohpe_updated.yaml"
    ).read_text(encoding="utf-8")
    mutation_prompt = (
        PROBLEM_DIR / "prompts" / "mutation" / "system.txt"
    ).read_text(encoding="utf-8")
    insights_prompt = (
        PROBLEM_DIR / "prompts" / "insights" / "system.txt"
    ).read_text(encoding="utf-8")

    runtime_block = metrics.split("  runtime:", 1)[1].split("  is_valid:", 1)[0]
    assert "higher_is_better: true" in runtime_block
    assert "low finite limits get a small fitness reward" not in base_algorithm
    assert "low finite limits get a small fitness reward" not in updated_algorithm
    assert "Treat faster runtime as useful only" not in mutation_prompt
    assert "productive runtime utilization" in mutation_prompt
    assert "productive runtime utilization" in insights_prompt


def test_policy_aux_omits_tohpeprefix_and_uses_todd_z_only() -> None:
    import importlib.util
    import sys

    problem_dir = REPO_ROOT / "problems" / "vartodd_evo_gf"
    spec = importlib.util.spec_from_file_location(
        "_test_vartodd_evo_gf_mcts_dao",
        problem_dir / "mcts_dao.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(problem_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(problem_dir))

    group = {
        "start_rank": 420,
        "end_rank": 410,
        "steps": 1,
        "profile_id": "P1",
        "split_reasons": [],
        "red": [10.0],
        "red_max": [10.0],
        "basis_dim": [2.0],
        "accepted_tohpe": [3.0],
        "accepted_tohpeprefix": [99.0],
        "accepted_todd": [2.0],
        "researched_z": [10_512.0],
        "researched_z_tohpeprefix": [10_000.0],
        "researched_z_todd": [512.0],
        "pool_size": [104.0],
        "pool_tohpe_size": [3.0],
        "pool_tohpeprefix_size": [99.0],
        "pool_todd_size": [2.0],
        "missing_stats": 0,
    }
    band_text = module.Path._format_band([group], 1)
    snapshot = (
        ("beamwidth", "3"),
        ("tohpe_pool", "8/2"),
        ("tohpeprefix_pool", "99/9"),
        ("todd_pool", "6/2"),
        ("tohpe_sampling", "oh:8/sp:2/de:0/w:2"),
        ("tohpeprefix_sampling", "oh:99/sp:9/de:9/w:9"),
        ("todd_sampling", "oh:2/sp:0/de:0/w:1"),
        ("tohpeprefix_actions_per_bucket", "99"),
        ("todd_actions_per_bucket", "4"),
        ("tohpeprefix_z_min_buckets", "99"),
        ("tohpeprefix_z_max_buckets", "999"),
        ("tohpeprefix_z_limit_bucket", "9999"),
        ("todd_z_min_buckets", "64"),
        ("todd_z_max_buckets", "256"),
        ("todd_z_limit_bucket", "1024"),
    )
    profile_text = module.Path._format_policy_profile(
        "P1",
        snapshot,
        [group],
        [group],
        include_scores=False,
    )

    assert "tohpeprefix" not in band_text.lower()
    assert "src=H:3/T:2" in band_text
    assert "pool:5(H3/T2)" in band_text
    assert "z:512" in band_text
    assert "10000" not in band_text
    assert "tohpeprefix" not in profile_text.lower()
    assert "pool=final:16/tohpe:8/2/todd:6/2" in profile_text
    assert "z_buckets=todd:64..256/1024" in profile_text


def test_policy_aux_uses_native_aggregate_z_when_todd_is_the_only_z_source() -> None:
    import importlib.util
    import sys

    problem_dir = REPO_ROOT / "problems" / "vartodd_evo_gf"
    spec = importlib.util.spec_from_file_location(
        "_test_vartodd_evo_gf_mcts_dao_native_stats",
        problem_dir / "mcts_dao.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(problem_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(problem_dir))

    root = SimpleNamespace(
        state=SimpleNamespace(rows=420),
        parent=None,
        incoming=None,
    )
    child = SimpleNamespace(
        state=SimpleNamespace(rows=419),
        parent=root,
        incoming=SimpleNamespace(
            # ActionInfo.total is not the z-bucket count.
            total=10_000,
            cand=SimpleNamespace(
                reduction=1,
                basis_dim=2,
                bucket_size=1,
                pool_tohpe_size=0,
                pool_tohpeprefix_size=0,
                pool_todd_size=1,
            ),
            # This mirrors the native Stats API: only aggregate z_researched
            # exists, while the default TOHPE-prefix source is disabled.
            global_info=SimpleNamespace(
                max_reduction=1,
                max_basis=2,
                accepted_tohpe=0,
                accepted_tohpeprefix=0,
                accepted_todd=1,
                z_researched=64,
            ),
        ),
    )
    path = module.Path(final_node=child, daos=[module.Dao()])

    text = path.format_path_stats()

    assert "src=H:0/T:1" in text
    assert "z:64" in text
    assert "z:0" not in text


def test_updated_algorithm_separates_light_builders_from_heavy_tail_refiners() -> None:
    updated_algorithm = (
        REPO_ROOT
        / "config"
        / "algorithm"
        / "vartodd_diverse_gf16_tohpe_updated.yaml"
    ).read_text(encoding="utf-8")

    probabilities = re.findall(
        r"^      probability: ([0-9.]+)$", updated_algorithm, re.M
    )
    assert probabilities == ["0.40", "0.35", "0.25"]
    assert updated_algorithm.count("## Required Island Regime") == 3
    assert "island_id: ab_initio" in updated_algorithm
    assert "island_id: mid_margin" in updated_algorithm
    assert "island_id: near_end" in updated_algorithm

    ab_initio, remainder = updated_algorithm.split("regime_id: mid_margin", 1)
    mid_margin, near_tail = remainder.split("regime_id: near_end", 1)
    assert 'path_name="init"' in ab_initio
    assert "TOHPE primary" in ab_initio
    assert "TODD disabled or light" in ab_initio
    assert "margin 30..100" in mid_margin
    assert "margin 5..30" in near_tail
    assert "terminal TODD" in near_tail
    assert "rare and diverse actions" in re.sub(r"\s+", " ", near_tail)
    assert "TOHPEprefix" not in updated_algorithm
