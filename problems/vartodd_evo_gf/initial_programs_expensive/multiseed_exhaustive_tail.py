"""Single-seed DE scout with two-seed high-coverage terminal search."""

from collections.abc import Iterable

from helper import (
    INITIAL_RANK,
    TARGET_FINAL_RANK,
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
    policy,
)
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


OPTIMIZER_FAMILY = "pymoo_post_restart_multiseed_exhaustive"
SCOUT_SEEDS = (51,)
TAIL_SEEDS = (51, 52)
SCOUT_EVALS = 80
TAIL_EVALS = 70
TARGET_TOTAL_EVALS = 220
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MARGIN = max(1, round(0.44 * RANK_SPAN))
REOPEN_MARGIN = max(8, round(0.10 * RANK_SPAN))
MAX_DEPTH = max(500, RANK_SPAN + 64)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    blend = fn.clip((0.5 - k.nrank_red) * 4.0, 0.0, 1.0)
    tight = k.nred * p.w(0) + k.ntohpe * p.w(1) - k.nrank_dim * fn.abs(p.w(2))
    spread = k.nred * p.w(3) + (k.red / fn.max(k.bucket, 1.0)) * p.w(4) + k.ndim * p.w(5)
    return blend * tight + (1.0 - blend) * spread + k.f_todd * p.w(6)


def rank_from_target(fraction: float) -> int:
    return min(INITIAL_RANK, TARGET_FINAL_RANK + max(1, round(fraction * RANK_SPAN)))


def rank_margin(fraction: float) -> int:
    return max(1, round(fraction * RANK_SPAN))


class Evaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        self.tail_mode = False
        self.active_seeds = SCOUT_SEEDS
        super().__init__(*args, **kwargs)

    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(
            lambda x: low + (high - low) / (1.0 + np.exp(-float(x))), group=group
        )

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) / (1.0 + np.exp(-float(x))))),
            group=group,
        )

    def tohpe_filter(self) -> tuple[int, int]:
        return TOHPE_FILTERS[self.int_range(0, len(TOHPE_FILTERS) - 1, group="tohpe_filter")]

    def policy_mapping(self):
        target_min_red, target_max_red = self.tohpe_filter()
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([self.float_range(-4.0, 4.0, group="scores") for _ in range(explore_score.n_params)]),
                final=final_score.bind([self.float_range(-4.0, 4.0, group="scores") for _ in range(final_score.n_params)]),
            )
        )
        scout = self._scout_policy()
        tail = self._tail_policy()
        selection, pool, tohpe, todd = tail if self.tail_mode else scout
        self.set_action_selection(selection)
        self.set_action_pool(pool)
        self.set_tohpe_search(tohpe)
        self.set_todd_search(todd)

    def _scout_policy(self):
        return (
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.28),
            ActionPool(final_size=self.int_range(12, 28, group="scout")),
            TohpeSearch(
                SamplingBudget(one_hot="all", sparse=0, dense=8, sparse_max_weight=2),
                SourcePool(
                    keep=self.int_range(9, 26, group="scout"),
                    reserve=self.int_range(1, 5, group="scout"),
                ),
                z_choices=self.int_range(4, 15, group="scout"),
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            ),
            ToddSearch(
                SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
                SourcePool(keep=0, reserve=0),
                actions_per_bucket=0,
                buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
            ),
        )

    def _tail_policy(self):
        cap = max(512, round(self.buckets_space * 0.22))
        return (
            ActionSelection(
                beamwidth=self.int_range(3, 4, group="tail"),
                mode="softmax",
                temperature=0.13,
            ),
            ActionPool(final_size=self.int_range(28, 52, group="tail")),
            TohpeSearch(
                SamplingBudget(one_hot="all", sparse=8, dense=12, sparse_max_weight=3),
                SourcePool(keep=10, reserve=2),
                z_choices=6,
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            ),
            ToddSearch(
                SamplingBudget(one_hot="all", sparse=12, dense=12, sparse_max_weight=3),
                SourcePool(
                    keep=self.int_range(5, 10, group="tail"),
                    reserve=self.int_range(1, 3, group="tail"),
                ),
                actions_per_bucket=self.int_range(3, 4, group="tail"),
                buckets=ZBucketSearch(
                    min_buckets=max(160, cap // 24),
                    max_buckets=cap,
                    limit_bucket=cap,
                ),
            ),
        )

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, self.active_seeds), dtype=float)
        return float(values.min() + 0.02 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        size = len(evaluator.extract_active())
        super().__init__(n_var=size, n_obj=1, xl=np.full(size, -1.0), xu=np.full(size, 1.0))
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize(evaluator: Evaluator, algorithm, evaluations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    result = minimize(
        Problem(evaluator), algorithm, termination=("n_eval", evaluations), seed=seed, verbose=False
    )
    return np.asarray(result.X if result.X is not None else current, dtype=float)


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    evaluator.select_parameter_groups("scores", "scout")
    params = optimize(
        evaluator,
        DE(pop_size=8, variant="DE/rand/1/bin", CR=0.88, F=0.65),
        SCOUT_EVALS,
        seed=51,
    )
    evaluator.tail_mode = True
    restarted = evaluator.set_up_new_init(
        0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=params
    )
    if restarted is not None:
        evaluator.active_seeds = TAIL_SEEDS
        evaluator.select_parameter_groups("scores", "tail")
        optimize(
            evaluator,
            PatternSearch(x0=np.clip(evaluator.extract_active(), -1.0, 1.0), init_delta=0.2),
            TAIL_EVALS,
            seed=53,
        )
    return evaluator.get_best()


AUX_DESCRIPTION = """
Pool role: the expensive robustness-and-coverage specialist. It postpones both
the second seed and high-z TODD until a strong path has been reopened, then
uses a wider beam to supply tree search with more rare terminal actions.
"""
