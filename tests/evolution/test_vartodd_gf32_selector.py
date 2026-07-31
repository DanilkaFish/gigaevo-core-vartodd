"""Retention rules for the GF32 VarTODD MAP-Elites archive."""

from custom.archive_selectors import RankAwareRetentionSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _program(*, fitness: float, loaded_rank: float, rank_improved: float) -> Program:
    program = Program(code="def entrypoint(): pass", state=ProgramState.DONE)
    program.add_metrics(
        {
            "fitness": fitness,
            "loaded_rank": loaded_rank,
            "rank_improved": rank_improved,
        }
    )
    return program


def _selector() -> RankAwareRetentionSelector:
    return RankAwareRetentionSelector(
        fitness_keys=["fitness"],
        fitness_key_higher_is_better=[False],
    )


def test_improving_program_beats_nonimproving_program() -> None:
    selector = _selector()
    new = _program(fitness=1210.0, loaded_rank=1240.0, rank_improved=1.0)
    current = _program(fitness=1200.0, loaded_rank=1400.0, rank_improved=0.0)

    assert selector(new, current) is True


def test_nonimproving_program_cannot_replace_improving_program() -> None:
    selector = _selector()
    new = _program(fitness=1190.0, loaded_rank=1701.0, rank_improved=0.0)
    current = _program(fitness=1210.0, loaded_rank=1240.0, rank_improved=1.0)

    assert selector(new, current) is False


def test_higher_loaded_rank_wins_between_equally_improving_programs() -> None:
    selector = _selector()
    new = _program(fitness=1210.0, loaded_rank=1400.0, rank_improved=1.0)
    current = _program(fitness=1200.0, loaded_rank=1240.0, rank_improved=1.0)

    assert selector(new, current) is True


def test_fitness_breaks_an_exact_rank_tie() -> None:
    selector = _selector()
    new = _program(fitness=1200.0, loaded_rank=1300.0, rank_improved=1.0)
    current = _program(fitness=1210.0, loaded_rank=1300.0, rank_improved=1.0)

    assert selector(new, current) is True
