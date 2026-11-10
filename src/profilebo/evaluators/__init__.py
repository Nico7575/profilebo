from profilebo.evaluators.base import Evaluation, Evaluator


def make_evaluator(cfg) -> Evaluator:
    """The evaluator a loaded config asks for."""
    if cfg.evaluator == "synthetic":
        from profilebo.evaluators.synthetic import SyntheticEvaluator
        return SyntheticEvaluator(cfg.synthetic)
    if cfg.evaluator == "serpent":
        raise NotImplementedError("the Serpent evaluator is not ported yet (step 5)")
    raise ValueError(f"unknown evaluator {cfg.evaluator!r}")


__all__ = ["Evaluation", "Evaluator", "make_evaluator"]
