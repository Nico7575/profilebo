from profilebo.evaluators.base import Evaluation, Evaluator


def make_evaluator(cfg) -> Evaluator:
    """The evaluator a loaded config asks for."""
    if cfg.evaluator == "synthetic":
        from profilebo.evaluators.synthetic import SyntheticEvaluator
        return SyntheticEvaluator(cfg.synthetic)
    if cfg.evaluator == "serpent":
        from profilebo.evaluators.serpent.evaluator import SerpentEvaluator
        return SerpentEvaluator(cfg.serpent)
    raise ValueError(f"unknown evaluator {cfg.evaluator!r}")


__all__ = ["Evaluation", "Evaluator", "make_evaluator"]
