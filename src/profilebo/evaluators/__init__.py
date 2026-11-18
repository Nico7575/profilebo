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


# Axis labels for the report, per evaluator: (t, y, J). The optimizer itself
# never needs units.
AXIS_LABELS = {
    "serpent": ("burnup time (days)", "reactivity (pcm)", "J (pcm$^2$)"),
    "synthetic": ("t (days)", "y (pcm)", "J (pcm$^2$)"),   # mimics a burnup curve
}


__all__ = ["AXIS_LABELS", "Evaluation", "Evaluator", "make_evaluator"]
