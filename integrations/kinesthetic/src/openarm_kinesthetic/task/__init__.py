"""Shared task contract used by collection, quality gates, and replay."""

from .evaluator import EvaluationSample, TaskEvaluator, TaskResult

__all__ = ["EvaluationSample", "TaskEvaluator", "TaskResult"]
