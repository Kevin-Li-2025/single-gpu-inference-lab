"""Epilogue-boundary analysis helpers for L20 serving experiments."""

from l20_stack.epilogue.compare import BoundaryImpact, build_boundary_impacts
from l20_stack.epilogue.logits_boundary import (
    LogitsBoundaryBudget,
    load_logits_boundary_budget,
)
from l20_stack.epilogue.intervention import (
    CONTINUE_EPILOGUE_PROTOTYPE,
    DO_NOT_CLAIM_WIN,
    NEEDS_MORE_RUNS,
    render_logits_boundary_ab_markdown,
    summarize_logits_boundary_ab,
)
from l20_stack.epilogue.sampler_epilogue import SamplerConfig, sampler_gate_reasons

__all__ = [
    "BoundaryImpact",
    "CONTINUE_EPILOGUE_PROTOTYPE",
    "DO_NOT_CLAIM_WIN",
    "LogitsBoundaryBudget",
    "NEEDS_MORE_RUNS",
    "SamplerConfig",
    "build_boundary_impacts",
    "load_logits_boundary_budget",
    "render_logits_boundary_ab_markdown",
    "sampler_gate_reasons",
    "summarize_logits_boundary_ab",
]
