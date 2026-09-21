"""Stage implementations, and the registry the driver builds from.

The registry is deliberately explicit rather than discovered by import scanning.
A stage that silently fails to register would show up as "nothing to do", which
is indistinguishable from a finished campaign -- and that is the one report a
driver must never get wrong.
"""

from __future__ import annotations

from pathlib import Path

from ..config.loader import ResolvedConfig
from .base import Stage, StageReport, WorkItem
from .analyze_stage import AnalyzeStage
# CalibrateStage is no longer part of the funnel (D126).  The module is kept
# because it records the measurement that retired it, and because
# `calibrate:` blocks in existing campaign.yaml files must still parse.
from .calibrate_stage import CalibrateStage  # noqa: F401
from .dedup_stage import DedupStage
from .dft_stage import DftStage
from .filter_stage import FilterStage
from .generate_stage import GenerateStage
from .reference_stage import ReferenceStage
from .screen_stage import ScreenStage
from .source_stage import SourceStage

__all__ = ["Stage", "StageReport", "WorkItem", "AnalyzeStage", "CalibrateStage", "DedupStage", "DftStage", "FilterStage", "GenerateStage", "ReferenceStage", "ScreenStage", "SourceStage",
           "build_registry", "IMPLEMENTED", "PLANNED"]

# What exists today, in funnel order.
IMPLEMENTED = ["source", "generate", "screen", "dedup", "reference",
               "filter", "dft", "analyze"]

# What does not, and which milestone brings it. Named here so `csp run` can say
# "screen arrives in M1" rather than "unknown stage". Empty now that every stage
# of the funnel exists; the registry test asserts IMPLEMENTED | PLANNED covers
# STAGE_ORDER, so a stage cannot be dropped from the funnel without failing.
PLANNED: dict[str, str] = {}


def build_registry(cfg: ResolvedConfig, base_dir: Path | None = None) -> list[Stage]:
    """Every stage that has an implementation, given this configuration."""
    stages: list[Stage] = [SourceStage(cfg, base_dir)]
    if cfg.campaign.generate is not None:
        stages.append(GenerateStage(cfg))
    return stages + [ScreenStage(cfg), DedupStage(cfg),
            ReferenceStage(cfg),
            FilterStage(cfg), DftStage(cfg), AnalyzeStage(cfg)]
