"""Top-level pipeline orchestration: wires the per-stage modules (detection,
filtering, region assignment, reporting, roi_extract, redraw) into the two
end-to-end pipelines (registered / not-registered) and a CLI dispatcher.
"""
from .run import run_pipeline
from .run_with_registration import run_with_registration
from .run_without_registration import run_without_registration

__all__ = ["run_pipeline", "run_with_registration", "run_without_registration"]
