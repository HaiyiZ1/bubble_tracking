"""Re-exports from bubble_common.run_config for back-compat.

This module previously held RunConfig, load, and mark_phase_done. They
moved to bubble_common/ in the phase-boundary refactor. Imports here
still work.
"""
from bubble_common.run_config import RunConfig, load, mark_phase_done

__all__ = ["RunConfig", "load", "mark_phase_done"]
