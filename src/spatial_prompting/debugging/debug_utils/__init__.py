"""
Shared utilities for TextBraTS spatial prompting debugging tools.

This package provides common functionality used across all debugging subcommands:
- atlas_ops: Atlas mask operations and coverage analysis
- gt_processing: Ground truth loading and TC/WT/ET conversion
- overlap_analysis: GT-atlas overlap calculations
- checkpoint_utils: Model checkpoint analysis
- visualization: PDF generation and plotting
"""

from . import atlas_ops
from . import gt_processing
from . import overlap_analysis
from . import checkpoint_utils
from . import visualization

__all__ = [
    'atlas_ops',
    'gt_processing',
    'overlap_analysis',
    'checkpoint_utils',
    'visualization',
]
