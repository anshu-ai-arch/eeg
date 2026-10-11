"""
scripts/classification

Classification and feature selection package for Bonn EEG reproduction.
Includes:
- MCFS: Supervised Multi-Cluster Feature Selection (Cai et al., 2010; Lashkari et al., 2026).
"""

from .mcfs import MCFS
from .nested_cv import NestedCVConfig, run_nested_cross_validation, compute_metrics

__all__ = ["MCFS", "NestedCVConfig", "run_nested_cross_validation", "compute_metrics"]
