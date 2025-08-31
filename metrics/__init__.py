"""
Audio quality evaluation metrics module.
"""

from .base import (
    BaseMetricCalculator,
    ModelConfig,
    MetricCalculationError,
    ModelLoadError
)
from .utmos import UTMOSCalculator
from .wer import WERCalculator
from .sim import SIMCalculator
from .ffe import FFECalculator
from .mcd import MCDCalculator

__all__ = [
    # Base classes and exceptions
    'BaseMetricCalculator',
    'ModelConfig',
    'MetricCalculationError',
    'ModelLoadError',

    # Metric calculators
    'UTMOSCalculator',
    'WERCalculator',
    'SIMCalculator',
    'FFECalculator',
    'MCDCalculator'
]

# Version information
__version__ = '1.0.0'

# Metric registry for easy access
METRIC_CALCULATORS = {
    'utmos': UTMOSCalculator,
    'wer': WERCalculator,
    'sim': SIMCalculator,
    'ffe': FFECalculator,
    'mcd': MCDCalculator
}


def create_calculator(metric_name: str, config: ModelConfig) -> BaseMetricCalculator:
    """
    Factory function to create metric calculators.

    Args:
        metric_name: Name of the metric calculator
        config: Model configuration

    Returns:
        Metric calculator instance

    Raises:
        ValueError: If metric name is not supported
    """
    if metric_name.lower() not in METRIC_CALCULATORS:
        available_metrics = ', '.join(METRIC_CALCULATORS.keys())
        raise ValueError(f"Unsupported metric: {metric_name}. Available metrics: {available_metrics}")

    calculator_class = METRIC_CALCULATORS[metric_name.lower()]
    return calculator_class(config)


def get_available_metrics() -> list[str]:
    """Get list of available metric names."""
    return list(METRIC_CALCULATORS.keys())