"""Utility functions for the predictive maintenance system."""

import logging
from typing import Any, Dict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded JSON data as dictionary
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        raise


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Path to save JSON file
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        raise


def format_seconds(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def calculate_cost_benefit(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    true_negatives: int,
    maintenance_cost: float = 10000,
    failure_cost: float = 100000,
) -> Dict[str, float]:
    """
    Calculate cost-benefit analysis for predictive maintenance.

    Args:
        true_positives: Number of correctly predicted failures
        false_positives: Number of false alarms
        false_negatives: Number of missed failures
        true_negatives: Number of correctly predicted non-failures
        maintenance_cost: Cost of scheduled maintenance
        failure_cost: Cost of unexpected failure

    Returns:
        Dictionary with cost-benefit metrics
    """
    # Costs
    scheduled_maintenance_cost = (true_positives + false_positives) * maintenance_cost
    unexpected_failure_cost = false_negatives * failure_cost

    total_cost = scheduled_maintenance_cost + unexpected_failure_cost

    # Benefits (avoided costs)
    avoided_failure_cost = true_positives * (failure_cost - maintenance_cost)

    # Net benefit
    net_benefit = avoided_failure_cost - (false_positives * maintenance_cost)

    # ROI
    if scheduled_maintenance_cost > 0:
        roi = (net_benefit / scheduled_maintenance_cost) * 100
    else:
        roi = 0

    return {
        "scheduled_maintenance_cost": scheduled_maintenance_cost,
        "unexpected_failure_cost": unexpected_failure_cost,
        "total_cost": total_cost,
        "avoided_failure_cost": avoided_failure_cost,
        "net_benefit": net_benefit,
        "roi_percent": roi,
    }
