"""Unit tests for visual backend import guards and decoupling."""

import sys
from unittest.mock import patch

import pytest

from tropicor.viz._common import require_matplotlib, require_plotly


def test_require_matplotlib_success() -> None:
    """Verify that require_matplotlib returns pyplot module when available."""
    plt = require_matplotlib()
    assert plt is not None
    assert hasattr(plt, "subplots")


def test_require_plotly_success() -> None:
    """Verify that require_plotly returns graph_objects and express."""
    go, px = require_plotly()
    assert go is not None
    assert px is not None
    assert hasattr(go, "Figure")


def test_require_matplotlib_missing_raises_actionable_error() -> None:
    """Verify informative ImportError when matplotlib is missing."""
    err = ImportError("No module named matplotlib")
    with patch.dict(sys.modules, {"matplotlib.pyplot": None}):
        with patch("builtins.__import__", side_effect=err):
            with pytest.raises(ImportError, match="pip install 'tropicor\\[viz\\]'"):
                require_matplotlib()


def test_require_plotly_missing_raises_actionable_error() -> None:
    """Verify informative ImportError when plotly is missing."""
    err = ImportError("No module named plotly")
    mock_dict = {"plotly.express": None, "plotly.graph_objects": None}
    with patch.dict(sys.modules, mock_dict):
        with patch("builtins.__import__", side_effect=err):
            match_msg = "pip install 'tropicor\\[viz-interactive\\]'"
            with pytest.raises(ImportError, match=match_msg):
                require_plotly()


def test_top_level_tropicor_import_does_not_depend_on_viz() -> None:
    """Verify that tropicor base package can be imported independently."""
    import tropicor

    assert hasattr(tropicor, "compute_validation_metrics")
    assert hasattr(tropicor, "taylor_statistics")
    # viz is intentionally NOT exported in top-level __init__.py
    assert not hasattr(tropicor, "plot_station_map")
