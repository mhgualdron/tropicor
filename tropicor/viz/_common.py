"""Common utilities, backend checkers, and type guards for visualization."""

from typing import Any


def require_matplotlib() -> Any:
    """Verify that matplotlib is installed and return the pyplot module.

    Returns:
        The matplotlib.pyplot module.

    Raises:
        ImportError: If matplotlib is not installed, providing instructions
            to install the optional extra tropicor[viz].
    """
    try:
        import matplotlib

        # Ensure headless 'Agg' backend if running non-interactively
        # or in CI/test environments
        if not matplotlib.is_interactive():
            try:
                matplotlib.use("Agg")
            except Exception:
                pass

        import matplotlib.pyplot as plt

        return plt
    except ImportError as err:
        msg = (
            "The 'tropicor.viz' module requires matplotlib for static plotting. "
            "Install it with: pip install 'tropicor[viz]'"
        )
        raise ImportError(msg) from err


def require_plotly() -> Any:
    """Verify that plotly is installed and return graph_objects and express.

    Returns:
        Tuple of (plotly.graph_objects, plotly.express).

    Raises:
        ImportError: If plotly is not installed, providing instructions
            to install the optional extra tropicor[viz-interactive].
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        return go, px
    except ImportError as err:
        msg = (
            "Interactive visualization requires plotly. "
            "Install it with: pip install 'tropicor[viz-interactive]'"
        )
        raise ImportError(msg) from err
