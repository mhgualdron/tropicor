"""Tests for Jupyter notebooks execution and correctness."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_quickstart_notebook_headless_execution() -> None:
    """Validate quickstart notebook execution and expected results."""
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / "tropicor_quickstart.ipynb"

    assert nb_path.exists(), f"Notebook not found at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=60, kernel_name="python3")
    client.execute()

    # Collect text outputs from all code cells
    all_stdout = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if (
                    output.get("output_type") == "stream"
                    and output.get("name") == "stdout"
                ):
                    all_stdout.append(output.get("text", ""))

    combined_text = "\n".join(all_stdout)

    # Verify key assertions and scientific outputs from the notebook
    assert "Loaded 48 Colombian benchmark stations" in combined_text
    assert "Detected Peaks: [4, 11]" in combined_text
    assert "Classified Regime: BIMODAL" in combined_text
    assert "Mean Thermal Dampening" in combined_text
    assert "Rendered national station network with dual insets." in combined_text
    assert "Generated annual climatology profile with regime badge." in combined_text
    assert "Computed Taylor stats:" in combined_text
    assert "Generated diurnal thermal range" in combined_text
