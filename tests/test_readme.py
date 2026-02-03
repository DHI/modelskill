"""Test that README.md code snippets are executable.

This module extracts Python code blocks from the README and executes them
to verify they work correctly. It substitutes file paths to point to the
test data directory.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


def extract_python_code_blocks(markdown_text: str) -> list[str]:
    """Extract Python code blocks from markdown text.

    Parameters
    ----------
    markdown_text : str
        The markdown content to parse

    Returns
    -------
    list[str]
        List of code strings from ```python blocks
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    return matches


def strip_doctest_prefix(code: str) -> str:
    """Remove >>> and ... prefixes from doctest-style code.

    Parameters
    ----------
    code : str
        Code that may have doctest prefixes

    Returns
    -------
    str
        Clean executable code
    """
    lines = []
    for line in code.split("\n"):
        if line.startswith(">>> "):
            lines.append(line[4:])
        elif line.startswith("... "):
            lines.append(line[4:])
        elif line.startswith(">>>"):
            lines.append(line[3:])
        # Skip output lines (lines that don't start with >>> or ...)
        # but keep empty lines within code
        elif not line.strip() and lines:
            lines.append("")
    return "\n".join(lines)


def substitute_file_paths(code: str, data_dir: str) -> str:
    """Substitute bare file names with full paths to test data.

    Parameters
    ----------
    code : str
        Code containing file references
    data_dir : str
        Path to the test data directory

    Returns
    -------
    str
        Code with substituted paths
    """
    # Files used in README examples
    files_to_substitute = [
        "HKZN_local_2017_DutchCoast.dfsu",
        "HKNA_Hm0.dfs0",
        "eur_Hm0.dfs0",
        "Alti_c2_Dutch.dfs0",
    ]

    for filename in files_to_substitute:
        code = code.replace(f'"{filename}"', f'"{data_dir}/{filename}"')

    return code


def test_readme_code_blocks_execute():
    """Test that all Python code blocks in README.md execute without error."""
    # Locate files
    repo_root = Path(__file__).parent.parent
    readme_path = repo_root / "README.md"
    data_dir = repo_root / "tests" / "testdata" / "SW"

    assert readme_path.exists(), f"README.md not found at {readme_path}"
    assert data_dir.exists(), f"Test data directory not found at {data_dir}"

    # Read and parse README
    readme_content = readme_path.read_text()
    code_blocks = extract_python_code_blocks(readme_content)

    assert len(code_blocks) > 0, "No Python code blocks found in README"

    # Build up namespace across all code blocks (they reference each other)
    namespace: dict = {}

    for i, block in enumerate(code_blocks):
        # Clean up the code
        code = strip_doctest_prefix(block)
        code = substitute_file_paths(code, str(data_dir))

        # Skip empty blocks
        if not code.strip():
            continue

        try:
            exec(code, namespace)
        except Exception as e:
            # Include context in error message
            raise AssertionError(
                f"README code block {i + 1} failed to execute:\n"
                f"---CODE---\n{code}\n---END---\n"
                f"Error: {type(e).__name__}: {e}"
            ) from e

        # Close any matplotlib figures to avoid memory issues
        plt.close("all")


def test_readme_skill_output_structure():
    """Test that the skill() output has the expected structure."""
    import modelskill as ms

    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "tests" / "testdata" / "SW"

    # Replicate README example
    mr = ms.model_result(
        str(data_dir / "HKZN_local_2017_DutchCoast.dfsu"),
        name="HKZN_local",
        item=0,
    )
    HKNA = ms.PointObservation(
        str(data_dir / "HKNA_Hm0.dfs0"),
        item=0,
        x=4.2420,
        y=52.6887,
        name="HKNA",
    )
    EPL = ms.PointObservation(
        str(data_dir / "eur_Hm0.dfs0"),
        item=0,
        x=3.2760,
        y=51.9990,
        name="EPL",
    )
    c2 = ms.TrackObservation(
        str(data_dir / "Alti_c2_Dutch.dfs0"),
        item=3,
        name="c2",
    )

    cc = ms.match([HKNA, EPL, c2], mr)
    skill = cc.skill()

    # Verify structure matches what README shows
    assert "HKNA" in skill.index
    assert "EPL" in skill.index
    assert "c2" in skill.index

    # Verify expected columns exist
    expected_columns = ["n", "bias", "rmse", "urmse", "mae", "cc", "si", "r2"]
    for col in expected_columns:
        assert col in skill.columns, f"Missing column: {col}"

    # Verify data is reasonable (not NaN, within expected ranges)
    # Access underlying DataFrame for numeric checks
    df = skill.to_dataframe()
    assert all(df["n"] > 0), "No data points"
    assert all(df["r2"] > 0), "r2 should be positive for this dataset"
    assert all(df["rmse"] > 0), "rmse should be positive"
