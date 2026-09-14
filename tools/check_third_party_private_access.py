"""Report access to private attributes on objects from other packages.

ModelSkill reaches into its own private attributes freely -- that is ordinary
intra-package access. Reaching into a *third-party* object's private attribute
(for example ``mikeio_dataset._zn``) is different: nothing stops the other
package from renaming or removing it in a patch release, and nothing in CI
would notice until a user hits it.

Ruff's ``SLF001`` finds every private attribute access but cannot tell whose
object it is, because it does not infer types. This script narrows ``SLF001``
by name: a member that modelskill defines somewhere in ``src/modelskill`` is
assumed to be ours; anything else is assumed to belong to another package.

That is a heuristic, not type inference. It misses an access whose attribute
name we happen to use ourselves, and it can misreport an attribute we only ever
read (never assign) as third-party. It is meant to make these accesses visible,
not to prove their absence.

Run with ``just private-access``. Exits non-zero if anything is reported.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "modelskill"


def own_names(src: Path) -> set[str]:
    """Collect every attribute or name modelskill itself defines.

    Parameters
    ----------
    src : Path
        Root of the modelskill package.

    Returns
    -------
    set of str
        Names bound anywhere in the package: functions, classes, assigned
        attributes, class-body and annotated assignments, and arguments.
    """
    names: set[str] = set()
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
                continue

            if isinstance(node, ast.Assign):
                targets: list[ast.expr] = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Attribute):
                    names.add(target.attr)
    return names


def private_accesses(src: Path) -> list[dict]:
    """Run ruff's SLF001 over the package and return its findings as JSON."""
    result = subprocess.run(
        [
            "ruff",
            "check",
            "--no-cache",
            "--isolated",
            "--select",
            "SLF001",
            "--output-format",
            "json",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        sys.exit(f"ruff failed:\n{result.stderr}")
    return json.loads(result.stdout or "[]")


def main() -> int:
    ours = own_names(SRC)
    root = SRC.parent.parent

    reported = []
    for hit in private_accesses(SRC):
        member = hit["message"].split("`")[1]
        if member in ours:
            continue
        path = Path(hit["filename"])
        if path.is_relative_to(root):
            path = path.relative_to(root)
        reported.append((path, hit["location"]["row"], member))

    if not reported:
        print("No private attribute access on third-party objects.")
        return 0

    print("Private attribute access on objects from other packages:\n")
    for path, row, member in reported:
        print(f"  {path}:{row}: {member}")
    print(
        "\nThese attributes are not part of any public API and can disappear "
        "without a deprecation. Ask the upstream package for a public accessor, "
        "or add the name to the docstring's list of known exceptions."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
