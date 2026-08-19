"""Freeze what the network loader produces, fixture by fixture.

These snapshots exist to be handed to another repository. ADR-013 moves the
topology layer to mikeio1d, and the move is verified rather than trusted: the
loader's output over every fixture is recorded here first, and the same
recordings become the upstream module's acceptance test.

So the cases below are named after the fixture and the load options, never after
the constructor that happens to read them today. Upstream the same six loads go
through one entry point, and the stored files must still match.

Regenerate with::

    pytest tests/test_network_snapshots.py --update-snapshots
"""

# ruff: noqa: E402
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("networkx")

from modelskill.model.adapters._inp import read_pipe_lengths
from modelskill.network import Network

from tests.network_snapshot import describe

requires_mikeio1d = pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)

_TESTDATA = Path(__file__).parent / "testdata"
_SNAPSHOTS = _TESTDATA / "network_snapshots"

_RES1D = str(_TESTDATA / "network.res1d")
_RES11 = str(_TESTDATA / "network_cali.res11")
_EPANET_RES = str(_TESTDATA / "epanet.res")
_EPANET_RESX = str(_TESTDATA / "epanet.resx")
_EPANET_INP = str(_TESTDATA / "epanet.inp")

# Each case is a fixture plus the options it is loaded with. Upstream, the two
# EPANET cases become companions=[...] and companions=[]; the difference in
# spelling is the reason a case is named for its options and not its constructor.
LOADS = {
    "res1d": lambda: Network.from_mike(_RES1D),
    "res1d_nodes_filtered": lambda: Network.from_mike(
        _RES1D, nodes=["108", "101"], reaches=[]
    ),
    "res1d_one_quantity": lambda: Network.from_mike(_RES1D, quantities="Discharge"),
    "res11": lambda: Network.from_mike(_RES11),
    "epanet_with_companions": lambda: Network.from_epanet(
        _EPANET_RES, resx=_EPANET_RESX, inp=_EPANET_INP
    ),
    "epanet_alone": lambda: Network.from_epanet(_EPANET_RES),
}


def _mismatches(actual, expected, where=""):
    """Compare two snapshots, tolerating the last bits of a float.

    Parameters
    ----------
    actual : object
        What the loader produced now.
    expected : object
        What the stored snapshot holds.
    where : str, optional
        Path through the structure, used to name a mismatch.

    Returns
    -------
    list of str
        One line per difference, empty when the two agree. A tolerance is
        applied to floats because these files are meant to travel between
        machines, where a sum over thousands of values need not land on the same
        final digit.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        problems = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                problems.append(f"{where}/{key}: missing, expected {expected[key]!r}")
            elif key not in expected:
                problems.append(f"{where}/{key}: unexpected, got {actual[key]!r}")
            else:
                problems += _mismatches(actual[key], expected[key], f"{where}/{key}")
        return problems

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: length {len(actual)}, expected {len(expected)}"]
        problems = []
        for i, (got, want) in enumerate(zip(actual, expected)):
            problems += _mismatches(got, want, f"{where}[{i}]")
        return problems

    if isinstance(expected, float) or isinstance(actual, float):
        if actual is None or expected is None:
            return (
                [] if actual == expected else [f"{where}: {actual!r} != {expected!r}"]
            )
        if actual == pytest.approx(expected, rel=1e-9, abs=1e-9):
            return []
        return [f"{where}: {actual!r} != {expected!r}"]

    return [] if actual == expected else [f"{where}: {actual!r} != {expected!r}"]


@requires_mikeio1d
@pytest.mark.parametrize("case", sorted(LOADS))
def test_loader_output_is_unchanged(case, update_snapshots):
    snapshot = _SNAPSHOTS / f"{case}.json"
    actual = describe(LOADS[case]())

    if update_snapshots:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(
            json.dumps(actual, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        return

    assert snapshot.exists(), f"No snapshot for '{case}'. Run --update-snapshots."
    expected = json.loads(snapshot.read_text(encoding="utf-8"))

    problems = _mismatches(actual, expected)
    assert not problems, "\n".join([f"'{case}' has changed:", *problems[:40]])


def test_the_inp_reader_ignores_line_endings(tmp_path):
    """A companion parsed the same whichever line ending it arrives with.

    mikeio1d keeps its copy of ``epanet.inp`` with LF endings and modelskill's
    has CRLF, so the snapshots only transfer if the parser cannot tell.
    """
    crlf = Path(_EPANET_INP).read_bytes()
    lf = crlf.replace(b"\r\n", b"\n")
    assert crlf != lf, "fixture is expected to have CRLF endings"

    as_lf = tmp_path / "lf.inp"
    as_lf.write_bytes(lf)

    assert read_pipe_lengths(as_lf) == read_pipe_lengths(_EPANET_INP)
