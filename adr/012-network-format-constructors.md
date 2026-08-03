# ADR-012: One Network Constructor per Modelling Product

**Status**: Draft

**Date**: 2026-08

## Context

`Network` is built from result files read through mikeio1d. Its single `Res1D` class opens nine extensions across five products — MIKE 1D (`.res1d`), MIKE 11 (`.res11`), MOUSE (`.prf`, `.crf`, `.xrf`), EPANET (`.res`), SWMM (`.out`), Water Hammer (`.whr`), and `.resx`, which is shared by the last three. There is no per-format reader and no per-format constructor argument, so from mikeio1d's side all nine look alike.

modelskill's constructor was named `from_res1d`, and its extension guard was briefly widened to accept everything mikeio1d could read. That made the name misleading: it promised one format and read nine.

Loading each of mikeio1d's own fixtures showed the nine are not interchangeable:

- `.res1d` and `.res11` produce a full network with real reach lengths and gridpoints. `.res11` initially failed because MIKE 11 keeps its timeseries on reach gridpoints, leaving nodes with no quantities at all — a bug in modelskill's adapter, now fixed.
- `.res` (EPANET) loads, but as a link-node model it reports no reach length and one synthetic gridpoint per reach. mikeio1d signals the missing length by returning `0`, which is indistinguishable from a genuine zero.
- `.out` (SWMM) and `.resx` expose no reach start/end nodes. There is no topology to rebuild. mikeio1d's own SWMM tests never touch reach connectivity.
- MOUSE and `.whr` have no test fixture anywhere, including in mikeio1d's testdata, so nothing about them can be verified.

## Decision

Name constructors after the product that writes the file, and only ship one where a committed fixture backs it:

| Constructor | Extensions |
|---|---|
| `Network.from_mike` | `.res1d`, `.res11` |
| `Network.from_epanet` | `.res` |

`from_res1d` is removed. It only ever shipped in the 1.4.0a3 alpha, and the network module is opt-in and absent from the API reference, so a deprecation shim would have added a second name for the tested path without protecting a real caller.

Every extension mikeio1d can read is accounted for in one of three module-level tables in `network.py` — readable by `from_mike`, readable by `from_epanet`, or refused with a specific reason. A test asserts the three cover exactly `Res1D.get_supported_file_extensions()`, so a mikeio1d release that adds a tenth format fails CI and forces a decision rather than leaving the format silently unreachable.

Two supporting rules:

- **A constructor requires a fixture.** Naming a product in the API is a support claim; it should be backed by a test that builds a `Network` from a real file of that product. MOUSE and Water Hammer are refused today for exactly this reason, and each becomes a six-line addition once a redistributable fixture exists.
- **Degenerate results are documented, not warned about.** EPANET's undefined reach lengths and absent breakpoints are stated in the `from_epanet` docstring and the user guide, and asserted in tests. A runtime warning would fire on correct usage and teach users to filter our warnings, and both consequences already raise where they bite.
- **An unreadable reach length is undefined, not zero.** `NetworkReach.length` is optional and defaults to `None`, and the adapter maps mikeio1d's `0` sentinel onto it. Reporting `0` would assert that an EPANET pipe has no extent, which is false — the length exists, mikeio1d just cannot read it — and it makes a length-weighted graph algorithm treat the reach as free to traverse. With `None`, `networkx` fails instead: shortest-path treats the edge as unreachable and weight-summing calls raise `TypeError`. Omitting the edge attribute altogether was rejected for the opposite reason, since `networkx` then defaults the weight to `1`. Nothing inside modelskill reads the length, so this only affects `Network.graph`; matching and extraction work from break point distances.

## Alternatives Considered

**One constructor per extension** — `from_res`, `from_out`, `from_whr` say nothing about the product they belong to, and MOUSE would need three identical methods.

**A generic catch-all (`from_file` or `from_mikeio1d`)** — a second way to do the same thing. With every extension either read or explicitly refused, the catch-all's only remaining job is forward compatibility with formats mikeio1d adds later, which the drift test handles more usefully by demanding a decision.

**Keep `from_res1d` permissive** — preserves the misleading name, and a constructor that accepts everything cannot tell an EPANET user which method to reach for instead.

**Ship all five product constructors regardless of coverage** — three of the five would either always fail (SWMM) or be unverifiable, so the method list would stop being a reliable statement of what works.

## Consequences

Positive:

- The method list is the format list; `Network.from_<TAB>` answers "which formats does this read".
- Refusals name the cause, so the SWMM and `.resx` gaps read as upstream limitations rather than modelskill bugs.
- Passing a file the other constructor handles raises a `ValueError` naming that constructor.
- One private implementation (`Network._from_mikeio1d`) does the version guard, extension validation, `Res1D` construction and node/reach filtering, so a new product constructor is a docstring and one call.

Negative:

- MIKE 11 is covered by a fixture but has no field-tested usage behind it yet.
- MOUSE and Water Hammer are refused even though mikeio1d may well read them correctly. This is deliberate: refusing with a reason is recoverable, while a method that silently produces a wrong graph is not.

## Relationship to ADR-009

[ADR-009](009-factory-pattern.md) argues for auto-detecting entry points such as `model_result()` and `observation()` so users need not know the class hierarchy. That is not in tension with this decision. Auto-detection resolves *which class* to build from the shape of the data; these constructors resolve *which product wrote the file*, which is information the call site should state rather than have guessed — particularly when the answer decides whether reach-based matching will work at all.

## See Also

- [ADR-010](010-optional-domain-dependencies.md) — why mikeio1d is an optional dependency
- `tests/testdata/README.md` — provenance of the result fixtures
