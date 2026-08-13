# ADR-013: The Network Topology Layer Belongs to mikeio1d

**Status**: Draft

**Date**: 2026-08

## Context

`modelskill.network` grew into a topology layer of its own: the abstract node/reach/breakpoint types, a `Res1D` adapter, one constructor per modelling product, the `.resx` and `.inp` companions, a table of which extensions we refuse and why, a networkx graph carrying reach lengths and boundary edges, an alias map, and `find`/`recall`/`to_dataset` on top. Roughly 630 code lines, against 25 for mikeio1d's `experimental.to_networkx`, which converts the same file and ignores gridpoints.

Most of that difference is work the upstream function declines to do rather than work done twice. But the layer sits on the wrong side of a line ADR-001 drew for mikeio: we call `mikeio.read()` and stop, without modelling dfsu geometry or policing its format list. Here we do both. `Res1D` reads nine extensions across five products; our tables decide which of the nine we accept, and a test fails our CI when a mikeio1d release adds a tenth. The fixtures those tables are checked against — `network.res1d`, `network_cali.res11`, `epanet.res/.resx/.inp`, `network_chinese.res1d` — are copies of mikeio1d's own. Meanwhile `NetworkModelResult` uses four things from `Network` and never traverses the graph.

## Decision

mikeio1d gains an optional network module that builds and owns `Network`. modelskill requires it and consumes what it produces.

| Owner | Pieces |
|---|---|
| mikeio1d | abstract types and `BasicNode`/`BasicReach`, the `Res1D` adapter, `from_mike`/`from_epanet`, the `.resx` and `.inp` companions, the extension policy tables, graph construction with its length and boundary semantics, the alias map, `find`, `recall`, `to_dataframe`, `to_dataset` |
| modelskill | `NetworkModelResult`, `NodeModelResult`, `NodeObservation`, `ReachObservation`, matching, the MIKE+ station resolver |

`NetworkModelResult` takes a `Network` the upstream module built, or a path it hands to that module. The module is an extra there, carrying networkx and xarray, so `to_dataset()` travels with the class rather than leaving a stub behind. modelskill's `networks` extra requires a mikeio1d release new enough to contain it.

Original IDs become the only identifier a user handles: `NodeObservation.at` takes a node name or a `(reach, distance)` pair, and no longer an integer. The alias integers stay an internal index — they exist because the ID space is mixed and a tuple cannot be an xarray coordinate value, not because anyone should type one. A saved comparer records the original ID beside the integer, so reloading it does not depend on the numbering the installed mikeio1d happened to hand out.

The move is verified rather than trusted: the current loader's output over every fixture is snapshotted first — graph edges with their lengths and boundary flags, the alias map, the dataframe, the answers `find` and `recall` give — and those snapshots become the upstream module's acceptance test. Code moves verbatim before it is cleaned up. Neither project releases the feature until both can: modelskill 1.4.0 waits for the mikeio1d release that carries the module.

## Alternatives Considered

**Keep the layer here.** Defensible while the API is private, but it means maintaining a format matrix, an EPANET `.inp` parser and a graph contract for traversals we never perform — and taking a CI failure when someone else's release adds a format.

**Move only the constructors and companions**, leaving the graph and the abstract types here. Splits the format knowledge from the topology it produces, and leaves `Res1DReach` here as the single adapter for a plug point with no second implementation.

**A separate `modelskill-network` package.** Rejected in ADR-010 for fragmenting the install, and it would still own format knowledge that mikeio1d has better.

**Ask mikeio1d to guarantee stable node numbering** instead of dropping integers from our API. Puts a promise on someone else's release process to protect a number users should never have been handling.

## Consequences

- ADR-012 is narrowed: the constructors, the companion arguments and the extension tables it describes become mikeio1d's, and the coverage test goes with them. Its reasoning about naming constructors after products still holds — upstream is simply where it now applies.
- ADR-010's open question about version constraints for optional dependencies is answered for this feature: the `networks` extra pins a minimum mikeio1d, and network support requires whatever Python that release requires.
- A hand-built network needs mikeio1d installed, since `BasicNode`/`BasicReach` move too. That costs a .NET dependency for users who touch no MIKE file, which only matters for tests and for a backend nobody has written.
- Dropping `at=<int>` is a breaking change for a signature that shipped in the 1.4.0a3 alpha only, while the network module is opt-in and absent from the API reference. Cheap now, expensive after 1.4.0.
- Releases become coupled in one direction: a fix to network file reading ships on mikeio1d's schedule, not ours. In exchange, a format mikeio1d adds no longer breaks our CI.
