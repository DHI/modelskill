# ADR-013: The Network Topology Layer Belongs to mikeio1d

**Status**: Accepted

**Date**: 2026-08

## Context

`modelskill.network` grew into a topology layer of its own: the abstract node/reach/breakpoint types, a `Res1D` adapter, one constructor per modelling product, the `.resx` and `.inp` companions, a table of which extensions we refuse and why, a networkx graph carrying reach lengths and boundary edges, an alias map, and `find`/`recall`/`to_dataset` on top. Roughly 630 code lines, against 25 for mikeio1d's `experimental.to_networkx`, which converts the same file and ignores gridpoints.

Most of those 630 lines do work the upstream function declines to do. The layer still sits on the wrong side of a line ADR-001 drew for mikeio: we call `mikeio.read()` and stop, without modelling dfsu geometry or policing its format list. Here we do both. `Res1D` reads nine extensions across five products; our tables decide which of the nine we accept, and a test fails our CI when a mikeio1d release adds a tenth. The fixtures those tables are checked against — `network.res1d`, `network_cali.res11`, `epanet.res/.resx/.inp` — are copies of mikeio1d's own. Meanwhile `NetworkModelResult` uses five members of `Network`, two of them private, and never traverses the graph.

## Decision

mikeio1d gains an optional network module that builds and owns `Network`. modelskill requires it and consumes what it produces.

| Owner | Pieces |
|---|---|
| mikeio1d | abstract types and `BasicNode`/`BasicReach`, the `Res1D` adapter, `Network.open`, the `.resx` and `.inp` companions, the extension policy tables, graph construction with its length and boundary semantics, the alias map, `find`, `recall`, `to_dataframe`, `to_dataset` |
| modelskill | `NetworkModelResult`, `NodeModelResult`, `NodeObservation`, `ReachObservation`, matching, the MIKE+ station resolver |

`NetworkModelResult` takes a `Network` the upstream module built, or a path it hands to that module. The module is an extra there, carrying networkx and xarray, so `to_dataset()` ships with the class. modelskill's `network` extra requires a mikeio1d release new enough to contain it.

Original IDs become the only identifier a user handles: `NodeObservation.at` takes a node name or a `(reach, distance)` pair, and no longer an integer. The alias integers stay an internal index. They exist because the ID space mixes names and break points, and a tuple cannot be an xarray coordinate value. A saved comparer records the original ID, with the integer beside it as `node_index`, so reloading does not depend on the numbering the installed mikeio1d handed out.

Before anything moved, the loader's output over six fixture loads was recorded: graph edges with their lengths and boundary flags, the alias map, the dataframe, and every answer `find` and `recall` give. Those snapshots are the upstream module's acceptance test.

Phase 1 landed as mikeio1d [#247](https://github.com/DHI/mikeio1d/pull/247), merged 2026-08-19. The snapshots pass there unchanged, twice: against the code moved verbatim, and again after the two product constructors collapsed into `Network.open`. The second pass covers the redesign.

modelskill 1.4.0 waits for the mikeio1d release carrying the module. That release is not out yet.

## Alternatives Considered

**Keep the layer here.** Defensible while the API is private. It means maintaining a format matrix, an EPANET `.inp` parser and a graph contract for traversals we never perform. It also means a CI failure whenever someone else's release adds a format.

**Move only the constructors and companions**, leaving the graph and the abstract types here. Splits the format knowledge from the topology it produces, and leaves `Res1DReach` here as the single adapter for a plug point with no second implementation.

**A separate `modelskill-network` package.** Rejected in ADR-010 for fragmenting the install. It would still own format knowledge that belongs with mikeio1d.

**Ask mikeio1d to guarantee stable node numbering** instead of dropping integers from our API. Puts a promise on someone else's release process, to protect a number users should not be handling.

## Consequences

- ADR-012 is narrowed: the constructors, the companion arguments, the extension tables and the coverage test become mikeio1d's. Naming a constructor after the product that wrote the file is still the rule, and mikeio1d applies it.
- ADR-010's open question about version constraints for optional dependencies is answered for this feature: the `network` extra pins a minimum mikeio1d, and network support requires whatever Python that release requires.
- A hand-built network needs mikeio1d installed, since `BasicNode`/`BasicReach` move too. That costs a .NET dependency for users who touch no MIKE file, which only matters for tests and for a backend nobody has written.
- Dropping `at=<int>` is a breaking change for a signature that shipped in the 1.4.0a3 alpha only, while the network module is opt-in and absent from the API reference. Removing it after 1.4.0 would cost more.
- Releases become coupled in one direction: a fix to network file reading ships on mikeio1d's schedule. A format mikeio1d adds no longer breaks our CI.
