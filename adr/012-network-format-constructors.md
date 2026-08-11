# ADR-012: One Network Constructor per Modelling Product

**Status**: Draft

**Date**: 2026-08

## Context

`Network` is built from result files read through mikeio1d, whose single `Res1D` class opens nine extensions across five products — MIKE 1D (`.res1d`), MIKE 11 (`.res11`), MOUSE (`.prf`, `.crf`, `.xrf`), EPANET (`.res`), SWMM (`.out`), Water Hammer (`.whr`), and `.resx`, which is shared by the last three. There is no per-format reader and no per-format constructor argument, so from mikeio1d's side all nine look alike. modelskill's constructor was named `from_res1d`, and its extension guard was briefly widened to accept everything mikeio1d could read — making the name promise one format while reading nine.

Loading mikeio1d's own fixtures showed the nine are not interchangeable. `.res1d` and `.res11` give a full network with real reach lengths and gridpoints. EPANET's `.res` loads, but as a link-node model it reports no reach length and one synthetic gridpoint per reach, so reach-based matching cannot work. `.out` (SWMM) and `.resx` carry no reach connectivity at all — it lives in a companion file: SWMM's `.inp`, and for `.resx` the sibling `.res` that defines the network the results are added to. MOUSE and `.whr` have no test fixture anywhere, upstream included, so nothing about them can be verified.

## Decision

Name constructors after the product that writes the file, and ship one only where a committed fixture backs it:

| Constructor | Extensions |
|---|---|
| `Network.from_mike` | `.res1d`, `.res11` |
| `Network.from_epanet` | `.res`, plus optional `.resx` and `.inp` companions |

`NetworkModelResult` is exempt: it accepts a path and reads it with the constructor its extension is mapped to, since every other model result class already takes a path.

A product's companion files are arguments rather than constructors of their own. A companion describes a network defined elsewhere and cannot stand alone, so `from_epanet(res, resx=..., inp=...)` and not a `from_resx()`. Each companion is validated against the main file — same time axis, no unknown IDs — because two unrelated runs would otherwise merge silently.

Every extension mikeio1d reads is accounted for in one of three module-level tables in `network.py`: readable by `from_mike`, readable by `from_epanet`, or refused with a reason that names the file or method which would lift it. A test asserts the tables cover exactly `Res1D.get_supported_file_extensions()`, so a mikeio1d release adding a tenth format fails CI instead of leaving that format silently unreachable. `from_res1d` is removed without a deprecation shim: it shipped only in the 1.4.0a3 alpha, and the network module is opt-in and absent from the API reference.

## Alternatives Considered

**One constructor per extension** - `from_res`, `from_out` and `from_whr` say nothing about the product they belong to, and MOUSE would need three identical methods.

**A generic catch-all (`from_file`, `from_mikeio1d`)** - a second way to do the same thing. With every extension either read or explicitly refused, its only remaining job is forward compatibility, which the coverage test handles more usefully by demanding a decision.

**Auto-detect the product, as ADR-009 does elsewhere** - factories such as `model_result()` resolve *which class* to build from the shape of the data. Here the question is *which product wrote the file*, which the call site should state rather than have guessed, since the answer decides whether reach-based matching works at all.

**Ship all five product constructors** - MOUSE and Water Hammer would be unverifiable, so the method list would stop being a reliable statement of what works. SWMM is deferred rather than impossible, since its `.inp` does carry the missing topology ([#689](https://github.com/DHI/modelskill/issues/689)).

## Consequences

- The method list is the format list: `Network.from_<TAB>` answers "which formats does this read", and passing a file the other constructor handles raises a `ValueError` naming that constructor.
- EPANET's degenerate geometry is stated in the `from_epanet` docstring and the user guide and asserted in tests, rather than warned about at runtime. A warning would fire on correct usage, and both consequences already raise where they bite.
- `NetworkModelResult(path)` reads the extension table rather than asking the caller, which is the one place the guessing objection above does not bite: the tables map each extension to exactly one product, and the answer is reported in `mr.network`. It picks up an EPANET file's `.resx` and `.inp` siblings for the same reason, since a network built without the `.inp` has no reach lengths at all. Anything needing named companions or selective loading still goes through `Network.from_*`.
- MOUSE and Water Hammer are refused even though mikeio1d may well read them correctly. Refusing with a reason is recoverable; a method that silently builds a wrong graph is not. Each becomes a six-line addition once a redistributable fixture exists.
- The `.inp` reader (`model/adapters/_inp.py`) is ours to maintain, since mikeio1d does not read `.inp` and pulling in `wntr` or `swmmio` for two sections would weigh more than the parser does (ADR-010). SWMM support will reuse it, as the two products share the layout.
