# ADR-012: Public vs. internal API convention

**Status**: Accepted

**Date**: 2026-05-14

## Context

ModelSkill had accumulated an inconsistent privacy convention. Leading underscores appeared on both **file names** (`_comparison.py`, `_misc.py`, `_timeseries.py`) and **function or constant names** (`_get_name`, `_parse_metric`, `_validate_data_var_name`), with no written rule about what each level meant or how the two interacted. Symptoms:

- `from modelskill.timeseries._timeseries import _validate_data_var_name` — privacy claimed at both the path segment *and* the function name, while the function was imported from 4+ other files across subpackages.
- `timeseries/__init__.py` placed five underscore-prefixed names in `__all__`, simultaneously declaring them "officially public" and "private."
- Pyright (Pylance) flagged `reportPrivateUsage` on every cross-module `_foo` import, generating ambient warnings during development.
- No written policy distinguished "what users can rely on" from "what the package uses internally."

Three candidate definitions of "public" were considered:

- **Strict**: only names in `modelskill.__all__` at the top level.
- **Documented**: anything appearing in `docs/api/`.
- **Conventional**: anything reachable as `modelskill.<...>` without a leading `_` in any path segment.

The conventional definition was chosen because the team did not want documentation gaps to silently shrink the public API: a function reachable via a non-underscored import path could already have been adopted by users, even if undocumented.

## Decision

**Privacy is set by the module path, not the function name.** A name is private if any segment of its import path starts with `_`. Functions inside private modules use plain names (no leading `_`); the path already carries the privacy signal. A leading `_` on a function or class name is reserved for symbols that live in a public module but are file-local — i.e., used only inside that one file.

Examples:

- `modelskill.timeseries._timeseries.validate_data_var_name` — private (module path).
- `modelskill._utils.get_name` — private (module path).
- `modelskill.metrics._format_directional_label` (hypothetical) — private (function name); used only inside `metrics.py`.
- `modelskill.matching.match` — public.

This is the same convention used by scikit-learn (`sklearn.utils._param_validation.validate_params`).

The convention is mechanically enforced by Ruff rule `PLC2701` (`import-private-name`), which flags `from x import _foo` style imports. After the renames applied with this ADR, the codebase has zero PLC2701 violations in `src/`. Tests are allowed to import private names via a per-file ignore, since white-box testing of internals is a legitimate need.

## Alternatives Considered

**Snapshot-test the public surface.** A test that asserts `dir(modelskill)` matches a frozen list catches accidental public additions but requires updating the expected list on every intentional public change. Rejected as ongoing maintenance cost for limited additional coverage beyond what Ruff and PR review already provide.

**Introduce a `modelskill/_internal/` subpackage.** Concentrating all cross-cutting internal helpers in one location would make "shared internal API" structurally obvious. Rejected as pure churn: the helpers already live in natural subpackage homes, and moving them does not change the privacy story under this convention.

**Rename `utils.py` → `_utils.py` and rely on a deprecation shim.** Would break legitimately-public functions (`rename_coords_xr`, `rename_coords_pd`, `make_unique_index`) that are imported from non-underscored paths internally and could already be used by downstream consumers. The actual cross-cutting *private* helpers (`_get_name`, `_get_idx`, `_RESERVED_NAMES`) were split out into a new `modelskill/_utils.py` module instead, leaving `utils.py` as a public module.

**Keep the status quo and configure Pyright to silence `reportPrivateUsage`.** Rejected because the warning is genuinely useful for catching future drift; removing the noise by suppression would also remove the signal.

## Consequences

- **Cross-module use of internal helpers is no longer flagged.** `from modelskill._utils import get_name` is unambiguously legal; the module path says "internal." Pyright's `reportPrivateUsage` falls silent.
- **Function-name underscores carry a sharper meaning.** A `_function_name` now signals "private to this one file" — never "private to this subpackage." Reviewers can use that signal at face value.
- **Future contributors have a clear default.** New internal helpers go in `_modules`, not in `_underscored_names` inside public modules. PRs that put internal helpers in `metrics.py` or `utils.py` with a leading underscore now stand out as the exception rather than the norm.
- **One breaking change is accepted.** Five underscored names were previously re-exported from `modelskill.timeseries` via `__all__` (`_parse_track_input`, etc.). These names are no longer reachable from `modelskill.timeseries`; the renamed functions live in `modelskill.timeseries._point`, `_track`, `_vertical`. By Python convention, leading-underscore names were never part of the stable API, so this is consistent with the new policy.
- **Tests retain access to internals** via a `tests/**/*.py` per-file ignore for `PLC2701`. White-box testing of private modules remains supported.
- **`metrics.py` no longer hosts `_parse_metric` or `_linear_regression`.** They moved to `comparison/_utils.py:parse_metric` and `_utils.py:linear_regression` respectively. The first is consumer-specific to the comparison subpackage; the second is a general regression helper used by both `metrics.lin_slope` and `plotting._scatter`.
