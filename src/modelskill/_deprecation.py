"""Deprecation utilities for ModelSkill.

This module provides utilities for deprecating positional arguments in functions,
following scikit-learn's SLEP009 approach.
"""

from __future__ import annotations

import warnings
from functools import wraps
from inspect import Parameter, signature
from typing import Any, Callable, TypeVar, overload

from . import __version__

# Type variable for the decorated function
F = TypeVar("F", bound=Callable[..., Any])


@overload
def _deprecate_positional_args(func: F) -> F: ...


@overload
def _deprecate_positional_args(
    func: None = None, *, version: str = ...
) -> Callable[[F], F]: ...


def _deprecate_positional_args(
    func: F | None = None, *, version: str = "2.0"
) -> F | Callable[[F], F]:
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in PEP 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    This is a temporary migration tool that will be removed in a future major version.
    The decorator preserves type annotations and works with mypy.

    Parameters
    ----------
    func : callable, optional
        Function to check arguments on.
    version : str, default="2.0"
        The version when positional arguments will result in an error.

    Returns
    -------
    callable
        Decorated function that warns when keyword-only args are passed positionally.

    Examples
    --------
    >>> @_deprecate_positional_args
    ... def function(data, option1=None, option2=None):
    ...     pass

    >>> @_deprecate_positional_args(version="2.0")
    ... def function(data, option1=None, option2=None):
    ...     pass
    """

    def _inner_deprecate_positional_args(f: F) -> F:
        sig = signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args: Any, **kwargs: Any) -> Any:
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = [
                f"{name}={arg}"
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg_str = ", ".join(args_msg)

            # Get current version without dev suffix
            current_version = __version__.split(".dev")[0]

            warnings.warn(
                f"Passing {args_msg_str} as positional argument(s) is deprecated "
                f"since version {current_version} and will raise an error in version {version}. "
                f"Please use keyword argument(s) instead.",
                FutureWarning,
                stacklevel=2,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f  # type: ignore[return-value]

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args
