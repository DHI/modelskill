"""
Tests for consistent plotting APIs across modelskill.
Does not test plotting functionality itself, only format and annotations of function / method signatures.
"""
import inspect
import modelskill as ms
import pytest
import matplotlib.pyplot as plt

plt.switch_backend("Agg")


# TODO: This is a hacky way to find public plotting functions/methods in modelskill to test their signatures.
#
# - Main purpose was to avoid writing tests that might change as plotting is refactored.
# - Works okay, but is not perfect. Could throw it away or keep it as a starting point.
# - Potential improvements if we keep it:
#   - Use Sphinx to generate a list of all public plotting functions/methods? What's the one true source?
#   - Mark public API functions/methods/classes with @api annotation?


# Determines if a function / method is considered a plotting function
def is_plotting_func(func):
    if not inspect.isfunction(func) and not inspect.ismethod(func):
        raise TypeError("obj must be a function or method")
    keywords = [
        "plot(",
        "plot_",
        "plot ",
        " plot.",
        " plots.",
        "plt.subplot",
        "plt.figure",
    ]

    source = inspect.getsource(func).lower()
    return any([k in source for k in keywords])


def is_public_api(alias, obj):
    """
    Determines whether the given object is part of the public API of the ModelSkill package.

    Args:
        alias (str): The namespace alias of the object, e.g. "ms.example".
        obj (object): The module / class / function / method that the alias refers to.

    Returns:
        bool: True if the object is part of the public API, False otherwise.
    """

    def obj_modulename_contains(keyword):
        module = inspect.getmodule(obj)
        if module:
            return keyword in module.__name__
        return False

    # Only modelskill, no third parties
    if not obj_modulename_contains("modelskill"):
        return False
    # Skip these modules
    modules_to_skip = ["connection"]
    if any([obj_modulename_contains(m) for m in modules_to_skip]):
        return False
    # No private aliases (e.g. ms.example is okay, but ms._example is not)
    if any([a.startswith("_") for a in alias.split(".")]):
        return False
    # Docstrings must exist for classes, functions, and methods
    if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
        if inspect.getdoc(obj) is None:
            return False
    return True


def find_plotting_funcs(module_or_class, _plotting_functions=None):
    """
    Recursively searches through a module or class and its submodules and subclasses to find all plotting functions/methods.

    Args:
        module_or_class (module or class): The module or class to search through.
        plotting_functions (list, optional): A list of plotting functions found so far. Only used for recursion.

    Returns:
        list: A list of all plotting functions/methods found in the module or class and its submodules and subclasses.
    """
    if _plotting_functions is None:
        _plotting_functions = []

    def add_if_is_plotting_function(func):
        try:
            if is_plotting_func(func):
                if func not in _plotting_functions:
                    _plotting_functions.append(func)
        except OSError:
            print(f"Could not add {func}")

    for alias, obj in inspect.getmembers(module_or_class):
        if not is_public_api(alias, obj):
            continue
        elif inspect.ismodule(obj) or inspect.isclass(obj):
            find_plotting_funcs(obj, _plotting_functions)
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            add_if_is_plotting_function(obj)

    return _plotting_functions


plotting_functions = find_plotting_funcs(ms)
plotting_functions_ids = [
    f"{f.__module__}.{f.__qualname__}" for f in plotting_functions
]


@pytest.mark.parametrize(
    "plotting_function", plotting_functions, ids=plotting_functions_ids
)
def test_does_setup_work(plotting_function):
    pass
