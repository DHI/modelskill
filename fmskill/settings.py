"""
The settings module holds package-wide configurables and provides
a uniform API for working with them.

This module is heavily inspired by pandas config module
https://github.com/pandas-dev/pandas/tree/main/pandas/_config 

Overview
========
This module supports the following requirements:
- options are referenced using keys in dot.notation, e.g. "x.y.option - z".
- keys are case-insensitive.
- functions should accept partial/regex keys, when unambiguous.
- options can be registered by modules at import time.
- options have a default value, and (optionally) a description and
  validation function associated with them.
- options can be reset to their default value.
- all option can be reset to their default value at once.
- all options in a certain sub - namespace can be reset at once.
- the user can set / get / reset or ask for the description of an option.
- a developer can register an option.

Implementation
==============
- Data is stored using nested dictionaries, and should be accessed
  through the provided API.
- "Registered options" have metadata associated
  with them, which are stored in auxiliary dictionaries keyed on the
  fully-qualified key, e.g. "x.y.z.option".

"""
import re
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Dict,
    Tuple,
    Type,
)
import warnings
import fmskill.metrics as mtr


class RegisteredOption(NamedTuple):
    key: str
    defval: object
    doc: str
    validator: Optional[Callable[[object], Any]]
    # cb: Optional[Callable[[str], Any]]


# holds registered option metadata
_registered_options: Dict[str, RegisteredOption] = {}

# holds the current values for registered options
_global_settings: Dict[str, Any] = {}


class OptionError(AttributeError, KeyError):
    pass


def _get_single_key(pat: str) -> str:
    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError(f"No such keys(s): {repr(pat)}")
    if len(keys) > 1:
        raise OptionError("Pattern matched multiple keys")
    key = keys[0]

    # key = _translate_key(key)  # deprecated keys

    return key


def get_option(pat: str) -> Any:
    key = _get_single_key(pat)

    # walk the nested dict
    root, k = _get_root(key)
    return root[k]


def set_option(*args, **kwargs) -> None:
    # must at least 1 arg deal with constraints later
    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword arguments")

    # default to false
    silent = kwargs.pop("silent", False)

    if kwargs:
        kwarg = list(kwargs.keys())[0]
        raise TypeError(f'_set_option() got an unexpected keyword argument "{kwarg}"')

    for k, v in zip(args[::2], args[1::2]):
        key = _get_single_key(k)  # , silent)

        o = _get_registered_option(key)
        if o and o.validator:
            o.validator(v)

        # walk the nested dict
        root, k = _get_root(key)
        root[k] = v

        # if o.cb:
        #     if silent:
        #         with warnings.catch_warnings(record=True):
        #             o.cb(key)
        #     else:
        #         o.cb(key)


def _describe_option_short(pat: str = "", _print_desc: bool = True) -> Optional[str]:

    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    s = "\n".join([f"{k} : {get_option(k)}" for k in keys])

    if _print_desc:
        print(s)
        return None
    return s


def describe_option(pat: str = "", _print_desc: bool = True) -> Optional[str]:

    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    s = "\n".join([_build_option_description(k) for k in keys])

    if _print_desc:
        print(s)
        return None
    return s


def reset_option(pat: str, silent: bool = False) -> None:

    keys = _select_options(pat)

    if len(keys) == 0:
        raise OptionError("No such keys(s)")

    if len(keys) > 1 and len(pat) < 4 and pat != "all":
        raise ValueError(
            "You must specify at least 4 characters when "
            "resetting multiple keys, use the special keyword "
            '"all" to reset all the options to their default value'
        )

    for k in keys:
        set_option(k, _registered_options[k].defval, silent=silent)


def get_default_val(pat: str):
    key = _get_single_key(pat, silent=True)
    return _get_registered_option(key).defval


class OptionsContainer:
    """provide attribute-style access to a nested dict"""

    def __init__(self, d: Dict[str, Any], prefix: str = "") -> None:
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "prefix", prefix)

    def __setattr__(self, key: str, val: Any) -> None:
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        # you can't set new keys
        # can you can't overwrite subtrees
        if key in self.d and not isinstance(self.d[key], dict):
            set_option(prefix, val)
        else:
            raise OptionError("You can only set the value of existing options")

    def __getattr__(self, key: str):
        prefix = object.__getattribute__(self, "prefix")
        if prefix:
            prefix += "."
        prefix += key
        try:
            v = object.__getattribute__(self, "d")[key]
        except KeyError as err:
            raise OptionError("No such option") from err
        if isinstance(v, dict):
            return OptionsContainer(v, prefix)
        else:
            return get_option(prefix)

    def __repr__(self) -> str:
        return _describe_option_short(self.prefix, False)

    def __dir__(self) -> Iterable[str]:
        return list(self.d.keys())


def _select_options(pat: str) -> List[str]:
    """
    returns a list of keys matching `pat`
    if pat=="all", returns all registered options
    """
    # short-circuit for exact key
    if pat in _registered_options:
        return [pat]

    # else look through all of them
    keys = sorted(_registered_options.keys())
    if pat == "all":  # reserved key
        return keys

    return [k for k in keys if re.search(pat, k, re.I)]


def _get_root(key: str) -> Tuple[Dict[str, Any], str]:
    path = key.split(".")
    cursor = _global_settings
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _get_registered_option(key: str):
    """
    Retrieves the option metadata if `key` is a registered option.
    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """
    return _registered_options.get(key)


def _build_option_description(k: str) -> str:
    """Builds a formatted description of a registered option and prints it"""
    o = _get_registered_option(k)

    s = f"{k} "

    if o.doc:
        s += "\n".join(o.doc.strip().split("\n"))
    else:
        s += "No description available."

    if o:
        s += f"\n    [default: {o.defval}] [currently: {get_option(k)}]"

    return s


options = OptionsContainer(_global_settings)


def register_option(
    key: str,
    defval: object,
    doc: str = "",
    validator: Optional[Callable[[object], Any]] = None,
    # cb: Optional[Callable[[str], Any]] = None,
) -> None:
    """
    Register an option in the package-wide pandas config object
    Parameters
    ----------
    key : str
        Fully-qualified key, e.g. "x.y.option - z".
    defval : object
        Default value of the option.
    doc : str
        Description of the option.
    validator : Callable, optional
        Function of a single argument, should raise `ValueError` if
        called with a value which is not a legal value for the option.
    disabled: cb
        a function of a single argument "key", which is called
        immediately after an option value is set/reset. key is
        the full name of the option.
    Raises
    ------
    ValueError if `validator` is specified and `defval` is not a valid value.
    """
    import keyword
    import tokenize

    key = key.lower()

    if key in _registered_options:
        raise OptionError(f"Option '{key}' has already been registered")
    # if key in _reserved_keys:
    #     raise OptionError(f"Option '{key}' is a reserved key")

    # the default value should be legal
    if validator:
        validator(defval)

    # walk the nested dict, creating dicts as needed along the path
    path = key.split(".")

    for k in path:
        if not re.match("^" + tokenize.Name + "$", k):
            raise ValueError(f"{k} is not a valid identifier")
        if keyword.iskeyword(k):
            raise ValueError(f"{k} is a python keyword")

    cursor = _global_settings
    msg = "Path prefix to option '{option}' is already an option"

    for i, p in enumerate(path[:-1]):
        if not isinstance(cursor, dict):
            raise OptionError(msg.format(option=".".join(path[:i])))
        if p not in cursor:
            cursor[p] = {}
        cursor = cursor[p]

    if not isinstance(cursor, dict):
        raise OptionError(msg.format(option=".".join(path[:-1])))

    cursor[path[-1]] = defval  # initialize

    # save the option metadata
    _registered_options[key] = RegisteredOption(
        key=key, defval=defval, doc=doc, validator=validator  # , cb=cb
    )


def is_type_factory(_type: Type[Any]) -> Callable[[Any], None]:
    """
    Parameters
    ----------
    `_type` - a type to be compared against (e.g. type(x) == `_type`)
    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if type(x) is not equal to `_type`
    """

    def inner(x) -> None:
        if type(x) != _type:
            raise ValueError(f"Value must have type '{_type}'")

    return inner


def is_instance_factory(_type) -> Callable[[Any], None]:
    """
    Parameters
    ----------
    `_type` - the type to be checked against
    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if x is not an instance of `_type`
    """
    if isinstance(_type, (tuple, list)):
        _type = tuple(_type)
        type_repr = "|".join(map(str, _type))
    else:
        type_repr = f"'{_type}'"

    def inner(x) -> None:
        if not isinstance(x, _type):
            raise ValueError(f"Value must be an instance of {type_repr}")

    return inner


# common type validators, for convenience
# usage: register_option(... , validator = is_int)
is_int = is_type_factory(int)
is_bool = is_type_factory(bool)
is_float = is_type_factory(float)
is_str = is_type_factory(str)
is_text = is_instance_factory((str, bytes))


def is_callable(obj) -> bool:
    """
    Parameters
    ----------
    `obj` - the object to be checked
    Returns
    -------
    validator - returns True if object is callable
        raises ValueError otherwise.
    """
    if not callable(obj):
        raise ValueError("Value must be a callable")
    return True
