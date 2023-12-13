import os
import pandas as pd
import yaml
from typing import Union

from . import model_result
from .observation import PointObservation, TrackObservation
from .connection import Connector


def from_config(conf: Union[dict, str], *, validate_eum=True, relative_path=True):
    """Load Connector from a config file (or dict)

    Parameters
    ----------
    configuration : Union[str, dict]
        path to config file or dict with configuration
    validate_eum : bool, optional
        require eum to match, by default True
    relative_path: bool, optional
            file path are relative to configuration file, and not current directory

    Returns
    -------
    Connector
        A Connector object with the given configuration

    Examples
    --------
    >>> import modelskill as ms
    >>> con = ms.from_config('Oresund.yml')
    >>> cc = con.extract()
    """
    if isinstance(conf, str):
        filename = conf
        ext = os.path.splitext(filename)[-1]
        dirname = os.path.dirname(filename)
        if (ext == ".yml") or (ext == ".yaml") or (ext == ".conf"):
            conf = _yaml_to_dict(filename)
        elif "xls" in ext:
            conf = _excel_to_dict(filename)
        else:
            raise ValueError("Filename extension not supported! Use .yml or .xlsx")
    else:
        dirname = ""

    modelresults = {}

    assert isinstance(conf, dict)
    for name, mr_dict in conf["modelresults"].items():
        if not mr_dict.get("include", True):
            continue
        if relative_path:
            filename = os.path.join(dirname, mr_dict["filename"])
        else:
            filename = mr_dict["filename"]
        item = mr_dict.get("item")
        mr = model_result(filename, name=name, item=item)
        modelresults[name] = mr
    mr_list = list(modelresults.values())

    observations = {}
    for name, obs_dict in conf["observations"].items():
        if not obs_dict.get("include", True):
            continue
        if relative_path:
            filename = os.path.join(dirname, obs_dict["filename"])
        else:
            filename = obs_dict["filename"]
        item = obs_dict.get("item")
        alt_name = obs_dict.get("name")
        name = name if alt_name is None else alt_name

        otype = obs_dict.get("type")
        if (otype is not None) and ("track" in otype.lower()):
            obs = TrackObservation(filename, item=item, name=name)  # type: ignore
        else:
            x, y = obs_dict.get("x"), obs_dict.get("y")
            obs = PointObservation(filename, item=item, x=x, y=y, name=name)  # type: ignore
        observations[name] = obs
    obs_list = list(observations.values())

    if "connections" in conf:
        raise NotImplementedError()
    else:
        con = Connector(obs_list, mr_list, validate=validate_eum)
    return con


def _yaml_to_dict(filename):
    with open(filename) as f:
        contents = f.read()
    conf = yaml.load(contents, Loader=yaml.FullLoader)
    return conf


def _excel_to_dict(filename):
    with pd.ExcelFile(filename, engine="openpyxl") as xls:
        dfmr = pd.read_excel(xls, "modelresults", index_col=0).T
        dfo = pd.read_excel(xls, "observations", index_col=0).T
        # try: dfc = pd.read_excel(xls, "connections", index_col=0).T
    conf = {}
    conf["modelresults"] = _remove_keys_w_nan_value(dfmr.to_dict())
    conf["observations"] = _remove_keys_w_nan_value(dfo.to_dict())
    return conf


def _remove_keys_w_nan_value(d):
    """Loops through dicts in dict and removes all entries where value is NaN
    e.g. x,y values of TrackObservations
    """
    dout = {}
    for key, subdict in d.items():
        dout[key] = {k: v for k, v in subdict.items() if pd.Series(v).notna().all()}
    return dout
