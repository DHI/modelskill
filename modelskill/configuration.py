from pathlib import Path
import pandas as pd
import yaml
from typing import Union

from . import model_result, match
from .obs import PointObservation, TrackObservation
from .comparison import ComparerCollection


def from_config(
    conf: Union[dict, str, Path], *, relative_path=True
) -> ComparerCollection:
    """Load ComparerCollection from a config file (or dict)

    Parameters
    ----------
    conf : Union[str, Path, dict]
        path to config file or dict with configuration
    relative_path: bool, optional
        True: file paths are relative to configuration file,
        False: file paths are absolute (relative to the current directory),
        by default True

    Returns
    -------
    ComparerCollection
        A ComparerCollection object from the given configuration

    Examples
    --------
    >>> import modelskill as ms
    >>> cc = ms.from_config('Oresund.yml')
    """
    if isinstance(conf, (str, Path)):
        p = Path(conf)
        ext = p.suffix
        dirname = Path(str(p.parents[0]))
        if (ext == ".yml") or (ext == ".yaml") or (ext == ".conf"):
            conf = _yaml_to_dict(p)
        elif "xls" in ext:
            conf = _excel_to_dict(p)
        else:
            raise ValueError("Filename extension not supported! Use .yml or .xlsx")
    else:
        dirname = Path(".")

    assert isinstance(conf, dict)
    modelresults = []
    for name, mr_dict in conf["modelresults"].items():
        if not mr_dict.get("include", True):
            continue
        fp = Path(mr_dict["filename"])
        if relative_path:
            fp = dirname / fp

        item = mr_dict.get("item")
        mr = model_result(fp, name=name, item=item)
        modelresults.append(mr)

    observations = []
    for name, obs_dict in conf["observations"].items():
        if not obs_dict.get("include", True):
            continue
        fp = Path(obs_dict["filename"])
        if relative_path:
            fp = dirname / fp

        item = obs_dict.get("item")
        alt_name = obs_dict.get("name")
        name = name if alt_name is None else alt_name

        otype = obs_dict.get("type")
        if (otype is not None) and ("track" in otype.lower()):
            obs = TrackObservation(fp, item=item, name=name)  # type: ignore
        else:
            x, y = obs_dict.get("x"), obs_dict.get("y")
            obs = PointObservation(fp, item=item, x=x, y=y, name=name)  # type: ignore
        observations.append(obs)

    # if "connections" in conf:
    #     raise NotImplementedError()
    return match(obs=observations, mod=modelresults)


def _yaml_to_dict(filename: Path) -> dict:
    with open(filename) as f:
        contents = f.read()
    conf = yaml.load(contents, Loader=yaml.FullLoader)
    return conf


def _excel_to_dict(filename: Path) -> dict:
    with pd.ExcelFile(filename, engine="openpyxl") as xls:
        dfmr = pd.read_excel(xls, "modelresults", index_col=0).T
        dfo = pd.read_excel(xls, "observations", index_col=0).T
        # try: dfc = pd.read_excel(xls, "connections", index_col=0).T
    conf = {}
    conf["modelresults"] = _remove_keys_w_nan_value(dfmr.to_dict())
    conf["observations"] = _remove_keys_w_nan_value(dfo.to_dict())
    return conf


def _remove_keys_w_nan_value(d: dict) -> dict:
    """Loops through dicts in dict and removes all entries where value is NaN
    e.g. x,y values of TrackObservations
    """
    dout = {}
    for key, subdict in d.items():
        dout[key] = {k: v for k, v in subdict.items() if pd.Series(v).notna().all()}
    return dout
