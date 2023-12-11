from __future__ import annotations
import warnings
from typing import Iterable, Collection, Optional, overload, Hashable
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# TODO remove ?
def _validate_multi_index(index, min_levels=2, max_levels=2):
    errors = []
    if isinstance(index, pd.MultiIndex):
        if len(index.levels) < min_levels:
            errors.append(
                f"not possible for MultiIndex with fewer than {min_levels} levels"
            )
        if len(index.levels) > max_levels:
            errors.append(
                f"not possible for MultiIndex with more than {max_levels} levels"
            )
    else:
        errors.append("only possible for MultiIndex skill objects")
    return errors


class SkillArrayPlotter:
    def __init__(self, skillarray):
        self.skillarray = skillarray

    def _name_to_title_in_kwargs(self, kwargs):
        if "title" not in kwargs:
            if self.skillarray.name is not None:
                kwargs["title"] = self.skillarray.name

    def _get_plot_df(self, level: int | str = 0) -> pd.DataFrame:
        s = self.skillarray

        if isinstance(s.ser.index, pd.MultiIndex):
            df = s.ser.unstack(level=level)
        else:
            df = s.ser.to_frame()
        return df

    def line(
        self,
        level: int | str = 0,
        **kwargs,
    ):
        """plot statistic as a lines using pd.DataFrame.plot.line()

        Primarily for MultiIndex skill objects, e.g. multiple models and multiple observations

        Parameters
        ----------
        level : int or str, optional
            level to unstack, by default 0
        kwargs : dict, optional
            key word arguments to be pased to pd.DataFrame.plot.line()
            e.g. marker, title, figsize, ...

        Examples
        --------
        >>> s = comparer.skill()["rmse"]
        >>> s.plot.line()
        >>> s.plot.line(marker="o", linestyle=':')
        >>> s.plot.line(color=['0.2', '0.4', '0.6'])
        """
        df = self._get_plot_df(level=level)
        self._name_to_title_in_kwargs(kwargs)
        axes = df.plot.line(**kwargs)

        xlabels = list(df.index)
        nx = len(xlabels)

        if not isinstance(axes, Iterable):
            axes = [axes]
        for ax in axes:
            if not isinstance(df.index, pd.DatetimeIndex):
                ax.set_xticks(np.arange(nx))
                ax.set_xticklabels(xlabels, rotation=90)
        return axes

    def bar(self, level: int | str = 0, **kwargs):
        """plot statistic as bar chart using pd.DataFrame.plot.bar()

        Parameters
        ----------
        level : int or str, optional
            level to unstack, by default 0
        kwargs : dict, optional
            key word arguments to be pased to pd.DataFrame.plot.bar()
            e.g. color, title, figsize, ...

        Returns
        -------
        AxesSubplot

        Examples
        --------
        >>> s = comparer.skill()["rmse"]
        >>> s.plot.bar()
        >>> s.plot.bar(level="observation")
        >>> s.plot.bar(title="Root Mean Squared Error")
        >>> s.plot.bar(color=["red","blue"])
        """
        df = self._get_plot_df(level=level)
        self._name_to_title_in_kwargs(kwargs)
        return df.plot.bar(**kwargs)

    def barh(self, level: int | str = 0, **kwargs):
        """plot statistic as horizontal bar chart using pd.DataFrame.plot.barh()

        Parameters
        ----------
        level : int or str, optional
            level to unstack, by default 0
        kwargs : dict, optional
            key word arguments to be passed to pd.DataFrame.plot.barh()
            e.g. color, title, figsize, ...

        Returns
        -------
        AxesSubplot

        Examples
        --------
        >>> s = comparer.skill()["rmse"]
        >>> s.plot.barh()
        >>> s.plot.barh(level="observation")
        >>> s.plot.barh(title="Root Mean Squared Error")
        """
        df = self._get_plot_df(level)
        self._name_to_title_in_kwargs(kwargs)
        return df.plot.barh(**kwargs)

    def grid(
        self,
        show_numbers=True,
        precision=3,
        fmt=None,
        figsize=None,
        title=None,
        cmap=None,
    ):
        """plot statistic as a colored grid, optionally with values in the cells.

        Primarily for MultiIndex skill objects, e.g. multiple models and multiple observations

        Parameters
        ----------
        show_numbers : bool, optional
            should values of the static be shown in the cells?, by default True
            if False, a colorbar will be displayed instead
        precision : int, optional
            number of decimals if show_numbers, by default 3
        fmt : str, optional
            format string, e.g. ".0%" to show value as percentage
        figsize : Tuple(float, float), optional
            figure size, by default None
        title : str, optional
            plot title, by default name of statistic
        cmap : str, optional
            colormap, by default "OrRd" ("coolwarm" if bias)

        Examples
        --------
        >>> s = comparer.skill()["rmse"]
        >>> s.plot.grid()
        >>> s.plot.grid(show_numbers=False, cmap="magma")
        >>> s.plot.grid(precision=1)
        >>> s.plot.grid(fmt=".0%", title="Root Mean Squared Error")
        """

        s = self.skillarray

        errors = _validate_multi_index(s.ser.index)
        if len(errors) > 0:
            warnings.warn("plot_grid: " + "\n".join(errors))
            return None
            # df = self.df[field]    TODO: at_least_2d...
        df = s.ser.unstack()

        vmin = None
        vmax = None
        if cmap is None:
            cmap = "OrRd"
            if s.name == "bias":
                cmap = "coolwarm"
                mm = s.ser.abs().max()
                vmin = -mm
                vmax = mm
        if title is None:
            title = s.name
        xlabels = list(df.keys())
        nx = len(xlabels)
        ylabels = list(df.index)
        ny = len(ylabels)

        if (fmt is not None) and fmt[0] != "{":
            fmt = "{:" + fmt + "}"

        if figsize is None:
            figsize = (nx, ny)
        plt.figure(figsize=figsize)
        plt.pcolormesh(df, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.gca().set_xticks(np.arange(nx) + 0.5)
        plt.gca().set_xticklabels(xlabels, rotation=90)
        plt.gca().set_yticks(np.arange(ny) + 0.5)
        plt.gca().set_yticklabels(ylabels)
        if show_numbers:
            mean_val = df.to_numpy().mean()
            for ii in range(ny):
                for jj in range(nx):
                    val = df.iloc[ii, jj].round(precision)
                    col = "w" if val > mean_val else "k"
                    if s.name == "bias":
                        col = "w" if np.abs(val) > (0.7 * mm) else "k"
                    if fmt is not None:
                        val = fmt.format(val)
                    plt.text(
                        jj + 0.5,
                        ii + 0.5,
                        val,
                        ha="center",
                        va="center",
                        # size=15,
                        color=col,
                    )
        else:
            plt.colorbar()
        plt.title(title, fontsize=14)


class SkillArray:
    def __init__(self, ser) -> None:
        self.ser = ser
        self.plot = SkillArrayPlotter(self)

    def to_dataframe(self):
        return self.ser.to_frame()

    def __repr__(self):
        return repr(self.to_dataframe())

    def _repr_html_(self):
        return self.to_dataframe()._repr_html_()

    @property
    def name(self):
        return self.ser.name


class SkillTable:
    """
    SkillTable object for visualization and analysis returned by
    the comparer's skill method. The object wraps the pd.DataFrame
    class which can be accessed from the attribute df.

    Examples
    --------
    >>> s = comparer.skill()
    >>> s.mod_names
    ['SW_1', 'SW_2']
    >>> s.style()
    >>> s.sel(model='SW_1').style()
    >>> s.rmse.plot.bar()
    """

    large_is_best_metrics = [
        "cc",
        "corrcoef",
        "r2",
        "spearmanr",
        "rho",
        "nash_sutcliffe_efficiency",
        "nse",
    ]
    small_is_best_metrics = [
        "mae",
        "mape",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "rmse",
        "root_mean_squared_error",
        "urmse",
        "scatter_index",
        "si",
        "mef",
        "model_efficiency_factor",
    ]
    one_is_best_metrics = ["lin_slope"]
    zero_is_best_metrics = ["bias"]

    def __init__(self, df):
        self.df = df

    @property
    def metrics(self) -> Collection[str]:
        """List of metrics (columns) in the dataframe"""
        return list(self.df.columns)

    def __len__(self):
        return len(self.df)

    def to_dataframe(self):
        return self.df

    def __repr__(self):
        return repr(self.df)

    def _repr_html_(self):
        return self.df._repr_html_()

    @overload
    def __getitem__(self, key: Hashable | int) -> SkillArray:
        ...

    @overload
    def __getitem__(self, key: Iterable[Hashable]) -> SkillTable:
        ...

    def __getitem__(self, key) -> SkillArray | SkillTable:
        result = self.df.iloc[key] if isinstance(key, int) else self.df[key]
        if isinstance(result, pd.Series):
            return SkillArray(result)
        elif isinstance(result, pd.DataFrame):
            return SkillTable(result)
        else:
            return result

    def __getattr__(self, item):
        # Use __getitem__ for DataFrame column access
        if item in self.df.columns:
            return self[item]  # Redirects to __getitem__

        # For other attributes, return them directly
        return getattr(self.df, item)

    @property
    def loc(self, *args, **kwargs):
        return self.df.loc(*args, **kwargs)

    # TODO: remove?
    def sort_index(self, *args, **kwargs):
        """Wrapping pd.DataFrame.sort_index() for e.g. sorting by observation"""
        return self.__class__(self.df.sort_index(*args, **kwargs))

    # TODO: remove?
    def swaplevel(self, *args, **kwargs):
        """Wrapping pd.DataFrame.swaplevel() for e.g. swapping model and observation"""
        return self.__class__(self.df.swaplevel(*args, **kwargs))

    @property
    def mod_names(self):
        """List of model names (in index)"""
        return self._get_index_level_by_name("model")

    @property
    def obs_names(self):
        """List of observation names (in index)"""
        return self._get_index_level_by_name("observation")

    @property
    def var_names(self):
        """List of variable names (in index)"""
        return self._get_index_level_by_name("variable")

    # TODO what does this method actually do?
    def _get_index_level_by_name(self, name):
        index = self.df.index
        if name in index.names:
            level = index.names.index(name)
            return index.get_level_values(level).unique()
        else:
            return []
            # raise ValueError(f"name {name} not in index {list(self.index.names)}")

    def _id_to_name(self, index, id):
        """Assumes that index is valid and id is int"""
        if isinstance(id, Iterable):
            name_list = []
            for i in id:
                name_list.append(self._id_to_name(index, i))
            print(name_list)
            return name_list
        names = self._get_index_level_by_name(index)
        n = len(names)
        if (id < 0) or (id >= n):
            raise KeyError(f"Id {id} is out of bounds for index {index} (0, {n})")
        return names[id]

    def _sel_from_index(self, df, key, value):
        if (not isinstance(value, str)) and isinstance(value, Iterable):
            for i, v in enumerate(value):
                dfi = self._sel_from_index(df, key, v)
                if i == 0:
                    dfout = dfi
                else:
                    dfout = pd.concat([dfout, dfi])
            return dfout

        if isinstance(value, int):
            value = self._id_to_name(key, value)

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(value, level=key, drop_level=False)
        else:
            df = df[df.index == value]  # .copy()
        return df

    def sel(self, query=None, reduce_index=True, **kwargs):
        """Select a subset of the SkillTable by a query,
           (part of) the index, or specific columns

        Parameters
        ----------
        query : str, optional
            string supported by pd.DataFrame.query(), by default None
        reduce_index : bool, optional
            Should unnecessary levels of the index be removed after subsetting?
            Removed levels will stay as columns. By default True
        **kwargs : dict, optional
            "metrics"=... to select specific metrics (=columns),
            "model"=... to select specific models (=rows),
            "observation"=... to select specific observations (=rows)

        Returns
        -------
        SkillTable
            A subset of the orignal SkillTable

        Examples
        --------
        >>> s = comparer.skill()
        >>> s.sel(query="rmse>0.3")
        >>> s.sel(model = "SW_1")
        >>> s.sel(observation = ["EPL", "HKNA"])
        >>> s.sel(metrics="rmse")
        >>> s.sel("rmse>0.2", observation=[0, 2], metrics=["n","rmse"])
        """
        df = self.df

        if query is not None:
            if isinstance(query, str):
                df = df.query(query)

        for key, value in kwargs.items():
            if key in df.index.names:
                df = self._sel_from_index(df, key, value)
            elif key == "metrics" or key == "columns":
                cols = [value] if isinstance(value, str) else value
                df = df[cols]
            else:
                raise KeyError(
                    f"Unknown index {key}. Valid index names are {df.index.names}"
                )

        if isinstance(df, pd.Series):
            return SkillArray(df)
            # df = df.to_frame()
        if reduce_index and isinstance(df.index, pd.MultiIndex):
            df = self._reduce_index(df)
        return self.__class__(df)

    def _reduce_index(self, df):
        """Remove unnecessary levels of MultiIndex"""
        df.index = df.index.remove_unused_levels()
        levels_to_reset = []
        for j, level in enumerate(df.index.levels):
            if len(level) == 1:
                levels_to_reset.append(j)
        return df.reset_index(level=levels_to_reset)

    # TODO remove plot_* methods in v1.1
    # def plot_line(self, field, level=0, **kwargs):
    #     warnings.warn(
    #         "plot_line() is deprecated, use plot.line() instead", FutureWarning
    #     )
    #     return self.plot.line(field, level, **kwargs)

    # def plot_bar(self, field, level=0, **kwargs):
    #     warnings.warn("plot_bar() is deprecated, use plot.bar() instead", FutureWarning)
    #     return self.plot.bar(field, level, **kwargs)

    # def plot_barh(self, field, level=0, **kwargs):
    #     warnings.warn(
    #         "plot_barh() is deprecated, use plot.barh() instead", FutureWarning
    #     )
    #     return self.plot.barh(field, level, **kwargs)

    # def plot_grid(
    #     self,
    #     field,
    #     show_numbers=True,
    #     precision=3,
    #     fmt=None,
    #     figsize=None,
    #     title=None,
    #     cmap=None,
    # ):
    #     warnings.warn(
    #         "plot_grid() is deprecated, use plot.grid() instead", FutureWarning
    #     )
    #     return self.plot.grid(
    #         field=field,
    #         show_numbers=show_numbers,
    #         precision=precision,
    #         fmt=fmt,
    #         figsize=figsize,
    #         title=title,
    #         cmap=cmap,
    #     )

    def round(self, decimals=3):
        """round all values in dataframe

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to (default: 3). If decimals is negative, it specifies the number of positions to the left of the decimal point.
        """

        return self.__class__(self.df.round(decimals=decimals))

    def style(
        self,
        decimals=3,
        metrics=None,
        cmap="OrRd",
        show_best=True,
        **kwargs,
    ):
        """style dataframe with colors using pandas style

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to (default: 3).
        metrics : str or List[str], optional
            apply background gradient color to these columns, by default all;
            if columns is [] then no background gradient will be applied.
        cmap : str, optional
            colormap of background gradient, by default "OrRd",
            except "bias" column which will always be "coolwarm"
        show_best : bool, optional
            indicate best of each column by underline, by default True

        Returns
        -------
        pd.Styler
            Returns a pandas Styler object.

        Examples
        --------
        >>> s = comparer.skill()
        >>> s.style()
        >>> s.style(precision=1, metrics="rmse")
        >>> s.style(cmap="Blues", show_best=False)
        """
        # identity metric columns
        float_cols = list(self.df.select_dtypes(include="number").columns)

        if "precision" in kwargs:
            warnings.warn(
                FutureWarning(
                    "precision is deprecated, it has been renamed to decimals"
                )
            )
            decimals = kwargs["precision"]

        # selected columns
        if metrics is None:
            metrics = float_cols
        else:
            if isinstance(metrics, str):
                if not metrics:
                    metrics = []
                else:
                    metrics = [metrics]
            for column in metrics:
                if column not in float_cols:
                    raise ValueError(
                        f"Invalid column name {column} (must be one of {float_cols})"
                    )

        sdf = self.df.style.format(precision=decimals)

        # apply background gradient
        bg_cols = list(set(metrics) & set(float_cols))
        if "bias" in bg_cols:
            mm = self.df.bias.abs().max()
            sdf = sdf.background_gradient(
                subset=["bias"], cmap="coolwarm", vmin=-mm, vmax=mm
            )
            bg_cols.remove("bias")
        if "lin_slope" in bg_cols:
            mm = (self.df.lin_slope - 1).abs().max()
            sdf = sdf.background_gradient(
                subset=["lin_slope"], cmap="coolwarm", vmin=(1 - mm), vmax=(1 + mm)
            )
            bg_cols.remove("lin_slope")
        if len(bg_cols) > 0:
            cols = list(set(self.small_is_best_metrics) & set(bg_cols))
            sdf = sdf.background_gradient(subset=cols, cmap=cmap)

            cols = list(set(self.large_is_best_metrics) & set(bg_cols))
            cmap_r = self._reverse_colormap(cmap)
            sdf = sdf.background_gradient(subset=cols, cmap=cmap_r)

        if show_best:
            cols = list(set(self.large_is_best_metrics) & set(float_cols))
            sdf = sdf.apply(self._style_max, subset=cols)
            cols = list(set(self.small_is_best_metrics) & set(float_cols))
            sdf = sdf.apply(self._style_min, subset=cols)
            cols = list(set(self.one_is_best_metrics) & set(float_cols))
            sdf = sdf.apply(self._style_one_best, subset=cols)
            if "bias" in float_cols:
                sdf = sdf.apply(self._style_abs_min, subset=["bias"])

        return sdf

    def _reverse_colormap(self, cmap):
        cmap_r = cmap
        if isinstance(cmap, str):
            if cmap[-2:] == "_r":
                cmap_r = cmap_r[:-2]
            else:
                cmap_r = cmap + "_r"
        else:
            cmap_r = cmap.reversed()
        return cmap_r

    def _style_one_best(self, s):
        """Using underline-etc to highlight the best in a Series."""
        is_best = (s - 1.0).abs() == (s - 1.0).abs().min()
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in is_best]

    def _style_abs_min(self, s):
        """Using underline-etc to highlight the best in a Series."""
        is_best = s.abs() == s.abs().min()
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in is_best]

    def _style_min(self, s):
        """Using underline-etc to highlight the best in a Series."""
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in (s == s.min())]

    def _style_max(self, s):
        """Using underline-etc to highlight the best in a Series."""
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in (s == s.max())]
