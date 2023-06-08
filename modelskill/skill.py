import sys
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd

# import matplotlib as mpl
from matplotlib import pyplot as plt

# from pandas.plotting import parallel_coordinates
# from typing import List, Union


class SkillDataFrame:
    def __init__(self, df):
        self.df = df

    def __repr__(self):
        return repr(self.df)

    def _repr_html_(self):
        return self.df._repr_html_()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, x):
        return self.df[x]

    @property
    def loc(self, *args, **kwargs):
        return self.df.loc(*args, **kwargs)

    @property
    def iloc(self, *args, **kwargs):
        return self.df.iloc(*args, **kwargs)

    @property
    def index(self):
        return self.df.index

    @property
    def columns(self):
        return self.df.columns

    @property
    def shape(self):
        return self.df.shape

    @property
    def size(self):
        return self.df.size

    @property
    def ndim(self):
        return self.df.ndim

    def to_html(self, *args, **kwargs):
        return self.df.to_html(*args, **kwargs)

    def to_markdown(self, *args, **kwargs):
        return self.df.to_markdown(*args, **kwargs)

    def to_excel(self, *args, **kwargs):
        return self.df.to_excel(*args, **kwargs)

    def to_csv(self, *args, **kwargs):
        return self.df.to_csv(*args, **kwargs)

    def to_dataframe(self, copy=True):
        if copy:
            return self.df.copy()
        else:
            return self.df

    def head(self, *args, **kwargs):
        return self.__class__(self.df.head(*args, **kwargs))

    def tail(self, *args, **kwargs):
        return self.__class__(self.df.tail(*args, **kwargs))

    def round(self, decimals, *args, **kwargs):
        return self.__class__(self.df.round(decimals))

    def sort_index(self, *args, **kwargs):
        return self.__class__(self.df.sort_index(*args, **kwargs))

    def sort_values(self, by, **kwargs):
        return self.__class__(self.df.sort_values(by, **kwargs))

    def query(self, expr, **kwargs):
        return self.__class__(self.df.query(expr, **kwargs))

    def xs(self, *args, **kwargs):
        return self.__class__(self.df.xs(*args, **kwargs))

    def reorder_levels(self, order, **kwargs):
        return self.__class__(self.df.reorder_levels(order, **kwargs))

    def swaplevel(self, *args, **kwargs):
        return self.__class__(self.df.swaplevel(*args, **kwargs))


class AggregatedSkill(SkillDataFrame):
    """
    AggregatedSkill object for visualization and analysis returned by
    the comparer's skill method. The object wraps the pd.DataFrame
    class which can be accessed from the attribute df.

    Examples
    --------
    >>> s = comparer.skill()
    >>> s.mod_names
    ['SW_1', 'SW_2']
    >>> s.style()
    >>> s.sel(model='SW_1').style()
    >>> s.plot_bar(field='rmse')
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

    @property
    def field_names(self):
        """List of field names (=dataframe columns)"""
        return list(self.df.columns)

    def _get_index_level_by_name(self, name):
        if name in self.index.names:
            level = self.index.names.index(name)
            return self.index.get_level_values(level).unique()
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
        """Select a subset of the AggregatedSkill by a query,
           (part of) the index, or specific columns

        Parameters
        ----------
        query : str, optional
            string supported by pd.DataFrame.query(), by default None
        reduce_index : bool, optional
            Should unnecessary levels of the index be removed after subsetting?
            Removed levels will stay as columns. By default True
        **kwargs : dict, optional
            "columns"=... to select specific columns,
            "model"=... to select specific models,
            "observation"=... to select specific observations, etc.

        Returns
        -------
        AggregatedSkill
            A subset of the orignal AggregatedSkill

        Examples
        --------
        >>> s = comparer.skill()
        >>> s.sel(query="rmse>0.3")
        >>> s.sel(model = "SW_1")
        >>> s.sel(observation = ["EPL", "HKNA"])
        >>> s.sel(columns="rmse")
        >>> s.sel("rmse>0.2", observation=[0, 2], columns=["n","rmse"])
        """
        df = self.df

        if query is not None:
            if isinstance(query, str):
                df = df.query(query)

        for key, value in kwargs.items():
            if key in df.index.names:
                df = self._sel_from_index(df, key, value)
            elif key == "columns":
                cols = [value] if isinstance(value, str) else value
                df = df[cols]
            else:
                raise KeyError(
                    f"Unknown index {key}. Valid index names are {df.index.names}"
                )

        if isinstance(df, pd.Series):
            df = df.to_frame()
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

    def parallel_coordinates(self):
        pass

    def _validate_multi_index(self, min_levels=2, max_levels=2):
        errors = []
        if isinstance(self.index, pd.MultiIndex):
            if len(self.index.levels) < min_levels:
                errors.append(
                    f"not possible for MultiIndex with fewer than {min_levels} levels"
                )
            if len(self.index.levels) > max_levels:
                errors.append(
                    f"not possible for MultiIndex with more than {max_levels} levels"
                )
        else:
            errors.append("only possible for MultiIndex skill objects")
        return errors

    def plot_line(self, field, level=0, **kwargs):
        """plot statistic as a lines using pd.DataFrame.plot.line()

        Primarily for MultiIndex skill objects, e.g. multiple models and multiple observations

        Parameters
        ----------
        field : str
            field (statistic) to plot e.g. "rmse"
        level : int or str, optional
            level to unstack, by default 0
        kwargs : dict, optional
            key word arguments to be pased to pd.DataFrame.plot.line()
            e.g. marker, title, figsize, ...

        Examples
        --------
        >>> s = comparer.skill()
        >>> s.plot_line("rmse")
        >>> s.plot_line("mae", marker="o", linestyle=':')
        >>> s.plot_line(field="bias", color=['0.2', '0.4', '0.6'])
        """
        if isinstance(self.index, pd.MultiIndex):
            df = self.df[field].unstack(level=level)
        else:
            df = self.df[field]
        if "title" not in kwargs:
            if isinstance(field, str):
                kwargs["title"] = field
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

    def plot_bar(self, field, level=0, **kwargs):
        """plot statistic as bar chart using pd.DataFrame.plot.bar()

        Parameters
        ----------
        field : str
            field (statistic) to plot e.g. "rmse"
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
        >>> s = comparer.skill()
        >>> s.plot_bar("rmse")
        >>> s.plot_bar("mae", level="observation")
        >>> s.plot_bar(field="si", title="scatter index")
        >>> s.plot_bar("si", color=["red","blue"])
        """
        if isinstance(self.index, pd.MultiIndex):
            df = self.df[field].unstack(level=level)
        else:
            df = self.df[field]
        if "title" not in kwargs:
            if isinstance(field, str):
                kwargs["title"] = field
        return df.plot.bar(**kwargs)

    def plot_barh(self, field, level=0, **kwargs):
        """plot statistic as horizontal bar chart using pd.DataFrame.plot.barh()

        Parameters
        ----------
        field : str
            field (statistic) to plot e.g. "rmse"
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
        >>> s = comparer.skill()
        >>> s.plot_barh("rmse")
        >>> s.plot_barh("mae", level="observation")
        >>> s.plot_barh(field="si", title="scatter index")
        """
        if isinstance(self.index, pd.MultiIndex):
            df = self.df[field].unstack(level=level)
        else:
            df = self.df[field]
        if "title" not in kwargs:
            if isinstance(field, str):
                kwargs["title"] = field
        return df.plot.barh(**kwargs)

    def plot_grid(
        self,
        field,
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
        field : str
            field (statistic) to plot e.g. "rmse"
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
        >>> s = comparer.skill()
        >>> s.plot_grid("rmse")
        >>> s.plot_grid("n", show_numbers=False, cmap="magma")
        >>> s.plot_grid(field="bias", precision=1)
        >>> s.plot_grid('si', fmt=".0%", title="scatter index")
        """
        errors = self._validate_multi_index()
        if len(errors) > 0:
            warnings.warn("plot_grid: " + "\n".join(errors))
            return None
            # df = self.df[field]    TODO: at_least_2d...
        df = self.df[field].unstack()

        vmin = None
        vmax = None
        if cmap is None:
            cmap = "OrRd"
            if field == "bias":
                cmap = "coolwarm"
                mm = self.df.bias.abs().max()
                vmin = -mm
                vmax = mm
        if title is None:
            title = field
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
                    if field == "bias":
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

    def style(
        self,
        precision=3,
        columns=None,
        cmap="OrRd",
        show_best=True,
    ):
        """style dataframe with colors using pandas style

        Parameters
        ----------
        precision : int, optional
            number of decimals, by default 3
        columns : str or List[str], optional
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
        >>> s.style(precision=1, columns="rmse")
        >>> s.style(cmap="Blues", show_best=False)
        """
        # identity metric columns
        float_list = ["float16", "float32", "float64"]
        float_cols = list(self.df.select_dtypes(include=float_list).columns.values)

        # selected columns
        if columns is None:
            columns = float_cols
        else:
            if isinstance(columns, str):
                if not columns:
                    columns = []
                else:
                    columns = [columns]
            for column in columns:
                if column not in float_cols:
                    raise ValueError(
                        f"Invalid column name {column} (must be one of {float_cols})"
                    )

        sdf = (
            self.df.style.format(precision=precision)
            if sys.version_info >= (3, 7)
            else self.df.style.set_precision(precision)
        )

        # apply background gradient
        bg_cols = list(set(columns) & set(float_cols))
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
