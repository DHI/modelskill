import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

# from pandas.plotting import parallel_coordinates

# import numpy as np
# import warnings
# from typing import List, Union
# from IPython.display import display


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
    def ndims(self):
        return self.df.ndims

    def to_html(self, *args, **kwargs):
        return self.df.to_html(*args, **kwargs)

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

    def sort_index(self, **kwargs):
        return self.__class__(self.df.sort_index(**kwargs))

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
        """List of model names"""
        return self._get_index_level_by_name("model")

    @property
    def obs_names(self):
        """List of observation names"""
        return self._get_index_level_by_name("observation")

    @property
    def var_names(self):
        """List of variable names"""
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

    # model=None, observation=None, variable=None,
    def sel(self, query=None, **kwargs):
        df = self.df
        # if model is not None:
        #     df = df.xs(model, level="model")
        # if observation is not None:
        #     df = df.xs(observation, level="observation")
        # if variable is not None:
        #     df = df.xs(variable, level="variable")
        for key, value in kwargs.items():
            if key in df.index.names:
                # TODO: if value is int: lookup name in self.mod_names ...
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(value, level=key)
                else:
                    df = df.loc[value].copy()
            else:
                raise ValueError(
                    f"Unknown index {key}. Valid index names are {df.index.names}"
                )
        if query is not None:
            if isinstance(query, str):
                df = df.query(query)

        if isinstance(df, pd.Series):
            df = df.to_frame()
        return self.__class__(df)  # .squeeze()

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
        >>> s.plot_line("mae", marker="o")
        >>> s.plot_line(field="bias", precision=1)
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
        """
        if isinstance(self.index, pd.MultiIndex):
            df = self.df[field].unstack(level=level)
        else:
            df = self.df[field]
        if "title" not in kwargs:
            if isinstance(field, str):
                kwargs["title"] = field
        return df.plot.bar(**kwargs)

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
            format string, e.g. "{:.0%}" to show value as percentage
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
        >>> s.plot_grid('si', fmt="{:.0%}", title="scatter index")
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

        if figsize is None:
            figsize = (nx, ny)  # (nx * ((4 + precision) / 7), ny * 0.7)
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

        sdf = self.df.style.set_precision(precision)

        # apply background gradient
        bg_cols = list(set(columns) & set(float_cols))
        if "bias" in bg_cols:
            mm = self.df.bias.abs().max()
            sdf = sdf.background_gradient(
                subset=["bias"], cmap="coolwarm", vmin=-mm, vmax=mm
            )
            bg_cols.remove("bias")
        if len(bg_cols) > 0:
            cols = list(set(self.small_is_best_metrics) & set(bg_cols))
            sdf = sdf.background_gradient(subset=cols, cmap=cmap)

            cols = list(set(self.large_is_best_metrics) & set(bg_cols))
            cmap_r = self._reverse_colormap(cmap)
            sdf = sdf.background_gradient(subset=cols, cmap=cmap_r)

            # TODO: lin_slope is not implemented!

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
        """Using blod-face to highlight the best in a Series."""
        is_best = (s - 1.0).abs() == (s - 1.0).abs().min()
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in is_best]

    def _style_abs_min(self, s):
        """Using blod-face to highlight the best in a Series."""
        is_best = s.abs() == s.abs().min()
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in is_best]

    def _style_min(self, s):
        """Using blod-face to highlight the best in a Series."""
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in (s == s.min())]

    def _style_max(self, s):
        """Using blod-face to highlight the best in a Series."""
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in (s == s.max())]
        # font-weight: bold; font-style: oblique; border-style: solid; border-color: #212121; border-style: solid; border-width: thin;

    def target_diagram(self):
        raise NotImplementedError()
