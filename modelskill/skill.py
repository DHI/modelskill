from __future__ import annotations
import warnings
from typing import Any, Iterable, Collection, overload, Hashable, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap

from .plotting._misc import _get_fig_ax


# TODO remove ?
def _validate_multi_index(index, min_levels=2, max_levels=2):  # type: ignore
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
    """SkillArrayPlotter object for visualization of a single metric (SkillArray)

    plot.line() : line plot
    plot.bar() : bar chart
    plot.barh() : horizontal bar chart
    plot.grid() : colored grid
    """

    def __init__(self, skillarray: "SkillArray") -> None:
        self.skillarray = skillarray

    def _name_to_title_in_kwargs(self, kwargs: Any) -> None:
        if "title" not in kwargs:
            if self.skillarray.name is not None:
                kwargs["title"] = self.skillarray.name

    def _get_plot_df(self, level: int | str = 0) -> pd.DataFrame:
        ser = self.skillarray._ser
        if isinstance(ser.index, pd.MultiIndex):
            df = ser.unstack(level=level)
        else:
            df = ser.to_frame()
        return df

    # TODO hide this for now until we are certain about the API
    # def map(self, **kwargs):
    #     if "model" in self.skillarray.data.index.names:
    #         n_models = len(self.skillarray.data.reset_index().model.unique())
    #         if n_models > 1:
    #             raise ValueError(
    #                 "map() is only possible for single model skill. Use .sel(model=...) to select a single model."
    #             )

    #     gdf = self.skillarray.to_geodataframe()
    #     column = self.skillarray.name
    #     kwargs = {"marker_kwds": {"radius": 10}} | kwargs

    #     return gdf.explore(column=column, **kwargs)

    def line(
        self,
        level: int | str = 0,
        **kwargs: Any,
    ) -> Axes:
        """Plot statistic as a lines using pd.DataFrame.plot.line()

        Primarily for MultiIndex skill objects, e.g. multiple models and multiple observations

        Parameters
        ----------
        level : int or str, optional
            level to unstack, by default 0
        **kwargs
            key word arguments to be pased to pd.DataFrame.plot.line()
            e.g. marker, title, figsize, ...

        Examples
        --------
        >>> sk = cc.skill()["rmse"]
        >>> sk.plot.line()
        >>> sk.plot.line(marker="o", linestyle=':')
        >>> sk.plot.line(color=['0.2', '0.4', '0.6'])
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

    def bar(self, level: int | str = 0, **kwargs: Any) -> Axes:
        """Plot statistic as bar chart using pd.DataFrame.plot.bar()

        Parameters
        ----------
        level : int or str, optional
            level to unstack, by default 0
        **kwargs
            key word arguments to be pased to pd.DataFrame.plot.bar()
            e.g. color, title, figsize, ...

        Returns
        -------
        AxesSubplot

        Examples
        --------
        >>> sk = cc.skill()["rmse"]
        >>> sk.plot.bar()
        >>> sk.plot.bar(level="observation")
        >>> sk.plot.bar(title="Root Mean Squared Error")
        >>> sk.plot.bar(color=["red","blue"])
        """
        df = self._get_plot_df(level=level)
        self._name_to_title_in_kwargs(kwargs)
        return df.plot.bar(**kwargs)

    def barh(self, level: int | str = 0, **kwargs: Any) -> Axes:
        """Plot statistic as horizontal bar chart using pd.DataFrame.plot.barh()

        Parameters
        ----------
        level : int or str, optional
            level to unstack, by default 0
        **kwargs
            key word arguments to be passed to pd.DataFrame.plot.barh()
            e.g. color, title, figsize, ...

        Returns
        -------
        AxesSubplot

        Examples
        --------
        >>> sk = cc.skill()["rmse"]
        >>> sk.plot.barh()
        >>> sk.plot.barh(level="observation")
        >>> sk.plot.barh(title="Root Mean Squared Error")
        """
        df = self._get_plot_df(level)
        self._name_to_title_in_kwargs(kwargs)
        return df.plot.barh(**kwargs)

    def grid(
        self,
        show_numbers: bool = True,
        precision: int = 3,
        fmt: str | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        cmap: str | Colormap | None = None,
    ) -> Axes | None:
        """Plot statistic as a colored grid, optionally with values in the cells.

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
        ax : Axes, optional
            matplotlib axes, by default None
        figsize : Tuple(float, float), optional
            figure size, by default None
        title : str, optional
            plot title, by default name of statistic
        cmap : str, optional
            colormap, by default "OrRd" ("coolwarm" if bias)

        Returns
        -------
        AxesSubplot

        Examples
        --------
        >>> sk = cc.skill()["rmse"]
        >>> sk.plot.grid()
        >>> sk.plot.grid(show_numbers=False, cmap="magma")
        >>> sk.plot.grid(precision=1)
        >>> sk.plot.grid(fmt=".0%", title="Root Mean Squared Error")
        """

        s = self.skillarray
        ser = s._ser

        errors = _validate_multi_index(ser.index)  # type: ignore
        if len(errors) > 0:
            warnings.warn("plot_grid: " + "\n".join(errors))
            # TODO raise error?
            return None
            # df = self.df[field]    TODO: at_least_2d...
        df = ser.unstack()

        vmin = None
        vmax = None
        if cmap is None:
            cmap = "OrRd"
            if s.name == "bias":
                cmap = "coolwarm"
                mm = ser.abs().max()
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
        fig, ax = _get_fig_ax(ax, figsize)
        assert ax is not None
        pcm = ax.pcolormesh(df, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(nx) + 0.5)
        ax.set_xticklabels(xlabels, rotation=90)
        ax.set_yticks(np.arange(ny) + 0.5)
        ax.set_yticklabels(ylabels)
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
                    ax.text(
                        jj + 0.5,
                        ii + 0.5,
                        val,
                        ha="center",
                        va="center",
                        # size=15,
                        color=col,
                    )
        else:
            fig.colorbar(pcm, ax=ax)
        ax.set_title(title, fontsize=14)
        return ax


class DeprecatedSkillPlotter:
    def __init__(self, skilltable):  # type: ignore
        self.skilltable = skilltable

    @staticmethod
    def _deprecated_warning(method, field):  # type: ignore
        warnings.warn(
            f"Selecting metric in plot functions like modelskill.skill().plot.{method}({field}) is deprecated and will be removed in a future version. Use modelskill.skill()['{field}'].plot.{method}() instead.",
            FutureWarning,
        )

    def line(self, field, **kwargs):  # type: ignore
        self._deprecated_warning("line", field)  # type: ignore
        return self.skilltable[field].plot.line(**kwargs)

    def bar(self, field, **kwargs):  # type: ignore
        self._deprecated_warning("bar", field)  # type: ignore
        return self.skilltable[field].plot.bar(**kwargs)

    def barh(self, field, **kwargs):  # type: ignore
        self._deprecated_warning("barh", field)  # type: ignore
        return self.skilltable[field].plot.barh(**kwargs)

    def grid(self, field, **kwargs):  # type: ignore
        self._deprecated_warning("grid", field)  # type: ignore
        return self.skilltable[field].plot.grid(**kwargs)


class SkillArray:
    """SkillArray object for visualization obtained by
    selecting a single metric from a SkillTable.

    Examples
    --------
    >>> sk = cc.skill()   # SkillTable
    >>> sk.rmse           # SkillArray
    >>> sk.rmse.plot.line()
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self._ser = data.iloc[:, -1]  # last column is the metric
        
        self.plot = SkillArrayPlotter(self)
        """Plot using the SkillArrayPlotter

        Examples
        --------
        >>> sk.rmse.plot.line()
        >>> sk.rmse.plot.bar()
        >>> sk.rmse.plot.barh()
        >>> sk.rmse.plot.grid()
        """

    def to_dataframe(self, drop_xy: bool = True) -> pd.DataFrame:
        """Convert SkillArray to pd.DataFrame

        Parameters
        ----------
        drop_xy : bool, optional
            Drop the x, y coordinates?, by default True

        Returns
        -------
        pd.DataFrame
            Skill data as pd.DataFrame
        """
        if drop_xy:
            return self._ser.to_frame()
        else:
            return self.data.copy()

    def __repr__(self) -> str:
        return repr(self.to_dataframe())

    def _repr_html_(self) -> Any:
        return self.to_dataframe()._repr_html_()

    @property
    def name(self) -> Any:
        """Name of the metric"""
        return self._ser.name

    def to_geodataframe(self, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """Convert SkillArray to geopandas.GeoDataFrame

        Note: requires geopandas to be installed

        Note: requires x and y columns to be present

        Parameters
        ----------
        crs : str, optional
            Coordinate reference system identifier passed to the
            GeoDataFrame constructor, by default "EPSG:4326"

        Returns
        -------
        gpd.GeoDataFrame
            Skill data as GeoDataFrame
        """
        import geopandas as gpd

        assert "x" in self.data.columns
        assert "y" in self.data.columns

        gdf = gpd.GeoDataFrame(
            self._ser,
            geometry=gpd.points_from_xy(self.data.x, self.data.y),
            crs=crs,
        )

        return gdf


class SkillTable:
    """
    SkillTable object for visualization and analysis returned by
    the comparer's `skill` method. The object wraps the pd.DataFrame
    class which can be accessed from the attribute `data`.

    The columns are assumed to be metrics and data for a single metric
    can be accessed by e.g. `s.rmse` or `s["rmse"]`. The resulting object
    can be used for plotting.

    Examples
    --------
    >>> sk = cc.skill()
    >>> sk.mod_names
    ['SW_1', 'SW_2']
    >>> sk.style()
    >>> sk.sel(model='SW_1').style()
    >>> sk.rmse.plot.bar()
    """

    _large_is_best_metrics = [
        "cc",
        "corrcoef",
        "r2",
        "spearmanr",
        "rho",
        "nash_sutcliffe_efficiency",
        "nse",
        "kge",
    ]
    _small_is_best_metrics = [
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
    _one_is_best_metrics = ["lin_slope"]
    _zero_is_best_metrics = ["bias"]

    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = (
            data if isinstance(data, pd.DataFrame) else data.to_dataframe()
        )
        # TODO remove in v1.1
        self.plot = DeprecatedSkillPlotter(self)  # type: ignore

    # TODO: remove?
    @property
    def _df(self) -> pd.DataFrame:
        """Data as DataFrame without x and y columns"""
        return self.to_dataframe(drop_xy=True)

    @property
    def metrics(self) -> Collection[str]:
        """List of metrics (columns) in the SkillTable"""
        return list(self._df.columns)

    # TODO: remove?
    def __len__(self) -> int:
        return len(self._df)

    def to_dataframe(self, drop_xy: bool = True) -> pd.DataFrame:
        """Convert SkillTable to pd.DataFrame

        Parameters
        ----------
        drop_xy : bool, optional
            Drop the x, y coordinates?, by default True

        Returns
        -------
        pd.DataFrame
            Skill data as pd.DataFrame
        """
        if drop_xy:
            return self.data.drop(columns=["x", "y"], errors="ignore")
        else:
            return self.data.copy()

    def to_geodataframe(self, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """Convert SkillTable to geopandas.GeoDataFrame

        Note: requires geopandas to be installed

        Note: requires x and y columns to be present

        Parameters
        ----------
        crs : str, optional
            Coordinate reference system identifier passed to the
            GeoDataFrame constructor, by default "EPSG:4326"

        Returns
        -------
        gpd.GeoDataFrame
            Skill data as GeoDataFrame
        """
        import geopandas as gpd

        assert "x" in self.data.columns
        assert "y" in self.data.columns

        df = self.to_dataframe(drop_xy=False)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.x, df.y),
            crs=crs,
        )

        return gdf

    def __repr__(self) -> str:
        return repr(self._df)

    def _repr_html_(self) -> Any:
        return self._df._repr_html_()

    @overload
    def __getitem__(self, key: Hashable | int) -> SkillArray:
        ...

    @overload
    def __getitem__(self, key: Iterable[Hashable]) -> SkillTable:
        ...

    def __getitem__(
        self, key: Hashable | Iterable[Hashable]
    ) -> SkillArray | SkillTable:
        if isinstance(key, int):
            key = list(self.data.columns)[key]
        result = self.data[key]
        if isinstance(result, pd.Series):
            # I don't think this should be necessary, but in some cases the input doesn't contain x and y
            if "x" in self.data.columns and "y" in self.data.columns:
                cols = ["x", "y", key]
                return SkillArray(self.data[cols])
            else:
                return SkillArray(result.to_frame())
        elif isinstance(result, pd.DataFrame):
            return SkillTable(result)
        else:
            raise NotImplementedError("Unexpected type of result")

    def __getattr__(self, item: str, *args, **kwargs) -> Any:
        # note: no help from type hints here!
        if item in self.data.columns:
            return self[item]  # Redirects to __getitem__
        else:
            # act as a DataFrame... (necessary for style() to work)
            # drawback is that methods such as head() etc would appear
            # as working but return a DataFrame instead of a SkillTable!
            return getattr(self.data, item, *args, **kwargs)
            # raise AttributeError(
            #     f"""
            #         SkillTable has no attribute {item}; Maybe you are
            #         looking for the corresponding DataFrame attribute?
            #         Try exporting the skill table to a DataFrame using sk.to_dataframe().
            #     """
            # )

    @property
    def iloc(self, *args, **kwargs):  # type: ignore
        return self.data.iloc(*args, **kwargs)

    @property
    def loc(self, *args, **kwargs):  # type: ignore
        return self.data.loc(*args, **kwargs)        

    def sort_index(self, *args, **kwargs) -> SkillTable:  # type: ignore
        """Sort by index (level) e.g. sorting by observation

        Wrapping pd.DataFrame.sort_index()

        Returns
        -------
        SkillTable
            A new SkillTable with sorted index

        Examples
        --------
        >>> sk = cc.skill()
        >>> sk.sort_index()
        >>> sk.sort_index(level="observation")
        """
        return self.__class__(self.data.sort_index(*args, **kwargs))

    def sort_values(self, *args, **kwargs) -> SkillTable:  # type: ignore
        """Sort by values e.g. sorting by rmse values

        Wrapping pd.DataFrame.sort_values()

        Returns
        -------
        SkillTable
            A new SkillTable with sorted values

        Examples
        --------
        >>> sk = cc.skill()
        >>> sk.sort_values("rmse")
        >>> sk.sort_values("rmse", ascending=False)
        >>> sk.sort_values(["n", "rmse"])
        """
        return self.__class__(self.data.sort_values(*args, **kwargs))

    def swaplevel(self, *args, **kwargs) -> SkillTable:  # type: ignore
        """Swap the levels of the MultiIndex e.g. swapping 'model' and 'observation'

        Wrapping pd.DataFrame.swaplevel()

        Returns
        -------
        SkillTable
            A new SkillTable with swapped levels

        Examples
        --------
        >>> sk = cc.skill()
        >>> sk.swaplevel().sort_index(level="observation")
        >>> sk.swaplevel("model", "observation")
        >>> sk.swaplevel(0, 1)
        """
        return self.__class__(self.data.swaplevel(*args, **kwargs))

    @property
    def mod_names(self) -> list[str]:
        """List of model names (in index)"""
        return self._get_index_level_by_name("model")

    @property
    def obs_names(self) -> list[str]:
        """List of observation names (in index)"""
        return self._get_index_level_by_name("observation")

    @property
    def quantity_names(self) -> list[str]:
        """List of quantity names (in index)"""
        return self._get_index_level_by_name("quantity")

    def _get_index_level_by_name(self, name: str) -> list[str]:
        # Helper function to get unique values of a level in the index (e.g. model)
        index = self._df.index
        if name in index.names:
            level = index.names.index(name)
            return list(index.get_level_values(level).unique())
        else:
            return []
            # raise ValueError(f"name {name} not in index {list(self.index.names)}")

    def query(self, query: str) -> SkillTable:
        """Select a subset of the SkillTable by a query string

        wrapping pd.DataFrame.query()

        Parameters
        ----------
        query : str
            string supported by pd.DataFrame.query()

        Returns
        -------
        SkillTable
            A subset of the original SkillTable

        Examples
        --------
        >>> sk = cc.skill()
        >>> sk_above_0p3 = sk.query("rmse>0.3")
        """
        return self.__class__(self.data.query(query))

    def sel(
        self, query: str | None = None, reduce_index: bool = True, **kwargs: Any
    ) -> SkillTable | SkillArray:
        """Select a subset of the SkillTable by a query,
           (part of) the index, or specific columns

        Parameters
        ----------
        reduce_index : bool, optional
            Should unnecessary levels of the index be removed after subsetting?
            Removed levels will stay as columns. By default True
        **kwargs
            Concrete keys depend on the index names of the SkillTable
            (from the "by" argument in cc.skill() method)
            "model"=... to select specific models,
            "observation"=... to select specific observations

        Returns
        -------
        SkillTable
            A subset of the original SkillTable

        Examples
        --------
        >>> sk = cc.skill()
        >>> sk_SW1 = sk.sel(model = "SW_1")
        >>> sk2 = sk.sel(observation = ["EPL", "HKNA"])
        """
        if query is not None:
            warnings.warn(
                "s.sel(query=...) is deprecated, use s.query(...) instead",
                FutureWarning,
            )
            return self.query(query)

        for key, value in kwargs.items():
            if key == "metrics" or key == "columns":
                warnings.warn(
                    f"s.sel({key}=...) is deprecated, use getitem s[...] instead",
                    FutureWarning,
                )
                return self[value]  # type: ignore

        df = self.to_dataframe(drop_xy=False)

        for key, value in kwargs.items():
            if key in df.index.names:
                df = self._sel_from_index(df, key, value)
            else:
                raise KeyError(
                    f"Unknown index {key}. Valid index names are {df.index.names}"
                )

        if isinstance(df, pd.Series):
            return SkillArray(df)
        if reduce_index and isinstance(df.index, pd.MultiIndex):
            df = self._reduce_index(df)
        return self.__class__(df)

    def _sel_from_index(
        self, df: pd.DataFrame, key: str, value: str | int
    ) -> pd.DataFrame:
        if (not isinstance(value, str)) and isinstance(value, Iterable):
            for i, v in enumerate(value):
                dfi = self._sel_from_index(df, key, v)
                if i == 0:
                    dfout = dfi
                else:
                    dfout = pd.concat([dfout, dfi])
            return dfout

        if isinstance(value, int):
            value = self._idx_to_name(key, value)

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(value, level=key, drop_level=False)
        else:
            df = df[df.index == value]  # .copy()
        return df

    def _idx_to_name(self, index_name: str, pos: int) -> str:
        """Assumes that index is valid and idx is int"""
        names = self._get_index_level_by_name(index_name)
        n = len(names)
        if (pos < 0) or (pos >= n):
            raise KeyError(f"Id {pos} is out of bounds for index {index_name} (0, {n})")
        return names[pos]

    def _reduce_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unnecessary levels of MultiIndex"""
        df.index = df.index.remove_unused_levels()
        levels_to_reset = []
        for j, level in enumerate(df.index.levels):
            if len(level) == 1:
                levels_to_reset.append(j)
        return df.reset_index(level=levels_to_reset)

    def round(self, decimals: int = 3) -> SkillTable:
        """Round all values in SkillTable

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to (default: 3).
            If decimals is negative, it specifies the number of
            positions to the left of the decimal point.

        Returns
        -------
        SkillTable
            A new SkillTable with rounded values
        """

        return self.__class__(self.data.round(decimals=decimals))

    def style(
        self,
        decimals: int = 3,
        metrics: Iterable[str] | None = None,
        cmap: str = "OrRd",
        show_best: bool = True,
        **kwargs: Any,
    ) -> pd.io.formats.style.Styler:
        """Style SkillTable with colors using pandas style

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
        >>> sk = cc.skill()
        >>> sk.style()
        >>> sk.style(precision=1, metrics="rmse")
        >>> sk.style(cmap="Blues", show_best=False)
        """
        # identity metric columns
        float_cols = list(self._df.select_dtypes(include="number").columns)

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

        sdf = self._df.style.format(precision=decimals)

        # apply background gradient
        bg_cols = list(set(metrics) & set(float_cols))
        if "bias" in bg_cols:
            mm = self._df.bias.abs().max()
            sdf = sdf.background_gradient(
                subset=["bias"], cmap="coolwarm", vmin=-mm, vmax=mm
            )
            bg_cols.remove("bias")
        if "lin_slope" in bg_cols:
            mm = (self._df.lin_slope - 1).abs().max()
            sdf = sdf.background_gradient(
                subset=["lin_slope"], cmap="coolwarm", vmin=(1 - mm), vmax=(1 + mm)
            )
            bg_cols.remove("lin_slope")
        if len(bg_cols) > 0:
            cols = list(set(self._small_is_best_metrics) & set(bg_cols))
            sdf = sdf.background_gradient(subset=cols, cmap=cmap)

            cols = list(set(self._large_is_best_metrics) & set(bg_cols))
            cmap_r = self._reverse_colormap(cmap)  # type: ignore
            sdf = sdf.background_gradient(subset=cols, cmap=cmap_r)

        if show_best:
            cols = list(set(self._large_is_best_metrics) & set(float_cols))
            sdf = sdf.apply(self._style_max, subset=cols)
            cols = list(set(self._small_is_best_metrics) & set(float_cols))
            sdf = sdf.apply(self._style_min, subset=cols)
            cols = list(set(self._one_is_best_metrics) & set(float_cols))
            sdf = sdf.apply(self._style_one_best, subset=cols)
            if "bias" in float_cols:
                sdf = sdf.apply(self._style_abs_min, subset=["bias"])

        return sdf

    def _reverse_colormap(self, cmap):  # type: ignore
        cmap_r = cmap
        if isinstance(cmap, str):
            if cmap[-2:] == "_r":
                cmap_r = cmap_r[:-2]
            else:
                cmap_r = cmap + "_r"
        else:
            cmap_r = cmap.reversed()
        return cmap_r

    def _style_one_best(self, s: pd.Series) -> list[str]:
        """Using underline-etc to highlight the best in a Series."""
        is_best = (s - 1.0).abs() == (s - 1.0).abs().min()
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in is_best]

    def _style_abs_min(self, s: pd.Series) -> list[str]:
        """Using underline-etc to highlight the best in a Series."""
        is_best = s.abs() == s.abs().min()
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in is_best]

    def _style_min(self, s: pd.Series) -> list[str]:
        """Using underline-etc to highlight the best in a Series."""
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in (s == s.min())]

    def _style_max(self, s: pd.Series) -> list[str]:
        """Using underline-etc to highlight the best in a Series."""
        cell_style = (
            "text-decoration: underline; font-style: italic; font-weight: bold;"
        )
        return [cell_style if v else "" for v in (s == s.max())]

    # =============== Deprecated methods ===============

    # TODO: remove plot_* methods in v1.1; warnings are not needed
    # as the refering method is also deprecated
    def plot_line(self, **kwargs):  # type: ignore
        return self.plot.line(**kwargs)  # type: ignore

    def plot_bar(self, **kwargs):  # type: ignore
        return self.plot.bar(**kwargs)  # type: ignore

    def plot_barh(self, **kwargs):  # type: ignore
        return self.plot.barh(**kwargs)  # type: ignore

    def plot_grid(self, **kwargs):  # type: ignore
        return self.plot.grid(**kwargs)  # type: ignore
