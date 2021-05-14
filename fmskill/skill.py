import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
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

    def sort_values(self, by, **kwargs):
        return self.__class__(self.df.sort_values(by, **kwargs))


class AggregatedSkill(SkillDataFrame):
    @property
    def mod_names(self):
        return self._get_index_level_by_name("model")

    @property
    def obs_names(self):
        return self._get_index_level_by_name("observation")

    def _get_index_level_by_name(self, name):
        if name in self.index.names:
            level = self.index.names.index(name)
            return self.index.get_level_values(level).unique()
        else:
            return []
            # raise ValueError(f"name {name} not in index {list(self.index.names)}")

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

    def plot_line(self, field, level=-1, **kwargs):
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
            ax.set_xticklabels(xlabels)
        return axes

    def plot_bar(self, field, level=-1, **kwargs):
        if isinstance(self.index, pd.MultiIndex):
            df = self.df[field].unstack(level=level)
        else:
            df = self.df[field]
        if "title" not in kwargs:
            if isinstance(field, str):
                kwargs["title"] = field
        return df.plot.bar(**kwargs)

    def plot_grid(
        self, field, show_numbers=True, precision=3, figsize=None, title=None, cmap=None
    ):
        errors = self._validate_multi_index()
        if len(errors) > 0:
            warnings.warn("plot_grid: " + "\n".join(errors))
            return None
            # df = self.df[field]    TODO: at_least_2d...
        df = self.df[field].unstack()

        vmin = None
        vmax = None
        if cmap is None:
            cmap = "Reds"
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
        plt.gca().set_xticklabels(xlabels)
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
                    plt.text(
                        jj + 0.5,
                        ii + 0.5,
                        val,
                        ha="center",
                        va="center",
                        # size=15,
                        color=col,
                    )
        plt.title(title, fontsize=14)

    def style(self, precision=3, columns=None, cmap="Reds", background_gradient=True):
        float_list = ["float16", "float32", "float64"]
        float_cols = list(self.df.select_dtypes(include=float_list).columns.values)
        if columns is None:
            columns = float_cols
        else:
            if isinstance(columns, str):
                columns = [columns]
            for column in columns:
                if column not in float_cols:
                    raise ValueError(
                        f"Invalid column name {column} (must be one of {float_cols})"
                    )

        styled_df = self.df.style.set_precision(precision)

        #'mef', 'model_efficiency_factor', 'nash_sutcliffe_efficiency', 'nse',
        large_is_best_metrics = [
            "cc",
            "corrcoef",
            "r2",
            "spearmanr",
            "rho",
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
        ]
        one_is_best_metrics = ["lin_slope"]
        zero_is_best_metrics = ["bias"]

        bg_cols = list(set(columns) & set(float_cols))
        if "bias" in bg_cols:
            bg_cols.remove("bias")

        if background_gradient and (len(bg_cols) > 0):
            styled_df = styled_df.background_gradient(subset=bg_cols, cmap=cmap)
        if background_gradient and ("bias" in columns):
            mm = self.df.bias.abs().max()
            styled_df = styled_df.background_gradient(
                subset=["bias"], cmap="coolwarm", vmin=-mm, vmax=mm
            )

        cols = list(set(large_is_best_metrics) & set(float_cols))
        styled_df = styled_df.apply(self._style_max, subset=cols)
        cols = list(set(small_is_best_metrics) & set(float_cols))
        styled_df = styled_df.apply(self._style_min, subset=cols)
        cols = list(set(one_is_best_metrics) & set(float_cols))
        styled_df = styled_df.apply(self._style_one_best, subset=cols)
        # cols = list(set(zero_is_good_metrics) & set(float_cols.append("bias")))
        if "bias" in float_cols:
            styled_df = styled_df.apply(self._style_abs_min, subset=["bias"])
        # , subset=

        return styled_df

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
