import pandas as pd
from pandas.plotting import parallel_coordinates

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

    def plot_bar(self):
        pass

    def plot_grid(self):
        pass

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
