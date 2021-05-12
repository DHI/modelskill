import pandas as pd

# import numpy as np
# import warnings
# from typing import List, Union
# from IPython.display import display


class BaseSkill(pd.DataFrame):
    # ALTERNATIVE approach: where the class inherits from DataFrame
    # instead of holding a DataFrame as an attribute.

    # https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
    # https://stackoverflow.com/questions/47466255/subclassing-a-pandas-dataframe-updates
    # This class variable tells Pandas the name of the attributes
    # that are to be ported over to derivative DataFrames.  There
    # is a method named `__finalize__` that grabs these attributes
    # and assigns them to newly created `BaseSkill`
    _metadata = ["my_attr"]

    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `BaseSkill` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `my_attr` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finalize__` every time."""

        def _c(*args, **kwargs):
            return BaseSkill(*args, **kwargs).__finalize__(self)

        return _c

    def __init__(self, *args, **kwargs):
        # grab the keyword argument that is supposed to be my_attr
        self.my_attr = kwargs.pop("my_attr", None)
        super().__init__(*args, **kwargs)

    # def my_method(self, other):
    #     return self * np.sign(self - other)


class AggregatedSkill:
    def __init__(self, df):
        # super().__init__()
        self.df = df

    def __repr__(self):
        #     display(self.df)
        return repr(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, x):
        # if isinstance(x, int):
        #     x = self._get_obs_name(x)
        return self.df[x]

    def _repr_html_(self):
        return self.df._repr_html_()

    def index(self):
        return self.df.index()

    def head(self, **kwargs):
        return AggregatedSkill(self.df.head(**kwargs))

    def round(self, precision):
        return AggregatedSkill(self.df.round(precision))

    def sort_values(self, field, **kwargs):
        return AggregatedSkill(self.df.sort_values(field, **kwargs))

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
            print("best bias")
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

    def taylor_diagram(self):
        raise NotImplementedError()

    def target_diagram(self):
        raise NotImplementedError()

    def to_html(self, **kwargs):
        return self.df.to_html(**kwargs)