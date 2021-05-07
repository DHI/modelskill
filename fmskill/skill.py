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

    # def index(self):
    #     return self.df.index()

    def round(self, precision):
        return self.df.round(precision)

    def style(self):
        raise NotImplementedError()

    def taylor_diagram(self):
        raise NotImplementedError()

    def target_diagram(self):
        raise NotImplementedError()