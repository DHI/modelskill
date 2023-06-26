from typing import List, Union

import matplotlib.pyplot as plt


from ._utils import _get_id
from ..plot import colors, scatter


class ComparerCollectionPlotter:
    def __init__(self, cc) -> None:
        self.cc = cc

    def scatter(
        self,
        *,
        model=None,
        bins: Union[int, float, List[int], List[float]] = 20,
        quantiles: Union[int, List[float]] = None,
        fit_to_quantiles: bool = False,
        show_points: Union[bool, int, float] = None,
        show_hist: bool = None,
        show_density: bool = None,
        backend: str = "matplotlib",
        figsize: List[float] = (8, 8),
        xlim: List[float] = None,
        ylim: List[float] = None,
        reg_method: str = "ols",
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        skill_table: Union[str, List[str], bool] = None,
        **kwargs,
    ):
        """Scatter plot showing compared data: observation vs modelled
        Optionally, with density histogram.

        Parameters
        ----------
        bins: (int, float, sequence), optional
            bins for the 2D histogram on the background. By default 20 bins.
            if int, represents the number of bins of 2D
            if float, represents the bin size
            if sequence (list of int or float), represents the bin edges
        quantiles: (int, sequence), optional
            number of quantiles for QQ-plot, by default None and will depend on the scatter data length (10, 100 or 1000)
            if int, this is the number of points
            if sequence (list of floats), represents the desired quantiles (from 0 to 1)
        fit_to_quantiles: bool, optional, by default False
            by default the regression line is fitted to all data, if True, it is fitted to the quantiles
            which can be useful to represent the extremes of the distribution
        show_points : (bool, int, float), optional
            Should the scatter points be displayed?
            None means: show all points if fewer than 1e4, otherwise show 1e4 sample points, by default None.
            float: fraction of points to show on plot from 0 to 1. eg 0.5 shows 50% of the points.
            int: if 'n' (int) given, then 'n' points will be displayed, randomly selected
        show_hist : bool, optional
            show the data density as a a 2d histogram, by default None
        show_density: bool, optional
            show the data density as a colormap of the scatter, by default None. If both `show_density` and `show_hist`
        are None, then `show_density` is used by default.
            for binning the data, the previous kword `bins=Float` is used
        backend : str, optional
            use "plotly" (interactive) or "matplotlib" backend, by default "matplotlib"
        figsize : tuple, optional
            width and height of the figure, by default (8, 8)
        xlim : tuple, optional
            plot range for the observation (xmin, xmax), by default None
        ylim : tuple, optional
            plot range for the model (ymin, ymax), by default None
        reg_method : str, optional
            method for determining the regression line
            "ols" : ordinary least squares regression
            "odr" : orthogonal distance regression,
            by default "ols"
        title : str, optional
            plot title, by default None
        xlabel : str, optional
            x-label text on plot, by default None
        ylabel : str, optional
            y-label text on plot, by default None
        skill_table : str, List[str], bool, optional
            list of modelskill.metrics or boolean, if True then by default modelskill.options.metrics.list.
            This kword adds a box at the right of the scatter plot,
            by default False
        kwargs

        Examples
        ------
        >>> cc.plot.scatter()
        >>> cc.plot.scatter(bins=0.2, backend='plotly')
        >>> cc.plot.scatter(show_points=False, title='no points')
        >>> cc.plot.scatter(xlabel='all observations', ylabel='my model')
        >>> cc.plot.scatter(model='HKZN_v2', figsize=(10, 10))
        >>> cc.plot.scatter(observations=['c2','HKNA'])
        """

        # select model
        mod_id = _get_id(model, self.cc.mod_names)
        mod_name = self.cc.mod_names[mod_id]

        cmp = self.cc

        if cmp.n_points == 0:
            raise ValueError("No data found in selection")

        df = cmp.to_dataframe()
        x = df.obs_val
        y = df.mod_val

        unit_text = self.cc[df.observation[0]]._unit_text

        xlabel = xlabel or f"Observation, {unit_text}"
        ylabel = ylabel or f"Model, {unit_text}"
        title = title or f"{mod_name} vs {cmp.name}"

        skill_df = None
        units = None
        if skill_table:
            metrics = None if skill_table is True else skill_table

            # TODO why is this here?
            if isinstance(self, ComparerCollectionPlotter) and cmp.n_observations == 1:
                skill_df = cmp.skill(metrics=metrics)
            else:
                skill_df = cmp.mean_skill(metrics=metrics)
            # TODO improve this
            try:
                units = unit_text.split("[")[1].split("]")[0]
            except IndexError:
                units = ""  # Dimensionless

        ax = scatter(
            x=x,
            y=y,
            bins=bins,
            quantiles=quantiles,
            fit_to_quantiles=fit_to_quantiles,
            show_points=show_points,
            show_hist=show_hist,
            show_density=show_density,
            backend=backend,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            reg_method=reg_method,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            skill_df=skill_df,
            units=units,
            # **kwargs, # TODO
        )
        return ax
