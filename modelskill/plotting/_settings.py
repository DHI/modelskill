import modelskill.settings as settings
from modelskill.settings import register_option


register_option("plot.scatter.points.size", 20, validator=settings.is_positive)
register_option("plot.scatter.points.alpha", 0.5, validator=settings.is_between_0_and_1)
register_option("plot.scatter.points.label", "", validator=settings.is_str)
register_option("plot.scatter.quantiles.marker", "X", validator=settings.is_str)
register_option(
    "plot.scatter.quantiles.markersize", 3.5, validator=settings.is_positive
)
register_option(
    "plot.scatter.quantiles.color",
    "darkturquoise",
    validator=settings.is_tuple_list_or_str,
)
register_option("plot.scatter.quantiles.label", "Q-Q", validator=settings.is_str)
register_option(
    "plot.scatter.quantiles.markeredgecolor",
    (0, 0, 0, 0.4),
    validator=settings.is_tuple_list_or_str,
)
register_option(
    "plot.scatter.quantiles.markeredgewidth", 0.5, validator=settings.is_positive
)
register_option("plot.scatter.quantiles.kwargs", {}, validator=settings.is_dict)
register_option("plot.scatter.oneone_line.label", "1:1", validator=settings.is_str)
register_option(
    "plot.scatter.oneone_line.color",
    "blue",
    validator=settings.is_tuple_list_or_str,
)
register_option("plot.scatter.legend.kwargs", {}, validator=settings.is_dict)
register_option(
    "plot.scatter.reg_line.kwargs", {"color": "r"}, validator=settings.is_dict
)
register_option(
    "plot.scatter.legend.bbox",
    {
        "facecolor": "white",
        "edgecolor": "lightgray",
        "linewidth": 1,
        "boxstyle": "round,pad=0.1",
        "alpha": 0.95,
    },
    validator=settings.is_dict,
)
# register_option("plot.scatter.table.show", False, validator=settings.is_bool)
register_option("plot.scatter.legend.fontsize", 12, validator=settings.is_positive)

# TODO: Auto-implement
# still requires plt.rcParams.update(modelskill.settings.get_option('plot.rcParams'))

# TODO does this work as intended? Mutable default values, e.g. dict, list, are usually not recommended
register_option(
    key="plot.rcParams", defval={}, validator=settings.is_dict
)  # still have to
