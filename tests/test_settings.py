import pytest

import fmskill
import fmskill.settings as settings


def test_options():
    o = fmskill.options
    assert isinstance(o, settings.OptionsContainer)
    assert isinstance(o.plot, settings.OptionsContainer)
    assert isinstance(o.metrics, settings.OptionsContainer)


def test_options_repr():
    o = fmskill.options
    assert "metrics.list" in repr(o)
    assert "plot.scatter.points.size" in repr(o.plot)
    assert "metrics.list" in repr(o.metrics)


def test_options_get_options():
    o = fmskill.options
    assert o.plot.scatter.points.size == 20
    assert fmskill.get_option("plot.scatter.points.size") == 20

    # does not need to be complete, just unique
    assert fmskill.get_option("scatter.points.size") == 20

    with pytest.raises(settings.OptionError):
        # there are many color options
        assert fmskill.get_option("color")


def test_options_get_options_dict():
    o = fmskill.options
    d0 = o.plot.scatter.quantiles.kwargs
    # dict is a special case, it cannot be accessed as an attribute
    # without some additional code
    assert not isinstance(d0, dict)
    assert isinstance(d0, settings.OptionsContainer)
    d1 = list(d0.to_dict().values())[0]
    assert isinstance(d1, dict)

    # this is the way to get the dict
    d = settings.get_option("plot.scatter.quantiles.kwargs")
    assert isinstance(d, dict)


def test_options_to_dict():
    o = fmskill.options
    d = o.to_dict()
    assert d["plot.scatter.points.size"] == 20
    assert d["metrics.list"][0] == o.metrics.list[0]


def test_options_set_options():
    o = fmskill.options
    o.plot.scatter.points.size = 300
    assert o.plot.scatter.points.size == 300
    o.plot.scatter.points.size = 100
    assert o.plot.scatter.points.size == 100

    fmskill.set_option("plot.scatter.points.size", 200)
    assert o.plot.scatter.points.size == 200

    # validation fails
    with pytest.raises(ValueError):
        fmskill.set_option("plot.scatter.points.size", "200")
        fmskill.set_option("plot.scatter.points.size", -1)


def test_options_reset_options():
    o = fmskill.options
    o.plot.scatter.points.size = 300
    assert o.plot.scatter.points.size == 300
    fmskill.reset_option("plot.scatter.points.size")
    assert o.plot.scatter.points.size == 20


def test_options_register_option():
    with pytest.raises(settings.OptionError):
        # non-existing option
        assert fmskill.get_option("plot.scatter.points.size2")

    settings.register_option("plot.scatter.points.size2", 200, "test")
    assert fmskill.get_option("plot.scatter.points.size2") == 200
    fmskill.set_option("plot.scatter.points.size2", 300)
    assert fmskill.get_option("plot.scatter.points.size2") == 300
    fmskill.reset_option("plot.scatter.points.size2")
    assert fmskill.get_option("plot.scatter.points.size2") == 200


def test_named_style():

    fmskill.load_style("MOOD")
    assert fmskill.get_option("plot.scatter.points.label") == "Data"


def test_missing_named_style():

    with pytest.raises(KeyError, match="not found"):
        fmskill.load_style("food")
