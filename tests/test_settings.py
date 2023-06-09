import pytest

import modelskill
import modelskill.settings as settings


def test_options():
    o = modelskill.options
    assert isinstance(o, settings.OptionsContainer)
    assert isinstance(o.plot, settings.OptionsContainer)
    assert isinstance(o.metrics, settings.OptionsContainer)


def test_options_repr():
    o = modelskill.options
    assert "metrics.list" in repr(o)
    assert "plot.scatter.points.size" in repr(o.plot)
    assert "metrics.list" in repr(o.metrics)


def test_options_get_options():
    o = modelskill.options
    assert o.plot.scatter.points.size == 20
    assert modelskill.get_option("plot.scatter.points.size") == 20

    # does not need to be complete, just unique
    assert modelskill.get_option("scatter.points.size") == 20

    with pytest.raises(settings.OptionError):
        # there are many color options
        assert modelskill.get_option("color")


def test_options_get_options_dict():
    o = modelskill.options
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
    o = modelskill.options
    d = o.to_dict()
    assert d["plot.scatter.points.size"] == 20
    assert d["metrics.list"][0] == o.metrics.list[0]


def test_options_set_options():
    o = modelskill.options
    o.plot.scatter.points.size = 300
    assert o.plot.scatter.points.size == 300
    o.plot.scatter.points.size = 100
    assert o.plot.scatter.points.size == 100

    modelskill.set_option("plot.scatter.points.size", 200)
    assert o.plot.scatter.points.size == 200

    # validation fails
    with pytest.raises(ValueError):
        modelskill.set_option("plot.scatter.points.size", "200")
        modelskill.set_option("plot.scatter.points.size", -1)


def test_options_set_invalid_metric_raises_error():
    o = modelskill.options

    # this is ok
    o.metrics.list = ["bias", "rmse"]

    # this is not
    with pytest.raises(ValueError):
        o.metrics.list = ["invalid_metric"]

    # neither is a mix of valid and invalid
    with pytest.raises(ValueError, match="invalid_metric"):
        o.metrics.list = ["bias", "invalid_metric"]


def test_options_reset_options():
    o = modelskill.options
    o.plot.scatter.points.size = 300
    assert o.plot.scatter.points.size == 300
    modelskill.reset_option("plot.scatter.points.size")
    assert o.plot.scatter.points.size == 20


def test_options_register_option():
    with pytest.raises(settings.OptionError):
        # non-existing option
        assert modelskill.get_option("plot.scatter.points.size2")

    settings.register_option("plot.scatter.points.size2", 200, "test")
    assert modelskill.get_option("plot.scatter.points.size2") == 200
    modelskill.set_option("plot.scatter.points.size2", 300)
    assert modelskill.get_option("plot.scatter.points.size2") == 300
    modelskill.reset_option("plot.scatter.points.size2")
    assert modelskill.get_option("plot.scatter.points.size2") == 200
