import fmskill
import fmskill.settings as settings


def test_options():
    o = fmskill.options
    assert isinstance(o, settings.OptionsContainer)
    assert isinstance(o.plot, settings.OptionsContainer)
    assert isinstance(o.metrics, settings.OptionsContainer)
