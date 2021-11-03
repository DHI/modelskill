import pytest
import fmskill
from fmskill.report import Reporter
import matplotlib as mpl

mpl.use("Agg")


def test_markdown(tmpdir):
    with pytest.warns(UserWarning, match="Item type mismatch!"):
        connector = fmskill.from_config("tests/testdata/conf.yml", validate_eum=False)
    reporter = Reporter(connector, tmpdir)

    filename = reporter.to_markdown()

    assert filename.exists
    assert "#" in filename.read_text()


def test_html(tmpdir):
    with pytest.warns(UserWarning, match="Item type mismatch!"):
        connector = fmskill.from_config("tests/testdata/conf.yml", validate_eum=False)
    reporter = Reporter(connector, tmpdir)

    filename = reporter.to_html()

    assert filename.exists

    assert "<h1>" in filename.read_text()
