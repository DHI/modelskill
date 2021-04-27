import fmskill
from fmskill.report import Reporter


def test_markdown(tmpdir):
    model_result = fmskill.from_config("tests/testdata/conf.yml")
    reporter = Reporter(model_result, tmpdir)

    filename = reporter.to_markdown()

    assert filename.exists
    assert "#" in filename.read_text()


def test_html(tmpdir):
    model_result = fmskill.from_config("tests/testdata/conf.yml")
    reporter = Reporter(model_result, tmpdir)

    filename = reporter.to_html()

    assert filename.exists

    assert "<h1>" in filename.read_text()