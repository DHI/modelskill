import fmskill
from fmskill.report import Reporter


def test_markdown(tmpdir):
    model_result = fmskill.create("tests/testdata/conf.yml")
    reporter = Reporter(model_result, tmpdir)

    filename = reporter.markdown()

    assert filename.exists
    assert "#" in filename.read_text()


def test_html(tmpdir):
    model_result = fmskill.create("tests/testdata/conf.yml")
    reporter = Reporter(model_result, tmpdir)

    filename = reporter.html()

    assert filename.exists

    assert "<h1>" in filename.read_text()