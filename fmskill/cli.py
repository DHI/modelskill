import click
import markdown
import fmskill
from fmskill.report import Reporter


@click.command("markdown")
@click.argument("configuration")
def report(configuration: str) -> None:

    model_result = fmskill.create(configuration)
    reporter = Reporter(model_result)

    filename = reporter.markdown()
    print(f"Report created at: {filename}")
