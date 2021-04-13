import click
import fmskill
from fmskill.report import Reporter


@click.command()
@click.argument("configuration")
@click.option("--output_folder", help="Folder to write output to")
@click.option(
    "--output_format",
    type=click.Choice(["md", "html"]),
    default="md",
    help="Output format, default is markdown",
)
def report(configuration: str, output_folder=None, output_format="md") -> None:
    """
    fmskill: Automatic model skill assessment
    """

    model_result = fmskill.create(configuration)
    reporter = Reporter(model_result, output_folder)

    if output_format == "md":
        filename = reporter.markdown()
    else:
        filename = reporter.html()
    print(f"Report created at: {filename}")
