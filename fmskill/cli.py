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

    con = fmskill.from_config(configuration)
    reporter = Reporter(con, output_folder)

    if output_format == "md":
        filename = reporter.to_markdown()
    else:
        filename = reporter.to_html()
    print(f"Report created at: {filename}")
