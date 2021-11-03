import click
import fmskill
from fmskill.report import Reporter


@click.command()
@click.argument("configuration")
@click.option("--output_folder", help="Folder to write output to")
@click.option(
    "--output_format",
    type=click.Choice(["md", "html"]),
    default="html",
    help="Output format, default is html",
)
def report(configuration: str, output_folder=None, output_format="html") -> None:
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
