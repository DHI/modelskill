import click
import fmskill
from fmskill.report import Reporter


@click.command()
@click.argument("configuration")
@click.option("--output_folder", help="Folder to write output to")
def report(configuration: str, output_folder=None) -> None:
    """
    fmskill: Automatic model skill assessment
    """

    model_result = fmskill.create(configuration)
    reporter = Reporter(model_result, output_folder)

    filename = reporter.markdown()
    print(f"Report created at: {filename}")
