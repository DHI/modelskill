from enum import Enum

import typer
import fmskill
from fmskill.report import Reporter
from pathlib import Path


app = typer.Typer(add_completion=False)


class OutputFormat(str, Enum):

    html = "html"
    markdown = "md"


@app.command()
def report(
    config_file: Path = typer.Argument(..., exists=True),
    output_folder: Path = typer.Argument(...),
    format: OutputFormat = OutputFormat.html,
):
    """
    fmskill: Automatic model skill assessment
    """

    con = fmskill.from_config(str(config_file), validate_eum=False)
    reporter = Reporter(con, output_folder)

    if format.value == "md":
        filename = reporter.to_markdown()
    else:
        filename = reporter.to_html()
    typer.echo(f"Report created at: {filename}")


if __name__ == "__main__":
    app()
