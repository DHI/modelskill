import typer
import fmskill
from fmskill.report import Reporter
from pathlib import Path


app = typer.Typer(add_completion=False)


@app.command()
def report(
    config_file: Path,
    output_folder: Path,
    output_format: str = typer.Argument(default="html", help="Choose: html or md"),
):
    """
    fmskill: Automatic model skill assessment
    """

    con = fmskill.from_config(str(config_file), validate_eum=False)
    reporter = Reporter(con, output_folder)

    if output_format == "md":
        filename = reporter.to_markdown()
    else:
        filename = reporter.to_html()
    typer.echo(f"Report created at: {filename}")


if __name__ == "__main__":
    app()
