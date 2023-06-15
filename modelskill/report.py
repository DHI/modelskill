import matplotlib.pyplot as plt
from pathlib import Path
import re
import markdown
import modelskill


class Reporter:
    def __init__(self, connector, folder=None):
        assert isinstance(connector, modelskill.Connector)
        self.connector = connector
        self.name = connector.mod_names[0]

        if folder is None:
            self.basedir = Path(self.name)
        else:
            self.basedir = Path(folder)
        self.basedir.mkdir(exist_ok=True)

    def _markdown(self):
        lines = []
        lines.append(f"# Validation report - {self.name}")
        lines.append("## Observations")

        self.connector.plot_observation_positions()
        plt.savefig(self.basedir / f"{self.safe_name}_map.png")
        lines.append(f"![map]({self.safe_name}_map.png)")

        lines.append("## Timeseries")
        cc = self.connector.extract()
        for key, value in cc.comparers.items():
            if "plot_timeseries" in dir(value):
                value.plot_timeseries()
                plt.savefig(self.basedir / f"{self.safe_name}_{key}_ts.png")
                lines.append(f"![{key}]({self.safe_name}_{key}_ts.png)")

        lines.append("## Aggregated skill")

        skilldf = cc.skill().round(3)
        lines.append(skilldf.to_html() + "\n")

        lines.append("## Scatter")

        for key, value in cc.comparers.items():
            value.scatter()
            plt.savefig(self.basedir / f"{self.safe_name}_{key}_scatter.png")
            lines.append(f"![{key}]({self.safe_name}_{key}_scatter.png)")

        return "\n".join(lines)

    def to_markdown(self):
        """Create report in markdown format"""

        filename = self.basedir / f"{self.safe_name}.md"

        contents = self._markdown()

        with open(filename, "w") as f:
            f.write(contents)

        return filename

    def to_html(self):
        """Create report in html format"""

        filename = self.basedir / "index.html"

        md = self._markdown()

        html = markdown.markdown(md)
        with open(filename, "w") as f:
            f.write(self._html_header())
            f.write(html)
            f.write(self._html_footer())

        return filename

    def _html_header(self):
        header = """
            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"
            integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">

        <body>
            <div class="container">
        """
        return header

    def _html_footer(self):
        footer = """
    </div>
    </body>
        """
        return footer

    @property
    def safe_name(self) -> str:
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", "", self.name)

        # Replace all runs of whitespace with a single dash
        s = re.sub(r"\s+", "-", s)

        return s
