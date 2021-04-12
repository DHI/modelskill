import matplotlib.pyplot as plt
from pathlib import Path
import re


class Reporter:
    def __init__(self, modelresults, folder=None):
        self.name = modelresults.name
        self.mr = modelresults

        if folder is None:
            self.basedir = Path(self.name)
        else:
            self.basedir = Path(folder)
        self.basedir.mkdir(exist_ok=True)

    def markdown(self):
        """Create report in markdown format"""

        filename = self.basedir / f"{self.safe_name}.md"

        with open(filename, "w") as f:

            f.write(f"# Validation report - {self.name}\n")
            f.write("## Observations\n")

            self.mr.plot_observation_positions()
            plt.savefig(self.basedir / f"{self.safe_name}_map.png")
            f.write(f"![map]({self.safe_name}_map.png)\n")

            f.write("## Timeseries\n")
            cc = self.mr.extract()
            for key, value in cc.comparers.items():
                if "plot_timeseries" in dir(value):
                    value.plot_timeseries()
                    plt.savefig(self.basedir / f"{self.safe_name}_{key}_ts.png")
                    f.write(f"![{key}]({self.safe_name}_{key}_ts.png)\n")

            f.write("## Scatter\n")

            for key, value in cc.comparers.items():
                value.scatter()
                plt.savefig(self.basedir / f"{self.safe_name}_{key}_scatter.png")
                f.write(f"![{key}]({self.safe_name}_{key}_scatter.png)\n")

        return filename

    @property
    def safe_name(self) -> str:
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", "", self.name)

        # Replace all runs of whitespace with a single dash
        s = re.sub(r"\s+", "-", s)

        return s
