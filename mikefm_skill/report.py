import matplotlib.pyplot as plt
from pathlib import Path


class Reporter:
    def __init__(self, modelresults):
        self.name = modelresults.name
        self.mr = modelresults
        self.basedir = Path(self.name)
        self.basedir.mkdir(exist_ok=True)

    def markdown(self, filename=None):

        if filename is None:
            filename = self.basedir / f"{self.name}.md"

        with open(filename, "w") as f:

            f.write(f"# Validation report - {self.name}\n")
            f.write("## Observations\n")

            self.mr.plot_observation_positions()
            plt.savefig(self.basedir / f"{self.name}_map.png")
            f.write(f"![map]({self.name}_map.png)\n")

            f.write("## Timeseries\n")
            cc = self.mr.extract()
            for key, value in cc.comparisons.items():
                if "plot_timeseries" in dir(value):
                    value.plot_timeseries()
                    plt.savefig(self.basedir / f"{self.name}_{key}_ts.png")
                    f.write(f"![{key}]({self.name}_{key}_ts.png)\n")

            f.write("## Scatter\n")

            for key, value in cc.comparisons.items():
                value.scatter()
                plt.savefig(self.basedir / f"{self.name}_{key}_scatter.png")
                f.write(f"![{key}]({self.name}_{key}_scatter.png)\n")

        return filename