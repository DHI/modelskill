import mikeio
from modelskill._rose import wind_rose

from shiny import App, render, ui

app_ui = ui.page_fluid(
    ui.panel_title("Wind rose (configurable)"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select("n", "# of sectors", [4, 8, 16]),
            ui.input_slider("calm", "Calm threshold", min=0.0, max=1.0, value=0.1, step=0.1),
            ui.input_slider("mag_step", "Magnitude step", min=0.1, max=1.0, value=0.2, step=0.2),
            ui.input_radio_buttons("datasets", label="Data", choices = ["model", "observation", "both"], selected = "both"),
        ),
        ui.panel_main(
            ui.output_plot("plot"),
            ui.output_table("table"),
            
        ),
    )
)

def get_data(datasets:str):
    ds = mikeio.read("../tests/testdata/wave_dir.dfs0")
    data = ds[[0,2,1,3]].to_dataframe()
    if datasets == "model":
        data = data.iloc[:,[0,1]]
    elif datasets== "observation":
        data = data.iloc[:,[2,3]]
    return data


def server(input, output, session):
    
    @output
    @render.plot
    def plot():
        data = get_data(input.datasets())
        n = int(input.n())
        calm = float(input.calm())
        mag_step = float(input.mag_step())
        ax = wind_rose(data.to_numpy(), n_sectors=n, calm_threshold=calm, mag_step=mag_step)
        return ax.figure

    @output
    @render.table
    def table():
        data = get_data(input.datasets())
        data = data.head()
        return data


app = App(app_ui, server)
