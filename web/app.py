import os
import tempfile
import streamlit as st
import mikeio
import modelskill
import matplotlib

"""
# ModelSkill Drag and Drop
"""

tmp_folder = tempfile.mkdtemp()

with st.sidebar:
    obs = st.file_uploader("Observation", type="dfs0")

    if obs:
        fn_obs = os.path.join(tmp_folder, "obs.dfs0")
        with open(fn_obs, "wb") as f:
            f.write(obs.getvalue())

        dfs: mikeio.Dfs0 = mikeio.open(fn_obs)  # type: ignore
        items = [item.name for item in dfs.items]
        obs_item = st.selectbox(label="Item", options=items)

    mod = st.file_uploader("Model", type="dfs0")

    if mod:
        fn_mod = os.path.join(tmp_folder, "mod.dfs0")
        with open(fn_mod, "wb") as f:
            f.write(mod.getvalue())
        mdfs: mikeio.Dfs0 = mikeio.open(fn_mod)  # type: ignore
        mitems = [item.name for item in mdfs.items]
        mod_item = st.selectbox(label="Item", options=mitems)

    metrics = st.multiselect(
        "Metrics",
        ["bias", "rmse", "mae", "cc", "r2"],
        default=["bias", "rmse", "mae", "cc", "r2"],
    )

if mod and obs:
    c: modelskill.Comparer = modelskill.match(
        fn_obs, fn_mod, obs_item=obs_item, mod_item=mod_item
    )  # type: ignore

    tabskill, tabts, tabscatter = st.tabs(["Skill", "Time series", "Scatter"])

    with tabskill:
        df = c.skill(metrics=metrics).to_dataframe()
        st.dataframe(df)

    with tabts:
        c.plot.timeseries()
        fig = matplotlib.pyplot.gcf()
        st.pyplot(fig)

    with tabscatter:
        c.plot.scatter()
        fig_sc = matplotlib.pyplot.gcf()
        st.pyplot(fig_sc)
