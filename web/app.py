import tempfile
from pathlib import Path
import streamlit as st
import mikeio
import modelskill
import matplotlib.pyplot as plt

"""
# ModelSkill Drag and Drop
"""

tmp = Path(tempfile.mkdtemp())

with st.sidebar:
    obs = st.file_uploader("Observation", type="dfs0")

    if obs:
        fn_obs = tmp / "obs.dfs0"
        fn_obs.write_bytes(obs.getvalue())

        dfs = mikeio.open(fn_obs)
        obs_item = st.selectbox("Item", options=[i.name for i in dfs.items])

    mod = st.file_uploader("Model", type="dfs0")

    if mod:
        fn_mod = tmp / "mod.dfs0"
        fn_mod.write_bytes(mod.getvalue())
        mdfs = mikeio.open(fn_mod)
        mod_item = st.selectbox("Item", options=[i.name for i in mdfs.items])

    metrics = st.multiselect(
        "Metrics",
        ["bias", "rmse", "mae", "cc", "r2", "si", "kge", "mape", "urmse"],
        default=["bias", "rmse", "r2"],
    )

if mod and obs:
    c = modelskill.match(
        fn_obs, fn_mod, obs_item=obs_item, mod_item=mod_item, gtype="point"
    )

    tskill, tts, tsc = st.tabs(["Skill", "Time series", "Scatter"])

    with tskill:
        df = c.skill(metrics=metrics).to_dataframe()
        st.dataframe(df)

    with tts:
        c.plot.timeseries()
        st.pyplot(plt.gcf())

    with tsc:
        c.plot.scatter()
        st.pyplot(plt.gcf())
