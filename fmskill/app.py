import os
import tempfile
import streamlit as st
import mikeio
import fmskill

"""
# FMskill Drag and Drop
"""

tmp_folder = tempfile.mkdtemp()

with st.sidebar:
    
    obs = st.file_uploader("Observation", type="dfs0")

    if obs:
        fn_obs = os.path.join(tmp_folder, "obs.dfs0")
        with open(fn_obs, "wb") as f:
            f.write(obs.getvalue())

        dfs = mikeio.open(fn_obs)
        items = [item.name for item in dfs.items]
        obs_item = st.selectbox(label="Item", options=items)

    mod = st.file_uploader("Model", type="dfs0")

    if mod:
        fn_mod = os.path.join(tmp_folder, "mod.dfs0")
        with open(fn_mod, "wb") as f:
            f.write(mod.getvalue())
        dfs = mikeio.open(fn_mod)
        items = [item.name for item in dfs.items]
        mod_item = st.selectbox(label="Item", options=items)

if mod and obs:

    c = fmskill.compare(fn_obs, fn_mod, obs_item=obs_item, mod_item=mod_item)

    df = c.skill().to_dataframe()
    st.dataframe(df)

    fig_ts = c.plot_timeseries(backend="plotly", return_fig=True)
    st.plotly_chart(fig_ts)

    fig_sc = c.scatter(backend="plotly", return_fig=True)
    st.plotly_chart(fig_sc)
