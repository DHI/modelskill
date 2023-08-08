import streamlit as st

import mikeio
from modelskill._rose import wind_rose

@st.cache_data
def get_data():
    ds = mikeio.read("tests/testdata/wave_dir.dfs0")
    df = ds[[0,2,1,3]].to_dataframe()
    data = df.to_numpy()
    return data

data = get_data()

# add all inputs in sidebar
with st.sidebar:
    n_sectors = st.selectbox("Number of sectors", [4, 8, 16], index=1)
    mag_step = st.slider("Magnitude step", min_value=0.1, max_value=1.0, value=0.1, step=0.2)
    calm = st.slider("Calm threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    factor = st.slider("Factor", min_value=1.0, max_value=2.0, value=2.0, step=0.1)

ax = wind_rose(data, n_sectors=n_sectors, mag_step=mag_step, calm_threshold=calm, secondary_dir_step_factor=factor)

st.pyplot(ax.figure)

cmd = f"""
import mikeio
from modelskill.rose import wind_rose
ds = mikeio.read("tests/testdata/wave_dir.dfs0")
df = ds[[0,2,1,3]].to_dataframe()
data = df.to_numpy()

wind_rose(data, n_sectors={n_sectors}, mag_step={mag_step}, calm_threshold={calm})"""
with st.expander("Show code"):
    st.markdown(f"```python\n{cmd}\n```")