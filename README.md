# mikefm-skill
Compare results from MIKE FM simulations with observations


## Usage
```
>>> from mikefm_skill.model import ModelResult
>>> from mikefm_skill.observation import PointObservation
>>> mr = ModelResult("../tests/testdata/Oresund2D.dfsu")
>>> mr.add_observation(PointObservation("../tests/testdata/smhi_2095_klagshamn.dfs0", item=0, x=0.36684415E+06, y=0.61542916E+07, name="Klagshamn") , item=0)
>>> mr.add_observation(PointObservation("../tests/testdata/dmi_30357_Drogden_Fyr.dfs0",item=0, x=355568.0, y=6156863.0), item=0)
>>> collection = mr.extract()
>>> report = collection.skill_report()
                       bias  rmse  corr_coef  scatter_index
Klagshamn              0.18  0.19       0.84           0.32
dmi_30357_Drogden_Fyr  0.26  0.28       0.51           0.53
```

### Overview of observation locations

![map](images/map.png)

### Scatter plot

![scatter](images/scatter.png)