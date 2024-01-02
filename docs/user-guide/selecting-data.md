# Selecting/filtering data

The primary data filtering method of ModelSkill is the `sel()` method which is accesible on most ModelSkill data structures. The `sel()` method is a wrapper around `xarray.Dataset.sel()` and can be used to select data based on time, location and/or variable. The `sel()` method returns a new data structure of the same type with the selected data.