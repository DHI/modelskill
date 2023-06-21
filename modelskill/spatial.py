

class SpatialSkill:
    """
    Spatial skill object for analysis and visualization of spatially
    gridded skill assessment. The object wraps the xr.DataSet class
    which can be accessed from the attribute ds.

    Examples
    --------
    >>> ss = comparer.spatial_skill()
    >>> ss.field_names
    ['n', 'bias', 'rmse', 'urmse', 'mae', 'cc', 'si', 'r2']

    >>> ss.mod_names
    ['SW_1', 'SW_2']

    >>> ss.plot(field='rmse', model='SW_1')
    """

    @property
    def x(self):
        """x-coordinate values"""
        return self.ds.x

    @property
    def y(self):
        """y-coordinate values"""
        return self.ds.y

    @property
    def coords(self):
        """Coordinates (same as xr.DataSet.coords) """
        return self.ds.coords

    @property
    def obs_names(self):
        """List of observation names"""
        if "observation" in self._coords_list:
            return list(self.coords["observation"].values)
        else:
            return []

    @property
    def mod_names(self):
        """List of model names"""
        if "model" in self._coords_list:
            return list(self.coords["model"].values)
        else:
            return []

    @property
    def field_names(self):
        """List of field names (=data vars)"""
        return list(self.ds.data_vars)

    @property
    def _coords_list(self):
        return [d for d in self.coords.dims]

    @property
    def n(self):
        """number of observations"""
        if "n" in self.ds:
            return self.ds.n

    def __init__(self, ds, name: str = None):
        # TODO: add type and unit info; add domain to plot outline on map
        self.ds = ds
        self.name = name
        self._set_attrs()

    def _set_attrs(self):

        # TODO: use type and unit to give better long name
        # self.ds["bias"].attrs = dict(long_name="Bias of Hm0", units="m")

        self.ds["n"].attrs = dict(long_name="Number of observations", units="-")
        if self._has_geographical_coords():
            self.ds["x"].attrs = dict(long_name="Longitude", units="degrees east")
            self.ds["y"].attrs = dict(long_name="Latitude", units="degrees north")
        else:
            self.ds["x"].attrs = dict(long_name="Easting", units="meter")
            self.ds["y"].attrs = dict(long_name="Northing", units="meter")

    def _has_geographical_coords(self):
        is_geo = True
        if (self.x.min() < -180.0) or (self.x.max() > 360.0):
            is_geo = False
        if (self.y.min() < -90.0) or (self.y.max() > 90.0):
            is_geo = False
        return is_geo

    def plot(self, field: str, model=None, **kwargs):
        """wrapper for xArray DataSet plot function

        Parameters
        ----------
        field : str
            The field to plot, e.g. 'rmse' or 'bias'
        model : str, optional
            Name of model to plot, by default all models
        **kwargs : keyword arguments passed to xr.DataSet plot()
            e.g. figsize

        Examples
        --------
        >>> ss = comparer.spatial_skill()
        >>> ss.plot(field='bias')
        >>> ss.plot('rmse', model='SW_1')
        >>> ss.plot(field='r2', cmap='YlOrRd', figsize=(10,10))
        """
        if field not in self.field_names:
            raise ValueError(f"field {field} not found in {self.field_names}")

        if model is None:
            ds = self.ds[field]
        else:
            if model not in self.mod_names:
                raise ValueError(f"model {model} not in model list ({self.mod_names})")
            ds = self.ds[field].sel({"model": model})

        extra_dims = [d for d in ds.coords.dims if d not in ["x", "y"]]
        if len(extra_dims) == 2:
            ax = ds.plot(col=extra_dims[0], row=extra_dims[1], **kwargs)
        elif len(extra_dims) == 1:
            ax = ds.plot(col=extra_dims[0], **kwargs)
        else:
            ax = ds.plot(**kwargs)
        return ax

    def to_dataframe(self):
        """export as pandas.DataFrame"""
        return self.ds.to_dataframe()

    def __repr__(self):
        return repr(self.ds)
