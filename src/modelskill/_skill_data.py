"""Internal data structures for skill calculations.

This module provides a long-format DataFrame abstraction that eliminates
MultiIndex complexity while maintaining backward compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable
import numpy as np
import pandas as pd

from .metrics import get_metric_names


@dataclass
class SkillDimensions:
    """Metadata tracking which columns represent dimensions vs metrics vs coordinates.

    This class encapsulates the dimensional structure of skill results,
    distinguishing between:
    - Dimensions: observation, model, quantity (categorical grouping variables)
    - Metrics: n, bias, rmse, r2, etc. (computed statistics)
    - Coords: x, y (spatial metadata)

    Parameters
    ----------
    all : list of str
        Names of dimension columns present in the data
    metrics : list of str
        Names of metric columns
    coords : list of str, optional
        Names of coordinate columns (e.g., x, y)

    Examples
    --------
    >>> dims = SkillDimensions(
    ...     all=["observation", "model"],
    ...     metrics=["n", "bias", "rmse"],
    ...     coords=["x", "y"]
    ... )
    """

    all: list[str]
    metrics: list[str]
    coords: list[str] = field(default_factory=lambda: ["x", "y"])

    @classmethod
    def infer(cls, df: pd.DataFrame) -> "SkillDimensions":
        """Infer dimensions from DataFrame columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to infer dimensions from

        Returns
        -------
        SkillDimensions
            Inferred dimensional structure
        """
        known_dims = {"observation", "model", "quantity"}
        known_coords = {"x", "y", "z"}
        known_metrics = set(get_metric_names()) | {"n"}

        dims = [c for c in df.columns if c in known_dims]
        coords = [c for c in df.columns if c in known_coords]
        # Metrics are either known metric names or numeric columns not in dims/coords
        metrics = [
            c for c in df.columns
            if c in known_metrics or (
                c not in dims
                and c not in coords
                and pd.api.types.is_numeric_dtype(df[c])
            )
        ]

        return cls(all=dims, metrics=metrics, coords=coords)

    def to_index_names(self) -> list[str]:
        """Get list of dimension names suitable for use as MultiIndex.

        Returns
        -------
        list of str
            Dimension names
        """
        return self.all.copy()


class _SkillData:
    """Internal storage for skill results using long-format DataFrame.

    This class wraps a long-format DataFrame where dimensions (observation, model,
    quantity) are regular columns rather than MultiIndex levels. This dramatically
    simplifies operations like filtering, aggregation, and coordinate assignment.

    The public API (SkillTable) converts to MultiIndex format for display and
    backward compatibility, but all internal operations work on the long format.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with dimension columns
    dims : SkillDimensions, optional
        Dimensional metadata. If None, will be inferred from df.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "observation": ["obs1", "obs1", "obs2"],
    ...     "model": ["m1", "m2", "m1"],
    ...     "n": [100, 100, 150],
    ...     "bias": [0.1, 0.2, 0.05],
    ...     "rmse": [0.2, 0.3, 0.15],
    ...     "x": [1.0, 1.0, 2.0],
    ...     "y": [2.0, 2.0, 3.0],
    ... })
    >>> skill_data = _SkillData(df)
    >>> skill_data.sel(model="m1")
    """

    def __init__(self, df: pd.DataFrame, dims: SkillDimensions | None = None):
        self._df = df.copy()
        self.dims = dims or SkillDimensions.infer(df)

        # Convert dimension columns to categorical for efficient groupby
        # But don't convert coordinates - they need to support mean()
        for dim in self.dims.all:
            if dim in self._df.columns and dim not in self.dims.coords:
                dtype = self._df[dim].dtype
                if not isinstance(dtype, pd.CategoricalDtype):
                    self._df[dim] = self._df[dim].astype("category")

        # Ensure coordinates are numeric (not categorical)
        for coord in self.dims.coords:
            if coord in self._df.columns:
                if isinstance(self._df[coord].dtype, pd.CategoricalDtype):
                    self._df[coord] = self._df[coord].astype(float)

    @classmethod
    def from_multiindex(cls, df: pd.DataFrame) -> "_SkillData":
        """Create from MultiIndex DataFrame (backward compatibility).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MultiIndex or regular index

        Returns
        -------
        _SkillData
            Long-format skill data
        """
        # Track which columns were in the index (these become dimensions)
        index_cols = []

        # Convert index to columns if it contains dimension names
        if isinstance(df.index, pd.MultiIndex):
            index_cols = list(df.index.names)
            df = df.reset_index()
        elif df.index.name is not None:
            # Single index with a name
            index_cols = [df.index.name]
            df = df.reset_index()

        # Infer metrics and coords from dataframe
        dims = SkillDimensions.infer(df)

        # IMPORTANT: dims.all should ONLY contain columns that were in the index
        # Columns like 'observation' added by _add_as_col_if_not_in_index() should remain as columns
        dims.all = [col for col in index_cols if col not in dims.coords]

        return cls(df, dims)

    def sel(self, **kwargs) -> "_SkillData":
        """Select by dimension values.

        Parameters
        ----------
        **kwargs
            Dimension names and values to filter by.
            Values can be single items or lists.
            Can select on any dimension in dims.all or any categorical column.

        Returns
        -------
        _SkillData
            Filtered skill data

        Examples
        --------
        >>> skill_data.sel(model="m1")
        >>> skill_data.sel(observation=["obs1", "obs2"])
        """
        df = self._df

        # Build set of selectable dimensions: tracked dimensions + categorical columns
        selectable = set(self.dims.all)
        for col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                selectable.add(col)

        for dim, value in kwargs.items():
            if dim not in selectable:
                raise KeyError(
                    f"Cannot select on '{dim}'. "
                    f"Selectable dimensions: {sorted(selectable)}"
                )

            if dim not in df.columns:
                raise KeyError(
                    f"Dimension '{dim}' not found in data. "
                    f"Available columns: {list(df.columns)}"
                )

            if isinstance(value, (list, tuple)):
                df = df[df[dim].isin(value)]
            else:
                df = df[df[dim] == value]

        return _SkillData(df, self.dims)

    def to_display(self, reduce_index: bool = False) -> pd.DataFrame:
        """Convert to MultiIndex DataFrame for display.

        Parameters
        ----------
        reduce_index : bool
            If True, only dimensions with >1 unique value are used as index.
            If False (default), all dimensions are used as index.

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex on dimensions
        """
        if not self.dims.all:
            return self._df.copy()

        if reduce_index:
            # Only use dimensions with multiple values as index
            active_dims = [
                d for d in self.dims.all
                if d in self._df.columns and self._df[d].nunique() > 1
            ]
        else:
            # Use all dimensions as index
            active_dims = [d for d in self.dims.all if d in self._df.columns]

        if active_dims:
            return self._df.set_index(active_dims)
        else:
            return self._df.copy()

    def aggregate(
        self,
        by: list[str],
        weights: dict | None = None,
        n_col: str = "n"
    ) -> "_SkillData":
        """Aggregate metrics across dimensions.

        Parameters
        ----------
        by : list of str
            Dimension names to group by. If empty, aggregates everything.
        weights : dict, optional
            Mapping from group values to weights for weighted aggregation
        n_col : str
            Name of the count column (summed, not averaged)

        Returns
        -------
        _SkillData
            Aggregated skill data
        """
        # Handle empty by - aggregate everything into single row
        if not by:
            result_dict = {n_col: [self._df[n_col].sum()]}
            for metric in self.dims.metrics:
                if metric != n_col:
                    if weights:
                        # Find the weight dimension
                        for dim in self.dims.all:
                            if dim in self._df.columns:
                                weight_col = self._df[dim].map(weights)
                                result_dict[metric] = [
                                    np.average(self._df[metric], weights=weight_col)
                                ]
                                break
                    else:
                        result_dict[metric] = [self._df[metric].mean()]

            # Also aggregate coordinates (convert from categorical if needed)
            for coord in self.dims.coords:
                if coord in self._df.columns:
                    coord_series = self._df[coord]
                    # Convert categorical to numeric if needed
                    if isinstance(coord_series.dtype, pd.CategoricalDtype):
                        coord_series = coord_series.astype(float)
                    result_dict[coord] = [coord_series.mean()]

            result = pd.DataFrame(result_dict)
            new_dims = SkillDimensions(all=[], metrics=self.dims.metrics, coords=self.dims.coords)
            return _SkillData(result, new_dims)

        # Build aggregation dictionary
        agg_dict = {}

        if weights is not None:
            # Weighted mean for metrics
            # Determine which dimension has the weights
            # Find the dimension NOT in the groupby (that's what we're aggregating over)
            weight_dim = None
            for dim in self.dims.all:
                if dim not in by:
                    weight_dim = dim
                    break

            if weight_dim is None:
                raise ValueError("Cannot determine weight dimension")

            # Add weights column to DataFrame for aggregation
            df_with_weights = self._df.copy()
            df_with_weights["_weight"] = df_with_weights[weight_dim].map(weights)

            def weighted_mean(x):
                # x is a Series, we need corresponding weights
                # Get the group's weights from the same rows
                w = df_with_weights.loc[x.index, "_weight"]
                valid = x.notna() & w.notna()
                if not valid.any():
                    return np.nan
                return np.average(x[valid], weights=w[valid])

            for metric in self.dims.metrics:
                if metric == n_col:
                    agg_dict[metric] = "sum"
                else:
                    agg_dict[metric] = weighted_mean

            # Use df_with_weights for groupby
            result = df_with_weights.groupby(by, observed=True).agg(agg_dict).reset_index()
            # Remove weight column
            if "_weight" in result.columns:
                result = result.drop(columns=["_weight"])
        else:
            # Simple mean for metrics
            for metric in self.dims.metrics:
                if metric == n_col:
                    agg_dict[metric] = "sum"
                else:
                    agg_dict[metric] = "mean"

            # Also aggregate coordinates (mean) - convert categorical to float first
            for coord in self.dims.coords:
                if coord in self._df.columns:
                    # If categorical, we can't use mean directly
                    # Convert to float first
                    if isinstance(self._df[coord].dtype, pd.CategoricalDtype):
                        agg_dict[coord] = lambda x: x.astype(float).mean()
                    else:
                        agg_dict[coord] = "mean"

            result = self._df.groupby(by, observed=True).agg(agg_dict).reset_index()

        # Update dimensions for aggregated result
        new_dims = SkillDimensions(
            all=by,
            metrics=self.dims.metrics,
            coords=self.dims.coords
        )

        return _SkillData(result, new_dims)

    def get_unique_values(self, dim: str) -> list:
        """Get unique values for a dimension.

        Parameters
        ----------
        dim : str
            Dimension name

        Returns
        -------
        list
            Unique values, or empty list if dimension not present
        """
        if dim not in self.dims.all or dim not in self._df.columns:
            return []
        return self._df[dim].unique().tolist()

    def copy(self) -> "_SkillData":
        """Create a copy of this SkillData.

        Returns
        -------
        _SkillData
            Copy of skill data
        """
        return _SkillData(self._df.copy(), self.dims)
