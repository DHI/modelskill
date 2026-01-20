"""Internal data structures for skill calculations.

This module provides a long-format DataFrame abstraction that eliminates
MultiIndex complexity while maintaining backward compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import pandas as pd

from .metrics import get_metric_names


@dataclass
class SkillDimensions:
    """Tracks which columns are dimensions vs metrics.

    Dimensions: observation, model, quantity (categorical grouping)
    Metrics: n, bias, rmse, r2, etc. (computed statistics)
    """

    all: list[str]
    metrics: list[str]

    @classmethod
    def infer(cls, df: pd.DataFrame) -> "SkillDimensions":
        """Infer dimensions from DataFrame columns."""
        known_dims = {"observation", "model", "quantity"}
        known_metrics = set(get_metric_names()) | {"n"}

        dims = [c for c in df.columns if c in known_dims]
        # Metrics are either known metric names or numeric columns not in dims
        metrics = [
            c
            for c in df.columns
            if c in known_metrics
            or (c not in dims and pd.api.types.is_numeric_dtype(df[c]))
        ]

        return cls(all=dims, metrics=metrics)


class _SkillData:
    """Internal storage for skill results using long-format DataFrame.

    Dimensions (observation, model, quantity) are regular columns rather than
    MultiIndex levels, simplifying filtering and aggregation.
    SkillTable converts to MultiIndex format for display/backward compatibility.
    """

    def __init__(self, df: pd.DataFrame, dims: SkillDimensions | None = None):
        self._df = df.copy()
        self.dims = dims or SkillDimensions.infer(df)

        # Convert dimension columns to categorical for efficient groupby
        for dim in self.dims.all:
            if dim in self._df.columns:
                dtype = self._df[dim].dtype
                if not isinstance(dtype, pd.CategoricalDtype):
                    self._df[dim] = self._df[dim].astype("category")

    @classmethod
    def from_multiindex(cls, df: pd.DataFrame) -> "_SkillData":
        """Create from MultiIndex DataFrame (backward compatibility)."""
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

        # Infer metrics from dataframe
        dims = SkillDimensions.infer(df)

        # All index columns become dimensions (including dt:month -> month, etc.)
        # This preserves grouping dimensions like month, weekday, hour, etc.
        dims.all = index_cols

        return cls(df, dims)

    def sel(self, **kwargs) -> "_SkillData":
        """Values can be single items or lists."""
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
        if not self.dims.all:
            return self._df.copy()

        if reduce_index:
            # Only use dimensions with multiple values as index
            active_dims = [
                d
                for d in self.dims.all
                if d in self._df.columns and self._df[d].nunique() > 1
            ]
        else:
            # Use all dimensions as index
            active_dims = [d for d in self.dims.all if d in self._df.columns]

        if active_dims:
            return self._df.set_index(active_dims)
        else:
            return self._df.copy()

    def _aggregate_all(self, weights: dict | None, n_col: str) -> "_SkillData":
        """Aggregate all rows into a single row."""
        result_dict = {n_col: [self._df[n_col].sum()]}

        # Aggregate metrics
        for metric in self.dims.metrics:
            if metric == n_col:
                continue

            if weights:
                # Find the weight dimension and compute weighted average
                weight_col = self._get_weight_column(weights)
                result_dict[metric] = [np.average(self._df[metric], weights=weight_col)]
            else:
                result_dict[metric] = [self._df[metric].mean()]

        result = pd.DataFrame(result_dict)
        new_dims = SkillDimensions(all=[], metrics=self.dims.metrics)
        return _SkillData(result, new_dims)

    def _get_weight_column(self, weights: dict) -> pd.Series:
        """Find the first dimension column and map it to weights."""
        for dim in self.dims.all:
            if dim in self._df.columns:
                return self._df[dim].map(weights)
        raise ValueError("No dimension column found for weight mapping")

    def aggregate(
        self, by: list[str], weights: dict | None = None, n_col: str = "n"
    ) -> "_SkillData":
        # Handle empty by - aggregate everything into single row
        if not by:
            return self._aggregate_all(weights, n_col)

        # Build aggregation dictionary
        agg_dict: dict[str, str | Callable[[Any], Any]] = {}

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

            def weighted_mean(x: pd.Series) -> float:
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
            result = (
                df_with_weights.groupby(by, observed=True).agg(agg_dict).reset_index()
            )
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

            result = self._df.groupby(by, observed=True).agg(agg_dict).reset_index()

        # Update dimensions for aggregated result
        new_dims = SkillDimensions(all=by, metrics=self.dims.metrics)

        return _SkillData(result, new_dims)

    def get_unique_values(self, dim: str) -> list:
        if dim not in self.dims.all or dim not in self._df.columns:
            return []
        return self._df[dim].unique().tolist()
