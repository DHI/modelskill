"""Tests for internal _SkillData structure"""

import numpy as np
import pandas as pd
import pytest

from modelskill._skill_data import _SkillData, SkillDimensions


def test_skill_dimensions_infer():
    """Test that SkillDimensions can be inferred from DataFrame"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs1", "obs2"],
        "model": ["m1", "m2", "m1"],
        "n": [100, 100, 150],
        "bias": [0.1, 0.2, 0.05],
        "rmse": [0.2, 0.3, 0.15],
        "x": [1.0, 1.0, 2.0],
        "y": [2.0, 2.0, 3.0],
    })

    dims = SkillDimensions.infer(df)

    assert "observation" in dims.all
    assert "model" in dims.all
    assert "n" in dims.metrics
    assert "bias" in dims.metrics
    assert "rmse" in dims.metrics
    assert "x" in dims.coords
    assert "y" in dims.coords


def test_skill_data_creation():
    """Test creating _SkillData from DataFrame"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs1"],
        "model": ["m1", "m2"],
        "n": [100, 100],
        "bias": [0.1, 0.2],
        "rmse": [0.2, 0.3],
        "x": [1.0, 1.0],
        "y": [2.0, 2.0],
    })

    skill_data = _SkillData(df)

    assert len(skill_data._df) == 2
    assert "observation" in skill_data.dims.all
    assert "model" in skill_data.dims.all


def test_skill_data_sel():
    """Test filtering _SkillData by dimension"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs1", "obs2", "obs2"],
        "model": ["m1", "m2", "m1", "m2"],
        "n": [100, 100, 150, 150],
        "bias": [0.1, 0.2, 0.05, 0.1],
        "rmse": [0.2, 0.3, 0.15, 0.25],
        "x": [1.0, 1.0, 2.0, 2.0],
        "y": [2.0, 2.0, 3.0, 3.0],
    })

    skill_data = _SkillData(df)

    # Select single model
    filtered = skill_data.sel(model="m1")
    assert len(filtered._df) == 2
    assert all(filtered._df["model"] == "m1")

    # Select multiple observations
    filtered = skill_data.sel(observation=["obs1", "obs2"])
    assert len(filtered._df) == 4

    # Chain selections
    filtered = skill_data.sel(model="m1").sel(observation="obs1")
    assert len(filtered._df) == 1


def test_skill_data_from_multiindex():
    """Test creating _SkillData from MultiIndex DataFrame"""
    df = pd.DataFrame({
        "n": [100, 100],
        "bias": [0.1, 0.2],
        "rmse": [0.2, 0.3],
        "x": [1.0, 1.0],
        "y": [2.0, 2.0],
    })
    df.index = pd.MultiIndex.from_tuples(
        [("obs1", "m1"), ("obs1", "m2")],
        names=["observation", "model"]
    )

    skill_data = _SkillData.from_multiindex(df)

    assert len(skill_data._df) == 2
    assert "observation" in skill_data._df.columns
    assert "model" in skill_data._df.columns
    assert "observation" in skill_data.dims.all
    assert "model" in skill_data.dims.all


def test_skill_data_to_display():
    """Test converting _SkillData to display format (MultiIndex)"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs1", "obs2"],
        "model": ["m1", "m2", "m1"],
        "n": [100, 100, 150],
        "bias": [0.1, 0.2, 0.05],
        "rmse": [0.2, 0.3, 0.15],
    })

    skill_data = _SkillData(df)
    display_df = skill_data.to_display()

    # Should have MultiIndex with both observation and model
    assert isinstance(display_df.index, pd.MultiIndex)
    assert "observation" in display_df.index.names
    assert "model" in display_df.index.names

    # Metrics should be columns
    assert "n" in display_df.columns
    assert "bias" in display_df.columns
    assert "rmse" in display_df.columns


def test_skill_data_to_display_single_dimension():
    """Test that single-value dimensions can be reduced from index"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs1"],  # Single value
        "model": ["m1", "m2"],  # Multiple values
        "n": [100, 100],
        "bias": [0.1, 0.2],
    })

    skill_data = _SkillData(df)

    # Default: all dimensions in index
    display_df = skill_data.to_display()
    assert isinstance(display_df.index, pd.MultiIndex)
    assert "observation" in display_df.index.names
    assert "model" in display_df.index.names

    # With reduce_index: only multi-value dimensions in index
    display_df_reduced = skill_data.to_display(reduce_index=True)
    assert display_df_reduced.index.name == "model"
    assert "observation" in display_df_reduced.columns  # Single-value dim is column


def test_skill_data_get_unique_values():
    """Test getting unique values for a dimension"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs1", "obs2"],
        "model": ["m1", "m2", "m1"],
        "n": [100, 100, 150],
    })

    skill_data = _SkillData(df)

    assert set(skill_data.get_unique_values("observation")) == {"obs1", "obs2"}
    assert set(skill_data.get_unique_values("model")) == {"m1", "m2"}
    assert skill_data.get_unique_values("quantity") == []  # Not present


def test_skill_data_aggregate():
    """Test aggregating _SkillData"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs1", "obs2", "obs2"],
        "model": ["m1", "m1", "m1", "m1"],
        "n": [50, 50, 75, 75],
        "bias": [0.1, 0.2, 0.05, 0.1],
        "rmse": [0.2, 0.3, 0.15, 0.25],
        "x": [1.0, 1.0, 2.0, 2.0],
        "y": [2.0, 2.0, 3.0, 3.0],
    })

    skill_data = _SkillData(df)

    # Aggregate across observations (keep model)
    aggregated = skill_data.aggregate(by=["model"])

    assert len(aggregated._df) == 1  # Single model
    assert aggregated._df["n"].iloc[0] == 250  # Sum of n
    # Mean of [0.1, 0.2, 0.05, 0.1] = 0.1125
    assert aggregated._df["bias"].iloc[0] == pytest.approx(0.1125)  # Mean
    assert "model" in aggregated.dims.all
    assert "observation" not in aggregated.dims.all


def test_skill_data_aggregate_weighted():
    """Test weighted aggregation"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs2"],
        "model": ["m1", "m1"],
        "n": [100, 200],
        "bias": [0.1, 0.2],  # Weighted mean should be (0.1*2 + 0.2*1)/3 = 0.133
        "rmse": [0.2, 0.3],
    })

    skill_data = _SkillData(df)

    # Aggregate with weights
    weights = {"obs1": 2.0, "obs2": 1.0}
    aggregated = skill_data.aggregate(by=["model"], weights=weights)

    assert len(aggregated._df) == 1
    assert aggregated._df["n"].iloc[0] == 300  # Sum
    assert aggregated._df["bias"].iloc[0] == pytest.approx(0.133, abs=0.01)


def test_skill_data_categorical_conversion():
    """Test that dimension columns are converted to categorical"""
    df = pd.DataFrame({
        "observation": ["obs1", "obs1", "obs2"],
        "model": ["m1", "m2", "m1"],
        "n": [100, 100, 150],
    })

    skill_data = _SkillData(df)

    assert isinstance(skill_data._df["observation"].dtype, pd.CategoricalDtype)
    assert isinstance(skill_data._df["model"].dtype, pd.CategoricalDtype)
