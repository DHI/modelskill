import pytest
from modelskill.model._base import SelectedItems, _parse_items


def test_select_items_values_can_not_also_be_aux():
    items = ["wind_speed", "wind_direction", "air_temperature"]
    with pytest.raises(ValueError) as m:
        _parse_items(
            items, item="wind_speed", aux_items=["wind_speed", "wind_direction"]
        )
        msg = str(m.excinfo.value)
        assert "wind_speed" in msg
        assert "wind_direction" not in msg


def test_select_items_simple():
    res = SelectedItems.parse(["wind_speed", "wind_direction"], item="wind_speed")
    assert res.values == "wind_speed"
    assert len(res.aux) == 0
    assert "wind_speed" in res.all
    assert "wind_direction" not in res.all


def test_select_items_aux():
    res = SelectedItems.parse(
        ["wind_speed", "wind_direction"],
        item="wind_speed",
        aux_items=["wind_direction"],
    )
    assert res.values == "wind_speed"
    assert len(res.aux) == 1
    assert "wind_speed" in res.all
    assert "wind_direction" in res.all
    assert "wind_speed" not in res.aux


def test_select_items_values_can_not_also_be_aux_static():
    items = ["wind_speed", "wind_direction", "air_temperature"]

    with pytest.raises(ValueError) as m:
        SelectedItems.parse(
            items, item="wind_speed", aux_items=["wind_speed", "wind_direction"]
        )
        msg = str(m.excinfo.value)
        assert "wind_speed" in msg
        assert "wind_direction" not in msg
