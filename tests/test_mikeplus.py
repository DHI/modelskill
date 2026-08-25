import sqlite3

import numpy as np
import pandas as pd
import pytest

from modelskill.obs import NodeObservation, ReachObservation

STATION_COLUMNS = [
    "muid",
    "locationid",
    "locationtype",
    "chainagevalue",
    "assetname",
]
MEASUREMENT_COLUMNS = [
    "measurementstationid",
    "tsfilename",
    "tsitemname",
    "resitemname",
]

JUNCTION = 8
LINK = 9
TANK = 12


def station(muid, locationid, locationtype, assetname, chainagevalue=None):
    return dict(
        muid=muid,
        locationid=locationid,
        locationtype=locationtype,
        chainagevalue=chainagevalue,
        assetname=assetname,
    )


def measurement(station_muid, item, quantity, file="calib.dfs0"):
    return dict(
        measurementstationid=station_muid,
        tsfilename=rf"..\Scripts\{file}",
        tsitemname=item,
        resitemname=f"{quantity};{quantity};100450",
    )


def build_db(path, stations, measurements, *, station_columns=STATION_COLUMNS):
    conn = sqlite3.connect(str(path))
    pd.DataFrame(stations, columns=station_columns).to_sql(
        "m_Station", conn, index=False
    )
    pd.DataFrame(measurements, columns=MEASUREMENT_COLUMNS).to_sql(
        "m_Measurement", conn, index=False
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def db(tmp_path):
    """Two pressure sensors on nodes, one flow meter on a link."""
    stations = [
        station("s1", "wNode_1", JUNCTION, "PT.401"),
        station("s2", "Tank_A", TANK, "LT.410"),
        station("s3", "Pipe_7", LINK, "FT.403"),
    ]
    measurements = [
        measurement("s1", "item_pressure_1", "Pressure"),
        measurement("s2", "item_pressure_2", "Pressure"),
        measurement("s3", "item_flow_1", "Flow"),
    ]
    return build_db(tmp_path / "mikeplus.sqlite", stations, measurements)


def frame(*items):
    """A data source holding one timeseries per named item."""
    time = pd.date_range("2024-01-01", periods=24, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {item: rng.normal(35.0, 1.0, len(time)) for item in items}, index=time
    )


def test_junction_and_tank_resolve_to_nodes(db):
    obs_list = NodeObservation.from_multiple(
        data=frame("item_pressure_1", "item_pressure_2"), db=db, quantity="Pressure"
    )

    assert {obs.at for obs in obs_list} == {"wNode_1", "Tank_A"}


def test_link_without_chainage_resolves_to_a_reach(db):
    (obs,) = ReachObservation.from_multiple(
        data=frame("item_flow_1"), db=db, quantity="Flow"
    )

    assert obs.reach == "Pipe_7"


def test_link_with_chainage_resolves_to_a_breakpoint(tmp_path):
    path = build_db(
        tmp_path / "chainage.sqlite",
        [station("s1", "Pipe_7", LINK, "FT.403", chainagevalue=24.5)],
        [measurement("s1", "item_flow_1", "Flow")],
    )

    (obs,) = NodeObservation.from_multiple(
        data=frame("item_flow_1"), db=path, quantity="Flow"
    )

    assert obs.at == ("Pipe_7", 24.5)


def test_several_items_at_one_location_all_survive(tmp_path):
    path = build_db(
        tmp_path / "shared.sqlite",
        [
            station("s1", "wNode_1", JUNCTION, "PT.401"),
            station("s2", "wNode_1", JUNCTION, "PT.402"),
        ],
        [
            measurement("s1", "before_valve", "Pressure"),
            measurement("s2", "after_valve", "Pressure"),
        ],
    )

    obs_list = NodeObservation.from_multiple(
        data=frame("before_valve", "after_valve"), db=path, quantity="Pressure"
    )

    assert [obs.name for obs in obs_list] == ["PT.401", "PT.402"]
    assert {obs.at for obs in obs_list} == {"wNode_1"}


def test_name_falls_back_to_item_name_when_assetnames_collide(tmp_path):
    path = build_db(
        tmp_path / "collide.sqlite",
        [
            station("s1", "wNode_1", JUNCTION, "same"),
            station("s2", "wNode_2", JUNCTION, "same"),
        ],
        [
            measurement("s1", "item_a", "Pressure"),
            measurement("s2", "item_b", "Pressure"),
        ],
    )

    obs_list = NodeObservation.from_multiple(
        data=frame("item_a", "item_b"), db=path, quantity="Pressure"
    )

    assert [obs.name for obs in obs_list] == ["item_a", "item_b"]


def test_quantity_is_inferred_when_unambiguous(db):
    obs_list = NodeObservation.from_multiple(
        data=frame("item_pressure_1", "item_pressure_2"), db=db
    )

    assert {obs.quantity.name for obs in obs_list} == {"Pressure"}


def test_ambiguous_quantity_raises_and_lists_options(tmp_path):
    path = build_db(
        tmp_path / "two_quantities.sqlite",
        [
            station("s1", "wNode_1", JUNCTION, "PT.401"),
            station("s2", "wNode_2", JUNCTION, "LT.410"),
        ],
        [
            measurement("s1", "item_pressure", "Pressure"),
            measurement("s2", "item_level", "Water Level"),
        ],
    )

    with pytest.raises(ValueError, match="cannot be inferred") as excinfo:
        NodeObservation.from_multiple(
            data=frame("item_pressure", "item_level"), db=path
        )

    assert "Pressure" in str(excinfo.value)
    assert "Water Level" in str(excinfo.value)


def test_the_observation_kind_narrows_the_pool_used_for_inference(db):
    # The database holds pressure on two nodes and flow on a reach. Asking for
    # node observations leaves only one quantity, so it needs no naming.
    obs_list = NodeObservation.from_multiple(
        data=frame("item_pressure_1", "item_pressure_2", "item_flow_1"), db=db
    )

    assert {obs.quantity.name for obs in obs_list} == {"Pressure"}
    assert len(obs_list) == 2


def test_unknown_quantity_raises(db):
    with pytest.raises(ValueError, match="not found"):
        NodeObservation.from_multiple(
            data=frame("item_pressure_1"), db=db, quantity="Discharge"
        )


def test_asking_for_a_node_when_the_station_is_a_reach_raises(db):
    with pytest.raises(ValueError, match="not 'node'"):
        NodeObservation.from_multiple(data=frame("item_flow_1"), db=db, quantity="Flow")


def test_missing_items_raise_and_separate_the_two_causes(db):
    with pytest.raises(ValueError) as excinfo:
        NodeObservation.from_multiple(
            data=frame("item_pressure_1", "PT.401", "never_heard_of_it"),
            db=db,
            quantity="Pressure",
        )

    message = str(excinfo.value)
    assert "no measurement registered" in message
    assert "PT.401" in message
    assert "Not found in the database" in message
    assert "never_heard_of_it" in message


def test_missing_items_can_be_skipped(db):
    obs_list = NodeObservation.from_multiple(
        data=frame("item_pressure_1", "never_heard_of_it"),
        db=db,
        quantity="Pressure",
        on_missing="skip",
    )

    assert [obs.name for obs in obs_list] == ["PT.401"]


def test_source_selects_between_files(tmp_path):
    path = build_db(
        tmp_path / "files.sqlite",
        [station("s1", "wNode_1", JUNCTION, "PT.401")],
        [
            measurement("s1", "item_a", "Pressure", file="main.dfs0"),
            measurement("s1", "item_a", "Pressure", file="other.dfs0"),
        ],
    )

    obs_list = NodeObservation.from_multiple(
        data=frame("item_a"), db=path, quantity="Pressure", source="main.dfs0"
    )

    assert len(obs_list) == 1


def test_source_accepts_a_full_path(tmp_path):
    path = build_db(
        tmp_path / "fullpath.sqlite",
        [station("s1", "wNode_1", JUNCTION, "PT.401")],
        [measurement("s1", "item_a", "Pressure", file="main.dfs0")],
    )

    obs_list = NodeObservation.from_multiple(
        data=frame("item_a"),
        db=path,
        quantity="Pressure",
        source="/some/where/main.dfs0",
    )

    assert len(obs_list) == 1


def test_item_registered_against_several_files_raises_without_source(tmp_path):
    path = build_db(
        tmp_path / "ambiguous.sqlite",
        [station("s1", "wNode_1", JUNCTION, "PT.401")],
        [
            measurement("s1", "item_a", "Pressure", file="main.dfs0"),
            measurement("s1", "item_a", "Pressure", file="other.dfs0"),
        ],
    )

    with pytest.raises(ValueError, match="more than one file"):
        NodeObservation.from_multiple(
            data=frame("item_a"), db=path, quantity="Pressure"
        )


def test_unknown_locationtype_raises(tmp_path):
    path = build_db(
        tmp_path / "weird.sqlite",
        [station("s1", "wNode_1", 99, "PT.401")],
        [measurement("s1", "item_a", "Pressure")],
    )

    with pytest.raises(ValueError, match="unsupported locationtype"):
        NodeObservation.from_multiple(
            data=frame("item_a"), db=path, quantity="Pressure"
        )


def test_accepts_an_open_connection(db):
    conn = sqlite3.connect(db)
    try:
        (obs,) = ReachObservation.from_multiple(
            data=frame("item_flow_1"), db=conn, quantity="Flow"
        )
    finally:
        conn.close()

    assert obs.reach == "Pipe_7"


def test_missing_table_raises(tmp_path):
    path = str(tmp_path / "empty.sqlite")
    conn = sqlite3.connect(path)
    pd.DataFrame({"a": [1]}).to_sql("something_else", conn, index=False)
    conn.close()

    with pytest.raises(ValueError, match="missing table"):
        NodeObservation.from_multiple(data=frame("item_a"), db=path)


def test_missing_column_raises(tmp_path):
    path = build_db(
        tmp_path / "thin.sqlite",
        [
            dict(muid="s1", locationid="wNode_1", locationtype=JUNCTION, assetname="a"),
        ],
        [measurement("s1", "item_a", "Pressure")],
        station_columns=["muid", "locationid", "locationtype", "assetname"],
    )

    with pytest.raises(ValueError, match="missing column"):
        NodeObservation.from_multiple(data=frame("item_a"), db=path)


@pytest.fixture
def calibration_data():
    """A data source holding both pressure and flow items."""
    time = pd.date_range("2024-01-01", periods=24, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "item_pressure_1": rng.normal(35.0, 1.0, len(time)),
            "item_pressure_2": rng.normal(36.0, 1.0, len(time)),
            "item_flow_1": rng.normal(120.0, 5.0, len(time)),
        },
        index=time,
    )


class TestNodeObservationFromDatabase:
    def test_builds_one_observation_per_item(self, db, calibration_data):
        obs_list = NodeObservation.from_multiple(
            data=calibration_data, db=db, quantity="Pressure"
        )

        assert len(obs_list) == 2
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)
        assert [obs.at for obs in obs_list] == ["wNode_1", "Tank_A"]

    def test_names_come_from_the_database(self, db, calibration_data):
        obs_list = NodeObservation.from_multiple(
            data=calibration_data, db=db, quantity="Pressure"
        )

        assert [obs.name for obs in obs_list] == ["PT.401", "LT.410"]

    def test_quantity_comes_from_the_database(self, db, calibration_data):
        obs_list = NodeObservation.from_multiple(
            data=calibration_data, db=db, quantity="Pressure"
        )

        assert all(obs.quantity.name == "Pressure" for obs in obs_list)

    def test_data_is_selected_per_item(self, db, calibration_data):
        obs_list = NodeObservation.from_multiple(
            data=calibration_data, db=db, quantity="Pressure"
        )

        expected = calibration_data["item_pressure_1"].to_numpy()
        assert obs_list[0].values == pytest.approx(expected)

    def test_quantity_is_inferred_when_only_nodes_are_wanted(
        self, db, calibration_data
    ):
        obs_list = NodeObservation.from_multiple(data=calibration_data, db=db)

        assert len(obs_list) == 2
        assert all(obs.quantity.name == "Pressure" for obs in obs_list)

    def test_several_sensors_at_one_node_are_all_kept(self, tmp_path):
        path = build_db(
            tmp_path / "shared.sqlite",
            [
                station("s1", "wNode_1", JUNCTION, "PT.401"),
                station("s2", "wNode_1", JUNCTION, "PT.402"),
            ],
            [
                measurement("s1", "before_valve", "Pressure"),
                measurement("s2", "after_valve", "Pressure"),
            ],
        )
        time = pd.date_range("2024-01-01", periods=5, freq="h")
        data = pd.DataFrame(
            {"before_valve": range(5), "after_valve": range(5, 10)}, index=time
        )

        obs_list = NodeObservation.from_multiple(data=data, db=path)

        assert [obs.at for obs in obs_list] == ["wNode_1", "wNode_1"]
        assert [obs.name for obs in obs_list] == ["PT.401", "PT.402"]

    def test_flow_on_a_link_points_at_reach_observation(self, db, calibration_data):
        with pytest.raises(ValueError, match="not 'node'"):
            NodeObservation.from_multiple(data=calibration_data, db=db, quantity="Flow")

    def test_unresolvable_item_raises(self, db, calibration_data):
        data = calibration_data.rename(columns={"item_pressure_1": "mystery_sensor"})

        with pytest.raises(ValueError, match="could not be resolved"):
            NodeObservation.from_multiple(data=data, db=db, quantity="Pressure")

    def test_unresolvable_item_can_be_skipped(self, db, calibration_data):
        data = calibration_data.rename(columns={"item_pressure_1": "mystery_sensor"})

        obs_list = NodeObservation.from_multiple(
            data=data, db=db, quantity="Pressure", on_missing="skip"
        )

        assert [obs.name for obs in obs_list] == ["LT.410"]

    def test_db_and_nodes_are_mutually_exclusive(self, db, calibration_data):
        with pytest.raises(ValueError, match="mutually exclusive"):
            NodeObservation.from_multiple(
                data=calibration_data, db=db, nodes={1: "item_pressure_1"}
            )

    def test_db_without_data_raises(self, db):
        with pytest.raises(ValueError, match="'data' is required"):
            NodeObservation.from_multiple(db=db)

    def test_quantity_string_without_db_raises(self, calibration_data):
        with pytest.raises(TypeError, match="must be a Quantity"):
            NodeObservation.from_multiple(
                data=calibration_data,
                nodes={1: "item_pressure_1"},
                quantity="Pressure",
            )


class TestReachObservationFromDatabase:
    def test_builds_reach_observations(self, db, calibration_data):
        obs_list = ReachObservation.from_multiple(
            data=calibration_data, db=db, quantity="Flow"
        )

        assert len(obs_list) == 1
        assert isinstance(obs_list[0], ReachObservation)
        assert obs_list[0].reach == "Pipe_7"
        assert obs_list[0].name == "FT.403"
        assert obs_list[0].quantity.name == "Flow"

    def test_quantity_is_inferred_when_only_reaches_are_wanted(
        self, db, calibration_data
    ):
        obs_list = ReachObservation.from_multiple(data=calibration_data, db=db)

        assert [obs.reach for obs in obs_list] == ["Pipe_7"]

    def test_pressure_on_a_node_points_at_node_observation(self, db, calibration_data):
        with pytest.raises(ValueError, match="not 'reach'"):
            ReachObservation.from_multiple(
                data=calibration_data, db=db, quantity="Pressure"
            )

    def test_db_and_reaches_are_mutually_exclusive(self, db, calibration_data):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ReachObservation.from_multiple(
                data=calibration_data, db=db, reaches={"r1": "item_flow_1"}
            )
