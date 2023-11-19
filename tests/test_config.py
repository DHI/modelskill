import modelskill as ms


def test_comparison_from_dict():
    # As an alternative to
    # mr = ModelResult()

    # o1 = PointObservation()
    # con = Connector(o1, mr)
    # c = con.extract()

    configuration = dict(
        modelresults=dict(
            HD=dict(
                filename="tests/testdata/Oresund2D.dfsu",
                item=0,
            ),
        ),
        observations=dict(
            klagshamn=dict(
                filename="tests/testdata/obs_two_items.dfs0",
                item=1,
                x=366844,
                y=6154291,
                name="Klagshamn2",
            ),
            Drogden=dict(
                filename="tests/testdata/dmi_30357_Drogden_Fyr.dfs0",
                item=0,
                x=355568.0,
                y=6156863.0,
            ),
        ),
    )

    con = ms.from_config(configuration, validate_eum=False)
    cc = con.extract()
    assert len(cc) == 2
    assert cc.n_comparers == 2
    assert cc.n_models == 1


def test_comparison_from_yml():
    con = ms.from_config("tests/testdata/conf.yml", validate_eum=False)
    cc = con.extract()

    assert len(cc) == 2
    assert cc.n_comparers == 2
    assert cc.n_models == 1
    assert con.observations["Klagshamn"].quantity.name == "Water Level"
