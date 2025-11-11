import modelskill as ms


def test_network_model_result_has_name():
    mod = ms.NetworkModelResult(
        "tests/testdata/network/Vida_1BaseDefault_Network_HD.res1d",
        name="Vida",
        item="Discharge",
    )

    assert mod.name == "Vida"
