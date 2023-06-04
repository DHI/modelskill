import pandas as pd

from modelskill.plot import format_skill_df


def test_format_skill_df():

    #
    #    	            n	bias	rmse	urmse	mae	cc	si	r2
    # observation
    # smhi_2095_klagshamn	167	1.033099e-09	0.040645	0.040645	0.033226	0.841135	0.376413	0.706335

    df = pd.DataFrame(
        {
            "n": [167],
            "bias": [1.033099e-09],
            "rmse": [0.040645],
            "urmse": [0.040645],
            "mae": [0.033226],
            "cc": [0.841135],
            "si": [0.376413],
            "r2": [0.706335],
        },
        index=["smhi_2095_klagshamn"],
    )

    lines = format_skill_df(df, units="degC")
    assert "N     =  167" in lines[0]
    assert "BIAS  =  0.00 degC" in lines[1]
    assert "RMSE  =  0.04 degC" in lines[2]
    assert "URMSE =  0.04 degC" in lines[3]
    assert "MAE   =  0.03 degC" in lines[4]
    assert "CC    =  0.84 " in lines[5]
