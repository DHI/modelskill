import requests
import pandas as pd
from datetime import datetime


class DMIOceanObsRepository:
    """Get Ocean observations from DMI"""

    def __init__(self, apikey: str) -> None:

        self.__api__key = apikey

    def get_observations(
        self,
        *,
        station_id: str,
        parameter_id="sealev_dvr",
        start_time: datetime = None,
        end_time: datetime = None,
        n=1000,
    ) -> pd.DataFrame:
        """
        Get ocean observations from DMI

        Parameters
        ==========
        station_id: str
            Id of station, e.g. "30336" # Kbh. havn
        parameter_id: str, optional
            Select one of "sea_reg", "sealev_dvr", "sealev_ln", "tw", default  is "sealev_dvr"
        start_time: datetime, optional
            Start time of  interval.
        end_time: datetime, , optional
            Start time of  interval.
        n: int
            Max number of observations to return, default 1000, max value: 300000

        Returns
        =======
        pd.DataFrame

        Examples
        ========
        >>>
        repo = DMIOceanObsRepository(apikey="e11...")

        df = repo.get_observations(
            station_id="30336", start_time=datetime(2018, 3, 4, 0, 0)
        )

        """

        available_parameters = {"sea_reg", "sealev_dvr", "sealev_ln", "tw"}
        if parameter_id not in available_parameters:
            raise ValueError(
                f"Selected parameter: {parameter_id} not available. Choose one of {available_parameters}"
            )

        params = {
            "api-key": self.__api__key,
            "stationId": station_id,
            "parameterId": parameter_id,
            "limit": n,
        }

        if start_time or end_time:

            if start_time and end_time is None:
                params["datetime"] = f"{start_time.isoformat()}Z/.."
            elif end_time and start_time is None:
                params["datetime"] = f"../{end_time.isoformat()}Z"
            else:
                params[
                    "datetime"
                ] = f"{start_time.isoformat()}Z/{end_time.isoformat()}Z"

        resp = requests.get(
            "https://dmigw.govcloud.dk/v2/oceanObs/collections/observation/items",
            params=params,
        )
        if not resp.ok:
            raise Exception(
                f"Failed to retrieve data for station: {station_id} from DMI API. Response: {resp.text}"
            )

        data = resp.json()
        ts = [
            {
                "time": p["properties"]["observed"].replace("Z", ""),
                parameter_id: p["properties"]["value"],
            }
            for p in data["features"]
        ]
        df = pd.DataFrame(ts)

        if parameter_id in {"sea_reg", "sealev_dvr", "sealev_ln"}:
            df[parameter_id] = df[parameter_id] / 100.0  # cm -> m

        df.index = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])
        df = df.sort_index()

        return df
