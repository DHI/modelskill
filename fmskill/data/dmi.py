from typing import Dict
import requests
import pandas as pd
from datetime import datetime


class DMIOceanObsRepository:
    """Get Ocean observations from DMI

    Notes
    =====
    Get a API key here: https://confluence.govcloud.dk/pages/viewpage.action?pageId=26476690
    """

    def __init__(self, apikey: str) -> None:

        self.__api__key = apikey
        self._stations = None

    def get_observations(
        self,
        *,
        station_id: str,
        parameter_id="sealev_dvr",
        start_time: datetime = None,
        end_time: datetime = None,
        limit=1000,
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
        limit: int
            Max number of observations to return, default 1000, max value: 300000

        Returns
        =======
        pd.DataFrame

        Examples
        ========
        >>> repo = DMIOceanObsRepository(apikey="e11...")

        >>> df = repo.get_observations(station_id="30336", start_time=datetime(2018, 3, 4, 0, 0))
        >>> df.head()
                            sealev_dvr
        time
        2021-06-01 18:20:00        0.06
        2021-06-01 18:30:00        0.04
        2021-06-01 18:40:00        0.03
        2021-06-01 18:50:00        0.02
        2021-06-01 19:00:00       -0.01
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
            "limit": limit,
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

    def get_stations_raw(self) -> Dict:
        resp = requests.get(
            "https://dmigw.govcloud.dk/v2/oceanObs/collections/station/items",
            params={"api-key": self.__api__key},
        )
        if not resp.ok:
            raise Exception(
                f"Failed to retrieve station info from DMI API. Response: {resp.text}"
            )

        data = resp.json()

        return data

    @property
    def stations(self) -> pd.DataFrame:
        if self._stations is None:
            self._stations = self.get_stations_raw()

        res = []
        for s in self._stations["features"]:
            pos = s["geometry"]["coordinates"]
            row = dict(
                station_id=s["properties"]["stationId"],
                lon=pos[0],
                lat=pos[1],
                name=s["properties"]["name"],
            )
            res.append(row)

        return pd.DataFrame(res)
