from typing import Dict
import numpy as np
import requests
import pandas as pd
from datetime import datetime


class DMIOceanObsRepository:
    """Get Ocean observations from DMI

    Notes
    =====
    Get a API key here: https://confluence.govcloud.dk/pages/viewpage.action?pageId=26476690

    Examples
    ========
    >>> dmi = DMIOceanObsRepository(api_key="e11...")
    >>> dmi.stations[dmi.stations.name.str.startswith('Køg')]
       station_id      lon      lat          name
    43      30478  12.1965  55.4555   Køge Havn I
    54      30479  12.1965  55.4555  Køge Havn II
    >>> df = dmi.get_observed_data(station_id="30478", start_time=datetime(2018, 3, 4))
    """

    def __init__(self, api_key: str) -> None:

        self.__api__key = api_key
        self._stations = None

    def get_observed_data(
        self,
        *,
        station_id: str,
        parameter_id="sealev_dvr",
        start_time: datetime = None,
        end_time: datetime = None,
        limit=100000,
        records_per_request=10000,
    ) -> pd.DataFrame:
        """
        Get ocean observations from DMI

        For historical data, always specify both a start_time and an end_time.

        Parameters
        ==========
        station_id: str
            Id of station, e.g. "30336" # Kbh. havn
        parameter_id: str, optional
            Select one of "sea_reg", "sealev_dvr", "sealev_ln", "tw", default  is "sealev_dvr"
        start_time: datetime, optional
            Start time of  interval.
        end_time: datetime, optional
            End time of  interval.
        limit: int, optional
            Max number of observations to return default 10000
        records_per_request: int, optional
            Tunable parameter for optimal performance, default value is 100000

        Returns
        =======
        pd.DataFrame

        Examples
        ========
        >>> from fmskill.data.dmi import DMIOceanObsRepository
        >>> dmi = DMIOceanObsRepository(api_key="e11...")
        >>> df = dmi.get_observed_data(station_id="30336", start_time="2018-03-04", end_time="2018-03-06")
        >>> df.head()
                            sealev_dvr
        time
        2018-03-04 00:00:00       -0.22
        2018-03-04 00:10:00       -0.23
        2018-03-04 00:20:00       -0.26
        2018-03-04 00:30:00       -0.27
        2018-03-04 00:40:00       -0.28

        >>> df = dmi.get_observed_data(station_id="30336", parameter_id = "tw", start_time="2018-07-01", end_time="2018-07-06")
        >>> df.head()
                            tw
        time
        2018-07-01 00:00:00  18.2
        2018-07-01 00:10:00  18.2
        2018-07-01 00:20:00  18.2
        2018-07-01 00:30:00  18.2
        2018-07-01 00:40:00  18.2
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
            "limit": records_per_request,
        }

        if start_time and isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if end_time and isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)

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

        ts = self._data_to_ts(data, parameter_id)

        if len(ts) == 0:
            return pd.DataFrame({parameter_id: []})

        next_link = None
        if len(data["links"]) > 1:
            next_link = data["links"][1]["href"]

        while next_link and (len(ts) < limit):

            resp = requests.get(next_link)
            data = resp.json()
            if data["numberReturned"] == 0:
                break
            else:
                ts = ts + self._data_to_ts(data, parameter_id)
                next_link = data["links"][1]["href"]

        df = pd.DataFrame(ts)

        if parameter_id in {"sea_reg", "sealev_dvr", "sealev_ln"}:
            df[parameter_id] = df[parameter_id] / 100.0  # cm -> m

        df.index = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])
        df = df.sort_index()

        return df

    def _data_to_ts(self, data, parameter_id):

        ts = [
            {
                "time": p["properties"]["observed"].replace("Z", ""),
                parameter_id: p["properties"]["value"],
            }
            for p in data["features"]
        ]
        return ts

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
        """Get DMI stations as a dataframe.

        Returns
        -------
        pd.DataFrame
            all stations in API

        Examples
        --------
        >>> dmi.stations.head(5)
          station_id      lon      lat              name      start end
        0    9007102   8.5739  55.2764          Mandø II 2000-11-02 NaT
        1      31572  11.3474  54.6551   Rødbyhavns Havn 1990-09-21 NaT
        2    9005110   8.1259  56.3716  Thorsminde Fjord 1990-12-03 NaT
        3      29392  11.1390  55.3355       Korsør Havn 1991-09-02 NaT
        4    9005201   8.1290  56.0005  Hvide Sande Havn 1990-10-05 NaT
        """
        if self._stations is None:
            self._stations = self.get_stations_raw()

        res = []
        for s in self._stations["features"]:
            pos = s["geometry"]["coordinates"]
            end_time = s["properties"]["validTo"]
            end_time = None if end_time is None else end_time[:-1]
            row = dict(
                station_id=s["properties"]["stationId"],
                lon=pos[0],
                lat=pos[1],
                name=s["properties"]["name"],
                start=pd.to_datetime(s["properties"]["validFrom"][:-1]),
                end=pd.to_datetime(end_time),
            )
            res.append(row)
        df = pd.DataFrame(res)

        return df

    def get_stations_in_interval(self, start_time=None, end_time=None) -> pd.DataFrame:
        """Get DMI stations with data in time interval.

        Parameters
        ==========
        start_time: (str, datetime), optional
            Start time of interval. Keep only stations with end after this time.
        end_time: (str, datetime), optional
            End time of interval. Keep only stations with start before this time.

        Returns
        -------
        pd.DataFrame
            all stations with data in time interval

        Examples
        --------
        >>> df = dmi.get_stations_in_interval(end_time="1990-1-1")
        >>> df.head(5)
        station_id      lon      lat              name      start end
        0    9007102   8.5739  55.2764          Mandø II 2000-11-02 NaT
        1      31572  11.3474  54.6551   Rødbyhavns Havn 1990-09-21 NaT
        2    9005110   8.1259  56.3716  Thorsminde Fjord 1990-12-03 NaT
        3      29392  11.1390  55.3355       Korsør Havn 1991-09-02 NaT
        4    9005201   8.1290  56.0005  Hvide Sande Havn 1990-10-05 NaT
        """
        df = self.stations
        if start_time:
            start_time = pd.to_datetime(start_time)
            df = df[np.logical_or(pd.isnull(df.end), df.end > start_time)]

        if end_time:
            end_time = pd.to_datetime(end_time)
            df = df[df.start > end_time]
        return df
