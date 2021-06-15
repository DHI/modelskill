from datetime import datetime
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mikeio import Dfs0, eum


class APIAuthenticationFailed(Exception):
    pass


class DownloadDataFromAPIFailed(Exception):
    pass


class InvalidSatelliteName(Exception):
    pass


class AltimetryData:
    def __init__(self, df, area=None, requested_start=None, requested_end=None):
        self.df = df
        self.area = area
        self._requested_start = requested_start
        self._requested_end = requested_end

    @property
    def satellites(self):
        return list(self.df.satellite.unique())

    @property
    def start_time(self):
        return self.df.index[0].to_pydatetime()

    @property
    def end_time(self):
        return self.df.index[0].to_pydatetime()

    def to_dfs0(self, filename, satellite=None, quality=0):
        """Save altimetry data to dfs0 file.

        Parameters
        ----------
        filename : str
            path to new dfs0 file
        satellite : str, optional
            short name of satellite to be saved, by default all
        """
        df = self.df
        if satellite is not None:
            df = df[df.satellite == satellite]

        if len(df) < 1:
            raise Exception("No data in data frame")

        cols = ["lon", "lat", "adt_dhi", "swh", "wind_speed"]
        items = []
        items.append(eum.ItemInfo("Longitude", eum.EUMType.Latitude_longitude))
        items.append(eum.ItemInfo("Latitude", eum.EUMType.Latitude_longitude))
        items.append(eum.ItemInfo("Water Level", eum.EUMType.Water_Level))
        items.append(
            eum.ItemInfo("Significant Wave Height", eum.EUMType.Significant_wave_height)
        )
        items.append(eum.ItemInfo("Wind Speed", eum.EUMType.Wind_speed))

        df[cols].to_dfs0(filename, items=items)

    def plot_map(self, fig_size=(12, 10)):
        """plot map of altimetry data

        Parameters
        ----------
        fig_size : Tuple(float), optionally
            size of figure, by default (12,10)
        """
        df = self.df

        plt.style.use("seaborn-whitegrid")
        plt.figure(figsize=fig_size)
        markers = ["o", "x", ".", ",", "+", "v", "^", "<", ">", "s", "d"]
        j = 0
        for sat in self.satellites:
            dfsub = df[df.satellite == sat]
            plt.plot(dfsub.lon, dfsub.lat, markers[j], label=sat, markersize=1)
            j = j + 1
        plt.legend(numpoints=1)
        plt.title(f"Altimetry data between {self.start_time} and {self.end_time}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    @staticmethod
    def from_csv(filename):
        """read altimetry data from csv file instead of api

        Parameters
        ----------
        filename : str
            path to csv file

        Returns
        -------
        DataFrame
            With datetime index containing the altimetry data
        """
        df = pd.read_csv(filename, parse_dates=True, index_col="date")
        print(f"Succesfully read {len(df)} rows from file {filename}")
        return AltimetryData(df)

    def get_dataframe_per_satellite(self, df=None):
        if df is None:
            df = self._data
        res = {}
        sats = self.get_satellites(df)
        for sat in sats:
            dfsub = df[df.satellite == sat]
            res[sat] = dfsub  # .drop(['satellite'], axis=1)
        return res

    def assign_track_id(self, data=None, max_jump=3):
        # loop over missions, then time to find consecutive points
        if data is None:
            data = self._data
        id = 0
        sats = self.get_satellites(data)

        # 1 step (=1second = 7.2km)
        # max_jump = 10  # what is an acceptable skip length 60?

        df = data.copy()

        if "track_id" not in df.columns:
            ids = np.zeros((len(df),), dtype=int)
            df.insert(len(df.columns), "track_id", ids, True)

        # find tracks for each satellite
        tot_tracks = 0
        for sat in sats:
            dfsub = df[df.satellite == sat]
            tvec = np.asarray([dt.timestamp() for dt in dfsub.index.to_pydatetime()])
            wl = dfsub.adt_dhi
            dtvec = np.zeros(np.size(tvec))
            nt = len(dtvec)
            dtvec[1:] = np.diff(tvec)

            ids = np.zeros(tvec.shape, dtype=int) - 1  # default is -1
            idx = 0
            ni = 0
            for j in range(nt):
                if (dtvec[j] > max_jump) & (ni > 0):
                    idx = idx + 1
                    ni = 0

                # only assign track id if actual data?
                # if not np.isnan(wl[j]):
                ids[j] = idx
                ni = ni + 1

            tot_tracks = tot_tracks + idx
            df.loc[df.satellite == sat, "track_id"] = ids

        print(f"Identified {tot_tracks} individual passings")

        return df

    def print_records_per_satellite(self, df, details=1):
        if df is None:
            df = self._data
        sats = self.get_satellites()
        print(f"For the selected area between {self.start_date} and {self.end_date}:")
        for sat in sats:
            dfsub = df[df.satellite == sat]
            print(f"Satellite {sat} has {len(dfsub)} records")
            if details > 1:
                print(dfsub.drop(["lon", "lat"], axis=1).describe())


class DHIAltimetryRepository:
    API_URL = "https://altimetry-shop-data-api.dhigroup.com/"
    HEADERS = None
    api_key = None

    def __init__(self, api_key):
        self.api_key = api_key
        self.HEADERS = {"authorization": api_key}
        self._api_conf = None

    @property
    def _conf(self):
        if self._api_conf is None:
            self._api_conf = self._get_config()
        return self._api_conf

    def _get_config(self):
        r = requests.get((self.API_URL + "/config"), headers=self.HEADERS)
        r.raise_for_status()
        return r.json()

    def get_satellites(self):
        sats = self._conf.get("satellites")
        return pd.DataFrame(sats).set_index("short_name")

    def get_quality_filters(self):
        sats = self._conf.get("quality_filters")
        return pd.DataFrame(sats).set_index("short_name")

    def get_observation_stats(self):
        r = requests.get(
            (self.API_URL + ("/observations-stats")),
            headers=self.HEADERS,
        )
        r.raise_for_status()
        stats = r.json()["stats"]
        df = pd.DataFrame(stats).set_index("short_name")
        df["min_date"] = pd.to_datetime(df["min_date"])
        df["max_date"] = pd.to_datetime(df["max_date"])
        return df

    def get_temporal_coverage(
        self, area, start_time="20200101", end_time=None, satellites=""
    ):
        query_url = self._create_coverage_query(
            "temporal",
            area=area,
            start_time=start_time,
            end_time=end_time,
            satellites=satellites,
        )
        r = requests.get(query_url, headers=self.HEADERS)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data["temporal_coverage"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_spatial_coverage(
        self, area, start_time="20200101", end_time=None, satellites=""
    ):
        query_url = self._create_coverage_query(
            "spatial",
            area=area,
            start_time=start_time,
            end_time=end_time,
            satellites=satellites,
        )
        r = requests.get(query_url, headers=self.HEADERS)
        r.raise_for_status()
        return r.json()

    def _create_coverage_query(
        self, type, area, start_time="20200101", end_time=None, satellites=""
    ):

        query = self.API_URL + f"{type}-coverage?"
        query += self._validate_area(area)

        satellites = self.parse_satellites(satellites)
        if satellites:
            query += "&satellites=" + ",".join(satellites) if satellites else ""

        start_time = self._parse_datetime(start_time)
        if start_time:
            query += "&start_date=" + start_time

        if end_time is None:
            end_time = datetime.now()
        end_time = self._parse_datetime(end_time)
        if end_time:
            query += "&end_date=" + end_time
        return query

    def get_altimetry_data(
        self,
        area,
        start_time="20200101",
        end_time=None,
        satellites="",
        quality_filter="",
    ):
        """Main function that creates url and retrieves altimetry data from api

        Parameters
        ----------
        area : str
            String specifying location of desired data.  The three forms allowed by the API are:
                - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
                - bbox=115,28,150,52
                - lon=10.9&lat=55.9&radius=100000
            A few named domains can also be used:
                - GS_NorthSea, GS_BalticSea, GS_SouthChinaSea
        start_date : str, datetime, optional
            Start of data to be retrieved, by default '20200101'
        end_date : str, datetime, optional
            End of data to be retrieved, by default datetime.now()
        satellites : str, list of str, optional
            Satellites to be downloaded, e.g. '', '3a', 'j3, by default ''
        quality_filter : str, optional
            Name of quality filter, e.g. 'dhi_combined', by default '' meaning no filter

        Returns
        -------
        DataFrame
            With columns 'lon', 'lat', 'adt', 'adt_dhi', 'swh', 'swh_rms' 'wind_speed', ...
        """
        if end_time is None:
            end_time = datetime.now()
        url_query = self._create_altimetry_query(
            area=area,
            start_time=start_time,
            end_time=end_time,
            quality_filter=quality_filter,
            satellites=satellites,
        )
        df = self.get_altimetry_data_raw(url_query)
        return AltimetryData(df, area=area)

    def _create_altimetry_query(
        self,
        area="bbox=-11.913345,48.592117,12.411167,63.084148",
        satellites="3a",
        start_time="20200101",
        end_time="",
        nan_value="",
        quality_filter="",
        numeric=False,
    ):
        """
        Create a query for satellite data as a URL pointing to the location of a CSV file with the data.

        Parameters
        ----------
        area : str
            String specifying location of desired data.  The three forms allowed by the API are:
                - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
                - bbox=115,28,150,52
                - lon=10.9&lat=55.9&radius=100000
            A few named domains can also be used:
                - GS_NorthSea, GS_BalticSea, GS_SouthChinaSea
        satellites : Union[List[str], str], optional
            List of short or long names of satellite to include, an empty string, or the string 'sentinels' to specify
                the two sentinel satellites 3a and 3b. Default: '3a'.
        start_time : str or datetime, optional
            First date for which data is wanted, in the format '20100101' or as an empty string. If an empty string is
                given, data starting from when it was first available is returned. Default: ''.
        end_time : str or datetime, optional
            Last date for which data is wanted, in the format '20100101' or as an empty string. If an empty string is
                given, data until the last time available is returned. Default: "20200101".
        nan_value : str, optional
            Value to use to indicate bad or missing data, or an empty string to use the default (-9999). Default: ''.
        quality_filter : str, optional
            Type of filter to apply before returning data. Currently, only the value 'dhi_combined' is allowed. If the
                empty string is given, no filter is applied. Default: 'dhi_combined'.
        numeric : bool, optional
            If True, return columns as numeric and return fewer columns in order to comply with the Met-Ocean on Demand
                analysis systems. If False, all columns are returned, and string types are preserved as such.
                Default: False.

        Returns
        -------
        str
            URL pointing to the location of data in a CSV file.
        """

        query_url = self.API_URL + "query-csv?"

        area = self._validate_area(area)
        satellites = self.parse_satellites(satellites)
        satellite_str = "satellites=" + ",".join(satellites) if satellites else ""

        if numeric:
            numeric_string = "numeric=true"
        else:
            numeric_string = "numeric=false"

        quality_filter = self._validate_quality_filter(quality_filter)
        if quality_filter != "":
            quality_filter = "qual_filters=" + quality_filter

        if end_time:
            end_time = self._parse_datetime(end_time)
            end_time = "end_date=" + end_time

        if start_time:
            start_time = self._parse_datetime(start_time)
            start_time = "start_date=" + start_time

        argument_strings = [
            arg_string
            for arg_string in [
                area,
                satellite_str,
                start_time,
                end_time,
                nan_value,
                quality_filter,
                numeric_string,
            ]
            if arg_string != ""
        ]

        query_url = "".join([query_url, "&".join(argument_strings)])
        return query_url

    def _validate_area(self, area):
        # polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
        # bbox=115.0,28.5,150.2,52.1
        # lon=10.9&lat=55.9&radius=10.0
        parsed = True
        message = None
        if area == "GS_NorthSea":
            area = "polygon=9.74351,55.87496,13.17873,56.09956,13.44641,56.96066,10.72500,60.56592,9.02970,63.01088,-12.29205,63.2,-12.64895,48.54273,-3.55120,47.97842,8.67280,50.31249,9.74351,55.87496"
        elif area == "GS_BalticSea":
            area = "polygon=6.42936234081148,58.06933188935258,8.279936182852992,56.60995884195779,9.77463043988601,53.640768085976816,21.58983266214966,53.725076403620875,30.735937996854545,59.86677430413451,28.60066048680588,61.31689046459479,22.08806408116112,60.36326141464659,22.83541120967695,62.70289239987201,27.426257856280074,65.61608475872717,22.47953162466976,66.52536961810017,19.73925882010812,64.24418831137172,15.860171343522126,61.28270990295982,15.860171343522126,56.76631664113353,13.511366082469124,56.25577473222242,10.87785715341036,60.64361921650416,8.42228801685593,58.629547789876256,6.42936234081148,58.06933188935258"
        elif area == "GS_SouthChinaSea":
            area = "polygon=98.86609661997,8.761756775260409,96.16575957545422,4.8984661910612886,102.51949379784571,-1.9254572721213634,100.45453017556935,-10.366906480390398,123.08970834284287,-11.61431933913046,125.47235867623976,-16.466767514033222,133.65279148756986,-13.860707843374854,136.11486349874792,-12.778732842985974,138.73577886548395,-9.349684246097908,139.76826067662364,-5.807512090082426,137.14734530988608,-2.877669323455109,138.4975138321447,35.63428101232233,133.09683974311162,35.11621506468174,128.56980410965576,37.231645155575706,125.55178035401951,40.44749626033976,121.18358807612458,41.823204815858105,117.6890342538083,40.80915702051942,116.5771307648904,37.168383431979905,120.70705800944455,29.06545223449251,114.35332378705306,23.58737830287147,106.25231265350266,22.050118028915463,104.425614064564,18.323806173034896,108.31727627577902,13.127339365868266,104.82272245346422,12.11983013962174,99.18378333109047,14.8995938250698,98.86609661997,8.761756775260409"
        else:
            parsed = False

        if area[0:5] == "bbox=":
            if area.count(",") == 3:
                parsed = True
            else:
                message = "bbox area should be provided as bbox=115.0,28.5,150.2,52.1"

        elif area[0:8] == "polygon=":
            if area.count(",") >= 7:
                parsed = True
            else:
                message = "polygon area should be provided as polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993"

        elif area[0:4] == "lon=":
            if (area.count("&lat=") == 1) & (area.count("&radius=") == 1):
                parsed = True
            else:
                message = (
                    "circle area should be provided as lon=10.9&lat=55.9&radius=10.0"
                )
        else:
            message = "area must be given as bbox=115.0,28.5,150.2,52.1 or polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993 or lon=10.9&lat=55.9&radius=10.0"

        if not parsed:
            raise Exception(f"Failed to parse area {area}! {message}")
        return area

    def _parse_datetime(self, date):
        """is the date accepted by the altimetry api?

        Arguments:
            date -- datetime or string in format yyyymmdd are accepted

        Raises:
            TypeError: if neither string nor datetime

        Returns:
            date -- as yyyymmdd string
        """
        if date is None:
            return None
        if not isinstance(date, (datetime, str)):
            raise TypeError(f"Date should be either str or datetime")
        if isinstance(date, str):
            date = pd.to_datetime(date)

        return date.strftime("%Y%m%d")  # %M%H

    def _validate_quality_filter(self, qa):
        """is the quality filter string allowed by the altimetry api

        Arguments:
            qa -- allowed strings: '', 'qual_swh', 'qual_wind', 'qual_combined'

        Returns:
            qa -- the accepted string
        """
        if qa == "nofilter":
            return ""
        elif (
            (qa == "")
            | (qa == "dhi_combined")
            | (qa == "qual_swh")
            | (qa == "qual_wind_speed")
        ):
            return qa
        else:
            raise Exception(
                f"Filter {qa} is unknown (dhi_combined, qual_swh, qual_wind_speed)"
            )

    def get_altimetry_data_raw(self, url_query: str) -> pd.DataFrame:
        """Request data from altimetry api

        Parameters
        ----------
        url_query : str
            query build with create_altimetry_query()

        Raises
        ------
        APIAuthenticationFailed: if api key is wrong

        Returns
        -------
        pd.DataFrame
            with altimetry data
        """
        t_start = time.time()
        r = requests.get(
            url_query,
            headers=self.HEADERS,
        )
        if r.status_code == 401:
            raise APIAuthenticationFailed
        r.raise_for_status()
        response_data = r.json()
        df = pd.read_csv(
            response_data["download_url"], parse_dates=True, index_col="date"
        )

        elapsed = time.time() - t_start
        nrecords = len(df)
        if nrecords > 0:
            print(
                f"Succesfully retrieved {nrecords} records from API in {elapsed:.2f} seconds"
            )
        else:
            print("No data retrieved!")

        return df

    def parse_satellites(self, satellites):
        """
        Parse a list of satellite names into an argument string to pass as part of a URL query.

        Parameters
        ----------
        satellites : Union[List[str], str]
            List of short or long names of satellite to include, an empty string, or the string 'sentinels' to specify
                the two sentinel satellites 3a and 3b.

        Returns
        -------
        str
            String representing argument specifying which satellites to retrieve data from.

        Raises
        --------
        InvalidSatelliteName
            If a string that is not the empty string, 'sentinels', or part of the following lists is passed:
            ['TOPEX', 'Poseidon', 'Jason-1', 'Envisat', 'Jason-2', 'SARAL', 'Jason-3',
                                'Geosat', 'GFO', 'ERS-1', 'ERS-2', 'CryoSat-2', 'Sentinel-3A', 'Sentinel-3B']
            ['tx', 'ps', 'j1', 'n1', 'j2', 'sa', 'j3', 'gs', 'g1', 'e1', 'e2', 'c2', '3a', '3b']
        """
        if not satellites:
            return ""
        if isinstance(satellites, str):
            satellites = [satellites]

        satellite_long_names = [
            "TOPEX",
            "Poseidon",
            "Jason-1",
            "Envisat",
            "Jason-2",
            "SARAL",
            "Jason-3",
            "Geosat",
            "GFO",
            "ERS-1",
            "ERS-2",
            "CryoSat-2",
            "Sentinel-3A",
            "Sentinel-3B",
        ]

        satellite_short_names = [
            "tx",
            "ps",
            "j1",
            "n1",
            "j2",
            "sa",
            "j3",
            "gs",
            "g1",
            "e1",
            "e2",
            "c2",
            "3a",
            "3b",
        ]

        satellite_dict = dict(zip(satellite_long_names, satellite_short_names))

        satellite_strings = []
        for sat in satellites:
            if sat in satellite_short_names:
                satellite_strings.append(sat)
            elif sat in satellite_long_names:
                satellite_strings.append(satellite_dict[sat])
            else:
                raise InvalidSatelliteName("Invalid satellite name: " + sat)
        return satellite_strings
