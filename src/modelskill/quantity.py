from typing import Dict, Mapping
from dataclasses import dataclass
import warnings
import mikeio


# TODO change name of fields to match CF conventions?
# https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch03s03.html
# long_name, standard_name & units
@dataclass(frozen=True)
class Quantity:
    """Quantity of data

    Parameters
    ----------
    name : str
        Name of the quantity
    unit : str
        Unit of the quantity
    is_directional : bool, optional
        Whether the quantity is directional (e.g. Wind Direction), by default False

    Examples
    --------
    ```{python}
    from modelskill import Quantity
    wl = Quantity(name="Water Level", unit="meter")
    wl
    ```
    ```{python}
    wl.name
    'Water Level'
    ```
    ```{python}
    wl.unit
    ```
    ```{python}
    wl.is_compatible(wl)
    ```
    ```{python}
    Quantity(name="Wind Direction", unit="degree", is_directional=True)
    ```
    """

    name: str
    unit: str
    is_directional: bool = False

    def __str__(self):
        return f"{self.name} [{self.unit}]"

    def __repr__(self):
        if self.is_directional:
            return (
                f"Quantity(name='{self.name}', unit='{self.unit}', is_directional=True)"
            )
        else:
            # hide is_directional if False to avoid clutter
            return f"Quantity(name='{self.name}', unit='{self.unit}')"

    def is_compatible(self, other: "Quantity") -> bool:
        """Check if the quantity is compatible with another quantity

        Two quantities are compatible when their units agree. Names are not
        compared, since the same physical quantity is often named differently
        in observation and model, e.g. "Water Level" and "Surface Elevation".
        Units are compared as written, so "meter" and "m" differ. An undefined
        quantity, or one without a unit, is compatible with any other.

        Examples
        --------
        ```{python}
        wl = Quantity(name="Water Level", unit="m")
        ws = Quantity(name="Wind Speed", unit="m/s")
        wl.is_compatible(ws)
        ```
        ```{python}
        wl.is_compatible(Quantity(name="Surface Elevation", unit="m"))
        ```
        ```{python}
        wl.is_compatible(Quantity.undefined())
        ```
        """
        if _is_undefined_unit(self.unit) or _is_undefined_unit(other.unit):
            return True

        return self.unit == other.unit

    @staticmethod
    def undefined() -> "Quantity":
        """Create an undefined Quantity.

        Returns
        -------
        Quantity
            A Quantity with empty name and unit
        """
        return Quantity(name="", unit="")

    def to_dict(self) -> Dict[str, str]:
        """Convert Quantity to a dictionary.

        Returns
        -------
        Dict[str, str]
            Dictionary with 'name' and 'unit' keys
        """
        return {"name": self.name, "unit": self.unit}

    @staticmethod
    def from_cf_attrs(attrs: Mapping[str, str]) -> "Quantity":
        """Create a Quantity from a CF compliant attributes dictionary

        If units is "degree", "degrees" or "Degree true", the quantity is assumed
        to be directional. Based on https://codes.ecmwf.int/grib/param-db/ and
        https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html

        Parameters
        ----------
        attrs : Mapping[str, str]
            Attributes dictionary

        Returns
        -------
        Quantity
            Quantity object created from CF attributes

        Examples
        --------
        ```{python}
        Quantity.from_cf_attrs({'long_name': 'Water Level', 'units': 'meter'})
        ```
        ```{python}
        Quantity.from_cf_attrs({'long_name': 'Wind direction', 'units': 'degree'})
        ```

        """
        quantity = Quantity.undefined()
        if long_name := attrs.get("long_name"):
            if units := attrs.get("units"):
                is_directional = units in ["degree", "degrees", "Degree true"]
                quantity = Quantity(
                    name=long_name,
                    unit=units,
                    is_directional=is_directional,
                )
        return quantity

    @staticmethod
    def from_mikeio_iteminfo(iteminfo: mikeio.ItemInfo) -> "Quantity":
        """Create a Quantity from mikeio ItemInfo

        If the unit is "degree", the quantity is assumed to be directional.
        """

        unit = iteminfo.unit.short_name
        is_directional = unit == "degree"
        return Quantity(
            name=repr(iteminfo.type), unit=unit, is_directional=is_directional
        )

    @staticmethod
    def from_mikeio_eum_name(type_name: str) -> "Quantity":
        """Create a Quantity from a name recognized by mikeio

        Parameters
        ----------
        type_name : str
            Name of the quantity

        Returns
        -------
        Quantity
            Quantity object created from mikeio EUM type

        Examples
        --------
        ```{python}
        Quantity.from_mikeio_eum_name("Water Level")
        ```
        """
        try:
            etype = mikeio.EUMType[type_name]
        except KeyError:
            name_underscore = type_name.replace(" ", "_")
            try:
                etype = mikeio.EUMType[name_underscore]
            except KeyError:
                raise ValueError(
                    f"{type_name=} is not recognized as a known type. Please create a Quantity(name='{type_name}' unit='<FILL IN UNIT>')"
                )
        unit = etype.units[0].short_name
        is_directional = unit == "degree"
        warnings.warn(f"{unit=} was automatically set for {type_name=}")
        return Quantity(name=type_name, unit=unit, is_directional=is_directional)


def _is_undefined_unit(unit: str) -> bool:
    # "" from Quantity.undefined() and from res1d/EPANET results, which carry no
    # unit; "undefined" from mikeio items of EUM type Undefined; "Undefined" from
    # earlier modelskill versions
    return unit in ("", "undefined", "Undefined")
