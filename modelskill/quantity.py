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
    >>> wl = Quantity(name="Water Level", unit="meter")
    >>> wl
    Quantity(name='Water Level', unit='meter')
    >>> wl.name
    'Water Level'
    >>> wl.unit
    'meter'
    >>> wl.is_compatible(wl)
    True
    >>> ws = Quantity(name="Wind Direction", unit="degree", is_directional=True)
    >>> ws
    Quantity(name='Wind Direction', unit='degree', is_directional=True)
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

    def is_compatible(self, other) -> bool:
        """Check if the quantity is compatible with another quantity

        Examples
        --------
        >>> wl = Quantity(name="Water Level", unit="meter")
        >>> ws = Quantity(name="Wind Speed", unit="meter per second")
        >>> wl.is_compatible(ws)
        False
        >>> uq = Quantity(name="Undefined", unit="Undefined")
        >>> wl.is_compatible(uq)
        True
        """

        if self == other:
            return True

        if (self.name == "Undefined") or (other.name == "Undefined"):
            return True

        return False

    @staticmethod
    def undefined() -> "Quantity":
        return Quantity(name="", unit="")

    def to_dict(self) -> Dict[str, str]:
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

        Examples
        --------
        >>> Quantity.from_cf_attrs({'long_name': 'Water Level', 'units': 'meter'})
        Quantity(name='Water Level', unit='meter')
        >>> Quantity.from_cf_attrs({'long_name': 'Wind direction', 'units': 'degree'})
        Quantity(name='Wind direction', unit='degree', is_directional=True)

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

        Examples
        --------
        >>> Quantity.from_mikeio_eum_name("Water Level")
        Quantity(name='Water Level', unit='meter')
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
        unit = etype.units[0].name
        is_directional = unit == "degree"
        warnings.warn(f"{unit=} was automatically set for {type_name=}")
        return Quantity(name=type_name, unit=unit, is_directional=is_directional)
