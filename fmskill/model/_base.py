from typing import Union

import pandas as pd
from fmskill import types


class ModelResultBase:
    def __init__(
        self,
        data: Union[types.ExtractableType, pd.DataFrame],
        item: str = None,
        name: str = None,
        quantity: str = None,
    ) -> None:

        self.data = data
        self.item = item
        self.name = name
        self.quantity = quantity

    def __repr__(self):
        txt = [f"<{self.__class__.__name__}> '{self.name}'"]
        txt.append(f"- Item: {self.item}")
        return "\n".join(txt)
