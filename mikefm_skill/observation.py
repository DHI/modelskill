import numpy as np
from mikeio import Dfs0

class PointObservation:
    
    filename = None
    dfs = None
    x = None
    y = None
    z = None
    item = None
    
    def __init__(self, filename, x, y, z=None):
        self.filename = filename
        self.dfs = Dfs0(filename)
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        out = []
        out.append("PointObservation")
        out.append(self.filename)
        return str.join("\n", out)

