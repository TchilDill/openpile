import math as m
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from typing_extensions import Literal
from pydantic import BaseModel, Field, root_validator
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

import openpile.utils.graphics as graphics
import openpile.utils.validation as validation

class PydanticConfig:
    arbitrary_types_allowed = True

@dataclass(config=PydanticConfig)
class SoilProfile:
    pass