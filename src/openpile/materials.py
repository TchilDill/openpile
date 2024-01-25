from enum import Enum

from pydantic.dataclasses import dataclass


class ConstitutiveModel:
    pass


class StructuralMaterial(ConstitutiveModel):
    pass


class StructuralMaterialEnum(str, Enum):
    Steel = "Steel"
    Concrete = "Concrete"
    

@dataclass
class Steel(StructuralMaterial):
    """
    Parameters
    ----------
    uw: float = 78.0
        Unit weight (kN/m3).
    young_modulus: float = 210.0e6
        Young modulus (kPa).
    nu: float = 0.3
        Poisson's ratio.
    """
    uw: float = 78.0
    young_modulus: float = 210.0e6
    nu: float = 0.3


@dataclass
class Concrete(StructuralMaterial):
    """
    Parameters
    ----------
    uw: float = 24.0
        Unit weight (kN/m3).
    young_modulus: float = 30.0e6
        Young modulus (kPa).
    nu: float = 0.2
        Poisson's ratio.
    """
    uw: float = 24.0
    young_modulus: float = 30.0e6
    nu: float = 0.2