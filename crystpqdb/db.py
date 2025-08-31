from dataclasses import dataclass, field
from typing import Annotated, Any

import numpy as np
import pandas as pd
from parquetdb import ParquetDB
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_serializer,
    field_validator,
)
from pymatgen.core.structure import Structure

NdArrayType = Annotated[
    np.ndarray,
    BeforeValidator(lambda v: v if isinstance(v, np.ndarray) else np.array(v)),
    PlainSerializer(lambda v: v.tolist(), return_type=list),
]

def validate_structure(structure: Structure | dict | None):
    if structure is None:
        return None
    if isinstance(structure, dict):
        return Structure.from_dict(structure)
    return structure

StructureType = Annotated[
    Structure,
    BeforeValidator(validate_structure),
    PlainSerializer(lambda v: v.as_dict() if v is not None else None, return_type=dict),
]

class SymmetryData(BaseModel):
    crystal_system: str | None = None
    symbol: str | None = None
    number: int | None = None
    point_group: str | None = None
    symprec: float | None = None
    angle_tolerance: float | None = None
    version: str | None = None
    
class LatticeData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    matrix: NdArrayType | None = None
    a: float | None = None
    b: float | None = None
    c: float | None = None
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    volume: float | None = None
    
class HasPropsData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    materials: bool | None = None
    thermo: bool | None = None
    xas: bool | None = None
    grain_boundaries: bool | None = None
    chemenv: bool | None = None
    electronic_structure: bool | None = None
    absorption: bool | None = None
    bandstructure: bool | None = None
    dos: bool | None = None
    magnetism: bool | None = None
    elasticity: bool | None = None
    dielectric: bool | None = None
    piezoelectric: bool | None = None
    surface_properties: bool | None = None
    oxi_states: bool | None = None
    provenance: bool | None = None
    charge_density: bool | None = None
    eos: bool | None = None
    phonon: bool | None = None
    insertion_electrodes: bool | None = None
    substrates: bool | None = None
    
    
class CrystPQData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    band_gap: float | None = None
    band_gap_ind: float | None = None
    band_gap_dir: float | None = None
    dos_ef: float | None = None
    
    energy_total: float | None = None
    energy_corrected: float | None = None
    energy_uncorrected: float | None = None
    energy_formation: float | None = None
    energy_above_hull: float | None = None
    energy_phase_seperation: float | None = None
    
    n: float | None = None
    piezoelectric_modulus: float | None = None
    e_electronic: float | None = None
    e_ionic: float | None = None
    e_total: float | None = None
    
    g_reuss: float | None = None
    g_voigt: float | None = None
    g_vrh: float | None = None
    k_reuss: float | None = None
    k_voigt: float | None = None
    k_vrh: float | None = None
    poisson_ratio: float | None = None
    
    surface_energy_anisotropy: float | None = None
    weighted_work_function: float | None = None
    weighted_surface_energy: float | None = None
    
    total_magnetization: float | None = None
    magnetic_ordering: str | None = None
    
    stress: NdArrayType | None = None
    
    
    is_gap_direct: bool | None = None
    is_stable: bool | None = None
    
    @field_validator("is_gap_direct", mode="before")
    def validate_is_gap_direct(cls, v):
        if isinstance(v, bool):
            return v
        elif isinstance(v, str) and v.lower() in ["true", "false"]:
            return bool(v)
        elif isinstance(v, int) and v in [0, 1]:
            return bool(v)
        else:
            return None
    
    @field_validator("is_stable", mode="before")
    def validate_is_stable(cls, v):
        if isinstance(v, bool):
            return v
        elif isinstance(v, str):
            return bool(v)
        elif isinstance(v, int):
            return bool(v)
        elif isinstance(v, float) and isinstance(v, np.nan):
            return None
        return v


class CrystPQRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_database: str
    source_dataset: str
    source_id: str
    species: list[str] | None = None
    frac_coords: NdArrayType | None = None
    cart_coords: NdArrayType | None = None
    lattice: LatticeData | None = None
    structure: StructureType | None = None
    symmetry: SymmetryData | None = None
    data: CrystPQData | None = None
    
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        # Compare arrays using np.array_equal
        is_equal = True
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                if not np.allclose(value, other.__dict__[key]):
                    is_equal = False
            else:
                if value != other.__dict__[key]:
                    is_equal = False
        return is_equal


