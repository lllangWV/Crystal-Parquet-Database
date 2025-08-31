from pandera.typing.pandas import Series
import pandera.pandas as pa

from typing import TypedDict

class LatticeDict(TypedDict):
    matrix: list[list[float]]
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    pbc: list[bool]
    volume: float

class SpeciesDict(TypedDict):
    element: str
    occu: float
    
class PropertiesDict(TypedDict):
    magmom: float
    charge: float
    forces: list[float]
    
class SiteDict(TypedDict):
    species: list[str]
    abc: list[float]
    xyz: list[float]
    properties: PropertiesDict
    label: str
    
class StructureDict(TypedDict):
    charge: float
    lattice: LatticeDict
    sites: list[SiteDict]
    
class HasPropsData(TypedDict):
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
    
    
class DataDict(TypedDict):
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
    
    stress: list[list[float]] | None = None
    
    is_stable: bool | None = None
    
class SymmetryData(TypedDict):
    crystal_system: str | None = None
    symbol: str | None = None
    number: int | None = None
    point_group: str | None = None
    symprec: float | None = None
    angle_tolerance: float | None = None
    version: str | None = None
    
class CrystPQData(pa.DataFrameModel):
    class Config:
        add_missing_columns=True
        
    source_database: Series[str] = pa.Field(nullable=True)
    source_dataset: Series[str] = pa.Field(nullable=True)
    source_id: Series[str] = pa.Field(nullable=True)
    species: Series[list[str]] = pa.Field(nullable=True)
    cart_coords: Series[list[list[float]]] = pa.Field(nullable=True)
    frac_coords: Series[list[list[float]]] = pa.Field(nullable=True)
    lattice: Series[LatticeDict] = pa.Field(nullable=True)
    structure: Series[StructureDict] = pa.Field(nullable=True)
    data: Series[DataDict] = pa.Field(nullable=True)
    symmetry: Series[SymmetryData] = pa.Field(nullable=True)
    has_props: Series[HasPropsData] = pa.Field(nullable=True)
    

