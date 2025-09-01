import numpy as np
import pytest
from pymatgen.core import Structure


@pytest.fixture
def symmetry():
    return {
        "crystal_system" : "cubic",
        "symbol" : "Pm-3m",
        "number" : 221,
        "point_group" : "m3m",
        "symprec" : 0.01,
        "angle_tolerance" : 5.0,
        "version" : "2.0.0",
    }

@pytest.fixture
def has_props():
    return {
    "materials" : True,
    "thermo" : True,
    "xas" : False,
    "grain_boundaries" : False,
    "chemenv" : True,
}

@pytest.fixture
def data(symmetry, has_props):
    return {
    "band_gap" : 1.1,
    "n" : 1.02,
    "piezoelectric_modulus" : 1.1,
    "e_electronic" : 1.1,
    "e_ionic" : 1.1,
    "e_total" : 1.1,
    "g_reuss" : 1.1,
    "g_voigt" : 1.1,
    "g_vrh" : 1.1,
    "k_reuss" : 1.1,
    "k_voigt" : 1.1,
    "k_vrh" : 1.1,
    "poisson_ratio" : 1.1,
    "surface_energy_anisotropy" : 1.1,
    "total_energy" : 1.1,
    "uncorrected_energy" : 1.1,
    "weighted_work_function" : 1.1,
    "weighted_surface_energy" : 1.1,
    "total_magnetization" : 1.1,
    "is_gap_direct" : True,
    "magnetic_ordering" : "FM",
    "formation_energy_per_atom" : 1.1,
    "e_above_hull" : 1.1,
    "is_stable" : True,
    "spacegroup" : "Fm-3m",
    "material_id" : "1234567890",
    "nelements" : 1,
    "nsites" : 40,
}

@pytest.fixture
def structure():
    return Structure(
        lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species = ["A", "B"],
    )
    
@pytest.fixture
def record(data, structure):
    return {
    "source_database" : "mp",
    "source_dataset" : "1d",
    "source_id" : "1234567890",
    "energy" : 1.1,
    "species" : ["A", "B", "C"],
    "frac_coords" : np.array([0.0, 0.0, 0.0]),
    "cart_coords" : np.array([0.0, 0.0, 0.0]),
    "lattice" : np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "structure" : structure,
    "data" : data,
        "symmetry" : symmetry,
    "has_props" : has_props,
}
    
    
@pytest.fixture
def record_other(data, structure):
    return {
    "source_database" : "mp",
    "source_dataset" : "1d",
    "source_id" : "1234567890",
    "energy" : 1.1,
    "species" : ["A", "B", "C"],
    "frac_coords" : np.array([0.0, 0.0, 0.0]),
    "cart_coords" : np.array([0.0, 0.0, 0.0]),
    "lattice" : np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "structure" : structure,
    "data" : data,
}

# class TestCrystPQRecord:
    
#     def test_init(self, record):
#         record = CrystPQRecord(**record)
#         assert record.source_database == "mp"
#         assert record.source_dataset == "1d"
#         assert record.source_id == "1234567890"
#         assert record.energy == 1.1
#         assert record.species == ["A", "B", "C"]
#         assert np.allclose(record.frac_coords, np.array([0.0, 0.0, 0.0]))
#         assert np.allclose(record.cart_coords, np.array([0.0, 0.0, 0.0]))
#         assert np.allclose(record.lattice, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
#         assert record.data.symmetry.number == 221
#         assert isinstance(record.structure, Structure)
        
#     def test_serialization(self, record):
#         record = CrystPQRecord(**record)
#         record_json = record.model_dump(mode="json")

#         assert isinstance(record_json, dict)
#         assert isinstance(record_json["frac_coords"], list)
#         assert isinstance(record_json["cart_coords"], list)
#         assert isinstance(record_json["lattice"], list)
#         assert isinstance(record_json["structure"], dict)
        
#     def test_deserialization(self, record):
#         record = CrystPQRecord(**record)
#         record_json = record.model_dump(mode="json")
#         print(record_json)
        
#         test_record = CrystPQRecord(**record_json)
#         print(test_record)
#         assert isinstance(test_record.structure, Structure)
#         assert test_record == record
        
#     def test_equality(self, record, record_other):
#         record = CrystPQRecord(**record)
#         record_other = CrystPQRecord(**record_other)
#         assert record == record
#         assert record != record_other
        