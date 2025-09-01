
import json
import logging
import os
from pathlib import Path
from typing import Final, Iterable

import pyarrow as pa
import pyarrow.compute as pc
from dotenv import load_dotenv
from mp_api.client import MPRester

from crystpqdb.loaders.base import BaseLoader
from crystpqdb.utils.pyarrow_utils import get_listArray_struct_fields

load_dotenv()

CHUNK_BYTES: Final[int] = 1024 * 1024
LOGGER = logging.getLogger(__name__)

class MPLoader(BaseLoader):

    DEFAULT_BASE_URL: Final[str] = "https://materialsproject.org/api"

    @property
    def source_database(self) -> str:
        return "materials_project"
    
    @property
    def source_dataset(self) -> str:
        return "summary"

    def _download(self, dirpath: Path = None) -> Path:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        # Use self.config.api_key if present; fallback to environment variable
        api_key = self.config.api_key or os.getenv("MP_API_KEY")
        if not api_key:
            LOGGER.error("Materials Project API key not provided. Set DownloadConfig.api_key or MP_API_KEY env var.")
            raise ValueError(
                "Materials Project API key not provided. Set DownloadConfig.api_key or MP_API_KEY env var."
            )

        with MPRester(api_key, monty_decode=False, use_document_model=False) as mpr:
            docs = mpr.materials.summary.search()
            
            docs_list = []
            for doc in docs:
                structure = doc.get("structure", {})
                structure = structure.as_dict() if hasattr(structure, "as_dict") else structure
                composition = doc.get("composition", {})
                composition = composition.as_dict() if hasattr(composition, "as_dict") else composition
                symmetry = doc.get("symmetry", {})
                crystal_system = str(symmetry.get("crystal_system", ""))
                symmetry.update({"crystal_system": crystal_system})
                
                record = {
                    "structure": structure,
                    "composition": composition,
                    "band_gap": doc.get("band_gap", None),
                    "n": doc.get("n", None),
                    "piezoelectric_modulus": doc.get("piezoelectric_modulus", None),
                    "e_electronic": doc.get("e_electronic", None),
                    "e_ionic": doc.get("e_ionic", None),
                    "e_total": doc.get("e_total", None),
                    "g_reuss": doc.get("g_reuss", None),
                    "g_voigt": doc.get("g_voigt", None),
                    "g_vrh": doc.get("g_vrh", None),
                    "k_reuss": doc.get("k_reuss", None),
                    "k_voigt": doc.get("k_voigt", None),
                    "k_vrh": doc.get("k_vrh", None),
                    "poisson_ratio": doc.get("poisson_ratio", None),
                    "surface_energy_anisotropy": doc.get("surface_energy_anisotropy", None),
                    "total_energy": doc.get("total_energy", None),
                    "uncorrected_energy": doc.get("uncorrected_energy", None),
                    "weighted_work_function": doc.get("weighted_work_function", None),
                    "weighted_surface_energy": doc.get("weighted_surface_energy", None),
                    "total_magnetization": doc.get("total_magnetization", None),
                    "is_gap_direct": doc.get("is_gap_direct", None),
                    "magnetic_ordering": doc.get("magnetic_ordering", None),
                    "formation_energy_per_atom": doc.get("formation_energy_per_atom", None),
                    "e_above_hull": doc.get("e_above_hull", None),
                    "is_stable": doc.get("is_stable", None),
                    "spacegroup": doc.get("spacegroup", None),
                    "has_props": doc.get("has_props", None),
                    "material_id": str(doc.get("material_id", "")),
                    "nelements": doc.get("nelements", None),
                    "nsites": doc.get("nsites", None),
                    "symmetry": doc.get("symmetry", {}),
                    
                }
                docs_list.append(record)
                
            with open(dirpath / "mp_data.json", "w") as f:
                json.dump(docs_list, f)

        return dirpath
    
    def _load(self, dirpath: Path) -> Iterable[dict]:
        json_files = dirpath.glob("*.json")
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
            yield data
    
    def _transform(self, table: pa.Table) -> pa.Table:
        n_rows = len(table)
        
        data_fields = table.to_struct_array().combine_chunks()
        structure = table["structure"].combine_chunks()
        sites=pc.struct_field(structure,"sites")
        charge = pc.struct_field(structure,"charge")
        lattice_struct_array = pc.struct_field(structure,"lattice")
        cart_coords = get_listArray_struct_fields(sites,["xyz"])["xyz"]
        frac_coords = get_listArray_struct_fields(sites,["abc"])["abc"]
        labels = get_listArray_struct_fields(sites,["label"])["label"]
        species_list_array = get_listArray_struct_fields(sites,["species"])["species"]
        
        species_list_array = get_listArray_struct_fields(sites,["species"])["species"]
        offsets = species_list_array.offsets
        raw_element = pc.struct_field(species_list_array.flatten(recursive=True),"element")
        elements = pa.ListArray.from_arrays(offsets,raw_element)

        
        source_database = [self.source_database] * n_rows
        source_dataset = [self.source_dataset] * n_rows
        source_id = pc.struct_field(data_fields,"material_id")
        
        has_props = pc.struct_field(data_fields,"has_props")
        
        data = pa.Table.from_pydict({
                "band_gap":pc.struct_field(data_fields,"band_gap"),
                "energy_total":pc.struct_field(data_fields,"total_energy"),
                "energy_uncorrected":pc.struct_field(data_fields,"uncorrected_energy"),
                "energy_corrected":pc.struct_field(data_fields,"total_energy"),
                "energy_formation":pc.struct_field(data_fields,"formation_energy_per_atom"),
                "energy_above_hull":pc.struct_field(data_fields,"e_above_hull"),
                "n":pc.struct_field(data_fields,"n"),
                "piezoelectric_modulus":pc.struct_field(data_fields,"piezoelectric_modulus"),
                "e_electronic":pc.struct_field(data_fields,"e_electronic"),
                "e_ionic":pc.struct_field(data_fields,"e_ionic"),
                "e_total":pc.struct_field(data_fields,"e_total"),
                "g_reuss":pc.struct_field(data_fields,"g_reuss"),
                "g_voigt":pc.struct_field(data_fields,"g_voigt"),
                "g_vrh":pc.struct_field(data_fields,"g_vrh"),
                "k_reuss":pc.struct_field(data_fields,"k_reuss"),
                "k_voigt":pc.struct_field(data_fields,"k_voigt"),
                "k_vrh":pc.struct_field(data_fields,"k_vrh"),
                "poisson_ratio":pc.struct_field(data_fields,"poisson_ratio"),
                "surface_energy_anisotropy":pc.struct_field(data_fields,"surface_energy_anisotropy"),
                
                "weighted_work_function":pc.struct_field(data_fields,"weighted_work_function"),
                "weighted_surface_energy":pc.struct_field(data_fields,"weighted_surface_energy"),
                "total_magnetization":pc.struct_field(data_fields,"total_magnetization"),
                
                "magnetic_ordering":pc.struct_field(data_fields,"magnetic_ordering"),
                
                "is_gap_direct":pc.struct_field(data_fields,"is_gap_direct"),
                "is_stable":pc.struct_field(data_fields,"is_stable"),
        })
        
        
        symmetry = pc.struct_field(data_fields,"symmetry")
        
        pqdb_data = {
            "source_database":source_database,
            "source_dataset":source_dataset,
            "source_id":source_id,
            "species":elements,
            "cart_coords":cart_coords,
            "frac_coords":frac_coords,
            "lattice":lattice_struct_array,
            "structure":structure,
            "data":data.to_struct_array().combine_chunks(),
            "has_props":has_props,
            "symmetry":symmetry,
        }
        
        return pa.Table.from_pydict(pqdb_data)
