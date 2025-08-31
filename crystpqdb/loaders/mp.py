
import bz2
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, Iterable, List, Optional, Type

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core.structure import Structure

from crystpqdb.db import (
    CrystPQData,
    DataDict,
    HasPropsData,
    LatticeDict,
    StructureDict,
    SymmetryData,
)
from crystpqdb.loaders.base import BaseLoader

load_dotenv()

CHUNK_BYTES: Final[int] = 1024 * 1024
LOGGER = logging.getLogger(__name__)

class MPLoader(BaseLoader):
    """Loader for Materials Project data via their API.

    Expects an API key if private endpoints are used.
    """

    DEFAULT_BASE_URL: Final[str] = "https://materialsproject.org/api"

    @property
    def source_database(self) -> str:
        return "materials_project"
    
    @property
    def source_dataset(self) -> str:
        return "summary"

    def _download(self, dirpath: Path | str | None = None) -> Path:
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
    
    def _load_json(self, filepath: Path) -> pd.DataFrame:
        with open(filepath, "r") as f:
            data = json.load(f)
              
        field_names = CrystPQRecord.model_fields.keys()
        columnar: Dict[str, List[Any]] = {k: [] for k in field_names}
        for idoc, doc in enumerate(data):
            if idoc % 10000 == 0:
                LOGGER.debug("Loading document %d of %d", idoc, len(data))
            structure_dict = doc.get("structure", None)
            lattice_data = None
            structure=None
            species = None
            frac_coords = None
            cart_coords = None
            
            if structure_dict is not None:
                structure = Structure.from_dict(structure_dict)
                frac_coords = structure.frac_coords
                cart_coords = structure.cart_coords
                species = [specie.name for specie in structure.species]
                lattice_data = LatticeData(
                    matrix=structure.lattice.matrix,
                    a=structure.lattice.a,
                    b=structure.lattice.b,
                    c=structure.lattice.c,
                    alpha=structure.lattice.alpha,
                    beta=structure.lattice.beta,
                    gamma=structure.lattice.gamma,
                    volume=structure.lattice.volume)

            sym = doc.get("symmetry") or {}
            symmetry = SymmetryData(
                crystal_system=str(sym.get("crystal_system")) if sym.get("crystal_system") is not None else None,
                symbol=sym.get("symbol"),
                number=sym.get("number"),
                point_group=sym.get("point_group"),
                symprec=sym.get("symprec"),
                angle_tolerance=sym.get("angle_tolerance"),
                version=sym.get("version"),
            )

            hp = doc.get("has_props") or {}
            has_props = HasPropsData(**hp) if isinstance(hp, dict) else None

            
            data = CrystPQData(
                band_gap=doc.get("band_gap"),
                
                energy_total=doc.get("total_energy"),
                energy_uncorrected=doc.get("uncorrected_energy"),
                energy_corrected=doc.get("total_energy"),
                energy_formation=doc.get("formation_energy_per_atom"),
                energy_above_hull=doc.get("e_above_hull"),
                
                
                n=doc.get("n"),
                piezoelectric_modulus=doc.get("piezoelectric_modulus"),
                e_electronic=doc.get("e_electronic"),
                e_ionic=doc.get("e_ionic"),
                e_total=doc.get("e_total"),
                g_reuss=doc.get("g_reuss"),
                g_voigt=doc.get("g_voigt"),
                g_vrh=doc.get("g_vrh"),
                k_reuss=doc.get("k_reuss"),
                k_voigt=doc.get("k_voigt"),
                k_vrh=doc.get("k_vrh"),
                poisson_ratio=doc.get("poisson_ratio"),
                surface_energy_anisotropy=doc.get("surface_energy_anisotropy"),
                
                weighted_work_function=doc.get("weighted_work_function"),
                weighted_surface_energy=doc.get("weighted_surface_energy"),
                total_magnetization=doc.get("total_magnetization"),
                
                magnetic_ordering=doc.get("magnetic_ordering"),
                
                is_gap_direct=doc.get("is_gap_direct", None),
                is_stable=doc.get("is_stable", None),
                
                has_props=has_props,
            )

            record = CrystPQRecord(
                source_database=self.source_database,
                source_dataset=self.source_dataset,
                source_id=str(doc.get("material_id", "")),
                species=species,
                frac_coords=frac_coords,
                cart_coords=cart_coords,
                lattice=lattice_data,
                structure=structure,
                symmetry=symmetry,
                data=data
            )
            
            record_dict = record.model_dump(mode="json")
            for k in field_names:
                columnar[k].append(record_dict[k])
                
        df = pd.DataFrame(columnar)
                
        return df
    
    def load(self, data_dirpath: Path) -> Iterable[pd.DataFrame]:
        json_files = data_dirpath.glob("*.json")
        for json_file in json_files:
            df = self._load_json(json_file)
            yield df
