
import bz2
import json
import logging
import os
import re
import shutil
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Type
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.core.structure import Structure

from crystpqdb.db import (
    CrystPQData,
    CrystPQRecord,
    HasPropsData,
    LatticeData,
    SymmetryData,
)
from crystpqdb.loaders.base import BaseLoader

load_dotenv()

CHUNK_BYTES: Final[int] = 1024 * 1024
LOGGER = logging.getLogger(__name__)

class MC3DLoader(BaseLoader):
    """Loader for MC3D data.

    The MC3D database is a collection of materials data from the Materials Cloud.
    It contains the following files:
    - files_description.md
    - MC3D-provenance.aiida
    - MC3D-structure.aiida
    """
    mc3d_cif_url: Final[str] = "https://archive.materialscloud.org/records/eqzc6-e2579/files/MC3D-cifs.zip?download=1"
    mc3d_provenance_url: Final[str] = "https://archive.materialscloud.org/records/eqzc6-e2579/files/MC3D-provenance.aiida?download=1"
    mc3d_structure_url: Final[str] = "https://archive.materialscloud.org/records/eqzc6-e2579/files/MC3D-structures.aiida?download=1"
    mc3d_file_description_url: Final[str] = "https://archive.materialscloud.org/records/eqzc6-e2579/files/files_description.md?download=1"
    

    @property
    def source_database(self) -> str:
        return "materialscloud"
    
    @property
    def source_dataset(self) -> str:
        return "mc3d"
    
    def download_url(self, dirpath: Path | str, url: str) -> str:
        dirpath = Path(dirpath)

        # Send GET request
        response = requests.get(url)

        # Check if request was successful
        if response.status_code == 200:
            cd = response.headers.get("content-disposition")
            filename = None
            if cd:
                match = re.findall('filename="?([^"]+)"?', cd)
                if match:
                    filename = match[0]
                    
            # If no filename in headers, fall back to URL
            if not filename:
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                filename = unquote(filename)  # decode %20 etc.
            output_file = dirpath / filename
                    
            with open(dirpath / output_file, "wb") as f:
                f.write(response.content)
            print(f"File downloaded and saved as {output_file}")
            
            return output_file
        else:
            print(f"Failed to download file. Status code: {response.status_code}")


    def _download(self, dirpath: Path | str | None = None) -> Path:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        mc3d_cif_zip = self.download_url(dirpath, self.mc3d_cif_url)
        mc3d_provenance = self.download_url(dirpath, self.mc3d_provenance_url)
        mc3d_structure = self.download_url(dirpath, self.mc3d_structure_url)
        mc3d_file_description = self.download_url(dirpath, self.mc3d_file_description_url)

        if mc3d_cif_zip.exists():
            # Unzip the directory
            with zipfile.ZipFile(mc3d_cif_zip, 'r') as zip_ref:
                zip_ref.extractall(dirpath)
            # Delete the zipped directory (zip file)
            mc3d_cif_zip.unlink()

        return dirpath
    
    def _load_cif(self, filepath: Path) -> pd.DataFrame:
        structure = Structure.from_file(filepath)          
        species = [specie.name for specie in structure.species]
        frac_coords = structure.frac_coords
        cart_coords = structure.cart_coords
        lattice_data = LatticeData(
            matrix=structure.lattice.matrix,
            a=structure.lattice.a,
            b=structure.lattice.b,
            c=structure.lattice.c,
            alpha=structure.lattice.alpha,
            beta=structure.lattice.beta,
            gamma=structure.lattice.gamma,
            volume=structure.lattice.volume)
        record = CrystPQRecord(
            source_database=self.source_database,
            source_dataset=self.source_dataset,
            source_id=str(filepath.stem),
            species=species,
            frac_coords=frac_coords,
            cart_coords=cart_coords,
            lattice=lattice_data,
            structure=structure,
            data=None
        )
        return record.model_dump(mode="json")
    
    def load(self, data_dirpath: Path) -> pd.DataFrame:
        structures_data_dirpath = data_dirpath / "MC3D-cifs" / "mc3d"
        cif_files = structures_data_dirpath.glob("*.cif")
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            record_dicts = list(executor.map(self._load_cif, cif_files))

        field_names = CrystPQRecord.model_fields.keys()
        columnar: Dict[str, List[Any]] = {k: [] for k in field_names}
        for record_dict in record_dicts:
            for k in field_names:
                columnar[k].append(record_dict[k])
                
        df = pd.DataFrame(columnar)
                
        return df
