
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
    CrystPQRecord,
    HasPropsData,
    LatticeData,
    SymmetryData,
)
from crystpqdb.loaders.base import BaseLoader, LoaderConfig

load_dotenv()

CHUNK_BYTES: Final[int] = 1024 * 1024
LOGGER = logging.getLogger(__name__)

def _http_get(url: str, timeout_seconds: int) -> requests.Response:
    response = requests.get(url, stream=True, timeout=timeout_seconds)
    response.raise_for_status()
    return response


def _stream_download(url: str, output_path: Path, timeout_seconds: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    response = _http_get(url, timeout_seconds)
    with output_path.open("wb") as output_file:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if chunk:
                output_file.write(chunk)


def _list_alexandria_files(index_url: str, timeout_seconds: int) -> List[str]:
    response = _http_get(index_url, timeout_seconds)
    soup = BeautifulSoup(response.text, "html.parser")
    pattern = re.compile(r"^alexandria_.*\.json\.bz2$")
    return [
        link.get("href", "")
        for link in soup.find_all("a", href=True)
        if pattern.match(link.get("href", ""))
    ]


def _decompress_bz2_file(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return
    with bz2.BZ2File(source_path, "rb") as file_in, dest_path.open("wb") as file_out:
        for chunk in iter(lambda: file_in.read(CHUNK_BYTES), b""):
            if chunk:
                file_out.write(chunk)


def _any_files(directory: Path) -> bool:
    return any(directory.iterdir()) if directory.exists() else False


class BaseAlexandriaLoader(BaseLoader):

    @property
    def source_database(self) -> str:
        return "alex"
    
    @property
    @abstractmethod
    def DEFAULT_BASE_URL(self) -> str:
        pass
    
    def _download(self, dirpath: Path | str | None = None) -> Path:
        dirpath = Path(dirpath)
        base_url = self.config.base_url or self.DEFAULT_BASE_URL
        timeout_seconds = self.config.timeout_seconds
        num_workers = self.config.num_workers

        # Stage compressed files under a temporary folder inside dirpath
        compressed_dir = dirpath / ".compressed_tmp"
        compressed_dir.mkdir(parents=True, exist_ok=True)
        dirpath.mkdir(parents=True, exist_ok=True)

        # If we already have uncompressed outputs in the root, skip
        if any(dirpath.glob("alexandria_*.json")):
            # Clean up any stale compressed staging if present
            if compressed_dir.exists():
                shutil.rmtree(compressed_dir, ignore_errors=True)
            return dirpath

        file_names = _list_alexandria_files(base_url, timeout_seconds)
        if not file_names:
            # Nothing to do; clean up staging
            if compressed_dir.exists():
                shutil.rmtree(compressed_dir, ignore_errors=True)
            return dirpath

        # Download compressed files in parallel
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = []
            for name in file_names:
                url = base_url.rstrip("/") + "/" + name
                out_path = compressed_dir / name
                LOGGER.info("Downloading %s", url)
                futures.append(
                    executor.submit(_stream_download, url, out_path, timeout_seconds)
                )
            for _ in as_completed(futures):
                pass

        # Decompress into the root dirpath
        to_decompress = list(compressed_dir.glob("*.bz2"))
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = []
            for src in to_decompress:
                dest = dirpath / src.name[:-4]
                LOGGER.info("Decompressing %s -> %s", src, dest)
                futures.append(executor.submit(_decompress_bz2_file, src, dest))
            for _ in as_completed(futures):
                pass

        # Remove compressed staging directory; leave only uncompressed files in root
        if compressed_dir.exists():
            shutil.rmtree(compressed_dir, ignore_errors=True)

        return dirpath
    
    
    def _load_json(self, filepath: Path) -> pd.DataFrame:
        with open(filepath, "r") as f:
            data = json.load(f)
              
        field_names = CrystPQRecord.model_fields.keys()
        columnar: Dict[str, List[Any]] = {k: [] for k in field_names}
        
        entries = data.get("entries", [])
        for doc in entries:            
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
        
            data_dict = doc.get("data", {})
            
            band_gap_ind = data_dict.get("band_gap_ind", None)
            band_gap_dir = data_dict.get("band_gap_dir", None)
            if band_gap_ind is not None and band_gap_dir is not None:
                if band_gap_ind == 0 and band_gap_dir > 0:
                    band_gap = band_gap_dir
                elif band_gap_ind > 0 and band_gap_dir == 0:
                    band_gap = band_gap_ind
                elif band_gap_ind > 0 and band_gap_dir > 0:
                    band_gap = min(band_gap_ind, band_gap_dir)
                else:
                    band_gap = None
                
            crystpq_data = CrystPQData(
                band_gap=band_gap,
                band_gap_ind = band_gap_ind,
                band_gap_dir = band_gap_dir,
                
                dos_ef = data_dict.get("dos_ef", None),
  
                energy_total=data_dict.get("energy_corrected", None),
                energy_corrected = data_dict.get("energy_corrected", None),
                energy_uncorrected = data_dict.get("energy_total", None),
                energy_formation = data_dict.get("e_form", None),
                energy_above_hull = data_dict.get("e_above_hull", None),
                energy_phase_seperation = data_dict.get("e_phase_seperation", None),
                
                total_magnetization=data_dict.get("total_mag"),

                stress=doc.get("stress"),
            )
            
            

            record = CrystPQRecord(
                source_database=self.source_database,
                source_dataset=self.source_dataset,
                source_id=str(doc.get("mat_id", None)),
                species=species,
                frac_coords=frac_coords,
                cart_coords=cart_coords,
                lattice=lattice_data,
                structure=structure,
                data=crystpq_data
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




class Alexandria3DLoader(BaseAlexandriaLoader):
    """Downloader for the Alexandria3D database.

    1) Load alexandria_*.json.bz2 files
    2) Return a pandas DataFrame
    """
    @property
    def DEFAULT_BASE_URL(self) -> str:
        return "https://alexandria.icams.rub.de/data/pbe/"
    
    @property
    def source_dataset(self) -> str:
        return "3d"

class Alexandria2DLoader(BaseAlexandriaLoader):
    """Loader for the Alexandria database.

    1) Load alexandria_*.json.bz2 files
    2) Return a pandas DataFrame
    """

    @property
    def DEFAULT_BASE_URL(self) -> str:
        return "https://alexandria.icams.rub.de/data/pbe_2d/"
    
    @property
    def source_dataset(self) -> str:
        return "2d"


class Alexandria1DLoader(BaseAlexandriaLoader):
    """Loader for the Alexandria database.

    1) Load alexandria_*.json.bz2 files
    2) Return a pandas DataFrame
    """
    
    @property
    def DEFAULT_BASE_URL(self) -> str:
        return "https://alexandria.icams.rub.de/data/pbe_1d/"
    
    @property
    def source_dataset(self) -> str:
        return "1d"
    

        
def get_alexandria_loader(source_dataset: str, config: LoaderConfig) -> BaseAlexandriaLoader:
    if source_dataset == "3d":
        return Alexandria3DLoader(config)
    elif source_dataset == "2d":
        return Alexandria2DLoader(config)
    elif source_dataset == "1d":
        return Alexandria1DLoader(config)
    else:
        raise ValueError(f"Invalid source dataset: {source_dataset}")