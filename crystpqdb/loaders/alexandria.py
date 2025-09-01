
import bz2
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Final, Iterable, List

import pyarrow as pa
import pyarrow.compute as pc
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from crystpqdb.loaders.base import BaseLoader, LoaderConfig
from crystpqdb.utils.pyarrow_utils import get_listArray_struct_fields

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
    
    def _load(self, data_dirpath: Path) -> Iterable[dict]:
        json_files = data_dirpath.glob("*.json")
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
            yield data.get("entries", [])
            
    def _transform(self, table: pa.Table) -> pa.Table:
        n_rows = len(table)
        
        data_fields = table["data"].combine_chunks()
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
        source_id = pc.struct_field(data_fields,"mat_id")
        
        data = pa.Table.from_pydict({
            "band_gap_ind":pc.struct_field(data_fields,"band_gap_ind"),
            "band_gap_dir":pc.struct_field(data_fields,"band_gap_dir"),
            "dos_ef":pc.struct_field(data_fields,"dos_ef"),
            "energy_total":pc.struct_field(data_fields,"energy_total"),
            "energy_uncorrected":pc.struct_field(data_fields,"energy_total"),
            "energy_corrected":pc.struct_field(data_fields,"energy_corrected"),
            "energy_formation":pc.struct_field(data_fields,"e_form"),
            "energy_above_hull":pc.struct_field(data_fields,"e_above_hull"),
            "energy_phase_seperation":pc.struct_field(data_fields,"e_phase_separation"),
            "total_magnetization":pc.struct_field(data_fields,"total_mag"),
        })
        
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
        }
        
        return pa.Table.from_pydict(pqdb_data)
 




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