
import logging
import os
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Iterable
from urllib.parse import unquote, urlparse

import pyarrow as pa
import requests
from dotenv import load_dotenv
from pymatgen.core import Structure
from pymatgen.core.structure import Structure

from crystpqdb.loaders.base import BaseLoader

load_dotenv()

CHUNK_BYTES: Final[int] = 1024 * 1024
LOGGER = logging.getLogger(__name__)

class MC3DLoader(BaseLoader):

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
    
    def _load(self, dirpath: Path) -> Iterable[dict]:
        
        dirpath = dirpath / "MC3D-cifs" / "mc3d"
        cif_files = dirpath.glob("*.cif")
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            data = list(executor.map(self._load_cif, cif_files))
        yield data
    
    def _load_cif(self, filepath: Path) -> dict:
        structure = Structure.from_file(filepath)          
        species = [specie.name for specie in structure.species]
        frac_coords = structure.frac_coords
        cart_coords = structure.cart_coords
        lattice_data = {
            "matrix":structure.lattice.matrix.tolist(),
            "a":structure.lattice.a,
            "b":structure.lattice.b,
            "c":structure.lattice.c,
            "alpha":structure.lattice.alpha,
            "beta":structure.lattice.beta,
            "gamma":structure.lattice.gamma,
            "pbc":structure.lattice.pbc,
            "volume":structure.lattice.volume}
        record = {
            "source_database":self.source_database,
            "source_dataset":self.source_dataset,
            "source_id":str(filepath.stem),
            "species":species,
            "frac_coords":frac_coords.tolist(),
            "cart_coords":cart_coords.tolist(),
            "lattice":lattice_data,
            "structure":structure.as_dict(),
        }
        return record
    
    def _transform(self, table: pa.Table) -> pa.Table:
        return table.drop_columns("id")