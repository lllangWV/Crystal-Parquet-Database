import bz2
import json
import logging
import os
import re
import shutil
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Final, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mp_api.client import MPRester

from .base import DatabaseDownloader, DownloadConfig

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


class BaseAlexandriaDownloader(DatabaseDownloader):
    
    @property
    @abstractmethod
    def DEFAULT_BASE_URL(self) -> str:
        pass
    
    def download(self, dirpath: Path | str) -> Path:
        dirpath = Path(dirpath)
        base_url = self.config.base_url or self.DEFAULT_BASE_URL
        timeout_seconds = self.config.timeout_seconds
        num_workers = self.config.num_workers

        if self.config.from_scratch and dirpath.exists():
            LOGGER.info("from_scratch=True, removing existing directory: %s", dirpath)
            shutil.rmtree(dirpath, ignore_errors=True)

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

class Alexandria3DDownloader(BaseAlexandriaDownloader):
    """Downloader for the Alexandria3D database.

    1) Scrape index to discover alexandria_*.json.bz2 files
    2) Download to 'compressed' subdirectory (parallel threads)
    3) Decompress to 'uncompressed' subdirectory (parallel threads)
    4) Return 'uncompressed' directory path
    """
    @property
    def DEFAULT_BASE_URL(self) -> str:
        return "https://alexandria.icams.rub.de/data/pbe/"

class Alexandria2DDownloader(BaseAlexandriaDownloader):
    """Downloader for the Alexandria database.

    1) Scrape index to discover alexandria_*.json.bz2 files
    2) Download to 'compressed' subdirectory (parallel threads)
    3) Decompress to 'uncompressed' subdirectory (parallel threads)
    4) Return 'uncompressed' directory path
    """

    @property
    def DEFAULT_BASE_URL(self) -> str:
        return "https://alexandria.icams.rub.de/data/pbe_2d/"


class Alexandria1DDownloader(BaseAlexandriaDownloader):
    """Downloader for the Alexandria database.

    1) Scrape index to discover alexandria_*.json.bz2 files
    2) Download to 'compressed' subdirectory (parallel threads)
    3) Decompress to 'uncompressed' subdirectory (parallel threads)
    4) Return 'uncompressed' directory path
    """

    @property
    def DEFAULT_BASE_URL(self) -> str:
        return "https://alexandria.icams.rub.de/data/pbe_1d/"

    



class MaterialsProjectDownloader(DatabaseDownloader):
    """Downloader for Materials Project data via their API.

    Expects an API key if private endpoints are used.
    """

    DEFAULT_BASE_URL: Final[str] = "https://materialsproject.org/api"

    def download(self, dirpath: Path) -> Path:
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


class JarvisDownloader(DatabaseDownloader):
    """Downloader for JARVIS datasets.

    """
    DEFAULT_BASE_URL: Final[str] = "https://jarvis.nist.gov/"

    def download(self, dirpath: Path) -> Path:
        dirpath.mkdir(parents=True, exist_ok=True)
        dataset_name = self.config.dataset_name

        try:
            from jarvis.db.figshare import data as jarvis_data
            from jarvis.db.figshare import get_db_info
        except Exception as exc:
            LOGGER.error(
                "jarvis-tools is required for JarvisDownloader: pip install jarvis-tools"
            )
            raise

        # Retrieve valid dataset names
        try:
            valid_names = sorted(list(get_db_info().keys()))
        except Exception:
            valid_names = []

        # Determine datasets to download
        download_all = (dataset_name is None) or (str(dataset_name).lower() == "all")
        if download_all:
            if not valid_names:
                raise ValueError("Could not retrieve JARVIS dataset list.")
            names_to_download = valid_names
        else:
            if valid_names and dataset_name not in valid_names:
                hint = ", ".join(valid_names)
                raise ValueError(
                    f"Unknown JARVIS dataset_name: '{dataset_name}'. Valid options are: {hint}"
                )
            names_to_download = [dataset_name]  # type: ignore[list-item]

        # Download each dataset. For 'all', place each in a subdirectory
        for name in names_to_download:
            target_dir = dirpath if not download_all else (dirpath / str(name))
            target_dir.mkdir(parents=True, exist_ok=True)

            if self.config.from_scratch and target_dir.exists():
                LOGGER.info("from_scratch=True, removing existing directory: %s", target_dir)
                shutil.rmtree(target_dir, ignore_errors=True)
                target_dir.mkdir(parents=True, exist_ok=True)

            LOGGER.info("Downloading JARVIS dataset '%s' into %s", name, target_dir)
            jarvis_data(str(name), store_dir=str(target_dir))

            # Unzip any downloaded zip files into target_dir and remove archives
            import zipfile

            for zip_path in target_dir.glob("*.zip"):
                LOGGER.info("Unzipping %s", zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(target_dir)
                try:
                    zip_path.unlink()
                except OSError:
                    pass

        return dirpath


