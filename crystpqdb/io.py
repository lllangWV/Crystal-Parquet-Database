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
from typing import Dict, Final, List, Optional, Type

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mp_api.client import MPRester

load_dotenv()

CHUNK_BYTES: Final[int] = 1024 * 1024
LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for database downloaders.

    Parameters
    ----
    source_name : str
        Canonical name of the data source (e.g., "alexandria3d").
    base_url : str, optional
        Base URL for the remote dataset or API.
    api_key : str, optional
        API key or token if the source requires authentication.
    timeout_seconds : int, default=60
        Network timeout to use for remote requests.
    num_workers : int, default=8
        Number of worker threads/processes to use for parallel I/O.
    from_scratch : bool, default=False
        If True, remove any existing data at destination before
        downloading.
    dataset_name : str, optional
        Dataset identifier for sources that provide multiple datasets
        (e.g., JARVIS). When provided, implementations may use this to
        determine which dataset to download.

    """

    source_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: int = 60
    num_workers: int = 8
    from_scratch: bool = False
    dataset_name: Optional[str] = None


class DatabaseDownloader(ABC):
    """Abstract interface for downloading a crystal database locally.

    Implementations should handle retrieving the dataset from the
    configured remote source and materializing it under the given
    directory path. Implementations must be idempotent and safe to
    re-run; callers may invoke ``download`` multiple times.

    Notes
    ----
    - Implementations should create the target directory if it does not
      exist.
    - The return value should be the directory containing the
      downloaded dataset for easy chaining.
    - Keep network I/O contained within this layer; transformation of
      the downloaded files should happen elsewhere.
    """

    def __init__(self, config: DownloadConfig) -> None:
        self._config = config

    @property
    def config(self) -> DownloadConfig:
        """Return the immutable downloader configuration."""

        return self._config

    @abstractmethod
    def download(self, dirpath: Path) -> Path:
        """Download or update the dataset under ``dirpath``.

        Parameters
        ----
        dirpath : pathlib.Path
            Directory path where the dataset should be stored
            (implementation may create subdirectories as needed).

        Returns
        ----
        pathlib.Path
            Path to the directory containing the downloaded dataset.
        """

        raise NotImplementedError





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



_REGISTRY: Dict[str, Type[DatabaseDownloader]] = {
    "alexandria1d": Alexandria1DDownloader,
    "alexandria2d": Alexandria2DDownloader,
    "alexandria3d": Alexandria3DDownloader,
    "materials_project": MaterialsProjectDownloader,
    "materials-project": MaterialsProjectDownloader,
    "materialsproject": MaterialsProjectDownloader,
    "jarvis": JarvisDownloader,
}


def get_downloader(source_name: str, config: DownloadConfig | None = None) -> DatabaseDownloader:
    """Create a downloader instance by source name.

    Parameters
    ----
    source_name : str
        Canonical or alias name of the source (e.g., "alexandria3d",
        "materials_project", "jarvis"). Case-insensitive.
    config : DownloadConfig, optional
        Optional explicit configuration. If not provided, a minimal
        configuration will be created from the ``source_name`` only.

    Returns
    ----
    DatabaseDownloader
        A constructed downloader instance for the requested source.

    Raises
    ----
    KeyError
        If the provided ``source_name`` is not registered.
    """

    normalized = source_name.strip().lower().replace(" ", "_")
    try:
        klass = _REGISTRY[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown downloader source: {source_name}") from exc

    if config is None:
        config = DownloadConfig(source_name=normalized)

    return klass(config=config)


