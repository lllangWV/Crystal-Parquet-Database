
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

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from jarvis.db.figshare import data as jarvis_data
from jarvis.db.figshare import get_db_info
from mp_api.client import MPRester

from crystpqdb.db import CrystPQData, DataDict, HasPropsData, SymmetryData
from crystpqdb.loaders.base import BaseLoader
from crystpqdb.db import LatticeDict, StructureDict

load_dotenv()

CHUNK_BYTES: Final[int] = 1024 * 1024
LOGGER = logging.getLogger(__name__)


class BaseJarvisDownloader(BaseLoader):
    """Downloader for JARVIS datasets.

    """
    DEFAULT_BASE_URL: Final[str] = "https://jarvis.nist.gov/"
    
    @property
    def source_database(self) -> str:
        return "jarvis"

    def _download(self, dirpath: Path | str | None = None) -> Path:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        jarvis_data(str(self.source_dataset), store_dir=str(dirpath))

        # Unzip any downloaded zip files into target_dir and remove archives
        import zipfile

        for zip_path in dirpath.glob("*.zip"):
            LOGGER.info("Unzipping %s", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dirpath)
            try:
                zip_path.unlink()
            except OSError:
                pass

        # # Retrieve valid dataset names
        # try:
        #     valid_names = sorted(list(get_db_info().keys()))
        # except Exception:
        #     valid_names = []

        # # Determine datasets to download
        # download_all = (dataset_name is None) or (str(dataset_name).lower() == "all")
        # if download_all:
        #     if not valid_names:
        #         raise ValueError("Could not retrieve JARVIS dataset list.")
        #     names_to_download = valid_names
        # else:
        #     if valid_names and dataset_name not in valid_names:
        #         hint = ", ".join(valid_names)
        #         raise ValueError(
        #             f"Unknown JARVIS dataset_name: '{dataset_name}'. Valid options are: {hint}"
        #         )
        #     names_to_download = [dataset_name]  # type: ignore[list-item]

        return dirpath
    
    
class JarvisAflow2Loader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "aflow2"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAflow2Loader is not implemented yet")
    
class JarvisAGRA_CHOLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "agra_cho"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAGRA_CHOLoader is not implemented yet")
    
    
class JarvisAGRA_COLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "agra_co"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAGRA_COLoader is not implemented yet")
    
class JarvisAGRA_COOHLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "agra_cooh"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAGRA_COOHLoader is not implemented yet")
    
class JarvisAGRA_OLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "agra_o"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAGRA_OLoader is not implemented yet")
    
class JarvisAGRA_OHLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "agra_oh"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAGRA_OHLoader is not implemented yet")
    

class JarvisAlexPBE1DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "alex_pbe_1d"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAlexPBE1DLoader is not implemented yet")
    
class JarvisAlexPBE2DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "alex_pbe_2d"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAlexPBE2DLoader is not implemented yet")
    
class JarvisAlexPBE3DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "alex_pbe_3d"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAlexPBE3DLoader is not implemented yet")
    
class JarvisAlexPBEHullLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "alex_pbe_hull"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAlexPBEHullLoader is not implemented yet")
    
    
class JarvisAlexPBESOL3DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "alex_pbesol_3d_all"
    
    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAlexPBESOL3DLoader is not implemented yet")
    
class JarvisAlexSCAN3DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "alex_scan_3d_all"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAlexSCAN3DLoader is not implemented yet")
    
    
class JarvisAlexSUPERCONLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "alex_supercon"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAlexSUPERCONLoader is not implemented yet")
    
class JarvisAlignnFFDBLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "alignn_ff_db"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisAlignnFFDBLoader is not implemented yet")
    
    
class JarvisarXivLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "arXiv"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisarXivLoader is not implemented yet")
    
class JarvisarXivSummaryLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "arxiv_summary"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisarXivSummaryLoader is not implemented yet")
    
class JarvisC2DBLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "c2db"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisC2DBLoader is not implemented yet")
    
class JarvisCCCBDBLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "cccbdb"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisCCCBDBLoader is not implemented yet")
    
class JarvisCFID3DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "cfid_3d"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisCFID3DLoader is not implemented yet")
    
class JarvisCFID3DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "cfid_3d"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisCFID3DLoader is not implemented yet")
    

class JarvisCodLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "cod"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisCodLoader is not implemented yet")


class JarvisCord19Loader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "cord19"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisCord19Loader is not implemented yet")


class JarvisDft2DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "dft_2d"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisDft2DLoader is not implemented yet")


class JarvisDft2D2021Loader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "dft_2d_2021"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisDft2D2021Loader is not implemented yet")


class JarvisDft3DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "dft_3d"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisDft3DLoader is not implemented yet")


class JarvisDft3D2021Loader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "dft_3d_2021"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisDft3D2021Loader is not implemented yet")


class JarvisEdosPdosLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "edos_pdos"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisEdosPdosLoader is not implemented yet")


class JarvisFoundryMLExpBandgapsLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "foundry_ml_exp_bandgaps"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisFoundryMLExpBandgapsLoader is not implemented yet")


class JarvisHalidePeroskitesLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "halide_peroskites"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisHalidePeroskitesLoader is not implemented yet")


class JarvisHMOFLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "hmof"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisHMOFLoader is not implemented yet")


class JarvisHOPVLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "hopv"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisHOPVLoader is not implemented yet")


class JarvisInterfaceDBLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "interfacedb"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisInterfaceDBLoader is not implemented yet")


class JarvisJFFLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "jff"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisJFFLoader is not implemented yet")


class JarvisM3GNetMPFLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "m3gnet_mpf"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisM3GNetMPFLoader is not implemented yet")


class JarvisM3GNetMPF1_5MilLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "m3gnet_mpf_1.5mil"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisM3GNetMPF1_5MilLoader is not implemented yet")


class JarvisMag2DChemLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "mag2d_chem"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisMag2DChemLoader is not implemented yet")


class JarvisMegNetLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "megnet"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisMegNetLoader is not implemented yet")


class JarvisMegNet2Loader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "megnet2"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisMegNet2Loader is not implemented yet")


class JarvisMLearnLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "mlearn"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisMLearnLoader is not implemented yet")


class JarvisMP3DLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "mp_3d"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisMP3DLoader is not implemented yet")


class JarvisMP3D2020Loader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "mp_3d_2020"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisMP3D2020Loader is not implemented yet")


class JarvisMXene275Loader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "mxene275"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisMXene275Loader is not implemented yet")


class JarvisOCPAllLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "ocp_all"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisOCPAllLoader is not implemented yet")


class JarvisOCP10kLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "ocp10k"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisOCP10kLoader is not implemented yet")


class JarvisOCP100kLoader(BaseJarvisDownloader):
    @property
    def source_dataset(self) -> str:
        return "ocp100k"

    def load(self, filepath: Path) -> pd.DataFrame:
        raise NotImplementedError("JarvisOCP100kLoader is not implemented yet")
