
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
from typing import Dict, Final, Iterable, List, Optional, Type

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mp_api.client import MPRester
from parquetdb import LoadConfig, ParquetDB
from parquetdb.utils import pyarrow_utils

from crystpqdb.db import crystpqdb_schema

# from crystpqdb.db import CrystPQData, DataDict, HasPropsData, SymmetryData

load_dotenv()

CHUNK_BYTES: Final[int] = 1024 * 1024
LOGGER = logging.getLogger(__name__)

@dataclass
class LoaderConfig:
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

    data_dir: Path | str = Path("./data")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: int = 60
    num_workers: int = 8
    download_from_scratch: bool = False
    ingest_from_scratch: bool = True
    transform_from_scratch: bool = True
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)


class BaseLoader(ABC):
    """Abstract interface for loading a crystal database locally.

    Implementations should handle retrieving the dataset from the
    configured remote source and materializing it under the given
    directory path. Implementations must be idempotent and safe to
    re-run; callers may invoke ``load`` multiple times.

    Notes
    ----
    - Implementations should create the target directory if it does not
      exist.
    - The return value should be the directory containing the
      downloaded dataset for easy chaining.
    - Keep network I/O contained within this layer; transformation of
      the downloaded files should happen elsewhere.
    """

    def __init__(self, config: LoaderConfig) -> None:
        self._config = config

    @property
    @abstractmethod
    def source_database(self) -> str:
        """Return the source name."""
        pass
    
    @property
    @abstractmethod
    def source_dataset(self) -> str:
        """Return the dataset name."""
        pass
    
    @property
    def dataset_dir(self) -> Path:
        """Return the directory path for the dataset."""
        return self.config.data_dir / self.source_database / self.source_dataset
    
    @property
    def raw_dir(self) -> Path:
        return self.dataset_dir / "raw"
    
    @property
    def interim_dir(self) -> Path:
        return self.dataset_dir / "interim"
    
    @property
    def pqdb_dir(self) -> Path:
        return self.interim_dir / "pqdb"
    
    @property
    def transformed_dir(self) -> Path:
        return self.interim_dir / "transformed_pqdv"
    
    @property
    def pqdb(self) -> ParquetDB:
        """Return the ParquetDB."""
        return ParquetDB(self.pqdb_dir)
    
    @property
    def transformed_pqdb(self) -> ParquetDB:
        """Return the ParquetDB."""
        return ParquetDB(self.transformed_dir)
    
    @property
    def config(self) -> LoaderConfig:
        """Return the immutable downloader configuration."""
        return self._config
    
    def download(self, dirpath: Path | str | None = None) -> Path:
        dirpath = Path(dirpath) if dirpath is not None else self.raw_dir
        if self.config.download_from_scratch and dirpath.exists():
            LOGGER.info("from_scratch=True, removing existing directory: %s", dirpath)
            shutil.rmtree(dirpath, ignore_errors=True)
        
        if dirpath.exists():
            LOGGER.info("Directory %s already exists and is not empty", dirpath)
            return dirpath
        else:
            LOGGER.info("Downloading dataset into %s", dirpath)
            return self._download(dirpath)

    @abstractmethod
    def _download(self, dirpath: Path) -> Path:
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
    
    def load(self, dirpath: Path | str) -> Iterable[pa.Table]:
        """Load the dataset from ``dirpath``."""
        dirpath = Path(dirpath)
        yield from self._load(dirpath)
        
    @abstractmethod
    def _load(self, filepath: Path) -> Iterable:
        """Load the dataset from ``filepath``."""
        raise NotImplementedError
    
    def ingest_pqdb(self, data: dict | list[dict] | pd.DataFrame) -> None:
        """Inject the ParquetDB into the dataset."""
        self.pqdb.create(data, convert_to_fixed_shape=False)
        
    def normalize_pqdb(self, **kwargs) -> None:
        """Normalize the ParquetDB."""
        self.pqdb.normalize(**kwargs)
        
    def read_pyarrow_table(self, **kwargs) -> pa.Table:
        """Read the ParquetDB into a PyArrow Table."""
        if "rebuild_nested_struct" not in kwargs:
            kwargs["rebuild_nested_struct"] = True
        
        return self.pqdb.read(**kwargs)
        
    def transform(self, table: pa.Table) -> pa.Table:
        """Transform raw records into validated ``CrystPQRecord`` objects.

        Parameters
        ----
        data : dict
            Data to transform.

        Returns
        ----
        list of CrystPQRecord
            Validated records ready to be serialized and ingested.
        """
        table = self._transform(table)
        
        merged_schema = pyarrow_utils.unify_schemas(
            [table.schema, crystpqdb_schema], promote_options="permissive"
        )
        modified_incoming_table = pyarrow_utils.table_schema_cast(
            table, merged_schema
        )
        return modified_incoming_table
    
    @abstractmethod
    def _transform(self, table: pa.Table) -> pa.Table:
        """Transform raw records into validated ``CrystPQRecord`` objects."""
        raise NotImplementedError
    
    
    def run(self, load_format: str = "table", load_config: LoadConfig = LoadConfig()):
        if self.config.download_from_scratch and self.raw_dir.exists():
            shutil.rmtree(self.raw_dir, ignore_errors=True)
            
        self.download()
        
        if self.config.ingest_from_scratch and self.pqdb_dir.exists():
            shutil.rmtree(self.pqdb_dir, ignore_errors=True)
            
            print(f"Loading from {self.raw_dir} into {self.pqdb_dir}")
            for data in self.load(self.raw_dir):
                self.ingest_pqdb(data)
            
            self.normalize_pqdb()
            
        table = self.read_pyarrow_table(load_format=load_format, load_config=load_config)
            
        if self.config.transform_from_scratch and self.transformed_dir.exists():
            shutil.rmtree(self.transformed_dir, ignore_errors=True)
        return self.transform(table)

