from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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


