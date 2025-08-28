from typing import Dict, Type  # type: ignore

from .base import DatabaseDownloader, DownloadConfig
from .downloaders import (
    Alexandria1DDownloader,
    Alexandria2DDownloader,
    Alexandria3DDownloader,
    JarvisDownloader,
    MaterialsProjectDownloader,
)

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


