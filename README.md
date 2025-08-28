# Crystal-Parquet-Database

Crystal Parquet Database (crystpqdb) is a Python library to build a unified local database of crystal structures by downloading datasets from multiple sources (Alexandria, Materials Project, JARVIS) into a consistent on-disk layout.

It provides:
- An abstract downloader interface and a simple factory to select sources by name
- Idempotent, parallel downloads with optional from-scratch cleanup
- Source-specific adapters for Alexandria (1D/2D/3D), Materials Project, and JARVIS


## Installation

To install and use this package we use conda package manager for `conda` packages and [Pixi](https://pixi.sh/latest/) to handle package depenedcies and virtual environements.

### 1. Install Miniforge

**Miniforge** is the community (conda-forge) driven minimalistic `conda` installer. Subsequent package installations come thus from conda-forge channel.

This is in comparison to **Miniconda** is the Anaconda (company) driven minimalistic `conda` installer. Subsequent package installations come from the `anaconda` channels (`default` or otherwise).

[Download here](https://github.com/conda-forge/miniforge#install)


### 2. Install Pixi package manager

**Linux/macOS**

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

**Windows (PowerShell)**

```bash
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

### 3. Cloning the repo

```bash
git clone https://github.com/YKK-xTechLab-Engineering/YKK-Point-Cloud.git
```

### 4. Install dependencies and virtual environments through Pixi

```bash
pixi install
```

## Quickstart

All downloads are created via a small factory and a per-source `DownloadConfig`.

```python
from pathlib import Path
from crystpqdb.io import get_downloader, DownloadConfig

data_root = Path("./data")

# Alexandria 3D
cfg = DownloadConfig(source_name="alexandria3d", num_workers=8)
dl = get_downloader("alexandria3d", cfg)
out = dl.download(data_root / "alexandria" / "3d")
print(out)  # → ./data/alexandria/3d
```

### Alexandria downloaders

Alexandria downloaders scrape the index to discover `alexandria_*.json.bz2`, download in parallel, and decompress into the target directory root. Only uncompressed files remain.

```python
from pathlib import Path
from crystpqdb.io import get_downloader, DownloadConfig

data_root = Path("./data/alexandria")

# 1D
cfg_1d = DownloadConfig(source_name="alexandria1d", num_workers=8)
get_downloader("alexandria1d", cfg_1d).download(data_root / "1d")

# 2D
cfg_2d = DownloadConfig(source_name="alexandria2d", num_workers=8)
get_downloader("alexandria2d", cfg_2d).download(data_root / "2d")

# 3D
cfg_3d = DownloadConfig(source_name="alexandria3d", num_workers=8)
get_downloader("alexandria3d", cfg_3d).download(data_root / "3d")
```

Use `from_scratch=True` to remove an existing directory before downloading:

```python
cfg = DownloadConfig(source_name="alexandria3d", num_workers=8, from_scratch=True)
get_downloader("alexandria3d", cfg).download(Path("./data/alexandria/3d"))
```

### Materials Project downloader

Requires an API key. Provide it via `DownloadConfig.api_key` or the `MP_API_KEY` environment variable.

```python
from pathlib import Path
from crystpqdb.io import get_downloader, DownloadConfig

# via env var MP_API_KEY or set api_key explicitly
cfg = DownloadConfig(source_name="materials_project", api_key="YOUR_API_KEY")
dl = get_downloader("materials_project", cfg)
out = dl.download(Path("./data/materials_project"))
print(out)
```

### JARVIS downloader

The JARVIS downloader requires `jarvis-tools` and a dataset name. It validates the name against `jarvis.db.figshare.get_db_info()` and lists valid options on error.

- Download a single dataset into a directory:

```python
from pathlib import Path
from crystpqdb.io import get_downloader, DownloadConfig

dataset = "alex_pbe_2d_all"
cfg = DownloadConfig(source_name="jarvis", dataset_name=dataset)
dl = get_downloader("jarvis", cfg)
out = dl.download(Path(f"./data/jarvis/{dataset}"))
print(out)
```

- Download all datasets (dataset_name=None or "all"). Each dataset is placed in a subdirectory under the target path:

```python
from pathlib import Path
from crystpqdb.io import get_downloader, DownloadConfig

cfg = DownloadConfig(source_name="jarvis", dataset_name=None)
dl = get_downloader("jarvis", cfg)
out_root = dl.download(Path("./data/jarvis"))
print(out_root)
```

### Logging

The package configures logging on import. Use your preferred pattern in your modules/scripts:

```python
import logging
LOGGER = logging.get_logger(__file__)
```

### Factory names

Valid `source_name` values for the factory:
- `alexandria1d`
- `alexandria2d`
- `alexandria3d`
- `materials_project` (aliases: `materials-project`, `materialsproject`)
- `jarvis`

### Notes

- Downloads are idempotent and safe to re-run. For Alexandria, uncompressed files in the target directory short-circuit re-download.
- Network `timeout_seconds`, parallelism `num_workers`, and `from_scratch` are configurable via `DownloadConfig`.



