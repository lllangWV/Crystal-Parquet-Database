# Crystal-Parquet-Database

Crystal Parquet Database (crystpqdb) is a Python library to build a unified local database of crystal structures by downloading datasets from multiple sources (Alexandria, Materials Project, Materials Cloud, andJARVIS) into a consistent on-disk layout.

## Installation

### 1. PyPi
```bash
pip install crystpqdb
```


### 2. Manually

To install and use this package we use conda package manager for `conda` packages and [Pixi](https://pixi.sh/latest/) to handle package depenedcies and virtual environements.

#### 1. Install Miniforge

**Miniforge** is the community (conda-forge) driven minimalistic `conda` installer. Subsequent package installations come thus from conda-forge channel.

This is in comparison to **Miniconda** is the Anaconda (company) driven minimalistic `conda` installer. Subsequent package installations come from the `anaconda` channels (`default` or otherwise).

[Download here](https://github.com/conda-forge/miniforge#install)


#### 2. Install Pixi package manager

**Linux/macOS**

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

**Windows (PowerShell)**

```bash
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

#### 3. Cloning the repo

```bash
git clone https://github.com/YKK-xTechLab-Engineering/YKK-Point-Cloud.git
```

#### 4. Install dependencies and virtual environments through Pixi

```bash
pixi install
```

## Quickstart

All downloads are created via a small factory and a per-source `DownloadConfig`.

### 1. Download the database combinded database

```python
from pathlib import Path
from crystpqdb import download

data_root = Path("./data")
db_dir = data_root / "crystpqdb"
db_dir = download(db_dir)
print("Downloaded to: {}".format(db_dir))
```

### 2. Uses the Loaders to download datasets from different sources

This package uses defines a common `BaseLoader` interface to download datasets and transform them into a unified schema.

A factory method `LoaderFactory` or `get_loader` is used to get the correct loader for a given source and dataset. The name of the source_database and source_dataset are used to get the correct loader. If you do not know the name of the source and dataset, you can use the `LoaderFactory` to list all available sources and datasets, or and error will be raised and it will list the available sources databases and datasets.

```python
import os
from crystpqdb.loaders import get_loader, LoaderConfig

# Define Configurations for the loader
config = LoaderConfig(
    api_key=os.getenv("MP_API_KEY"),
    download_from_scratch=False,
    ingest_from_scratch=True,
    transform_from_scratch=True
    )

# Get the loader
loader = get_loader("mp", "summary", data_dir=data_root, config=config)

# Run the loader
table = loader.run()
print(table.shape)
```

### 3. Loading all datasets into a single ParuqetDB

```python
import os
from pathlib import Path
from parquetdb import ParquetDB

from crystpqdb.loaders import get_loader, LoaderConfig

datasets = [
    ("alex", "3d"),
    ("alex", "2d"),
    ("alex", "1d"),
    ("mp", "summary"),
    ("materialscloud", "mc3d"),
]

for source_database, source_dataset in datasets:
    loader = get_loader(source_database, source_dataset, data_dir=data_dir)
    table = loader.run()
    pqdb.create(table, convert_to_fixed_shape=False)

table = pqdb.read(columns = ["id"])
print(table.shape)
```

> Note: This requires alot of memory (~64GB RAM) to load all the datasets into a single ParquetDB. Batch support is not yet implemented.

## Current Loaders

| Loader Class         | (source_database, source_dataset) | Working? |
|----------------------|-----------------------------------|----------|
| Alexandria1DLoader   | ("alex", "1d")                    | ✅        |
| Alexandria2DLoader   | ("alex", "2d")                    | ✅        |
| Alexandria3DLoader   | ("alex", "3d")                    | ✅        |
| MPLoader             | ("mp", "summary")                 | ✅        |
| MC3DLoader           | ("materialscloud", "mc3d")        | ✅        |
| JarvisLoader          |              | ❌        |

All listed loaders are currently implemented and functional. If you attempt to use a (source_database, source_dataset) pair not in this table, a `ValueError` will be raised and the available options will be listed.

