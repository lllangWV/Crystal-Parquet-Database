## Loaders module

This module provides a consistent interface to download and load data from
multiple crystal databases and their datasets/collections into a unified
schema. It standardizes how raw data are retrieved (JSON files on disk) and
how they are transformed into a canonical pandas DataFrame of serialized core
objects.

### Core concepts

- **BaseLoader**: Abstract base class that defines the contract every loader
  must follow. Subclasses specify which database/dataset they represent and how
  to download and load records.
- **LoaderConfig**: Configuration object shared by all loaders (data directory,
  API key, base URL, timeouts, parallelism, etc.).
- **LoaderFactory**: Helper that returns the correct loader instance given a
  `database_name` and `dataset_name`.

### BaseLoader contract

Each loader subclass must implement the following members:

- `source_database: str` (property)
- `source_dataset: str` (property)
- `_download(dirpath: Path) -> Path`
  - Download or update the raw dataset into `dirpath` as a directory of JSON
    files and return that directory path.
- `load(filepath: Path) -> pandas.DataFrame`
  - Read a single raw JSON file and return a pandas DataFrame where each row is
    a serialized core record conforming to the unified schema.

BaseLoader also provides:

- `download(dirpath: Optional[Path] = None) -> Path`
  - Orchestrates downloading. By default, uses
    `data_dir/<source_database>/<source_dataset>`.
- `transform(df: pandas.DataFrame) -> pandas.DataFrame`
  - Optional hook to post-process the DataFrame (identity by default).
- `__iter__()`
  - Iterates over raw JSON files and yields processed DataFrames per file.

### Data locations

By default, raw files are materialized under:

```
<data_dir>/<source_database>/<source_dataset>/
```

For example: `data/mp/summary/` or `data/alex/3d/`.

### Supported loaders

- `mp/summary` via `MPLoader` (Materials Project API; requires API key)
- `alex/1d`, `alex/2d`, `alex/3d` via `Alexandria*Loader` (HTTP index of
  compressed JSON, downloaded and decompressed locally)

### Quickstart

```python
from crystpqdb.loaders import LoaderConfig, get_loader

# Configure where data are stored and any source-specific settings
config = LoaderConfig(
    data_dir="./data",      # root data directory
    api_key="<MP_API_KEY>", # for MP; or set env var MP_API_KEY
    from_scratch=False,
)

# Obtain a loader via the factory
loader = get_loader("mp", "summary", config)

# Download raw dataset (directory of JSON files)
raw_dir = loader.download()
print(raw_dir)

# Iterate over raw files; each iteration yields a processed DataFrame
for df in loader:
    print(df.shape)

# Optionally concatenate into one DataFrame
# import pandas as pd
# all_df = pd.concat(list(loader), ignore_index=True)
```

Alexandria example:

```python
alex_loader = get_loader("alex", "3d", LoaderConfig(data_dir="./data"))
for df in alex_loader:
    print(df.shape)
```

### Configuration

`LoaderConfig` fields (common across loaders):

- `data_dir: Path | str` — Root directory to store datasets
- `api_key: Optional[str]` — API token if required by the source (e.g., MP)
- `base_url: Optional[str]` — Override default remote base URL
- `timeout_seconds: int` — Network request timeout
- `num_workers: int` — Parallelism for I/O bound work (downloads/decompression)
- `from_scratch: bool` — If true, remove any existing local data before
  downloading

Notes:
- For Materials Project, provide `api_key` or set environment variable
  `MP_API_KEY`.

### Implementing a new loader

1) Subclass `BaseLoader` and implement the required members.
2) Register your loader in `LoaderFactory.loaders` so the factory can return it.

Skeleton:

```python
from pathlib import Path
import pandas as pd
from crystpqdb.loaders.base import BaseLoader

class MySourceFooLoader(BaseLoader):
    @property
    def source_database(self) -> str:
        return "my_source"

    @property
    def source_dataset(self) -> str:
        return "foo"

    def _download(self, dirpath: Path) -> Path:
        # Fetch remote data and write JSON files under dirpath
        dirpath.mkdir(parents=True, exist_ok=True)
        # ... write one or more *.json files ...
        return dirpath

    def load(self, filepath: Path) -> pd.DataFrame:
        # Read one raw JSON file and produce a DataFrame of unified records
        # (each row corresponds to a serialized core record)
        # return pd.DataFrame([...])
        raise NotImplementedError
```

Finally, add an entry to the factory mapping so clients can call
`get_loader("my_source", "foo", config)`.

### Error handling and discovery

- If an invalid `(database_name, dataset_name)` pair is requested, the factory
  raises a `ValueError`. Use the factory helpers during development:
  - `LoaderFactory(config).list_databases()`
  - `LoaderFactory(config).list_datasets(database_name)`
  - `LoaderFactory(config).available_loaders()`

### What the loader returns

- `download()` returns the directory containing the raw JSON files.
- `load(filepath)` returns a pandas DataFrame with rows that serialize the core
  data model (e.g., lattice, structure, symmetry, and properties) into a
  unified schema.


