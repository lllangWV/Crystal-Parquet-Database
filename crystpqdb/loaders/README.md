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

### Intended Workflow

The intended workflow for use of these loaders is the following. 

```python
loader = Loader()
table = loader.run()
```

Some method will create the instances of the loader and then the ``run`` method will called to run the pipeline of the loader.

Interally the run method is executing the following steps in order:

1. `loader.download()`: Download the data from the source database and dataset in its raw format.
2. `loader.load()`: Load the data into a consistent dataframe to be ingested into ParquetDB to perform data inference and normalization.
3. `loader.ingest_pqdb()`: Ingest the data into ParquetDB to perform data inference, normalization, and validation.
4. `loader.normalize_pqdb()`: Normalize the disitribution of data across files and row groups.
5. `loader.read_pyarrow_table(rebuild_nested_struct=True)`: Read the ParquetDB into a PyArrow Table rebuilding any nested structure
6. `loader.transform()`: Transform the data into a unified schema.


### BaseLoader contract

Each loader subclass must implement the following members:

- `source_database: str` (property)
- `source_dataset: str` (property)
- `_download(dirpath: Path) -> Path`
  - Download or update the raw dataset into `dirpath` as a directory of JSON
    files and return that directory path.
- `_load(dirpath: Path) -> Iterable`
  - This method defines how the data is loaded into a consistent dataframe to be ingested into ParquetDB to perform data inference and normalization. This is typically a list of records, a dictionary of arrays, a `pandas DataFrame`, or a `pyarrow Table`.
- `_transform(table: pyarrow.Table) -> pyarrow.Table`
  - This method defines how the data is transformed into a unified schema. The incoming data will be in a `pyarrow Table` and the outgoing data will be a `pyarrow Table` with the unified schema. The logic of how this is done is left to the subclass.

The above abstract methods are wraped with the following public methods. This is done to provide a consistent interface to the user and take care of some business logic in the procesing of the data.

- `download(dirpath: Optional[Path] = None) -> Path`
  - Orchestrates downloading. By default, uses
    `data_dir/<source_database>/<source_dataset>`.
- `load(dirpath: Path) -> Iterable`
  - Load the data into a consistent dataframe to be ingested into ParquetDB to perform data inference and normalization.
- `transform(df: pandas.DataFrame) -> pandas.DataFrame`
  - Transform the data into a unified schema.

### Data locations


#### 1. Raw Data Files
By default, raw files are materialized under:

```
<data_dir>/<source_database>/<source_dataset>/raw/
```

For example: `data/mp/summary/raw` or `data/alex/3d/raw`.

#### 2. Interim Data Files

Intermediate steps of in the loading process are materialized under:

```
<data_dir>/<source_database>/<source_dataset>/interim/
```

For example: `data/mp/summary/interim` or `data/alex/3d/interim`.

