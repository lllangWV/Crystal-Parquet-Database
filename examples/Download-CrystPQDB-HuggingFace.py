import os
import shutil
from pathlib import Path

from parquetdb import ParquetDB

from crystpqdb.loaders import get_loader

CURRENT_DIR = Path(__file__).parent
ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = ROOT_DIR / "data"

print("ROOT_DIR: {}".format(ROOT_DIR))
print("DATA_DIR: {}".format(DATA_DIR))
print("CURRENT_DIR: {}".format(CURRENT_DIR))

DB_DIR = DATA_DIR / "test-crystpqdb"

from crystpqdb.download import download

output_dir = download(DB_DIR)
print(f"Downloaded to: {output_dir}")
