import logging
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

db_dir = DATA_DIR / "crystpqdb"
if db_dir.exists():
    shutil.rmtree(db_dir)
pqdb = ParquetDB(db_dir)


datasets = [
    ("alex", "3d"),
    ("alex", "2d"),
    ("alex", "1d"),
    ("mp", "summary"),
    ("materialscloud", "mc3d"),
]


for dataset in datasets:
    database_name, dataset_name = dataset
    
    print(f"Creating {database_name} {dataset_name}...")
    loader = get_loader(database_name, dataset_name, data_dir=DATA_DIR)
    for idx, df in enumerate(loader):
        print(f"Creating {database_name} {dataset_name}... {idx}")
        
        pqdb.create(df, convert_to_fixed_shape=False)
    
    