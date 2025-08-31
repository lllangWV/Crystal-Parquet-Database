from pathlib import Path

from crystpqdb.loaders.alexandria import (
    Alexandria1DLoader,
    Alexandria2DLoader,
    Alexandria3DLoader,
    get_alexandria_loader,
)
from crystpqdb.loaders.base import LoaderConfig
from crystpqdb.loaders.mc3d import MC3DLoader
from crystpqdb.loaders.mp import MPLoader

# from crystpqdb.loaders.jarvis import JarvisLoader
# from crystpqdb.loaders.mp import MPLoader


class LoaderFactory:
    def __init__(self, config: LoaderConfig):
        self.config = config
        self.loaders = {
            "alex": {"1d": Alexandria1DLoader, "2d": Alexandria2DLoader, "3d": Alexandria3DLoader},
            "mp": {"summary": MPLoader},
            "materialscloud": {"mc3d": MC3DLoader},
            # "jarvis": {"summary": JarvisLoader},
        }
        
    def get_loader(self, database_name: str, dataset_name: str):
        database_loaders = self.loaders.get(database_name, {})
        loader = database_loaders.get(dataset_name, None)
        if loader is None:
            error_msg = f"No loader found for database_name: {database_name} and dataset_name: {dataset_name}"
            error_msg += "\nAvailable loaders: \n" + self.available_loaders()
            raise ValueError(f"No loader found for database_name: {database_name} and dataset_name: {dataset_name}")
        return loader(self.config)
    
    def list_databases(self):
        return list(self.loaders.keys())
    
    def list_datasets(self, database_name: str):
        return list(self.loaders[database_name].keys())
    
    def available_loaders(self):
        tmp = "Available loaders: \n"
        for database_name in self.loaders.keys():
            tmp += f"{database_name}: \n"
            for dataset_name in self.loaders[database_name].keys():
                tmp += f"  {dataset_name}: \n"
        return tmp
    
    
def get_loader(database_name: str, dataset_name: str, data_dir: Path | str = None, config: LoaderConfig = LoaderConfig()):
    if data_dir is not None:
        config.data_dir = data_dir
    return LoaderFactory(config).get_loader(database_name, dataset_name)
