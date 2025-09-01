import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "lllangWV/CrystPQDB"
REPO_TYPE = "dataset"

def download(dirpath: Path | str):
    dirpath = Path(dirpath)
    outpath =snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                local_dir=dirpath)
    print(f"Downloaded: {outpath}")
    return outpath

import threading


def _upload_file_thread(api, file):
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file.name,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

def upload(db_path: Path | str):
    api = HfApi()
    threads = []
    for file in Path(db_path).glob("*.parquet"):
        t = threading.Thread(target=_upload_file_thread, args=(api, file))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()