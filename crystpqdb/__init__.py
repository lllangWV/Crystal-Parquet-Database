import logging

from crystpqdb.download import download, upload

from .utils.log import setup_logging

# Configure logging for the package and expose logging.get_logger
setup_logging()

