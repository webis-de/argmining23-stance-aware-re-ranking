from nltk.downloader import Downloader

from stare import logger
from stare.config import CONFIG


def download_nltk_dependencies(*dependencies: str):
    downloader = Downloader()
    for dependency in dependencies:
        if not downloader.is_installed(dependency):
            logger.info(f"Downloading NLTK dependency {dependency}.")
            downloader.download(dependency)
