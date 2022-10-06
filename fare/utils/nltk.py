from nltk.downloader import Downloader

from fare import logger
from fare.config import CONFIG


def download_nltk_dependencies(*dependencies: str):
    if CONFIG.offline:
        # Skip download.
        return

    downloader = Downloader()
    for dependency in dependencies:
        if not downloader.is_installed(dependency):
            logger.info(f"Downloading NLTK dependency {dependency}.")
            downloader.download(dependency)
