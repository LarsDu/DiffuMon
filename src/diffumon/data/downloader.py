"""Utils for automatic downloading of data for training
"""

import os
import shutil
import tarfile
from dataclasses import dataclass

import requests


@dataclass
class DownloadImageSource:
    """Encapsulates the source for downloading images

    Attributes:
        url: The URL to download the images from.
        internal_image_dirs: Once downloaded and unzipped, the directory containing
            the images actually intended for training
    """

    url: str
    internal_image_dirs: list[str]

    @property
    def name(self) -> str:
        return self.url.split("/")[-1].split(".")[0]


def download_file(url: str, output_path: str) -> None:
    """Download a file from a URL to a path

    Args:
        url: The URL to download the file from
        output_path: The path to save the downloaded file
    """
    with requests.get(url, stream=True) as r:
        with open(output_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    return output_path


def unpack_tarball(
    tarball_path: str,
    output_dir: str,
    delete_tarball: bool = True,
) -> None:
    """Unpack a tarball to a directory

    Args:
        tarball_path: The path to the tarball to unpack
        output_dir: The directory to unpack the tarball to
        delete_tarball: Whether to delete the tarball after unpacking
    """
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(output_dir)

    if delete_tarball:
        os.remove(tarball_path)


def download_unpack_images(
    source: DownloadImageSource,
    staging_dir: str,
    output_dir: str,
    delete_tarball: bool = True,
    delete_staging_dir: bool = True,
) -> None:
    """Download and unpack images from a URL

    Args:
        source: The source to download the images from
        staging_dir: The staging directory to unpack the images to
        output_dir: Once unpacked, copy the images in internal_image_dirs
            to this directory
        delete_tarball: Whether to delete the tarball after unpacking
        delete_staging_dir: Whether to delete the staging directory after unpacking
    """

    # Create the output directory if it doesn't exist
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Download the tarball from url to staging dir
    tarball_file = os.path.join(staging_dir, f"{source.name}.tar.gz")
    download_file(source.url, tarball_file)

    # Unpack the tarball to the staging directory
    unpack_tarball(tarball_file, staging_dir, delete_tarball=delete_tarball)

    # Move the images from various folders in the staging dir to a single output dir
    for subdir in source.internal_image_dirs:
        # Move every image in subdir to output_dir
        for file in os.listdir(os.path.join(staging_dir, subdir)):
            shutil.move(
                os.path.join(staging_dir, subdir, file),
                os.path.join(output_dir, file),
            )

    if delete_staging_dir:
        shutil.rmtree(staging_dir)
