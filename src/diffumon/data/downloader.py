"""Utils for automatic downloading of data for training
"""

import gzip
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str | Path, output_path: str) -> None:
    """Download a file from a URL to a path

    Args:
        url: The URL to download the file from
        output_path: The path to save the downloaded file
    """
    url = Path(url)
    print(f"Downloading {url} to {output_path}")
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            # Show progress bar for the download
            for data in tqdm(
                r.iter_content(chunk_size=1024),
                total=total_size,
                unit="B",
                unit_scale=True,
            ):
                f.write(data)

    return output_path


def unpack_tarball(
    tarball_path: str | Path,
    output_dir: str | Path,
    delete_tarball: bool = False,
    extension: str = "gz",
) -> None:
    """Unpack a tarball to a directory

    Args:
        tarball_path: The path to the tarball to unpack
        output_dir: The directory to unpack the tarball to
        delete_tarball: Whether to delete the tarball after unpacking
        extension: The extension of the tarball
    """
    tarball_path = Path(tarball_path)
    with tarfile.open(tarball_path, f"r:{extension}") as tar:
        tar.extractall(output_dir)

    if delete_tarball:
        os.remove(tarball_path)


def unpack_gzip(
    gzip_file: str | Path, output_file: str | Path, delete_gzip: bool = False
) -> None:
    """Unpack a gzip file to an output file

    Args:
        gzip_file: The path to the gzip file to unpack
        output_file: The path to save the unpacked file
    """
    gzip_file = Path(gzip_file)
    output_file = Path(output_file)

    with gzip.open(gzip_file, "rb") as f_in:
        with open(output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if delete_gzip:
        os.remove(gzip_file)


def download_unpack_images(
    url: str | Path,
    archive_image_path: str | Path,
    output_dir: str,
    delete_archive: bool = False,
) -> None:
    """Download and unpack images from a URL

    Args:
        source: The source to download the images from
        archive_image_path: The path within the archive to the desired images
        output_dir: Once unpacked, copy the images in internal_image_dirs
            to this directory
        delete_archive: Whether to delete the tarball or gzip after unpacking
    """
    url = Path(url)
    staging_dir = Path(staging_dir)

    # Create the output directory if it doesn't exist
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Download the tarball from url to staging dir
    archive_file: Path = staging_dir / url.name
    download_file(url, archive_file)

    # Unpack the tarball to the staging directory
    if archive_file.suffix == ".gz":
        unpack_gzip(archive_file, archive_image_path, delete_gzip=delete_archive)
    elif archive_file.suffix == ".tar.gz":
        unpack_tarball(
            archive_file,
            staging_dir,
            extension=".tar.gz",
            delete_tarball=delete_archive,
        )
    elif archive_file.suffix == ".tar":
        unpack_tarball(
            archive_file, staging_dir, extension=".tar", delete_tarball=delete_archive
        )
    else:
        raise ValueError(f"Unsupported archive extension in {archive_file.name}")

    if delete_archive:
        os.remove(archive_file)


### DATASET SPECIFIC DOWNLOADERS ###
@dataclass
class SplitDir:
    """Where to find the training, validation, and test splits

    Attributes:
        train: The training split directory
        val: The validation split directory
        test: The test split directory
    """

    train: Path
    val: Path
    test: Path


def download_pokemon(
    url: str = "https://github.com/PokeAPI/sprites/archive/refs/tags/2.0.0.tar.gz",
) -> SplitDir:
    pass


def download_mnist(
    train_url: str = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    test_url: str = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    # train_labels_url: str = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    # test_labels_url: str = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    val_size: int = 10000,
) -> SplitDir:
    pass
