"""Utils for automatic downloading of data for training
"""

import gzip
import os
import shutil
import tarfile
from pathlib import Path
from random import random
from typing import Sequence

import requests
from tqdm import tqdm


def download_file(url: str, output_path: str) -> None:
    """Download a file from a URL to a path

    Args:
        url: The URL to download the file from
        output_path: The path to save the downloaded file
    """
    # Skip if already downloaded
    if Path(output_path).exists():
        print(f"Found existing file at {output_path}")
        return

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
    internal_dirs: list[Path],
    delete_tarball: bool = False,
    extension: str = "tar.gz",
) -> None:
    """Unpack a tarball to a directory

    Args:
        tarball_path: The path to the tarball to unpack
        output_dir: The directory to unpack the tarball to
        internal_dirs: The specific internal directories to unpack. If empty, unpack everything
        delete_tarball: Whether to delete the tarball after unpacking
        extension: The extension of the tarball
    """
    tarball_path = Path(tarball_path)
    output_dir = Path(output_dir)
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball_path, f"r:{extension}") as tar:
        if internal_dirs:
            members_to_extract = []
            # Extract only the specified internal directories
            for member in tar.getmembers():
                if any(str(member).startswith(str(d)) for d in internal_dirs):
                    members_to_extract.append(member)
            tar.extractall(path=output_dir, members=members_to_extract)
        else:
            # Extract everything
            tar.extractall(output_dir)

    if delete_tarball:
        os.remove(tarball_path)


def unpack_gzip(
    gzip_file: str | Path,
    output_dir: str | Path,
    delete_gzip: bool = False,
) -> None:
    """Unpack a gzip file to an output file

    Args:
        gzip_file: The path to the gzip file to unpack
        output_dir: The path to save the unpacked file
        delete_gzip: Whether to delete the gzip file after unpacking
    """
    gzip_file = Path(gzip_file)
    output_dir = Path(output_dir)

    with gzip.open(gzip_file, "rb") as f_in:
        with open(output_dir, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if delete_gzip:
        os.remove(gzip_file)


def download_unpack_images(
    url: str,
    output_dir: str,
    internal_dirs: Sequence[Path | str] | None = None,
    delete_archive: bool = False,
) -> None:
    """Download and unpack images from a URL

    Args:
        url: The source to download the images from
        output_dir: Once unpacked, copy the images in
            internal_dirs to this directory
        internal_dirs: The paths within the archive to
            the desired images. If None, unpack the whole archive
        delete_archive: Whether to delete the tarball or
            gzip after unpacking

    Returns:
        The path to the output directory
    """

    # Enforce Path type for consistency
    output_dir = Path(output_dir)
    if internal_dirs:
        internal_dirs = [Path(d) for d in internal_dirs]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download the tarball from url to output_dir
    # Extract basename from url
    basename = os.path.basename(url)
    archive_file: Path = output_dir / basename
    download_file(url, archive_file)

    # Unpack the archive to the staging directory
    if archive_file.suffix == ".tar.gz":
        unpack_tarball(
            archive_file,
            output_dir=output_dir,
            internal_dirs=internal_dirs,
            delete_tarball=delete_archive,
            extension=".tar.gz",
        )
    elif archive_file.suffix == ".gz":
        unpack_gzip(
            archive_file,
            output_dir=output_dir,
            delete_gzip=delete_archive,
        )
    elif archive_file.suffix == ".tar":
        unpack_tarball(
            archive_file,
            output_dir=output_dir,
            internal_dirs=internal_dirs,
            delete_tarball=delete_archive,
            extension=".tar",
        )
    else:
        raise ValueError(f"Unsupported archive extension in {archive_file.name}")

    if delete_archive:
        os.remove(archive_file)


def download_pokemon_sprites(
    url: str = "https://github.com/PokeAPI/sprites/archive/refs/tags/2.0.0.tar.gz",
    test_size: float = 0.15,
    output_dir: str | Path = "downloads/pokemon_sprites",
    archive_image_path: str | Path = "sprites-2.0.0/sprites/pokemon",
    split_seed: int = 1999,
    delete_archive: bool = True,
    delete_staging: bool = True,
) -> tuple[Path, Path]:
    """Download a pokemon sprite dataset

    Partition the dataset into 'train' and 'test' subdirectories

    Args:
        url: The URL to download the dataset
        test_size: The proportion of samples to use for the test set
        output_dir: The directory to save the downloaded dataset. Will create 'train' and 'test' subdirectories
        split_seed: The seed to use for the random train/test split

    Returns:
        The paths to the 'train' and 'test' subdirectories
    """
    output_dir = Path(output_dir)
    archive_image_path = Path(archive_image_path)

    # Download the tarball to the staging directory
    staging_dir = output_dir / "staging"

    # Create the staging directory if it doesn't exist
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Download the tarball from url to staging dir
    download_unpack_images(
        url,
        internal_dirs=[archive_image_path],
        output_dir=staging_dir,
        delete_archive=delete_archive,
    )

    # Split the images into 'train' and 'test' subdirectories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # Short circuit if these directories already exist
    if train_dir.exists() and test_dir.exists():
        print(f"Found existing train and test directories in {output_dir}")
        print("Skipping download and unpacking...")
        return train_dir, test_dir

    # Create the 'train' and 'test' directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Randomly select images for the test set
    images = list(staging_dir.glob("*"))
    random.seed(split_seed)
    random.shuffle(images)
    # The first test_size proportion of images are for the test set
    test_images = images[: int(test_size * len(images))]
    # The remaining images are for the training set
    train_images = images[int(test_size * len(images)) :]

    # Copy the images to the 'train' and 'test' directories
    for img in train_images:
        shutil.copy(img, train_dir / img.name)
    for img in test_images:
        shutil.copy(img, test_dir / img.name)

    if delete_staging:
        shutil.rmtree(staging_dir)

    return train_dir, test_dir


def download_mnist(
    train_url: str = "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    test_url: str = "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    output_dir: str | Path = "downloads/mnist",
    delete_archives: bool = True,
) -> tuple[Path, Path]:
    """Download the MNIST dataset
    Partition the 60,000 train, 10,000 test samples into 'train' and 'test' subdirectories

    Args:
        train_url: The URL to download the training set
        test_url: The URL to download the test set
        output_dir: The directory to save the downloaded dataset. Will create 'train' and 'test' subdirectories
        delete_archives: Whether to delete the downloaded files after unpacking
    Returns:
        The paths to the 'train' and 'test' subdirectories
    """

    # Enforce Path type for consistency
    output_dir = Path(output_dir)

    # Download the tarball to the staging directory
    staging_dir = output_dir / "staging"

    # Create the staging directory if it doesn't exist
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Create train and test directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # Short circuit if these directories already exist
    if train_dir.exists() and test_dir.exists():
        print(f"Found existing train and test directories in {output_dir}")
        print("Skipping download and unpacking...")
        return train_dir, test_dir

    # Create the 'train' and 'test' directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Download the tarball from url to staging dir
    download_unpack_images(
        train_url,
        archive_image_path=None,  # Unpack the whole thing
        output_dir=train_dir,
        delete_archive=delete_archives,
    )

    download_unpack_images(
        test_url,
        archive_image_path=None,  # Unpack the whole thing
        output_dir=test_dir,
        delete_archive=delete_archives,
    )

    return train_dir, test_dir
