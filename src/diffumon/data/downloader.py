"""Utils for automatic downloading of data for training
"""

import gzip
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import Callable, Sequence

import requests
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from diffumon.data.transforms import forward_transform


def download_file(
    url: str, output_file: str | Path, headers: dict[str, str] | None = None
) -> None:
    """Download a file from a URL to a path

    Args:
        url: The URL to download the file from
        output_path: The path to save the downloaded file
    """
    output_file = Path(output_file)
    # Skip if already downloaded
    if os.path.exists(output_file):
        print(f"Found existing file at {output_file}")
        return

    print(f"Downloading {url} to {output_file}")
    if headers is not None:
        print(f"Using headers {headers}")
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        # Show progress bar for the download
        # ref: https://stackoverflow.com/questions/56795229
        pbar = tqdm(total=int(r.headers.get("Content-Length", 0)))
        with open(output_file, "wb") as f:
            for data in r.iter_content(chunk_size=1024):
                if data:
                    f.write(data)
                    pbar.update(len(data))

    return output_file


def unpack_tarball(
    tarball_path: str | Path,
    output_dir: str | Path,
    internal_dirs: list[Path],
    delete_tarball: bool = False,
    extension: str = "gz",
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
                if any(str(member.name).startswith(str(d)) for d in internal_dirs):
                    # print(f"Extracting {member.name}")
                    members_to_extract.append(member)
            tar.extractall(path=output_dir, members=members_to_extract)
        else:
            # Extract everything
            tar.extractall(output_dir)

    if delete_tarball and os.path.exists(tarball_path):
        print(f"Deleting tarball at {tarball_path}")
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

    if delete_gzip and gzip_file.exists():
        print(f"Deleting gzip file at {gzip_file}")
        os.remove(gzip_file)


def download_unpack_images(
    url: str,
    output_dir: str,
    internal_dirs: Sequence[Path | str] | None = None,
    delete_archive: bool = False,
    headers: dict[str, str] | None = None,
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
        headers: Headers to pass to the request

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

    download_file(url, archive_file, headers=headers)

    # Unpack the archive to the staging directory
    if archive_file.name.endswith(".tar.gz"):
        unpack_tarball(
            archive_file,
            output_dir=output_dir,
            internal_dirs=internal_dirs,
            delete_tarball=delete_archive,
            extension="gz",
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


def download_pokemon_sprites(
    url: str = "https://github.com/PokeAPI/sprites/archive/refs/tags/2.0.0.tar.gz",
    transform: Callable | None = None,
    test_size: float = 0.15,
    output_dir: str | Path = "downloads/pokemon_sprites",
    archive_image_path: str | Path = "sprites-2.0.0/sprites/pokemon",
    split_seed: int = 1999,
    delete_archive: bool = True,
    delete_staging: bool = True,
) -> tuple[Dataset, Dataset]:
    """Download a pokemon sprite dataset

    Partition the dataset into 'train' and 'test' subdirectories

    Args:
        url: The URL to download the dataset
        transform: The transform to apply to the images
        test_size: The proportion of samples to use for the test set
        output_dir: The directory to save the downloaded dataset. Will create 'train' and 'test'
            subdirectories
        archive_image_path: The path within the tarball to the images.
            If specified, only unpack the images in this directory
        split_seed: The seed to use for the random train/test split
        delete_archive: Whether to delete the downloaded files after unpacking
        delete_staging: Whether to delete the staging directory after completion

    Returns:
        The 'train' and 'test' ImageFolder datasets
    """
    output_dir = Path(output_dir)
    archive_image_path = Path(archive_image_path)

    if transform is None:
        transform = forward_transform()

    # Split the images into 'train' and 'test' subdirectories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # Short circuit if these directories already exist
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Found existing train and test directories in {output_dir}")
        print("Skipping download and unpacking...")
        return train_dir, test_dir

    # Download the tarball to the staging directory
    staging_dir = output_dir / "staging"

    # Create the staging directory if it doesn't exist
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Download the tarball from url to staging dir
    # Unpack only the contents of the 'archive_image_path' directory
    download_unpack_images(
        url,
        internal_dirs=[archive_image_path],
        output_dir=staging_dir,
        delete_archive=delete_archive,
    )

    # Create the 'train' and 'test' directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Randomly select images for the test set
    images = list((staging_dir / archive_image_path).glob("*.png"))
    random.seed(split_seed)
    random.shuffle(images)
    # The first test_size proportion of images are for the test set
    test_images = images[: int(test_size * len(images))]
    # The remaining images are for the training set
    train_images = images[int(test_size * len(images)) :]

    print(
        f"Randomly partitioning images into {len(train_images)} train and {len(test_images)} test samples with seed {split_seed}"
    )
    # Copy the images to the 'train' and 'test' directories
    for img in train_images:
        if img.is_dir():
            continue
        shutil.copy(img, train_dir / "class_0" / img.name)
    for img in test_images:
        if img.is_dir():
            continue
        shutil.copy(img, test_dir / "class_0" / img.name)

    if delete_staging:
        shutil.rmtree(staging_dir)

    return (
        ImageFolder(train_dir, transform=transform),
        ImageFolder(test_dir, transform=transform),
    )


def download_mnist(
    train_url: str = "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    test_url: str = "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    output_dir: str | Path = "downloads/mnist",
) -> tuple[Dataset, Dataset]:
    """Download the MNIST dataset
    Partition the 60,000 train, 10,000 test samples into 'train' and 'test' subdirectories

    Slight issue with 403 forbidden error when downloading from the original source
    ref: https://stackoverflow.com/questions/60548000

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

    # Create train and test directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # Short circuit if these directories already exist
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Found existing train and test directories in {output_dir}")
        print("Skipping download and unpacking...")
        return train_dir, test_dir

    # Create the 'train' and 'test' directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    lecunn_header = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
    }

    # Download the training set
    train_archive = download_file(
        train_url, output_dir / os.path.basename(train_url), headers=lecunn_header
    )

    test_archive = download_file(
        test_url, output_dir / os.path.basename(test_url), headers=lecunn_header
    )

    return train_dir, test_dir
