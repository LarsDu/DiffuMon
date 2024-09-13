"""Utils for automatic downloading of data for training
"""

import gzip
import hashlib
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import Callable, Sequence

import py7zr
import requests
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from diffumon.data.transforms import forward_transform


def download_file(
    url: str,
    output_file: str | Path,
    headers: dict[str, str] | None = None,
    md5sum: str | None = None,
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
    # Check the md5sum of the downloaded file
    if md5sum is not None:
        print(f"Checking md5sum of downloaded file...")
        with open(output_file, "rb") as f:
            data = f.read()
            md5 = hashlib.md5(data).hexdigest()
            if md5 != md5sum:
                raise ValueError(
                    f"MD5 checksum mismatch for downloaded file {output_file}. Expected {md5sum}, got {md5}"
                )
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


def unpack_7z(
    archive_file: str | Path,
    output_dir: str | Path,
    internal_dirs: list[Path] | None = None,
    delete_archive: bool = False,
) -> None:
    """Unpack a 7z archive to a directory

    Args:
        archive_file: The path to the 7z archive to unpack
        output_dir: The directory to unpack the archive to
        internal_dirs: The specific internal directories to unpack. If None, unpack everything
        delete_archive: Whether to delete the archive file after unpacking
    """
    archive_file = Path(archive_file)
    output_dir = Path(output_dir)
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the py7zr library to extract the archive

    with py7zr.SevenZipFile(archive_file, mode="r") as z:
        target_files = []
        if internal_dirs:
            for file_info in z.list():
                if any(
                    str(file_info.filename).startswith(str(d)) for d in internal_dirs
                ):
                    target_files.append(file_info.filename)
            z.extract(target_files, path=output_dir)
        else:
            z.extractall(output_dir)

    if delete_archive and os.path.exists(archive_file):
        print(f"Deleting archive file at {archive_file}")
        os.remove(archive_file)


def download_unpack_images(
    url: str,
    output_dir: str,
    internal_dirs: Sequence[Path | str] | None = None,
    delete_archive: bool = False,
    headers: dict[str, str] | None = None,
    md5sum: str | None = None,
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
        md5sum: (Optional) The expected md5sum of the downloaded file

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

    download_file(url, archive_file, headers=headers, md5sum=md5sum)

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
    elif archive_file.suffix == ".7z":
        unpack_7z(
            archive_file,
            output_dir=output_dir,
            internal_dirs=internal_dirs,
            delete_archive=delete_archive,
        )
    else:
        raise ValueError(f"Unsupported archive extension in {archive_file.name}")


def convert_to_rgb_with_white_bg(
    input_path: str | Path, output_path: str | Path, output_format: str = "PNG"
) -> None:
    """
    Converts a PNG image with alpha transparency to an RGB image with a white background.

    Args:
        input_path: Path to the input PNG image.
        output_path (str): Path to save the converted RGB image.
    """
    # Open the input image
    image = Image.open(input_path)

    # Check if the image has an alpha channel
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        # Create a white background image
        bg_image = Image.new("RGB", image.size, (255, 255, 255))
        # Paste the input image onto the white background using the alpha channel as a mask
        bg_image.paste(image, mask=image.split()[3])  # Use the alpha channel as mask
    else:
        # If the image has no alpha channel, convert it directly to RGB
        bg_image = image.convert("RGB")

    # Save the image in RGB format
    bg_image.save(output_path, format=output_format)


def download_pokemon_sprites(
    url: str = "https://github.com/PokeAPI/sprites/archive/refs/tags/2.0.0.tar.gz",
    transform: Callable | None = None,
    test_size: float = 0.15,
    output_dir: str | Path = "downloads/pokemon_sprites",
    internal_dirs: Sequence[str | Path] | None = ("sprites-2.0.0/sprites/pokemon",),
    md5sum: str | None = "5068352117f3cc6e5641b7c5c426592c",
    split_seed: int = 1999,
    delete_archive: bool = True,
    delete_staging: bool = True,
    img_file_extension: str = ".png",
    convert_alpha_to_white: bool = True,
) -> tuple[Dataset, Dataset]:
    """Download a pokemon sprite dataset

    Partition the dataset into 'train' and 'test' subdirectories

    Args:
        url: The URL to download the dataset
        transform: The transform to apply to the images
        test_size: The proportion of samples to use for the test set
        output_dir: The directory to save the downloaded dataset. Will create 'train' and 'test'
            subdirectories
        internal_dirs: The paths within the tarball to the images.
            If specified, only unpack the images in these directories
        split_seed: The seed to use for the random train/test split
        delete_archive: Whether to delete the downloaded files after unpacking
        delete_staging: Whether to delete the staging directory after completion
        img_file_extension: The file extension of the images in the dataset
        convert_alpha_to_white: Whether to convert the alpha channel to white
            for images with transparency
    Returns:
        The 'train' and 'test' ImageFolder datasets
    """
    output_dir = Path(output_dir)
    if internal_dirs is not None:
        internal_dirs = [Path(d) for d in internal_dirs]

    if transform is None:
        transform = forward_transform()

    # Split the images into 'train' and 'test' subdirectories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # Short circuit if these directories already exist
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Found existing train and test directories in {output_dir}")
        print("Skipping download and unpacking...")
        return (
            ImageFolder(train_dir, transform=transform),
            ImageFolder(test_dir, transform=transform),
        )

    # Download the tarball to the staging directory
    staging_dir = output_dir / "staging"

    # Create the staging directory if it doesn't exist
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Download the tarball from url to staging dir
    # Unpack only the contents of the 'archive_image_path' directory
    download_unpack_images(
        url,
        internal_dirs=internal_dirs,
        output_dir=staging_dir,
        delete_archive=delete_archive,
        md5sum=md5sum,
    )

    # Create the 'train' and 'test' directories
    os.makedirs(train_dir / "class_0", exist_ok=True)
    os.makedirs(test_dir / "class_0", exist_ok=True)

    # Randomly select images for the test set

    images = [img for img in staging_dir.rglob(f"*{img_file_extension}")]
    for image in images:
        if convert_alpha_to_white:
            # Convert the alpha channel to white
            convert_to_rgb_with_white_bg(image)
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


def download_pokemon_sprites_11k(
    transform: Callable,
    test_size: float = 0.15,
    split_seed: int = 1999,
) -> tuple[ImageFolder, ImageFolder]:
    """Download the 11,779 Pokemon sprites dataset consisting of 96x96 color png images

    NOTE: I think this dataset might actually be from https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images?resource=download

    TODO: May want to start hosting this somewhere else

    Args:
        transform: The transform to apply to the images
        test_size: The proportion of samples to use for the test set
        split_seed: The seed to use for the random train/test split

    Returns:
        The 'train' and 'test' ImageFolder datasets
    """
    return download_pokemon_sprites(
        url="https://raw.githubusercontent.com/jonasgrebe/tf-pokemon-generation/master/data/pokemon_sprite_dataset.7z",
        internal_dirs=None,
        transform=transform,
        test_size=test_size,
        output_dir="downloads/pokemon_sprites_11k",
        md5sum="8b620579e0731115e8b30d24998b8c8b",
        split_seed=split_seed,
        img_file_extension=".png",
        delete_archive=True,
        delete_staging=True,
    )
