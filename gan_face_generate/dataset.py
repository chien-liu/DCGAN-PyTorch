import zipfile
from pathlib import Path
from typing import Any

import requests
from platformdirs import user_cache_dir
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class DigiFace1MDataset(ImageFolder):
    """Dataset class for DigiFace1M images.

    Automatically downloads and extracts the dataset if not found in the specified root directory.

    Args:
        root (str | Path | None): Root directory of the dataset.
            If None, defaults to a cache directory specific to the application.
        **kwargs: Additional keyword arguments passed to `torchvision.datasets.ImageFolder`.
    """

    def __init__(self, *, root: str | Path | None = None, **kwargs: Any) -> None:
        if root is None:
            root = (
                Path(user_cache_dir(appname="gan_face_generate"))
                / self.__class__.__name__
            )

            if not root.exists():
                root.mkdir(parents=True)
                self._download_dataset(root)

        self.root = Path(root)

        super().__init__(root=root, **kwargs)

    def _download_dataset(self, root: Path) -> None:
        """Download all dataset zip files and extract them into the root directory.

        Args:
            root (Path): Directory where the dataset will be stored.
        """
        urls = [
            "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_0-1999_72_imgs.zip",
            "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_2000-3999_72_imgs.zip",
            "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_4000-5999_72_imgs.zip",
            "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_6000-7999_72_imgs.zip",
            "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_8000-9999_72_imgs.zip",
        ]

        for url in urls:
            dest = root / url.split("/")[-1]
            self._download_file(url, root / dest)
            self._extract_zip(dest, root)

    def _download_file(self, url: str, dest: str | Path) -> Path:
        """Download a file from a URL to the specified destination.

        Args:
            url (str): The URL to download from.
            dest (str | Path): The path where the file will be saved.

        Returns:
            Path: Path to the downloaded file.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Stream download with progress bar
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with (
                open(dest, "wb") as f,
                tqdm(
                    desc=dest.name,
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return dest

    def _extract_zip(self, zip_path: str | Path, extract_to: str | Path = ".") -> None:
        """Extract a ZIP archive into the specified directory.

        Args:
            zip_path (str | Path): Path to the ZIP file.
            extract_to (str | Path): Directory to extract files into. Defaults to current directory.
        """
        zip_path = Path(zip_path)
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted to: {extract_to.resolve()}")
