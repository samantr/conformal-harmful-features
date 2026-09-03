import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DRY_BEAN_URL = (
    "https://archive.ics.uci.edu/static/public/602/"
    "dry%2Bbean%2Bdataset.zip"
)
DRY_BEAN_ARCHIVE_SHA256 = (
    "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
)
DRY_BEAN_ARCHIVE_MEMBER = "DryBeanDataset/Dry_Bean_Dataset.arff"
DRY_BEAN_ARFF_ATTRIBUTES = (
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
    "Class",
)
DRY_BEAN_FEATURE_NAMES = (
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
)
DRY_BEAN_CLASS_NAMES = (
    "SEKER",
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "HOROZ",
    "SIRA",
    "DERMASON",
)


@dataclass(frozen=True)
class TabularFeature:
    index: int
    name: str
    role: str = "measured"
    source_feature: str | None = None


@dataclass(frozen=True)
class RealTabularDataset:
    name: str
    features: np.ndarray
    labels: np.ndarray
    feature_manifest: tuple[TabularFeature, ...]
    class_names: tuple[str, ...]
    source_url: str
    archive_sha256: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.feature_manifest)


def load_dry_bean(
    archive_path: str | Path,
    *,
    source_url: str = DRY_BEAN_URL,
    expected_sha256: str = DRY_BEAN_ARCHIVE_SHA256,
) -> RealTabularDataset:
    """Load the official UCI Dry Bean ARFF from a verified local cache.

    If the archive is absent, it is downloaded atomically from UCI. The
    published archive digest is checked before any data are parsed so a stale
    or silently changed source cannot alter an experiment unnoticed.
    """
    archive = Path(archive_path)
    if not archive.exists():
        _download_verified_archive(archive, source_url, expected_sha256)
    observed_sha256 = _file_sha256(archive)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Dry Bean archive checksum mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )

    with zipfile.ZipFile(archive) as bundle:
        try:
            arff_bytes = bundle.read(DRY_BEAN_ARCHIVE_MEMBER)
        except KeyError as error:
            raise ValueError(
                f"Dry Bean archive lacks {DRY_BEAN_ARCHIVE_MEMBER}"
            ) from error
    features, labels = _parse_dry_bean_arff(arff_bytes)
    manifest = tuple(
        TabularFeature(index=index, name=name)
        for index, name in enumerate(DRY_BEAN_FEATURE_NAMES)
    )
    return RealTabularDataset(
        name="dry_bean",
        features=features,
        labels=labels,
        feature_manifest=manifest,
        class_names=DRY_BEAN_CLASS_NAMES,
        source_url=source_url,
        archive_sha256=observed_sha256,
    )


def _download_verified_archive(
    destination: Path, source_url: str, expected_sha256: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(source_url, timeout=60) as response:
            payload = response.read()
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                "Downloaded Dry Bean archive checksum mismatch: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        temporary.write_bytes(payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_dry_bean_arff(
    arff_bytes: bytes, *, expected_n_samples: int = 13_611
) -> tuple[np.ndarray, np.ndarray]:
    try:
        text = arff_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Dry Bean ARFF is not valid UTF-8") from error
    lines = [line.strip() for line in io.StringIO(text)]
    data_markers = [index for index, line in enumerate(lines) if line.lower() == "@data"]
    if len(data_markers) != 1:
        raise ValueError("Dry Bean ARFF must contain exactly one @DATA marker")
    attributes = [
        line.split()[1]
        for line in lines[: data_markers[0]]
        if line.lower().startswith("@attribute ")
    ]
    if tuple(attributes) != DRY_BEAN_ARFF_ATTRIBUTES:
        raise ValueError(
            "Dry Bean ARFF attributes or their order differ from the expected schema"
        )
    data_lines = [
        line
        for line in lines[data_markers[0] + 1 :]
        if line and not line.startswith("%")
    ]
    if len(data_lines) != expected_n_samples:
        raise ValueError(
            "Dry Bean ARFF must contain "
            f"{expected_n_samples} rows, observed {len(data_lines)}"
        )

    features = np.empty((len(data_lines), len(DRY_BEAN_FEATURE_NAMES)), dtype=np.float64)
    label_names: list[str] = []
    for row_index, line in enumerate(data_lines):
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(DRY_BEAN_FEATURE_NAMES) + 1:
            raise ValueError(
                f"Dry Bean row {row_index} has {len(values)} fields; expected 17"
            )
        if any(value == "?" or not value for value in values):
            raise ValueError(f"Dry Bean row {row_index} contains a missing value")
        try:
            features[row_index] = values[:-1]
        except ValueError as error:
            raise ValueError(
                f"Dry Bean row {row_index} contains a non-numeric feature"
            ) from error
        label_names.append(values[-1].upper())

    if not np.isfinite(features).all():
        raise ValueError("Dry Bean features must all be finite")
    class_to_index = {
        class_name: index for index, class_name in enumerate(DRY_BEAN_CLASS_NAMES)
    }
    unknown = set(label_names).difference(class_to_index)
    if unknown:
        raise ValueError(f"Dry Bean ARFF contains unknown classes: {sorted(unknown)}")
    if set(label_names) != set(DRY_BEAN_CLASS_NAMES):
        raise ValueError("Dry Bean ARFF does not contain all seven expected classes")
    labels = np.fromiter(
        (class_to_index[label] for label in label_names),
        dtype=np.int64,
        count=len(label_names),
    )
    return features, labels
