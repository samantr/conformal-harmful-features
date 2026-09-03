import gzip
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

COVERTYPE_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
COVERTYPE_ARCHIVE_SHA256 = (
    "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d"
)
COVERTYPE_ARCHIVE_MEMBER = "covtype.data.gz"
COVERTYPE_QUANTITATIVE_FEATURE_NAMES = (
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
)
COVERTYPE_FEATURE_NAMES = (
    *COVERTYPE_QUANTITATIVE_FEATURE_NAMES,
    *(f"Wilderness_Area_{index}" for index in range(1, 5)),
    *(f"Soil_Type_{index}" for index in range(1, 41)),
)
COVERTYPE_CLASS_NAMES = (
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
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


def load_covertype(
    archive_path: str | Path,
    *,
    source_url: str = COVERTYPE_URL,
    expected_sha256: str = COVERTYPE_ARCHIVE_SHA256,
) -> RealTabularDataset:
    """Load the official UCI Covertype data from a verified local cache."""
    archive = Path(archive_path)
    if not archive.exists():
        _download_verified_archive(
            archive, source_url, expected_sha256, artifact_name="Covertype"
        )
    observed_sha256 = _file_sha256(archive)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Covertype archive checksum mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )

    with zipfile.ZipFile(archive) as bundle:
        try:
            compressed_data = bundle.read(COVERTYPE_ARCHIVE_MEMBER)
        except KeyError as error:
            raise ValueError(
                f"Covertype archive lacks {COVERTYPE_ARCHIVE_MEMBER}"
            ) from error
    try:
        csv_bytes = gzip.decompress(compressed_data)
    except (EOFError, gzip.BadGzipFile) as error:
        raise ValueError("Covertype data member is not valid gzip data") from error
    features, labels = _parse_covertype_csv(csv_bytes)
    manifest = tuple(
        TabularFeature(index=index, name=name, role="quantitative")
        for index, name in enumerate(COVERTYPE_QUANTITATIVE_FEATURE_NAMES)
    ) + tuple(
        TabularFeature(
            index=index,
            name=name,
            role="one_hot",
            source_feature=(
                "Wilderness_Area" if index < 14 else "Soil_Type"
            ),
        )
        for index, name in enumerate(
            COVERTYPE_FEATURE_NAMES[len(COVERTYPE_QUANTITATIVE_FEATURE_NAMES) :],
            start=len(COVERTYPE_QUANTITATIVE_FEATURE_NAMES),
        )
    )
    return RealTabularDataset(
        name="covertype",
        features=features,
        labels=labels,
        feature_manifest=manifest,
        class_names=COVERTYPE_CLASS_NAMES,
        source_url=source_url,
        archive_sha256=observed_sha256,
    )


def _download_verified_archive(
    destination: Path,
    source_url: str,
    expected_sha256: str,
    *,
    artifact_name: str = "Dry Bean",
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(source_url, timeout=60) as response:
            payload = response.read()
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"Downloaded {artifact_name} archive checksum mismatch: "
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


def _parse_covertype_csv(
    csv_bytes: bytes, *, expected_n_samples: int = 581_012
) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw = np.loadtxt(io.BytesIO(csv_bytes), delimiter=",", dtype=np.float64)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("Covertype data must be a complete numeric CSV") from error
    expected_columns = len(COVERTYPE_FEATURE_NAMES) + 1
    if raw.shape != (expected_n_samples, expected_columns):
        raise ValueError(
            "Covertype data shape differs from the official schema: "
            f"expected {(expected_n_samples, expected_columns)}, observed {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError("Covertype data must contain only finite values")
    if not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Covertype source columns must all contain integer values")

    binary = raw[:, 10:54]
    if not np.isin(binary, (0.0, 1.0)).all():
        raise ValueError("Covertype wilderness and soil columns must be binary")
    if not np.equal(binary[:, :4].sum(axis=1), 1.0).all():
        raise ValueError(
            "Each Covertype row must select exactly one wilderness area"
        )
    if not np.equal(binary[:, 4:].sum(axis=1), 1.0).all():
        raise ValueError("Each Covertype row must select exactly one soil type")

    source_labels = raw[:, -1].astype(np.int64)
    expected_labels = np.arange(1, len(COVERTYPE_CLASS_NAMES) + 1)
    if not np.array_equal(np.unique(source_labels), expected_labels):
        raise ValueError("Covertype data must contain all seven labels 1 through 7")
    features = np.ascontiguousarray(raw[:, :-1], dtype=np.float64)
    return features, source_labels - 1
