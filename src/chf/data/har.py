import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .real import TabularFeature


HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
HAR_N_SAMPLES = 10_299
HAR_N_FEATURES = 561
HAR_N_CLASSES = 6
HAR_N_SUBJECTS = 30
HAR_CLASS_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)


@dataclass(frozen=True)
class HumanActivityRecognitionDataset:
    name: str
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    feature_manifest: tuple[TabularFeature, ...]
    class_names: tuple[str, ...]
    source_url: str
    archive_sha256: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.feature_manifest)


def load_human_activity_recognition(
    archive_path: str | Path,
    *,
    source_url: str = HAR_URL,
    expected_sha256: str | None = None,
) -> HumanActivityRecognitionDataset:
    """Load the official UCI HAR release and preserve subject identifiers.

    The current UCI download is an outer ZIP containing ``UCI HAR Dataset.zip``;
    older mirrors expose that inner archive directly. Both layouts are accepted.
    When ``expected_sha256`` is supplied, the outer/local archive is pinned before
    parsing. The observed SHA-256 is always recorded for experiment provenance.
    """
    archive = Path(archive_path)
    if not archive.exists():
        _download_archive(archive, source_url)
    observed_sha256 = _file_sha256(archive)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            "UCI HAR archive checksum mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )

    with zipfile.ZipFile(archive) as outer:
        inner_bytes = _locate_inner_archive(outer)
        if inner_bytes is None:
            features, labels, groups, names = _parse_har_bundle(outer)
        else:
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                features, labels, groups, names = _parse_har_bundle(inner)

    manifest = tuple(
        TabularFeature(index=index, name=f"{index + 1:03d}:{name}")
        for index, name in enumerate(names)
    )
    return HumanActivityRecognitionDataset(
        name="human_activity_recognition",
        features=features,
        labels=labels,
        groups=groups,
        feature_manifest=manifest,
        class_names=HAR_CLASS_NAMES,
        source_url=source_url,
        archive_sha256=observed_sha256,
    )


def _download_archive(destination: Path, source_url: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(source_url, timeout=120) as response:
            temporary.write_bytes(response.read())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locate_inner_archive(bundle: zipfile.ZipFile) -> bytes | None:
    candidates = [
        name
        for name in bundle.namelist()
        if Path(name).name.lower() == "uci har dataset.zip"
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("UCI HAR outer archive contains multiple inner dataset ZIPs")
    return bundle.read(candidates[0])


def _member_name(bundle: zipfile.ZipFile, suffix: str) -> str:
    normalized_suffix = suffix.replace("\\", "/").lower()
    matches = [
        name
        for name in bundle.namelist()
        if name.replace("\\", "/").lower().endswith(normalized_suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"UCI HAR archive must contain exactly one member ending in {suffix!r}"
        )
    return matches[0]


def _read_numeric_matrix(bundle: zipfile.ZipFile, suffix: str) -> np.ndarray:
    member = _member_name(bundle, suffix)
    try:
        matrix = np.loadtxt(io.BytesIO(bundle.read(member)), dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"UCI HAR member {suffix} is not a numeric matrix") from error
    if not np.isfinite(matrix).all():
        raise ValueError(f"UCI HAR member {suffix} contains non-finite values")
    return matrix


def _parse_feature_names(bundle: zipfile.ZipFile) -> tuple[str, ...]:
    member = _member_name(bundle, "UCI HAR Dataset/features.txt")
    try:
        lines = bundle.read(member).decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("UCI HAR features.txt is not valid UTF-8") from error
    parsed: list[str] = []
    for expected_index, line in enumerate(lines, start=1):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or int(parts[0]) != expected_index:
            raise ValueError("UCI HAR features.txt has an unexpected index schema")
        parsed.append(parts[1])
    if len(parsed) != HAR_N_FEATURES:
        raise ValueError(
            f"UCI HAR must define {HAR_N_FEATURES} features; observed {len(parsed)}"
        )
    return tuple(parsed)


def _parse_activity_labels(bundle: zipfile.ZipFile) -> None:
    member = _member_name(bundle, "UCI HAR Dataset/activity_labels.txt")
    try:
        lines = bundle.read(member).decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("UCI HAR activity_labels.txt is not valid UTF-8") from error
    observed = []
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("UCI HAR activity_labels.txt has an unexpected schema")
        observed.append((int(parts[0]), parts[1]))
    expected = list(enumerate(HAR_CLASS_NAMES, start=1))
    if observed != expected:
        raise ValueError("UCI HAR activity labels differ from the official six-class schema")


def _parse_har_bundle(
    bundle: zipfile.ZipFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    names = _parse_feature_names(bundle)
    _parse_activity_labels(bundle)

    blocks = []
    for partition in ("train", "test"):
        X = _read_numeric_matrix(
            bundle, f"UCI HAR Dataset/{partition}/X_{partition}.txt"
        )
        y = _read_numeric_matrix(
            bundle, f"UCI HAR Dataset/{partition}/y_{partition}.txt"
        ).astype(np.int64)
        subjects = _read_numeric_matrix(
            bundle, f"UCI HAR Dataset/{partition}/subject_{partition}.txt"
        ).astype(np.int64)
        y = np.asarray(y).reshape(-1)
        subjects = np.asarray(subjects).reshape(-1)
        if X.ndim != 2 or X.shape[1] != HAR_N_FEATURES:
            raise ValueError(
                f"UCI HAR {partition} matrix must have {HAR_N_FEATURES} columns"
            )
        if not (len(X) == len(y) == len(subjects)):
            raise ValueError(f"UCI HAR {partition} X/y/subject row counts differ")
        blocks.append((X, y, subjects))

    features = np.ascontiguousarray(
        np.vstack([block[0] for block in blocks]), dtype=np.float64
    )
    source_labels = np.concatenate([block[1] for block in blocks])
    groups = np.concatenate([block[2] for block in blocks])
    if features.shape != (HAR_N_SAMPLES, HAR_N_FEATURES):
        raise ValueError(
            "UCI HAR data shape differs from the official release: "
            f"expected {(HAR_N_SAMPLES, HAR_N_FEATURES)}, observed {features.shape}"
        )
    if not np.array_equal(np.unique(source_labels), np.arange(1, HAR_N_CLASSES + 1)):
        raise ValueError("UCI HAR must contain all activity labels 1 through 6")
    if not np.array_equal(np.unique(groups), np.arange(1, HAR_N_SUBJECTS + 1)):
        raise ValueError("UCI HAR must contain subject identifiers 1 through 30")
    return features, source_labels - 1, groups, names
