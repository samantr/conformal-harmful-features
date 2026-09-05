"""Content-validated, atomic checkpoints for long-running experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible protocol state using a stable representation."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash scientific configuration while excluding runtime replay switches."""
    normalized = json.loads(json.dumps(dict(config), default=str))
    checkpointing = normalized.get("checkpointing")
    if isinstance(checkpointing, dict):
        checkpointing.pop("resume", None)
    return canonical_sha256(normalized)


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace ``path`` only after a complete JSON payload reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically replace a CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    os.close(descriptor)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class CheckpointStore:
    """A manifest-bound collection of individually replayable result shards."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.manifest = json.loads(json.dumps(dict(manifest), default=str))
        self.manifest_hash = canonical_sha256(self.manifest)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def initialize(self, *, resume: bool) -> None:
        """Create the manifest or reject an unsafe/mismatched resume."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            saved = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if saved != self.manifest:
                raise ValueError(
                    "checkpoint manifest differs from the requested run; "
                    "use a new output directory"
                )
            if not resume:
                raise FileExistsError(
                    "checkpoint directory already exists; pass resume=True or "
                    "use a new output directory"
                )
            return
        if resume and any(self.root.iterdir()):
            raise ValueError(
                "cannot resume a non-empty checkpoint directory without a manifest"
            )
        atomic_write_json(self.manifest_path, self.manifest)

    def _shard_path(self, logical_key: Mapping[str, Any]) -> Path:
        digest = canonical_sha256(logical_key)
        return self.root / "shards" / f"{digest}.json"

    def has(self, logical_key: Mapping[str, Any]) -> bool:
        return self._shard_path(logical_key).exists()

    def save_frame(
        self, logical_key: Mapping[str, Any], frame: pd.DataFrame
    ) -> Path:
        path = self._shard_path(logical_key)
        payload = {
            "manifest_sha256": self.manifest_hash,
            "logical_key": dict(logical_key),
            "columns": list(frame.columns),
            "records": frame.to_dict(orient="records"),
        }
        atomic_write_json(path, payload)
        return path

    def load_frame(self, logical_key: Mapping[str, Any]) -> pd.DataFrame | None:
        path = self._shard_path(logical_key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("manifest_sha256") != self.manifest_hash:
            raise ValueError("checkpoint shard belongs to a different manifest")
        if payload.get("logical_key") != dict(logical_key):
            raise ValueError("checkpoint shard key does not match its filename")
        return pd.DataFrame(payload["records"], columns=payload["columns"])
