import json

import pandas as pd
import pytest

from chf.experiments.checkpoints import CheckpointStore, canonical_sha256


def test_checkpoint_round_trip_is_bound_to_manifest_and_logical_key(tmp_path) -> None:
    manifest = {"dataset": "har", "seed": 45, "grid": {"alpha": [0.1, 0.05]}}
    store = CheckpointStore(tmp_path / "checkpoints", manifest)
    store.initialize(resume=False)
    key = {"model": "mlp", "selected_indices": [0, 2]}
    frame = pd.DataFrame({"mean_size": [1.2], "coverage": [0.9]})

    shard = store.save_frame(key, frame)
    restored = store.load_frame(key)

    assert shard.exists()
    assert restored is not None
    pd.testing.assert_frame_equal(restored, frame)
    saved_manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert canonical_sha256(saved_manifest) == store.manifest_hash


def test_checkpoint_resume_rejects_manifest_drift(tmp_path) -> None:
    root = tmp_path / "checkpoints"
    CheckpointStore(root, {"seed": 1}).initialize(resume=False)

    CheckpointStore(root, {"seed": 1}).initialize(resume=True)
    with pytest.raises(ValueError, match="manifest differs"):
        CheckpointStore(root, {"seed": 2}).initialize(resume=True)
    with pytest.raises(FileExistsError, match="resume=True"):
        CheckpointStore(root, {"seed": 1}).initialize(resume=False)


def test_checkpoint_resume_rejects_orphaned_files(tmp_path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "partial.tmp").write_text("incomplete", encoding="utf-8")

    with pytest.raises(ValueError, match="without a manifest"):
        CheckpointStore(root, {"seed": 1}).initialize(resume=True)

