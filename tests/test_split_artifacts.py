import numpy as np

from chf.data import load_split_artifact, make_four_way_split, save_split_artifact


def test_split_indices_seed_and_metadata_round_trip(tmp_path) -> None:
    labels = np.repeat(np.arange(4), 25)
    split = make_four_way_split(labels, (40, 20, 20, 20), seed=91)
    path = save_split_artifact(
        tmp_path / "split_indices.npz",
        split,
        seed=91,
        metadata={"dataset": "hand-check"},
    )

    restored = load_split_artifact(path)

    assert restored.seed == 91
    assert restored.metadata == {"dataset": "hand-check"}
    for name in ("train", "tune", "calibration", "test"):
        np.testing.assert_array_equal(getattr(restored.split, name), getattr(split, name))
