import numpy as np
import pytest

from chf.data.real import (
    COVERTYPE_CLASS_NAMES,
    COVERTYPE_FEATURE_NAMES,
    DRY_BEAN_ARFF_ATTRIBUTES,
    DRY_BEAN_CLASS_NAMES,
    DRY_BEAN_FEATURE_NAMES,
    _parse_covertype_csv,
    _parse_dry_bean_arff,
)
from chf.models import fit_classifier


def _small_valid_arff() -> bytes:
    header = ["@RELATION Dry_Beans_Dataset"]
    header.extend(
        f"@ATTRIBUTE {name} REAL"
        for name in DRY_BEAN_ARFF_ATTRIBUTES[:-1]
    )
    header.append(
        "@ATTRIBUTE Class {SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON}"
    )
    rows = [
        ",".join([str(index + offset / 10) for index in range(16)] + [label])
        for offset, label in enumerate(DRY_BEAN_CLASS_NAMES)
    ]
    return "\n".join([*header, "@DATA", *rows]).encode("utf-8")


def _small_valid_covertype_csv() -> bytes:
    rows = []
    for row_index in range(7):
        quantitative = [100 + row_index + column for column in range(10)]
        wilderness = [int(column == row_index % 4) for column in range(4)]
        soil = [int(column == row_index % 40) for column in range(40)]
        rows.append(
            ",".join(
                str(value)
                for value in [
                    *quantitative,
                    *wilderness,
                    *soil,
                    row_index + 1,
                ]
            )
        )
    return "\n".join(rows).encode("utf-8")


def test_dry_bean_parser_preserves_schema_and_encodes_all_classes() -> None:
    features, labels = _parse_dry_bean_arff(
        _small_valid_arff(), expected_n_samples=7
    )

    assert features.shape == (7, len(DRY_BEAN_FEATURE_NAMES))
    assert features.dtype == np.float64
    np.testing.assert_array_equal(labels, np.arange(7))


def test_dry_bean_parser_rejects_changed_column_order() -> None:
    changed = _small_valid_arff().replace(
        b"@ATTRIBUTE Area REAL\n@ATTRIBUTE Perimeter REAL",
        b"@ATTRIBUTE Perimeter REAL\n@ATTRIBUTE Area REAL",
    )

    with pytest.raises(ValueError, match="attributes or their order"):
        _parse_dry_bean_arff(changed, expected_n_samples=7)


def test_classifier_standardizer_uses_training_partition_only() -> None:
    labels = np.repeat(np.arange(3), 4)
    training = np.arange(24, dtype=float).reshape(12, 2)
    held_out = training + 10_000
    fitted = fit_classifier(
        training,
        labels,
        {"type": "logistic_regression", "max_iter": 200},
        seed=42,
    )

    np.testing.assert_allclose(fitted.scaler.mean_, training.mean(axis=0))
    assert not np.allclose(
        fitted.scaler.mean_, np.vstack((training, held_out)).mean(axis=0)
    )


def test_covertype_parser_preserves_schema_and_encodes_all_classes() -> None:
    features, labels = _parse_covertype_csv(
        _small_valid_covertype_csv(), expected_n_samples=7
    )

    assert features.shape == (7, len(COVERTYPE_FEATURE_NAMES))
    assert features.dtype == np.float64
    assert len(COVERTYPE_CLASS_NAMES) == 7
    np.testing.assert_array_equal(labels, np.arange(7))


def test_covertype_parser_rejects_invalid_one_hot_groups() -> None:
    lines = _small_valid_covertype_csv().decode("utf-8").splitlines()
    first = lines[0].split(",")
    first[10] = "1"
    first[11] = "1"
    lines[0] = ",".join(first)

    with pytest.raises(ValueError, match="exactly one wilderness area"):
        _parse_covertype_csv(
            "\n".join(lines).encode("utf-8"), expected_n_samples=7
        )
