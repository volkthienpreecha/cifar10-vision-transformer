import pytest

from cifar10_vit.history import EpochMetrics, parse_training_history


def test_parser_extracts_complete_metric_line() -> None:
    line = "[30] train loss: 0.779 | train accuracy: 0.724 | val loss: 1.023 | val accuracy: 0.660"
    assert parse_training_history(line) == [
        EpochMetrics(30, 0.779, 0.724, 1.023, 0.660)
    ]


def test_parser_ignores_unrelated_output_and_preserves_order() -> None:
    text = (
        "starting run\n"
        "[ 1] train loss: 1.845 | train accuracy: 0.302 | val loss: 1.752 | val accuracy: 0.364\n"
        "progress\n"
        "[ 2] train loss: 1.584 | train accuracy: 0.413 | val loss: 1.523 | val accuracy: 0.453"
    )
    assert [row.epoch for row in parse_training_history(text)] == [1, 2]


def test_parser_rejects_partial_metric_line() -> None:
    with pytest.raises(ValueError, match="partial"):
        parse_training_history("[ 1] train loss: 1.845 | train accuracy: 0.302")


def test_parser_rejects_duplicate_epoch() -> None:
    line = "[ 1] train loss: 1.0 | train accuracy: 0.2 | val loss: 1.1 | val accuracy: 0.1"
    with pytest.raises(ValueError, match="duplicate"):
        parse_training_history(f"{line}\n{line}")
