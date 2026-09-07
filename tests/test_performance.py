from unittest.mock import MagicMock

import pytest

from src.performance import (
    BenchmarkResult,
    format_benchmark_result,
    measure_operation,
)


def test_benchmark_statistics() -> None:
    result = BenchmarkResult(
        name="Test",
        durations=(1.0, 2.0, 3.0),
    )

    assert result.average_duration == 2.0
    assert result.minimum_duration == 1.0
    assert result.maximum_duration == 3.0


def test_empty_durations_raise_error() -> None:
    with pytest.raises(
        ValueError,
        match="mindestens",
    ):
        BenchmarkResult(
            name="Test",
            durations=(),
        )


def test_negative_duration_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="nicht negativ",
    ):
        BenchmarkResult(
            name="Test",
            durations=(-1.0,),
        )


def test_invalid_repetitions_raise_error() -> None:
    operation = MagicMock()

    with pytest.raises(
        ValueError,
        match="größer als 0",
    ):
        measure_operation(
            name="Test",
            operation=operation,
            repetitions=0,
        )

    operation.assert_not_called()


def test_operation_is_called_repeatedly() -> None:
    operation = MagicMock()

    result = measure_operation(
        name="Testoperation",
        operation=operation,
        repetitions=3,
    )

    assert operation.call_count == 3
    assert len(result.durations) == 3
    assert all(
        duration >= 0
        for duration in result.durations
    )


def test_format_benchmark_result() -> None:
    result = BenchmarkResult(
        name="RAG-Test",
        durations=(1.0, 2.0, 3.0),
    )

    formatted = format_benchmark_result(
        result
    )

    assert "RAG-Test" in formatted
    assert "Durchschnitt" in formatted
    assert "2.00 Sekunden" in formatted