from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    """Zeitmessungen einer Operation."""

    name: str
    durations: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Der Benchmark benötigt einen Namen."
            )

        if not self.durations:
            raise ValueError(
                "Der Benchmark benötigt mindestens "
                "eine Zeitmessung."
            )

        if any(
            duration < 0
            for duration in self.durations
        ):
            raise ValueError(
                "Zeitmessungen dürfen nicht negativ sein."
            )

    @property
    def average_duration(self) -> float:
        """Berechnet die durchschnittliche Dauer."""

        return mean(self.durations)

    @property
    def minimum_duration(self) -> float:
        """Gibt die schnellste Messung zurück."""

        return min(self.durations)

    @property
    def maximum_duration(self) -> float:
        """Gibt die langsamste Messung zurück."""

        return max(self.durations)


def measure_operation(
    name: str,
    operation: Callable[[], Any],
    repetitions: int = 3,
) -> BenchmarkResult:
    """Misst die Laufzeit einer Funktion mehrfach."""

    if repetitions <= 0:
        raise ValueError(
            "Die Anzahl Wiederholungen muss "
            "größer als 0 sein."
        )

    durations: list[float] = []

    for _ in range(repetitions):
        start_time = perf_counter()
        operation()
        duration = perf_counter() - start_time
        durations.append(duration)

    return BenchmarkResult(
        name=name,
        durations=tuple(durations),
    )


def format_benchmark_result(
    result: BenchmarkResult,
) -> str:
    """Formatiert ein Benchmark-Ergebnis."""

    return (
        f"{result.name}\n"
        f"  Durchschnitt: "
        f"{result.average_duration:.2f} Sekunden\n"
        f"  Schnellste Messung: "
        f"{result.minimum_duration:.2f} Sekunden\n"
        f"  Langsamste Messung: "
        f"{result.maximum_duration:.2f} Sekunden"
    )