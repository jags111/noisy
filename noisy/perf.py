'''Utilities to measure the performance and runtime of parts of the code. Used
mostly during trainig.'''
from typing import Optional, Dict
from dataclasses import dataclass
import time

from .utils import ema


@dataclass
class PerfEntry:
    total: float = 0.
    count: int = 0
    ema: Optional[float] = None
    last: Optional[float] = None

    def update(self, time_taken: float) -> None:
        self.total += time_taken
        self.count += 1
        self.ema = ema(time_taken, self.ema)
        self.last = time_taken


_perf_data: Dict[str, PerfEntry] = dict()


def get_perf_data() -> Dict[str, PerfEntry]:
    return _perf_data


def set_perf_data(perf_data: Dict[str, PerfEntry]) -> None:
    global _perf_data
    _perf_data = perf_data


def get_totals(prefix: str = '') -> Dict[str, Optional[float]]:
    return {prefix + k: pe.total for k, pe in get_perf_data().items()}


class PerfMeasurer:
    '''A context manager that measures the execution time of the code in the
    `with` block.'''

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> None:
        self.start_time = time.perf_counter()

    def __exit__(self, *_) -> None:
        stop_time = time.perf_counter()
        time_taken = stop_time - self.start_time
        if self.name not in get_perf_data():
            get_perf_data()[self.name] = PerfEntry()
        pe = get_perf_data()[self.name]
        pe.update(time_taken)
