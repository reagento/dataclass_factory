import abc
import datetime
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Protocol, TypedDict


@dataclass(frozen=True)
class BenchMeta:
    benchmark_name: str
    benchmark_subname: str


class BenchSchemaProto(Protocol):
    tags: Iterable[str]
    base: str


class BenchAccessProto(Protocol):
    meta: BenchMeta

    @abc.abstractmethod
    @cached_property
    def schemas(self) -> Sequence[BenchSchemaProto]:
        raise NotImplementedError

    @abc.abstractmethod
    def bench_result_file(self, bench_id: str) -> Path:
        raise NotImplementedError

    @abc.abstractmethod
    def get_id(self, schema) -> str:
        raise NotImplementedError


class BenchRecord(TypedDict):
    is_actual: bool
    benchmark_name: str
    benchmark_subname: str
    base: str
    local_id: str
    global_id: str
    tags: str
    kwargs: str
    distributions: str
    data: str
    created_at: datetime.datetime


class BenchOperator(Protocol):

    @abc.abstractmethod
    def write_bench_result(self, record: BenchRecord) -> None:
        return

    @abc.abstractmethod
    def get_all_bench_results(self) -> Sequence[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_bench_result(self, schema: Any) -> Optional[str]:
        return None
