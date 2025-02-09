import abc
import datetime
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, TypedDict


@dataclass(frozen=True)
class BenchMeta:
    benchmark_name: str
    benchmark_subname: str


class BenchSchemaProto(Protocol):
    tags: Iterable[str]
    base: str


class BenchAccessProto(Protocol):
    meta: BenchMeta
    data_dir: Path

    @abc.abstractmethod
    @cached_property
    def schemas(self) -> Sequence[BenchSchemaProto]:
        ...

    @abc.abstractmethod
    def get_id(self, schema) -> str:
        ...


class BenchRecord(TypedDict):
    is_actual: bool
    benchmark_name: str
    benchmark_subname: str
    base: str
    local_id: str
    global_id: str
    tags: Iterable[str]
    kwargs: Mapping[str, Any]
    distributions: dict[str, str]
    data: str


class BenchOperator(Protocol):

    @abc.abstractmethod
    def write_bench_record(self, record: BenchRecord) -> None:
        ...

    @abc.abstractmethod
    def get_all_bench_results(self) -> Sequence[str]:
        ...

    @abc.abstractmethod
    def get_bench_result(self, schema: Any) -> Optional[str]:
        ...
