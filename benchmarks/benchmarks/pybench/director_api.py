# pylint: disable=import-error,no-name-in-module
# ruff: noqa: T201, S603
import datetime
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional, TypeVar, Union

import pyperf
from pyperf._cli import format_checks

from benchmarks.pybench.persistence.common import BenchAccessProto, BenchMeta, BenchOperator, BenchSchemaProto
from benchmarks.pybench.persistence.database import BenchRecord, sqlite_operator_factory
from benchmarks.pybench.persistence.filesystem import filesystem_operator_factory
from benchmarks.pybench.utils import get_function_object_ref, load_by_object_ref

__all__ = (
    "BenchAccessor",
    "BenchChecker",
    "BenchSchema",
    "BenchmarkDirector",
    "CheckParams",
    "PlotParams",
)

EnvSpec = Mapping[str, str]


def operator_factory(accessor: BenchAccessProto, *, sqlite: bool) -> BenchOperator:
    if sqlite:
        return sqlite_operator_factory(accessor)
    return filesystem_operator_factory(accessor)


@dataclass(frozen=True)
class CheckParams:
    stdev_rel_threshold: Optional[float] = None
    ignore_pyperf_warnings: Optional[bool] = None


@dataclass(frozen=True)
class BenchSchema(BenchSchemaProto):
    entry_point: Union[Callable, str]
    base: str
    tags: Iterable[str]
    kwargs: Mapping[str, Any]
    used_distributions: Sequence[str]
    skip_if: Optional[Callable[[EnvSpec], bool]] = None
    check_params: Callable[[EnvSpec], CheckParams] = lambda env_spec: CheckParams()


@dataclass(frozen=True)
class PlotParams:
    title: str
    fig_size: tuple[float, float] = (8, 4.8)
    label_padding: float = 0
    trim_after: Optional[float] = None
    label_format: str = ".1f"


BUILTIN_CHECK_PARAMS = CheckParams(
    ignore_pyperf_warnings=False,
)


class BenchAccessor(BenchAccessProto):
    def __init__(
        self,
        data_dir: Path,
        env_spec: EnvSpec,
        check_params: Callable[[EnvSpec], CheckParams],
        schemas: Sequence[BenchSchema],
        meta: BenchMeta,
    ):
        self.meta = meta
        self.data_dir = data_dir
        self.env_spec = env_spec
        self.all_schemas = schemas
        self.id_to_schema: dict[str, BenchSchema] = {self.get_id(schema): schema for schema in schemas}
        self._base_check_params = check_params

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--data-dir", action="store", required=False, type=Path)
        parser.add_argument("--env-spec", action="extend", nargs="+", required=False, metavar="KEY=VALUE")

    def override_state(self, env_spec: Optional[Iterable[str]], data_dir: Optional[Path] = None):
        if data_dir is not None:
            self.data_dir = data_dir

        if env_spec is not None:
            update_data = {}
            for item in env_spec:
                key, value = item.split("=")
                if key not in self.env_spec:
                    raise KeyError(f"Unexpected key {key}")
                update_data[key] = value
            self.env_spec = {**self.env_spec, **update_data}

    def apply_state(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def env_spec_str(self) -> str:
        return "[" + "-".join(f"{k}={v}" for k, v in self.env_spec.items()) + "]"

    def get_id(self, schema: BenchSchema) -> str:
        return self.get_local_id(schema) + self.env_spec_str()

    def get_local_id(self, schema: BenchSchema) -> str:
        if schema.tags:
            tags_str = "-" + "-".join(schema.tags)
            return schema.base + tags_str
        return schema.base

    def get_label(self, schema: BenchSchema) -> str:
        if schema.tags:
            tags_str = ", ".join(schema.tags)
            return f"{schema.base}\n({tags_str})"
        return schema.base

    def _chain_check_param(self, base_check_params: CheckParams, key: str, value: Any) -> Any:
        if value is not None:
            return value
        base_value = getattr(base_check_params, key)
        if base_value is not None:
            return base_value
        builtin_value = getattr(BUILTIN_CHECK_PARAMS, key)
        if builtin_value is None:
            raise ValueError(f"Check param {key!r} must be filled")
        return builtin_value

    def resolve_check_params(self, schema: BenchSchema) -> CheckParams:
        base_check_params = self._base_check_params(self.env_spec)
        return CheckParams(
            **{
                key: self._chain_check_param(base_check_params, key, value)
                for key, value in vars(schema.check_params(self.env_spec)).items()
            },
        )

    @cached_property
    def schemas(self) -> Sequence[BenchSchema]:
        return [
            schema
            for schema in self.all_schemas
            if schema.skip_if is None or not schema.skip_if(self.env_spec)
        ]


class BenchChecker:
    def __init__(self, accessor: BenchAccessor):
        self.accessor = accessor

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--local-id-list", action="store_true", required=False, default=False,
            help="print only schema list with errors",
        )

    def _process_pyperf_warnings(
        self,
        schema: BenchSchema,
        bench: pyperf.Benchmark,
        check_params: CheckParams,
        warnings: Iterable[str],
    ) -> Iterable[str]:
        if check_params.ignore_pyperf_warnings:
            return []
        return [
            line
            for line in warnings
            if not line.startswith("Use")
        ]

    def get_warnings(self, schema: BenchSchema, bench_operator: BenchOperator) -> Optional[Sequence[str]]:
        data = bench_operator.get_bench_result(schema)
        if data is None:
            return None
        bench = pyperf.Benchmark.loads(data)
        check_params = self.accessor.resolve_check_params(schema)
        warnings = self._process_pyperf_warnings(schema, bench, check_params, format_checks(bench))
        self_warnings = self._check_yourself(schema, bench, check_params)
        return [*warnings, *self_warnings]

    def _check_yourself(self, schema: BenchSchema, bench: pyperf.Benchmark, check_params: CheckParams) -> Sequence[str]:
        lines: list[str] = []
        stdev = bench.stdev()
        mean = bench.mean()
        rate = stdev / mean
        if rate >= check_params.stdev_rel_threshold:
            lines.append(
                f"the relative standard deviation is {rate:.1%},"
                f" max allowed is {check_params.stdev_rel_threshold:.0%}",
            )
        return lines

    def check_results(self, *, local_id_list: bool = False, sqlite: bool = False):
        lines = []
        schemas_with_warnings = []
        reader = operator_factory(self.accessor, sqlite=sqlite)
        for schema in self.accessor.schemas:
            warnings = self.get_warnings(schema, reader)
            if warnings is None:
                lines.append(f"Result file of {self.accessor.get_id(schema)!r}")
                lines.append("")
                schemas_with_warnings.append(schema)
            elif warnings:
                lines.append(self.accessor.get_id(schema))
                lines.extend(warnings)
                schemas_with_warnings.append(schema)

        if local_id_list:
            print(json.dumps([self.accessor.get_local_id(schema) for schema in schemas_with_warnings]))
        else:
            print("\n".join(lines))


class BenchRunner:
    def __init__(self, accessor: BenchAccessor, checker: BenchChecker, meta: BenchMeta):
        self.meta = meta
        self.accessor = accessor
        self.checker = checker

    def add_arguments(self, parser: ArgumentParser) -> None:
        selective_group = parser.add_mutually_exclusive_group()
        selective_group.add_argument("--include", "-i", action="extend", nargs="+", required=False)
        selective_group.add_argument("--exclude", "-e", action="extend", nargs="+", required=False)

        scope_group = parser.add_mutually_exclusive_group()
        scope_group.add_argument(
            "--missing", action="store_true", required=False, default=False,
            help="run only missing benchmarks",
        )
        scope_group.add_argument(
            "--unstable", action="store_true", required=False, default=False,
            help="run only unstable or missing benchmarks",
        )
        parser.add_argument("--sqlite", action="store_true", required=False, default=False)

    def run_benchmarks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        missing: bool = False,
        unstable: bool = False,
        sqlite: bool = False,
    ) -> None:
        operator = operator_factory(self.accessor, sqlite=sqlite)
        schemas: Sequence[BenchSchema]
        if missing:
            schemas = [
                schema for schema in self.accessor.schemas
                if not operator.get_bench_result(schema)
            ]
        elif unstable:
            schemas = [
                schema for schema, warnings in (
                    (schema, self.checker.get_warnings(schema, operator))
                    for schema in self.accessor.schemas
                )
                if warnings is None or warnings
            ]
        else:
            schemas = self.accessor.schemas

        local_id_to_schema = {
            self.accessor.get_local_id(schema): schema
            for schema in schemas
        }

        benchmarks_to_run: list[str]
        if exclude is not None:
            benchmarks_to_run = [
                self.accessor.get_local_id(schema)
                for schema in schemas
                if self.accessor.get_local_id(schema) not in set(exclude)
            ]
        elif include is not None:
            wild_local_ids = set(include) - local_id_to_schema.keys()
            if wild_local_ids:
                raise ValueError(f"Unknown local ids {wild_local_ids}")
            benchmarks_to_run = list(include)
        else:
            benchmarks_to_run = [self.accessor.get_local_id(schema) for schema in schemas]

        print("Benchmarks to run: " + " ".join(benchmarks_to_run))
        for tag in benchmarks_to_run:
            self.run_one_benchmark(local_id_to_schema[tag], operator)

    def run_one_benchmark(self, schema: BenchSchema, bench_operator: BenchOperator) -> None:
        distributions = {
            dist: importlib.metadata.version(dist)
            for dist in schema.used_distributions
        }
        bench_id = self.accessor.get_id(schema)
        sig = inspect.signature(
            load_by_object_ref(schema.entry_point)
            if isinstance(schema.entry_point, str) else
            schema.entry_point,
        )
        with TemporaryDirectory() as dir_name:
            temp_file = Path(dir_name) / f"{bench_id}.json"
            print(f"start: {bench_id}")
            self.launch_benchmark(
                bench_name=bench_id,
                entrypoint=(
                    schema.entry_point
                    if isinstance(schema.entry_point, str) else
                    get_function_object_ref(schema.entry_point)
                ),
                params=[schema.kwargs[param] for param in sig.parameters],
                extra_args=["-o", str(temp_file)],
            )
            result_data = json.loads(temp_file.read_text())
            result_data["pybench_data"] = {
                "base": schema.base,
                "tags": schema.tags,
                "kwargs": schema.kwargs,
                "distributions": distributions,
            }
            bench_data_text = json.dumps(
                    result_data,
                    ensure_ascii=False,
                    check_circular=False,
            )
            bench = pyperf.Benchmark.loads(bench_data_text)
            check_params = self.accessor.resolve_check_params(schema)
            rel_stddev = bench.stdev() / bench.mean()
            print(f"Relative stdev is {rel_stddev:.1%} (max allowed is {check_params.stdev_rel_threshold:.1%})")
            print()
            bench_data: BenchRecord = {
                "base": schema.base,
                "kwargs": schema.kwargs,
                "distributions": distributions,
                "is_actual": True,
                "tags": schema.tags,
                "data": bench_data_text,
                "local_id": self.accessor.get_local_id(schema),
                "global_id": self.accessor.get_id(schema),
                "benchmark_subname": self.meta.benchmark_subname,
                "benchmark_name": self.meta.benchmark_name,
            }
            bench_operator.write_bench_record(bench_data)

    def launch_benchmark(
        self,
        bench_name: str,
        entrypoint: str,
        params: list[Any],
        extra_args: Iterable[str] = (),
    ) -> None:
        subprocess.run(
            [
                sys.executable, "-m", "benchmarks.pybench.pyperf_runner",
                "--inherit-environ", "PYBENCH_NAME,PYBENCH_ENTRYPOINT,PYBENCH_PARAMS",
                *extra_args,
            ],
            env={
                **os.environ,
                "PYBENCH_NAME": bench_name,
                "PYBENCH_ENTRYPOINT": entrypoint,
                "PYBENCH_PARAMS": json.dumps(params),
            },
            check=True,
        )


class BenchPlotter:
    def __init__(self, params: PlotParams, accessor: BenchAccessor):
        self.params = params
        self.accessor = accessor

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--output", "-o", action="store", required=False, type=Path)
        parser.add_argument("--dpi", action="store", required=False, type=float, default=100)


    def draw_plot(self, output: Optional[Path], dpi: float, *,  sqlite: bool = False):
        operator = operator_factory(self.accessor, sqlite=sqlite)
        if output is None:
            output = self.accessor.data_dir / f"plot{self.accessor.env_spec_str()}.png"
        benches_data = operator.get_all_bench_results()
        self._render_plot(
            output=output,
            dpi=dpi,
            benchmarks=[pyperf.Benchmark.loads(data) for data in benches_data],
        )

    def _render_plot(self, output: Path, dpi: float, benchmarks: Iterable[pyperf.Benchmark]) -> None:
        # pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt

        from benchmarks.pybench.matplotlib_utils import bar_left_aligned_label

        benchmarks = sorted(benchmarks, key=lambda b: b.mean())
        _, ax = plt.subplots(figsize=self.params.fig_size)
        x_pos = range(len(benchmarks))
        means = [bench.mean() * 10 ** 6 for bench in benchmarks]
        errors = [bench.stdev() * 10 ** 6 for bench in benchmarks]
        hbars = ax.barh(
            x_pos,
            means,
            xerr=errors,
            align="center",
            alpha=1,
            color="orange",
            ecolor="black",
            capsize=5,
            edgecolor="black",
        )

        if self.params.trim_after is not None:
            upper_bound = next(mean * 1.08 for mean in reversed(means) if mean <= self.params.trim_after)
            ax.set_xbound(upper=upper_bound)

        bar_left_aligned_label(
            ax,
            hbars,
            padding=self.params.label_padding,
            fontsize=9,
            labels=[format(mean, self.params.label_format) for mean in means],
        )
        ax.set_xlabel("Time (μs)")
        ax.set_yticks(x_pos)
        ax.tick_params(bottom=False, left=False)
        ax.set_yticklabels(
            [
                self.accessor.get_label(self.accessor.id_to_schema[bench.get_name()])
                for bench in benchmarks
            ],
        )
        ax.set_title(self.params.title)
        ax.xaxis.grid(visible=True)
        plt.tight_layout(w_pad=1000)
        ax.set_axisbelow(True)
        plt.savefig(output, dpi=dpi)


T = TypeVar("T")


def call_by_namespace(func: Callable[..., T], namespace: Namespace) -> T:
    sig = inspect.signature(func)
    kwargs_for_func = (vars(namespace).keys() & sig.parameters.keys())
    return func(**{key: getattr(namespace, key) for key in kwargs_for_func})


BD = TypeVar("BD", bound="BenchmarkDirector")


class BenchmarkDirector:
    def __init__(
        self,
        *,
        data_dir: Path,
        plot_params: PlotParams,
        env_spec: EnvSpec,
        check_params: Callable[[EnvSpec], CheckParams],
        schemas: Iterable[BenchSchema] = (),
        meta: BenchMeta,
    ):
        self.meta = meta
        self.data_dir = data_dir
        self.env_spec = env_spec
        self.plot_params = plot_params
        self.schemas: list[BenchSchema] = list(schemas)
        self.check_params = check_params

    def add(self, *schemas: BenchSchema) -> None:
        self.schemas.extend(schemas)

    def add_iter(self, schemas: Iterable[BenchSchema]) -> None:
        self.schemas.extend(schemas)

    def cli(self, args: Optional[Sequence[str]] = None):
        accessor = self.make_accessor()
        self._validate_schemas(accessor)

        checker = self.make_bench_checker(accessor)
        runner = self.make_bench_runner(accessor, checker)
        plotter = self.make_bench_plotter(accessor)

        parser = self._make_parser(accessor, runner, plotter, checker)
        namespace = parser.parse_args(args)

        call_by_namespace(accessor.override_state, namespace)
        accessor.apply_state()
        if namespace.command == "run":
            call_by_namespace(runner.run_benchmarks, namespace)
        elif namespace.command == "render":
            call_by_namespace(plotter.draw_plot, namespace)
        elif namespace.command == "run-render":
            call_by_namespace(runner.run_benchmarks, namespace)
            call_by_namespace(plotter.draw_plot, namespace)
        elif namespace.command == "check":
            call_by_namespace(checker.check_results, namespace)
        else:
            raise TypeError

    def _make_parser(
        self,
        accessor: BenchAccessor,
        runner: BenchRunner,
        plotter: BenchPlotter,
        checker: BenchChecker,
    ) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(
            "--sqlite",
            action="store_true",
            default=False,
            required=False,
        )

        subparsers = parser.add_subparsers(required=True)

        run_parser = subparsers.add_parser("run")
        run_parser.set_defaults(command="run")
        accessor.add_arguments(run_parser)
        runner.add_arguments(run_parser)

        render_parser = subparsers.add_parser("render")
        render_parser.set_defaults(command="render")
        accessor.add_arguments(render_parser)
        plotter.add_arguments(render_parser)

        run_render_parser = subparsers.add_parser("run-render")
        run_render_parser.set_defaults(command="run-render")
        accessor.add_arguments(run_render_parser)
        runner.add_arguments(run_render_parser)
        plotter.add_arguments(run_render_parser)

        check_parser = subparsers.add_parser("check")
        check_parser.set_defaults(command="check")
        accessor.add_arguments(check_parser)
        checker.add_arguments(check_parser)

        return parser

    def make_accessor(self) -> BenchAccessor:
        return BenchAccessor(
            data_dir=self.data_dir,
            env_spec=self.env_spec,
            check_params=self.check_params,
            schemas=self.schemas,
            meta=self.meta,
        )

    def make_bench_runner(self, accessor: BenchAccessor, checker: BenchChecker) -> BenchRunner:
        return BenchRunner(accessor, checker, self.meta)

    def make_bench_plotter(self, accessor: BenchAccessor) -> BenchPlotter:
        return BenchPlotter(self.plot_params, accessor)

    def make_bench_checker(self, accessor: BenchAccessor) -> BenchChecker:
        return BenchChecker(accessor)

    def _validate_schemas(self, accessor: BenchAccessor):
        local_id_set: set[str] = set()
        for schema in self.schemas:
            local_id = accessor.get_local_id(schema)
            if local_id in local_id_set:
                raise ValueError(f"Local id {local_id} is duplicated")
            local_id_set.add(local_id)

    def replace(self: BD, *, env_spec: EnvSpec) -> BD:
        self_copy = copy(self)
        self_copy.env_spec = env_spec
        return self_copy
