import concurrent
import importlib
import inspect
import json
import subprocess
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
    cast,
)
from zipfile import ZIP_BZIP2, ZipFile

import plotly
import plotly.graph_objects as go
import pyperf

from adaptix._internal.utils import pairs
from benchmarks.pybench.director_api import BenchAccessor, BenchChecker, BenchmarkDirector, operator_factory
from benchmarks.pybench.persistence.filesystem import FileSystemBenchOperator

T = TypeVar("T")


def call_by_namespace(func: Callable[..., T], namespace: Namespace) -> T:
    sig = inspect.signature(func)
    kwargs_for_func = (vars(namespace).keys() & sig.parameters.keys())
    return func(**{key: getattr(namespace, key) for key in kwargs_for_func})


@dataclass
class BenchmarkMeasure:
    base: str
    tags: Sequence[str]
    kwargs: Mapping[str, Any]
    distributions: Mapping[str, str]
    pyperf: pyperf.Benchmark


def pyperf_bench_to_measure(data: Union[str, bytes]) -> BenchmarkMeasure:
    pybench_data = json.loads(data)["pybench_data"]
    return BenchmarkMeasure(
        base=pybench_data["base"],
        tags=pybench_data["tags"],
        kwargs=pybench_data["kwargs"],
        distributions=pybench_data["distributions"],
        pyperf=pyperf.Benchmark.loads(data),
    )


@dataclass(frozen=True)
class EnvDescription:
    key: str
    title: str
    tox_env: str


BENCHMARK_ENVS: Iterable[EnvDescription] = [
    EnvDescription(
        title="CPython 3.8",
        key="py38",
        tox_env="py38-bench",
    ),
    EnvDescription(
        title="CPython 3.9",
        key="py39",
        tox_env="py39-bench",
    ),
    EnvDescription(
        title="CPython 3.10",
        key="py310",
        tox_env="py310-bench",
    ),
    EnvDescription(
        title="CPython 3.11",
        key="py311",
        tox_env="py311-bench",
    ),
    EnvDescription(
        title="CPython 3.12",
        key="py312",
        tox_env="py312-bench",
    ),
    EnvDescription(
        title="PyPy 3.8",
        key="pypy38",
        tox_env="pypy38-bench",
    ),
    EnvDescription(
        title="PyPy 3.9",
        key="pypy39",
        tox_env="pypy39-bench",
    ),
    EnvDescription(
        title="PyPy 3.10",
        key="pypy310",
        tox_env="pypy310-bench",
    ),
]
KEY_TO_ENV = {
    env_description.key: env_description
    for env_description in BENCHMARK_ENVS
}


class Foundation(ABC):
    def print(self, *args: str):
        print(*args)

    def run(self, command: str) -> None:
        subprocess.run(command, shell=True, check=True)  # nosec  # noqa: DUO116

    def call(self, command: str) -> str:
        proc = subprocess.run(command, shell=True, capture_output=True, check=True)  # nosec  # noqa: DUO116
        return proc.stdout.decode("utf-8")

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        ...

    @abstractmethod
    def start(self) -> None:
        ...


class AxisBounder(ABC):
    @abstractmethod
    def get_hub_x_bound(self, env_to_measures: Mapping[EnvDescription, Sequence[BenchmarkMeasure]]) -> float:
        pass


class ClusterAxisBounder(AxisBounder):
    def __init__(self, last_cluster_idx: int, boundary_rate: float):
        self.last_cluster_idx = last_cluster_idx
        self.boundary_rate = boundary_rate

    def _split_into_clusters(self, measures: Iterable[BenchmarkMeasure]) -> Sequence[Sequence[BenchmarkMeasure]]:
        clusters: List[List[BenchmarkMeasure]] = []
        current_cluster: List[BenchmarkMeasure] = []
        for prev, current in pairs(measures):
            if current.pyperf.mean() / prev.pyperf.mean() >= self.boundary_rate:
                clusters.append(current_cluster)
                current_cluster = [current]
            else:
                current_cluster.append(current)
        clusters.append(current_cluster)
        return clusters

    def get_hub_x_bound(self, env_to_measures: Mapping[EnvDescription, Sequence[BenchmarkMeasure]]) -> float:
        max_bound_value = max(
            self._split_into_clusters(measures)[self.last_cluster_idx][-1].pyperf.mean()
            for measures in env_to_measures.values()
        )
        return max_bound_value * 10 ** 6 * 1.08


@dataclass(frozen=True)
class HubDescription:
    key: str
    module: str
    title: str
    x_bounder: AxisBounder


RELEASE_DATA = Path(__file__).parent.parent / "release_data"

BENCHMARK_HUBS: Iterable[HubDescription] = [
    HubDescription(
        key="simple_structures-loading",
        title="Simple Structures (loading)",
        module="benchmarks.simple_structures.hub_loading",
        x_bounder=ClusterAxisBounder(
            last_cluster_idx=-2,
            boundary_rate=3,
        ),
    ),
    HubDescription(
        key="simple_structures-dumping",
        title="Simple Structures (dumping)",
        module="benchmarks.simple_structures.hub_dumping",
        x_bounder=ClusterAxisBounder(
            last_cluster_idx=-3,
            boundary_rate=2,
        ),
    ),
    HubDescription(
        key="gh_issues-loading",
        title="Github Issues (loading)",
        module="benchmarks.gh_issues.hub_loading",
        x_bounder=ClusterAxisBounder(
            last_cluster_idx=-2,
            boundary_rate=2,
        ),
    ),
    HubDescription(
        key="gh_issues-dumping",
        title="Github Issues (dumping)",
        module="benchmarks.gh_issues.hub_dumping",
        x_bounder=ClusterAxisBounder(
            last_cluster_idx=-3,
            boundary_rate=1.5,
        ),
    ),
]
KEY_TO_HUB = {
    hub_description.key: hub_description
    for hub_description in BENCHMARK_HUBS
}


class HubProcessor(Foundation, ABC):
    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        hub_selective_group = parser.add_mutually_exclusive_group()
        hub_selective_group.add_argument(
            "--include",
            "-i",
            action="extend",
            nargs="+",
            required=False,
        )
        hub_selective_group.add_argument(
            "--exclude",
            "-e",
            action="extend",
            nargs="+",
            required=False,
        )
        env_selective_group = parser.add_mutually_exclusive_group()
        env_selective_group.add_argument(
            "--env-include",
            "-ei",
            action="extend",
            nargs="+",
            required=False,
        )
        env_selective_group.add_argument(
            "--env-exclude",
            "-ee",
            action="extend",
            nargs="+",
            required=False,
        )
        parser.add_argument(
            "--sqlite",
            action="store_true",
            default=False,
            required=False,
        )

    def __init__(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        env_include: Optional[Sequence[str]] = None,
        env_exclude: Optional[Sequence[str]] = None,
        sqlite: Optional[bool] = None,
    ):
        self.sqlite = sqlite
        self.include = include
        self.exclude = exclude
        self.env_include = env_include
        self.env_exclude = env_exclude

    def filtered_hubs(self) -> Iterable[HubDescription]:
        if self.include:
            wild_hubs = set(self.include) - {hub_description.key for hub_description in BENCHMARK_HUBS}
            if wild_hubs:
                raise ValueError(f"Unknown hubs {wild_hubs}")
            return [
                hub_description
                for hub_description in BENCHMARK_HUBS
                if hub_description.key in self.include
            ]
        if self.exclude:
            wild_hubs = set(self.exclude) - {hub_description.key for hub_description in BENCHMARK_HUBS}
            if wild_hubs:
                raise ValueError(f"Unknown hubs {wild_hubs}")
            return [
                hub_description
                for hub_description in BENCHMARK_HUBS
                if hub_description.key not in self.exclude
            ]
        return BENCHMARK_HUBS

    def filtered_envs(self) -> Iterable[EnvDescription]:
        if self.env_include and self.env_exclude:
            wild_envs = set(self.env_exclude) - {env_description.key for env_description in BENCHMARK_ENVS}
            if wild_envs:
                raise ValueError(f"Unknown envs {wild_envs}")
            return [
                env_description
                for env_description in BENCHMARK_ENVS
                if env_description.key in self.env_include
            ]
        if self.env_exclude:
            wild_envs = set(self.env_exclude) - {env_description.key for env_description in BENCHMARK_ENVS}
            if wild_envs:
                raise ValueError(f"Unknown envs {wild_envs}")
            return [
                env_description
                for env_description in BENCHMARK_ENVS
                if env_description.key not in self.env_exclude
            ]
        return BENCHMARK_ENVS

    def _submit_python(
        self,
        executor: Executor,
        env_description: EnvDescription,
        code: str,
    ) -> concurrent.futures.Future[str]:
        return executor.submit(
            lambda: self.call(
                f"tox exec -e {env_description.tox_env} --skip-pkg-install -qq -- python -c '{code}'",
            ),
        )

    def load_directors(
        self,
        hub_descriptions: Iterable[HubDescription],
    ) -> Mapping[HubDescription, Mapping[EnvDescription, BenchmarkDirector]]:
        env_spec_printers = [
            dedent(
                f"""
                    import json
                    from {hub_description.module} import director
                    print(json.dumps(director.env_spec))
                """,
            )
            for hub_description in hub_descriptions
        ]
        with ThreadPoolExecutor() as executor:
            hub_to_env_to_future = {
                hub_description: {
                    env_description: self._submit_python(executor, env_description, env_spec_printer)
                    for env_description in self.filtered_envs()
                }
                for hub_description, env_spec_printer in zip(hub_descriptions, env_spec_printers)
            }
            return {
                hub_description: {
                    env_description: (
                        importlib.import_module(hub_description.module).director.replace(
                            env_spec=json.loads(future.result()),
                        )
                    )
                    for env_description, future in env_to_future.items()
                }
                for hub_description, env_to_future in hub_to_env_to_future.items()
            }


@dataclass
class CaseState:
    director: BenchmarkDirector
    accessor: BenchAccessor
    checker: BenchChecker
    local_ids_with_warnings: Sequence[str]
    max_tries: Optional[int]
    tries_count: int = 0

    @property
    def is_completed(self) -> bool:
        return not self.local_ids_with_warnings

    @property
    def is_out_of_tries(self) -> bool:
        return self.max_tries is not None and self.tries_count >= self.max_tries


class Orchestrator(HubProcessor):
    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--series",
            action="store",
            required=False,
            type=int,
            default=2,
        )
        parser.add_argument(
            "--max-tries",
            action="store",
            required=False,
            type=int,
            default=None,
        )

    def __init__(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        env_include: Optional[Sequence[str]] = None,
        env_exclude: Optional[Sequence[str]] = None,
        series: int = 2,
        max_tries: Optional[int] = None,
        sqlite: bool = False,
    ):
        super().__init__(
            sqlite=sqlite,
            include=include,
            exclude=exclude,
            env_include=env_include,
            env_exclude=env_exclude,
        )
        self.series = series
        self.max_tries = max_tries

    def start(self) -> None:
        self.print("Start environments preparation")
        joined_envs = ",".join(env_description.tox_env for env_description in self.filtered_envs())
        self.run(f"tox --notest -p auto -e {joined_envs}")

        self.print("Loading all cases")
        hub_to_env_to_case_state: Dict[HubDescription, Dict[EnvDescription, CaseState]] = {}
        for hub_description, env_to_director in self.load_directors(self.filtered_hubs()).items():
            env_to_case_state = {}
            for env_description, director in env_to_director.items():
                accessor = director.make_accessor()
                case_state = CaseState(
                    director=director,
                    accessor=accessor,
                    checker=director.make_bench_checker(accessor),
                    local_ids_with_warnings=(),
                    max_tries=self.max_tries,
                    tries_count=0,
                )
                self.update_local_ids_with_warnings(case_state)
                env_to_case_state[env_description] = case_state

            hub_to_env_to_case_state[hub_description] = env_to_case_state

        self.process_all_cases(hub_to_env_to_case_state)

    def _render_tries(self, tries_max_size: int, case_state: CaseState) -> str:
        if case_state.max_tries is None:
            return str(case_state.tries_count).ljust(tries_max_size) + " " * (tries_max_size + 1)
        return f"{str(case_state.tries_count).ljust(tries_max_size)}/{str(case_state.max_tries).ljust(tries_max_size)}"

    def _render_cases_to_complete(
        self,
        hub_to_env_to_case_state: Mapping[HubDescription, Mapping[EnvDescription, CaseState]],
    ) -> str:
        hub_to_env_to_case_state = {
            hub_description: env_to_case_state
            for hub_description, env_to_case_state in hub_to_env_to_case_state.items()
            if any(not case_state.is_completed for case_state in env_to_case_state.values())
        }
        hub_max_size = max(
            len(hub_description.key)
            for hub_description in hub_to_env_to_case_state.keys()
        )
        env_max_size = max(
            len(env_description.key)
            for env_to_case_state in hub_to_env_to_case_state.values()
            for env_description in env_to_case_state.keys()
        )
        tries_max_size = max(
            1 if case_state.max_tries is None else len(str(case_state.max_tries))
            for env_to_case_state in hub_to_env_to_case_state.values()
            for case_state in env_to_case_state.values()
        )
        return "\n".join(
            f"{hub_description.key.rjust(hub_max_size)} tries:  "
            + "  ".join(
                f"{env_description.key.ljust(env_max_size)} - {self._render_tries(tries_max_size, case_state)}"
                for env_description, case_state in env_to_case_state.items()
                if not case_state.is_completed
            )
            for hub_description, env_to_case_state in hub_to_env_to_case_state.items()
        )

    def process_all_cases(
        self,
        hub_to_env_to_case_state: Mapping[HubDescription, Mapping[EnvDescription, CaseState]],
    ) -> None:
        while True:
            has_uncompleted = False
            for hub_description, env_to_case_state in hub_to_env_to_case_state.items():
                for env_description, case_state in env_to_case_state.items():
                    if case_state.is_completed or case_state.is_out_of_tries:
                        continue

                    has_uncompleted = True
                    cases_to_complete = (
                        "\n" + self._render_cases_to_complete(hub_to_env_to_case_state)
                    ).replace("\n", "\n  ")
                    self.print("Cases to complete:" + cases_to_complete)
                    self.print()

                    self.run_case(hub_description, env_description, case_state)

            if not has_uncompleted:
                return

    def update_local_ids_with_warnings(self, case_state: CaseState) -> None:
        local_ids_with_warnings = []
        reader = operator_factory(case_state.accessor, sqlite=bool(self.sqlite))
        for schema in case_state.accessor.schemas:
            warnings = case_state.checker.get_warnings(schema, reader)
            if warnings is None or warnings:
                local_ids_with_warnings.append(
                    case_state.accessor.get_local_id(schema),
                )

        case_state.local_ids_with_warnings = local_ids_with_warnings

    def run_case(self, hub_description: HubDescription, env_description: EnvDescription, case_state: CaseState) -> None:
        hub_module = hub_description.module
        env = env_description.tox_env
        for _ in range(self.series):
            self.print(f"Running   hub: {hub_description.key:<30} env: {env}")
            self.run(f"tox exec -e {env} --skip-pkg-install -- python -m {hub_module} run --unstable")
            case_state.tries_count += 1
            self.update_local_ids_with_warnings(case_state)
            if case_state.is_completed:
                case_state.director.make_bench_plotter(case_state.accessor).draw_plot(output=None, dpi=100)
                return
            self.print("Got unstable results for " + " ".join(case_state.local_ids_with_warnings))
            if case_state.is_out_of_tries:
                self.print(
                    f"WARNING: too many tries to get stable benchmark results -- {case_state.tries_count},"
                    f" restarting is stopped",
                )
                return


@dataclass
class ColorScheme:
    bg_color: str
    bar_color: str
    bar_bordercolor: str
    button_bordercolor: str
    button_bgcolor: str
    button_font_color: str
    font_color: str
    hoverlabel_bgcolor: str
    template: str
    updatemenus_showactive: bool
    selected_env_annotation_color: str


class Renderer(HubProcessor):
    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--output",
            "-o",
            action="store",
            required=False,
        )

    def __init__(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        env_include: Optional[Sequence[str]] = None,
        env_exclude: Optional[Sequence[str]] = None,
        output: Optional[str] = None,
        sqlite: bool = False,
    ):
        super().__init__(
            sqlite=sqlite,
            include=include,
            exclude=exclude,
            env_include=env_include,
            env_exclude=env_exclude,
        )
        self.output = output

    HTML_TEMPLATE = dedent(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
        </head>
        <body>
            {body}
        </body>
        </html>
        """,
    )

    def start(self) -> None:
        body = ""
        for hub_description, env_to_director in self.load_directors(self.filtered_hubs()).items():
            figure = self.render_hub_from_workbench(hub_description, env_to_director)
            figure.update_layout(width=809)
            body += plotly.offline.plot(
                figure,
                config={"displaylogo": False},
                include_plotlyjs="cdn",
                output_type="div",
            )
            body += "<p></p>"
        output = "benchmark_plots.html" if self.output is None else self.output
        Path(output).write_text(self.HTML_TEMPLATE.format(body=body))
        self.print(f"Open file://{Path(output).absolute()}")

    def _director_to_measures(self, director: BenchmarkDirector) -> Sequence[BenchmarkMeasure]:
        reader = operator_factory(director.make_accessor(), sqlite=bool(self.sqlite))
        measures = [pyperf_bench_to_measure(d) for d in reader.get_all_bench_results()]
        measures.sort(key=lambda x: x.pyperf.mean())
        return measures

    def _release_zip_to_measures(
        self,
        hub_description: HubDescription,
    ) -> Mapping[EnvDescription, Sequence[BenchmarkMeasure]]:
        with ZipFile(RELEASE_DATA / f"{hub_description.key}.zip") as release_zip:
            index = json.loads(release_zip.read("index.json"))
            env_to_files = {
                KEY_TO_ENV[env_key]: files
                for env_key, files in index["env_files"].items()
            }
            return {
                env: sorted(
                    (
                        pyperf_bench_to_measure(release_zip.read(file))
                        for file in files
                    ),
                    key=lambda x: x.pyperf.mean(),
                )
                for env, files in env_to_files.items()
            }

    def render_hub_from_workbench(
        self,
        hub_description: HubDescription,
        env_to_director: Mapping[EnvDescription, BenchmarkDirector],
    ) -> go.Figure:
        env_to_measures = {
            env_description: self._director_to_measures(director)
            for env_description, director in env_to_director.items()
        }
        return self.create_hub_plot(hub_description, env_to_measures, self.LIGHT_COLOR_SCHEME)

    def render_hub_from_release(
        self,
        hub_description: HubDescription,
        color_scheme: ColorScheme,
    ) -> go.Figure:
        figure = self.create_hub_plot(hub_description, self._release_zip_to_measures(hub_description), color_scheme)
        return figure

    BASE_RENAMING = {
        "pydantic": "pydantic v2",
    }

    def _get_label(self, measure: BenchmarkMeasure) -> str:
        base = self.BASE_RENAMING.get(measure.base) or measure.base
        if measure.tags:
            tags_str = ", ".join(measure.tags)
            return f"{base}<br>({tags_str})"
        return base

    LIGHT_COLOR_SCHEME = ColorScheme(
        bg_color="white",
        bar_color="rgba(129, 182, 230, 1.0)",
        bar_bordercolor="rgba(17, 95, 143, 1.0)",
        button_bordercolor="black",
        button_bgcolor="white",
        button_font_color="black",
        font_color="black",
        hoverlabel_bgcolor="white",
        template="plotly_white",
        updatemenus_showactive=True,
        selected_env_annotation_color="rgb(171, 171, 171)",
    )
    DARK_COLOR_SCHEME = ColorScheme(
        bg_color="#131416",
        bar_color="#1a687d",
        bar_bordercolor="#4abdd4",
        button_bordercolor="white",
        button_bgcolor="#202020",
        button_font_color="rgb(171, 171, 171)",
        font_color="white",
        hoverlabel_bgcolor="#202020",
        template="plotly_dark",
        updatemenus_showactive=False,
        selected_env_annotation_color="rgb(171, 171, 171)",
    )

    def _create_bar_chart(self, color_scheme: ColorScheme, measures: Sequence[BenchmarkMeasure]) -> go.Bar:
        return go.Bar(
            x=[
                measure.pyperf.mean() * 10 ** 6
                for measure in measures
            ],
            y=[
                self._get_label(measure)
                for measure in measures
            ],
            error_x={
                "type": "data",
                "array": [
                    measure.pyperf.stdev() * 10 ** 6
                    for measure in measures
                ],
                "width": 4,
                "thickness": 1.5,
                "color": color_scheme.bar_bordercolor,
            },
            texttemplate="%{x:,.1f}",
            textposition="inside",
            insidetextanchor="start",
            insidetextfont={
                "family": "Helvetica",
                "color": "white",
            },
            orientation="h",
            marker={
                "color": color_scheme.bar_color,
                "line": {
                    "color": color_scheme.bar_bordercolor,
                    "width": 1.2,
                },
            },
            hovertemplate=(
                "<b>%{customdata.base}</b><br>"
                "%{x:,.1f} μs ± %{error_x.array:.1f} μs (%{customdata.rel_error:.1f}%)<br>"
                "<br>"
                "%{customdata.kwargs}"
                "<extra></extra>"
            ),
            customdata=[
                {
                    "base": measure.base,
                    "rel_error": measure.pyperf.stdev() / measure.pyperf.mean() * 100,
                    "kwargs": "<br>".join(
                        f"{key}={value}"
                        for key, value in measure.kwargs.items()
                    ),
                }
                for measure in measures
            ],
        )

    DEFAULT_ENV_KEY = "py312"

    def create_hub_plot(
        self,
        hub_description: HubDescription,
        env_to_measures: Mapping[EnvDescription, Sequence[BenchmarkMeasure]],
        color_scheme: ColorScheme,
    ) -> go.Figure:
        bar_charts = [
            self._create_bar_chart(color_scheme, measures).update(visible=False)
            for measures in env_to_measures.values()
        ]
        visible_by_default = next(
            idx for idx, env_description in enumerate(self.filtered_envs())
            if env_description.key == self.DEFAULT_ENV_KEY
        )
        bar_charts[visible_by_default].update(visible=True)
        buttons = [
            {
                "args": [
                    {"visible": [env_idx == bar_idx for bar_idx in range(len(bar_charts))]},
                    {"annotations[0].text": env_description.title},
                ],
                "label": env_description.title,
                "method": "update",
            }
            for env_idx, env_description in enumerate(self.filtered_envs())
        ]
        height = max(len(measures) for accessor, measures in env_to_measures.items()) * 40 + 185
        # pylint: disable=no-member
        figure = go.Figure(
            bar_charts,
        ).update_layout(
            title={
                "text": f"<b>{hub_description.title}</b>",
                "x": 0.5,
                "y": 0.97,
                "xanchor": "center",
                "yanchor": "top",
                "font": {
                    "size": 18,
                },
            },
            xaxis_title="Time (μs)",
            font={
                "family": "Helvetica",
                "size": 12,
                "color": color_scheme.font_color,
            },
            template=color_scheme.template,
            margin_pad=10,
            hoverlabel={
                "bgcolor": color_scheme.hoverlabel_bgcolor,
            },
            paper_bgcolor=color_scheme.bg_color,
            plot_bgcolor=color_scheme.bg_color,
        ).update_xaxes(
            range=[0, hub_description.x_bounder.get_hub_x_bound(env_to_measures)],
        ).update_layout(
            height=height,
            autosize=True,
            xaxis_fixedrange=True,
            yaxis_fixedrange=True,
            modebar_remove=["zoom", "pan", "select", "zoomIn", "zoomOut", "autoScale", "resetScale", "lasso2d"],
            clickmode="none",
            dragmode=False,
        ).update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "buttons": buttons,
                    "borderwidth": 0.5,
                    "bordercolor": color_scheme.button_bordercolor,
                    "bgcolor": color_scheme.button_bgcolor,
                    "showactive": color_scheme.updatemenus_showactive,
                    "font": {
                        "color": color_scheme.button_font_color,
                    },
                    "active": visible_by_default,
                    "x": 0.5,
                    "y": 1 + (64 / height),
                    "xanchor": "center",
                    "yanchor": "top",
                },
            ],
        ).add_annotation(
            xref="x domain",
            yref="y domain",
            x=0.95,
            y=0.05,
            text=next(
                env_description.title for env_description in self.filtered_envs()
                if env_description.key == self.DEFAULT_ENV_KEY
            ),
            xanchor="right",
            showarrow=False,
            font=dict(
                size=13,
                color=color_scheme.selected_env_annotation_color,
            ),
        )
        return figure


class HubValidator(HubProcessor):
    def start(self) -> None:
        errors = self.validate(self.filtered_hubs())
        if errors:
            self.print("\n".join(errors))

    def validate(self, hub_descriptions: Iterable[HubDescription]) -> Sequence[str]:
        hub_to_director_to_env = self.load_directors(hub_descriptions)
        errors: List[str] = []
        errors.extend(self._validate_distributions(hub_to_director_to_env))
        return errors

    def _validate_distributions(
        self,
        hub_to_director_to_env: Mapping[HubDescription, Mapping[EnvDescription, BenchmarkDirector]],
    ) -> Sequence[str]:
        dist_to_versions: DefaultDict[str, Set[str]] = defaultdict(set)
        for _hub_description, env_to_director in hub_to_director_to_env.items():
            for director in env_to_director.values():
                reader = operator_factory(director.make_accessor(), sqlite=bool(self.sqlite))
                for bench_result in reader.get_all_bench_results():
                    for dist, version in json.loads(bench_result)["pybench_data"]["distributions"].items():
                        dist_to_versions[dist].add(version)
        return [
            f"Benchmarks using distribution {dist!r} were taken with different versions {versions!r}"
            for dist, versions in dist_to_versions.items()
            if len(versions) > 1
        ]


class Releaser(HubProcessor):
    def start(self) -> None:
        validator = HubValidator()
        validator.validate(self.filtered_hubs())
        self._release()

    def _get_index_data(self, env_to_files: Mapping[EnvDescription, Iterable[str]]) -> Dict[str, Any]:
        return {
            "env_files": {
                env_description.key: [
                    file for file in files
                ]
                for env_description, files in env_to_files.items()
            },
        }

    def _release_from_files(
        self,
        operator: FileSystemBenchOperator,
        release_zip: ZipFile,
        bench_ids: list[str],
        bench_results: Sequence[str],
    ):
        for file_path, data in zip(
            [
                operator.bench_result_file(id_)
                for id_ in bench_ids
            ],
            bench_results,
        ):
            release_zip.writestr(file_path.name, data)

    def _release_from_sqlite(
        self,
        release_zip: ZipFile,
        bench_ids: list[str],
        bench_results: Sequence[str],
    ):
        for name, data in zip(bench_ids, bench_results):
            release_zip.writestr(name, data)

    def _release(self):
        hub_to_director_to_env = self.load_directors(self.filtered_hubs())
        for hub_description, env_to_director in hub_to_director_to_env.items():
            env_with_accessor = [
                (env_description, director.make_accessor())
                for env_description, director in env_to_director.items()
            ]
            env_to_accessor = {
                env_description: accessor for env_description, accessor in env_with_accessor
            }

            with ZipFile(
                file=RELEASE_DATA / f"{hub_description.key}.zip",
                mode="w",
                compression=ZIP_BZIP2,
                compresslevel=9,
            ) as release_zip:
                for env in env_to_accessor:
                    accessor = env_to_accessor[env]
                    bench_operator = operator_factory(accessor, sqlite=bool(self.sqlite))
                    bench_results = bench_operator.get_all_bench_results()
                    bench_ids = [accessor.get_id(schema) for schema in accessor.schemas]
                    for name, data in zip(bench_ids, bench_results):
                        release_zip.writestr(name + ".json", data)

                release_zip.writestr(
                    "index.json",
                    json.dumps(
                        self._get_index_data(
                            {
                                env_description: [
                                    accessor.get_id(schema) + ".json"
                                    for schema in accessor.schemas
                                ]
                                for env_description, accessor in env_with_accessor
                            },
                        ),
                    ),
                )


class ListGetter(Foundation):
    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        pass

    def start(self) -> None:
        self.print("Hubs:")
        self.print("\n".join([hub_description.key for hub_description in BENCHMARK_HUBS]))
        self.print()
        self.print("Envs:")
        self.print("\n".join([env_description.key for env_description in BENCHMARK_ENVS]))


COMMAND_TO_CLS = {
    "run": Orchestrator,
    "render": Renderer,
    "validate": HubValidator,
    "release": Releaser,
    "list": ListGetter,
}


def main():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(required=True)
    for command, cls in COMMAND_TO_CLS.items():
        command_parser = subparsers.add_parser(command)
        command_parser.set_defaults(command=command)
        cls.add_arguments(command_parser)

    namespace = parser.parse_args()
    cls = COMMAND_TO_CLS[namespace.command]
    instance = call_by_namespace(cls, namespace)
    instance.start()


if __name__ == "__main__":
    main()
