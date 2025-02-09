import sys
from pathlib import Path

from adaptix import DebugTrail
from benchmarks.pybench.director_api import BenchmarkDirector, BenchSchema, CheckParams, PlotParams
from benchmarks.pybench.persistence.common import BenchMeta
from benchmarks.pybench.persistence.database import SQLite3BenchOperator, sqlite_operator_factory
from benchmarks.simple_structures import (
    bench_adaptix,
    bench_cattrs,
    bench_dataclass_factory,
    bench_marshmallow,
    bench_mashumaro,
    bench_pydantic,
    bench_schematics,
)

REVIEWS_COUNT = 100

director = BenchmarkDirector(
    data_dir=Path(__file__).parent.parent.parent / "data" / "simple_structures" / "loading",
    plot_params=PlotParams(
        title="Small Structures Benchmark (loading)",
        fig_size=(9, 7.5),
        label_padding=5,
        trim_after=300,
    ),
    env_spec={
        "py": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "py_impl": sys.implementation.name,
    },
    check_params=lambda env_spec: CheckParams(
        stdev_rel_threshold=0.07 if env_spec["py_impl"] == "pypy" else 0.04,
    ),
    meta=BenchMeta(benchmark_subname="loading", benchmark_name="simple_structures"),
)

director.add(
    BenchSchema(
        entry_point=bench_adaptix.bench_loading,
        base="adaptix",
        tags=["sc", "dt_all"],
        kwargs={"strict_coercion": True, "debug_trail": DebugTrail.ALL.value, "reviews_count": REVIEWS_COUNT},
        used_distributions=["adaptix"],
    ),
    BenchSchema(
        entry_point=bench_adaptix.bench_loading,
        base="adaptix",
        tags=["sc", "dt_first"],
        kwargs={"strict_coercion": True, "debug_trail": DebugTrail.FIRST.value, "reviews_count": REVIEWS_COUNT},
        used_distributions=["adaptix"],
    ),
    BenchSchema(
        entry_point=bench_adaptix.bench_loading,
        base="adaptix",
        tags=["sc", "dt_disable"],
        kwargs={"strict_coercion": True, "debug_trail": DebugTrail.DISABLE.value, "reviews_count": REVIEWS_COUNT},
        used_distributions=["adaptix"],
    ),
    BenchSchema(
        entry_point=bench_adaptix.bench_loading,
        base="adaptix",
        tags=["dt_all"],
        kwargs={"strict_coercion": False, "debug_trail": DebugTrail.ALL.value, "reviews_count": REVIEWS_COUNT},
        used_distributions=["adaptix"],
    ),
    BenchSchema(
        entry_point=bench_adaptix.bench_loading,
        base="adaptix",
        tags=["dt_first"],
        kwargs={"strict_coercion": False, "debug_trail": DebugTrail.FIRST.value, "reviews_count": REVIEWS_COUNT},
        used_distributions=["adaptix"],
    ),
    BenchSchema(
        entry_point=bench_adaptix.bench_loading,
        base="adaptix",
        tags=["dt_disable"],
        kwargs={"strict_coercion": False, "debug_trail": DebugTrail.DISABLE.value, "reviews_count": REVIEWS_COUNT},
        used_distributions=["adaptix"],
    ),
)

director.add(
    BenchSchema(
        entry_point=bench_mashumaro.bench_loading,
        base="mashumaro",
        tags=[],
        kwargs={"lazy_compilation": False, "reviews_count": REVIEWS_COUNT},
        used_distributions=["mashumaro"],
    ),
    BenchSchema(
        entry_point=bench_mashumaro.bench_loading,
        base="mashumaro",
        tags=["lc"],
        kwargs={"lazy_compilation": True, "reviews_count": REVIEWS_COUNT},
        used_distributions=["mashumaro"],
    ),
)

director.add(
    BenchSchema(
        entry_point=bench_pydantic.bench_loading,
        base="pydantic",
        tags=["strict"],
        kwargs={"strict": True, "reviews_count": REVIEWS_COUNT},
        used_distributions=["pydantic"],
        check_params=lambda env_spec: CheckParams(
            stdev_rel_threshold=0.3 if env_spec["py_impl"] == "pypy" else None,
            ignore_pyperf_warnings=True if env_spec["py_impl"] == "pypy" else None,
        ),
    ),
    BenchSchema(
        entry_point=bench_pydantic.bench_loading,
        base="pydantic",
        tags=[],
        kwargs={"strict": False, "reviews_count": REVIEWS_COUNT},
        used_distributions=["pydantic"],
        check_params=lambda env_spec: CheckParams(
            stdev_rel_threshold=0.3 if env_spec["py_impl"] == "pypy" else None,
            ignore_pyperf_warnings=True if env_spec["py_impl"] == "pypy" else None,
        ),
    ),
)

director.add(
    BenchSchema(
        entry_point=bench_cattrs.bench_loading,
        base="cattrs",
        tags=["dv"],
        kwargs={"detailed_validation": True, "reviews_count": REVIEWS_COUNT},
        used_distributions=["cattrs"],
    ),
    BenchSchema(
        entry_point=bench_cattrs.bench_loading,
        base="cattrs",
        tags=[],
        kwargs={"detailed_validation": False, "reviews_count": REVIEWS_COUNT},
        used_distributions=["cattrs"],
    ),
)

director.add(
    BenchSchema(
        entry_point=bench_schematics.bench_loading,
        base="schematics",
        tags=[],
        kwargs={"reviews_count": REVIEWS_COUNT},
        used_distributions=["schematics"],
    ),
)

director.add(
    BenchSchema(
        entry_point=bench_dataclass_factory.bench_loading,
        base="dataclass_factory",
        tags=["dp"],
        kwargs={"debug_path": True, "reviews_count": REVIEWS_COUNT},
        used_distributions=["dataclass_factory"],
    ),
    BenchSchema(
        entry_point=bench_dataclass_factory.bench_loading,
        base="dataclass_factory",
        tags=[],
        kwargs={"debug_path": False, "reviews_count": REVIEWS_COUNT},
        used_distributions=["dataclass_factory"],
    ),
)

director.add(
    BenchSchema(
        entry_point=bench_marshmallow.bench_loading,
        base="marshmallow",
        tags=[],
        kwargs={"reviews_count": REVIEWS_COUNT},
        used_distributions=["marshmallow"],
    ),
)

director.add(
    BenchSchema(
        entry_point="benchmarks.simple_structures.bench_msgspec:bench_loading",
        base="msgspec",
        tags=[],
        kwargs={"strict": False, "no_gc": False, "reviews_count": REVIEWS_COUNT},
        skip_if=lambda env_spec: env_spec["py_impl"] == "pypy",
        used_distributions=["msgspec"],
    ),
    BenchSchema(
        entry_point="benchmarks.simple_structures.bench_msgspec:bench_loading",
        base="msgspec",
        tags=["strict"],
        kwargs={"strict": True, "no_gc": False, "reviews_count": REVIEWS_COUNT},
        skip_if=lambda env_spec: env_spec["py_impl"] == "pypy",
        used_distributions=["msgspec"],
    ),
    BenchSchema(
        entry_point="benchmarks.simple_structures.bench_msgspec:bench_loading",
        base="msgspec",
        tags=["no_gc"],
        kwargs={"strict": False, "no_gc": True, "reviews_count": REVIEWS_COUNT},
        skip_if=lambda env_spec: env_spec["py_impl"] == "pypy",
        used_distributions=["msgspec"],
    ),
    BenchSchema(
        entry_point="benchmarks.simple_structures.bench_msgspec:bench_loading",
        base="msgspec",
        tags=["strict", "no_gc"],
        kwargs={"strict": True, "no_gc": True, "reviews_count": REVIEWS_COUNT},
        skip_if=lambda env_spec: env_spec["py_impl"] == "pypy",
        used_distributions=["msgspec"],
    ),
)

if __name__ == "__main__":
    director.cli()
