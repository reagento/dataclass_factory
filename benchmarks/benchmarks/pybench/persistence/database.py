import datetime
import json
from collections.abc import Sequence
from sqlite3 import Connection, connect
from typing import Any, Optional

from benchmarks.pybench.persistence.common import BenchAccessProto, BenchOperator, BenchRecord

DATABASE_FILE_NAME = "adaptix_bench.db"


class _RecordNotFound(Exception):
    def __init__(self, gid: str, name: str, sub_name: str):
        self.gid = gid
        self.name = name
        self.sub_name = sub_name

    def __str__(self):
        return f"""Record not found for {self.name}/{self.sub_name}({self.gid})."""


def database_initialization(database_name: str):
    create_table_ddl_q = """
    BEGIN;
    CREATE TABLE IF NOT EXISTS bench (
    is_actual BOOLEAN,
    benchmark_name TEXT,
    benchmark_subname TEXT,
    base TEXT,
    local_id TEXT,
    global_id TEXT,
    tags TEXT,
    kwargs TEXT,
    distributions TEXT,
    data TEXT,
    created_at DATETIME
    );
    CREATE UNIQUE INDEX IF NOT EXISTS bench_unique ON bench (
     benchmark_name, benchmark_subname, global_id
    ) WHERE is_actual=TRUE;
    COMMIT;
     """
    with connect(database_name) as con:
        con.executescript(create_table_ddl_q)


class SQLite3BenchOperator(BenchOperator):
    GET_BENCH_DATA_Q = """
    SELECT data, max(created_at)
    FROM bench
    WHERE benchmark_name = ?
        AND benchmark_subname = ?
        AND global_id = ?;
    """
    INSERT_BENCH_DATA_Q = """
    INSERT OR REPLACE INTO bench (
    is_actual,
    benchmark_name,
    benchmark_subname,
    base,
    local_id,
    global_id,
    tags,
    kwargs,
    distributions,
    data,
    created_at
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?);
    """

    def __init__(self, accessor: BenchAccessProto, db_name: str):
        self.accessor = accessor
        self.db_name = db_name

    def get_all_bench_results(self) -> Sequence[str]:
        con = connect(self.db_name)
        content_container = []
        for schema in self.accessor.schemas:
            global_id = self.accessor.get_id(schema)
            content_container.append(self._bench_data(con, global_id,
                                                      self.accessor.meta.benchmark_name,
                                                      self.accessor.meta.benchmark_subname))
        con.close()
        return content_container

    def get_bench_result(self, schema: Any) -> Optional[str]:
        con = connect(self.db_name)
        result = con.execute(self.GET_BENCH_DATA_Q,
                             (
                                 self.accessor.meta.benchmark_name,
                                 self.accessor.meta.benchmark_subname,
                                 self.accessor.get_id(schema),
                             ))
        data = result.fetchone()
        con.close()
        if not data:
            return None
        return data[0]

    def _bench_data(self, connection: Connection, bench_id, bench_name: str, bench_sub_name: str) -> str:
        result = connection.execute(self.GET_BENCH_DATA_Q, (bench_name, bench_sub_name, bench_id))
        data = result.fetchone()
        if not data:
            raise _RecordNotFound(bench_id, bench_name, bench_sub_name)
        return data[0]

    def write_bench_record(self, record: BenchRecord) -> None:

        with connect(self.db_name) as conn:
            conn.execute(self.INSERT_BENCH_DATA_Q, (
                record["is_actual"],
                record["benchmark_name"],
                record["benchmark_subname"],
                record["base"],
                record["local_id"],
                record["global_id"],
                json.dumps(record["tags"]),
                json.dumps(record["kwargs"]),
                json.dumps(record["distributions"]),
                record["data"],
                datetime.datetime.now(tz=datetime.timezone.utc),
            ))
            conn.commit()


def sqlite_operator_factory(accessor: BenchAccessProto) -> SQLite3BenchOperator:
    database_initialization(DATABASE_FILE_NAME)
    return SQLite3BenchOperator(accessor, DATABASE_FILE_NAME)
