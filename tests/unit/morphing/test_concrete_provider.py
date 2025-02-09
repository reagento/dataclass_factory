# ruff: noqa: DTZ001
import re
import typing
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from io import BytesIO
from typing import Union

import pytest
from tests_helpers import cond_list, raises_exc
from unit.integrations.sqlalchemy.test_orm import retort

from adaptix import DebugTrail, Omittable, Omitted, Retort
from adaptix._internal.feature_requirement import HAS_PY_311, IS_PYPY
from adaptix._internal.morphing.concrete_provider import (
    DatetimeFormatProvider,
    DateTimestampProvider,
    DatetimeTimestampProvider,
)
from adaptix._internal.morphing.dump_error import SentinelDumpError
from adaptix._internal.morphing.load_error import LoadError, UnionLoadError
from adaptix.load_error import FormatMismatchLoadError, TypeLoadError, ValueLoadError

INVALID_INPUT_ISO_FORMAT = (
    None,
    10,
    datetime(2011, 11, 4, 0, 0),
    date(2019, 12, 4),
    time(4, 23, 1),
)

INVALID_INPUT_TIMESTAMP = (
    None,
    datetime(2011, 11, 4, 0, 0),
    date(2019, 12, 4),
    time(4, 23, 1),
)


@pytest.mark.parametrize(
    "tp",
    [datetime, date, time],
)
@pytest.mark.parametrize(
    "value",
    INVALID_INPUT_ISO_FORMAT,
)
def test_invalid_input_iso_format(
    strict_coercion,
    debug_trail,
    value,
    tp,
):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    loader = retort.get_loader(tp)

    raises_exc(
        TypeLoadError(str, value),
        lambda: loader(value),
    )


@pytest.mark.parametrize(
    "value",
    INVALID_INPUT_ISO_FORMAT,
)
def test_invalid_input_datetime_format(
    strict_coercion,
    debug_trail,
    value,
):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
        recipe=[
            DatetimeFormatProvider("%Y-%m-%d"),
        ],
    )

    loader = retort.get_loader(datetime)

    raises_exc(
        TypeLoadError(str, value),
        lambda: loader(value),
    )


@pytest.mark.parametrize(
    ["tp", "loader"],
    [
        (datetime, DatetimeTimestampProvider(tz=timezone.utc)),
        (date, DateTimestampProvider()),
    ],
)
@pytest.mark.parametrize(
    "value",
    INVALID_INPUT_TIMESTAMP,
)
def test_invalid_input_timestamp(
    strict_coercion,
    debug_trail,
    value,
    tp,
    loader,
):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
        recipe=[
            loader,
        ],
    )

    loader = retort.get_loader(tp)

    raises_exc(
        TypeLoadError(Union[float, int], value),
        lambda: loader(value),
    )


def test_iso_format_provider_datetime(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    loader = retort.get_loader(datetime)
    assert loader("2011-11-04") == datetime(2011, 11, 4, 0, 0)
    assert loader("2011-11-04T00:05:23") == datetime(2011, 11, 4, 0, 5, 23)
    assert loader("2011-11-04T00:05:23+04:00") == datetime(
        2011, 11, 4, 0, 5, 23,
        tzinfo=timezone(timedelta(seconds=14400)),
    )

    raises_exc(
        ValueLoadError("Invalid isoformat string", "some string"),
        lambda: loader("some string"),
    )

    dumper = retort.get_dumper(datetime)
    assert dumper(datetime(2011, 11, 4, 0, 0)) == "2011-11-04T00:00:00"


def test_iso_format_provider_date(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    loader = retort.get_loader(date)
    assert loader("2019-12-04") == date(2019, 12, 4)

    raises_exc(
        ValueLoadError("Invalid isoformat string", "some string"),
        lambda: loader("some string"),
    )

    dumper = retort.get_dumper(date)
    assert dumper(date(2019, 12, 4)) == "2019-12-04"


def test_iso_format_provider_time(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    loader = retort.get_loader(time)
    assert loader("04:23:01") == time(4, 23, 1)
    assert loader("04:23:01+04:00") == time(
        4, 23, 1,
        tzinfo=timezone(timedelta(seconds=14400)),
    )

    raises_exc(
        ValueLoadError("Invalid isoformat string", "some string"),
        lambda: loader("some string"),
    )

    dumper = retort.get_dumper(time)
    assert dumper(time(4, 23, 1)) == "04:23:01"


def test_datetime_format_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
        recipe=[
            DatetimeFormatProvider("%Y-%m-%d"),
        ],
    )

    loader = retort.get_loader(datetime)
    assert loader("3045-02-13") == datetime(year=3045, month=2, day=13)

    raises_exc(
        FormatMismatchLoadError("%Y-%m-%d", "some string"),
        lambda: loader("some string"),
    )

    dumper = retort.get_dumper(datetime)
    assert dumper(datetime(year=3045, month=2, day=13)) == "3045-02-13"


@pytest.mark.parametrize(
    "tz",
    [
        None,
        timezone(timedelta(hours=3)),
    ],
)
def test_datetime_timestamp_provider(strict_coercion, debug_trail, tz: timezone):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
        recipe=[
            DatetimeTimestampProvider(tz=tz),
        ],
    )

    loader = retort.get_loader(datetime)

    dt = datetime(2011, 11, 4, 6, 38, tzinfo=tz)
    ts = dt.timestamp()

    assert loader(ts) == dt

    overflow_ts = float("inf")
    raises_exc(
        ValueLoadError("Timestamp is out of the range of supported values", overflow_ts),
        lambda: loader(overflow_ts),
    )

    nan = float("nan")
    raises_exc(
        ValueLoadError("Unexpected value", nan),
        lambda: loader(nan),
    )

    dumper = retort.get_dumper(datetime)
    assert dumper(dt) == ts


def test_date_timestamp_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
        recipe=[
            DateTimestampProvider(),
        ],
    )

    loader = retort.get_loader(date)
    dt = datetime(2011, 11, 4, tzinfo=timezone.utc)
    today = dt.date()

    ts = dt.timestamp()

    assert loader(ts) == today

    overflow_ts = float("inf")
    raises_exc(
        ValueLoadError("Timestamp is out of the range of supported values", overflow_ts),
        lambda: loader(overflow_ts),
    )

    nan = float("nan")
    raises_exc(
        ValueLoadError("Unexpected value", nan),
        lambda: loader(nan),
    )

    dumper = retort.get_dumper(date)
    assert dumper(dt) == ts


def test_seconds_timedelta_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    loader = retort.get_loader(timedelta)
    assert loader(10) == timedelta(seconds=10)
    assert loader(600) == timedelta(minutes=10)
    assert loader(0.123) == timedelta(milliseconds=123)
    assert loader(Decimal("0.123")) == timedelta(milliseconds=123)

    dumper = retort.get_dumper(timedelta)
    assert dumper(timedelta(minutes=10)) == 600


def test_none_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    loader = retort.get_loader(None)

    assert loader(None) is None
    raises_exc(
        TypeLoadError(None, 10),
        lambda: loader(10),
    )

    dumper = retort.get_dumper(None)
    assert dumper(None) is None


@pytest.mark.parametrize(
    ["provider_type", "get_string", "get_bytes"],
    [
        (bytes, lambda x: x.decode(), lambda x: x.encode()),
        (bytearray, lambda x: x.decode(), lambda x: bytearray(x.encode())),
        (BytesIO, lambda x: x.getvalue().decode(), lambda x: BytesIO(x.encode())),
        (typing.IO[bytes], lambda x: x.read().decode(), lambda x: BytesIO(x.encode())),
    ],
)
def test_bytes_like_provider(
    strict_coercion,
    debug_trail,
    provider_type,
    get_string,
    get_bytes,
):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    loader = retort.get_loader(provider_type)
    string = "abcd"
    b64_string = b"YWJjZA=="

    assert get_string(loader(b64_string.decode())) == string

    raises_exc(
        ValueLoadError("Bad base64 string", "Hello, world"),
        lambda: loader("Hello, world"),
    )

    raises_exc(
        ValueLoadError(
            msg="Invalid base64-encoded string: number of data characters (5)"
                " cannot be 1 more than a multiple of 4",
            input_value="aaaaa=",
        ),
        lambda: loader("aaaaa="),
    )

    raises_exc(
        ValueLoadError("Incorrect padding", "YWJjZA"),
        lambda: loader("YWJjZA"),
    )

    raises_exc(
        TypeLoadError(str, 108),
        lambda: loader(108),
    )

    dumper = retort.get_dumper(provider_type)
    assert dumper(get_bytes(string)) == b64_string.decode()


def test_regex_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    loader = retort.get_loader(re.Pattern)

    assert loader(r"\w") == re.compile(r"\w")

    raises_exc(
        TypeLoadError(str, 10),
        lambda: loader(10),
    )
    raises_exc(
        ValueLoadError("bad escape (end of pattern) at position 0", "\\"),
        lambda: loader("\\"),
    )

    dumper = retort.get_dumper(re.Pattern)
    assert dumper(re.compile(r"\w")) == r"\w"


def test_int_loader_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )
    loader = retort.get_loader(int)

    assert loader(100) == 100

    if strict_coercion:
        raises_exc(TypeLoadError(int, None), lambda: loader(None))
        raises_exc(TypeLoadError(int, "foo"), lambda: loader("foo"))
        raises_exc(TypeLoadError(int, "100"), lambda: loader("100"))
    else:
        raises_exc(TypeLoadError(Union[int, float, str], None), lambda: loader(None))
        raises_exc(ValueLoadError("Bad string format", "foo"), lambda: loader("foo"))
        assert loader("100") == 100


def test_float_loader_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )
    loader = retort.get_loader(float)

    assert loader(100.0) == 100.0
    assert isinstance(loader(100), float)

    if strict_coercion:
        raises_exc(TypeLoadError(Union[int, float], None), lambda: loader(None))
        raises_exc(TypeLoadError(Union[int, float], "foo"), lambda: loader("foo"))
        raises_exc(TypeLoadError(Union[int, float], "100"), lambda: loader("100"))
    else:
        raises_exc(TypeLoadError(Union[int, float, str], None), lambda: loader(None))
        raises_exc(ValueLoadError("Bad string format", "foo"), lambda: loader("foo"))
        assert loader("100") == 100


@pytest.mark.parametrize("tp", [str, *cond_list(HAS_PY_311, lambda: [typing.LiteralString])])
def test_str_loader_provider(strict_coercion, debug_trail, tp):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )
    loader = retort.get_loader(tp)

    assert loader("foo") == "foo"

    if strict_coercion:
        raises_exc(TypeLoadError(str, None), lambda: loader(None))
    else:
        assert loader(None) == "None"


def test_bool_loader_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )
    loader = retort.get_loader(bool)

    assert loader(True) is True  # noqa: FBT003

    if strict_coercion:
        raises_exc(TypeLoadError(bool, None), lambda: loader(None))
    else:
        assert loader(None) is False


def test_decimal_loader_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )
    loader = retort.get_loader(Decimal)

    assert loader("100") == Decimal("100")
    assert loader(Decimal("100")) == Decimal("100")
    raises_exc(TypeLoadError(Union[str, Decimal], None), lambda: loader(None))
    raises_exc(ValueLoadError("Bad string format", "foo"), lambda: loader("foo"))
    raises_exc(TypeLoadError(Union[str, Decimal], None), lambda: loader(None))

    if strict_coercion:
        raises_exc(TypeLoadError(Union[str, Decimal], []), lambda: loader([]))
    else:
        if IS_PYPY:
            description = (
                "Invalid tuple size in creation of Decimal from list or tuple."
                "  The list or tuple should have exactly three elements."
            )
        else:
            description = "argument must be a sequence of length 3"

        raises_exc(ValueLoadError(description, []), lambda: loader([]))


def test_fraction_loader_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )
    loader = retort.get_loader(Fraction)

    assert loader("100") == Fraction("100")
    assert loader(Fraction("100")) == Fraction("100")
    raises_exc(TypeLoadError(Union[str, Fraction], None), lambda: loader(None))
    raises_exc(ValueLoadError("Bad string format", "foo"), lambda: loader("foo"))
    raises_exc(TypeLoadError(Union[str, Fraction], None), lambda: loader(None))
    raises_exc(TypeLoadError(Union[str, Fraction], []), lambda: loader([]))


def test_complex_loader_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )
    loader = retort.get_loader(complex)

    assert loader("100") == complex("100")
    assert loader(complex("100")) == complex("100")
    raises_exc(TypeLoadError(Union[str, complex], None), lambda: loader(None))
    raises_exc(ValueLoadError("Bad string format", "foo"), lambda: loader("foo"))
    raises_exc(TypeLoadError(Union[str, complex], None), lambda: loader(None))
    raises_exc(TypeLoadError(Union[str, complex], []), lambda: loader([]))


def test_omittable_provider(strict_coercion, debug_trail):
    retort = Retort(
        strict_coercion=strict_coercion,
        debug_trail=debug_trail,
    )

    dumper = retort.get_dumper(Omittable[int])
    raises_exc(
        SentinelDumpError(Omitted),
        lambda: dumper(Omitted()),
    )
    loader = retort.get_loader(Omittable[int])
    assert loader(1) == 1

    if debug_trail == DebugTrail.DISABLE:
        raises_exc(
            LoadError(),
            lambda: loader(Omitted()),
        )
        raises_exc(
            LoadError(),
            lambda: loader([]),
        )
    elif debug_trail in (DebugTrail.FIRST, DebugTrail.ALL):
        if not strict_coercion:
            return
        raises_exc(
            UnionLoadError(
                f"while loading {Union[int, Omitted]}",
                [
                    ValueLoadError("Field value required", "100"),
                    TypeLoadError(int, "100"),
                ],
            ),
            lambda: loader("100"),
        )
