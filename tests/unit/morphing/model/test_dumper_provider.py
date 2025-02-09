# ruff: noqa: FBT003, PT011
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Dict, Optional, Type
from unittest.mock import ANY

import pytest
from tests_helpers import DebugCtx, full_match, parametrize_bool, raises_exc, with_trail

from adaptix import DebugTrail, Dumper, Retort, bound
from adaptix._internal.common import Catchable
from adaptix._internal.compat import CompatExceptionGroup
from adaptix._internal.model_tools.definitions import (
    Accessor,
    DefaultFactory,
    DefaultValue,
    NoDefault,
    OutputField,
    OutputShape,
    create_attr_accessor,
    create_key_accessor,
)
from adaptix._internal.morphing.model.crown_definitions import (
    ExtraExtract,
    ExtraTargets,
    OutDictCrown,
    OutFieldCrown,
    OutListCrown,
    OutNoneCrown,
    OutputNameLayout,
    OutputNameLayoutRequest,
)
from adaptix._internal.morphing.model.dumper_provider import ModelDumperProvider
from adaptix._internal.morphing.request_cls import DumperRequest
from adaptix._internal.provider.shape_provider import OutputShapeRequest
from adaptix._internal.provider.value_provider import ValueProvider
from adaptix._internal.struct_trail import Attr, TrailElement, TrailElementMarker
from adaptix._internal.utils import SingletonMeta


@dataclass
class TestField:
    id: str
    accessor: Accessor


def shape(*fields: TestField):
    return OutputShape(
        fields=tuple(
            OutputField(
                type=int,
                id=fld.id,
                default=NoDefault(),
                accessor=fld.accessor,
                metadata=MappingProxyType({}),
                original=None,
            )
            for fld in fields
        ),
        overriden_types=frozenset(fld.id for fld in fields),
    )


def int_dumper(data):
    if isinstance(data, BaseException):
        raise data
    return data


class Dummy:
    def __init__(self, items: Optional[Dict[str, Any]] = None, **kwargs: Any):
        if items is None:
            items = {}

        self.items = items

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self.items[item]


def dummy_items(**kwargs: Any):
    return Dummy(items=kwargs)


def make_dumper_getter(
    shape: OutputShape,
    name_layout: OutputNameLayout,
    debug_trail: DebugTrail,
    debug_ctx: DebugCtx,
) -> Callable[[], Dumper]:
    def getter():
        retort = Retort(
            recipe=[
                ValueProvider(OutputShapeRequest, shape),
                ValueProvider(OutputNameLayoutRequest, name_layout),
                bound(int, ValueProvider(DumperRequest, int_dumper)),
                ModelDumperProvider(),
                debug_ctx.accum,
            ],
        )
        return retort.replace(
            debug_trail=debug_trail,
        ).get_dumper(
            Dummy,
        )

    return getter


class Skip(metaclass=SingletonMeta):
    pass


def skipper(value):
    return value != Skip()


@dataclass(eq=False)
class SomeError(Exception):
    value: int = 0


def stub(value):
    return value


@dataclass
class AccessSchema:
    dummy: Callable
    accessor_maker: Callable[[str, bool], Accessor]
    access_error: Type[Exception]
    trail_element_maker: Callable[[str], Any]


class MyAccessError(Exception):
    pass


@dataclass(frozen=True)
class MyTrailElemMarker(TrailElementMarker):
    value: Any


@dataclass
class MyAccessor(Accessor):
    value: Any
    is_required: bool

    @property
    def getter(self) -> Callable[[Any], Any]:
        def my_getter(obj):
            try:
                return getattr(obj, self.value)
            except AttributeError as e:
                raise MyAccessError(*e.args)

        return my_getter

    @property
    def access_error(self) -> Optional[Catchable]:
        return None if self.is_required else MyAccessError

    @property
    def trail_element(self) -> TrailElement:
        return MyTrailElemMarker(self.value)

    def __hash__(self) -> int:
        return hash(self.value)


def make_str_item_accessor(name: str, is_required: bool):  # noqa: FBT001
    return create_key_accessor(
        key=name,
        access_error=None if is_required else KeyError,
    )


@pytest.fixture(
    params=[
        AccessSchema(
            dummy=Dummy,
            accessor_maker=lambda name, is_required: create_attr_accessor(name, is_required=is_required),
            access_error=AttributeError,
            trail_element_maker=Attr,
        ),
        AccessSchema(
            dummy=dummy_items,
            accessor_maker=make_str_item_accessor,
            access_error=KeyError,
            trail_element_maker=stub,
        ),
        AccessSchema(
            dummy=Dummy,
            accessor_maker=MyAccessor,
            access_error=MyAccessError,
            trail_element_maker=MyTrailElemMarker,
        ),
    ],
    ids=["attrs", "items", "custom"],
)
def acc_schema(request):
    return request.param


@parametrize_bool("is_required_a", "is_required_b")
def test_flat(debug_ctx, debug_trail, trail_select, is_required_a, is_required_b, acc_schema):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required_a)),
            TestField("b", acc_schema.accessor_maker("b", is_required_b)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "a": OutFieldCrown("a"),
                    "b": OutFieldCrown("b"),
                },
                sieves={
                    "b": skipper,
                },
            ),
            extra_move=None,
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )

    dumper = dumper_getter()

    assert dumper(acc_schema.dummy(a=1, b=2)) == {"a": 1, "b": 2}
    assert dumper(acc_schema.dummy(a=1, b=2, c=3)) == {"a": 1, "b": 2}
    assert dumper(acc_schema.dummy(a=1, b=Skip())) == {"a": 1}
    assert dumper(acc_schema.dummy(a=1, b=Skip(), c=3)) == {"a": 1}

    assert dumper(acc_schema.dummy(a=Skip(), b=2)) == {"a": Skip(), "b": 2}
    assert dumper(acc_schema.dummy(a=Skip(), b=2, c=3)) == {"a": Skip(), "b": 2}
    assert dumper(acc_schema.dummy(a=Skip(), b=Skip())) == {"a": Skip()}
    assert dumper(acc_schema.dummy(a=Skip(), b=Skip(), c=3)) == {"a": Skip()}

    if is_required_a:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("a")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("a")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(b=1)),
        )

    if is_required_b:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("b")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("b")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(a=1)),
        )

    if is_required_a and is_required_b:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("a")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("a")],
                        ),
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("b")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy()),
        )

    if not is_required_a:
        assert dumper(acc_schema.dummy(b=1)) == {"b": 1}
        assert dumper(acc_schema.dummy(b=Skip())) == {}

    if not is_required_b:
        assert dumper(acc_schema.dummy(a=1)) == {"a": 1}
        assert dumper(acc_schema.dummy(a=Skip())) == {"a": Skip()}

    if not is_required_a and not is_required_b:
        assert dumper(acc_schema.dummy()) == {}

    raises_exc(
        trail_select(
            disable=SomeError(),
            first=with_trail(
                SomeError(),
                [acc_schema.trail_element_maker("a")],
            ),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(
                        SomeError(),
                        [acc_schema.trail_element_maker("a")],
                    ),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(), b=Skip())),
    )

    raises_exc(
        trail_select(
            disable=SomeError(),
            first=with_trail(
                SomeError(),
                [acc_schema.trail_element_maker("b")],
            ),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(
                        SomeError(),
                        [acc_schema.trail_element_maker("b")],
                    ),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=1, b=SomeError())),
    )

    raises_exc(
        trail_select(
            disable=SomeError(0),
            first=with_trail(
                SomeError(0),
                [acc_schema.trail_element_maker("a")],
            ),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(
                        SomeError(0),
                        [acc_schema.trail_element_maker("a")],
                    ),
                    with_trail(
                        SomeError(1),
                        [acc_schema.trail_element_maker("b")],
                    ),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(0), b=SomeError(1))),
    )


def test_wild_extra_targets(debug_ctx, debug_trail, acc_schema):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required=True)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "a": OutFieldCrown("a"),
                },
                sieves={},
            ),
            extra_move=ExtraTargets(("b",)),
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )

    pytest.raises(ValueError, dumper_getter).match(
        full_match("ExtraTargets ['b'] are attached to non-existing fields"),
    )


@parametrize_bool("is_required_a", "is_required_b")
def test_one_extra_target(debug_ctx, debug_trail, trail_select, is_required_a, is_required_b, acc_schema):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required=is_required_a)),
            TestField("b", acc_schema.accessor_maker("b", is_required=is_required_b)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "a": OutFieldCrown("a"),
                },
                sieves={},
            ),
            extra_move=ExtraTargets(("b",)),
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )
    dumper = dumper_getter()

    assert dumper(acc_schema.dummy(a=1, b={"e": 2})) == {"a": 1, "e": 2}
    assert dumper(acc_schema.dummy(a=1, b={"b": 2})) == {"a": 1, "b": 2}

    if is_required_a:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("a")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("a")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(b=1)),
        )

    if is_required_b:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("b")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("b")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(a=1)),
        )

    if is_required_a and is_required_b:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("a")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("a")],
                        ),
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("b")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy()),
        )

    if not is_required_a:
        assert dumper(acc_schema.dummy(b={"f": 2})) == {"f": 2}

    if not is_required_b:
        assert dumper(acc_schema.dummy(a=1)) == {"a": 1}

    if not is_required_a and not is_required_b:
        assert dumper(acc_schema.dummy()) == {}

    raises_exc(
        trail_select(
            disable=SomeError(),
            first=with_trail(
                SomeError(),
                [acc_schema.trail_element_maker("a")],
            ),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(
                        SomeError(),
                        [acc_schema.trail_element_maker("a")],
                    ),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(), b=Skip())),
    )

    raises_exc(
        trail_select(
            disable=SomeError(),
            first=with_trail(
                SomeError(),
                [acc_schema.trail_element_maker("b")],
            ),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(
                        SomeError(),
                        [acc_schema.trail_element_maker("b")],
                    ),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=1, b=SomeError())),
    )

    raises_exc(
        trail_select(
            disable=SomeError(0),
            first=with_trail(
                SomeError(0),
                [acc_schema.trail_element_maker("a")],
            ),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(
                        SomeError(0),
                        [acc_schema.trail_element_maker("a")],
                    ),
                    with_trail(
                        SomeError(1),
                        [acc_schema.trail_element_maker("b")],
                    ),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(0), b=SomeError(1))),
    )


@parametrize_bool("is_required_a", "is_required_b", "is_required_c", "is_required_d")
def test_several_extra_target(
    debug_ctx, debug_trail, trail_select, is_required_a, is_required_b, is_required_c, is_required_d, acc_schema,
):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required=is_required_a)),
            TestField("b", acc_schema.accessor_maker("b", is_required=is_required_b)),
            TestField("c", acc_schema.accessor_maker("c", is_required=is_required_c)),
            TestField("d", acc_schema.accessor_maker("d", is_required=is_required_d)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "a": OutFieldCrown("a"),
                },
                sieves={},
            ),
            extra_move=ExtraTargets(("b", "c", "d")),
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )
    dumper = dumper_getter()

    assert (
        dumper(acc_schema.dummy(a=1, b={"b1": 2}, c={"c1": 3}, d={"d1": 4}))
        ==
        {"a": 1, "b1": 2, "c1": 3, "d1": 4}
    )
    assert (
        dumper(acc_schema.dummy(a=1, b={"b1": 2, "b2": 3}, c={"c1": 4, "c2": 5}, d={"d1": 6, "d2": 7}))
        ==
        {"a": 1, "b1": 2, "b2": 3, "c1": 4, "c2": 5, "d1": 6, "d2": 7}
    )
    assert dumper(acc_schema.dummy(a=1, b={"d": 2}, c={"e": 3}, d={})) == {"a": 1, "d": 2, "e": 3}

    assert (
        dumper(acc_schema.dummy(a=1, b={"b1": 2, "b2": 3}, c={"c1": 4, "b2": 5}, d={}))
        ==
        {"a": 1, "b1": 2, "c1": 4, "b2": 5}
    )

    if is_required_b:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("b")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("b")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(a=1, c={"c1": 2}, d={"d1": 3})),
        )

    if is_required_c:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("c")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("c")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(a=1, b={"b1": 2}, d={"d1": 3})),
        )

    if is_required_b and is_required_c and is_required_d:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("b")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("b")],
                        ),
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("c")],
                        ),
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("d")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(a=1)),
        )

    requirement = {
        "b": is_required_b,
        "c": is_required_c,
        "d": is_required_d,
    }

    assert (
        dumper(acc_schema.dummy(a=1, **{k: {k: 1} for k, v in requirement.items() if v}))
        ==
        {"a": 1, **{k: 1 for k, v in requirement.items() if v}}
    )


def my_extractor(obj):
    try:
        return int_dumper(obj.b)
    except AttributeError:
        try:
            return int_dumper(obj["b"])
        except KeyError:
            return {}


@parametrize_bool("is_required_a")
def test_extra_extract(debug_ctx, debug_trail, trail_select, is_required_a, acc_schema):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required=is_required_a)),
            TestField("b", acc_schema.accessor_maker("b", is_required=True)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "a": OutFieldCrown("a"),
                },
                sieves={},
            ),
            extra_move=ExtraExtract(my_extractor),
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )
    dumper = dumper_getter()

    assert dumper(acc_schema.dummy(a=1, b={"e": 2})) == {"a": 1, "e": 2}
    assert dumper(acc_schema.dummy(a=1, b={"b": 2})) == {"a": 1, "b": 2}

    if not is_required_a:
        assert dumper(acc_schema.dummy(b={"f": 2})) == {"f": 2}
        assert dumper(acc_schema.dummy()) == {}

    assert dumper(acc_schema.dummy(a=1)) == {"a": 1}

    if is_required_a:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("a")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("a")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy()),
        )
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(
                    acc_schema.access_error(ANY),
                    [acc_schema.trail_element_maker("a")],
                ),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(
                            acc_schema.access_error(ANY),
                            [acc_schema.trail_element_maker("a")],
                        ),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(b=1)),
        )

    raises_exc(
        trail_select(
            disable=SomeError(),
            first=SomeError(),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [SomeError()],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=1, b=SomeError())),
    )

    raises_exc(
        trail_select(
            disable=SomeError(0),
            first=with_trail(
                SomeError(0),
                [acc_schema.trail_element_maker("a")],
            ),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(
                        SomeError(0),
                        [acc_schema.trail_element_maker("a")],
                    ),
                    SomeError(1),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(0), b=SomeError(1))),
    )


def test_optional_fields_at_list(debug_ctx, debug_trail, acc_schema):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required=True)),
            TestField("b", acc_schema.accessor_maker("b", is_required=False)),
        ),
        name_layout=OutputNameLayout(
            crown=OutListCrown(
                (
                    OutFieldCrown("a"),
                    OutFieldCrown("b"),
                ),
            ),
            extra_move=None,
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )

    pytest.raises(ValueError, dumper_getter).match(
        full_match("Optional fields ['b'] are found at list crown"),
    )


@dataclass
class FlatMap:
    field: str
    mapped: str


@parametrize_bool("is_required_a", "is_required_b")
@pytest.mark.parametrize(
    "mp", [
        SimpleNamespace(a=FlatMap("a", "a"), b=FlatMap("b", "b")),
        SimpleNamespace(a=FlatMap("a", "m_a"), b=FlatMap("b", "m_b")),
    ],
    ids=["as_is", "diff"],
)
def test_flat_mapping(debug_ctx, debug_trail, trail_select, is_required_a, is_required_b, acc_schema, mp):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField(mp.a.field, acc_schema.accessor_maker("a", is_required_a)),
            TestField(mp.b.field, acc_schema.accessor_maker("b", is_required_b)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    mp.a.mapped: OutFieldCrown(mp.a.field),
                    mp.b.mapped: OutFieldCrown(mp.b.field),
                },
                sieves={
                    mp.b.mapped: skipper,
                },
            ),
            extra_move=None,
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )

    dumper = dumper_getter()

    assert dumper(acc_schema.dummy(a=1, b=2)) == {mp.a.mapped: 1, mp.b.mapped: 2}
    assert dumper(acc_schema.dummy(a=1, b=2, c=3)) == {mp.a.mapped: 1, mp.b.mapped: 2}
    assert dumper(acc_schema.dummy(a=1, b=Skip())) == {mp.a.mapped: 1}
    assert dumper(acc_schema.dummy(a=1, b=Skip(), c=3)) == {mp.a.mapped: 1}

    assert dumper(acc_schema.dummy(a=Skip(), b=2)) == {mp.a.mapped: Skip(), mp.b.mapped: 2}
    assert dumper(acc_schema.dummy(a=Skip(), b=2, c=3)) == {mp.a.mapped: Skip(), mp.b.mapped: 2}
    assert dumper(acc_schema.dummy(a=Skip(), b=Skip())) == {mp.a.mapped: Skip()}
    assert dumper(acc_schema.dummy(a=Skip(), b=Skip(), c=3)) == {mp.a.mapped: Skip()}

    if is_required_a:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(acc_schema.access_error(ANY), [acc_schema.trail_element_maker(mp.a.field)]),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(acc_schema.access_error(ANY), [acc_schema.trail_element_maker(mp.a.field)]),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(b=1)),
        )

    if is_required_b:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(acc_schema.access_error(ANY), [acc_schema.trail_element_maker(mp.b.field)]),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(acc_schema.access_error(ANY), [acc_schema.trail_element_maker(mp.b.field)]),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy(a=1)),
        )

    if is_required_a and is_required_b:
        raises_exc(
            trail_select(
                disable=acc_schema.access_error(ANY),
                first=with_trail(acc_schema.access_error(ANY), [acc_schema.trail_element_maker(mp.a.field)]),
                all=CompatExceptionGroup(
                    f"while dumping model {Dummy}",
                    [
                        with_trail(acc_schema.access_error(ANY), [acc_schema.trail_element_maker(mp.a.field)]),
                        with_trail(acc_schema.access_error(ANY), [acc_schema.trail_element_maker(mp.b.field)]),
                    ],
                ),
            ),
            lambda: dumper(acc_schema.dummy()),
        )

    if not is_required_a:
        assert dumper(acc_schema.dummy(b=1)) == {mp.b.mapped: 1}
        assert dumper(acc_schema.dummy(b=Skip())) == {}

    if not is_required_b:
        assert dumper(acc_schema.dummy(a=1)) == {mp.a.mapped: 1}
        assert dumper(acc_schema.dummy(a=Skip())) == {mp.a.mapped: Skip()}

    if not is_required_a and not is_required_b:
        assert dumper(acc_schema.dummy()) == {}

    raises_exc(
        trail_select(
            disable=SomeError(),
            first=with_trail(SomeError(), [acc_schema.trail_element_maker(mp.a.field)]),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(SomeError(), [acc_schema.trail_element_maker(mp.a.field)]),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(), b=Skip())),
    )

    raises_exc(
        trail_select(
            disable=SomeError(),
            first=with_trail(SomeError(), [acc_schema.trail_element_maker(mp.b.field)]),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(SomeError(), [acc_schema.trail_element_maker(mp.b.field)]),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=1, b=SomeError())),
    )

    raises_exc(
        trail_select(
            disable=SomeError(0),
            first=with_trail(SomeError(0), [acc_schema.trail_element_maker(mp.a.field)]),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(SomeError(0), [acc_schema.trail_element_maker(mp.a.field)]),
                    with_trail(SomeError(1), [acc_schema.trail_element_maker(mp.b.field)]),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(0), b=SomeError(1))),
    )


def test_direct_list(debug_ctx, debug_trail, trail_select, acc_schema):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", True)),
            TestField("b", acc_schema.accessor_maker("b", True)),
        ),
        name_layout=OutputNameLayout(
            crown=OutListCrown(
                (
                    OutFieldCrown("a"),
                    OutFieldCrown("b"),
                ),
            ),
            extra_move=None,
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )

    dumper = dumper_getter()
    assert dumper(acc_schema.dummy(a=1, b=2)) == [1, 2]

    raises_exc(
        trail_select(
            disable=SomeError(1),
            first=with_trail(SomeError(1), [acc_schema.trail_element_maker("a")]),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(SomeError(1), [acc_schema.trail_element_maker("a")]),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(1), b=2)),
    )

    raises_exc(
        trail_select(
            disable=SomeError(1),
            first=with_trail(SomeError(1), [acc_schema.trail_element_maker("a")]),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(SomeError(1), [acc_schema.trail_element_maker("a")]),
                    with_trail(SomeError(2), [acc_schema.trail_element_maker("b")]),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=SomeError(1), b=SomeError(2))),
    )

    raises_exc(
        trail_select(
            disable=SomeError(2),
            first=with_trail(SomeError(2), [acc_schema.trail_element_maker("b")]),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(SomeError(2), [acc_schema.trail_element_maker("b")]),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=1, b=SomeError(2))),
    )


def dict_skipper(data):
    return Skip() not in data.values()


def list_skipper(data):
    return Skip() not in data


def test_structure_flattening(debug_ctx, debug_trail, trail_select, acc_schema):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", True)),
            TestField("b", acc_schema.accessor_maker("b", True)),
            TestField("c", acc_schema.accessor_maker("c", True)),
            TestField("d", acc_schema.accessor_maker("d", True)),
            TestField("e", acc_schema.accessor_maker("e", True)),
            TestField("f", acc_schema.accessor_maker("f", True)),
            TestField("g", acc_schema.accessor_maker("g", True)),
            TestField("h", acc_schema.accessor_maker("h", True)),
            TestField("extra", acc_schema.accessor_maker("extra", True)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "z": OutDictCrown(
                        {
                            "y": OutFieldCrown("a"),
                            "x": OutFieldCrown("b"),
                        },
                        sieves={},
                    ),
                    "w": OutFieldCrown("c"),
                    "v": OutListCrown(
                        (
                            OutFieldCrown("d"),
                            OutDictCrown(
                                {
                                    "u": OutFieldCrown("e"),
                                },
                                sieves={},
                            ),
                            OutListCrown(
                                (
                                    OutFieldCrown("f"),
                                ),
                            ),
                        ),
                    ),
                    "t": OutDictCrown(
                        {
                            "s": OutFieldCrown("g"),
                        },
                        sieves={},
                    ),
                    "r": OutListCrown(
                        (
                            OutFieldCrown("h"),
                        ),
                    ),
                },
                sieves={
                    "t": dict_skipper,
                    "r": list_skipper,
                },
            ),
            extra_move=ExtraTargets(("extra",)),
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )
    dumper = dumper_getter()

    assert dumper(
        acc_schema.dummy(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, extra={}),
    ) == {
        "z": {
            "y": 1,
            "x": 2,
        },
        "w": 3,
        "v": [
            4,
            {"u": 5},
            [6],
        ],
        "t": {
            "s": 7,
        },
        "r": [
            8,
        ],
    }

    assert dumper(
        acc_schema.dummy(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, extra={"i": 9}),
    ) == {
        "z": {
            "y": 1,
            "x": 2,
        },
        "w": 3,
        "v": [
            4,
            {"u": 5},
            [6],
        ],
        "t": {
            "s": 7,
        },
        "r": [
            8,
        ],
        "i": 9,
    }

    assert dumper(
        acc_schema.dummy(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=Skip(), extra={}),
    ) == {
        "z": {
            "y": 1,
            "x": 2,
        },
        "w": 3,
        "v": [
            4,
            {"u": 5},
            [6],
        ],
        "t": {
            "s": 7,
        },
    }

    assert dumper(
        acc_schema.dummy(a=1, b=2, c=3, d=4, e=5, f=6, g=Skip(), h=8, extra={}),
    ) == {
        "z": {
            "y": 1,
            "x": 2,
        },
        "w": 3,
        "v": [
            4,
            {"u": 5},
            [6],
        ],
        "r": [
            8,
        ],
    }

    assert dumper(
        acc_schema.dummy(a=1, b=2, c=3, d=4, e=5, f=6, g=Skip(), h=Skip(), extra={}),
    ) == {
        "z": {
            "y": 1,
            "x": 2,
        },
        "w": 3,
        "v": [
            4,
            {"u": 5},
            [6],
        ],
    }

    assert dumper(
        acc_schema.dummy(a=1, b=2, c=3, d=4, e=5, f=6, g=Skip(), h=Skip(), extra={"v": "foo"}),
    ) == {
        "z": {
            "y": 1,
            "x": 2,
        },
        "w": 3,
        "v": "foo",  # sorry, merging is not implemented
    }

    raises_exc(
        trail_select(
            disable=SomeError(5),
            first=with_trail(SomeError(5), [acc_schema.trail_element_maker("e")]),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(SomeError(5), [acc_schema.trail_element_maker("e")]),
                ],
            ),
        ),
        lambda: dumper(acc_schema.dummy(a=1, b=2, c=3, d=4, e=SomeError(5), f=6, g=Skip(), h=Skip(), extra={})),
    )

    raises_exc(
        trail_select(
            disable=SomeError(1),
            first=with_trail(SomeError(1), [acc_schema.trail_element_maker("a")]),
            all=CompatExceptionGroup(
                f"while dumping model {Dummy}",
                [
                    with_trail(SomeError(1), [acc_schema.trail_element_maker("a")]),
                    with_trail(SomeError(2), [acc_schema.trail_element_maker("b")]),
                    with_trail(SomeError(3), [acc_schema.trail_element_maker("c")]),
                    with_trail(SomeError(4), [acc_schema.trail_element_maker("d")]),
                    with_trail(SomeError(5), [acc_schema.trail_element_maker("e")]),
                    with_trail(SomeError(6), [acc_schema.trail_element_maker("f")]),
                    with_trail(SomeError(7), [acc_schema.trail_element_maker("g")]),
                    with_trail(SomeError(8), [acc_schema.trail_element_maker("h")]),
                    with_trail(SomeError(9), [acc_schema.trail_element_maker("extra")]),
                ],
            ),
        ),
        lambda: dumper(
            acc_schema.dummy(
                a=SomeError(1),
                b=SomeError(2),
                c=SomeError(3),
                d=SomeError(4),
                e=SomeError(5),
                f=SomeError(6),
                g=SomeError(7),
                h=SomeError(8),
                extra=SomeError(9),
            ),
        ),
    )


@parametrize_bool("is_required_a", "is_required_b")
def test_extra_target_at_crown(debug_ctx, debug_trail, acc_schema, is_required_a, is_required_b):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required_a)),
            TestField("b", acc_schema.accessor_maker("b", is_required_b)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "m_a": OutFieldCrown("a"),
                    "m_b": OutFieldCrown("b"),
                },
                sieves={},
            ),
            extra_move=ExtraTargets(("b",)),
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )
    pytest.raises(ValueError, dumper_getter).match(
        full_match("Extra targets ['b'] are found at crown"),
    )

    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required_a)),
            TestField("b", acc_schema.accessor_maker("b", is_required_b)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "m_a": OutFieldCrown("a"),
                    "m_b": OutFieldCrown("b"),
                },
                sieves={},
            ),
            extra_move=ExtraTargets(("b",)),
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )
    pytest.raises(ValueError, dumper_getter).match(
        full_match("Extra targets ['b'] are found at crown"),
    )


@dataclass
class SomeClass:
    value: int


@parametrize_bool("is_required_a")
def test_none_crown_at_dict_crown(debug_ctx, debug_trail, acc_schema, is_required_a):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", is_required_a)),
        ),
        name_layout=OutputNameLayout(
            crown=OutDictCrown(
                {
                    "w": OutNoneCrown(placeholder=DefaultValue(None)),
                    "x": OutNoneCrown(placeholder=DefaultValue(SomeClass(2))),
                    "y": OutFieldCrown("a"),
                    "z": OutNoneCrown(placeholder=DefaultFactory(list)),
                },
                sieves={},
            ),
            extra_move=None,
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )
    dumper = dumper_getter()

    assert dumper(acc_schema.dummy(a=1)) == {"w": None, "x": SomeClass(2), "y": 1, "z": []}


def test_none_crown_at_list_crown(debug_ctx, debug_trail, acc_schema):
    dumper_getter = make_dumper_getter(
        shape=shape(
            TestField("a", acc_schema.accessor_maker("a", True)),
        ),
        name_layout=OutputNameLayout(
            crown=OutListCrown(
                (
                    OutNoneCrown(placeholder=DefaultValue(None)),
                    OutNoneCrown(placeholder=DefaultValue(SomeClass(2))),
                    OutFieldCrown("a"),
                    OutNoneCrown(placeholder=DefaultFactory(list)),
                ),
            ),
            extra_move=None,
        ),
        debug_trail=debug_trail,
        debug_ctx=debug_ctx,
    )
    dumper = dumper_getter()

    assert dumper(acc_schema.dummy(a=1)) == [None, SomeClass(2), 1, []]
