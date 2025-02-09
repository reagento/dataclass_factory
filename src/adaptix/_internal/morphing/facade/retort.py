from abc import ABC
from collections.abc import ByteString, Iterable, Mapping, MutableMapping  # noqa: PYI057
from datetime import date, datetime, time
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from itertools import chain
from pathlib import Path, PosixPath, PurePath, PurePosixPath, PureWindowsPath, WindowsPath
from typing import Any, Optional, TypeVar, overload
from uuid import UUID

from ...common import Dumper, Loader, TypeHint, VarTuple
from ...definitions import DebugTrail
from ...provider.essential import Provider, Request
from ...provider.loc_stack_filtering import LocStack, P
from ...provider.location import TypeHintLoc
from ...provider.shape_provider import BUILTIN_SHAPE_PROVIDER
from ...provider.value_provider import ValueProvider
from ...retort.operating_retort import OperatingRetort
from ...struct_trail import render_trail_as_note
from ...type_tools.basic_utils import is_generic_class
from ..concrete_provider import (
    BOOL_PROVIDER,
    COMPLEX_PROVIDER,
    DECIMAL_PROVIDER,
    FLOAT_PROVIDER,
    FRACTION_PROVIDER,
    INT_PROVIDER,
    STR_PROVIDER,
    BytearrayBase64Provider,
    BytesBase64Provider,
    BytesIOBase64Provider,
    IOBytesBase64Provider,
    IsoFormatProvider,
    LiteralStringProvider,
    NoneProvider,
    OmittedProvider,
    RegexPatternProvider,
    SecondsTimedeltaProvider,
    SelfTypeProvider,
)
from ..constant_length_tuple_provider import ConstantLengthTupleProvider
from ..dict_provider import DefaultDictProvider, DictProvider
from ..generic_provider import (
    ForwardRefEvaluatingProvider,
    LiteralProvider,
    NewTypeUnwrappingProvider,
    PathLikeProvider,
    TypeAliasUnwrappingProvider,
    TypeHintTagsUnwrappingProvider,
    UnionProvider,
)
from ..iterable_provider import IterableProvider
from ..json_schema.providers import InlineJSONSchemaProvider, JSONSchemaRefProvider
from ..model.crown_definitions import ExtraSkip
from ..model.dumper_provider import ModelDumperProvider
from ..model.loader_provider import ModelLoaderProvider
from ..model.request_filtering import AnyModelLSC
from ..name_layout.component import BuiltinExtraMoveAndPoliciesMaker, BuiltinSievesMaker, BuiltinStructureMaker
from ..name_layout.name_mapping import SkipPrivateFieldsNameMappingProvider
from ..name_layout.provider import BuiltinNameLayoutProvider
from ..provider_template import ABCProxy
from ..request_cls import DebugTrailRequest, DumperRequest, LoaderRequest, StrictCoercionRequest
from .provider import (
    as_is_dumper,
    as_is_loader,
    bound,
    dumper,
    enum_by_exact_value,
    flag_by_exact_value,
    loader,
    name_mapping,
)


class FilledRetort(OperatingRetort, ABC):
    """A retort contains builtin providers"""

    recipe = [
        NoneProvider(),

        as_is_loader(Any),
        as_is_dumper(Any),
        as_is_loader(object),
        as_is_dumper(object),

        IsoFormatProvider(datetime),
        IsoFormatProvider(date),
        IsoFormatProvider(time),
        SecondsTimedeltaProvider(),

        flag_by_exact_value(),
        enum_by_exact_value(),  # it has higher priority than scalar types for Enum with mixins

        INT_PROVIDER,
        FLOAT_PROVIDER,
        STR_PROVIDER,
        BOOL_PROVIDER,
        DECIMAL_PROVIDER,
        FRACTION_PROVIDER,
        COMPLEX_PROVIDER,

        BytesBase64Provider(),
        BytesIOBase64Provider(),
        IOBytesBase64Provider(),
        BytearrayBase64Provider(),

        *chain.from_iterable(
            (
                loader(tp, tp),
                dumper(tp, tp.__str__),  # type: ignore[arg-type]
            )
            for tp in [
                UUID,
                IPv4Address, IPv6Address,
                IPv4Network, IPv6Network,
                IPv4Interface, IPv6Interface,
            ]
        ),
        *chain.from_iterable(
            (
                loader(tp, tp),
                dumper(tp, tp.__fspath__),  # type: ignore[attr-defined]
            )
            for tp in [
                PurePath, Path,
                PurePosixPath, PosixPath,
                PureWindowsPath, WindowsPath,
            ]
        ),
        PathLikeProvider(),

        LiteralProvider(),
        UnionProvider(),
        ConstantLengthTupleProvider(),
        IterableProvider(),
        DictProvider(),
        DefaultDictProvider(),
        RegexPatternProvider(),
        SelfTypeProvider(),
        LiteralStringProvider(),
        OmittedProvider(),

        ABCProxy(Mapping, dict),
        ABCProxy(MutableMapping, dict),
        ABCProxy(ByteString, bytes),

        name_mapping(
            chain=None,
            skip=(),
            only=P.ANY,
            map=[
                SkipPrivateFieldsNameMappingProvider(),
            ],
            trim_trailing_underscore=True,
            name_style=None,
            as_list=False,
            omit_default=False,
            extra_in=ExtraSkip(),
            extra_out=ExtraSkip(),
        ),
        BuiltinNameLayoutProvider(
            structure_maker=BuiltinStructureMaker(),
            sieves_maker=BuiltinSievesMaker(),
            extra_move_maker=BuiltinExtraMoveAndPoliciesMaker(),
            extra_policies_maker=BuiltinExtraMoveAndPoliciesMaker(),
        ),
        ModelLoaderProvider(),
        ModelDumperProvider(),

        bound(AnyModelLSC(), InlineJSONSchemaProvider(inline=False)),
        InlineJSONSchemaProvider(inline=True),
        JSONSchemaRefProvider(),

        BUILTIN_SHAPE_PROVIDER,

        NewTypeUnwrappingProvider(),
        TypeHintTagsUnwrappingProvider(),
        TypeAliasUnwrappingProvider(),
        ForwardRefEvaluatingProvider(),
    ]


T = TypeVar("T")
RequestT = TypeVar("RequestT", bound=Request)
AR = TypeVar("AR", bound="AdornedRetort")


class AdornedRetort(OperatingRetort):
    """A retort implementing high-level user interface"""

    def __init__(
        self,
        *,
        recipe: Iterable[Provider] = (),
        strict_coercion: bool = True,
        debug_trail: DebugTrail = DebugTrail.ALL,
        hide_traceback: bool = True,
    ):
        self._strict_coercion = strict_coercion
        self._debug_trail = debug_trail
        super().__init__(recipe=recipe, hide_traceback=hide_traceback)

    def _calculate_derived(self):
        super()._calculate_derived()
        self._loader_cache = {}
        self._dumper_cache = {}

    def replace(
        self: AR,
        *,
        strict_coercion: Optional[bool] = None,
        debug_trail: Optional[DebugTrail] = None,
        hide_traceback: Optional[bool] = None,
    ) -> AR:
        with self._clone() as clone:
            if strict_coercion is not None:
                clone._strict_coercion = strict_coercion
            if debug_trail is not None:
                clone._debug_trail = debug_trail
            if hide_traceback is not None:
                clone._hide_traceback = hide_traceback
        return clone

    def extend(self: AR, *, recipe: Iterable[Provider]) -> AR:
        with self._clone() as clone:
            clone._instance_recipe = (
                tuple(recipe) + clone._instance_recipe
            )

        return clone

    def _get_recipe_tail(self) -> VarTuple[Provider]:
        return (
            ValueProvider(StrictCoercionRequest, self._strict_coercion),
            ValueProvider(DebugTrailRequest, self._debug_trail),
        )

    def get_loader(self, tp: type[T]) -> Loader[T]:
        try:
            return self._loader_cache[tp]
        except KeyError:
            pass
        loader_ = self._make_loader(tp)
        self._loader_cache[tp] = loader_
        return loader_

    def _make_loader(self, tp: type[T]) -> Loader[T]:
        loader_ = self._facade_provide(
            LoaderRequest(loc_stack=LocStack(TypeHintLoc(type=tp))),
            error_message=f"Cannot produce loader for type {tp!r}",
        )
        if self._debug_trail == DebugTrail.FIRST:
            def trail_rendering_wrapper(data):
                try:
                    return loader_(data)
                except Exception as e:
                    render_trail_as_note(e)
                    raise

            return trail_rendering_wrapper

        return loader_

    def get_dumper(self, tp: type[T]) -> Dumper[T]:
        try:
            return self._dumper_cache[tp]
        except KeyError:
            pass
        dumper_ = self._make_dumper(tp)
        self._dumper_cache[tp] = dumper_
        return dumper_

    def _make_dumper(self, tp: type[T]) -> Dumper[T]:
        dumper_ = self._facade_provide(
            DumperRequest(loc_stack=LocStack(TypeHintLoc(type=tp))),
            error_message=f"Cannot produce dumper for type {tp!r}",
        )
        if self._debug_trail == DebugTrail.FIRST:
            def trail_rendering_wrapper(data):
                try:
                    return dumper_(data)
                except Exception as e:
                    render_trail_as_note(e)
                    raise

            return trail_rendering_wrapper

        return dumper_

    @overload
    def load(self, data: Any, tp: type[T], /) -> T:
        ...

    @overload
    def load(self, data: Any, tp: TypeHint, /) -> Any:
        ...

    def load(self, data: Any, tp: TypeHint, /):
        return self.get_loader(tp)(data)

    @overload
    def dump(self, data: T, tp: type[T], /) -> Any:
        ...

    @overload
    def dump(self, data: Any, tp: Optional[TypeHint] = None, /) -> Any:
        ...

    def dump(self, data: Any, tp: Optional[TypeHint] = None, /) -> Any:
        if tp is None:
            tp = type(data)
            if is_generic_class(tp):
                raise ValueError(
                    f"Can not infer the actual type of generic class instance ({tp!r}),"
                    " you have to explicitly pass the type of object",
                )
        return self.get_dumper(tp)(data)


class Retort(FilledRetort, AdornedRetort):
    pass
