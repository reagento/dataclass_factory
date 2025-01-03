from abc import ABC, abstractmethod
from collections.abc import Container
from typing import final

from ..common import Dumper, Loader, TypeHint
from ..provider.essential import CannotProvide, Mediator
from ..provider.loc_stack_filtering import ExactOriginLSC
from ..provider.located_request import LocatedRequestMethodsProvider
from ..provider.methods_provider import method_handler
from ..type_tools import normalize_type
from .json_schema.definitions import JSONSchema
from .json_schema.request_cls import JSONSchemaRequest
from .json_schema.schema_model import JSONSchemaDialect
from .request_cls import DumperRequest, LoaderRequest


class LoaderProvider(LocatedRequestMethodsProvider, ABC):
    @method_handler
    @abstractmethod
    def provide_loader(self, mediator: Mediator[Loader], request: LoaderRequest) -> Loader:
        ...


class DumperProvider(LocatedRequestMethodsProvider, ABC):
    @method_handler
    @abstractmethod
    def provide_dumper(self, mediator: Mediator[Dumper], request: DumperRequest) -> Dumper:
        ...


class JSONSchemaProvider(LocatedRequestMethodsProvider, ABC):
    SUPPORTED_JSON_SCHEMA_DIALECTS: Container[str] = (JSONSchemaDialect.DRAFT_2020_12, )

    @final
    @method_handler
    def provide_json_schema(self, mediator: Mediator, request: JSONSchemaRequest) -> JSONSchema:
        if request.ctx.dialect not in self.SUPPORTED_JSON_SCHEMA_DIALECTS:
            raise CannotProvide(f"Dialect {request.ctx.dialect} is not supported for this type")
        return self._generate_json_schema(mediator, request)

    @abstractmethod
    def _generate_json_schema(self, mediator: Mediator, request: JSONSchemaRequest) -> JSONSchema:
        ...


class MorphingProvider(
    LoaderProvider,
    DumperProvider,
    JSONSchemaProvider,
    ABC,
):
    pass


class ABCProxy(LoaderProvider, DumperProvider):
    def __init__(self, abstract: TypeHint, impl: TypeHint, *, for_loader: bool = True, for_dumper: bool = True):
        self._abstract = normalize_type(abstract).origin
        self._impl = impl
        self._loc_stack_checker = ExactOriginLSC(self._abstract)
        self._for_loader = for_loader
        self._for_dumper = for_dumper

    def provide_loader(self, mediator: Mediator, request: LoaderRequest) -> Loader:
        if not self._for_loader:
            raise CannotProvide

        return mediator.mandatory_provide(
            LoaderRequest(
                loc_stack=request.loc_stack.replace_last_type(self._impl),
            ),
            lambda x: f"Cannot create loader for {self._abstract}. Loader for {self._impl} cannot be created",
        )

    def provide_dumper(self, mediator: Mediator, request: DumperRequest) -> Dumper:
        if not self._for_dumper:
            raise CannotProvide

        return mediator.mandatory_provide(
            DumperRequest(
                loc_stack=request.loc_stack.replace_last_type(self._impl),
            ),
            lambda x: f"Cannot create dumper for {self._abstract}. Dumper for {self._impl} cannot be created",
        )
