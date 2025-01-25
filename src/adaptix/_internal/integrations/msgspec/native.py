from typing import Any, Callable, Iterable, Mapping, TypeVar

from adaptix import Dumper, Loader, Mediator, Omittable, Omitted, Provider
from adaptix._internal.morphing.load_error import LoadError
from adaptix._internal.morphing.provider_template import DumperProvider, LoaderProvider
from adaptix._internal.morphing.request_cls import DumperRequest, LoaderRequest
from adaptix._internal.provider.facade.provider import bound_by_any
from adaptix._internal.provider.loc_stack_filtering import Pred

try:
    import msgspec
except ImportError:
    pass

T = TypeVar("T")


class NativeMsgspecProvider(LoaderProvider, DumperProvider):
    def __init__(
        self,
        conversion_params: Mapping[str, Omittable[Any]],
        to_builtins_params: Mapping[str, Omittable[Any]],
    ):
        self.conversion_params = conversion_params
        self.to_builtins_params = to_builtins_params

    def _skip_omitted(self, mapping: Mapping[str, T]) -> Mapping[str, T]:
        return {k: v for k, v in mapping.items() if v != Omitted()}

    def provide_loader(self, mediator: Mediator[Loader], request: LoaderRequest) -> Loader:
        conversion_params = self._skip_omitted(self.conversion_params)

        if conversion_params:
            def native_msgspec_loader(data):
                try:
                    return msgspec.convert(data, type=request.last_loc.type, **conversion_params)
                except msgspec.ValidationError as e:
                    raise LoadError() from e

            return native_msgspec_loader

        def native_msgspec_loader_no_params(data):
            try:
                return msgspec.convert(data, type=request.last_loc.type)
            except msgspec.ValidationError as e:
                raise LoadError() from e

        return native_msgspec_loader_no_params

    def provide_dumper(self, mediator: Mediator[Dumper], request: DumperRequest) -> Dumper:
        to_builtins_params = self._skip_omitted(self.to_builtins_params)
        if to_builtins_params:
            def native_msgspec_dumper_with_params(data):
                return msgspec.to_builtins(data, **to_builtins_params)

            return native_msgspec_dumper_with_params

        return msgspec.to_builtins


def native_msgspec(
    *preds: Pred,
    enc_hook: Omittable[Callable[[Any], Any]] = Omitted(),
    to_builtins_builtin_types: Omittable[Iterable[type]] = Omitted(),
    to_builtins_str_keys: Omittable[bool] = Omitted(),
    dec_hook: Omittable[Callable[[Any], Any]] = Omitted(),
    convert_builtin_types: Omittable[Iterable[type]] = Omitted(),
    convert_str_keys: Omittable[bool] = Omitted(),
    strict: Omittable[bool] = Omitted(),
    from_attributes: Omittable[bool] = Omitted(),
) -> Provider:
    return bound_by_any(
        preds,
        NativeMsgspecProvider(
            conversion_params={
                "builtin_types": convert_builtin_types,
                "str_keys": convert_str_keys,
                "strict": strict,
                "from_attributes": from_attributes,
                "dec_hook": dec_hook,
            },
            to_builtins_params={
                "builtin_types": to_builtins_builtin_types,
                "str_keys": to_builtins_str_keys,
                "enc_hook": enc_hook,
            },
        ),
    )
