from typing import Mapping, Protocol, Tuple

from ...code_tools import BasicClosureCompiler, BuiltinContextNamespace
from ...common import Loader
from ...provider.essential import CannotProvide, Mediator
from ...provider.model.definitions import CodeGenerator, InputFigure, InputFigureRequest, VarBinder
from ...provider.model.input_extraction_gen import BuiltinInputExtractionGen
from ...provider.provider_template import LoaderProvider
from ...provider.request_cls import LoaderFieldRequest, LoaderRequest
from .basic_gen import (
    CodeGenHookRequest,
    NameSanitizer,
    compile_closure_with_globals_capturing,
    get_extra_targets_at_crown,
    get_optional_fields_at_list_crown,
    get_skipped_fields,
    has_collect_policy,
    strip_figure,
    stub_code_gen_hook,
)
from .crown_definitions import InputNameMapping, InputNameMappingRequest
from .input_creation_gen import BuiltinInputCreationGen


class InputExtractionMaker(Protocol):
    def __call__(self, mediator: Mediator, request: LoaderRequest) -> Tuple[CodeGenerator, InputFigure]:
        ...


class InputCreationMaker(Protocol):
    def __call__(self, mediator: Mediator, request: LoaderRequest, figure: InputFigure) -> CodeGenerator:
        ...


class BuiltinInputExtractionMaker(InputExtractionMaker):
    def __call__(self, mediator: Mediator, request: LoaderRequest) -> Tuple[CodeGenerator, InputFigure]:
        figure: InputFigure = mediator.provide(
            InputFigureRequest(type=request.type)
        )

        name_mapping = mediator.provide(
            InputNameMappingRequest(
                type=request.type,
                figure=figure,
            )
        )

        processed_figure = self._process_figure(figure, name_mapping)
        self._validate_params(figure, processed_figure, name_mapping)

        field_loaders = {
            field.name: mediator.provide(
                LoaderFieldRequest(
                    strict_coercion=request.strict_coercion,
                    debug_path=request.debug_path,
                    field=field,
                    type=field.type,
                )
            )
            for field in processed_figure.fields
        }

        extraction_gen = self._create_extraction_gen(request, figure, name_mapping, field_loaders)

        return extraction_gen, figure

    def _process_figure(self, figure: InputFigure, name_mapping: InputNameMapping) -> InputFigure:
        skipped_fields = get_skipped_fields(figure, name_mapping)

        skipped_required_fields = [
            field.name
            for field in figure.fields
            if field.is_required and field.name in skipped_fields
        ]

        if skipped_required_fields:
            raise ValueError(
                f"Required fields {skipped_required_fields} are skipped"
            )

        return strip_figure(figure, skipped_fields)

    def _validate_params(self, figure: InputFigure, processed_figure: InputFigure, name_mapping: InputNameMapping):
        if figure.extra is None and has_collect_policy(name_mapping.crown):
            raise ValueError(
                "Cannot create loader that collect extra data"
                " if InputFigure does not take extra data",
            )

        extra_targets_at_crown = get_extra_targets_at_crown(figure, name_mapping)
        if extra_targets_at_crown:
            raise ValueError(
                f"Extra targets {extra_targets_at_crown} are found at crown"
            )

        optional_fields_at_list_crown = get_optional_fields_at_list_crown(
            {field.name: field for field in processed_figure.fields},
            name_mapping.crown,
        )
        if optional_fields_at_list_crown:
            raise ValueError(
                f"Optional fields {optional_fields_at_list_crown} are found at list crown"
            )

    def _create_extraction_gen(
        self,
        request: LoaderRequest,
        figure: InputFigure,
        name_mapping: InputNameMapping,
        field_loaders: Mapping[str, Loader],
    ) -> CodeGenerator:
        return BuiltinInputExtractionGen(
            figure=figure,
            crown=name_mapping.crown,
            debug_path=request.debug_path,
            field_loaders=field_loaders,
        )


def make_input_creation(mediator: Mediator, request: LoaderRequest, figure: InputFigure) -> CodeGenerator:
    return BuiltinInputCreationGen(figure=figure)


class ModelLoaderProvider(LoaderProvider):
    def __init__(
        self,
        name_sanitizer: NameSanitizer,
        extraction_maker: InputExtractionMaker,
        creation_maker: InputCreationMaker,
    ):
        self._name_sanitizer = name_sanitizer
        self._extraction_maker = extraction_maker
        self._creation_maker = creation_maker

    def _provide_loader(self, mediator: Mediator, request: LoaderRequest) -> Loader:
        extraction_gen, figure = self._extraction_maker(mediator, request)
        creation_gen = self._creation_maker(mediator, request, figure)

        try:
            code_gen_hook = mediator.provide(CodeGenHookRequest())
        except CannotProvide:
            code_gen_hook = stub_code_gen_hook

        binder = self._get_binder()
        ctx_namespace = BuiltinContextNamespace()

        extraction_code_builder = extraction_gen(binder, ctx_namespace)
        creation_code_builder = creation_gen(binder, ctx_namespace)

        return compile_closure_with_globals_capturing(
            compiler=self._get_compiler(),
            code_gen_hook=code_gen_hook,
            binder=binder,
            namespace=ctx_namespace.dict,
            body_builders=[
                extraction_code_builder,
                creation_code_builder,
            ],
            closure_name=self._get_closure_name(request),
            file_name=self._get_file_name(request),
        )

    def _get_closure_name(self, request: LoaderRequest) -> str:
        tp = request.type
        if isinstance(tp, type):
            name = tp.__name__
        else:
            name = str(tp)

        s_name = self._name_sanitizer.sanitize(name)
        if s_name != "":
            s_name = "_" + s_name
        return "model_loader" + s_name

    def _get_file_name(self, request: LoaderRequest) -> str:
        return self._get_closure_name(request)

    def _get_compiler(self):
        return BasicClosureCompiler()

    def _get_binder(self):
        return VarBinder()
