import contextlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Callable, NamedTuple, Optional, Protocol, Union

from ...code_tools.cascade_namespace import BuiltinCascadeNamespace, CascadeNamespace
from ...code_tools.code_gen_tree import (
    CodeBlock,
    CodeExpr,
    DictItem,
    DictKeyValue,
    DictLiteral,
    Expression,
    LinesWriter,
    ListLiteral,
    MappingUnpack,
    RawExpr,
    RawStatement,
    Statement,
    StringLiteral,
    TextSliceWriter,
    TryExcept,
    statements,
)
from ...code_tools.utils import get_literal_expr, get_literal_from_factory, is_singleton
from ...common import Dumper
from ...compat import CompatExceptionGroup
from ...definitions import DebugTrail
from ...model_tools.definitions import (
    DefaultFactory,
    DefaultFactoryWithSelf,
    DefaultValue,
    DescriptorAccessor,
    ItemAccessor,
    OutputField,
    OutputShape,
)
from ...special_cases_optimization import as_is_stub, get_default_clause
from ...struct_trail import TrailElement, append_trail, extend_trail, render_trail_as_note
from ...utils import Omittable, Omitted
from ..json_schema.definitions import JSONSchema
from ..json_schema.schema_model import JSONSchemaType, JSONValue
from .basic_gen import ModelDumperGen
from .crown_definitions import (
    CrownPath,
    CrownPathElem,
    ExtraExtract,
    ExtraTargets,
    OutCrown,
    OutDictCrown,
    OutExtraMove,
    OutFieldCrown,
    OutListCrown,
    OutNoneCrown,
    OutputNameLayout,
    Placeholder,
    Sieve,
)


class GenState:
    def __init__(self, namespace: CascadeNamespace, debug_trail: DebugTrail, error_handler_name: str):
        self.namespace = namespace
        self.debug_trail = debug_trail

        self.field_id_to_path: dict[str, CrownPath] = {}
        self.path_to_suffix: dict[CrownPath, str] = {}

        self._last_path_idx = 0
        self._path: CrownPath = ()

        self.trail_element_to_name_idx: dict[TrailElement, int] = {}
        self.error_collectors: list[Statement] = []
        self.overriden_error_collectors: dict[TrailElement, Callable[[Statement], Statement]] = {}
        self.trail_to_collector_idx: dict[TrailElement, int] = {}
        self.error_handler_name = error_handler_name

    def _ensure_path_idx(self, path: CrownPath) -> str:
        try:
            return self.path_to_suffix[path]
        except KeyError:
            self._last_path_idx += 1
            suffix = str(self._last_path_idx)
            self.path_to_suffix[path] = suffix
            return suffix

    @property
    def path(self):
        return self._path

    @contextlib.contextmanager
    def add_key(self, key: CrownPathElem):
        past = self._path

        self._path += (key,)
        self._ensure_path_idx(self._path)
        yield
        self._path = past

    def suffix(self, basis: str, key: Optional[CrownPathElem] = None) -> str:
        path = self._path if key is None else (*self._path, key)
        if not path:
            return basis
        return basis + "_" + self._ensure_path_idx(path)

    def var_suffix(self, basis: str, key: Optional[CrownPathElem] = None) -> str:
        var = self.suffix(basis, key)
        self.namespace.register_var(var)
        return var

    @property
    def v_crown(self) -> str:
        return self.suffix("result")


class VarExpr(Expression):
    def __init__(self, name: str):
        self.name = name

    def write_lines(self, writer: TextSliceWriter) -> None:
        writer.write(self.name)


class AssignmentStatement(Statement):
    def __init__(self, var: VarExpr, value: Expression):
        self.var = var
        self.value = value

    def write_lines(self, writer: TextSliceWriter) -> None:
        CodeBlock(
            "<var> = <value>",
            var=self.var,
            value=self.value,
        ).write_lines(
            writer,
        )


class ReturnVarStatement(Statement):
    def __init__(self, var: VarExpr):
        self.var = var

    def write_lines(self, writer: TextSliceWriter) -> None:
        CodeBlock(
            "return <var>",
            var=self.var,
        ).write_lines(
            writer,
        )


class ErrorCatching(Statement):
    def __init__(self, state: GenState, trail_element: Optional[TrailElement], stmt: Statement):
        self.state = state
        self.trail_element = trail_element
        self.stmt = stmt

    def _get_trail_element_expr(self, trail_element: TrailElement) -> Expression:
        literal_expr = get_literal_expr(trail_element)
        if literal_expr is not None:
            return RawExpr(literal_expr)

        if trail_element in self.state.trail_element_to_name_idx:
            idx = self.state.trail_element_to_name_idx[trail_element]
            v_trail_element = f"trail_element_{idx}"
        else:
            idx = len(self.state.trail_element_to_name_idx)
            self.state.trail_element_to_name_idx[trail_element] = idx
            v_trail_element = f"trail_element_{idx}"
            self.state.namespace.add_constant(v_trail_element, trail_element)
        return RawExpr(v_trail_element)

    def _get_append_trail(self, trail_element: TrailElement) -> Expression:
        if self.state.debug_trail == DebugTrail.ALL:
            return CodeExpr(
                "render_trail_as_note(append_trail(e, <trail_element>))",
                trail_element=self._get_trail_element_expr(trail_element),
            )
        return CodeExpr(
            "append_trail(e, <trail_element>)",
            trail_element=self._get_trail_element_expr(trail_element),
        )

    def _wrap_stmt(self, stmt: Statement) -> Statement:
        if self.state.debug_trail == DebugTrail.DISABLE:
            return stmt
        if self.state.debug_trail == DebugTrail.FIRST:
            if self.trail_element is None:
                return stmt
            return CodeBlock(
                """
                try:
                    <stmt>
                except Exception as e:
                    <append_trail>
                    raise
                """,
                stmt=self.stmt,
                append_trail=self._get_append_trail(self.trail_element),
            )
        if self.state.debug_trail == DebugTrail.ALL:
            idx = self._process_error_collecting(stmt)
            return CodeBlock(
                """
                try:
                    <stmt>
                except Exception as e:
                    raise <error_handler>(<idx>, data, <append_trail>) from None
                """,
                stmt=self.stmt,
                idx=RawExpr(repr(idx)),
                append_trail=(
                    RawExpr("e")
                    if self.trail_element is None else
                    self._get_append_trail(self.trail_element)
                ),
                error_handler=RawExpr(self.state.error_handler_name),
            )
        raise ValueError

    def _process_error_collecting(self, stmt: Statement) -> int:
        idx = len(self.state.error_collectors)

        if self.trail_element is None:
            self.state.error_collectors.append(stmt)
        else:
            if self.trail_element in self.state.trail_to_collector_idx:
                if self.trail_element not in self.state.overriden_error_collectors:
                    raise ValueError
                return self.state.trail_to_collector_idx[self.trail_element]

            self.state.error_collectors.append(self._get_error_collector(stmt, self.trail_element))
            self.state.trail_to_collector_idx[self.trail_element] = idx
        return idx

    def _get_error_collector(self, stmt: Statement, trail_element: TrailElement) -> Statement:
        error_saving = CodeBlock(
            "errors.append(<append_trail>)",
            append_trail=self._get_append_trail(trail_element),
        )

        if self.trail_element in self.state.overriden_error_collectors:
            return self.state.overriden_error_collectors[self.trail_element](error_saving)

        return CodeBlock(
            """
            try:
                <stmt>
            except Exception as e:
                <error_saving>
            """,
            stmt=stmt,
            error_saving=error_saving,
        )

    def write_lines(self, writer: TextSliceWriter) -> None:
        self._wrap_stmt(self.stmt).write_lines(writer)


class OutVarStatement(NamedTuple):
    stmt: Statement
    var: VarExpr


class OutVarStatementMaker(Protocol):
    def __call__(
        self,
        *,
        on_access_ok: Statement,
        on_access_error: Statement,
        on_unexpected_error: Optional[Statement],
    ) -> Statement:
        ...


class OptionalOutVarStatement(NamedTuple):
    var: VarExpr
    stmt_maker: OutVarStatementMaker


OutStatement = Union[OutVarStatement, OptionalOutVarStatement]


class DictBuilder:
    def __init__(self) -> None:
        self.before_stmts: list[Statement] = []
        self.dict_items: list[DictItem] = []
        self.after_stmts: list[Statement] = []


class BuiltinModelDumperGen(ModelDumperGen):
    def __init__(
        self,
        shape: OutputShape,
        name_layout: OutputNameLayout,
        debug_trail: DebugTrail,
        fields_dumpers: Mapping[str, Dumper],
        model_identity: str,
    ):
        self._shape = shape
        self._name_layout = name_layout
        self._debug_trail = debug_trail
        self._fields_dumpers = fields_dumpers
        self._id_to_field: dict[str, OutputField] = {field.id: field for field in self._shape.fields}
        self._model_identity = model_identity

    def _v_dumper(self, field: OutputField) -> str:
        return f"dumper_{field.id}"

    def _create_state(self, namespace: CascadeNamespace) -> GenState:
        return GenState(namespace, self._debug_trail, f"error_handler")

    def _alloc_var(self, state: GenState, name: str) -> VarExpr:
        state.namespace.register_var(name)
        return VarExpr(name)

    def _get_header(self, state: GenState) -> Statement:
        writer = LinesWriter()
        if state.path_to_suffix:
            writer.write("# suffix to path")
            for path, suffix in state.path_to_suffix.items():
                writer.write(f"# {suffix} -> {list(path)}")

            writer.write("")

        if state.field_id_to_path:
            writer.write("# field to path")
            for f_name, path in state.field_id_to_path.items():
                writer.write(f"# {f_name} -> {list(path)}")

            writer.write("")

        return RawStatement(writer.make_string())

    def produce_code(self, closure_name: str) -> tuple[str, Mapping[str, object]]:
        namespace = BuiltinCascadeNamespace()
        namespace.add_constant("CompatExceptionGroup", CompatExceptionGroup)
        namespace.add_constant("append_trail", append_trail)
        namespace.add_constant("extend_trail", extend_trail)
        namespace.add_constant("render_trail_as_note", render_trail_as_note)
        for field_id, dumper in self._fields_dumpers.items():
            namespace.add_constant(self._v_dumper(self._id_to_field[field_id]), dumper)

        state = self._create_state(namespace)
        body = self._get_body_statement(state)
        header = self._get_header(state)

        writer = LinesWriter()
        closure = CodeBlock(
            """
            def <closure_name>(data):
                <header>
                <body>
            """,
            closure_name=RawExpr(closure_name),
            header=header,
            body=body,
        )
        closure.write_lines(writer)

        result = writer.make_string()
        if state.debug_trail == DebugTrail.ALL:
            error_handler_writer = LinesWriter()
            self._get_error_handler(state).write_lines(error_handler_writer)
            result += "\n\n\n" + error_handler_writer.make_string()
        return result, namespace.all_constants

    def _get_body_statement(self, state: GenState) -> Statement:
        crown_out_stmt = self._get_root_crown_stmt(state)
        extra_extraction_out_stmt = self._get_extra_extraction(state)

        if extra_extraction_out_stmt is None:
            extending_stmt = crown_out_stmt
        elif isinstance(extra_extraction_out_stmt, OutVarStatement):
            extending_stmt = OutVarStatement(
                var=crown_out_stmt.var,
                stmt=statements(
                    crown_out_stmt.stmt,
                    CodeBlock(
                        "<result>.update(<extra>)",
                        result=crown_out_stmt.var,
                        extra=extra_extraction_out_stmt.var,
                    ),
                ),
            )
        elif isinstance(extra_extraction_out_stmt, OptionalOutVarStatement):
            extending_stmt = OutVarStatement(
                var=extra_extraction_out_stmt.var,
                stmt=extra_extraction_out_stmt.stmt_maker(
                    on_access_ok=statements(
                        crown_out_stmt.stmt,
                        CodeBlock(
                            "<result>.update(<extra>)",
                            result=crown_out_stmt.var,
                            extra=extra_extraction_out_stmt.var,
                        ),
                    ),
                    on_access_error=CodeBlock.PASS,
                    on_unexpected_error=None,
                ),
            )
        else:
            raise TypeError

        return statements(
            extending_stmt.stmt,
            ReturnVarStatement(extending_stmt.var),
        )

    def _get_extra_extraction(self, state: GenState) -> Optional[OutStatement]:
        if isinstance(self._name_layout.extra_move, ExtraTargets):
            return self._get_extra_target_extraction(state, self._name_layout.extra_move)
        if isinstance(self._name_layout.extra_move, ExtraExtract):
            return self._get_extra_extract_extraction(state, self._name_layout.extra_move)
        if self._name_layout.extra_move is None:
            return None
        raise ValueError

    def _get_extra_target_extraction(self, state: GenState, extra_targets: ExtraTargets) -> OutStatement:
        if len(extra_targets.fields) == 1:
            return self._get_field_extraction(state, self._id_to_field[extra_targets.fields[0]])

        out_stmts = [
            self._get_field_extraction(state, self._id_to_field[field_id])
            for field_id in extra_targets.fields
        ]
        builder = DictBuilder()

        for out_stmt in out_stmts:
            if isinstance(out_stmt, OutVarStatement):
                builder.before_stmts.append(out_stmt.stmt)
                builder.dict_items.append(MappingUnpack(out_stmt.var))
            elif isinstance(out_stmt, OptionalOutVarStatement):
                builder.after_stmts.append(
                    out_stmt.stmt_maker(
                        on_access_ok=CodeBlock(
                            "extra.update(<var>)",
                            var=out_stmt.var,
                        ),
                        on_access_error=CodeBlock.PASS,
                        on_unexpected_error=None,
                    ),
                )

        var = self._alloc_var(state, "extra")
        return OutVarStatement(
            stmt=statements(
                *builder.before_stmts,
                AssignmentStatement(
                    var=var,
                    value=DictLiteral(builder.dict_items),
                ),
                *builder.after_stmts,
            ),
            var=var,
        )

    def _get_extra_extract_extraction(self, state: GenState, extra_move: ExtraExtract) -> OutStatement:
        state.namespace.add_constant("extractor", extra_move.func)
        var = self._alloc_var(state, "extra")
        return OutVarStatement(
            stmt=statements(
                ErrorCatching(
                    state=state,
                    trail_element=None,
                    stmt=AssignmentStatement(var=var, value=RawExpr("extractor(data)")),
                ),
            ),
            var=var,
        )

    def _get_error_handler(self, state: GenState) -> Statement:
        error_collectors = [
            CodeBlock(
                """
                if idx < <idx>:
                    <stmt>
                """,
                idx=RawExpr(repr(idx)),
                stmt=stmt,
            )
            for idx, stmt in enumerate(state.error_collectors)
        ]
        return CodeBlock(
            """
            def <error_handler>(idx, data, e):
                errors = [e]
                <error_collectors>
                return ExceptionGroup(<error_msg>, errors)
            """,
            error_collectors=statements(*error_collectors),
            error_msg=StringLiteral(f"while dumping model {self._model_identity}"),
            error_handler=RawExpr(state.error_handler_name),
        )

    def _get_access_expr(self, namespace: CascadeNamespace, field: OutputField) -> str:
        accessor = field.accessor
        if isinstance(accessor, DescriptorAccessor):
            if accessor.attr_name.isidentifier():
                return f"data.{accessor.attr_name}"
            return f"getattr(data, {accessor.attr_name!r})"
        if isinstance(accessor, ItemAccessor):
            return f"data[{accessor.key!r}]"

        accessor_getter = f"accessor_getter_{field.id}"
        namespace.add_constant(accessor_getter, field.accessor.getter)
        return f"{accessor_getter}(data)"

    def _get_access_error_expr(self, namespace: CascadeNamespace, field: OutputField) -> str:
        access_error = field.accessor.access_error
        literal_expr = get_literal_expr(access_error)
        if literal_expr is not None:
            return literal_expr

        access_error_getter = f"access_error_{field.id}"
        namespace.add_constant(access_error_getter, field.accessor.getter)
        return f"{access_error_getter}(data)"

    def _get_trail_element_expr(self, namespace: CascadeNamespace, field: OutputField) -> str:
        trail_element = field.accessor.trail_element
        literal_expr = get_literal_expr(trail_element)
        if literal_expr is not None:
            return literal_expr

        v_trail_element = f"trail_element_{field.id}"
        namespace.add_constant(v_trail_element, trail_element)
        return v_trail_element

    def _get_field_out_variable(self, namespace: CascadeNamespace, field: OutputField) -> str:
        out_variable = f"r_{field.id}"
        namespace.register_var(out_variable)
        return out_variable

    def _get_field_extraction(self, state: GenState, field: OutputField) -> OutStatement:
        return (
            self._get_required_field_extraction(state, field)
            if field.is_required else
            self._get_optional_field_extraction(state, field)
        )

    def _get_required_field_extraction(self, state: GenState, field: OutputField) -> OutVarStatement:
        var = self._alloc_var(state, f"r_{field.id}")
        return OutVarStatement(
            stmt=ErrorCatching(
                state=state,
                trail_element=field.accessor.trail_element,
                stmt=AssignmentStatement(var=var, value=RawExpr(self._get_access_expr(state.namespace, field))),
            ),
            var=var,
        )

    def _get_optional_field_extraction(self, state: GenState, field: OutputField) -> OutStatement:
        access_expr = self._get_access_expr(state.namespace, field)
        access_error_expr = self._get_access_error_expr(state.namespace, field)
        var = self._alloc_var(state, f"r_{field.id}")

        def stmt_maker(
            *,
            on_access_ok: Statement,
            on_access_error: Statement,
            on_unexpected_error: Optional[Statement],
        ) -> Statement:
            excepts = [
                (RawExpr(access_error_expr), on_access_error),
            ]
            if on_unexpected_error is not None:
                excepts.append(
                    (RawExpr("Exception as e"), on_unexpected_error),
                )
            return TryExcept(
                try_=AssignmentStatement(var=var, value=RawExpr(access_expr)),
                excepts=excepts,
                else_=on_access_ok,
            )

        return OptionalOutVarStatement(
            var=var,
            stmt_maker=stmt_maker,
        )

    def _get_root_crown_stmt(self, state: GenState) -> OutVarStatement:
        if isinstance(self._name_layout.crown, OutDictCrown):
            result = self._get_dict_crown_out_stmt(state, self._name_layout.crown)
        elif isinstance(self._name_layout.crown, OutListCrown):
            result = self._get_list_crown_out_stmt(state, self._name_layout.crown)
        else:
            raise TypeError
        if isinstance(result, OptionalOutVarStatement):
            raise TypeError
        return result

    def _get_crown_out_stmt(self, state: GenState, key: CrownPathElem, crown: OutCrown) -> OutStatement:
        with state.add_key(key):
            if isinstance(crown, OutDictCrown):
                return self._get_dict_crown_out_stmt(state, crown)
            if isinstance(crown, OutListCrown):
                return self._get_list_crown_out_stmt(state, crown)
            if isinstance(crown, OutFieldCrown):
                return self._get_field_crown_out_stmt(state, crown)
            if isinstance(crown, OutNoneCrown):
                return self._get_none_crown_out_stmt(state, crown)
        raise TypeError

    def _get_dict_crown_out_stmt(self, state: GenState, crown: OutDictCrown) -> OutStatement:
        builder = DictBuilder()
        for key, sub_crown in crown.map.items():
            if key not in crown.sieves:
                self._process_dict_sub_crown(
                    state=state,
                    builder=builder,
                    key=key,
                    sub_crown=sub_crown,
                    out_stmt=self._get_crown_out_stmt(state, key, sub_crown),
                )
            else:
                self._process_dict_sieved_sub_crown(
                    state=state,
                    builder=builder,
                    key=key,
                    sieve=crown.sieves[key],
                    sub_crown=sub_crown,
                    out_stmt=self._get_crown_out_stmt(state, key, sub_crown),
                )

        var = self._alloc_var(state, state.v_crown)
        return OutVarStatement(
            stmt=statements(
                *builder.before_stmts,
                AssignmentStatement(
                    var=var,
                    value=DictLiteral(builder.dict_items),
                ),
                *builder.after_stmts,
            ),
            var=var,
        )

    def _get_dumper_call(self, state: GenState, sub_crown: OutCrown, var: VarExpr) -> OutVarStatement:
        if not isinstance(sub_crown, OutFieldCrown):
            return OutVarStatement(var=var, stmt=statements())

        field = self._id_to_field[sub_crown.id]
        dumped_var = self._alloc_var(state, f"dumped_{field.id}")
        dumper_call = (
            var
            if self._fields_dumpers[sub_crown.id] == as_is_stub else
            CodeExpr(
                "<dumper>(<expr>)",
                dumper=RawExpr(self._v_dumper(field)),
                expr=var,
            )
        )
        return OutVarStatement(
            var=dumped_var,
            stmt=ErrorCatching(
                state=state,
                trail_element=field.accessor.trail_element,
                stmt=AssignmentStatement(var=dumped_var, value=dumper_call),
            ),
        )

    def _merge_error_catching(
        self,
        out_stmt: OutVarStatement,
        dumper_call: OutVarStatement,
    ) -> OutVarStatement:
        if (
            isinstance(out_stmt.stmt, ErrorCatching)
            and isinstance(dumper_call.stmt, ErrorCatching)
            and out_stmt.stmt.trail_element == dumper_call.stmt.trail_element
        ):
            return OutVarStatement(
                var=dumper_call.var,
                stmt=ErrorCatching(
                    state=out_stmt.stmt.state,
                    trail_element=out_stmt.stmt.trail_element,
                    stmt=statements(
                        out_stmt.stmt.stmt,
                        dumper_call.stmt.stmt,
                    ),
                ),
            )
        return OutVarStatement(
            var=dumper_call.var,
            stmt=statements(
                out_stmt.stmt,
                dumper_call.stmt,
            ),
        )

    def _process_dict_sub_crown(
        self,
        state: GenState,
        builder: DictBuilder,
        key: str,
        sub_crown: OutCrown,
        out_stmt: OutStatement,
    ) -> None:
        dumper_call = self._get_dumper_call(
            state=state,
            sub_crown=sub_crown,
            var=out_stmt.var,
        )
        if isinstance(out_stmt, OutVarStatement):
            builder.before_stmts.append(
                self._merge_error_catching(out_stmt=out_stmt, dumper_call=dumper_call).stmt,
            )
            builder.dict_items.append(
                DictKeyValue(StringLiteral(key), dumper_call.var),
            )
        elif isinstance(out_stmt, OptionalOutVarStatement):
            stmt = out_stmt.stmt_maker(
                on_access_ok=statements(
                    dumper_call.stmt,
                    self._get_dict_append(
                        state=state,
                        key=key,
                        value=dumper_call.var,
                    ),
                ),
                on_access_error=CodeBlock.PASS,
                on_unexpected_error=...,
            )
            builder.after_stmts.append(stmt)
        else:
            raise TypeError

    def _process_dict_sieved_sub_crown(
        self,
        state: GenState,
        builder: DictBuilder,
        key: str,
        sieve: Sieve,
        sub_crown: OutCrown,
        out_stmt: OutStatement,
    ) -> None:
        dumper_call = self._get_dumper_call(state, sub_crown, out_stmt.var)
        condition = self._get_sieve_condition(state, sieve, key, out_stmt.var)
        conditional_append = statements(
            CodeBlock(
                """
                if <condition>:
                    <dumper_call>
                    <dict_append>
                """,
                condition=condition,
                dumper_call=dumper_call.stmt,
                dict_append=self._get_dict_append(state, key, dumper_call.var),
            ),
        )
        if isinstance(out_stmt, OptionalOutVarStatement):
            stmt = out_stmt.stmt_maker(
                on_access_ok=conditional_append,
                on_access_error=CodeBlock.PASS,
                on_unexpected_error=...,
            )
            builder.after_stmts.append(stmt)
        elif isinstance(out_stmt, OutVarStatement):
            builder.after_stmts.append(
                statements(
                    out_stmt.stmt,
                    conditional_append,
                ),
            )
            if isinstance(sub_crown, OutFieldCrown):
                assert isinstance(out_stmt.stmt, ErrorCatching)
                assert isinstance(dumper_call.stmt, ErrorCatching)
                trail_element = self._id_to_field[sub_crown.id].accessor.trail_element
                state.overriden_error_collectors[trail_element] = (
                    lambda error_saving: CodeBlock(
                        """
                        try:
                            <stmt>
                        except Exception as e:
                            <error_saving>
                        else:
                            if <condition>:
                                try:
                                    <dumper_call>
                                except Exception as e:
                                    <error_saving>
                        """,
                        stmt=out_stmt.stmt.stmt,
                        condition=condition,
                        dumper_call=dumper_call.stmt.stmt,
                        error_saving=error_saving,
                    )
                )
        else:
            raise TypeError

    def _get_dict_append(
        self,
        state: GenState,
        key: str,
        value: Expression,
    ) -> Statement:
        return CodeBlock(
            "<crown>[<key>] = <value>",
            key=StringLiteral(key),
            crown=RawExpr(state.v_crown),
            value=value,
        )

    def _get_sieve_condition(self, state: GenState, sieve: Sieve, key: str, test_var: VarExpr) -> Expression:
        default_clause = get_default_clause(sieve)
        if default_clause is None:
            v_sieve = state.suffix("sieve", key)
            state.namespace.add_constant(v_sieve, sieve)
            return RawExpr(f"{v_sieve}({test_var.name})")

        if isinstance(default_clause, DefaultValue):
            literal_expr = get_literal_expr(default_clause.value)
            if literal_expr is not None:
                return RawExpr(
                    f"{test_var.name} is not {literal_expr}"
                    if is_singleton(default_clause.value) else
                    f"{test_var.name} != {literal_expr}",
                )
            v_default = state.suffix("default", key)
            state.namespace.add_constant(v_default, default_clause.value)
            return RawExpr(f"{test_var.name} != {v_default}")

        if isinstance(default_clause, DefaultFactory):
            literal_expr = get_literal_from_factory(default_clause.factory)
            if literal_expr is not None:
                return RawExpr(f"{test_var.name} != {literal_expr}")
            v_default = state.suffix("default", key)
            state.namespace.add_constant(v_default, default_clause.factory)
            return RawExpr(f"{test_var.name} != {v_default}()")

        if isinstance(default_clause, DefaultFactoryWithSelf):
            v_default = state.suffix("default", key)
            state.namespace.add_constant(v_default, default_clause.factory)
            return RawExpr(f"{test_var.name} != {v_default}(data)")

        raise TypeError

    def _get_list_crown_out_stmt(self, state: GenState, crown: OutListCrown) -> OutStatement:
        dumped_out_stmts = [
            self._merge_error_catching(
                out_stmt=out_stmt,
                dumper_call=self._get_dumper_call(
                    state,
                    sub_crown,
                    out_stmt.var,
                ),
            )
            for sub_crown, out_stmt in zip(
                crown.map,
                (
                    self._get_crown_out_stmt(state, idx, sub_crown)
                    for idx, sub_crown in enumerate(crown.map)
                ),
            )
        ]
        var = self._alloc_var(state, state.v_crown)
        return OutVarStatement(
            stmt=statements(
                *(out_stmt.stmt for out_stmt in dumped_out_stmts),
                AssignmentStatement(
                    var=var,
                    value=ListLiteral([out_stmt.var for out_stmt in dumped_out_stmts]),
                ),
            ),
            var=var,
        )

    def _get_field_crown_out_stmt(self, state: GenState, crown: OutFieldCrown) -> OutStatement:
        state.field_id_to_path[crown.id] = state.path
        return self._get_field_extraction(state, self._id_to_field[crown.id])

    def _get_placeholder_expr(self, state: GenState, placeholder: Placeholder) -> Expression:
        if isinstance(placeholder, DefaultFactory):
            literal_expr = get_literal_from_factory(placeholder.factory)
            if literal_expr is not None:
                return RawExpr(literal_expr)

            v_placeholder = state.suffix("placeholder")
            state.namespace.add_constant(v_placeholder, placeholder.factory)
            return RawExpr(v_placeholder + "()")

        if isinstance(placeholder, DefaultValue):
            literal_expr = get_literal_expr(placeholder.value)
            if literal_expr is not None:
                return RawExpr(literal_expr)

            v_placeholder = state.suffix("placeholder")
            state.namespace.add_constant(v_placeholder, placeholder.value)
            return RawExpr(v_placeholder)

        raise TypeError

    def _get_none_crown_out_stmt(self, state: GenState, crown: OutNoneCrown) -> OutStatement:
        var = self._alloc_var(state, state.v_crown)
        return OutVarStatement(
            var=var,
            stmt=AssignmentStatement(
                var=var,
                value=self._get_placeholder_expr(state, crown.placeholder),
            ),
        )


class ModelOutputJSONSchemaGen:
    def __init__(
        self,
        shape: OutputShape,
        extra_move: OutExtraMove,
        field_json_schema_getter: Callable[[OutputField], JSONSchema],
        field_default_dumper: Callable[[OutputField], Omittable[JSONValue]],
        placeholder_dumper: Callable[[Any], JSONValue],
    ):
        self._shape = shape
        self._extra_move = extra_move
        self._field_json_schema_getter = field_json_schema_getter
        self._field_default_dumper = field_default_dumper
        self._placeholder_dumper = placeholder_dumper

    def _convert_dict_crown(self, crown: OutDictCrown) -> JSONSchema:
        return JSONSchema(
            type=JSONSchemaType.OBJECT,
            required=[
                key
                for key, value in crown.map.items()
                if self._is_required_crown(value)
            ],
            properties={
                key: self.convert_crown(value)
                for key, value in crown.map.items()
            },
            additional_properties=self._extra_move is not None,
        )

    def _convert_list_crown(self, crown: OutListCrown) -> JSONSchema:
        items = [
            self.convert_crown(sub_crown)
            for sub_crown in crown.map
        ]
        return JSONSchema(
            type=JSONSchemaType.ARRAY,
            prefix_items=items,
            max_items=len(items),
            min_items=len(items),
        )

    def _convert_field_crown(self, crown: OutFieldCrown) -> JSONSchema:
        field = self._shape.fields_dict[crown.id]
        json_schema = self._field_json_schema_getter(field)
        default = self._field_default_dumper(field)
        if default != Omitted():
            return replace(json_schema, default=default)
        return json_schema

    def _convert_none_crown(self, crown: OutNoneCrown) -> JSONSchema:
        value = (
            crown.placeholder.factory()
            if isinstance(crown.placeholder, DefaultFactory) else
            crown.placeholder.value
        )
        return JSONSchema(const=self._placeholder_dumper(value))

    def _is_required_crown(self, crown: OutCrown) -> bool:
        if isinstance(crown, OutFieldCrown):
            return self._shape.fields_dict[crown.id].is_required
        return True

    def convert_crown(self, crown: OutCrown) -> JSONSchema:
        if isinstance(crown, OutDictCrown):
            return self._convert_dict_crown(crown)
        if isinstance(crown, OutListCrown):
            return self._convert_list_crown(crown)
        if isinstance(crown, OutFieldCrown):
            return self._convert_field_crown(crown)
        if isinstance(crown, OutNoneCrown):
            return self._convert_none_crown(crown)
        raise TypeError
