import dataclasses

from adaptix._internal.utils import fix_dataclass_from_builtin


@dataclasses.dataclass(eq=False)
@fix_dataclass_from_builtin
class SentinelDumpError(Exception):
    sentinel: type

    def __str__(self):
        return f"Cannot dump {self.sentinel!r}"
