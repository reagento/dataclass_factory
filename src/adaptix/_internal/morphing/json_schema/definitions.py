from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar, Union

from ...provider.loc_stack_filtering import LocStack
from ...type_tools.fwd_ref import FwdRef
from .schema_model import BaseJSONSchema

T = TypeVar("T")

JSONSchemaT = TypeVar("JSONSchemaT")


@dataclass(frozen=True)
class RefSource(Generic[JSONSchemaT]):
    value: Optional[str]
    json_schema: JSONSchemaT
    loc_stack: LocStack = field(repr=False)


Boolable = Union[T, bool]


class JSONSchema(BaseJSONSchema[RefSource[FwdRef["JSONSchema"]], Boolable[FwdRef["JSONSchema"]]]):
    pass


class ResolvedJSONSchema(BaseJSONSchema[str, Boolable[FwdRef["ResolvedJSONSchema"]]]):
    pass

